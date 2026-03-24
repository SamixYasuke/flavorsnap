from __future__ import annotations

from time import perf_counter
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool

from .models import (
    ClassificationResponse,
    ErrorResponse,
    HealthResponse,
    PreprocessingOptions,
)

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Check API and model readiness",
)
async def health_check(request: Request) -> HealthResponse:
    classifier = request.app.state.classifier
    return HealthResponse(
        status="ok" if classifier.ready else "starting",
        model_loaded=classifier.ready,
        classes=classifier.class_names,
        startup_error=request.app.state.startup_error,
    )


@router.post(
    "/classify",
    response_model=ClassificationResponse,
    tags=["classification"],
    summary="Classify a food image",
    responses={
        400: {"model": ErrorResponse, "description": "Bad request"},
        413: {"model": ErrorResponse, "description": "Uploaded file too large"},
        415: {"model": ErrorResponse, "description": "Unsupported media type"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
        500: {"model": ErrorResponse, "description": "Internal server error"},
    },
)
async def classify_image(
    request: Request,
    image: Annotated[
        UploadFile,
        File(description="Food image to classify."),
    ],
    resize: Annotated[
        int | None,
        Form(description="Optional square resize target, in pixels."),
    ] = None,
    center_crop: Annotated[
        bool | None,
        Form(description="Apply center crop after resize."),
    ] = None,
    normalize: Annotated[
        bool | None,
        Form(description="Apply ImageNet normalization."),
    ] = None,
    top_k: Annotated[
        int | None,
        Form(description="Number of predictions to return."),
    ] = None,
) -> ClassificationResponse:
    settings = request.app.state.settings
    classifier = request.app.state.classifier

    if image.content_type not in settings.api.allowed_mime_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported media type. Use JPEG, PNG, or WebP.",
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    max_size_bytes = settings.api.max_upload_size_mb * 1024 * 1024
    if len(image_bytes) > max_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Uploaded file exceeds the {settings.api.max_upload_size_mb} MB limit.",
        )

    defaults = settings.api.default_preprocessing
    preprocessing = PreprocessingOptions(
        resize=resize if resize is not None else defaults.resize,
        center_crop=center_crop if center_crop is not None else defaults.center_crop,
        normalize=normalize if normalize is not None else defaults.normalize,
        top_k=top_k if top_k is not None else defaults.top_k,
    )

    start = perf_counter()
    try:
        predictions = await run_in_threadpool(classifier.classify, image_bytes, preprocessing)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc

    processing_time_ms = round((perf_counter() - start) * 1000, 3)
    top_prediction = predictions[0]

    return ClassificationResponse(
        prediction=top_prediction.label,
        confidence=top_prediction.confidence,
        predictions=predictions,
        preprocessing=preprocessing,
        processing_time_ms=processing_time_ms,
        filename=image.filename or "upload",
        request_id=getattr(request.state, "request_id", "unknown"),
    )
