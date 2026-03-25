from __future__ import annotations

from pydantic import BaseModel, Field


class PredictionScore(BaseModel):
    label: str = Field(..., description="Predicted class label.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the prediction.",
    )


class PreprocessingOptions(BaseModel):
    resize: int = Field(
        default=224,
        ge=224,
        le=1024,
        description="Square resize target used before model inference.",
    )
    center_crop: bool = Field(
        default=True,
        description="Apply center crop after resize for model-ready input.",
    )
    normalize: bool = Field(
        default=True,
        description="Apply ImageNet normalization before inference.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of ranked predictions to return.",
    )


class ClassificationResponse(BaseModel):
    prediction: str = Field(..., description="Top predicted class label.")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the top prediction.",
    )
    predictions: list[PredictionScore] = Field(
        ...,
        description="Ranked list of predictions and confidence scores.",
    )
    preprocessing: PreprocessingOptions
    processing_time_ms: float = Field(
        ...,
        ge=0.0,
        description="Time spent running preprocessing and inference in milliseconds.",
    )
    filename: str = Field(..., description="Original uploaded filename.")
    request_id: str = Field(..., description="Request correlation identifier.")


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str]
    startup_error: str | None = None


class RateLimitSettings(BaseModel):
    requests: int = Field(default=30, ge=1)
    window_seconds: int = Field(default=60, ge=1)
    exempt_paths: list[str] = Field(
        default_factory=lambda: ["/docs", "/redoc", "/openapi.json", "/health"]
    )


class CorsSettings(BaseModel):
    allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    allow_headers: list[str] = Field(default_factory=lambda: ["*"])


class APISettings(BaseModel):
    title: str = "FlavorSnap REST API"
    version: str = "1.0.0"
    prefix: str = "/api/v1"
    max_upload_size_mb: int = Field(default=10, ge=1)
    allowed_mime_types: list[str] = Field(
        default_factory=lambda: ["image/jpeg", "image/png", "image/webp"]
    )
    default_preprocessing: PreprocessingOptions = Field(
        default_factory=PreprocessingOptions
    )
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)
    cors: CorsSettings = Field(default_factory=CorsSettings)


class ModelSettings(BaseModel):
    path: str = "models/best_model.pth"
    classes_path: str = "food_classes.txt"


class AppSettings(BaseModel):
    api: APISettings = Field(default_factory=APISettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
