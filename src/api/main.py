from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool

from .classifier import PyTorchFoodClassifier
from .middleware import configure_middleware
from .models import AppSettings
from .routes import router


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_settings(config_path: Path | None = None) -> AppSettings:
    config_file = config_path or repo_root() / "config.yaml"
    if not config_file.exists():
        return AppSettings()

    with config_file.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    return AppSettings.model_validate(data)


def create_app(
    settings: AppSettings | None = None,
    classifier: PyTorchFoodClassifier | None = None,
) -> FastAPI:
    resolved_settings = settings or load_settings()
    resolved_classifier = classifier or PyTorchFoodClassifier(
        model_path=repo_root() / resolved_settings.model.path,
        classes_path=repo_root() / resolved_settings.model.classes_path,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            await run_in_threadpool(app.state.classifier.load)
            app.state.startup_error = None
        except Exception as exc:
            app.state.startup_error = str(exc)
        yield

    app = FastAPI(
        title=resolved_settings.api.title,
        version=resolved_settings.api.version,
        description=(
            "Programmatic REST API for FlavorSnap food classification. "
            "Upload a food image as multipart/form-data and receive ranked predictions."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    app.state.settings = resolved_settings
    app.state.classifier = resolved_classifier
    app.state.startup_error = None

    configure_middleware(app, resolved_settings.api)
    app.include_router(router, prefix=resolved_settings.api.prefix)

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {
            "message": "FlavorSnap REST API",
            "docs": "/docs",
            "openapi": "/openapi.json",
        }

    return app


app = create_app()
