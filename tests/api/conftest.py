from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from src.api.main import create_app
from src.api.models import AppSettings, PredictionScore


class FakeClassifier:
    def __init__(self) -> None:
        self.ready = False
        self.class_names = ["Akara", "Bread", "Egusi", "Moi Moi"]

    def load(self) -> None:
        self.ready = True

    def classify(self, image_bytes: bytes, options) -> list[PredictionScore]:
        if image_bytes == b"bad-image":
            raise ValueError("Uploaded file is not a valid image.")
        if image_bytes == b"boom":
            raise RuntimeError("Classifier crashed.")
        return [
            PredictionScore(label="Moi Moi", confidence=0.91),
            PredictionScore(label="Akara", confidence=0.06),
            PredictionScore(label="Bread", confidence=0.03),
        ][: options.top_k]


@pytest.fixture
def settings() -> AppSettings:
    return AppSettings.model_validate(
        {
            "api": {
                "max_upload_size_mb": 1,
                "rate_limit": {
                    "requests": 10,
                    "window_seconds": 60,
                    "exempt_paths": ["/docs", "/redoc", "/openapi.json", "/health"],
                },
            }
        }
    )


@pytest.fixture
def client(settings: AppSettings) -> TestClient:
    app = create_app(settings=settings, classifier=FakeClassifier())
    with TestClient(app) as test_client:
        yield test_client
