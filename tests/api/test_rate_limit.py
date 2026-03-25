from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import create_app
from src.api.models import AppSettings, PredictionScore


class RateLimitClassifier:
    ready = False
    class_names = ["Akara", "Bread"]

    def load(self) -> None:
        self.ready = True

    def classify(self, image_bytes: bytes, options) -> list[PredictionScore]:
        return [PredictionScore(label="Akara", confidence=0.88)]


def test_rate_limit_returns_429():
    settings = AppSettings.model_validate(
        {
            "api": {
                "rate_limit": {
                    "requests": 2,
                    "window_seconds": 60,
                    "exempt_paths": ["/docs", "/redoc", "/openapi.json", "/health"],
                }
            }
        }
    )
    app = create_app(settings=settings, classifier=RateLimitClassifier())

    with TestClient(app) as client:
        first = client.post(
            "/api/v1/classify",
            files={"image": ("meal.png", b"image-1", "image/png")},
        )
        second = client.post(
            "/api/v1/classify",
            files={"image": ("meal.png", b"image-2", "image/png")},
        )
        third = client.post(
            "/api/v1/classify",
            files={"image": ("meal.png", b"image-3", "image/png")},
        )

    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.json()["detail"] == "Rate limit exceeded. Try again later."
    assert int(third.headers["Retry-After"]) >= 1
