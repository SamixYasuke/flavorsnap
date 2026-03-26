from __future__ import annotations


def test_classify_image_returns_ranked_predictions(client):
    response = client.post(
        "/api/v1/classify",
        files={"image": ("meal.png", b"fake-image-bytes", "image/png")},
        data={"top_k": "2", "resize": "256", "center_crop": "true", "normalize": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["prediction"] == "Moi Moi"
    assert payload["confidence"] == 0.91
    assert len(payload["predictions"]) == 2
    assert payload["predictions"][0]["label"] == "Moi Moi"
    assert payload["processing_time_ms"] >= 0
    assert payload["preprocessing"] == {
        "resize": 256,
        "center_crop": True,
        "normalize": True,
        "top_k": 2,
    }
    assert payload["filename"] == "meal.png"
    assert response.headers["X-Request-ID"]


def test_classify_rejects_empty_upload(client):
    response = client.post(
        "/api/v1/classify",
        files={"image": ("empty.png", b"", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is empty."


def test_classify_rejects_unsupported_media_type(client):
    response = client.post(
        "/api/v1/classify",
        files={"image": ("meal.gif", b"gif-bytes", "image/gif")},
    )

    assert response.status_code == 415
    assert "Unsupported media type" in response.json()["detail"]


def test_classify_returns_400_for_invalid_image(client):
    response = client.post(
        "/api/v1/classify",
        files={"image": ("bad.png", b"bad-image", "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image."
