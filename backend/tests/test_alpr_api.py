from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.services.alpr_processor import ALPRError
from examples import alpr_api


@pytest.fixture
def client(monkeypatch):
    processor = Mock()
    processor.process_image.return_value = {"plate_text": "123ABC01", "confidence": .9,
                                             "warp_image_base64": ""}
    factory = Mock(return_value=processor)
    monkeypatch.setattr(alpr_api, "ALPRProcessor", factory)
    monkeypatch.setenv("ALPR_OCR_MODEL_DIR", "local-model")
    with TestClient(alpr_api.app) as test_client:
        yield test_client, processor
    factory.assert_called_once()


def png_bytes():
    return cv2.imencode(".png", np.zeros((110, 470, 3), np.uint8))[1].tobytes()


def test_api_success(client):
    http, processor = client
    for _ in range(2):
        response = http.post("/recognize", content=png_bytes())
        assert response.status_code == 200
        assert response.json()["plate_text"] == "123ABC01"
    assert processor.process_image.call_args.args[0].shape == (110, 470, 3)


def test_api_rejects_invalid_image(client):
    http, processor = client
    assert http.post("/recognize", content=b"invalid").status_code == 400
    processor.process_image.assert_not_called()


def test_api_byte_limit(client, monkeypatch):
    monkeypatch.setattr(alpr_api, "MAX_BYTES", 4)
    http, processor = client
    assert http.post("/recognize", content=b"12345").status_code == 413
    processor.process_image.assert_not_called()


def test_api_pixel_limit_before_decode(client, monkeypatch):
    monkeypatch.setattr(alpr_api, "MAX_PIXELS", 100)
    http, processor = client
    assert http.post("/recognize", content=png_bytes()).status_code == 413
    processor.process_image.assert_not_called()


def test_api_failure_hides_internal_details(client):
    http, processor = client
    processor.process_image.side_effect = ALPRError("private path or content")
    response = http.post("/recognize", content=png_bytes())
    assert response.status_code == 503
    assert "private" not in response.text
