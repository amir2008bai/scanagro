import base64
from pathlib import Path
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from app.core.config import Settings
from app.services.alpr_processor import ALPRError
from app.services.vision.local import alpr_reader, geometry
from app.services.vision.local_provider import build_config


@pytest.fixture
def reader(monkeypatch):
    fallback = Mock(accept_confidence=.55, accept_confidence_unformatted=.88)
    processor = Mock()
    processor.process_image.return_value = {
        "plate_text": "123ABC10", "confidence": .95,
        "warp_image_base64": base64.b64encode(cv2.imencode(".png", np.zeros((110, 470, 3), np.uint8))[1]).decode(),
    }
    monkeypatch.setattr(alpr_reader, "ALPRProcessor", Mock(return_value=processor))
    instance = alpr_reader.ALPRPlateReader(fallback, Path("model"), None)
    return instance, fallback, processor


def test_new_reader_returns_existing_schema(reader):
    instance, fallback, _ = reader
    result = instance.read(np.zeros((80, 300, 3), np.uint8))
    assert result.text == "123 ABC 10"
    assert result.engine == "paddleocr_alpr"
    assert result.region == "Kostanay"
    assert result.readable and result.rectified_image.shape == (110, 470, 3)
    fallback.read.assert_not_called()


def test_no_alpr_read_uses_legacy(reader):
    instance, fallback, processor = reader
    processor.process_image.return_value["plate_text"] = ""
    assert instance.read(np.zeros((80, 300, 3), np.uint8)) is fallback.read.return_value


def test_operational_failure_is_not_hidden(reader):
    instance, fallback, processor = reader
    processor.process_image.side_effect = ALPRError("broken model")
    with pytest.raises(ALPRError):
        instance.read(np.zeros((80, 300, 3), np.uint8))
    fallback.read.assert_not_called()


def test_high_confidence_badge_does_not_bypass_plate_grammar(reader):
    instance, fallback, processor = reader
    processor.process_image.return_value["plate_text"] = "KAMAZ"
    processor.process_image.return_value["confidence"] = .99
    assert instance.read(np.zeros((80, 300, 3), np.uint8)) is fallback.read.return_value


def test_square_plate_not_stretched(monkeypatch):
    quad = np.float32([[5, 5], [70, 5], [70, 60], [5, 60]])
    monkeypatch.setattr(geometry, "estimate_quad", lambda _: geometry.QuadEstimate(quad, "glyph_row", 0))
    assert alpr_reader.PlateCropCorners().detect(np.zeros((80, 100, 3), np.uint8)) == []


def test_alpr_settings_reach_pipeline():
    settings = Settings(vision_alpr_enabled=True, vision_alpr_ocr_dir=Path("model"),
                        vision_alpr_sr_path=Path("sr.pb"))
    config = build_config(settings)
    assert config.alpr_enabled and config.alpr_ocr_dir == Path("model")
    assert config.alpr_sr_path == Path("sr.pb")
