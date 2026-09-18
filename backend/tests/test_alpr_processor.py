"""Deterministic contract/geometry tests; no downloads or OCR weights needed."""
import base64
from itertools import permutations
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from app.services.alpr_processor import (
    ALPRConfig,
    ALPRError,
    ALPRProcessor,
    OpenCVCornerDetector,
    PaddleRecognizer,
    order_quad,
)


@pytest.fixture
def scene():
    image = np.full((240, 640, 3), 35, np.uint8)
    quad = np.float32([[90, 70], [550, 95], [520, 180], [100, 165]])
    cv2.fillConvexPoly(image, quad.astype(np.int32), (235, 235, 235))
    cv2.putText(image, "123ABC01", (135, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (5, 5, 5), 3)
    return image, quad


def processor(scene, outputs, **kwargs):
    detector = Mock()
    detector.detect.return_value = [scene[1]]
    reader = Mock()
    reader.recognize.side_effect = outputs
    return ALPRProcessor(recognizer=reader, detector=detector, **kwargs), reader


def test_ensemble_and_png(scene):
    module, reader = processor(scene, [("123ABC01", .75), ("123ABC02", .93), ("123ABC03", .81)])
    original = scene[0].copy()
    result = module.process_image(scene[0])
    assert result["plate_text"] == "123ABC02"
    assert result["confidence"] == .93
    assert set(result) == {"plate_text", "confidence", "warp_image_base64"}
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(result["warp_image_base64"]), np.uint8), 1)
    assert decoded.shape == (110, 470, 3)
    assert reader.recognize.call_count == 3
    assert np.array_equal(original, scene[0])


def test_order_all_permutations_and_actual_homography(scene):
    for points in permutations(scene[1]):
        assert np.allclose(order_quad(np.array(points)), scene[1])
    image = np.zeros_like(scene[0])
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for point, color in zip(scene[1], colors, strict=True):
        cv2.circle(image, tuple(point.astype(int)), 8, color, -1)
    warped = ALPRProcessor._warp(image, scene[1])
    for pixel, color in zip([warped[0, 0], warped[0, -1], warped[-1, -1], warped[-1, 0]], colors, strict=True):
        assert np.max(np.abs(pixel.astype(int) - color)) < 5


def test_automatic_detection(scene):
    quads = OpenCVCornerDetector(ALPRConfig()).detect(scene[0])
    assert quads
    assert min(np.linalg.norm(q - scene[1], axis=1).mean() for q in quads) < 8


@pytest.mark.parametrize("raw,pattern,expected", [
    ("OIBSZ", "DDDDD", "01852"), ("01852", "LLLLL", "OIBSZ"),
    ("OIBSZ", None, "OIBSZ"), ("12ЖABC01", None, ""),
    ("123/ABC", None, ""), ("123 abc-01", None, "123ABC01"),
    ("123ABC01", "DDDLLLDD", "123ABC01"), ("123ABC01", "DDDDDDDD", ""),
])
def test_corrections(scene, raw, pattern, expected):
    module, _ = processor(scene, [], config=ALPRConfig(patterns=(pattern,) if pattern else ()))
    assert module._correct(raw) == expected


def test_ambiguous_patterns_abstain(scene):
    module, _ = processor(scene, [], config=ALPRConfig(patterns=("DLLL", "LDLL")))
    assert module._correct("OOAA") == ""


def test_correction_does_not_inflate_confidence(scene):
    module, _ = processor(scene, [("I23ABC0I", .88)] * 3,
                          config=ALPRConfig(patterns=("DDDLLLDD",)))
    assert module.process_image(scene[0])["confidence"] == .88
    assert module._correct("I23ABC0I") == "123ABC01"


def test_no_detection_does_not_run_ocr():
    reader = Mock()
    result = ALPRProcessor(recognizer=reader).process_image(np.zeros((200, 600, 3), np.uint8))
    assert result == {"plate_text": "", "confidence": 0.0, "warp_image_base64": ""}
    reader.recognize.assert_not_called()


def test_unreadable_keeps_crop(scene):
    module, _ = processor(scene, [("", .99), ("123ABC01", .2), ("123ЖBC01", .99)])
    result = module.process_image(scene[0])
    assert result["plate_text"] == "" and result["confidence"] == 0
    assert result["warp_image_base64"]


def test_partial_failure_and_invalid_confidence(scene):
    module, _ = processor(scene, [RuntimeError("inference failed"), ("123ABC01", float("nan")),
                                  ("123ABC02", .9)])
    assert module.process_image(scene[0])["plate_text"] == "123ABC02"


def test_total_failure_is_error(scene):
    module, _ = processor(scene, [RuntimeError("failed")] * 3)
    with pytest.raises(ALPRError, match="All OCR"):
        module.process_image(scene[0])


@pytest.mark.parametrize("bad", [None, np.zeros((10, 10)), np.zeros((2, 2, 3), np.uint8),
                                 np.zeros((12, 12, 4), np.uint8)])
def test_bad_inputs(bad):
    with pytest.raises(ValueError):
        ALPRProcessor(recognizer=Mock()).process_image(bad)


def test_pixel_limit():
    module = ALPRProcessor(recognizer=Mock(), config=ALPRConfig(max_pixels=64))
    with pytest.raises(ValueError):
        module.process_image(np.zeros((9, 9, 3), np.uint8))


def test_bad_detector_coordinates(scene):
    module, _ = processor((scene[0], scene[1] + 1000), [])
    with pytest.raises(ALPRError, match="outside"):
        module.process_image(scene[0])


def test_local_model_required(tmp_path):
    with pytest.raises(ValueError, match="required"):
        ALPRProcessor()
    with pytest.raises(ValueError, match="local Paddle"):
        PaddleRecognizer(tmp_path)
    with pytest.raises(FileNotFoundError):
        PaddleRecognizer(tmp_path / "missing")


def test_paddle_adapter_contract(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    for name in ("inference.yml", "inference.json", "inference.pdiparams"):
        (tmp_path / name).touch()
    (tmp_path / "inference.yml").write_text("Global:\n  model_name: en_PP-OCRv4_mobile_rec\n")
    model = Mock()
    model.predict.return_value = [{"rec_text": "123ABC01", "rec_score": .9}]
    constructor = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(TextRecognition=constructor))
    monkeypatch.setenv("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "False")
    monkeypatch.setenv("DISABLE_MODEL_SOURCE_CHECK", "False")
    reader = PaddleRecognizer(tmp_path)
    constructor.assert_called_once_with(model_name="en_PP-OCRv4_mobile_rec",
                                        model_dir=str(tmp_path.resolve()), device="cpu")
    assert reader.recognize(np.zeros((110, 470, 3), np.uint8)) == ("123ABC01", .9)


def test_sr_load_and_blur_gate(tmp_path, monkeypatch, scene):
    weights = tmp_path / "FSRCNN_x2.pb"
    weights.touch()
    sr = Mock()
    sr.upsample.side_effect = lambda image: cv2.resize(image, None, fx=2, fy=2)
    from types import SimpleNamespace
    monkeypatch.setattr(cv2, "dnn_superres", SimpleNamespace(DnnSuperResImpl_create=lambda: sr), raising=False)
    module, _ = processor(scene, [], sr_model_path=weights,
                          config=ALPRConfig(sr_blur_threshold=1))
    variants = module._variants(np.full((110, 470, 3), 100, np.uint8))
    sr.readModel.assert_called_once_with(str(weights.resolve()))
    sr.setModel.assert_called_once_with("fsrcnn", 2)
    assert len(variants) == 3 and all(v.shape == (220, 940, 3) for v in variants)
    sr.upsample.assert_called_once()
    module._variants(module._warp(*scene))
    sr.upsample.assert_called_once()


def test_selects_corresponding_crop_across_candidates(scene):
    module, reader = processor(scene, [("123ABC01", .7)] * 3 + [("456DEF02", .98)] * 3)
    second = scene[1] * .5
    module._detector.detect.return_value = [scene[1], second]
    result = module.process_image(scene[0])
    assert result["plate_text"] == "456DEF02"
    crop = cv2.imdecode(np.frombuffer(base64.b64decode(result["warp_image_base64"]), np.uint8), 1)
    assert np.array_equal(crop, module._warp(scene[0], second))
    assert reader.recognize.call_count == 6
