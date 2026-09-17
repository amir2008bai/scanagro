from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.services.vision.local import runtime
from app.services.vision.local.attributes import AttributeReader
from app.services.vision.local.reference_gallery import ReferenceGallery, vehicle_crop


def test_context_crop_clips_to_frame_and_keeps_cab():
    image = np.arange(100 * 200 * 3).reshape(100, 200, 3)
    box = SimpleNamespace(x1=10, y1=5, x2=110, y2=85)
    result = vehicle_crop(image, box, 0.3, 0.1)
    assert np.array_equal(result, image[:93, :140])


def test_pipeline_passes_context_to_gallery_but_tight_crop_to_badge(monkeypatch, tmp_path):
    from app.services.vision.local.attributes import VehicleAttributes
    from app.services.vision.local.pipeline import LocalRecognitionPipeline, PipelineConfig
    from app.services.vision.local.reference_gallery import ReferenceMatch
    from app.services.vision.local.vehicle import VehicleBox

    pipe = LocalRecognitionPipeline(PipelineConfig(model_dir=tmp_path))
    box = VehicleBox(10, 5, 110, 85, 0.9, "truck", "truck")
    monkeypatch.setattr(pipe.vehicle_detector, "detect", lambda *_: [box])
    monkeypatch.setattr(pipe.plate_detector, "detect", lambda *_, **kw: [])
    badge_shapes, reference_shapes = [], []

    def badge(crop, *_):
        badge_shapes.append(crop.shape)
        return VehicleAttributes()

    def classify(crop):
        reference_shapes.append(crop.shape)
        return ReferenceMatch("tractor", 0.85, 0.15)

    monkeypatch.setattr(pipe.attribute_reader, "read", badge)
    pipe.reference_gallery = SimpleNamespace(
        available=True,
        classify=classify,
        vehicle_crop=lambda image, box: vehicle_crop(image, box, 0.3, 0.1),
    )
    result, _ = pipe.analyze_array(np.zeros((100, 200, 3), np.uint8))
    assert badge_shapes == [(80, 100, 3)]
    assert reference_shapes == [(93, 140, 3)]
    assert result.detections[0].vehicle_type == "tractor"


def test_old_gallery_uses_tight_crop_and_new_gallery_uses_manifest(monkeypatch, tmp_path):
    gallery = ReferenceGallery(tmp_path, tmp_path)
    index = {"manifest": {}}
    monkeypatch.setattr(ReferenceGallery, "index", property(lambda self: index))
    image = np.zeros((100, 200, 3), np.uint8)
    box = SimpleNamespace(x1=10, y1=5, x2=110, y2=85)
    assert gallery.vehicle_crop(image, box).shape == (80, 100, 3)
    index["manifest"]["crop_policy"] = {"margin_x": 0.3, "margin_y": 0.1}
    assert gallery.vehicle_crop(image, box).shape == (93, 140, 3)


@pytest.mark.parametrize(
    "token,expected",
    [
        ("КИРОВЕЦ", "KIROVETS"),
        ("КАМАЗ", "KAMAZ"),
        ("БЕЛАРУС", "BELARUS"),
        ("МТЗ", "BELARUS"),
        ("МАЗ", None),
        ("СТАЙЕР", None),
        ("СТАИЕР", None),
        ("STAYER", None),
    ],
)
def test_cyrillic_badge_and_decorative_text(monkeypatch, token, expected):
    monkeypatch.setattr(AttributeReader, "_body_text", lambda *_: [(token, 0.95, ())])
    result = AttributeReader(Path(".")).read(np.zeros((20, 40, 3), np.uint8), "truck")
    assert result.manufacturer == expected
    assert result.evidence[0]["text"] == token


def test_missing_cyrillic_model_is_not_silently_reported_as_no_badge(tmp_path):
    runtime.release_all()
    reader = AttributeReader(tmp_path)
    with pytest.raises(FileNotFoundError, match="fetch_models"):
        _ = reader._cyrillic_ocr


def test_model_check_only_does_not_delete_a_corrupt_file(tmp_path, monkeypatch):
    from app.services.vision.local.model_registry import CYRILLIC_OCR
    from scripts import fetch_models

    target = tmp_path / CYRILLIC_OCR.filename
    target.write_bytes(b"bad model")
    monkeypatch.setattr(fetch_models, "DOWNLOADABLE", (CYRILLIC_OCR,))
    monkeypatch.setattr("sys.argv", ["fetch_models.py", "--dir", str(tmp_path), "--check-only"])
    assert fetch_models.main() == 1
    assert target.read_bytes() == b"bad model"


def test_broken_ocr_is_not_an_unreadable_plate(monkeypatch, tmp_path):
    from app.services.vision.local.plate_ocr import PlateReader

    reader = PlateReader(tmp_path)
    blank = np.zeros((20, 40, 3), np.uint8)
    monkeypatch.setattr(reader, "build_variants", lambda _: ({"raw": blank}, None))

    def broken(*_):
        raise RuntimeError("model inference failed")

    monkeypatch.setattr(reader, "_read_cct", broken)
    monkeypatch.setattr(reader, "_read_ppocr", broken)
    with pytest.raises(RuntimeError, match="All configured plate OCR engines failed"):
        reader.read(blank)
    # A working fallback is allowed to find no text; that is not a model outage.
    monkeypatch.setattr(reader, "_read_ppocr", lambda *_: None)
    assert reader.read(blank).reason == "no_text_recognised"
