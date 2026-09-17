"""Wiring tests for the local provider.

These deliberately avoid loading any model: they cover the contracts that hold the
pipeline to the rest of the service — the adapter registration, the schema, and the
mapping onto database columns — so they run anywhere, including CI without weights.
"""

import pytest

from app.schemas.vision import NormalizedBBox, PlateObservation, VisionDetection


def plate(**overrides) -> PlateObservation:
    payload = {
        "bbox": NormalizedBBox(x1=0.4, y1=0.5, x2=0.46, y2=0.53),
        "detection_confidence": 0.91,
        "ocr_confidence": 0.98,
        "text": "160 ALV 10",
        "text_raw": "160ALV10",
        "readable": True,
        "plate_format": "kz_civil_2012",
        "region": "Kostanay",
        "ocr_engine": "ppocr",
        "ocr_variant": "rectified",
        "rectified": True,
        "perspective_skew": 1.18,
        "status": "accepted",
    }
    payload.update(overrides)
    return PlateObservation(**payload)


def vehicle(**overrides) -> VisionDetection:
    payload = {
        "vehicle_type": "truck",
        "bbox": NormalizedBBox(x1=0.1, y1=0.2, x2=0.8, y2=0.9),
        "vehicle_confidence": 0.89,
        "license_plate": "160 ALV 10",
        "plate_readable": True,
        "plate": plate(),
    }
    payload.update(overrides)
    return VisionDetection(**payload)


def test_factory_rejects_unknown_providers():
    from app.services.vision.factory import get_vision_provider

    with pytest.raises(RuntimeError, match="Unknown VISION_PROVIDER"):
        get_vision_provider("nonsense")


def test_local_provider_is_registered_and_names_itself():
    """`local` resolves through the same adapter layer as the other providers.

    Construction must not touch the weights: only `analyze` is allowed to load them, so
    a service configured for `local` still starts when the model volume is cold.
    """
    pytest.importorskip("cv2", reason="opencv is only installed with the local pipeline")
    from app.services.vision.factory import get_vision_provider

    provider = get_vision_provider("local", model="test-pipeline-id")
    assert provider.name == "local"
    assert provider.model == "test-pipeline-id"


def test_mock_provider_still_works_unchanged():
    """The mock path must keep working: it is what the wiring tests rely on."""
    from app.services.vision.factory import get_vision_provider

    provider = get_vision_provider("mock")
    assert provider.name == "mock"


def test_plate_box_is_separate_from_vehicle_box():
    det = vehicle()
    assert det.plate is not None
    assert (det.plate.bbox.x1, det.plate.bbox.y1) != (det.bbox.x1, det.bbox.y1)
    # The plate must sit inside the machine it was attributed to.
    assert det.bbox.x1 <= det.plate.bbox.x1 and det.plate.bbox.x2 <= det.bbox.x2


def test_stage_confidences_stay_distinct():
    det = vehicle(
        vehicle_confidence=0.42, plate=plate(detection_confidence=0.95, ocr_confidence=0.30)
    )
    assert det.vehicle_confidence == 0.42
    assert det.plate.detection_confidence == 0.95
    assert det.plate.ocr_confidence == 0.30
    # The legacy field mirrors the vehicle score, never the OCR score.
    assert det.confidence == 0.42


def test_legacy_confidence_and_vehicle_confidence_mirror_each_other():
    assert vehicle(confidence=0.7, vehicle_confidence=None).vehicle_confidence == 0.7
    assert vehicle(confidence=None, vehicle_confidence=0.7).confidence == 0.7


def test_unreadable_plate_carries_no_text_but_keeps_the_raw_read():
    observation = plate(
        readable=False,
        text="597 AQ 10",
        text_raw="597AQ10",
        plate_format=None,
        status="no_matching_plate_format",
    )
    assert observation.text is None, "an unreadable plate must not expose a plate string"
    assert observation.text_raw == "597AQ10", "the raw read is kept for human review"


def test_readable_detection_requires_text():
    with pytest.raises(ValueError):
        vehicle(plate_readable=True, license_plate=None)


def test_unreadable_detection_clears_the_plate_string():
    det = vehicle(plate_readable=False, license_plate="160 ALV 10")
    assert det.license_plate is None


def test_detection_row_mapping_keeps_every_stage():
    from uuid import uuid4

    from app.services.processing import build_detection

    image_id, job_id = uuid4(), uuid4()
    row = build_detection(image_id, job_id, vehicle())
    assert row.vehicle_confidence == 0.89
    assert row.plate_bbox_x1 == 0.4 and row.plate_bbox_y2 == 0.53
    assert row.bbox_x1 == 0.1, "vehicle box must not be overwritten by the plate box"
    assert row.plate_ocr_confidence == 0.98
    assert row.plate_detection_confidence == 0.91
    assert row.plate_format == "kz_civil_2012"
    assert row.plate_text_raw == "160ALV10"
    assert row.plate_rectified is True
    assert row.plate_perspective_skew == 1.18


def test_detection_row_mapping_tolerates_providers_without_plates():
    """mock and openai_compatible produce no plate geometry; the columns stay null."""
    from uuid import uuid4

    from app.services.processing import build_detection

    row = build_detection(
        uuid4(), uuid4(), vehicle(plate=None, plate_readable=False, license_plate=None)
    )
    assert row.plate_bbox_x1 is None
    assert row.plate_quad is None
    assert row.plate_ocr_confidence is None
    assert row.vehicle_type == "truck"


def test_extra_carries_the_audit_trail():
    from uuid import uuid4

    from app.services.processing import build_detection

    det = vehicle(
        plate=plate(
            alternatives=[{"engine": "ppocr", "text": "160ALV10"}], disagreement_positions=[4]
        ),
        evidence=[{"source": "body_text_ocr", "text": "KAMAZ"}],
    )
    row = build_detection(uuid4(), uuid4(), det)
    assert row.extra["plate_ocr_candidates"][0]["text"] == "160ALV10"
    assert row.extra["plate_ocr_disagreement_positions"] == [4]
    assert row.extra["attribute_evidence"][0]["text"] == "KAMAZ"


def test_brand_matching_refuses_an_ambiguous_short_token():
    """`MAZ` read off a KAMAZ grille must not be asserted as the manufacturer."""
    pytest.importorskip("cv2", reason="opencv is only installed with the local pipeline")
    from pathlib import Path
    from unittest.mock import patch

    import numpy as np

    from app.services.vision.local.attributes import AttributeReader

    reader = AttributeReader(Path("."), enabled=True)
    blank = np.zeros((64, 128, 3), np.uint8)

    with patch.object(AttributeReader, "_body_text", return_value=[("MA3", 0.85, ())]):
        result = reader.read(blank, "truck")
    assert result.manufacturer is None
    assert result.evidence[0]["rejected"] == "ambiguous_brand_token"
    assert "KAMAZ" in result.evidence[0]["could_be"]

    # The unambiguous full token is accepted, and outranks the shorter match.
    with patch.object(
        AttributeReader, "_body_text", return_value=[("MA3", 0.99, ()), ("KAMAZ", 0.80, ())]
    ):
        result = reader.read(blank, "truck")
    assert result.manufacturer == "KAMAZ"
    assert result.vehicle_type == "truck"


def test_no_brand_text_means_no_manufacturer():
    pytest.importorskip("cv2", reason="opencv is only installed with the local pipeline")
    from pathlib import Path
    from unittest.mock import patch

    import numpy as np

    from app.services.vision.local.attributes import AttributeReader

    reader = AttributeReader(Path("."), enabled=True)
    with patch.object(
        AttributeReader, "_body_text", return_value=[("OLZHAAGRO", 0.9, ()), ("2026", 0.99, ())]
    ):
        result = reader.read(np.zeros((64, 128, 3), np.uint8), "truck")
    assert result.manufacturer is None
    assert result.model is None


def test_plate_assignment_picks_the_smallest_containing_vehicle():
    pytest.importorskip("cv2", reason="opencv is only installed with the local pipeline")
    from app.services.vision.local.pipeline import LocalRecognitionPipeline
    from app.services.vision.local.plate_detect import PlateBox
    from app.services.vision.local.vehicle import VehicleBox

    big = VehicleBox(0, 0, 1000, 800, 0.9, "truck", "truck")
    small = VehicleBox(100, 100, 400, 500, 0.8, "truck", "truck")
    inside_both = PlateBox(200, 300, 260, 330, 0.9, "vehicle")
    outside = PlateBox(2000, 2000, 2060, 2030, 0.9, "frame")

    assignment = LocalRecognitionPipeline._assign([inside_both, outside], [big, small])
    assert assignment[1] == [inside_both], "the more specific box wins"
    assert assignment[0] == []
    assert assignment[-1] == [outside], "a plate in no box stays unassigned"


def test_model_registry_is_reproducible():
    """Every pinned artefact must be fetchable and verifiable without guesswork."""
    from app.services.vision.local.model_registry import DOWNLOADABLE, VENDORED_NOTES

    assert DOWNLOADABLE, "the registry must not be empty"
    filenames = [spec.filename for spec in DOWNLOADABLE]
    assert len(filenames) == len(set(filenames)), "two artefacts would collide on disk"
    for spec in DOWNLOADABLE:
        assert spec.url.startswith("https://"), f"{spec.filename} must be fetched over TLS"
        assert spec.license, f"{spec.filename} has no recorded licence"
        assert spec.source.startswith("https://"), f"{spec.filename} has no recorded source"
        assert spec.size_bytes > 0
        # The one exception is documented inline: a tiny JSON validated by parsing.
        if not spec.sha256:
            assert spec.filename.endswith(".json"), f"{spec.filename} must pin a checksum"
        else:
            assert len(spec.sha256) == 64 and all(c in "0123456789abcdef" for c in spec.sha256)
    assert "text_ocr_ppocr" in VENDORED_NOTES


def test_verify_rejects_a_substituted_weight(tmp_path):
    """A checksum mismatch must be a hard failure, not a warning."""
    from app.services.vision.local.model_registry import YOLO_V9_PLATE, verify

    missing = tmp_path / YOLO_V9_PLATE.filename
    with pytest.raises(FileNotFoundError, match="fetch_models"):
        verify(YOLO_V9_PLATE, missing)

    missing.write_bytes(b"not the real weights")
    with pytest.raises(RuntimeError, match="Checksum mismatch"):
        verify(YOLO_V9_PLATE, missing)
