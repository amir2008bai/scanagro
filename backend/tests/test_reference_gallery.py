"""Reference-gallery tests.

The decision logic is what matters here — when the gallery answers, when it refuses, and
whether an excluded reference really stops influencing the result — so these build a tiny
index by hand and stub the encoder. No weights, no network, runs anywhere.
"""

import numpy as np
import pytest

pytest.importorskip("cv2", reason="opencv is only installed with the local pipeline")

from app.services.vision.local import runtime  # noqa: E402
from app.services.vision.local.reference_gallery import ReferenceGallery  # noqa: E402


def unit(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


@pytest.fixture(autouse=True)
def clean_runtime():
    """Sessions and indexes are cached per process; tests must not inherit each other's."""
    runtime.release_all()
    yield
    runtime.release_all()


def build(monkeypatch, entries, query, tmp_path, **kwargs):
    """entries: (label, name, source, vector). `query` is what the encoder will return."""
    gallery = ReferenceGallery(tmp_path, tmp_path, **kwargs)
    index = {
        "vectors": np.stack([e[3] for e in entries]).astype(np.float32),
        "labels": [e[0] for e in entries],
        "names": [e[1] for e in entries],
        "sources": [e[2] for e in entries],
        "manifest": {"references": []},
    }
    monkeypatch.setattr(type(gallery), "index", property(lambda self: index))
    monkeypatch.setattr(gallery.encoder, "embed", lambda image: query)
    return gallery


def test_accepts_a_clear_match(monkeypatch, tmp_path):
    entries = [
        ("tractor", "kirovets.jpg", "site", unit(1, 0, 0)),
        ("truck", "kamaz.jpg", "site", unit(0, 1, 0)),
    ]
    gallery = build(monkeypatch, entries, unit(0.99, 0.14, 0), tmp_path)
    match = gallery.classify(np.zeros((32, 32, 3), np.uint8))
    assert match is not None
    assert match.reason == "accepted"
    assert match.vehicle_type == "tractor"
    assert match.similarity > 0.9
    assert match.neighbours[0]["reference"] == "kirovets.jpg"


def test_refuses_when_nothing_is_similar_enough(monkeypatch, tmp_path):
    """A machine unlike anything in the gallery must not be forced into a class."""
    entries = [
        ("tractor", "kirovets.jpg", "site", unit(1, 0, 0)),
        ("truck", "kamaz.jpg", "site", unit(0, 1, 0)),
    ]
    gallery = build(monkeypatch, entries, unit(0, 0, 1), tmp_path, min_similarity=0.62)
    match = gallery.classify(np.zeros((32, 32, 3), np.uint8))
    assert match is not None
    assert match.reason == "below_similarity_floor"


def test_refuses_when_two_classes_are_too_close(monkeypatch, tmp_path):
    """This is the rear-view case: a light truck's body and a trailer look alike."""
    entries = [
        ("truck_light", "gaz_rear.jpg", "site", unit(1, 0.02, 0)),
        ("trailer", "trailer_rear.jpg", "site", unit(1, 0.0, 0.02)),
    ]
    gallery = build(
        monkeypatch, entries, unit(1, 0.01, 0.01), tmp_path, min_similarity=0.5, min_margin=0.04
    )
    match = gallery.classify(np.zeros((32, 32, 3), np.uint8))
    assert match is not None
    assert match.reason == "classes_too_close"
    assert match.margin < 0.04


def test_excluding_a_reference_changes_the_answer(monkeypatch, tmp_path):
    """Without this, scoring a frame against a crop of itself would be self-measurement."""
    entries = [
        ("tractor", "frame5.jpg", "site_frames", unit(1, 0, 0)),
        ("truck", "frame1.jpg", "site_frames", unit(0.9, 0.44, 0)),
    ]
    gallery = build(monkeypatch, entries, unit(1, 0, 0), tmp_path, min_margin=0.0)

    assert gallery.classify(np.zeros((32, 32, 3), np.uint8)).vehicle_type == "tractor"
    excluded = gallery.classify(np.zeros((32, 32, 3), np.uint8), exclude_sources={"frame5.jpg"})
    assert excluded.vehicle_type == "truck", "the excluded reference still decided the answer"
    assert all(n["reference"] != "frame5.jpg" for n in excluded.neighbours)


def test_excluding_everything_yields_no_answer(monkeypatch, tmp_path):
    entries = [("tractor", "only.jpg", "site_frames", unit(1, 0, 0))]
    gallery = build(monkeypatch, entries, unit(1, 0, 0), tmp_path)
    assert gallery.classify(np.zeros((32, 32, 3), np.uint8), exclude_sources={"only.jpg"}) is None


def test_a_single_outlier_cannot_outvote_a_consistent_class(monkeypatch, tmp_path):
    """Weighted top-k voting, so one lucky neighbour does not carry the decision."""
    entries = [
        ("truck", "a.jpg", "site", unit(1, 0.30, 0)),
        ("truck", "b.jpg", "site", unit(1, 0.31, 0)),
        ("truck", "c.jpg", "site", unit(1, 0.32, 0)),
        ("tractor", "d.jpg", "site", unit(1, 0.28, 0)),
    ]
    gallery = build(monkeypatch, entries, unit(1, 0.29, 0), tmp_path, min_margin=0.0)
    assert gallery.classify(np.zeros((32, 32, 3), np.uint8)).vehicle_type == "truck"


def test_missing_index_disables_the_stage(tmp_path):
    gallery = ReferenceGallery(tmp_path / "nothing-here", tmp_path)
    assert gallery.available is False
    assert gallery.classify(np.zeros((32, 32, 3), np.uint8)) is None
    assert gallery.describe() == {"available": False, "references": 0, "classes": {}}


def test_describe_counts_classes_and_sources(monkeypatch, tmp_path):
    entries = [
        ("tractor", "a.jpg", "site_frames", unit(1, 0, 0)),
        ("tractor", "b.jpg", "web", unit(0, 1, 0)),
        ("truck", "c.jpg", "web", unit(0, 0, 1)),
    ]
    gallery = build(monkeypatch, entries, unit(1, 0, 0), tmp_path)
    described = gallery.describe()
    assert described["references"] == 3
    assert described["classes"] == {"tractor": 2, "truck": 1}
    assert described["sources"] == {"site_frames": 1, "web": 2}


def test_empty_crop_is_declined(monkeypatch, tmp_path):
    entries = [("tractor", "a.jpg", "site", unit(1, 0, 0))]
    gallery = build(monkeypatch, entries, unit(1, 0, 0), tmp_path)
    assert gallery.classify(np.zeros((0, 0, 3), np.uint8)) is None
    assert gallery.classify(None) is None


# ----- how the pipeline combines the three sources of a type ----------------------


class _Attributes:
    def __init__(self, vehicle_type=None, manufacturer_confidence=None):
        self.vehicle_type = vehicle_type
        self.manufacturer = "KAMAZ" if vehicle_type else None
        self.manufacturer_confidence = manufacturer_confidence
        self.evidence = []


def _pipeline_stub(gallery):
    from app.services.vision.local.pipeline import LocalRecognitionPipeline

    stub = LocalRecognitionPipeline.__new__(LocalRecognitionPipeline)
    stub.reference_gallery = gallery
    return stub


def _vehicle_box():
    from app.services.vision.local.vehicle import VehicleBox

    return VehicleBox(0, 0, 100, 100, 0.77, "truck", "truck")


def test_type_falls_back_to_the_detector_without_a_gallery():
    stub = _pipeline_stub(None)
    crop = np.zeros((32, 32, 3), np.uint8)
    vehicle_type, source, confidence, evidence = stub._resolve_type(
        crop, _vehicle_box(), _Attributes()
    )
    assert (vehicle_type, source) == ("truck", "detector")
    assert confidence == 0.77
    assert evidence == []


def test_badge_text_beats_the_detector_but_loses_to_the_gallery(monkeypatch, tmp_path):
    crop = np.zeros((32, 32, 3), np.uint8)
    badge = _Attributes(vehicle_type="truck", manufacturer_confidence=0.99)

    # No gallery: the badge decides.
    stub = _pipeline_stub(None)
    assert stub._resolve_type(crop, _vehicle_box(), badge)[1] == "badge_text"

    # A confident gallery overrides it, because it looks at the whole machine.
    entries = [("trailer", "t.jpg", "site", unit(1, 0, 0))]
    gallery = build(monkeypatch, entries, unit(1, 0, 0), tmp_path)
    vehicle_type, source, confidence, evidence = _pipeline_stub(gallery)._resolve_type(
        crop, _vehicle_box(), badge
    )
    assert (vehicle_type, source) == ("trailer", "reference_gallery")
    assert confidence > 0.9
    assert evidence[0]["source"] == "reference_gallery"


def test_an_abstaining_gallery_still_records_why(monkeypatch, tmp_path):
    """The refusal has to be visible, or a `detector` type looks like a considered answer."""
    entries = [
        ("tractor", "a.jpg", "site", unit(1, 0, 0)),
        ("truck", "b.jpg", "site", unit(0, 1, 0)),
    ]
    gallery = build(monkeypatch, entries, unit(0, 0, 1), tmp_path, min_similarity=0.62)
    vehicle_type, source, _confidence, evidence = _pipeline_stub(gallery)._resolve_type(
        np.zeros((32, 32, 3), np.uint8), _vehicle_box(), _Attributes()
    )
    assert (vehicle_type, source) == ("truck", "detector")
    assert evidence[0]["status"] == "below_similarity_floor"


def test_close_gallery_match_cannot_override_detector_and_badge():
    from types import SimpleNamespace

    from app.services.vision.local.reference_gallery import ReferenceMatch

    gallery = SimpleNamespace(available=True, classify=lambda _: ReferenceMatch(
        "trailer", .87, .05, [], "accepted"
    ))
    result = _pipeline_stub(gallery)._resolve_type(
        np.zeros((32, 32, 3), np.uint8), _vehicle_box(), _Attributes("truck", .99)
    )
    assert result[:2] == ("truck", "badge_text")
    assert result[3][0]["status"] == "conflict_with_detector_and_badge"


def test_weak_badge_cannot_veto_gallery():
    from types import SimpleNamespace

    from app.services.vision.local.reference_gallery import ReferenceMatch

    gallery = SimpleNamespace(available=True, classify=lambda _: ReferenceMatch(
        "trailer", .87, .05, [], "accepted"
    ))
    result = _pipeline_stub(gallery)._resolve_type(
        np.zeros((32, 32, 3), np.uint8), _vehicle_box(), _Attributes("truck", .60)
    )
    assert result[:2] == ("trailer", "reference_gallery")


def test_a_broken_gallery_does_not_fail_the_image(monkeypatch, tmp_path):
    entries = [("tractor", "a.jpg", "site", unit(1, 0, 0))]
    gallery = build(monkeypatch, entries, unit(1, 0, 0), tmp_path)

    def explode(*_args, **_kwargs):
        raise RuntimeError("corrupt index")

    monkeypatch.setattr(gallery, "classify", explode)
    vehicle_type, source, _confidence, _evidence = _pipeline_stub(gallery)._resolve_type(
        np.zeros((32, 32, 3), np.uint8), _vehicle_box(), _Attributes()
    )
    assert (vehicle_type, source) == ("truck", "detector")
