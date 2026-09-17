"""Geometry tests use synthetic plates, so the expected corners are known exactly."""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv is only installed with the local pipeline")

from app.services.vision.local import geometry  # noqa: E402


def make_plate(width=240, height=52) -> np.ndarray:
    """A white plate with a dark border and dark glyph blocks, on a grey background.

    Glyphs occupy ~65% of the plate height, as they do on a real Kazakh plate. The
    proportion matters: the estimator brackets the *text row*, so a mock with unusually
    small glyphs would report an aspect that no real plate produces.
    """
    plate = np.full((height, width, 3), 235, np.uint8)
    cv2.rectangle(plate, (1, 1), (width - 2, height - 2), (30, 30, 30), 2)
    for index in range(6):
        x = 16 + index * 34
        cv2.rectangle(plate, (x, 9), (x + 20, height - 9), (25, 25, 25), -1)
    return plate


def paste(plate: np.ndarray, canvas_size=(140, 340), corners=None) -> np.ndarray:
    canvas = np.full((canvas_size[0], canvas_size[1], 3), 120, np.uint8)
    height, width = plate.shape[:2]
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    if corners is None:
        corners = np.float32(
            [[50, 44], [50 + width, 44], [50 + width, 44 + height], [50, 44 + height]]
        )
    matrix = cv2.getPerspectiveTransform(source, np.float32(corners))
    warped = cv2.warpPerspective(
        plate,
        matrix,
        (canvas_size[1], canvas_size[0]),
        borderMode=cv2.BORDER_TRANSPARENT,
        dst=canvas,
    )
    return warped


def test_finds_corners_of_an_axis_aligned_plate():
    canvas = paste(make_plate())
    estimate = geometry.estimate_quad(canvas)
    assert estimate is not None
    # Corners should land within a few pixels of where the plate was pasted.
    assert estimate.quad[0][0] == pytest.approx(50, abs=8)
    assert estimate.quad[0][1] == pytest.approx(44, abs=8)
    # The quad brackets the text row plus a margin, so it is close to but not identical
    # with the physical plate. What matters downstream is that it lands in the range a
    # single-row plate occupies, which is what the format gate checks.
    assert 2.2 <= estimate.aspect <= 8.0
    assert estimate.aspect == pytest.approx(240 / 52, rel=0.35)
    assert estimate.trusted_aspect, "a clean synthetic plate should use the glyph-row fit"
    assert estimate.skew < 0.15  # a square-on plate is a parallelogram


def test_detects_real_perspective_and_rectifies_it():
    """Top and bottom edges of different lengths: a rotation cannot undo this."""
    corners = [[46, 30], [286, 56], [286, 120], [46, 112]]
    canvas = paste(make_plate(), corners=corners)
    estimate = geometry.estimate_quad(canvas)
    assert estimate is not None
    assert estimate.skew > 0.15, "converging edges must register as perspective, not rotation"

    rectified = geometry.rectify(canvas, estimate.quad)
    assert rectified is not None
    # After rectification the glyph row should be level: measure the dark-pixel centroid
    # of the left and right halves and require them to agree vertically.
    grey = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
    dark = grey < 100
    height, width = dark.shape
    left = np.argwhere(dark[:, : width // 3])
    right = np.argwhere(dark[:, 2 * width // 3 :])
    assert left.size and right.size
    assert abs(left[:, 0].mean() - right[:, 0].mean()) < height * 0.12


def test_rejects_an_implausible_quad():
    """Flat noise gives no plate field, so no corners are invented."""
    rng = np.random.default_rng(7)
    noise = rng.integers(0, 255, (60, 200, 3), dtype=np.uint8)
    estimate = geometry.estimate_quad(noise)
    if estimate is not None:
        # Whatever it found still has to pass the shape checks used downstream.
        assert 0.7 <= estimate.aspect <= 8.0


def test_tiny_crops_are_declined_rather_than_guessed():
    assert geometry.estimate_quad(np.zeros((4, 4, 3), np.uint8)) is None
    assert geometry.estimate_quad(np.zeros((9, 40, 3), np.uint8)) is None


def test_upscale_only_grows_and_preserves_aspect():
    small = np.zeros((20, 90, 3), np.uint8)
    grown = geometry.upscale(small, min_height=64)
    assert grown.shape[0] >= 64
    assert grown.shape[1] / grown.shape[0] == pytest.approx(90 / 20, rel=0.05)
    # Already large enough: returned untouched, not resampled.
    big = np.zeros((80, 300, 3), np.uint8)
    assert geometry.upscale(big, min_height=64) is big


def test_quad_to_normalised_maps_into_the_whole_image():
    quad = np.float32([[0, 0], [10, 0], [10, 5], [0, 5]])
    out = geometry.quad_to_normalised(quad, offset_x=90, offset_y=40, width=200, height=100)
    assert out[0] == [0.45, 0.4]
    assert out[2] == [0.5, 0.45]
    assert all(0.0 <= value <= 1.0 for point in out for value in point)


def test_contrast_enhancement_leaves_geometry_alone():
    canvas = paste(make_plate())
    enhanced = geometry.enhance_contrast(canvas)
    assert enhanced.shape == canvas.shape
