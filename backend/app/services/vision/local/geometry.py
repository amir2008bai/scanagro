"""Stage 3 — recover the four plate corners and remove the perspective.

A plate photographed from the side is not merely rotated: its top and bottom edges stop
being parallel, so a rotation leaves the glyphs sheared and unevenly scaled. We isolate
the plate field inside the detector box, sample its boundary column by column and row by
row, fit a line to each of the four edges and intersect them. That yields a genuine
quadrilateral, which ``cv2.getPerspectiveTransform`` turns into a fronto-parallel crop.

The estimate is validated before use (position, plausible aspect, plausible area). When
validation fails we return ``None`` and the caller keeps the unrectified crop, because an
unreliable warp is worse than no warp.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

Quad = np.ndarray  # (4, 2) float32, ordered top-left, top-right, bottom-right, bottom-left


@dataclass(slots=True)
class QuadEstimate:
    quad: Quad
    method: str
    #: How far the quad departs from a parallelogram, in units of its own height.
    #: Near zero means a rotation would have sufficed; larger means real perspective.
    skew: float

    @property
    def trusted_aspect(self) -> bool:
        """Whether `aspect` is sound enough to gate a decision on.

        Only the glyph-row fit measures the plate itself. The bright-field fallback can
        latch onto bumper or shadow merged with the plate, so its aspect is good enough to
        warp by but not good enough to reject a read with.
        """
        return self.method == "glyph_row"

    @property
    def aspect(self) -> float:
        """The plate's true width/height, measured along its own edges.

        This is the figure to compare against a physical plate's proportions. The
        detector's axis-aligned box is not: a plate seen at an angle has a much taller
        bounding box than its 520x110 mm shape would suggest.
        """
        quad = self.quad
        edge_w = (
            float(np.linalg.norm(quad[1] - quad[0])) + float(np.linalg.norm(quad[2] - quad[3]))
        ) / 2
        edge_h = (
            float(np.linalg.norm(quad[3] - quad[0])) + float(np.linalg.norm(quad[2] - quad[1]))
        ) / 2
        return edge_w / edge_h if edge_h > 1e-6 else 0.0


def _plate_masks(gray: np.ndarray) -> list[np.ndarray]:
    """Candidate binary masks of the bright plate field, best first."""
    height, width = gray.shape
    blurred = cv2.bilateralFilter(gray, 7, 50, 50)
    block = max(15, (min(height, width) // 2) * 2 + 1)
    binaries = [
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, -8
        ),
    ]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, width // 12), max(3, height // 4)))
    masks = []
    for binary in binaries:
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, 8)
        best, best_score = None, -1e9
        for index in range(1, count):
            area = stats[index, cv2.CC_STAT_AREA]
            if area < 0.10 * height * width or area > 0.985 * height * width:
                continue
            cx, cy = centroids[index]
            # The detector box is centred on the plate, so the plate field should be too.
            offset = ((cx - width / 2) / width) ** 2 + ((cy - height / 2) / height) ** 2
            score = area / (height * width) - 2.5 * offset
            if score > best_score:
                best_score, best = score, index
        if best is not None:
            masks.append((labels == best).astype(np.uint8) * 255)
    return masks


def _fit_line(points: list[tuple[int, int]], horizontal: bool):
    """Huber-robust line fit, returned as (a, b, c) with a*x + b*y + c = 0."""
    if len(points) < 6:
        return None
    array = np.asarray(points, np.float32)
    vx, vy, x0, y0 = cv2.fitLine(array, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    if horizontal and abs(vx) < 1e-6:
        return None
    if not horizontal and abs(vy) < 1e-6:
        return None
    return float(-vy), float(vx), float(vy * x0 - vx * y0)


def _intersect(first, second):
    if first is None or second is None:
        return None
    a1, b1, c1 = first
    a2, b2, c2 = second
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    return np.array([(b1 * c2 - b2 * c1) / det, (c1 * a2 - c2 * a1) / det], np.float32)


def _trim(sequence: list, fraction: float = 0.12) -> list:
    """Discard the ends, where boundary samples ride the plate's rounded corners."""
    total = len(sequence)
    cut = int(total * fraction)
    return sequence[cut : total - cut] if total - 2 * cut >= 6 else sequence


def _quad_from_mask(mask: np.ndarray) -> Quad | None:
    height, width = mask.shape
    top, bottom, left, right = [], [], [], []
    for x in range(width):
        ys = np.flatnonzero(mask[:, x])
        if ys.size:
            top.append((x, int(ys[0])))
            bottom.append((x, int(ys[-1])))
    for y in range(height):
        xs = np.flatnonzero(mask[y])
        if xs.size:
            left.append((int(xs[0]), y))
            right.append((int(xs[-1]), y))
    line_top = _fit_line(_trim(top), True)
    line_bottom = _fit_line(_trim(bottom), True)
    line_left = _fit_line(_trim(left), False)
    line_right = _fit_line(_trim(right), False)
    corners = [
        _intersect(line_top, line_left),
        _intersect(line_top, line_right),
        _intersect(line_bottom, line_right),
        _intersect(line_bottom, line_left),
    ]
    if any(corner is None for corner in corners):
        return None
    return np.array(corners, np.float32)


def _skew(quad: Quad) -> float:
    top = quad[1] - quad[0]
    bottom = quad[2] - quad[3]
    height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    if height < 1e-6:
        return 0.0
    # Difference between the two horizontal edges; exactly zero for a parallelogram.
    return float(np.linalg.norm(top - bottom) / height)


def _valid(quad: Quad | None, height: int, width: int, min_area_fraction: float = 0.12) -> bool:
    if quad is None or not np.isfinite(quad).all():
        return False
    if (quad[:, 0] < -0.25 * width).any() or (quad[:, 0] > 1.25 * width).any():
        return False
    if (quad[:, 1] < -0.25 * height).any() or (quad[:, 1] > 1.25 * height).any():
        return False
    if cv2.contourArea(quad.astype(np.float32)) < min_area_fraction * height * width:
        return False
    edge_w = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    edge_h = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    if edge_h < 5 or not (0.7 <= edge_w / edge_h <= 8.0):
        return False
    return True


def _glyph_boxes(gray: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Bounding boxes of the dark characters sitting on the bright plate field.

    Working from the glyphs rather than from the plate's outline is what makes this
    robust here: the detector box often includes bumper and shadow that a brightness
    threshold happily merges with the plate, whereas the characters are unambiguous —
    dark, similar in height, and in a row.
    """
    height, width = gray.shape
    equalised = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
    binary = cv2.threshold(equalised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = (
            stats[index, cv2.CC_STAT_LEFT],
            stats[index, cv2.CC_STAT_TOP],
            stats[index, cv2.CC_STAT_WIDTH],
            stats[index, cv2.CC_STAT_HEIGHT],
            stats[index, cv2.CC_STAT_AREA],
        )
        if not (0.16 * height <= h <= 0.80 * height):
            continue
        if w < 2 or w > 0.35 * width:
            continue
        if not (0.10 <= w / h <= 1.6):
            continue
        if area < 0.25 * w * h:  # hollow outlines are frame fragments, not glyphs
            continue
        boxes.append((int(x), int(y), int(w), int(h)))
    if len(boxes) < 3:
        return []
    # Keep the dominant height cluster: characters on one plate are near-identical in
    # height, while stray blobs from the bumper or shadow are not.
    heights = np.array([b[3] for b in boxes], float)
    median = float(np.median(heights))
    boxes = [b for b in boxes if 0.62 * median <= b[3] <= 1.45 * median]
    return sorted(boxes, key=lambda b: b[0]) if len(boxes) >= 3 else []


def _quad_from_glyphs(
    boxes: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> Quad | None:
    """Fit the text row's top and bottom edges, then close the quad on the extremes.

    The two horizontal edges are fitted independently, so when the plate recedes the
    fitted lines converge exactly as the plate's own edges do. That convergence is the
    part a rotation cannot reproduce.
    """
    height, width = shape
    tops = [(x + w / 2, y) for x, y, w, h in boxes]
    bottoms = [(x + w / 2, y + h) for x, y, w, h in boxes]
    line_top = _fit_line(tops, True)
    line_bottom = _fit_line(bottoms, True)
    if line_top is None or line_bottom is None:
        return None

    glyph_h = float(np.median([b[3] for b in boxes]))
    glyph_w = float(np.median([b[2] for b in boxes]))
    # Reach past the outermost glyphs to take in the plate border, the KZ band and the
    # region box, which is what the OCR models were trained to see.
    left_x = min(b[0] for b in boxes) - 0.9 * glyph_w
    right_x = max(b[0] + b[2] for b in boxes) + 1.6 * glyph_w
    left_x = float(np.clip(left_x, -0.15 * width, width))
    right_x = float(np.clip(right_x, 0.0, 1.15 * width))
    if right_x - left_x < 3 * glyph_w:
        return None

    pad = 0.28 * glyph_h

    def at(line, x, offset):
        a, b, c = line
        if abs(b) < 1e-9:
            return None
        return np.array([x, (-a * x - c) / b + offset], np.float32)

    corners = [
        at(line_top, left_x, -pad),
        at(line_top, right_x, -pad),
        at(line_bottom, right_x, pad),
        at(line_bottom, left_x, pad),
    ]
    if any(corner is None for corner in corners):
        return None
    return np.array(corners, np.float32)


def estimate_quad(crop_bgr: np.ndarray, work_height: int = 160) -> QuadEstimate | None:
    """Corners of the plate inside ``crop_bgr``, in the crop's own pixel coordinates.

    Two estimators are tried in order. The glyph-row fit is preferred because it locks
    onto the characters themselves; the bright-field fit is the fallback for crops where
    too few characters segment cleanly.
    """
    height, width = crop_bgr.shape[:2]
    if height < 10 or width < 14:
        return None
    scale = work_height / height
    work = cv2.resize(
        crop_bgr, (max(10, int(width * scale)), work_height), interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)

    glyphs = _glyph_boxes(gray)
    if glyphs:
        quad = _quad_from_glyphs(glyphs, gray.shape)
        if _valid(quad, *gray.shape, min_area_fraction=0.04):
            return QuadEstimate(quad=quad / scale, method="glyph_row", skew=_skew(quad))

    for order, mask in enumerate(_plate_masks(gray)):
        quad = _quad_from_mask(mask)
        if _valid(quad, *gray.shape):
            return QuadEstimate(
                quad=quad / scale,
                method="bright_field_otsu" if order == 0 else "bright_field_adaptive",
                skew=_skew(quad),
            )
    return None


def rectify(
    crop_bgr: np.ndarray, quad: Quad, out_height: int = 112, margin: float = 0.02
) -> np.ndarray | None:
    """Warp the quadrilateral to a fronto-parallel rectangle of its own aspect ratio."""
    points = np.asarray(quad, np.float32)
    centre = points.mean(axis=0)
    points = centre + (points - centre) * (1.0 + margin)
    edge_w = (np.linalg.norm(points[1] - points[0]) + np.linalg.norm(points[2] - points[3])) / 2
    edge_h = (np.linalg.norm(points[3] - points[0]) + np.linalg.norm(points[2] - points[1])) / 2
    if edge_h < 4 or edge_w < 8:
        return None
    aspect = float(np.clip(edge_w / edge_h, 0.7, 8.0))
    out_w = int(round(out_height * aspect))
    destination = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_height - 1], [0, out_height - 1]], np.float32
    )
    matrix = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(
        crop_bgr,
        matrix,
        (out_w, out_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def rotate_only(crop_bgr: np.ndarray) -> np.ndarray | None:
    """Rotation-only baseline, kept so the perspective step can be measured against it."""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    for mask in _plate_masks(gray):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        (_, _), (rect_w, rect_h), angle = cv2.minAreaRect(max(contours, key=cv2.contourArea))
        if rect_w < rect_h:
            angle += 90
        angle = (angle + 45) % 90 - 45
        height, width = crop_bgr.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
        return cv2.warpAffine(
            crop_bgr,
            matrix,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
    return None


def enhance_contrast(image_bgr: np.ndarray, clip: float = 2.0) -> np.ndarray:
    """CLAHE on the L channel. Offered as a candidate, never applied unconditionally."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2BGR)


def upscale(image_bgr: np.ndarray, min_height: int = 64) -> np.ndarray:
    """Bicubic upscale so a tiny crop reaches the OCR model's working height.

    This is interpolation, not generation: it invents no strokes that were not sampled.
    """
    height = image_bgr.shape[0]
    if height >= min_height:
        return image_bgr
    scale = min_height / height
    return cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def quad_to_normalised(
    quad: Quad, offset_x: int, offset_y: int, width: int, height: int
) -> list[list[float]]:
    """Move a crop-local quad into normalised whole-image coordinates."""
    out = []
    for x, y in np.asarray(quad, np.float32):
        out.append(
            [
                round(float(min(max((x + offset_x) / max(1, width), 0.0), 1.0)), 6),
                round(float(min(max((y + offset_y) / max(1, height), 0.0), 1.0)), 6),
            ]
        )
    return out
