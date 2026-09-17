import os
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont

from app.db.models import Detection

VEHICLE_COLOUR = "#e23b3b"
PLATE_COLOUR = "#20c45a"
PLATE_UNREADABLE_COLOUR = "#f5a623"
QUAD_COLOUR = "#2f7fe0"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


def _label(draw, text, x, y, font, fill, width, height):
    """Draw a filled caption, clamped inside the frame."""
    while text and draw.textlength(text, font=font) > width - 8:
        text = text[:-1]
    if not text:
        return
    box = draw.textbbox((0, 0), text, font=font)
    text_w, text_h = box[2] - box[0], box[3] - box[1]
    left = max(0, min(x, width - text_w - 8))
    top = max(0, min(y - text_h - 8, height - text_h - 8))
    draw.rectangle((left, top, left + text_w + 8, top + text_h + 8), fill=fill)
    draw.text((left + 4 - box[0], top + 4 - box[1]), text, font=font, fill="white")


def annotate_image(source: Path, target: Path, detections: Iterable[Detection]) -> None:
    """Render vehicle boxes, plate boxes and plate corners onto a copy of the image.

    The vehicle box and the plate box are drawn in different colours and captioned
    separately, because they are different objects with different confidences. A plate
    that was detected but not read is drawn in amber rather than being hidden: the
    operator can see that the pipeline found something it could not confirm.
    """
    with Image.open(source) as original, original.convert("RGB") as image:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        line_width = max(2, round(min(width, height) / 300))
        font = _font(max(12, round(min(width, height) / 40)))
        small = _font(max(10, round(min(width, height) / 55)))

        for det in detections:
            x1, y1 = int(det.bbox_x1 * (width - 1)), int(det.bbox_y1 * (height - 1))
            x2, y2 = int(det.bbox_x2 * (width - 1)), int(det.bbox_y2 * (height - 1))
            draw.rectangle((x1, y1, x2, y2), outline=VEHICLE_COLOUR, width=line_width)

            # A type from the reference gallery is marked with '~': on this footage the
            # detector alone calls every tractor a truck, so a reviewer needs to see at a
            # glance which labels were actually matched against reference photos.
            type_text = det.vehicle_type
            if getattr(det, "vehicle_type_source", None) == "reference_gallery":
                type_text = f"~{type_text}"
            parts = [type_text, det.manufacturer, det.model, det.license_plate]
            label = " | ".join(part for part in parts if part)
            confidence = (
                det.vehicle_confidence if det.vehicle_confidence is not None else det.confidence
            )
            if confidence is not None:
                label = f"{label}  {confidence:.2f}"
            _label(draw, label, x1, y1, font, VEHICLE_COLOUR, width, height)

            if getattr(det, "plate_bbox_x1", None) is None:
                continue
            px1 = int(det.plate_bbox_x1 * (width - 1))
            py1 = int(det.plate_bbox_y1 * (height - 1))
            px2 = int(det.plate_bbox_x2 * (width - 1))
            py2 = int(det.plate_bbox_y2 * (height - 1))
            colour = PLATE_COLOUR if det.plate_readable else PLATE_UNREADABLE_COLOUR
            draw.rectangle((px1, py1, px2, py2), outline=colour, width=line_width)

            quad = getattr(det, "plate_quad", None)
            if quad and len(quad) == 4:
                points = [(int(px * (width - 1)), int(py * (height - 1))) for px, py in quad]
                draw.line(points + [points[0]], fill=QUAD_COLOUR, width=max(1, line_width - 1))

            if det.plate_readable and det.license_plate:
                caption = det.license_plate
                if det.plate_ocr_confidence is not None:
                    caption = f"{caption}  ocr {det.plate_ocr_confidence:.2f}"
            else:
                raw = getattr(det, "plate_text_raw", None)
                caption = f"unreadable ({raw})" if raw else "unreadable"
            _label(
                draw,
                caption,
                px1,
                py2 + int(2.4 * line_width) + small.size,
                small,
                colour,
                width,
                height,
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(target.name + "." + uuid4().hex + ".tmp")
        try:
            with temp.open("xb") as stream:
                image.save(stream, format="JPEG", quality=92)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, target)
        finally:
            temp.unlink(missing_ok=True)
