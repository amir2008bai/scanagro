"""Kazakhstan plate grammar.

This module only ever *classifies and spaces* a string that OCR already produced. It
never substitutes, pads or repairs characters: turning ``38?CBA10`` into ``384CBA10``
would be inventing evidence, and a plate whose characters are not legible must come back
marked unreadable rather than plausible. The grammar is used for two things only:

* deciding whether a read is structurally possible for this country, which is what lets
  us reject the CCTV timestamp overlay and the ``KAMAZ`` grille badge; and
* formatting an accepted read for display.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: Region codes issued in Kazakhstan. 10 is Kostanay, which is where this dataset is from.
REGION_CODES: dict[str, str] = {
    "01": "Astana",
    "02": "Almaty (city)",
    "03": "Akmola",
    "04": "Aktobe",
    "05": "Almaty region",
    "06": "Atyrau",
    "07": "West Kazakhstan",
    "08": "Zhambyl",
    "09": "Karaganda",
    "10": "Kostanay",
    "11": "Kyzylorda",
    "12": "Mangystau",
    "13": "Turkestan",
    "14": "Pavlodar",
    "15": "North Kazakhstan",
    "16": "East Kazakhstan",
    "17": "Shymkent",
    "18": "Abai",
    "19": "Jetisu",
    "20": "Ulytau",
}

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_STRIP = re.compile(r"[^A-Z0-9]")


@dataclass(frozen=True, slots=True)
class PlateFormat:
    name: str
    pattern: re.Pattern[str]
    #: Index groups used to insert spaces, e.g. (3, 6) -> "123 ABC 10".
    breaks: tuple[int, ...]
    description: str
    region_slice: tuple[int, int] | None = None
    #: Width/height range the *rectified plate* must fall in for this layout to be
    #: possible, measured along the quad's own edges. A 520x110 mm single-row plate
    #: cannot be square, and a square machinery plate cannot be four times as wide as it
    #: is tall — this is what stops a wide strip of yard signage from being accepted as a
    #: two-row plate. It is not applied when geometry could not recover the corners.
    aspect_range: tuple[float, float] = (0.4, 12.0)
    #: Layouts inferred from sample images rather than from a published specification.
    provisional: bool = False


FORMATS: tuple[PlateFormat, ...] = (
    PlateFormat(
        name="kz_civil_2012",
        pattern=re.compile(r"^[0-9]{3}[A-Z]{3}[0-9]{2}$"),
        breaks=(3, 6),
        description="Civil plate since 2012: 123 ABC 10",
        region_slice=(6, 8),
        aspect_range=(2.2, 8.0),
    ),
    PlateFormat(
        name="kz_private_pre2012",
        pattern=re.compile(r"^[A-Z][0-9]{3}[A-Z]{3}$"),
        breaks=(1, 4),
        description="Private plate before 2012: A 643 KCG",
        aspect_range=(2.2, 8.0),
    ),
    PlateFormat(
        name="kz_other_pre2012",
        pattern=re.compile(r"^[A-Z][0-9]{3}[A-Z]{2}$"),
        breaks=(1, 4),
        description="Non-private plate before 2012: A 644 DC",
        aspect_range=(2.2, 8.0),
    ),
    # Square two-row plates used on trailers and agricultural machinery. Which row PP-OCR
    # emits first depends on the crop, so both orders are accepted. The region code is
    # validated in either case, which is what keeps this from matching arbitrary noise.
    # The row semantics are inferred from this dataset and not from an official spec:
    # treat the grouping as provisional, and see docs/RECOGNITION.md.
    PlateFormat(
        name="kz_machinery_square",
        pattern=re.compile(r"^[A-Z]{3}[0-9]{4}$"),
        breaks=(3, 5),
        description="Two-row machinery plate, letter row first: AAH 10 98",
        region_slice=(3, 5),
        aspect_range=(0.6, 2.2),
        provisional=True,
    ),
    PlateFormat(
        name="kz_machinery_square_alt",
        pattern=re.compile(r"^[0-9]{4}[A-Z]{3}$"),
        breaks=(2, 4),
        description="Two-row machinery plate, digit row first: 10 98 AAH",
        region_slice=(0, 2),
        aspect_range=(0.6, 2.2),
        provisional=True,
    ),
)


def normalise(text: str | None) -> str:
    """Upper-case and drop separators. No character is replaced."""
    if not text:
        return ""
    return _STRIP.sub("", text.upper())


def classify(text: str | None, box_aspect: float | None = None) -> PlateFormat | None:
    """Match a read against the grammar.

    ``box_aspect`` is the rectified plate's width/height, measured along the recovered
    quad. When supplied, a layout whose physical shape cannot produce that plate is
    rejected — the geometry is evidence independent of the text, and it is what separates
    a real two-row machinery plate from a wide strip of signage that happens to read as
    three letters and four digits. Pass ``None`` when the corners were not recovered.
    """
    compact = normalise(text)
    if not compact:
        return None
    for fmt in FORMATS:
        if not fmt.pattern.match(compact):
            continue
        if fmt.region_slice is not None:
            start, end = fmt.region_slice
            if compact[start:end] not in REGION_CODES:
                continue
        if box_aspect is not None:
            low, high = fmt.aspect_range
            if not (low <= box_aspect <= high):
                continue
        return fmt
    return None


def format_for_display(text: str | None) -> str | None:
    """Insert the canonical spaces. Returns the compact text when no format matches."""
    compact = normalise(text)
    if not compact:
        return None
    fmt = classify(compact)
    if fmt is None:
        return compact
    parts, previous = [], 0
    for index in fmt.breaks:
        parts.append(compact[previous:index])
        previous = index
    parts.append(compact[previous:])
    return " ".join(part for part in parts if part)


def region_name(text: str | None) -> str | None:
    fmt = classify(text)
    if fmt is None or fmt.region_slice is None:
        return None
    start, end = fmt.region_slice
    return REGION_CODES.get(normalise(text)[start:end])


def plausibility(text: str | None) -> float:
    """A soft structural score in [0, 1], used to rank competing OCR candidates.

    A full format match scores 1.0. Partial credit is given for the length and the
    digit/letter mix a Kazakh plate would have, so that a read which is nearly right
    still outranks a random string scraped off a grille badge. This score only ever
    *ranks* candidates; it never edits one.
    """
    compact = normalise(text)
    if not compact:
        return 0.0
    if classify(compact) is not None:
        return 1.0
    score = 0.0
    if 6 <= len(compact) <= 9:
        score += 0.35
    digits = sum(character.isdigit() for character in compact)
    letters = len(compact) - digits
    if digits >= 3 and letters >= 2:
        score += 0.3
    # The 2012 layout starts with three digits and ends with a known region code.
    if re.match(r"^[0-9]{3}[A-Z]", compact):
        score += 0.2
    if len(compact) >= 2 and compact[-2:] in REGION_CODES:
        score += 0.15
    return min(score, 0.95)


def disagreements(first: str | None, second: str | None) -> list[int]:
    """Positions where two OCR reads of the same plate differ.

    Only meaningful for equal-length reads; different lengths mean the engines did not
    even agree on how many characters are present, which is reported separately.
    """
    a, b = normalise(first), normalise(second)
    if not a or not b or len(a) != len(b):
        return []
    return [index for index, (x, y) in enumerate(zip(a, b, strict=True)) if x != y]
