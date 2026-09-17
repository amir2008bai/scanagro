"""The plate grammar decides what may be asserted, so its edges are worth pinning."""

import pytest

from app.services.vision.local import plate_format


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("160ALV10", "kz_civil_2012"),
        ("592 LBA 10", "kz_civil_2012"),
        ("041ahf10", "kz_civil_2012"),
        ("A643KCG", "kz_private_pre2012"),
        ("A644DC", "kz_other_pre2012"),
    ],
)
def test_recognises_kazakh_layouts(text, expected):
    fmt = plate_format.classify(text)
    assert fmt is not None and fmt.name == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "WED1355507",  # burnt-in CCTV timestamp
        "KAMAZ",  # grille badge
        "CAMERA01",
        "597AQ10",  # two letters where the 2012 layout needs three
        "160ALV99",  # 99 is not an issued region code
    ],
)
def test_rejects_strings_that_are_not_plates(text):
    assert plate_format.classify(text) is None


def test_region_code_must_be_real():
    assert plate_format.classify("123ABC10") is not None
    assert plate_format.classify("123ABC77") is None
    assert plate_format.region_name("123ABC10") == "Kostanay"


def test_display_spacing_does_not_alter_characters():
    assert plate_format.format_for_display("160ALV10") == "160 ALV 10"
    assert plate_format.normalise("160 alv-10") == "160ALV10"


def test_never_repairs_a_broken_read():
    """A read missing a character stays broken; the grammar must not pad it."""
    assert plate_format.classify("60ALV10") is None
    assert plate_format.format_for_display("60ALV10") == "60ALV10"
    assert plate_format.normalise("60ALV10") == "60ALV10"


def test_shape_gates_two_row_layouts():
    """A wide strip cannot be a square machinery plate, however it reads."""
    # "TOO" + "2026" off a yard banner matches the character pattern...
    assert plate_format.classify("TOO2026") is not None
    # ...but not once the recovered plate shape is known to be a wide strip.
    assert plate_format.classify("TOO2026", box_aspect=4.6) is None
    assert plate_format.classify("AAH1098", box_aspect=1.3) is not None


def test_shape_gates_single_row_layouts():
    assert plate_format.classify("160ALV10", box_aspect=4.6) is not None
    assert plate_format.classify("160ALV10", box_aspect=1.1) is None


def test_plausibility_prefers_structured_reads():
    assert plate_format.plausibility("160ALV10") == 1.0
    assert plate_format.plausibility("WED1355507") < 0.6
    assert plate_format.plausibility("") == 0.0
    # A near-miss still outranks noise, so ranking can prefer it without accepting it.
    assert plate_format.plausibility("160ALV1") > plate_format.plausibility("KAMAZ")


def test_disagreements_locate_differing_characters():
    assert plate_format.disagreements("384CBA10", "384C8A10") == [4]
    assert plate_format.disagreements("384CBA10", "384CBA10") == []
    # Different lengths mean the engines did not agree on the character count at all.
    assert plate_format.disagreements("384CBA10", "384CBA1") == []
