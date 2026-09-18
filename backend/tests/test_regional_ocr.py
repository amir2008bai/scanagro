from unittest.mock import Mock

import numpy as np
import pytest

from app.services.alpr_processor import PaddleRecognizer, RestrictedCTC
from app.services.vision.local.alpr_reader import ALPRPlateReader


def test_ctc_mask_keeps_blank_indices_and_original_probability():
    class CTCLabelDecode:
        character = ["blank", "O", "0", "5", "S"]

        def get_ignored_tokens(self):
            return [0]

        def __call__(self, pred, **kwargs):
            return pred[0]

    pred = np.array([[[.02, .70, .20, .03, .05]]])
    result = RestrictedCTC(CTCLabelDecode(), "0123456789")([pred])
    assert result.argmax(axis=-1).item() == 2
    assert result[0, 0, 2] == .20  # No renormalization or fake high confidence.
    assert result[0, 0, 0] == .02
    assert pred[0, 0, 1] == .70  # Input model output remains untouched.


def regional_reader(region_outputs):
    reader = ALPRPlateReader.__new__(ALPRPlateReader)
    recognizer = Mock(spec=PaddleRecognizer)
    recognizer.recognize.return_value = ("X123BC", .95)
    recognizer.recognize_allowed.side_effect = region_outputs
    reader.processor = Mock(_recognizer=recognizer)
    reader.fallback = Mock(accept_confidence=.55)
    return reader


def test_uncertain_region_is_review_candidate_not_confirmed_plate():
    reader = regional_reader([("77", .4), ("77", .3), ("77", .5), ("77", .1)])
    result = reader._read_region(np.zeros((110, 470, 3), np.uint8), None)
    assert result.raw_text == "X123BC77"
    assert result.text is None and not result.readable
    assert result.reason == "region_needs_review"
    assert result.ocr_confidence == pytest.approx(.4)


def test_disagreeing_regions_abstain():
    reader = regional_reader([("77", .8), ("78", .8), ("77", .8), ("78", .8)])
    assert reader._read_region(np.zeros((110, 470, 3), np.uint8), None) is None


def test_high_confidence_region_is_accepted_without_specific_plate_hardcoding():
    reader = regional_reader([("123", .9)] * 4)
    result = reader._read_region(np.zeros((110, 470, 3), np.uint8), None)
    assert result.text == "X 123 BC 123" and result.readable
    assert result.ocr_confidence == .9
