import pytest
from pydantic import ValidationError

from app.schemas.vision import VisionImageResult


def test_vision_result_validates_and_normalizes_plate():
    result = VisionImageResult.model_validate(
        {
            "detections": [
                {
                    "vehicle_type": "dump truck",
                    "manufacturer": "KamAZ",
                    "model": None,
                    "license_plate": " 123 abc 01 ",
                    "plate_readable": True,
                    "confidence": 0.91,
                    "bbox": {"x1": 0.1, "y1": 0.2, "x2": 0.8, "y2": 0.9},
                }
            ]
        }
    )
    assert result.detections[0].license_plate == "123 ABC 01"


def test_bbox_rejects_invalid_order():
    with pytest.raises(ValidationError):
        VisionImageResult.model_validate(
            {
                "detections": [
                    {
                        "vehicle_type": "truck",
                        "bbox": {"x1": 0.8, "y1": 0.1, "x2": 0.2, "y2": 0.9},
                    }
                ]
            }
        )
