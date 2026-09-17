from pathlib import Path
from typing import Any

from app.schemas.vision import VisionImageResult
from app.services.vision.base import VisionProvider


class MockVisionProvider(VisionProvider):
    name = "mock"
    model = "mock-empty"

    async def analyze(
        self, image_path: Path, mime_type: str
    ) -> tuple[VisionImageResult, dict[str, Any]]:
        result = VisionImageResult(detections=[])
        return result, {"provider": "mock", "warning": "No real recognition was executed."}
