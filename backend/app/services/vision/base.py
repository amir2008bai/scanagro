from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.schemas.vision import VisionImageResult


class VisionProvider(ABC):
    name: str
    model: str

    @abstractmethod
    async def analyze(
        self, image_path: Path, mime_type: str
    ) -> tuple[VisionImageResult, dict[str, Any]]:
        raise NotImplementedError
