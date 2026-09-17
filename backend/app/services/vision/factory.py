from app.core.config import get_settings
from app.services.vision.base import VisionProvider
from app.services.vision.mock import MockVisionProvider
from app.services.vision.openai_compatible import OpenAICompatibleVisionProvider


def get_vision_provider(provider: str | None = None, model: str | None = None) -> VisionProvider:
    provider = provider or get_settings().vision_provider
    if provider == "mock":
        return MockVisionProvider()
    if provider == "openai_compatible":
        return OpenAICompatibleVisionProvider(model=model)
    if provider == "local":
        # Imported lazily: the heavy CV stack must not be a hard import for mock runs.
        from app.services.vision.local_provider import LocalVisionProvider

        return LocalVisionProvider(model=model)
    raise RuntimeError(f"Unknown VISION_PROVIDER: {provider}")
