import pytest
from pydantic import ValidationError

from app.core.config import Settings


@pytest.mark.parametrize(
    "overrides",
    [
        {"vision_provider": "openai_compatible", "vision_model": ""},
        {"vision_base_url": "ftp://host"},
        {"vision_base_url": "http://user:pass@host"},
        {"vision_base_url": "http://host?key=secret"},
        {"database_url": "sqlite:///test.db"},
        {"job_lease_seconds": 10},
        {"max_upload_mb": 0},
        {"processing_mode": "unknown"},
        {"app_env": "production", "vision_provider": "mock"},
        {
            "app_env": "production",
            "vision_provider": "openai_compatible",
            "vision_model": "vision",
            "api_key": "short",
        },
        {
            "app_env": "production",
            "vision_provider": "openai_compatible",
            "vision_model": "vision",
            "cors_origins": "*",
        },
    ],
)
def test_configuration_rejects_unsafe_or_inconsistent_values(overrides):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, **overrides)


def test_production_configuration_and_secret_masking():
    settings = Settings(
        _env_file=None,
        app_env="production",
        vision_provider="openai_compatible",
        vision_model="served-model",
        api_key="s" * 32,
        vision_api_key="private-key",
    )
    assert "private-key" not in repr(settings) and "s" * 32 not in repr(settings)
