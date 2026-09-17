import json

import httpx
import pytest

from app.core.config import get_settings
from app.services.vision.openai_compatible import (
    OpenAICompatibleVisionProvider,
    VisionProviderError,
)


async def analyze(tmp_path, image_bytes, handler):
    path = tmp_path / "image.jpg"
    path.write_bytes(image_bytes)
    return await OpenAICompatibleVisionProvider(
        model="test-model", transport=httpx.MockTransport(handler)
    ).analyze(path, "image/jpeg")


async def test_provider_payload_and_fenced_json(tmp_path, image_bytes):
    def handler(request):
        payload = json.loads(request.content)
        assert payload["model"] == "test-model"
        assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith(
            "data:image/jpeg;base64,"
        )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '```json\n{"detections": []}\n```'}}]}
        )

    result, _ = await analyze(tmp_path, image_bytes, handler)
    assert result.detections == []


@pytest.mark.parametrize(
    "response",
    [
        {"choices": []},
        {"choices": [{"message": {"content": "{}"}}]},
        {"choices": [{"message": {"content": "null"}}]},
        {"choices": [{"message": {"content": {}}}]},
        {"choices": [{"message": {"content": '{"detections": []}'}, "finish_reason": "length"}]},
    ],
)
async def test_bad_provider_responses(tmp_path, image_bytes, response):
    with pytest.raises(VisionProviderError):
        await analyze(tmp_path, image_bytes, lambda _: httpx.Response(200, json=response))


async def test_auth_error_is_not_retried_or_exposed(tmp_path, image_bytes):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(401, text="secret-provider-error")

    with pytest.raises(VisionProviderError) as error:
        await analyze(tmp_path, image_bytes, handler)
    assert len(calls) == 1 and "secret" not in str(error.value)


async def test_transient_retry(tmp_path, image_bytes, monkeypatch):
    import app.services.vision.openai_compatible as provider_module

    calls = []
    delays = []

    async def sleep(delay):
        delays.append(delay)

    monkeypatch.setattr(provider_module.asyncio, "sleep", sleep)

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"detections": []}'}}]}
        )

    await analyze(tmp_path, image_bytes, handler)
    assert len(calls) == 2 and delays == [2]


async def test_response_size_limit(tmp_path, image_bytes, monkeypatch):
    monkeypatch.setattr(get_settings(), "vision_max_response_bytes", 100)
    with pytest.raises(VisionProviderError, match="size limit"):
        await analyze(tmp_path, image_bytes, lambda _: httpx.Response(200, content=b"x" * 101))
