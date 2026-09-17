import secrets

from fastapi import Security
from fastapi.security import APIKeyHeader
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from app.core.config import get_settings

key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def api_key_schema(key: str | None = Security(key_header)):
    """OpenAPI security declaration; enforcement happens before multipart parsing."""


class APIKeyMiddleware:
    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send):
        settings = get_settings()
        if scope["type"] == "http":
            path = scope["path"]
            public = {
                "/",
                "/docs",
                "/redoc",
                "/openapi.json",
                "/docs/oauth2-redirect",
                settings.api_v1_prefix + "/health",
                settings.api_v1_prefix + "/health/live",
            }
            expected = settings.api_key.get_secret_value()
            # The review UI is static HTML/CSS/JS with no secrets in it, and it has to
            # load before the operator can type a key. The API it talks to stays guarded:
            # every /api/v1 call from the page carries the header like any other client.
            is_ui = path == "/ui" or path.startswith("/ui/")
            if expected and path not in public and not is_ui and scope["method"] != "OPTIONS":
                headers = dict(scope["headers"])
                supplied = headers.get(b"x-api-key", b"")
                if not secrets.compare_digest(supplied, expected.encode("utf-8")):
                    await JSONResponse({"detail": "Invalid or missing API key"}, status_code=401)(
                        scope, receive, send
                    )
                    return
        await self.app(scope, receive, send)
