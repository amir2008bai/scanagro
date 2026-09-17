import time
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

import structlog
from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.body_limit import RequestBodyLimitMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.security import APIKeyMiddleware, api_key_schema
from app.db.session import engine
from app.services.processing import running_workers
from app.services.storage import LocalStorage

settings = get_settings()
configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    LocalStorage().ensure_directories()
    try:
        if settings.processing_mode == "inprocess":
            async with running_workers():
                yield
        else:
            yield
    finally:
        await engine.dispose()


app = FastAPI(
    title=settings.app_name,
    version="1.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url="/openapi.json" if settings.docs_enabled else None,
)
app.add_middleware(RequestBodyLimitMiddleware, max_body_size=settings.max_request_mb * 1024 * 1024)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.allowed_host_list)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key"],
    expose_headers=["X-Request-ID"],
)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    # Do not echo arbitrary request input or non-JSON exception objects back to clients.
    return JSONResponse(
        status_code=422,
        content={
            "detail": [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        },
    )


@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = uuid4().hex
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        await logger.aerror(
            "unhandled_request_error",
            request_id=request_id,
            path=request.url.path,
            error_type=type(exc).__name__,
        )
        response = JSONResponse(
            status_code=500, content={"detail": "Internal server error", "request_id": request_id}
        )
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    await logger.ainfo(
        "request",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return response


app.include_router(
    api_router, prefix=settings.api_v1_prefix, dependencies=[Depends(api_key_schema)]
)


# The review UI is optional: the service runs headless without it, and mounting only
# when the directory exists keeps an API-only deployment from failing on a missing path.
_ui_dir = Path(__file__).resolve().parents[1] / "frontend"
if _ui_dir.is_dir():
    app.mount("/ui", StaticFiles(directory=_ui_dir, html=True), name="ui")


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {
        "service": settings.app_name,
        "docs": "/docs" if settings.docs_enabled else None,
        "ui": "/ui/" if _ui_dir.is_dir() else None,
    }
