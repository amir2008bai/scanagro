import asyncio
import signal

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.session import engine
from app.services.processing import running_workers
from app.services.storage import LocalStorage


async def warm_recognition_models():
    """Build the local pipeline before taking work.

    Loading ONNX weights takes seconds. Doing it here means the first job is not the one
    that pays for it, and that a missing or corrupt weight file fails at startup with a
    clear message rather than as a mysterious job failure minutes later.
    """
    settings = get_settings()
    if settings.vision_provider != "local":
        return
    import structlog

    from app.services.vision.local_provider import get_pipeline

    log = structlog.get_logger(__name__)
    try:
        loaded = await asyncio.to_thread(lambda: get_pipeline(settings).warm_up())
        await log.ainfo("recognition_models_ready", models=loaded)
    except Exception as exc:
        await log.aerror("recognition_models_unavailable", error=str(exc))
        raise


async def main():
    configure_logging(get_settings().log_level)
    LocalStorage().ensure_directories()
    await warm_recognition_models()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop.set)
        except NotImplementedError:  # Windows
            signal.signal(signum, lambda *_: loop.call_soon_threadsafe(stop.set))
    try:
        async with running_workers():
            await stop.wait()
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
