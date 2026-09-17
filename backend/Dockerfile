FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VISION_MODEL_DIR=/data/models \
    MODEL_CACHE_DIR=/data/models \
    VISION_REFERENCE_DIR=/data/reference
WORKDIR /app
# libgl1/libglib2.0-0 are OpenCV's runtime dependencies; the headless wheel still links
# against glib. fonts-dejavu-core is what the annotation renderer draws labels with.
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
COPY requirements.txt .
# rapidocr-onnxruntime depends on the GUI build of OpenCV, which needs libxcb/libX11 and
# would drag a windowing stack into a server image. Both wheels install the same `cv2`
# package, so we settle it explicitly: remove whichever landed, reinstall only the
# headless build, and prove it imports. The final `pip check` will note that rapidocr
# asked for opencv-python; that is expected and the headless wheel satisfies its use.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip uninstall -y opencv-python opencv-python-headless \
    && pip install --no-cache-dir --no-deps opencv-python-headless==5.0.0.93 \
    && python -c "import cv2, onnxruntime; print('cv2', cv2.__version__, 'ort', onnxruntime.__version__)"
COPY --chown=app:app . .
RUN chmod +x /app/docker-entrypoint.sh \
    && mkdir -p /data/storage /data/models /data/reference \
    && chown -R app:app /data
USER 10001:10001

# Weights are fetched into the /data/models volume at first start, not baked into the
# image: they are large, they are versioned separately from the code, and pinning them
# by checksum in the registry means a rebuild cannot silently change recognition.
# Pre-fetch at build time instead with:
#   RUN python scripts/fetch_models.py --dir /data/models
# The seeded reference gallery ships in the image under /app/reference_seed. The
# fetch-models service copies it onto the empty /data/reference volume on first start, so
# a fresh deployment has a working type classifier; anything already on the volume is
# left alone, because those are the operator's own photos.
EXPOSE 8000
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--limit-concurrency", "32", "--timeout-keep-alive", "5", "--no-proxy-headers"]
