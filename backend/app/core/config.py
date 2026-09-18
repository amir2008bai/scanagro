from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Fable Vehicle Recognition Backend"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    api_v1_prefix: str = "/api/v1"
    api_key: SecretStr = SecretStr("")
    docs_enabled: bool = True
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    allowed_hosts: str = "localhost,127.0.0.1,testserver"

    database_url: str = "postgresql+asyncpg://app:app@localhost:5432/fable"
    storage_dir: Path = Path("./data/storage")
    max_upload_mb: int = Field(25, ge=1, le=100)
    max_request_mb: int = Field(100, ge=1, le=500)
    max_upload_files: int = Field(25, ge=1, le=100)
    max_image_pixels: int = Field(40_000_000, ge=1, le=100_000_000)
    min_free_disk_mb: int = Field(256, ge=0)
    allowed_image_types: str = "image/jpeg,image/png,image/webp"

    vision_provider: Literal["mock", "openai_compatible", "local"] = "mock"
    vision_base_url: str = "http://localhost:8001/v1"
    vision_api_key: SecretStr = SecretStr("")
    vision_model: str = Field("", max_length=200)
    vision_timeout_seconds: float = Field(90, gt=0, le=300)
    vision_max_retries: int = Field(2, ge=0, le=5)
    vision_use_response_format: bool = True
    vision_max_tokens: int = Field(4096, ge=128, le=32768)
    vision_max_response_bytes: int = Field(2_000_000, ge=1024, le=10_000_000)
    vision_max_image_side: int = Field(2560, ge=512, le=8192)
    vision_store_raw_response: bool = False

    # --- local recognition pipeline (VISION_PROVIDER=local) ---------------------
    #: Where ONNX weights live. Mounted as a volume in Docker so they survive rebuilds.
    vision_model_dir: Path = Path("./data/models")
    #: Identifier recorded on each job, so results stay traceable to a pipeline version.
    vision_local_model_id: str = Field("local-alpr-v1", max_length=200)
    vision_device: Literal["cpu", "cuda", "auto"] = "cpu"
    #: 0 lets ONNX Runtime pick; set it explicitly when pinning CPU budget per worker.
    vision_onnx_threads: int = Field(0, ge=0, le=64)
    #: Images allowed inside the pipeline at once. This, not the session count, caps RSS.
    vision_max_parallel_images: int = Field(1, ge=1, le=8)
    vision_vehicle_input_size: int = Field(960, ge=320, le=1600)
    vision_vehicle_threshold: float = Field(0.35, ge=0.05, le=0.95)
    vision_plate_model: str = Field("yolo-v9-s-608-license-plate-end2end", max_length=120)
    vision_plate_threshold: float = Field(0.25, ge=0.05, le=0.95)
    vision_plate_tile: int = Field(608, ge=256, le=1280)
    vision_plate_tile_overlap: float = Field(0.35, ge=0.0, le=0.7)
    vision_plate_frame_sweep: bool = True
    vision_ocr_model: str = Field("cct-s-v2-global-model", max_length=120)
    #: PP-OCR is slower but markedly more accurate here; it also reads body lettering.
    vision_use_ppocr: bool = True
    vision_alpr_enabled: bool = False
    vision_alpr_ocr_dir: Path | None = None
    vision_alpr_sr_path: Path | None = None
    vision_ocr_accept_confidence: float = Field(0.55, ge=0.0, le=1.0)
    vision_ocr_accept_confidence_unformatted: float = Field(0.88, ge=0.0, le=1.0)
    vision_read_attributes: bool = True
    vision_cyrillic_attributes: bool = True
    # --- reference gallery: machine type by similarity to reference photos -------
    #: COCO has no class for agricultural machinery, so type comes from retrieval
    #: against photos you supply. Empty or missing gallery simply disables the stage.
    vision_reference_gallery: bool = True
    vision_reference_dir: Path = Path("./data/reference")
    #: Cosine similarity floor. Below it the gallery says nothing and the detector's
    #: coarse label stands. Raise it if you see confident wrong types.
    vision_reference_min_similarity: float = Field(0.62, ge=0.0, le=1.0)
    #: Required gap between the best class and the runner-up. This is what makes rear
    #: views of a light truck and a trailer come back as "unknown" instead of a guess.
    vision_reference_min_margin: float = Field(0.04, ge=0.0, le=0.5)
    vision_reference_top_k: int = Field(5, ge=1, le=50)
    #: These cameras burn a timestamp into the top of the frame and the plate detector
    #: fires on it; ignore that band. Set to 0 for footage without an overlay.
    vision_overlay_band_top: float = Field(0.045, ge=0.0, le=0.2)
    vision_save_plate_crops: bool = True

    # Both modes consume the same durable PostgreSQL queue.
    processing_mode: Literal["inprocess", "worker"] = "inprocess"
    worker_concurrency: int = Field(2, ge=1, le=16)
    worker_poll_seconds: float = Field(1, gt=0, le=60)
    job_timeout_seconds: float = Field(300, ge=1, le=1800)
    job_lease_seconds: int = Field(360, ge=2, le=3600)
    job_max_attempts: int = Field(3, ge=1, le=10)
    export_max_images: int = Field(1000, ge=1, le=10000)

    @property
    def allowed_content_types(self) -> set[str]:
        return {v.strip().lower() for v in self.allowed_image_types.split(",") if v.strip()}

    @property
    def cors_origin_list(self) -> list[str]:
        return [v.strip() for v in self.cors_origins.split(",") if v.strip()]

    @property
    def allowed_host_list(self) -> list[str]:
        return [v.strip() for v in self.allowed_hosts.split(",") if v.strip()]

    @model_validator(mode="after")
    def check_configuration(self):
        if not self.database_url.startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_URL must use postgresql+asyncpg://")
        if not self.api_v1_prefix.startswith("/") or self.api_v1_prefix.endswith("/"):
            raise ValueError("API_V1_PREFIX must start with / and have no trailing /")
        if not self.allowed_content_types or not self.allowed_content_types <= {
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            raise ValueError("Only JPEG, PNG and WEBP can be enabled")
        if self.job_lease_seconds <= self.job_timeout_seconds + 10:
            raise ValueError("JOB_LEASE_SECONDS must exceed JOB_TIMEOUT_SECONDS by over 10 seconds")
        if self.max_request_mb < self.max_upload_mb:
            raise ValueError("MAX_REQUEST_MB must be >= MAX_UPLOAD_MB")
        endpoint = urlsplit(self.vision_base_url)
        if endpoint.scheme not in {"http", "https"} or not endpoint.hostname or endpoint.username:
            raise ValueError("VISION_BASE_URL must be an HTTP(S) URL without embedded credentials")
        if endpoint.query or endpoint.fragment:
            raise ValueError("VISION_BASE_URL must not contain a query or fragment")
        if self.vision_provider == "openai_compatible" and not self.vision_model.strip():
            raise ValueError("Set VISION_MODEL to the actual model ID exposed by your server")
        if self.vision_provider == "local" and self.vision_plate_tile % 32:
            raise ValueError("VISION_PLATE_TILE must be a multiple of 32")
        if self.app_env == "production":
            if len(self.api_key.get_secret_value()) < 32:
                raise ValueError("Production requires API_KEY with at least 32 characters")
            if "*" in self.cors_origin_list or "*" in self.allowed_host_list:
                raise ValueError("Production requires explicit CORS_ORIGINS and ALLOWED_HOSTS")
            if self.vision_provider == "mock":
                raise ValueError("Production requires a real vision provider")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
