from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FP_BACKEND_", env_file=".env", extra="ignore")

    # --- compute tier ---
    # Accept the shared names too, so a single .env works for direct runs.
    compute_url: str = Field(
        "http://127.0.0.1:8801",
        validation_alias=AliasChoices("FP_BACKEND_COMPUTE_URL", "COMPUTE_URL"),
    )
    compute_secret: str = Field(
        "",
        validation_alias=AliasChoices("FP_BACKEND_COMPUTE_SECRET", "COMPUTE_SECRET"),
    )
    compute_timeout_s: float = 30.0

    # --- auth ---
    # When set, /analyze and /feedback require `Authorization: Bearer <token>`.
    kiosk_token: str = Field(
        "", validation_alias=AliasChoices("FP_BACKEND_KIOSK_TOKEN", "KIOSK_TOKEN")
    )

    # --- storage ---
    data_dir: Path = Path("./data")
    retention_days: int = Field(
        14, validation_alias=AliasChoices("FP_BACKEND_RETENTION_DAYS", "RETENTION_DAYS")
    )

    # --- uploads ---
    max_upload_bytes: int = 4 * 1024 * 1024
    min_image_edge: int = 64
    max_image_edge: int = 8000

    # --- face verify-and-discard ---
    # required    : reject the request if the detector cannot load (fail closed)
    # best_effort : log and continue if the detector cannot load
    # off         : skip the check entirely (dev only)
    face_check: Literal["required", "best_effort", "off"] = "required"
    yunet_path: Path = Path("/models/face_detection_yunet_2023mar.onnx")
    face_score_threshold: float = 0.6

    # --- CORS (only needed when the frontend is served from another origin) ---
    cors_origins: list[str] = []

    @property
    def db_path(self) -> Path:
        return self.data_dir / "fashion_police.sqlite"

    @property
    def overlay_dir(self) -> Path:
        return self.data_dir / "overlays"


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()
