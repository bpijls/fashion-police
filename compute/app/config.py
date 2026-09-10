from __future__ import annotations

import functools
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FP_COMPUTE_", env_file=".env", extra="ignore")

    # Path to the shared taxonomy file (mounted read-only in the container).
    styles_path: Path = Path("/config/styles.yaml")

    # "cuda", "cpu", or "auto" (cuda when available, else cpu).
    device: str = "auto"

    # Hugging Face model ids. Overridable for pinning / mirrors.
    segmentation_model: str = "mattmdjaga/segformer_b2_clothes"
    style_model: str = "patrickjohncyh/fashion-clip"

    # Which scorer implementation to use for style ranking.
    scorer: Literal["fashionclip", "sklearn"] = "fashionclip"

    # Shared secret required on the X-Compute-Secret header (empty = disabled, dev only).
    # Env var: FP_COMPUTE_SECRET
    secret: str = ""

    # A prediction is "uncertain" when the top softmax score is below this.
    uncertain_below: float = 0.22

    # Long edge the input image is resized to before inference (keeps latency bounded).
    max_image_edge: int = 1024


@functools.lru_cache
def get_settings() -> Settings:
    return Settings()


class Style(BaseModel):
    name: str
    display_name: str
    prompt: str
    blurb: str = ""
    accent: str = "#888888"


class Taxonomy(BaseModel):
    version: int = 1
    styles: list[Style]

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.styles]

    @property
    def prompts(self) -> list[str]:
        return [s.prompt for s in self.styles]

    def by_name(self, name: str) -> Style | None:
        return next((s for s in self.styles if s.name == name), None)


def load_taxonomy(path: Path) -> Taxonomy:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    styles_block = raw.get("styles") or {}
    if not styles_block:
        raise ValueError(f"No styles defined in {path}")
    styles = [
        Style(
            name=key,
            display_name=val.get("display_name", key.replace("_", " ").title()),
            prompt=val["prompt"],
            blurb=val.get("blurb", ""),
            accent=val.get("accent", "#888888"),
        )
        for key, val in styles_block.items()
    ]
    return Taxonomy(version=int(raw.get("version", 1)), styles=styles)


def resolve_device(setting: str) -> str:
    if setting != "auto":
        return setting
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"
