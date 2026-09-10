from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main as main_mod
from app.config import Settings, get_settings
from app.store import Store


def make_jpeg(w: int = 400, h: int = 800, colour=(120, 120, 140)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), colour).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def jpeg_bytes() -> bytes:
    return make_jpeg()


class FakeCompute:
    """Stands in for ComputeClient."""

    def __init__(self) -> None:
        self.infer_calls = 0
        self.infer_result = {
            "ranked": [
                {"name": "gothic", "score": 0.7, "display_name": "Gothic"},
                {"name": "preppy", "score": 0.3, "display_name": "Preppy"},
            ],
            "top": {"name": "gothic", "score": 0.7, "display_name": "Gothic"},
            "uncertain": False,
            "anonymized_overlay_b64": "/9j/",  # not a real jpeg; store may skip it
            "display_overlay_b64": "ZGlzcGxheQ==",
            "timings_ms": {"total": 40.0},
        }
        self.raise_on_infer: Exception | None = None

    async def healthz(self) -> dict:
        return {"status": "ok", "ready": True}

    async def labels(self) -> dict:
        return {
            "version": 1,
            "labels": [
                {"name": "gothic", "display_name": "Gothic", "blurb": "", "accent": "#000"},
                {"name": "preppy", "display_name": "Preppy", "blurb": "", "accent": "#00f"},
            ],
        }

    async def infer(self, image_bytes: bytes, content_type: str) -> dict:
        self.infer_calls += 1
        if self.raise_on_infer is not None:
            raise self.raise_on_infer
        return self.infer_result

    async def aclose(self) -> None:
        pass


class FakeDetector:
    def __init__(self, result: bool = False) -> None:
        self.result = result
        self.calls = 0

    def has_face(self, image) -> bool:
        self.calls += 1
        return self.result


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    """A configured backend with fakes wired into `state`, plus a TestClient.

    Not using TestClient as a context manager -> the real lifespan is skipped.
    """
    get_settings.cache_clear()
    settings = Settings(
        compute_url="http://compute.invalid",
        compute_secret="s3cret",
        data_dir=tmp_path / "data",
        face_check="required",
        kiosk_token="",
    )
    monkeypatch.setattr(main_mod, "get_settings", lambda: settings)
    settings.overlay_dir.mkdir(parents=True, exist_ok=True)

    store = Store(settings.db_path)
    compute = FakeCompute()
    detector = FakeDetector(result=False)

    main_mod.state.clear()
    main_mod.state.update(
        settings=settings,
        store=store,
        compute=compute,
        detector=detector,
        labels={"version": 1, "labels": []},
        label_names={"gothic", "preppy"},
    )

    client = TestClient(main_mod.app)
    yield type("Ctx", (), {"client": client, "settings": settings, "store": store,
                           "compute": compute, "detector": detector})
    store.close()
    main_mod.state.clear()
    get_settings.cache_clear()
