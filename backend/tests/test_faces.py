from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.config import Settings, get_settings
from app.faces import FaceDetector, FaceDetectorUnavailable
from app import main as main_mod
from tests.conftest import make_jpeg

YUNET = Path(__file__).resolve().parents[2] / "models" / "face_detection_yunet_2023mar.onnx"


def test_missing_model_raises():
    with pytest.raises(FaceDetectorUnavailable):
        FaceDetector(Path("/nonexistent/yunet.onnx"))


@pytest.mark.skipif(not YUNET.exists(), reason="run scripts/download-models.sh")
def test_flat_image_has_no_face():
    det = FaceDetector(YUNET)
    img = Image.fromarray(np.full((480, 320, 3), 127, dtype=np.uint8))
    assert det.has_face(img) is False


def test_kiosk_token_enforced(tmp_path, monkeypatch):
    get_settings.cache_clear()
    settings = Settings(data_dir=tmp_path / "d", face_check="off", kiosk_token="letmein")
    monkeypatch.setattr(main_mod, "get_settings", lambda: settings)
    settings.overlay_dir.mkdir(parents=True, exist_ok=True)

    from app.store import Store
    from tests.conftest import FakeCompute

    main_mod.state.clear()
    main_mod.state.update(
        settings=settings, store=Store(settings.db_path), compute=FakeCompute(),
        detector=None, labels={"version": 1, "labels": []}, label_names={"gothic"},
    )
    from fastapi.testclient import TestClient

    client = TestClient(main_mod.app)
    files = {"frame": ("f.jpg", make_jpeg(), "image/jpeg")}

    assert client.post("/api/v1/analyze", files=files).status_code == 401
    assert client.post(
        "/api/v1/analyze", files=files, headers={"Authorization": "Bearer letmein"}
    ).status_code == 200

    main_mod.state["store"].close()
    main_mod.state.clear()
    get_settings.cache_clear()
