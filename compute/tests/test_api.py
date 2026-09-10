from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import Settings, load_taxonomy
from app.pipeline import InferResult

REPO_STYLES = Path(__file__).resolve().parents[2] / "styles.yaml"


class StubClassifier:
    def classify(self, image) -> InferResult:
        return InferResult(
            ranked=[{"name": "gothic", "score": 0.8}, {"name": "preppy", "score": 0.2}],
            top={"name": "gothic", "score": 0.8},
            uncertain=False,
            anonymized_overlay_b64="YQ==",
            display_overlay_b64="Yg==",
            timings_ms={"total": 1.0},
        )


@pytest.fixture
def client(monkeypatch) -> TestClient:
    tax = load_taxonomy(REPO_STYLES)
    monkeypatch.setattr(main, "get_settings", lambda: Settings(styles_path=REPO_STYLES, secret="s3cret"))
    main.state.clear()
    main.state.update({"ready": True, "device": "cpu", "taxonomy": tax, "classifier": StubClassifier()})
    # Not using TestClient as a context manager -> lifespan (real model load) is skipped.
    yield TestClient(main.app)
    main.state.clear()


def test_healthz(client: TestClient) -> None:
    body = client.get("/healthz").json()
    assert body["ready"] is True and body["device"] == "cpu"


def test_labels(client: TestClient) -> None:
    body = client.get("/labels").json()
    names = [l["name"] for l in body["labels"]]
    assert "gothic" in names and "urban_streetwear" in names


def test_infer_requires_secret(client: TestClient, jpeg_bytes: bytes) -> None:
    r = client.post("/infer", files={"frame": ("f.jpg", jpeg_bytes, "image/jpeg")})
    assert r.status_code == 401


def test_infer_happy_path(client: TestClient, jpeg_bytes: bytes) -> None:
    r = client.post(
        "/infer",
        files={"frame": ("f.jpg", jpeg_bytes, "image/jpeg")},
        headers={"X-Compute-Secret": "s3cret"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["top"]["name"] == "gothic"
    assert body["display_overlay_b64"] and body["anonymized_overlay_b64"]


def test_infer_rejects_garbage(client: TestClient) -> None:
    r = client.post(
        "/infer",
        files={"frame": ("f.jpg", b"not an image", "image/jpeg")},
        headers={"X-Compute-Secret": "s3cret"},
    )
    assert r.status_code == 400
