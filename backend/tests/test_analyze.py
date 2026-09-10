from __future__ import annotations

import io

from PIL import Image

from app.compute import ComputeError
from tests.conftest import make_jpeg


def _post(client, data: bytes, content_type: str = "image/jpeg"):
    return client.post(
        "/api/v1/analyze",
        files={"frame": ("frame.jpg", data, content_type)},
    )


def test_happy_path_stores_and_returns(ctx):
    r = _post(ctx.client, make_jpeg())
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["top"]["name"] == "gothic"
    assert body["display_overlay_b64"] == "ZGlzcGxheQ=="
    assert ctx.compute.infer_calls == 1
    assert ctx.detector.calls == 1

    stats = ctx.client.get("/api/v1/stats").json()
    assert stats["total_predictions"] == 1
    assert stats["top_predictions"][0] == ["gothic", 1]


def test_face_visible_is_rejected_and_nothing_stored(ctx):
    ctx.detector.result = True
    r = _post(ctx.client, make_jpeg())
    assert r.status_code == 422
    assert r.json()["error"] == "face_visible"
    assert ctx.compute.infer_calls == 0  # never forwarded
    assert ctx.client.get("/api/v1/stats").json()["total_predictions"] == 0


def test_rejects_wrong_content_type(ctx):
    r = _post(ctx.client, make_jpeg(), content_type="image/gif")
    assert r.status_code == 400


def test_rejects_oversized(ctx):
    ctx.settings.max_upload_bytes = 1000
    r = _post(ctx.client, make_jpeg(1200, 1200))
    assert r.status_code == 413


def test_rejects_garbage_bytes(ctx):
    r = _post(ctx.client, b"not an image at all")
    assert r.status_code == 400


def test_rejects_tiny_image(ctx):
    buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buf, format="JPEG")
    r = _post(ctx.client, buf.getvalue())
    assert r.status_code == 400


def test_compute_timeout_maps_to_504(ctx):
    ctx.compute.raise_on_infer = ComputeError("compute timed out", status=504)
    r = _post(ctx.client, make_jpeg())
    assert r.status_code == 504


def test_compute_down_maps_to_502(ctx):
    ctx.compute.raise_on_infer = ComputeError("compute unreachable", status=502)
    r = _post(ctx.client, make_jpeg())
    assert r.status_code == 502


def test_overlay_file_written_when_valid_b64(ctx):
    # a real 1x1 jpeg, base64'd
    import base64

    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, format="JPEG")
    ctx.compute.infer_result["anonymized_overlay_b64"] = base64.b64encode(buf.getvalue()).decode()

    r = _post(ctx.client, make_jpeg())
    assert r.status_code == 200
    written = list(ctx.settings.overlay_dir.glob("*.jpg"))
    assert len(written) == 1
    assert written[0].stat().st_size > 0
