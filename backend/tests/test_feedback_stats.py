from __future__ import annotations

from tests.conftest import make_jpeg


def _analyze(client) -> str:
    r = client.post("/api/v1/analyze", files={"frame": ("f.jpg", make_jpeg(), "image/jpeg")})
    assert r.status_code == 200
    return r.json()["record_id"]


def test_feedback_updates_stats(ctx):
    rec = _analyze(ctx.client)

    r = ctx.client.post(f"/api/v1/feedback/{rec}", json={"correct_label": "preppy"})
    assert r.status_code == 200 and r.json() == {"ok": True}

    stats = ctx.client.get("/api/v1/stats").json()
    assert stats["total_predictions"] == 1
    assert stats["total_feedback"] == 1
    assert stats["feedback_rate"] == 1.0
    assert stats["user_corrections"][0] == ["preppy", 1]


def test_feedback_unknown_record_404(ctx):
    r = ctx.client.post("/api/v1/feedback/nope", json={"correct_label": "preppy"})
    assert r.status_code == 404


def test_feedback_unknown_label_400(ctx):
    rec = _analyze(ctx.client)
    r = ctx.client.post(f"/api/v1/feedback/{rec}", json={"correct_label": "not_a_style"})
    assert r.status_code == 400


def test_stats_empty(ctx):
    stats = ctx.client.get("/api/v1/stats").json()
    assert stats == {
        "total_predictions": 0,
        "total_feedback": 0,
        "feedback_rate": 0.0,
        "top_predictions": [],
        "user_corrections": [],
    }


def test_labels_proxied_from_compute(ctx):
    body = ctx.client.get("/api/v1/labels").json()
    assert [l["name"] for l in body["labels"]] == ["gothic", "preppy"]


def test_health_reports_compute_and_face_check(ctx):
    body = ctx.client.get("/api/v1/health").json()
    assert body["backend"] == "ok"
    assert body["compute"] == "ok"
    assert body["face_check"] == "loaded"
