from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image, UnidentifiedImageError

from .compute import ComputeClient, ComputeError
from .config import Settings, get_settings
from .faces import FaceDetector, FaceDetectorUnavailable
from .retention import retention_loop
from .schemas import (
    AnalyzeResponse,
    FeedbackRequest,
    HealthResponse,
    LabelsResponse,
    StatsResponse,
)
from .store import Store

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("fp.backend")
log.setLevel(logging.INFO)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}

state: dict = {}


def _load_face_detector(settings: Settings) -> FaceDetector | None:
    if settings.face_check == "off":
        log.warning("face_check=off — the verify-and-discard safety net is DISABLED")
        return None
    try:
        detector = FaceDetector(settings.yunet_path, settings.face_score_threshold)
        log.info("face detector loaded (%s)", settings.yunet_path)
        return detector
    except FaceDetectorUnavailable as exc:
        if settings.face_check == "required":
            raise RuntimeError(
                f"face_check=required but the detector could not load: {exc}"
            ) from exc
        log.warning("face detector unavailable (face_check=best_effort): %s", exc)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.overlay_dir.mkdir(parents=True, exist_ok=True)

    store = Store(settings.db_path)
    compute = ComputeClient(settings.compute_url, settings.compute_secret, settings.compute_timeout_s)
    detector = _load_face_detector(settings)

    try:
        labels = await compute.labels()
        label_names = {l["name"] for l in labels.get("labels", [])}
        log.info("compute reachable, %d labels", len(label_names))
    except ComputeError as exc:
        labels, label_names = {"version": 1, "labels": []}, set()
        log.warning("could not fetch labels at startup: %s", exc)

    state.update(
        settings=settings,
        store=store,
        compute=compute,
        detector=detector,
        labels=labels,
        label_names=label_names,
    )
    sweeper = asyncio.create_task(
        retention_loop(store, settings.overlay_dir, settings.retention_days)
    )
    try:
        yield
    finally:
        sweeper.cancel()
        try:
            await sweeper
        except asyncio.CancelledError:
            pass
        await compute.aclose()
        store.close()
        state.clear()


app = FastAPI(title="Fashion Police — backend", lifespan=lifespan)

_cors = get_settings().cors_origins
if _cors:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


def require_kiosk(authorization: str | None = Header(default=None)) -> None:
    token = get_settings().kiosk_token
    if not token:
        return
    expected = f"Bearer {token}"
    if authorization is None or not secrets.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail="missing or invalid kiosk token")


def _record_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


# ----------------------------------------------------------------------------


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    compute: ComputeClient = state["compute"]
    try:
        hz = await compute.healthz()
        compute_status = str(hz.get("status", "ok"))
    except Exception:
        compute_status = "unreachable"
    detector = state.get("detector")
    settings: Settings = state["settings"]
    fc = "off" if settings.face_check == "off" else ("loaded" if detector else "unavailable")
    return HealthResponse(compute=compute_status, face_check=fc)


@app.get("/api/v1/labels", response_model=LabelsResponse)
async def labels() -> LabelsResponse:
    compute: ComputeClient = state["compute"]
    try:
        data = await compute.labels()
        state["labels"] = data
        state["label_names"] = {l["name"] for l in data.get("labels", [])}
    except ComputeError:
        data = state.get("labels") or {"version": 1, "labels": []}
    return LabelsResponse(**data)


@app.post(
    "/api/v1/analyze",
    response_model=AnalyzeResponse,
    responses={422: {"description": "a face was still visible; nothing stored"}},
    dependencies=[Depends(require_kiosk)],
)
async def analyze(frame: UploadFile = File(...)) -> AnalyzeResponse:
    settings: Settings = state["settings"]

    if frame.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="expected image/jpeg, image/png or image/webp")

    data = await frame.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(status_code=413, detail="image too large")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="not a decodable image")

    w, h = image.size
    if min(w, h) < settings.min_image_edge or max(w, h) > settings.max_image_edge:
        raise HTTPException(status_code=400, detail="image dimensions out of range")

    # --- verify-and-discard: a face here means the client redaction failed ---
    detector: FaceDetector | None = state.get("detector")
    if detector is not None:
        if await asyncio.to_thread(detector.has_face, image):
            log.info("rejected upload: face still visible")
            return JSONResponse(
                status_code=422,
                content={
                    "error": "face_visible",
                    "message": (
                        "A face was still visible in the image. Nothing was sent or "
                        "stored. Step back into frame and try again."
                    ),
                },
            )
    elif settings.face_check == "required":  # pragma: no cover - guarded at startup
        raise HTTPException(status_code=503, detail="face detector unavailable")

    # --- forward the clean frame to compute ---
    compute: ComputeClient = state["compute"]
    try:
        result = await compute.infer(data, frame.content_type)
    except ComputeError as exc:
        raise HTTPException(status_code=exc.status, detail=str(exc))

    record_id = _record_id()
    overlay_rel: str | None = None
    b64 = result.get("anonymized_overlay_b64")
    if b64:
        try:
            raw = base64.b64decode(b64)
            path = settings.overlay_dir / f"{record_id}.jpg"
            path.write_bytes(raw)
            overlay_rel = f"overlays/{path.name}"
        except (binascii.Error, ValueError, OSError) as exc:  # pragma: no cover
            log.warning("could not store overlay for %s: %s", record_id, exc)

    top = result["top"]
    ranked = result["ranked"]
    store: Store = state["store"]
    await store.save_prediction(
        record_id,
        top_label=top["name"],
        top_score=float(top["score"]),
        uncertain=bool(result.get("uncertain", False)),
        ranked=ranked,
        overlay_path=overlay_rel,
    )

    return AnalyzeResponse(
        record_id=record_id,
        top=top,
        ranked=ranked,
        uncertain=bool(result.get("uncertain", False)),
        display_overlay_b64=result.get("display_overlay_b64", ""),
    )


@app.post("/api/v1/feedback/{record_id}", dependencies=[Depends(require_kiosk)])
async def feedback(record_id: str, body: FeedbackRequest) -> dict:
    label_names: set[str] = state.get("label_names") or set()
    if label_names and body.correct_label not in label_names:
        raise HTTPException(status_code=400, detail="unknown label")
    store: Store = state["store"]
    if not await store.save_feedback(record_id, body.correct_label):
        raise HTTPException(status_code=404, detail="unknown record_id")
    return {"ok": True}


@app.get("/api/v1/stats", response_model=StatsResponse)
async def stats() -> StatsResponse:
    store: Store = state["store"]
    return StatsResponse(**await store.statistics())
