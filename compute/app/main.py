from __future__ import annotations

import io
import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError

from .config import get_settings, load_taxonomy, resolve_device
from .pipeline import OutfitClassifier

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("fp.compute")
log.setLevel(logging.INFO)

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    device = resolve_device(settings.device)
    taxonomy = load_taxonomy(settings.styles_path)
    log.info("loading models on %s (scorer=%s, styles=%d)", device, settings.scorer, len(taxonomy.styles))
    state["settings"] = settings
    state["device"] = device
    state["taxonomy"] = taxonomy
    state["classifier"] = OutfitClassifier(settings, device, taxonomy)
    state["ready"] = True
    log.info("compute ready")
    yield
    state.clear()


app = FastAPI(title="Fashion Police — compute", lifespan=lifespan)


def require_secret(x_compute_secret: str | None = Header(default=None)) -> None:
    expected = get_settings().secret
    if expected and x_compute_secret != expected:
        raise HTTPException(status_code=401, detail="bad or missing X-Compute-Secret")


@app.get("/healthz")
def healthz() -> dict:
    ready = bool(state.get("ready"))
    cuda = False
    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:
        pass
    return {"status": "ok" if ready else "loading", "ready": ready, "device": state.get("device"), "cuda": cuda}


@app.get("/labels")
def labels() -> dict:
    taxonomy = state.get("taxonomy")
    if taxonomy is None:
        raise HTTPException(status_code=503, detail="not ready")
    return {
        "version": taxonomy.version,
        "labels": [
            {"name": s.name, "display_name": s.display_name, "blurb": s.blurb, "accent": s.accent}
            for s in taxonomy.styles
        ],
    }


@app.post("/infer", dependencies=[Depends(require_secret)])
async def infer(frame: UploadFile = File(...)) -> dict:
    classifier: OutfitClassifier | None = state.get("classifier")
    if classifier is None:
        raise HTTPException(status_code=503, detail="not ready")

    data = await frame.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="not a decodable image")

    result = classifier.classify(image)
    log.info("infer top=%s score=%.3f timings=%s", result.top["name"], result.top["score"], result.timings_ms)
    return {
        "ranked": result.ranked,
        "top": result.top,
        "uncertain": result.uncertain,
        "anonymized_overlay_b64": result.anonymized_overlay_b64,
        "display_overlay_b64": result.display_overlay_b64,
        "timings_ms": result.timings_ms,
    }
