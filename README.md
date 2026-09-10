# Fashion Police — kiosk (clean-room rebuild)

A digital-mirror installation: it photographs a visitor, reads their clothing, and
tells them which "class" they belong to. Concept after Ahnjili Zhuparris.

Three separate tiers:

| Tier | Runs on | Stack | Status |
|---|---|---|---|
| **`compute/`** | a CUDA host | FastAPI · SegFormer · FashionCLIP (zero-shot) | ✅ built · deployed |
| **`backend/`** | the reverse-proxy host | FastAPI gateway · SQLite · face verify-and-discard | ⬜ next |
| **`frontend/`** | the kiosk device (browser only) | React · TF.js MoveNet · client-side face redaction | ⬜ after |

The kiosk device runs **only a browser**. The camera frame is redacted **in the
browser** (face blacked out) before anything is sent — the raw image never leaves
the device.

## Configuration

All deployment-specific values — hostnames, addresses, secrets, the public domain —
live in `.env` (gitignored). Copy `.env.example` and fill it in:

```bash
cp .env.example .env
```

## Taxonomy

What the installation sorts people into lives entirely in [`styles.yaml`](styles.yaml).
Edit it, restart `compute` + `backend`, done. No retraining, no dataset — FashionCLIP
scores the photo against the text prompts zero-shot.

## Models

Weights are **not** committed. They download from Hugging Face on first use, or
ahead of time:

```bash
./scripts/download-models.sh
```

## Build order

1. **Compute** — `compute/README.md`. Stateless inference. `POST /infer` → ranked
   styles + anonymised overlays.
2. **Backend** — gateway between kiosk and compute: validates, rejects any frame
   with a detectable face, forwards clean frames, stores overlay + prediction +
   feedback, serves `/api/*`.
3. **Frontend** — the kiosk React app: live mirror, pose-gated capture, client-side
   redaction, result screen, minimal feedback tap.

Deploy manifests are in `deploy/`.

## Layout

```
compute/    inference service (SegFormer + FashionCLIP)
backend/    API gateway            (stage 2)
frontend/   kiosk app              (stage 3)
deploy/     docker compose manifests
scripts/    helper scripts
styles.yaml the taxonomy
```

The `studio-rai-group-project-HX1R/` and `fp2/` directories are earlier
implementations kept for reference and are gitignored.
