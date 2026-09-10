# Fashion Police — kiosk (clean-room rebuild)

A digital-mirror installation: it photographs a visitor, reads their clothing, and
tells them which "class" they belong to. Concept after Ahnjili Zhuparris.

Three separate tiers:

| Tier | Runs on | Stack | Status |
|---|---|---|---|
| **`compute/`** | a CUDA host | FastAPI · SegFormer · FashionCLIP (zero-shot) | ✅ built · deployed |
| **`backend/`** | the reverse-proxy host | FastAPI gateway · SQLite · YuNet face verify-and-discard | ✅ built · deployed |
| **`frontend/`** | the kiosk device (browser only) | React · TF.js MoveNet + BlazeFace · client-side face redaction | ✅ built · deployed |

The whole chain is live end-to-end: kiosk browser → Caddy → frontend (SPA +
`/api` proxy) → backend → compute. Verified with a fake camera driving a real
capture, redaction, prediction, and feedback.

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

## Deploy — edge tier (backend + frontend)

The edge tier runs on a host reached over SSH that already runs a
[Caddy Docker Proxy](https://github.com/lucaslorentz/caddy-docker-proxy) on an
external ingress network. All host-specific values stay in `.env`.

```bash
ssh <edge-host>
git clone git@github.com:bpijls/fashion-police.git && cd fashion-police

# 1. config — never commit this file
cp .env.example .env
#    fill in at least:
#      COMPUTE_URL      — where the backend reaches the compute tier
#      COMPUTE_SECRET   — shared secret the compute host expects
#      APP_DOMAIN       — public hostname Caddy should serve
#      CADDY_NETWORK    — name of the Caddy ingress network (e.g. proxy)

# 2. the backend needs the YuNet face detector as a plain file
mkdir -p models
curl -fL -o models/face_detection_yunet_2023mar.onnx \
  https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx

# 3. build + start (the --env-file flag is required: compose otherwise
#    looks for .env next to the manifest, in deploy/)
docker compose --env-file .env -f deploy/edge.compose.yml up -d --build
```

Caddy picks up the container labels, routes `APP_DOMAIN`, and issues a TLS
certificate automatically. Verify:

```bash
curl -s https://$APP_DOMAIN/api/v1/health      # {"backend":"ok","compute":"ok",...}
```

Redeploy after a change:

```bash
ssh <edge-host> 'cd fashion-police && git pull \
  && docker compose --env-file .env -f deploy/edge.compose.yml up -d --build'
```

## Layout

```
compute/    inference service (SegFormer + FashionCLIP)   deploy/compute.compose.yml
backend/    API gateway (validation, face check, storage)  deploy/edge.compose.yml
frontend/   kiosk SPA (camera, redaction, result)          deploy/edge.compose.yml
deploy/     docker compose manifests
scripts/    helper scripts (download-models.sh)
styles.yaml the taxonomy
```

`compute` runs on a CUDA host; `backend` + `frontend` run together on the
reverse-proxy host behind Caddy.

The `studio-rai-group-project-HX1R/` and `fp2/` directories are earlier
implementations kept for reference and are gitignored.
