# Fashion Police — backend gateway

Sits between the kiosk and the compute tier. It never persists the raw upload.

Flow of `POST /api/v1/analyze`:

1. validate (content-type ∈ jpeg/png/webp, size ≤ 4 MB, decodable, sane dimensions)
2. **verify-and-discard** — run YuNet on the upload. If a face is detectable the
   client redaction failed: return `422 {"error":"face_visible"}`, forward nothing,
   store nothing.
3. forward the clean frame to `COMPUTE_URL/infer` (shared secret header)
4. persist the *anonymised* overlay (white bg / black face) + the ranked labels
   in SQLite; a retention job deletes both after `RETENTION_DAYS`
5. return `{record_id, top, ranked, uncertain, display_overlay_b64}`

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/api/v1/health` | `{backend, compute, face_check}` |
| `GET` | `/api/v1/labels` | taxonomy, proxied from compute |
| `POST` | `/api/v1/analyze` | multipart `frame`; `422` if a face is visible |
| `POST` | `/api/v1/feedback/{record_id}` | `{ "correct_label": "..." }` |
| `GET` | `/api/v1/stats` | totals, feedback rate, correction histogram |

Set `KIOSK_TOKEN` to require `Authorization: Bearer <token>` on `/analyze` and `/feedback`.

## Configuration

Env, prefix `FP_BACKEND_`. The shared names (`COMPUTE_URL`, `COMPUTE_SECRET`,
`KIOSK_TOKEN`, `RETENTION_DAYS`) are also accepted so one `.env` works. See
`../.env.example`.

| Var | Default | Meaning |
|---|---|---|
| `COMPUTE_URL` | `http://127.0.0.1:8801` | compute base URL (private network) |
| `COMPUTE_SECRET` | _(empty)_ | shared secret sent to compute |
| `KIOSK_TOKEN` | _(empty)_ | bearer token; empty = open |
| `FP_BACKEND_FACE_CHECK` | `required` | `required` \| `best_effort` \| `off` |
| `FP_BACKEND_YUNET_PATH` | `/models/face_detection_yunet_2023mar.onnx` | detector model |
| `FP_BACKEND_DATA_DIR` | `./data` (`/srv/data` in container) | sqlite + overlays |
| `RETENTION_DAYS` | `14` | overlay + row lifetime |
| `FP_BACKEND_CORS_ORIGINS` | `[]` | only if the frontend is another origin |

## Local development

```bash
cd backend
uv sync --extra dev
uv run pytest                                    # fast; compute + detector faked

# run against a live compute tier:
set -a; source ../.env; set +a
export COMPUTE_URL=http://<compute-host>:8801
export FP_BACKEND_YUNET_PATH="$PWD/../models/face_detection_yunet_2023mar.onnx"
export FP_BACKEND_DATA_DIR="$PWD/data"
uv run uvicorn app.main:app --port 8800
```

Fetch the YuNet model first: `./scripts/download-models.sh` (repo root).

## Deploy

```bash
# reverse-proxy host, repo synced (excluding .env), .env filled in with
# COMPUTE_URL=<compute private address>, COMPUTE_SECRET (same as compute),
# APP_DOMAIN, CADDY_NETWORK:
./scripts/download-models.sh
docker compose --env-file .env -f deploy/backend.compose.yml up -d --build
```
