# Fashion Police — kiosk frontend

React + Vite. Runs in a browser on the kiosk device; that device runs **nothing
else** — no Python, no local model server.

## What it does

1. Shows a full-screen mirror of the camera (the video never leaves the device).
2. In-browser TF.js: **MoveNet** (person + "hand at eye level" pose gate) and
   **BlazeFace** (face boxes).
3. On a held pose or a button/key press it captures one frame, then **in the
   browser**: blacks out every face region (detected boxes ∪ a pose-derived head
   box), crops to the person, and shows the result — *"this is what we send"*.
4. Uploads only that redacted JPEG to `POST /api/v1/analyze`.
5. Renders the verdict, the top-3, and a "set it straight" feedback tap.
6. Returns to the attract loop after ~40 s.

The raw frame is only ever held inside the capture handler — never in component
state, a ref, `localStorage`, or the network.

## State machine (`src/kiosk.ts`)

`loading → idle → ready → redacting → analysing → result → (idle)`
with `error` on failure and `ready` (+ hint) if the backend still sees a face.

## Configuration (build-time, `VITE_` prefix)

| Var | Default | Meaning |
|---|---|---|
| `VITE_API_BASE` | `/api/v1` | backend base path (same origin in production) |
| `VITE_MODEL_BASE` | _(empty)_ | if set (e.g. `/models`), load the TF.js detectors from bundled files instead of the Google CDN |

Runtime knobs (hold time, result timeout, capture keys) are in `src/config.ts`.

## Develop

```bash
cd frontend
npm install
# proxy /api to a running backend:
VITE_DEV_API=http://127.0.0.1:8800 npm run dev
```

Then open the printed URL (`getUserMedia` needs `localhost` or HTTPS).

## Offline model bundling

The kiosk needs internet on first boot to fetch the TF.js models (MoveNet +
BlazeFace, ~1 MB total), which the browser then caches. To make it fully
offline, vendor the model files:

1. Get `model.json` + weight shards for MoveNet SinglePose Lightning and
   BlazeFace (the `@tensorflow-models` packages document their current sources;
   hosting has moved between tfhub / Kaggle Models).
2. Place them at `frontend/public/models/movenet/` and
   `frontend/public/models/blazeface/`.
3. Build with `VITE_MODEL_BASE=/models`:
   ```bash
   docker compose --env-file .env -f deploy/edge.compose.yml build \
     --build-arg VITE_MODEL_BASE=/models
   ```

`src/lib/vision.ts` then loads `/models/movenet/model.json` and
`/models/blazeface/model.json` instead of the CDN.

## Deploy

Part of `deploy/edge.compose.yml` — the frontend container serves the SPA and
proxies `/api/*` to the backend container, so the kiosk sees one origin. The
outer Caddy Docker Proxy routes `APP_DOMAIN` to it.

The kiosk device itself just needs a browser in kiosk mode pointed at
`https://<APP_DOMAIN>` (Chromium: `--kiosk --incognito`), plus `systemd` to
launch it on boot.
