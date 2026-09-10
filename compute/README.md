# Fashion Police — compute tier

Stateless inference service. Given one image it returns a ranked list of style
scores plus two anonymised overlays.

- **Segmentation:** SegFormer `mattmdjaga/segformer_b2_clothes` (18 clothing/body classes).
- **Style scoring:** FashionCLIP `patrickjohncyh/fashion-clip`, zero-shot against the
  text prompts in `../styles.yaml`. Pluggable via `FP_COMPUTE_SCORER`.

The taxonomy lives entirely in `../styles.yaml`. Change it, restart the service —
no retraining.

## API

| Method | Path | Notes |
|---|---|---|
| `GET` | `/healthz` | `{status, ready, device, cuda}` |
| `GET` | `/labels` | current taxonomy (name, display_name, blurb, accent) |
| `POST` | `/infer` | multipart `frame` (image). Requires `X-Compute-Secret` when `FP_COMPUTE_SECRET` is set. |

`/infer` response:

```json
{
  "ranked": [{"name": "gothic", "score": 0.62, "display_name": "Gothic", "accent": "#4a3f5c"}, ...],
  "top": {"name": "gothic", "score": 0.62, ...},
  "uncertain": false,
  "anonymized_overlay_b64": "<jpeg>",
  "display_overlay_b64": "<jpeg>",
  "timings_ms": {"segmentation": 88.1, "scoring": 19.4, "total": 108.0}
}
```

## Configuration

Everything is read from the environment (see `../.env.example`). Prefix `FP_COMPUTE_`.

| Var | Default | Meaning |
|---|---|---|
| `FP_COMPUTE_STYLES_PATH` | `/config/styles.yaml` | taxonomy file |
| `FP_COMPUTE_DEVICE` | `auto` | `auto` \| `cuda` \| `cpu` |
| `FP_COMPUTE_SCORER` | `fashionclip` | `fashionclip` \| `sklearn` (stub) |
| `FP_COMPUTE_SEGMENTATION_MODEL` | `mattmdjaga/segformer_b2_clothes` | HF repo id |
| `FP_COMPUTE_STYLE_MODEL` | `patrickjohncyh/fashion-clip` | HF repo id |
| `FP_COMPUTE_SECRET` | _(empty)_ | shared secret for `/infer` |
| `FP_COMPUTE_UNCERTAIN_BELOW` | `0.22` | top-score threshold for the `uncertain` flag |
| `FP_COMPUTE_MAX_IMAGE_EDGE` | `1024` | input is downscaled to this long edge |
| `HF_HOME` | `/models` | weight cache (mount a volume) |

## Local development (CPU)

```bash
cd compute
uv sync --extra dev
FP_COMPUTE_STYLES_PATH=../styles.yaml uv run pytest          # fast, models mocked

# real run — first request downloads ~700 MB and runs on CPU (3-5 s):
set -a; source ../.env; set +a
FP_COMPUTE_STYLES_PATH=../styles.yaml uv run uvicorn app.main:app --port 8801
```

Pre-fetch the weights instead of waiting on the first request:

```bash
./scripts/download-models.sh          # from the repo root
```

## Deploy (on the CUDA host)

```bash
# from the repo root, with a filled-in .env
docker compose --env-file .env -f deploy/compute.compose.yml up -d --build
```

Bump the `nvcr.io/nvidia/pytorch` tag in `compute/Dockerfile` if the pinned one
predates the host GPU's arch.
