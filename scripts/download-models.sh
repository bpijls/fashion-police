#!/usr/bin/env bash
# Download the models the services need. Nothing here is committed to git.
#
#   ./scripts/download-models.sh            # fetch what's missing
#   ./scripts/download-models.sh --force    # re-verify / re-fetch everything
#
# - compute tier: SegFormer + FashionCLIP, into the HF cache ($HF_HOME).
# - backend tier: the YuNet face-detector ONNX, as a plain file in ./models/.
#
# Reads HF_HOME, HF_TOKEN and the model ids from .env if present.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Resolve HF_HOME to an absolute path (it may be repo-relative in .env, and the
# downloader can run from a subdirectory).
case "${HF_HOME:-}" in
  "")   HF_HOME="$repo_root/models" ;;
  /*)   : ;;
  *)    HF_HOME="$repo_root/${HF_HOME#./}" ;;
esac
export HF_HOME
mkdir -p "$HF_HOME"

MODELS_DIR="$repo_root/models"
mkdir -p "$MODELS_DIR"

SEG_MODEL="${FP_COMPUTE_SEGMENTATION_MODEL:-mattmdjaga/segformer_b2_clothes}"
STYLE_MODEL="${FP_COMPUTE_STYLE_MODEL:-patrickjohncyh/fashion-clip}"
YUNET_REPO="${FP_BACKEND_YUNET_REPO:-opencv/face_detection_yunet}"
YUNET_FILE="${FP_BACKEND_YUNET_FILE:-face_detection_yunet_2023mar.onnx}"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN

# --- pick a downloader -----------------------------------------------------
if command -v hf >/dev/null 2>&1; then
  cli=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  cli=huggingface-cli
else
  cli=""
fi

if [[ -n "$cli" ]]; then
  dl_repo() {
    local a=("$1"); [[ $FORCE -eq 1 ]] && a+=(--force-download)
    "$cli" download "${a[@]}" >/dev/null
  }
  dl_file() {  # repo file dest_dir
    local a=("$1" "$2" --local-dir "$3"); [[ $FORCE -eq 1 ]] && a+=(--force-download)
    "$cli" download "${a[@]}" >/dev/null
  }
elif command -v uv >/dev/null 2>&1 && [[ -d compute/.venv ]]; then
  _py() { (cd compute && FORCE=$FORCE uv run python - "$@"); }
  dl_repo() {
    _py "$1" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], force_download=os.environ.get("FORCE") == "1")
PY
  }
  dl_file() {
    _py "$1" "$2" "$3" <<'PY'
import os, sys
from huggingface_hub import hf_hub_download
hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3],
                force_download=os.environ.get("FORCE") == "1")
PY
  }
else
  echo "error: need the 'hf' CLI (pip install huggingface_hub) or a synced compute/.venv" >&2
  exit 1
fi

echo "hf cache:    $HF_HOME"
for model in "$SEG_MODEL" "$STYLE_MODEL"; do
  echo "  fetch $model"
  dl_repo "$model"
done

echo "models dir:  $MODELS_DIR"
if [[ $FORCE -eq 1 || ! -f "$MODELS_DIR/$YUNET_FILE" ]]; then
  echo "  fetch $YUNET_REPO/$YUNET_FILE"
  dl_file "$YUNET_REPO" "$YUNET_FILE" "$MODELS_DIR"
else
  echo "  have  $YUNET_FILE"
fi

# The kiosk's TF.js detectors (MoveNet, BlazeFace) are loaded by the
# @tensorflow-models libraries at runtime and cached in the browser. Bundling
# them for a fully-offline kiosk is optional and done separately — see
# frontend/README.md ("Offline model bundling").

echo "done."
