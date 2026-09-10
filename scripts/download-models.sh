#!/usr/bin/env bash
# Download the Hugging Face models the compute tier needs, into the local weight
# cache (HF_HOME). Idempotent: a model already present in the cache is skipped by
# `hf download`. Nothing here is committed to git.
#
#   ./scripts/download-models.sh            # into $HF_HOME (default ./models)
#   ./scripts/download-models.sh --force    # re-verify / re-fetch everything
#
# Reads HF_HOME, HF_TOKEN and the model ids from .env if present.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# --- load .env (only the keys we care about) ---------------------------------
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Resolve HF_HOME to an absolute path (it may be a repo-relative value in .env,
# and the downloader runs from a subdirectory).
case "${HF_HOME:-}" in
  "")   HF_HOME="$repo_root/models" ;;
  /*)   : ;;
  *)    HF_HOME="$repo_root/${HF_HOME#./}" ;;
esac
export HF_HOME
mkdir -p "$HF_HOME"

SEG_MODEL="${FP_COMPUTE_SEGMENTATION_MODEL:-mattmdjaga/segformer_b2_clothes}"
STYLE_MODEL="${FP_COMPUTE_STYLE_MODEL:-patrickjohncyh/fashion-clip}"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# --- pick a downloader ------------------------------------------------------
if command -v hf >/dev/null 2>&1; then
  cli=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
  cli=huggingface-cli
else
  cli=""
fi

if [[ -n "$cli" ]]; then
  dl() {
    local args=("$1")
    [[ $FORCE -eq 1 ]] && args+=(--force-download)
    "$cli" download "${args[@]}" >/dev/null
  }
elif command -v uv >/dev/null 2>&1 && [[ -d compute/.venv ]]; then
  dl() {
    (cd compute && FORCE=$FORCE uv run python - "$1" <<'PY'
import os, sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], force_download=os.environ.get("FORCE") == "1")
PY
    )
  }
else
  echo "error: need the 'hf' CLI (pip install huggingface_hub) or a synced compute/.venv" >&2
  exit 1
fi

[[ -n "${HF_TOKEN:-}" ]] && export HF_TOKEN

echo "cache:  $HF_HOME"
for model in "$SEG_MODEL" "$STYLE_MODEL"; do
  echo "  fetch $model"
  dl "$model"
done
echo "done."
