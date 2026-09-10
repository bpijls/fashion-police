"""Real end-to-end smoke test — downloads the actual models and runs one inference.

Not part of the pytest suite (slow, network). Run manually:
    FP_COMPUTE_STYLES_PATH=../styles.yaml uv run python tests/smoke_real.py [image.jpg]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings, load_taxonomy, resolve_device
from app.pipeline import OutfitClassifier

styles = Path(__file__).resolve().parents[2] / "styles.yaml"
settings = Settings(styles_path=styles)
taxonomy = load_taxonomy(styles)
device = resolve_device(settings.device)
print(f"device={device} styles={len(taxonomy.styles)}")

clf = OutfitClassifier(settings, device, taxonomy)
print("models loaded")

if len(sys.argv) > 1:
    img = Image.open(sys.argv[1])
else:
    # synthetic: dark rectangle over a light ground (no real person, just exercises the path)
    from PIL import ImageDraw

    img = Image.new("RGB", (512, 900), (235, 235, 235))
    d = ImageDraw.Draw(img)
    d.rectangle([160, 120, 360, 780], fill=(30, 30, 35))

res = clf.classify(img)
print("timings_ms:", res.timings_ms)
print("uncertain:", res.uncertain)
for r in res.ranked[:5]:
    print(f"  {r['score']:.3f}  {r['name']}")
assert abs(sum(r["score"] for r in res.ranked) - 1.0) < 1e-3
assert len(res.anonymized_overlay_b64) > 100
print("OK")
