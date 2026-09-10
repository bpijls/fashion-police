from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field

from PIL import Image

from .config import Settings, Taxonomy
from .scorers import Scorer, build_scorer
from .segmentation import SegmentationModel


def _b64_jpeg(image: Image.Image, quality: int = 90) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _fit(image: Image.Image, max_edge: int) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_edge:
        return image
    scale = max_edge / max(w, h)
    return image.resize((round(w * scale), round(h * scale)), Image.LANCZOS)


@dataclass
class InferResult:
    ranked: list[dict]
    top: dict
    uncertain: bool
    anonymized_overlay_b64: str
    display_overlay_b64: str
    timings_ms: dict[str, float] = field(default_factory=dict)


class OutfitClassifier:
    """SegFormer segmentation + a pluggable style scorer."""

    def __init__(self, settings: Settings, device: str, taxonomy: Taxonomy) -> None:
        self.settings = settings
        self.taxonomy = taxonomy
        self.seg = SegmentationModel(settings.segmentation_model, device)
        self.scorer: Scorer = build_scorer(
            settings.scorer,
            model_name=settings.style_model,
            device=device,
            taxonomy=taxonomy,
        )

    def classify(self, image: Image.Image) -> InferResult:
        image = _fit(image.convert("RGB"), self.settings.max_image_edge)

        t0 = time.perf_counter()
        seg = self.seg.segment(image)
        t1 = time.perf_counter()
        raw = self.scorer.score(image, seg)
        t2 = time.perf_counter()

        ranked = sorted(raw, key=lambda r: r["score"], reverse=True)
        for r in ranked:
            style = self.taxonomy.by_name(r["name"])
            if style is not None:
                r["display_name"] = style.display_name
                r["accent"] = style.accent
        top = ranked[0]

        return InferResult(
            ranked=ranked,
            top=top,
            uncertain=bool(top["score"] < self.settings.uncertain_below),
            anonymized_overlay_b64=_b64_jpeg(seg.anonymized_overlay),
            display_overlay_b64=_b64_jpeg(seg.display_overlay),
            timings_ms={
                "segmentation": round((t1 - t0) * 1000, 1),
                "scoring": round((t2 - t1) * 1000, 1),
                "total": round((t2 - t0) * 1000, 1),
            },
        )
