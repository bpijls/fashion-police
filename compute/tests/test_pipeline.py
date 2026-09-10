from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image

from app import pipeline
from app.config import Settings, Taxonomy, load_taxonomy
from app.segmentation import SegmentationResult
from pathlib import Path

REPO_STYLES = Path(__file__).resolve().parents[2] / "styles.yaml"


class FakeSeg:
    def __init__(self, *_a, **_kw) -> None:
        pass

    def segment(self, image: Image.Image) -> SegmentationResult:
        w, h = image.size
        return SegmentationResult(
            seg_map=np.zeros((h, w), dtype=np.uint8),
            display_overlay=Image.new("RGB", (w, h), (10, 20, 30)),
            anonymized_overlay=Image.new("RGB", (w, h), (255, 255, 255)),
            face_pixels=0,
            person_pixels=w * h,
        )


class FakeScorer:
    def __init__(self, names: list[str], winner: str) -> None:
        self.names = names
        self.winner = winner

    def score(self, image, seg):
        return [{"name": n, "score": 0.9 if n == self.winner else 0.1} for n in self.names]


def _build(monkeypatch, winner: str, uncertain_below: float = 0.22) -> pipeline.OutfitClassifier:
    tax: Taxonomy = load_taxonomy(REPO_STYLES)
    monkeypatch.setattr(pipeline, "SegmentationModel", FakeSeg)
    monkeypatch.setattr(
        pipeline, "build_scorer", lambda *a, **kw: FakeScorer(tax.names, winner)
    )
    settings = Settings(styles_path=REPO_STYLES, uncertain_below=uncertain_below, max_image_edge=128)
    return pipeline.OutfitClassifier(settings, "cpu", tax)


def test_classify_ranks_and_enriches(monkeypatch) -> None:
    clf = _build(monkeypatch, winner="gothic")
    result = clf.classify(Image.new("RGB", (400, 900), (128, 128, 128)))

    assert result.top["name"] == "gothic"
    assert result.top["display_name"] == "Gothic"
    assert result.ranked == sorted(result.ranked, key=lambda r: r["score"], reverse=True)
    assert result.uncertain is False
    assert result.timings_ms["total"] >= 0

    png = base64.b64decode(result.display_overlay_b64)
    assert Image.open(io.BytesIO(png)).size == (57, 128)  # long edge clamped to 128


def test_uncertain_flag(monkeypatch) -> None:
    clf = _build(monkeypatch, winner="preppy", uncertain_below=0.95)
    result = clf.classify(Image.new("RGB", (100, 100)))
    assert result.uncertain is True


def test_fit_no_upscale() -> None:
    img = Image.new("RGB", (50, 50))
    assert pipeline._fit(img, 128).size == (50, 50)
