"""Style scorers.

A scorer turns (image, segmentation) into a ranked list of style scores over the
current taxonomy. The default is zero-shot FashionCLIP; `SklearnScorer` is a stub
kept so a trained classifier can be dropped in later without touching the API.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from PIL import Image

from .config import Taxonomy
from .segmentation import SegmentationResult


class Scorer(Protocol):
    def score(self, image: Image.Image, seg: SegmentationResult) -> list[dict]:
        """Return [{"name": str, "score": float}, ...] summing to ~1.0, unsorted."""
        ...


class FashionClipScorer:
    """Zero-shot: cosine similarity between the image embedding and one text
    embedding per style prompt, softmaxed. Ported from
    responsibleIT/fashion-police `src/scripts/style_predictor.py`."""

    def __init__(self, model_name: str, device: str, taxonomy: Taxonomy) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self._torch = torch
        self.device = device
        self.taxonomy = taxonomy
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self._text_embeds = self._embed_text(taxonomy.prompts)

    @staticmethod
    def _as_tensor(out):
        # transformers < 5 returns a tensor; >= 5 returns BaseModelOutputWithPooling
        # whose `pooler_output` is the projected CLIP embedding.
        return getattr(out, "pooler_output", out)

    def _embed_text(self, prompts: list[str]):
        torch = self._torch
        inputs = self.processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
        with torch.inference_mode():
            feats = self._as_tensor(self.model.get_text_features(**inputs))
        return feats / feats.norm(dim=-1, keepdim=True)

    def _embed_image(self, image: Image.Image):
        torch = self._torch
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.inference_mode():
            feats = self._as_tensor(self.model.get_image_features(**inputs))
        return feats / feats.norm(dim=-1, keepdim=True)

    def score(self, image: Image.Image, seg: SegmentationResult) -> list[dict]:
        torch = self._torch
        image_embed = self._embed_image(image)
        sims = (image_embed @ self._text_embeds.T).squeeze(0)
        probs = torch.softmax(sims * 100.0, dim=0).to("cpu").numpy()
        return [
            {"name": name, "score": float(p)}
            for name, p in zip(self.taxonomy.names, probs, strict=True)
        ]


class SklearnScorer:
    """Placeholder for a trained classifier over hand-built features.

    Implement `score()` against the same contract when a joblib model + feature
    extractor are available. Selected via FP_COMPUTE_SCORER=sklearn.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError(
            "SklearnScorer is not implemented yet. Use FP_COMPUTE_SCORER=fashionclip."
        )

    def score(self, image: Image.Image, seg: SegmentationResult) -> list[dict]:  # pragma: no cover
        raise NotImplementedError


def build_scorer(kind: str, *, model_name: str, device: str, taxonomy: Taxonomy) -> Scorer:
    if kind == "fashionclip":
        return FashionClipScorer(model_name, device, taxonomy)
    if kind == "sklearn":
        return SklearnScorer()
    raise ValueError(f"Unknown scorer: {kind!r}")
