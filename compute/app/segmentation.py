"""Clothing segmentation via SegFormer (mattmdjaga/segformer_b2_clothes).

Ported from responsibleIT/fashion-police `src/scripts/load_model.py`, adapted for
GPU execution and to return the two overlays the kiosk needs:

  * display_overlay    -> original photo, background whitened, face blacked,
                          clothing regions tinted (shown to the visitor)
  * anonymized_overlay -> original photo, background whitened, face blacked,
                          clothing left untouched (persisted by the backend)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# SegFormer b2_clothes label space.
ID2LABEL: dict[int, str] = {
    0: "Background", 1: "Hat", 2: "Hair", 3: "Sunglasses", 4: "Upper-clothes",
    5: "Skirt", 6: "Pants", 7: "Dress", 8: "Belt", 9: "Left-shoe", 10: "Right-shoe",
    11: "Face", 12: "Left-leg", 13: "Right-leg", 14: "Left-arm", 15: "Right-arm",
    16: "Bag", 17: "Scarf",
}
FACE_CLASS = 11
BACKGROUND_CLASS = 0

PALETTE: np.ndarray = np.array(
    [
        (255, 255, 255), (128, 0, 0), (255, 0, 0), (255, 165, 0),
        (255, 192, 203), (255, 105, 180), (255, 0, 255), (219, 112, 147),
        (255, 255, 0), (0, 128, 0), (34, 139, 34), (0, 0, 0),
        (75, 0, 130), (138, 43, 226), (0, 191, 255), (135, 206, 235),
        (0, 255, 255), (255, 20, 147),
    ],
    dtype=np.uint8,
)


@dataclass
class SegmentationResult:
    seg_map: np.ndarray  # (H, W) uint8 class indices
    display_overlay: Image.Image
    anonymized_overlay: Image.Image
    face_pixels: int
    person_pixels: int


class SegmentationModel:
    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

        self.device = device
        self._torch = torch
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForSemanticSegmentation.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

    def segment(self, image: Image.Image) -> SegmentationResult:
        torch = self._torch
        image = image.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            upsampled = torch.nn.functional.interpolate(
                logits, size=image.size[::-1], mode="bilinear", align_corners=False
            )
            seg_map = upsampled.argmax(dim=1)[0].to("cpu").numpy().astype(np.uint8)

        base = np.array(image)
        face_mask = seg_map == FACE_CLASS
        bg_mask = seg_map == BACKGROUND_CLASS
        other_mask = ~(face_mask | bg_mask)

        anonymized = base.copy()
        anonymized[face_mask] = (0, 0, 0)
        anonymized[bg_mask] = (255, 255, 255)

        display = anonymized.copy()
        if other_mask.any():
            colours = PALETTE[seg_map % len(PALETTE)]
            blended = (0.6 * display + 0.4 * colours).astype(np.uint8)
            display[other_mask] = blended[other_mask]

        return SegmentationResult(
            seg_map=seg_map,
            display_overlay=Image.fromarray(display),
            anonymized_overlay=Image.fromarray(anonymized),
            face_pixels=int(face_mask.sum()),
            person_pixels=int(other_mask.sum() + face_mask.sum()),
        )
