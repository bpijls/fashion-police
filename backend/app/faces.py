"""Face verify-and-discard.

The kiosk redacts faces in the browser before uploading. This is the server-side
safety net: if a face is still detectable in what arrives, the request is
rejected and nothing is forwarded or stored.

Uses OpenCV's YuNet detector (same model the earlier project relied on). The
ONNX file is not committed — fetch it with scripts/download-models.sh.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("fp.backend.faces")


class FaceDetectorUnavailable(RuntimeError):
    pass


class FaceDetector:
    def __init__(self, model_path: Path, score_threshold: float = 0.6) -> None:
        if not model_path.exists():
            raise FaceDetectorUnavailable(
                f"YuNet model not found at {model_path}. Run scripts/download-models.sh."
            )
        try:
            # input_size is set per-image in detect()
            self._detector = cv2.FaceDetectorYN.create(
                model=str(model_path),
                config="",
                input_size=(320, 320),
                score_threshold=float(score_threshold),
                nms_threshold=0.3,
                top_k=50,
            )
        except cv2.error as exc:  # pragma: no cover - depends on opencv build
            raise FaceDetectorUnavailable(f"could not initialise YuNet: {exc}") from exc
        self.score_threshold = float(score_threshold)

    def has_face(self, image: Image.Image) -> bool:
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]

        # YuNet expects a bounded input; downscale very large frames for speed.
        max_edge = 1024
        scale = min(1.0, max_edge / max(h, w))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            h, w = bgr.shape[:2]

        self._detector.setInputSize((w, h))
        _, faces = self._detector.detect(bgr)
        if faces is None:
            return False
        # column 14 is the detection score
        return any(float(row[14]) >= self.score_threshold for row in faces)
