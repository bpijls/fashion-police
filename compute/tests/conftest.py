from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def rgb_image() -> Image.Image:
    arr = (np.random.default_rng(0).random((240, 160, 3)) * 255).astype("uint8")
    return Image.fromarray(arr, "RGB")


@pytest.fixture
def jpeg_bytes(rgb_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    rgb_image.save(buf, format="JPEG")
    return buf.getvalue()
