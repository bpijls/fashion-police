from __future__ import annotations

from pydantic import BaseModel, Field


class RankedStyle(BaseModel):
    name: str
    score: float
    display_name: str | None = None
    accent: str | None = None


class AnalyzeResponse(BaseModel):
    record_id: str
    top: RankedStyle
    ranked: list[RankedStyle]
    uncertain: bool
    display_overlay_b64: str


class FaceVisibleResponse(BaseModel):
    error: str = "face_visible"
    message: str = (
        "A face was still visible in the image. Nothing was sent or stored. "
        "Step back into frame and try again."
    )


class FeedbackRequest(BaseModel):
    correct_label: str = Field(min_length=1)


class LabelInfo(BaseModel):
    name: str
    display_name: str
    blurb: str = ""
    accent: str = "#888888"


class LabelsResponse(BaseModel):
    version: int = 1
    labels: list[LabelInfo]


class StatsResponse(BaseModel):
    total_predictions: int
    total_feedback: int
    feedback_rate: float
    top_predictions: list[tuple[str, int]]
    user_corrections: list[tuple[str, int]]


class HealthResponse(BaseModel):
    backend: str = "ok"
    compute: str
    face_check: str
