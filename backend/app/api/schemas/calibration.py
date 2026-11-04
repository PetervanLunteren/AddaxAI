"""Schemas for the Calibration page (`routers/calibration.py`)."""

from typing import Literal

from pydantic import BaseModel, Field

CalibrationAnswerValue = Literal["right", "wrong", "unsure"]


class CalibrationPoint(BaseModel):
    score: float
    p: float = Field(..., description="Chance the label is right at this score")
    lower: float
    upper: float


class CalibrationThreshold(BaseModel):
    target: float = Field(..., description="Share of right labels, e.g. 0.95")
    score: float | None = Field(
        ..., description="Score from which the target holds; None when not reached"
    )


class CalibrationLabel(BaseModel):
    label: str = Field(..., description="The AI label, as stored on the detection")
    common_name: str
    scientific_name: str
    detections: int = Field(..., description="Detections the sample can draw from")
    right: int
    wrong: int
    unsure: int
    right_needed: int = Field(..., description="Right answers still needed for a curve")
    wrong_needed: int = Field(..., description="Wrong answers still needed for a curve")
    thresholds: list[CalibrationThreshold] | None = Field(
        ..., description="None until the minimum is reached"
    )
    curve: list[CalibrationPoint] | None = Field(
        ..., description="Over the answered score range only; None until the minimum"
    )
    no_error_bound: float | None = Field(
        ...,
        description=(
            "With answers but no wrong one: lower 95% bound on the share "
            "right (rule of three)"
        ),
    )


class CalibrationResponse(BaseModel):
    labels: list[CalibrationLabel]


class CalibrationSampleItem(BaseModel):
    detection_id: str
    score: float


class CalibrationAnswerCreate(BaseModel):
    detection_id: str
    answer: CalibrationAnswerValue
