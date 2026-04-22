"""
Schemas for the classification performance endpoint
(confusion matrix + classification report).
"""

from pydantic import BaseModel, Field


class ClassMetrics(BaseModel):
    class_name: str
    display_name: str
    support: int
    precision: float | None
    recall: float | None
    f1: float | None


class PerformanceResponse(BaseModel):
    taxonomic_rank: str = Field(
        ...,
        description="Aggregation rank used (all / class / order / family / genus / species)",
    )
    classes: list[str] = Field(..., description="Class identifiers in matrix order")
    class_display_names: list[str] = Field(
        ...,
        description="Human-friendly labels, aligned with `classes` by index",
    )
    class_taxonomy_ids: list[str | None] = Field(
        ...,
        description=(
            "LabelTaxonomy UUID per class, aligned with `classes`. None for "
            "detector categories (animal / person / vehicle), the 'other' "
            "bucket, and any class with no direct taxonomy row."
        ),
    )
    matrix: list[list[int]] = Field(
        ...,
        description="matrix[i][j] = count of (true_class=i, predicted_class=j)",
    )
    row_totals: list[int]
    col_totals: list[int]
    grand_total: int
    per_class: list[ClassMetrics]
    macro_precision: float | None
    macro_recall: float | None
    macro_f1: float | None
    weighted_precision: float | None
    weighted_recall: float | None
    weighted_f1: float | None
    skipped_no_prediction: int = Field(
        ...,
        description=(
            "Verified detections excluded because the original prediction "
            "was unavailable (typically analysed before the prediction-"
            "history columns existed)."
        ),
    )
    skipped_unverified: int = Field(
        ...,
        description="Detections matching the filter but not yet verified.",
    )
    has_classifier: bool = Field(
        ...,
        description="False for detector-only projects.",
    )
    top_n_applied: int | None
    other_bucket_present: bool
