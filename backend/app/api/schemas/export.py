"""Schemas for the project export endpoints that answer with JSON rather
than a file: today the recognition-file export, which writes into the
deployment folders and reports what it did per deployment."""

from pydantic import BaseModel


class RecognitionJsonWritten(BaseModel):
    deployment_id: str
    folder_path: str
    output_path: str
    image_count: int


class RecognitionJsonSkipped(BaseModel):
    deployment_id: str
    folder_path: str | None
    reason: str


class RecognitionJsonExportResult(BaseModel):
    """One entry per deployment in scope, written or skipped. A folder
    that is unmounted or read-only is a skipped entry with its reason,
    not a failed request: the other deployments' files are still worth
    writing, and a project's folders live on several drives."""

    written: list[RecognitionJsonWritten]
    skipped: list[RecognitionJsonSkipped]
