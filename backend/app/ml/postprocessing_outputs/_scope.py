"""Which files a media export covers.

A folder run copies every file of its one deployment. A project export
picks deployments and can narrow further to liked files, verified files
or a capture-date range. The three media modules (separation, annotated
copies, the preview) must agree exactly on that set: annotation writes
any file separation did not place to a fallback name at the media root,
so a module with a wider scope than its sibling leaks files there. One
object, built once by the worker and the preview endpoint, passed to all
three, is what keeps them equal.

Only "which files". Where they land is the layout's business
(``deployment_folder_in_path`` in ``separate_folders``).

``None`` in place of a scope means today's behaviour: every file of the
project.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from app.api.crud.file_filters import file_flag_clauses
from app.models import Deployment


@dataclass(frozen=True)
class MediaScope:
    # None = every deployment of the project. An empty tuple is an empty
    # export; the endpoint refuses that before a job exists.
    deployment_ids: tuple[str, ...] | None = None
    favorited_only: bool = False
    verified_only: bool = False
    date_from: date | None = None
    date_to: date | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> MediaScope:
        """Build from a job payload or request body as stored on the job
        row: dates are ISO strings there (``model_dump(mode="json")``),
        the id list may be absent or ``None``."""
        ids = payload.get("deployment_ids")
        return cls(
            deployment_ids=tuple(ids) if ids is not None else None,
            favorited_only=bool(payload.get("favorited_only", False)),
            verified_only=bool(payload.get("verified_only", False)),
            date_from=_iso_date(payload.get("date_from")),
            date_to=_iso_date(payload.get("date_to")),
        )

    def deployment_clauses(self) -> list:
        """The deployment part alone. The event vote uses only this, so a
        burst is judged on all its files and the file-level filters only
        decide which of them are copied."""
        if self.deployment_ids is None:
            return []
        return [Deployment.id.in_(self.deployment_ids)]

    def file_clauses(self) -> list:
        """Every clause: deployments plus the file-level filters. Apply to
        a ``File`` query that is joined to ``Deployment``."""
        return [
            *self.deployment_clauses(),
            *file_flag_clauses(
                date_from=self.date_from,
                date_to=self.date_to,
                verified=True if self.verified_only else None,
                favorited=True if self.favorited_only else None,
            ),
        ]


def _iso_date(value: str | date | None) -> date | None:
    if value is None or isinstance(value, date):
        return value
    return date.fromisoformat(value)
