"""The detector class a species belongs to.

A box drawn by hand, a species added by hand on the Counts page, and a
person or vehicle box relabelled to a species all need a detector
category that no detector gave them. That category is the detector's
`primary_class` from the catalog ("animal", "fish", "elasmobranch"),
read here and nowhere else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from app.ml.label_exclusion import is_non_label, is_wildlife
from app.ml.manifest_manager import ManifestManager
from app.ml.schemas.model_manifest import ModelManifest

if TYPE_CHECKING:
    from app.models import Project


def primary_class_of(manifest: ModelManifest) -> str:
    """The detector's primary class. A detector without one, or with one
    it does not list in `classes`, is a catalog error, reported by name
    rather than guessed."""
    value = manifest.primary_class
    if not value or value not in (manifest.classes or []):
        raise ValueError(
            f"detector {manifest.model_id} declares primary_class {value!r} in the "
            f"catalog; expected one of its classes {manifest.classes!r}"
        )
    return value


def project_primary_class(project: Project) -> str:
    """The primary class of the project's detector. Raises ValueError
    naming the model when it is not installed or declares none."""
    manifest = ManifestManager().get_model(project.detection_model_id)
    return primary_class_of(manifest)


def category_after_human_edit(
    current: str | None,
    sent: str | None,
    label: str | None,
    primary: Callable[[], str],
) -> str:
    """The detector category a box or count row holds after a person set
    its label: the one rule for drawing, relabelling and adding by hand.

    - A species keeps a wildlife box's own category: it says what the
      animal is, not what the detector called it. A person or vehicle box,
      or a new one, takes the detector's primary class. A category the
      client sent beside a species is ignored (saved hotkeys still carry
      the old "animal").
    - No species (a detector class picked, a label cleared, a non-label
      such as "false detection"): the class picked, else the box's own,
      else the primary class.

    ``primary`` is called only when needed, because it reads the catalog.
    """
    if label and not is_non_label(label):
        if current is not None and is_wildlife(current):
            return current
        return primary()
    return sent or current or primary()
