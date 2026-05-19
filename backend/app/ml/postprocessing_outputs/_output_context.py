"""Shared state passed between the folder-run Save modules.

``separate_folders`` populates ``OutputContext.resolved_paths`` as it
places each file. Downstream modules (``annotated_copies``,
``observations_csv``, ``observations_xlsx``) consult the same context
to discover where each source file ended up, instead of writing into
siloed wrapper folders next to the separated tree.

When the user did not enable separation, ``resolved_paths`` stays
empty; downstream modules allocate fresh destinations under
``output_root`` themselves. The context is intentionally a dumb
record-keeping object — name allocation and collision handling stay
with the modules that need them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OutputContext:
    output_root: Path
    # file_id -> ordered list of on-disk paths where ``separate_folders``
    # placed the file. Multiple entries only for multi-species files
    # that landed in more than one label folder.
    resolved_paths: dict[str, list[Path]] = field(default_factory=dict)

    def record(self, file_id: str, path: Path) -> None:
        """Append one destination path for ``file_id``."""
        self.resolved_paths.setdefault(file_id, []).append(path)

    def resolved_for(self, file_id: str) -> list[Path] | None:
        """Destinations separation placed the file at, or ``None`` when
        separation did not place it (skipped, or never ran). Callers
        treat the ``None`` case as "allocate a fresh destination under
        ``output_root``"."""
        paths = self.resolved_paths.get(file_id)
        return list(paths) if paths else None
