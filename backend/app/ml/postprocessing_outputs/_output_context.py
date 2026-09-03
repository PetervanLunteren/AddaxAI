"""Shared state passed between the folder-run media Save modules.

``separate_folders`` populates ``OutputContext.resolved_paths`` as it
places each file. ``annotated_copies`` consults the same context to
discover where each source file ended up, instead of writing into
siloed wrapper folders next to the separated tree.

The context belongs to the media modules only. The worker points
``output_root`` at the ``addaxai-media`` subfolder of the user's
output dir; the loose data exports (CSV / XLSX / recognition JSON /
summary) take their target dir directly and never see this context.

When the user did not enable separation, ``resolved_paths`` stays
empty; downstream modules allocate fresh destinations under
``output_root`` themselves. The context is intentionally a dumb
record-keeping object — name allocation and collision handling stay
with the modules that need them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Subfolder of the user's output dir that receives all media copies
# (separated folders, annotated / anonymised images). Keeping media in
# one prefixed subfolder is what lets the output dir default to the
# source folder itself: originals are never overwritten, and the
# scan-skip marker goes on this subfolder only, never on the source
# root. The loose data exports share the prefix ("addaxai-…") so all
# run outputs sort together between the user's own files.
MEDIA_SUBDIR = "addaxai-media"


@dataclass
class OutputContext:
    output_root: Path
    # file_id -> ordered list of on-disk paths where ``separate_folders``
    # placed the file. Multiple entries only for multi-species files
    # that landed in more than one label folder.
    resolved_paths: dict[str, list[Path]] = field(default_factory=dict)
    # file_id -> the name allocated for a video's annotated still beside
    # its placed container. Allocated by ``separate_folders`` together
    # with every other name it hands out, so it cannot collide with a
    # photo or a second clip; read by ``annotated_copies``. Absent for
    # images and for a video placed as its still (blur mode).
    still_paths: dict[str, Path] = field(default_factory=dict)

    def record(self, file_id: str, path: Path) -> None:
        """Append one destination path for ``file_id``."""
        self.resolved_paths.setdefault(file_id, []).append(path)

    def record_still(self, file_id: str, path: Path) -> None:
        """Remember where ``file_id``'s annotated still may be written."""
        self.still_paths[file_id] = path

    def still_for(self, file_id: str) -> Path | None:
        """The still beside a placed container, or ``None``."""
        return self.still_paths.get(file_id)

    def resolved_for(self, file_id: str) -> list[Path] | None:
        """Destinations separation placed the file at, or ``None`` when
        separation did not place it (skipped, or never ran). Callers
        treat the ``None`` case as "allocate a fresh destination under
        ``output_root``"."""
        paths = self.resolved_paths.get(file_id)
        return list(paths) if paths else None
