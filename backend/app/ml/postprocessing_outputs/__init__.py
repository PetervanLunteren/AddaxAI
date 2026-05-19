"""Postprocessing output modules for the folder-run Save step.

Each module produces one kind of user-facing deliverable. The
folder-run save worker orchestrates them; nothing here knows about
HTTP, jobs, or the queue.

Shipped modules:

- ``separate_folders``: place files into ``<output_root>/<label>/``
  for browsing by species in the file manager.
- ``annotated_copies``: combined per-file pass that can blur people /
  vehicles, draw detection boxes, or both. Writes into the file's
  separated destination(s) when separation also ran, or into
  ``output_root`` directly otherwise.
- ``observations_csv`` / ``observations_xlsx``: flat observation table
  for spreadsheet / R / pandas / QGIS consumption.
- ``recognition_json``: Timelapse-compatible recognition file.
- ``run_readme``: plain-text manifest of the run.

The shared ``OutputContext`` (in ``_output_context.py``) carries
``output_root`` plus the resolved on-disk paths ``separate_folders``
placed each file at, so downstream modules write to the same tree
instead of creating siloed wrapper folders.

EXIF prediction tags are embedded silently by ``separate_folders`` and
``annotated_copies`` on every image they write; there is no
standalone EXIF module.
"""
