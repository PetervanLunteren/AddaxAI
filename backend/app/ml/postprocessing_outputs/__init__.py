"""Postprocessing output modules for folder runs.

Each module takes a project + a target directory and produces one
kind of user-facing deliverable. The folder-run save endpoint
orchestrates them; nothing here knows about HTTP, jobs, or the queue.

This package is the home for the four legacy-AddaxAI outputs we are
porting:

- `separate_folders`: copy files into target/<label>/ for browsing by
  species in the file manager. SHIPPED.
- `visualised_images`: draw bounding boxes + labels on copies. TODO.
- `blur_people`: blur person / vehicle bounding boxes on copies. TODO.
- `crops`: per-detection crops grouped by label. TODO.

Modules return a typed result dict so the endpoint can surface counts
to the user without needing to know the internals of each output.
"""
