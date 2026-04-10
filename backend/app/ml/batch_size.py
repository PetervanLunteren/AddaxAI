"""Batch size defaults and resolver for ML model subprocesses.

Each subprocess (MegaDetector, classification worker, embedding script)
runs in its own conda env with its own torch/tensorflow install and its
own GPU detection. The backend process does NOT have torch installed and
cannot detect GPU. So:

- Default (project value is NULL): don't pass a batch_size flag to the
  subprocess. Let it use its own built-in default, which already adapts
  to the hardware it finds.
- Custom (project value is an integer): pass it through as an override.
  The subprocess uses it regardless of its own GPU detection.

The constants below are DISPLAY-ONLY: they show up in the Settings UI
label ("Default (4 on GPU, 1 on CPU)") so users know roughly what to
expect when they leave the setting at Default. The actual subprocess
defaults may differ slightly (e.g. embedding auto-selects 32 on MPS),
but these are close enough to be informative.
"""

# Display-only defaults for the SettingsPage Performance card.
# These are NOT used by any resolver — they only populate the
# "Default (X on GPU, Y on CPU)" label in the UI so users know roughly
# what to expect. Must match the actual subprocess defaults:
#
# - Detection: MegaDetector package default = 1 (always, GPU or CPU).
# - Classification: classification_worker.py fallback = 8 on GPU, 1 on CPU.
#   Conservative enough to fit on any GPU (1.1 GB at batch=8 with 480px
#   input), while giving a 2x speedup over the old hardcoded 4.
# - Embedding: embedding_script.py auto-select = 32 on CUDA/MPS, 8 on CPU.
#   At 224px input and batch=32, peak VRAM is ~1.0 GB — safe on any GPU
#   including a base M1 Air.
DETECTION_DEFAULT_GPU = 1
DETECTION_DEFAULT_CPU = 1
CLASSIFICATION_DEFAULT_GPU = 8
CLASSIFICATION_DEFAULT_CPU = 1
EMBEDDING_DEFAULT_GPU = 32
EMBEDDING_DEFAULT_CPU = 8
