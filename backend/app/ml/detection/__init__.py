"""Detection module for running ML detection models."""

from .megadetector_runner import MegaDetectorRunner

# Single source of truth for the detection confidence floor.
#
# Anything below this threshold is dropped at the source by MegaDetector
# and never reaches results.json, the database, or the smoother. The
# detection worker (`app/workers/detection_worker.py`) passes this value
# to detect_to_json and to the postprocessing smoother.
#
# The user-visible "Detection threshold" in project settings is a
# *display* filter that hides detections below a user-chosen value
# (typically 0.5); it does not replace this floor.
#
# To change the global floor, edit this constant; every call site
# follows automatically. Don't reintroduce literal 0.1 in call sites.
DETECTION_CONFIDENCE_FLOOR = 0.1

__all__ = ["DETECTION_CONFIDENCE_FLOOR", "MegaDetectorRunner"]
