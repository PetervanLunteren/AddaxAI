"""
Standalone smoothing script that runs in the ML environment.

Called as a subprocess by the postprocessing service because
megadetector is only installed in the ML environment, not the backend.

Usage:
    python smoothing_script.py <input_json> <options_json> <output_json>

Where options_json contains:
    {
        "event_smoothing": bool,
        "smoothing_strength": "mild" | "normal" | "aggressive",
        "smoother_input": list[dict] | null
    }

`smoother_input` is the CCT-format list produced by
`app.ml.postprocessing.build_smoother_input`. It is passed through to
MegaDetector's `cct_sequence_information=` parameter verbatim.

Label exclusion and taxonomic rollup are handled upstream before this
script runs. The input JSON already has those transformations applied.
"""

import json
import sys

from megadetector.postprocessing.classification_postprocessing import (
    ClassificationSmoothingOptions,
    smooth_classification_results_image_level,
    smooth_classification_results_sequence_level,
)

# Preset parameter mappings for smoothing strength levels.
# "normal" matches MegaDetector's defaults exactly.
SMOOTHING_PRESETS = {
    "mild": {
        "classification_confidence_threshold": 0.6,
        "min_detections_to_overwrite_other": 3,
        "min_detections_to_overwrite_secondary": 6,
        "max_detections_nondominant_class": 1,
        "min_detections_to_overwrite_secondary_same_family": -1,
        "max_detections_nondominant_class_same_family": -1,
    },
    "normal": {
        "classification_confidence_threshold": 0.5,
        "min_detections_to_overwrite_other": 2,
        "min_detections_to_overwrite_secondary": 4,
        "max_detections_nondominant_class": 1,
        "min_detections_to_overwrite_secondary_same_family": 2,
        "max_detections_nondominant_class_same_family": -1,
    },
    "aggressive": {
        "classification_confidence_threshold": 0.3,
        "min_detections_to_overwrite_other": 1,
        "min_detections_to_overwrite_secondary": 2,
        "max_detections_nondominant_class": 2,
        "min_detections_to_overwrite_secondary_same_family": 2,
        "max_detections_nondominant_class_same_family": 2,
    },
}


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: smoothing_script.py <input_json> <options_json> <output_json>", file=sys.stderr
        )
        sys.exit(1)

    input_path = sys.argv[1]
    options_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(input_path) as f:
        md_results = json.load(f)

    with open(options_path) as f:
        opts = json.load(f)

    event_smoothing = opts.get("event_smoothing", False)
    detection_threshold = opts.get("detection_threshold", 0.15)
    smoothing_strength = opts.get("smoothing_strength", "normal")
    smoother_input = opts.get("smoother_input")

    # Configure smoothing options
    options = ClassificationSmoothingOptions()
    options.propagate_classifications_through_taxonomy = True
    options.detection_confidence_threshold = detection_threshold
    options.detection_category_names_to_smooth = ["animal"]

    # Apply strength preset
    preset = SMOOTHING_PRESETS.get(smoothing_strength, SMOOTHING_PRESETS["normal"])
    for param, value in preset.items():
        setattr(options, param, value)

    # Generic "other" categories that the smoother can overwrite with a dominant
    # real label. Non-label classes (blank, empty, false detection, none) are
    # already stripped by label exclusion before smoothing runs.
    base_other = [
        "other",
        "unknown",
        "no cv result",
        "animal",
        "mammal",
    ]
    options.other_category_names = [name.lower() for name in base_other]

    options.modify_in_place = True

    # Image-level smoothing
    smoothed = smooth_classification_results_image_level(
        input_file=md_results,
        output_file=None,
        options=options,
    )

    # Event-level (aka sequence-level in MegaDetector) smoothing.
    if event_smoothing and smoother_input:
        smoothed = smooth_classification_results_sequence_level(
            input_file=smoothed,
            cct_sequence_information=smoother_input,
            output_file=None,
            options=options,
        )

    with open(output_path, "w") as f:
        json.dump(smoothed, f)


if __name__ == "__main__":
    main()
