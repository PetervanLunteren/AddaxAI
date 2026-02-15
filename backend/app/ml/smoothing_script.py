"""
Standalone smoothing script that runs in the ML environment.

Called as a subprocess by the postprocessing service because
megadetector is only installed in the ML environment, not the backend.

Usage:
    python smoothing_script.py <input_json> <options_json> <output_json>

Where options_json contains:
    {
        "taxonomic_rollup": bool,
        "event_smoothing": bool,
        "excluded_classes": list[str],
        "sequence_info": list[dict] | null
    }
"""

import json
import sys

from megadetector.postprocessing.classification_postprocessing import (
    ClassificationSmoothingOptions,
    smooth_classification_results_image_level,
    smooth_classification_results_sequence_level,
)


def main() -> None:
    if len(sys.argv) != 4:
        print("Usage: smoothing_script.py <input_json> <options_json> <output_json>", file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    options_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(input_path) as f:
        md_results = json.load(f)

    with open(options_path) as f:
        opts = json.load(f)

    taxonomic_rollup = opts.get("taxonomic_rollup", False)
    event_smoothing = opts.get("event_smoothing", False)
    excluded_classes = opts.get("excluded_classes", [])
    detection_threshold = opts.get("detection_threshold", 0.15)
    sequence_info = opts.get("sequence_info")

    # Configure smoothing options
    options = ClassificationSmoothingOptions()
    options.propagate_classifications_through_taxonomy = taxonomic_rollup
    options.detection_confidence_threshold = detection_threshold
    options.detection_category_names_to_smooth = ["animal"]

    base_other = ["other", "unknown", "no cv result", "animal", "blank", "mammal"]
    excluded = [c.lower() for c in excluded_classes]
    options.other_category_names = list(set(base_other + excluded))

    options.modify_in_place = True

    # Image-level smoothing
    smoothed = smooth_classification_results_image_level(
        input_file=md_results,
        output_file=None,
        options=options,
    )

    # Sequence-level smoothing (if event_smoothing and sequence info provided)
    if event_smoothing and sequence_info:
        smoothed = smooth_classification_results_sequence_level(
            input_file=smoothed,
            cct_sequence_information=sequence_info,
            output_file=None,
            options=options,
        )

    with open(output_path, "w") as f:
        json.dump(smoothed, f)


if __name__ == "__main__":
    main()
