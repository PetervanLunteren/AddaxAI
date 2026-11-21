"""
Standalone taxa list script that runs in the ML environment.

Called as a subprocess by the postprocessing service because
megadetector is only installed in the ML environment, not the backend.

Usage:
    python taxa_mapping_script.py <input_json> <options_json> <output_json>

Where options_json contains:
    {
        "taxa_list": str,       # .csv written by taxa_mapping.write_md_taxa_list
        "taxonomy_file": str    # the model's taxonomy_release.*.txt
    }
"""

import json
import sys

from megadetector.postprocessing.classification_postprocessing import (
    restrict_to_taxa_list,
)


def main() -> None:
    if len(sys.argv) != 4:
        print(
            "Usage: taxa_mapping_script.py <input_json> <options_json> <output_json>",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path, options_path, output_path = sys.argv[1:]
    with open(options_path) as f:
        opts = json.load(f)

    # "human" is protected because MegaDetector recognises its own human
    # category by the full SpeciesNet string, GUID included, and AddaxAI's
    # descriptions carry the label in that first token instead. Without
    # this, a list with no primate row would turn every person into
    # "animal".
    restrict_to_taxa_list(
        taxa_list=opts["taxa_list"],
        speciesnet_taxonomy_file=opts["taxonomy_file"],
        input_file=input_path,
        output_file=output_path,
        protected_common_names=["human"],
        add_pre_filtering_description=False,
        add_post_filtering_description=False,
        verbose=False,
    )


if __name__ == "__main__":
    main()
