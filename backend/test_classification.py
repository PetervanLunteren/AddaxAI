"""
Test script to verify YOLOv8 classification produces exact same results as streamlit-AddaxAI.

Test case: /Users/peter/Downloads/test-img/giraffe.jpg
Expected: /Users/peter/Downloads/test-img/image_recognition_file.json

Run from backend directory with venv activated:
    source venv/bin/activate && python test_classification.py
"""

import json
import sys
from pathlib import Path

from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.ml.inference.base import BoundingBox
from app.ml.inference.yolov8_classifier import YOLOv8Classifier


def test_classification():
    """Test YOLOv8 classification on giraffe detection"""

    # Setup
    test_image = Path("/Users/peter/Downloads/test-img/giraffe.jpg")
    expected_results_file = Path(
        "/Users/peter/Downloads/test-img/image_recognition_file.json"
    )
    model_path = (
        Path.home() / "AddaxAI/models/cls/NAM-ADS-v1/namib_desert_v1.pt"
    )
    taxonomy_path = Path.home() / "AddaxAI/models/cls/NAM-ADS-v1/taxonomy.csv"

    print(f"Testing classification on: {test_image}")
    print(f"Using model: {model_path}")

    # Load expected results
    with open(expected_results_file) as f:
        expected = json.load(f)

    expected_detection = expected["images"][0]["detections"][0]
    expected_bbox = expected_detection["bbox"]
    expected_classifications = expected_detection["classifications"]

    # Get classification categories mapping
    cls_categories = expected["classification_categories"]

    # Top classification is first in the list
    top_cls_id, top_conf = expected_classifications[0]
    top_species = cls_categories[top_cls_id]

    print(f"\nExpected classification results:")
    print(f"  Top species: {top_species} (ID: {top_cls_id})")
    print(f"  Top confidence: {top_conf}")
    print(f"  Total classifications: {len(expected_classifications)}")

    # Print top 5 expected
    print(f"\n  Top 5 expected classes:")
    for i, (cls_id, conf) in enumerate(expected_classifications[:5], 1):
        species = cls_categories[cls_id]
        print(f"    {i}. {species:20s} {conf:.5f}")

    # Load image and create classifier
    image = Image.open(test_image)
    classifier = YOLOv8Classifier(model_path, taxonomy_path)

    # Create bbox from expected detection
    bbox = BoundingBox(
        x=expected_bbox[0],
        y=expected_bbox[1],
        width=expected_bbox[2],
        height=expected_bbox[3],
    )

    # Run classification
    print(f"\nRunning YOLOv8 classification...")
    result = classifier.classify(image, bbox)

    print(f"\nActual classification results:")
    print(f"  Top species: {result.species}")
    print(f"  Top confidence: {result.confidence}")
    print(f"  Total classes: {len(result.all_probabilities)}")

    # Print top 5 actual
    sorted_probs = sorted(
        result.all_probabilities.items(), key=lambda x: x[1], reverse=True
    )
    print(f"\n  Top 5 actual classes:")
    for i, (species, conf) in enumerate(sorted_probs[:5], 1):
        print(f"    {i}. {species:20s} {conf:.5f}")

    # Verify results match
    tolerance = 0.00001  # Very tight tolerance for float precision

    # Check top species
    species_match = result.species == top_species

    # Check top confidence
    confidence_match = abs(result.confidence - top_conf) < tolerance

    # Check all probabilities match (at least check the top 10)
    all_probs_match = True
    mismatches = []

    for cls_id, expected_conf in expected_classifications[:10]:
        expected_species = cls_categories[cls_id]
        actual_conf = result.all_probabilities.get(expected_species, -1.0)

        if abs(actual_conf - expected_conf) >= tolerance:
            all_probs_match = False
            mismatches.append(
                (expected_species, expected_conf, actual_conf)
            )

    print(f"\n{'='*60}")
    print("Validation Results:")
    print(f"{'='*60}")
    print(
        f"  Top species match: {'✅ PASS' if species_match else '❌ FAIL'}"
    )
    if not species_match:
        print(f"    Expected: {top_species}")
        print(f"    Got:      {result.species}")

    print(
        f"  Top confidence match: {'✅ PASS' if confidence_match else '❌ FAIL'}"
    )
    if not confidence_match:
        print(f"    Expected: {top_conf}")
        print(f"    Got:      {result.confidence}")
        print(f"    Diff:     {abs(result.confidence - top_conf)}")

    print(
        f"  All probabilities match (top 10): {'✅ PASS' if all_probs_match else '❌ FAIL'}"
    )
    if not all_probs_match:
        print(f"    Mismatches found:")
        for species, exp, act in mismatches:
            print(f"      {species:20s} Expected: {exp:.5f}, Got: {act:.5f}")

    all_pass = species_match and confidence_match and all_probs_match

    print(f"\n{'='*60}")
    if all_pass:
        print(
            "✅ ALL TESTS PASSED - Classification results match exactly!"
        )
    else:
        print("❌ TESTS FAILED - Results do not match")
    print(f"{'='*60}\n")

    return all_pass


if __name__ == "__main__":
    success = test_classification()
    sys.exit(0 if success else 1)
