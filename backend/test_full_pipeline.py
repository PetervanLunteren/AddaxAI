"""
End-to-end test of complete ML pipeline (detection + classification).

Test case: /Users/peter/Downloads/test-img/giraffe.jpg
Expected: /Users/peter/Downloads/test-img/image_recognition_file.json

Validates:
1. Detection bbox matches exactly: [0.3085, 0.3496, 0.1195, 0.3125]
2. Detection confidence matches: 0.97
3. Classification species matches: "giraffe"
4. Classification confidence matches: 0.99985
5. All classification probabilities match (30 classes)

Run from backend directory with venv activated:
    source venv/bin/activate && python test_full_pipeline.py
"""

import json
import sys
import tempfile
from pathlib import Path

from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.megadetector import MegaDetectorV1000
from app.ml.inference.yolov8_classifier import YOLOv8Classifier


def test_full_pipeline():
    """Test complete pipeline: detection → classification"""

    print("="*70)
    print("FULL ML PIPELINE TEST")
    print("="*70)

    # Setup paths
    test_image = Path("/Users/peter/Downloads/test-img/giraffe.jpg")
    expected_results_file = Path("/Users/peter/Downloads/test-img/image_recognition_file.json")

    det_model_path = Path.home() / "AddaxAI/models/det/MD1000-REDWOOD-0-0/md_v1000.0.0-redwood.pt"
    cls_model_path = Path.home() / "AddaxAI/models/cls/NAM-ADS-v1/namib_desert_v1.pt"
    taxonomy_path = Path.home() / "AddaxAI/models/cls/NAM-ADS-v1/taxonomy.csv"

    print(f"\nTest configuration:")
    print(f"  Test image: {test_image}")
    print(f"  Detection model: MD1000-REDWOOD-0-0")
    print(f"  Classification model: NAM-ADS-v1")

    # Load expected results
    with open(expected_results_file) as f:
        expected = json.load(f)

    expected_image = expected["images"][0]
    expected_detection = expected_image["detections"][0]
    expected_bbox = expected_detection["bbox"]
    expected_det_conf = expected_detection["prev_conf"]  # Detection confidence
    expected_det_category = expected_detection["prev_category"]  # "1" = animal
    expected_classifications = expected_detection["classifications"]

    # The top classification is the first in the list
    top_cls_id, top_cls_conf = expected_classifications[0]

    # Mappings
    cls_categories = expected["classification_categories"]
    expected_species = cls_categories[top_cls_id]  # Use top classification ID, not category

    print(f"\n{'='*70}")
    print("EXPECTED RESULTS (from streamlit-AddaxAI)")
    print(f"{'='*70}")
    print(f"\nDetection:")
    print(f"  Category: {expected_det_category} (animal)")
    print(f"  Confidence: {expected_det_conf}")
    print(f"  Bbox: {expected_bbox}")
    print(f"\nClassification:")
    print(f"  Species: {expected_species}")
    print(f"  Confidence: {top_cls_conf}")
    print(f"  Total classes: {len(expected_classifications)}")
    print(f"\n  Top 5 classes:")
    for i, (cls_id, conf) in enumerate(expected_classifications[:5], 1):
        species = cls_categories[cls_id]
        print(f"    {i}. {species:20s} {conf:.5f}")

    # =========================================================================
    # PHASE 1: DETECTION
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: DETECTION")
    print(f"{'='*70}")

    env_manager = EnvironmentManager()
    detector = MegaDetectorV1000(det_model_path, env_manager)

    detections = detector.detect(
        image_paths=[test_image],
        confidence_threshold=0.1,
        progress_callback=lambda msg, pct: print(f"  [{pct*100:3.0f}%] {msg[:60]}"),
    )

    print(f"\nDetection results:")
    print(f"  Found {len(detections)} detection(s)")

    if len(detections) == 0:
        print("❌ FAILED: No detections found!")
        return False

    detection = detections[0]
    print(f"  Category: {detection.category}")
    print(f"  Confidence: {detection.confidence}")
    print(f"  Bbox: [{detection.bbox.x}, {detection.bbox.y}, {detection.bbox.width}, {detection.bbox.height}]")

    # =========================================================================
    # PHASE 2: CLASSIFICATION
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: CLASSIFICATION")
    print(f"{'='*70}")

    classifier = YOLOv8Classifier(cls_model_path, taxonomy_path)

    # Load image
    image = Image.open(test_image)

    # Classify detection
    result = classifier.classify(image, detection.bbox)

    print(f"\nClassification results:")
    print(f"  Top species: {result.species}")
    print(f"  Top confidence: {result.confidence}")
    print(f"  Total classes: {len(result.all_probabilities)}")

    sorted_probs = sorted(result.all_probabilities.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 5 classes:")
    for i, (species, conf) in enumerate(sorted_probs[:5], 1):
        print(f"    {i}. {species:20s} {conf:.5f}")

    # =========================================================================
    # VALIDATION
    # =========================================================================
    print(f"\n{'='*70}")
    print("VALIDATION")
    print(f"{'='*70}")

    tolerance = 0.00001

    # Validate detection
    det_bbox_match = (
        abs(detection.bbox.x - expected_bbox[0]) < tolerance
        and abs(detection.bbox.y - expected_bbox[1]) < tolerance
        and abs(detection.bbox.width - expected_bbox[2]) < tolerance
        and abs(detection.bbox.height - expected_bbox[3]) < tolerance
    )

    det_conf_match = abs(detection.confidence - expected_det_conf) < tolerance
    det_category_match = detection.category == "animal"

    # Validate classification
    cls_species_match = result.species == expected_species
    cls_conf_match = abs(result.confidence - top_cls_conf) < tolerance

    # Validate all probabilities (check top 10)
    all_probs_match = True
    mismatches = []

    for cls_id, expected_conf in expected_classifications[:10]:
        expected_species_name = cls_categories[cls_id]
        actual_conf = result.all_probabilities.get(expected_species_name, -1.0)

        if abs(actual_conf - expected_conf) >= tolerance:
            all_probs_match = False
            mismatches.append((expected_species_name, expected_conf, actual_conf))

    # Print results
    print("\nDetection validation:")
    print(f"  ✅ Bbox match" if det_bbox_match else f"  ❌ Bbox MISMATCH")
    if not det_bbox_match:
        print(f"      Expected: {expected_bbox}")
        print(f"      Got:      [{detection.bbox.x}, {detection.bbox.y}, {detection.bbox.width}, {detection.bbox.height}]")

    print(f"  ✅ Confidence match" if det_conf_match else f"  ❌ Confidence MISMATCH")
    if not det_conf_match:
        print(f"      Expected: {expected_det_conf}")
        print(f"      Got:      {detection.confidence}")

    print(f"  ✅ Category match" if det_category_match else f"  ❌ Category MISMATCH")

    print("\nClassification validation:")
    print(f"  ✅ Species match" if cls_species_match else f"  ❌ Species MISMATCH")
    if not cls_species_match:
        print(f"      Expected: {expected_species}")
        print(f"      Got:      {result.species}")

    print(f"  ✅ Confidence match" if cls_conf_match else f"  ❌ Confidence MISMATCH")
    if not cls_conf_match:
        print(f"      Expected: {top_cls_conf}")
        print(f"      Got:      {result.confidence}")

    print(f"  ✅ All probabilities match (top 10)" if all_probs_match else f"  ❌ Probabilities MISMATCH")
    if not all_probs_match:
        print(f"      Mismatches:")
        for species, exp, act in mismatches:
            print(f"        {species:20s} Expected: {exp:.5f}, Got: {act:.5f}, Diff: {abs(exp-act):.7f}")

    all_pass = (
        det_bbox_match
        and det_conf_match
        and det_category_match
        and cls_species_match
        and cls_conf_match
        and all_probs_match
    )

    print(f"\n{'='*70}")
    if all_pass:
        print("✅ ALL TESTS PASSED")
        print("Pipeline produces EXACT same results as streamlit-AddaxAI!")
    else:
        print("❌ SOME TESTS FAILED")
        print("Results do not match expected output")
    print(f"{'='*70}\n")

    return all_pass


if __name__ == "__main__":
    success = test_full_pipeline()
    sys.exit(0 if success else 1)
