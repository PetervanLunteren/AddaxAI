"""
Test script to verify MegaDetector produces exact same results as streamlit-AddaxAI.

Test case: /Users/peter/Downloads/test-img/giraffe.jpg
Expected: /Users/peter/Downloads/test-img/image_recognition_file.json

Run from backend directory:
    python test_detection.py
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.megadetector import MegaDetectorV1000


def test_detection():
    """Test MegaDetector on giraffe.jpg"""

    # Setup
    test_image = Path("/Users/peter/Downloads/test-img/giraffe.jpg")
    expected_results_file = Path("/Users/peter/Downloads/test-img/image_recognition_file.json")
    model_path = Path.home() / "AddaxAI/models/det/MD1000-REDWOOD-0-0/md_v1000.0.0-redwood.pt"

    print(f"Testing detection on: {test_image}")
    print(f"Using model: {model_path}")

    # Load expected results
    with open(expected_results_file) as f:
        expected = json.load(f)

    expected_detection = expected["images"][0]["detections"][0]
    expected_bbox = expected_detection["bbox"]
    expected_conf = expected_detection["prev_conf"]  # Detection confidence (before classification)
    expected_category = expected_detection["prev_category"]  # "1" = animal

    print(f"\nExpected detection results:")
    print(f"  Category: {expected_category} -> animal")
    print(f"  Confidence: {expected_conf}")
    print(f"  Bbox: {expected_bbox}")

    # Run detection
    env_manager = EnvironmentManager()
    detector = MegaDetectorV1000(model_path, env_manager)

    def progress(msg, pct):
        print(f"  [{pct*100:.0f}%] {msg}")

    detections = detector.detect(
        image_paths=[test_image],
        confidence_threshold=0.1,
        progress_callback=progress,
    )

    print(f"\nGot {len(detections)} detection(s)")

    if len(detections) == 0:
        print("❌ FAILED: No detections found!")
        return False

    detection = detections[0]
    print(f"\nActual detection results:")
    print(f"  Category: {detection.category}")
    print(f"  Confidence: {detection.confidence}")
    print(f"  Bbox: [{detection.bbox.x}, {detection.bbox.y}, {detection.bbox.width}, {detection.bbox.height}]")

    # Verify results match
    tolerance = 0.0001  # Allow tiny float precision differences

    bbox_match = (
        abs(detection.bbox.x - expected_bbox[0]) < tolerance
        and abs(detection.bbox.y - expected_bbox[1]) < tolerance
        and abs(detection.bbox.width - expected_bbox[2]) < tolerance
        and abs(detection.bbox.height - expected_bbox[3]) < tolerance
    )

    confidence_match = abs(detection.confidence - expected_conf) < tolerance

    category_match = detection.category == "animal"

    print(f"\n{'='*60}")
    print("Validation Results:")
    print(f"{'='*60}")
    print(f"  Bbox match: {'✅ PASS' if bbox_match else '❌ FAIL'}")
    if not bbox_match:
        print(f"    Expected: {expected_bbox}")
        print(f"    Got:      [{detection.bbox.x}, {detection.bbox.y}, {detection.bbox.width}, {detection.bbox.height}]")

    print(f"  Confidence match: {'✅ PASS' if confidence_match else '❌ FAIL'}")
    if not confidence_match:
        print(f"    Expected: {expected_conf}")
        print(f"    Got:      {detection.confidence}")

    print(f"  Category match: {'✅ PASS' if category_match else '❌ FAIL'}")
    if not category_match:
        print(f"    Expected: animal")
        print(f"    Got:      {detection.category}")

    all_pass = bbox_match and confidence_match and category_match

    print(f"\n{'='*60}")
    if all_pass:
        print("✅ ALL TESTS PASSED - Detection results match exactly!")
    else:
        print("❌ TESTS FAILED - Results do not match")
    print(f"{'='*60}\n")

    return all_pass


if __name__ == "__main__":
    success = test_detection()
    sys.exit(0 if success else 1)
