"""
Test persistent worker implementation for classification models.

Tests both NAM-ADS-v1 and TAS-BB-v1 with the new worker architecture.

Run from backend directory:
    python test_persistent_worker.py
"""

import json
import time
from pathlib import Path

from PIL import Image

from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.base import BoundingBox
from app.ml.inference.custom_classification_model import CustomClassificationModel
from app.ml.manifest_manager import ManifestManager
from app.ml.model_storage import ModelStorage


def load_expected_results(json_path: Path, model_id: str) -> dict:
    """Load expected results from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract classification results from first detection
    detection = data["images"][0]["detections"][0]
    classifications = detection["classifications"]
    cls_categories = data["classification_categories"]

    expected = {}
    for cls_id, conf in classifications:
        class_name = cls_categories[cls_id]
        expected[class_name] = conf

    return expected


def test_model(model_id: str, test_image: Path, expected_json: Path, expected_bbox: tuple):
    """Test a classification model with persistent worker."""
    print(f"\n{'=' * 80}")
    print(f"Testing {model_id} with Persistent Worker")
    print(f"{'=' * 80}")

    # Load expected results
    print("\n1. Loading expected results...")
    expected = load_expected_results(expected_json, model_id)
    print(f"   ✓ Loaded {len(expected)} expected classifications")

    # Initialize infrastructure
    print("\n2. Initializing ML infrastructure...")
    manifest_manager = ManifestManager()
    env_manager = EnvironmentManager()
    model_storage = ModelStorage()

    manifest = manifest_manager.get_model(model_id)
    model_path = model_storage.get_model_file(manifest)
    model_dir = model_storage.get_model_path(manifest)
    env_name = manifest.env

    print(f"   Model: {model_id}")
    print(f"   Environment: {env_name}")
    print(f"   Model path: {model_path}")

    # Initialize model
    print("\n3. Creating CustomClassificationModel...")
    cls_model = CustomClassificationModel(model_dir, model_path, env_name, env_manager)
    print("   ✓ Model initialized")

    # Load image
    print("\n4. Loading test image...")
    image = Image.open(test_image)
    print(f"   ✓ Image loaded: {image.size}")

    # Create bbox
    bbox = BoundingBox(
        x=expected_bbox[0],
        y=expected_bbox[1],
        width=expected_bbox[2],
        height=expected_bbox[3],
    )

    # Test with context manager (3 classifications)
    print("\n5. Testing with context manager (3 classifications)...")
    start_time = time.time()

    with cls_model:
        print("   Worker started")
        results = []
        for i in range(3):
            result = cls_model.classify(image, bbox)
            if result:
                results.append(result)
                print(f"   Classification {i+1}: {result.species} ({result.confidence:.5f})")
            else:
                print(f"   Classification {i+1}: FAILED (returned None)")
        print("   Worker stopped")

    elapsed = time.time() - start_time
    print(f"   ✓ Completed 3 classifications in {elapsed:.2f}s ({elapsed/3:.2f}s per classification)")

    if not results:
        print("\n   ✗ All classifications failed!")
        return False

    # Validate first result
    print("\n6. Validating results...")
    result = results[0]

    # Check top prediction
    expected_top = max(expected.items(), key=lambda x: x[1])
    if result.species != expected_top[0]:
        print(f"   ✗ Top species mismatch: got {result.species}, expected {expected_top[0]}")
        return False

    if abs(result.confidence - expected_top[1]) >= 0.00001:
        print(
            f"   ✗ Top confidence mismatch: got {result.confidence:.5f}, "
            f"expected {expected_top[1]:.5f}"
        )
        return False

    # Check all probabilities
    mismatches = []
    for class_name, expected_conf in expected.items():
        actual_conf = result.all_probabilities.get(class_name, 0.0)
        diff = abs(actual_conf - expected_conf)
        if diff >= 0.00001:
            mismatches.append((class_name, actual_conf, expected_conf, diff))

    if mismatches:
        print(f"   ✗ Found {len(mismatches)} confidence mismatches:")
        for cls, actual, exp, diff in sorted(mismatches, key=lambda x: x[3], reverse=True)[:5]:
            print(f"      {cls:30s} actual={actual:.5f} expected={exp:.5f} diff={diff:.7f}")
        return False

    print(f"   ✓ All {len(expected)} classifications match exactly!")
    print(f"\n{'=' * 80}")
    print(f"✓ {model_id} TEST PASSED")
    print(f"{'=' * 80}")
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("PERSISTENT WORKER CLASSIFICATION TESTS")
    print("=" * 80)
    print("\nTesting new persistent worker architecture for faster classification.\n")

    # Test NAM-ADS-v1
    nam_pass = test_model(
        model_id="NAM-ADS-v1",
        test_image=Path("/Users/peter/Downloads/test-img/giraffe.jpg"),
        expected_json=Path("/Users/peter/Downloads/test-img/NAM-ADS-v1+REDWOOD.json"),
        expected_bbox=(0.3085, 0.3496, 0.1195, 0.3125),
    )

    # Test TAS-BB-v1
    tas_pass = test_model(
        model_id="TAS-BB-v1",
        test_image=Path("/Users/peter/Downloads/test-img/giraffe.jpg"),
        expected_json=Path("/Users/peter/Downloads/test-img/TAS-BB-v1+REDWOOD.json"),
        expected_bbox=(0.3085, 0.3496, 0.1195, 0.3125),
    )

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"NAM-ADS-v1 (YOLOv8):     {'✓ PASS' if nam_pass else '✗ FAIL'}")
    print(f"TAS-BB-v1 (MEWC Keras): {'✓ PASS' if tas_pass else '✗ FAIL'}")
    print("=" * 80)

    if nam_pass and tas_pass:
        print("\n✓ ALL TESTS PASSED - Persistent worker working correctly!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
