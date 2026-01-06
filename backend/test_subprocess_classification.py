"""
Test subprocess-based classification for both NAM-ADS-v1 and TAS-BB-v1.

This test validates that:
1. classification_runner.py works in isolated environments
2. CustomClassificationModel correctly uses subprocess execution
3. Results match expected values exactly

Run from backend directory:
    python test_subprocess_classification.py
"""

import json
from pathlib import Path

from PIL import Image

from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.base import BoundingBox
from app.ml.inference.custom_classification_model import CustomClassificationModel
from app.ml.manifest_manager import ManifestManager
from app.ml.model_storage import ModelStorage


def load_expected_results_nam(json_path: Path) -> dict:
    """Load expected results for NAM-ADS-v1 from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract first detection's classifications
    detection = data["images"][0]["detections"][0]
    classifications = detection["classifications"]
    cls_categories = data["classification_categories"]

    # Build dict: class_name -> confidence
    expected = {}
    for cls_id, conf in classifications:
        class_name = cls_categories[cls_id]
        expected[class_name] = conf

    return expected


def load_expected_results_tas(json_path: Path) -> dict:
    """Load expected results for TAS-BB-v1 from JSON."""
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract first detection's classifications
    detection = data["images"][0]["detections"][0]
    classifications = detection["classifications"]
    cls_categories = data["classification_categories"]

    # Build dict: class_name -> confidence
    expected = {}
    for cls_id, conf in classifications:
        class_name = cls_categories[cls_id]
        expected[class_name] = conf

    return expected


def test_nam_ads_v1():
    """Test NAM-ADS-v1 with subprocess isolation."""
    print("=" * 80)
    print("Testing NAM-ADS-v1 (YOLOv8) with Subprocess Isolation")
    print("=" * 80)

    model_id = "NAM-ADS-v1"
    test_image = Path("/Users/peter/Downloads/test-img/giraffe.jpg")
    expected_json = Path("/Users/peter/Downloads/test-img/NAM-ADS-v1+REDWOOD.json")
    expected_bbox = (0.3085, 0.3496, 0.1195, 0.3125)

    # Step 1: Load expected results
    print("\n1. Loading expected results...")
    expected = load_expected_results_nam(expected_json)
    print(f"   ✓ Loaded {len(expected)} expected classifications")

    # Step 2: Initialize infrastructure
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

    # Step 3: Initialize CustomClassificationModel
    print("\n3. Initializing CustomClassificationModel with subprocess...")
    cls_model = CustomClassificationModel(model_dir, model_path, env_name, env_manager)
    print("   ✓ Model initialized")

    # Step 4: Load image
    print("\n4. Loading test image...")
    image = Image.open(test_image)
    print(f"   ✓ Image loaded: {image.size}")

    # Step 5: Create BoundingBox
    bbox = BoundingBox(
        x=expected_bbox[0],
        y=expected_bbox[1],
        width=expected_bbox[2],
        height=expected_bbox[3],
    )

    # Step 6: Run classification
    print("\n5. Running classification via subprocess...")
    result = cls_model.classify(image, bbox)
    print(f"   ✓ Classification complete")
    print(f"   Top prediction: {result.species} ({result.confidence:.5f})")

    # Step 7: Validate results
    print("\n6. Validating results...")

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
    print("\n" + "=" * 80)
    print("✓ NAM-ADS-v1 TEST PASSED")
    print("=" * 80)
    return True


def test_tas_bb_v1():
    """Test TAS-BB-v1 with subprocess isolation."""
    print("\n" + "=" * 80)
    print("Testing TAS-BB-v1 (MEWC Keras) with Subprocess Isolation")
    print("=" * 80)

    model_id = "TAS-BB-v1"
    test_image = Path("/Users/peter/Downloads/test-img/giraffe.jpg")
    expected_json = Path("/Users/peter/Downloads/test-img/TAS-BB-v1+REDWOOD.json")
    expected_bbox = (0.3085, 0.3496, 0.1195, 0.3125)

    # Step 1: Load expected results
    print("\n1. Loading expected results...")
    expected = load_expected_results_tas(expected_json)
    print(f"   ✓ Loaded {len(expected)} expected classifications")

    # Step 2: Initialize infrastructure
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

    # Step 3: Initialize CustomClassificationModel
    print("\n3. Initializing CustomClassificationModel with subprocess...")
    cls_model = CustomClassificationModel(model_dir, model_path, env_name, env_manager)
    print("   ✓ Model initialized")

    # Step 4: Load image
    print("\n4. Loading test image...")
    image = Image.open(test_image)
    print(f"   ✓ Image loaded: {image.size}")

    # Step 5: Create BoundingBox
    bbox = BoundingBox(
        x=expected_bbox[0],
        y=expected_bbox[1],
        width=expected_bbox[2],
        height=expected_bbox[3],
    )

    # Step 6: Run classification
    print("\n5. Running classification via subprocess...")
    result = cls_model.classify(image, bbox)
    print(f"   ✓ Classification complete")
    print(f"   Top prediction: {result.species} ({result.confidence:.5f})")

    # Step 7: Validate results
    print("\n6. Validating results...")

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
    print("\n" + "=" * 80)
    print("✓ TAS-BB-v1 TEST PASSED")
    print("=" * 80)
    return True


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("SUBPROCESS CLASSIFICATION TESTS")
    print("=" * 80)
    print("\nThis test validates proper environment isolation for classification models.")
    print("Each model runs in its designated environment via subprocess.\n")

    # Test NAM-ADS-v1
    nam_pass = test_nam_ads_v1()

    # Test TAS-BB-v1
    tas_pass = test_tas_bb_v1()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"NAM-ADS-v1 (YOLOv8):     {'✓ PASS' if nam_pass else '✗ FAIL'}")
    print(f"TAS-BB-v1 (MEWC Keras): {'✓ PASS' if tas_pass else '✗ FAIL'}")
    print("=" * 80)

    if nam_pass and tas_pass:
        print("\n✓ ALL TESTS PASSED - Environment isolation working correctly!")
        return 0
    else:
        print("\n✗ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    exit(main())
