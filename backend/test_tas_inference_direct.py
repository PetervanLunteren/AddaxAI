"""
Direct test of TAS-BB-v1 inference.py script (without backend dependencies).

This test loads the inference script directly to verify it works in the
tensorflow environment.

Run with tensorflow environment:
/Users/peter/AddaxAI/bin/micromamba run -p /Users/peter/AddaxAI/envs/env-tensorflow-v2 python test_tas_inference_direct.py
"""

import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image

# Test configuration
MODEL_DIR = Path("/Users/peter/AddaxAI/models/cls/TAS-BB-v1")
MODEL_PATH = MODEL_DIR / "tas_ens_mewc.keras"
TEST_IMAGE = Path("/Users/peter/Downloads/test-img/giraffe.jpg")
EXPECTED_RESULTS = Path("/Users/peter/Downloads/test-img/TAS-BB-v1+REDWOOD.json")
INFERENCE_SCRIPT = MODEL_DIR / "inference.py"

# Expected detection bbox from the JSON
# "bbox": [0.3085, 0.3496, 0.1195, 0.3125]
EXPECTED_BBOX = (0.3085, 0.3496, 0.1195, 0.3125)


def load_inference_module():
    """Dynamically load the inference.py module."""
    spec = importlib.util.spec_from_file_location("tas_inference", INFERENCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)

    # Inject required variables
    module.MODEL_DIR = MODEL_DIR
    module.MODEL_PATH = MODEL_PATH

    # Add to sys.modules
    sys.modules["tas_inference"] = module

    # Execute module
    spec.loader.exec_module(module)

    return module


def load_expected_results():
    """Load expected results from JSON file."""
    with open(EXPECTED_RESULTS, 'r') as f:
        data = json.load(f)

    # Extract classification results from first detection
    detection = data['images'][0]['detections'][0]
    classifications = detection['classifications']

    # Convert to dict mapping class_name -> confidence
    cls_categories = data['classification_categories']

    expected = {}
    for cls_id, conf in classifications:
        class_name = cls_categories[cls_id]
        expected[class_name] = conf

    return expected


def test_inference_direct():
    """Test inference script directly."""
    print("=" * 80)
    print("Testing TAS-BB-v1 Inference Script Directly")
    print("=" * 80)

    # Step 1: Check files
    print("\n1. Checking files...")
    assert MODEL_DIR.exists()
    assert MODEL_PATH.exists()
    assert INFERENCE_SCRIPT.exists()
    assert TEST_IMAGE.exists()
    assert EXPECTED_RESULTS.exists()
    print("   ✓ All files exist")

    # Step 2: Load expected results
    print("\n2. Loading expected results...")
    expected = load_expected_results()
    print(f"   ✓ Loaded {len(expected)} expected classifications")
    sorted_expected = sorted(expected.items(), key=lambda x: x[1], reverse=True)
    print(f"   Top 3 expected:")
    for i, (cls, conf) in enumerate(sorted_expected[:3], 1):
        print(f"      {i}. {cls:30s} {conf:.5f}")

    # Step 3: Load inference module
    print("\n3. Loading inference module...")
    try:
        inf = load_inference_module()
        print("   ✓ Module loaded")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        raise

    # Step 4: Check GPU
    print("\n4. Checking GPU...")
    gpu_available = inf.check_gpu()
    print(f"   GPU available: {gpu_available}")

    # Step 5: Load model
    print("\n5. Loading model...")
    try:
        inf.load_model()
        print("   ✓ Model loaded")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        raise

    # Step 6: Get crop
    print("\n6. Cropping image...")
    try:
        image = Image.open(TEST_IMAGE)
        crop = inf.get_crop(image, EXPECTED_BBOX)
        print(f"   ✓ Crop size: {crop.size}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        raise

    # Step 7: Run classification
    print("\n7. Running classification...")
    try:
        results = inf.get_classification(crop)
        print(f"   ✓ Got {len(results)} classifications")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        raise

    # Step 8: Display top predictions
    print("\n8. Top 5 predictions:")
    for i, (class_name, confidence) in enumerate(results[:5], 1):
        expected_conf = expected.get(class_name, 0.0)
        diff = abs(confidence - expected_conf)
        status = "✓" if diff < 0.00001 else "✗"
        print(f"   {status} {i}. {class_name:30s} {confidence:.5f} (expected: {expected_conf:.5f}, diff: {diff:.7f})")

    # Step 9: Validate exact match
    print("\n9. Validating exact match...")

    # Convert results to dict
    result_dict = {name: conf for name, conf in results}

    # Check class count
    assert len(result_dict) == len(expected), \
        f"Class count mismatch: got {len(result_dict)}, expected {len(expected)}"

    # Check all classes present
    result_classes = set(result_dict.keys())
    expected_classes = set(expected.keys())
    missing = expected_classes - result_classes
    extra = result_classes - expected_classes

    if missing:
        print(f"   ✗ Missing classes: {missing}")
        raise AssertionError(f"Missing {len(missing)} classes")

    if extra:
        print(f"   ✗ Extra classes: {extra}")
        raise AssertionError(f"Extra {len(extra)} classes")

    # Check confidence values
    mismatches = []
    for class_name, expected_conf in expected.items():
        actual_conf = result_dict[class_name]
        diff = abs(actual_conf - expected_conf)

        # Allow tiny floating point differences (< 0.00001)
        if diff >= 0.00001:
            mismatches.append((class_name, actual_conf, expected_conf, diff))

    if mismatches:
        print(f"   ✗ Found {len(mismatches)} confidence mismatches:")
        for cls, actual, expected_val, diff in sorted(mismatches, key=lambda x: x[3], reverse=True)[:10]:
            print(f"      {cls:30s} actual={actual:.5f} expected={expected_val:.5f} diff={diff:.7f}")
        raise AssertionError(f"{len(mismatches)} values don't match exactly")

    print(f"   ✓ All {len(expected)} classifications match exactly!")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED - Results match EXACTLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_inference_direct()
