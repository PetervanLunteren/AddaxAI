"""
End-to-end test for TAS-BB-v1 MEWC Keras classification model.

Tests that:
1. CustomInferenceLoader can load inference.py for MEWC model
2. CustomClassificationModel works correctly
3. Results match expected output EXACTLY

Run: python test_tas_bb_v1.py
"""

import json
from pathlib import Path

from PIL import Image

from app.ml.inference.base import BoundingBox
from app.ml.inference.custom_classification_model import CustomClassificationModel

# Test configuration
MODEL_DIR = Path("/Users/peter/AddaxAI/models/cls/TAS-BB-v1")
MODEL_PATH = MODEL_DIR / "tas_ens_mewc.keras"
TEST_IMAGE = Path("/Users/peter/Downloads/test-img/giraffe.jpg")
EXPECTED_RESULTS = Path("/Users/peter/Downloads/test-img/TAS-BB-v1+REDWOOD.json")

# Expected detection bbox from the JSON
# "bbox": [0.3085, 0.3496, 0.1195, 0.3125]
EXPECTED_BBOX = BoundingBox(x=0.3085, y=0.3496, width=0.1195, height=0.3125)


def load_expected_results():
    """Load expected results from JSON file."""
    with open(EXPECTED_RESULTS, 'r') as f:
        data = json.load(f)

    # Extract classification results from first detection
    detection = data['images'][0]['detections'][0]
    classifications = detection['classifications']

    # Convert to dict mapping class_name -> confidence
    # The JSON has classification_categories mapping ID -> name
    cls_categories = data['classification_categories']

    expected = {}
    for cls_id, conf in classifications:
        class_name = cls_categories[cls_id]
        expected[class_name] = conf

    return expected


def test_tas_bb_v1():
    """Test TAS-BB-v1 model end-to-end."""
    print("=" * 80)
    print("Testing TAS-BB-v1 MEWC Keras Model")
    print("=" * 80)

    # Step 1: Check files exist
    print("\n1. Checking files...")
    assert MODEL_DIR.exists(), f"Model directory not found: {MODEL_DIR}"
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"
    assert (MODEL_DIR / "inference.py").exists(), "inference.py not found"
    assert (MODEL_DIR / "class_list.yaml").exists(), "class_list.yaml not found"
    assert TEST_IMAGE.exists(), f"Test image not found: {TEST_IMAGE}"
    assert EXPECTED_RESULTS.exists(), f"Expected results not found: {EXPECTED_RESULTS}"
    print("   ✓ All files exist")

    # Step 2: Load expected results
    print("\n2. Loading expected results...")
    expected = load_expected_results()
    print(f"   ✓ Loaded {len(expected)} expected classifications")
    print(f"   Top 3 expected:")
    sorted_expected = sorted(expected.items(), key=lambda x: x[1], reverse=True)
    for i, (cls, conf) in enumerate(sorted_expected[:3], 1):
        print(f"      {i}. {cls:30s} {conf:.5f}")

    # Step 3: Load model
    print("\n3. Loading TAS-BB-v1 model...")
    try:
        model = CustomClassificationModel(MODEL_DIR, MODEL_PATH)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        raise

    # Step 4: Run classification
    print("\n4. Running classification...")
    try:
        image = Image.open(TEST_IMAGE)
        result = model.classify(image, EXPECTED_BBOX)
        print(f"   ✓ Classification complete")
        print(f"   Top prediction: {result.species} ({result.confidence:.5f})")
        print(f"   Total classes: {len(result.all_probabilities)}")
    except Exception as e:
        print(f"   ✗ Classification failed: {e}")
        raise

    # Step 5: Display top predictions
    print("\n5. Top 5 predictions:")
    sorted_results = sorted(result.all_probabilities.items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, confidence) in enumerate(sorted_results[:5], 1):
        expected_conf = expected.get(class_name, 0.0)
        diff = abs(confidence - expected_conf)
        status = "✓" if diff < 0.00001 else "✗"
        print(f"   {status} {i}. {class_name:30s} {confidence:.5f} (expected: {expected_conf:.5f}, diff: {diff:.7f})")

    # Step 6: Validate results match exactly
    print("\n6. Validating results match expected...")

    # Check same number of classes
    assert len(result.all_probabilities) == len(expected), \
        f"Class count mismatch: got {len(result.all_probabilities)}, expected {len(expected)}"

    # Check all classes present
    result_classes = set(result.all_probabilities.keys())
    expected_classes = set(expected.keys())
    missing = expected_classes - result_classes
    extra = result_classes - expected_classes

    if missing:
        print(f"   ✗ Missing classes: {missing}")
        raise AssertionError(f"Missing {len(missing)} classes")

    if extra:
        print(f"   ✗ Extra classes: {extra}")
        raise AssertionError(f"Extra {len(extra)} classes")

    # Check confidence values match
    mismatches = []
    for class_name, expected_conf in expected.items():
        actual_conf = result.all_probabilities[class_name]
        diff = abs(actual_conf - expected_conf)

        # Allow tiny floating point differences (< 0.00001 = 1e-5)
        if diff >= 0.00001:
            mismatches.append((class_name, actual_conf, expected_conf, diff))

    if mismatches:
        print(f"   ✗ Found {len(mismatches)} confidence mismatches:")
        for cls, actual, expected_val, diff in sorted(mismatches, key=lambda x: x[3], reverse=True)[:10]:
            print(f"      {cls:30s} actual={actual:.5f} expected={expected_val:.5f} diff={diff:.7f}")
        raise AssertionError(f"{len(mismatches)} confidence values don't match exactly")

    print(f"   ✓ All {len(expected)} classifications match exactly!")

    print("\n" + "=" * 80)
    print("✓ All tests passed - Results match EXACTLY!")
    print("=" * 80)


if __name__ == "__main__":
    test_tas_bb_v1()
