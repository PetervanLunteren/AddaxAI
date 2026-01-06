"""
Test script for custom inference system.

Tests that:
1. CustomInferenceLoader can load inference.py
2. CustomClassificationModel works correctly
3. Results match expected format

Run: python test_custom_inference.py
"""

from pathlib import Path

from PIL import Image

from app.ml.inference.base import BoundingBox
from app.ml.inference.custom_classification_model import CustomClassificationModel

# Test configuration
MODEL_DIR = Path("/Users/peter/AddaxAI/models/cls/NAM-ADS-v1")
MODEL_PATH = MODEL_DIR / "namib_desert_v1.pt"
TEST_IMAGE = Path("/Users/peter/Downloads/test-img/giraffe.jpg")

# Fake bbox covering whole image
BBOX = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)


def test_custom_inference():
    """Test custom inference system end-to-end."""
    print("=" * 80)
    print("Testing Custom Inference System")
    print("=" * 80)

    # Step 1: Check files exist
    print("\n1. Checking files...")
    assert MODEL_DIR.exists(), f"Model directory not found: {MODEL_DIR}"
    assert MODEL_PATH.exists(), f"Model file not found: {MODEL_PATH}"
    assert (MODEL_DIR / "inference.py").exists(), "inference.py not found"
    assert TEST_IMAGE.exists(), f"Test image not found: {TEST_IMAGE}"
    print("   ✓ All files exist")

    # Step 2: Load model
    print("\n2. Loading custom classification model...")
    try:
        model = CustomClassificationModel(MODEL_DIR, MODEL_PATH)
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        raise

    # Step 3: Run classification
    print("\n3. Running classification...")
    try:
        image = Image.open(TEST_IMAGE)
        result = model.classify(image, BBOX)
        print(f"   ✓ Classification complete")
        print(f"   Top prediction: {result.species} ({result.confidence:.3f})")
        print(f"   Total classes: {len(result.all_probabilities)}")
    except Exception as e:
        print(f"   ✗ Classification failed: {e}")
        raise

    # Step 4: Validate result format
    print("\n4. Validating result format...")
    assert result.species, "Species name is empty"
    assert 0 <= result.confidence <= 1, f"Invalid confidence: {result.confidence}"
    assert len(result.all_probabilities) > 0, "No probabilities returned"

    # Check that all_probabilities is a dictionary
    assert isinstance(result.all_probabilities, dict), "all_probabilities should be a dictionary"

    # Check dictionary contains valid data
    for class_name, confidence in result.all_probabilities.items():
        assert isinstance(class_name, str), "Class name should be string"
        assert isinstance(confidence, float), "Confidence should be float"
        assert 0 <= confidence <= 1, f"Invalid confidence for {class_name}: {confidence}"

    # Check top species is in dictionary
    assert result.species in result.all_probabilities, "Top species not in all_probabilities"
    assert abs(result.all_probabilities[result.species] - result.confidence) < 0.0001, \
        "Top confidence mismatch"

    print("   ✓ Result format is valid")

    # Step 5: Display top 5 predictions
    print("\n5. Top 5 predictions:")
    # Sort by confidence descending
    sorted_probs = sorted(result.all_probabilities.items(), key=lambda x: x[1], reverse=True)
    for i, (class_name, confidence) in enumerate(sorted_probs[:5], 1):
        print(f"   {i}. {class_name:30s} {confidence:.5f}")

    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


if __name__ == "__main__":
    test_custom_inference()
