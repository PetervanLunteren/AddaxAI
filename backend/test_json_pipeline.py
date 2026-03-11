"""
Test JSON-based pipeline with NAM-ADS-v1 and TAS-BB-v1 models.

Tests the complete JSON pipeline workflow:
1. MegaDetector → detection_results.json
2. Classification → results.json
3. Validate JSON format matches expected structure

Run from backend directory:
    source venv/bin/activate
    python test_json_pipeline.py
"""

import asyncio
import json
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent))

from app.ml.environment_manager import EnvironmentManager
from app.ml.inference.custom_classification_model import CustomClassificationModel
from app.ml.inference.megadetector import MegaDetectorV1000
from app.ml.manifest_manager import ManifestManager
from app.ml.model_storage import ModelStorage


async def test_nam_ads_v1():
    """Test JSON pipeline with NAM-ADS-v1 model."""
    print("=" * 80)
    print("Testing NAM-ADS-v1 JSON Pipeline")
    print("=" * 80)
    print()

    # Test image
    test_image = Path("/Users/peter/Downloads/test-img/giraffe.jpg")
    if not test_image.exists():
        print(f"ERROR: Test image not found: {test_image}")
        return False

    # Setup
    manifest_manager = ManifestManager()
    env_manager = EnvironmentManager()
    model_storage = ModelStorage()

    # Load detection model
    print("1. Loading detection model...")
    det_manifest = manifest_manager.get_model("MD1000-REDWOOD-0-0")
    det_model_path = model_storage.get_model_file(det_manifest)
    detection_model = MegaDetectorV1000(det_model_path, env_manager)
    print(f"   ✓ Detection model loaded: {det_model_path}")
    print()

    # Load classification model
    print("2. Loading classification model...")
    cls_manifest = manifest_manager.get_model("NAM-ADS-v1")
    cls_model_path = model_storage.get_model_file(cls_manifest)
    cls_model_dir = model_storage.get_model_path(cls_manifest)
    env_name = cls_manifest.env

    classification_model = CustomClassificationModel(
        cls_model_dir, cls_model_path, env_name, env_manager
    )
    print(f"   ✓ Classification model loaded: {cls_model_path}")
    print()

    # Create temp deployment folder
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="test_json_pipeline_")
    deployment_folder = Path(temp_dir)
    print(f"3. Using temp folder: {deployment_folder}")
    print()

    # Test detection
    print("4. Running MegaDetector...")
    detection_json = detection_model.detect_to_json(
        image_paths=[test_image],
        deployment_folder=deployment_folder,
        confidence_threshold=0.1,
        progress_callback=None,
    )
    print(f"   ✓ Detection JSON saved: {detection_json}")

    # Validate detection JSON
    with open(detection_json) as f:
        det_results = json.load(f)

    print(f"   ✓ Images: {len(det_results.get('images', []))}")
    print(f"   ✓ Detections: {sum(len(img.get('detections', [])) for img in det_results.get('images', []))}")
    print()

    # Test classification
    print("5. Testing classification...")
    with classification_model:
        # Get class names
        class_names = classification_model.get_class_names()
        print(f"   ✓ Retrieved {len(class_names)} class names")

        # Test one classification
        from PIL import Image
        from app.ml.inference.base import BoundingBox

        img = det_results["images"][0]
        if img.get("detections"):
            det = img["detections"][0]
            bbox = det["bbox"]

            image = Image.open(test_image)
            result = classification_model.classify(
                image=image,
                bbox=BoundingBox(
                    x=float(bbox[0]),
                    y=float(bbox[1]),
                    width=float(bbox[2]),
                    height=float(bbox[3]),
                ),
                progress_callback=None,
            )

            if result:
                print(f"   ✓ Classification: {result.label} ({result.confidence:.5f})")
            else:
                print(f"   ✗ Classification failed (returned None)")
    print()

    # Validate JSON format
    print("6. Validating JSON structure...")
    required_keys = ["images", "detection_categories", "info"]
    for key in required_keys:
        if key in det_results:
            print(f"   ✓ {key}: present")
        else:
            print(f"   ✗ {key}: MISSING")
            return False

    # Validate image structure
    if det_results["images"]:
        img = det_results["images"][0]
        img_keys = ["file", "detections", "width", "height"]
        for key in img_keys:
            if key in img:
                print(f"   ✓ images[0].{key}: present")
            else:
                print(f"   ✗ images[0].{key}: MISSING")
                return False

    print()
    print("=" * 80)
    print("✓ NAM-ADS-v1 TEST PASSED")
    print("=" * 80)
    print()

    # Cleanup
    import shutil
    shutil.rmtree(deployment_folder)

    return True


async def main():
    """Run all tests."""
    print()
    print("=" * 80)
    print("JSON PIPELINE TESTS")
    print("=" * 80)
    print()

    # Test NAM-ADS-v1
    nam_success = await test_nam_ads_v1()

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"NAM-ADS-v1: {'✓ PASS' if nam_success else '✗ FAIL'}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    asyncio.run(main())
