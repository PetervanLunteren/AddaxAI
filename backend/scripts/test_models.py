#!/usr/bin/env python3
"""
Check that every classification model runs, picks the right compute
device, and still agrees with the legacy AddaxAI.

Run this by hand on each OS after porting models or bumping an
environment. It is deliberately not part of CI: it needs real weights and
real micromamba envs (many GB), which no other test in this repo touches.

What it checks, per model:

  runs      the classification subprocess exits 0 and returns a label
  device    the device the model actually chose, reported by the worker
            itself. Some models are legitimately CPU-only (see
            EXPECTED_CPU), and those are not failures.
  taxonomy  every class the model emits has a row in taxonomy.csv.
            Detection.label is matched against label_taxonomy.name by
            plain string equality with no foreign key, so one typo drops
            a species out of the filter tree and out of rollup with no
            error anywhere. Cheap to check here, invisible in production.
  matches   top-1 label equals what the legacy AddaxAI produced for the
            same image and box, with confidence within CONF_TOLERANCE.
            Expectations come from tests/data/model_expectations.json,
            generated once by scripts/generate_legacy_ground_truth.py.

Confidence is compared loosely on purpose. Ground truth is generated on
one machine (MPS), and the same weights on CUDA or CPU differ in the last
few decimals. A tight bound would fail on Windows for no reason, so the
label must match exactly while the confidence only has to be close. The
delta is always printed, so a real drift is still visible.

Usage
-----
    cd backend
    source venv/bin/activate

    python scripts/test_models.py                     # every cls model
    python scripts/test_models.py --model AHDRIFT-v1  # just one
    python scripts/test_models.py --skip-missing      # only what is downloaded
    python scripts/test_models.py --model-dir ./staged/AHDRIFT-v1

Options:
    --model         Model id. Repeatable. Default: every cls model in the catalog.
    --skip-missing  Skip models whose weights are not downloaded, instead of
                    downloading them (a full run pulls many GB).
    --model-dir     Test a directory directly, bypassing the catalog. Use to
                    validate a staged inference.py before uploading it to
                    HuggingFace.
    --json          Emit machine-readable results instead of the table.

Exit code is 0 when every model tested passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ml.environment_manager import EnvironmentManager  # noqa: E402
from app.ml.inference.custom_classification_model import (  # noqa: E402
    CustomClassificationModel,
)
from app.ml.manifest_manager import ManifestManager  # noqa: E402
from app.ml.model_storage import ModelStorage  # noqa: E402

EXPECTATIONS_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "model_expectations.json"
)

# Top-1 confidence may drift this far from the recorded ground truth
# before it is called a failure. Wide enough to absorb MPS/CUDA/CPU
# float differences, tight enough that a wrong preprocessing step (which
# moves confidence by far more than this) still trips it.
CONF_TOLERANCE = 0.02

# Models that cannot use the GPU, with the reason. Being on this list
# turns "ran on CPU" from a failure into an expected result.
EXPECTED_CPU: dict[str, str] = {
    # andrea-anpc calls tf.config.set_visible_devices([], 'GPU') on
    # Darwin, so TropiCam-AI is CPU-only on macOS by the model's own
    # choice, not ours.
    "NEO-MNCN-v1-0": "darwin",
}


@dataclass
class TestImage:
    """One shared test image, with the box every model classifies."""

    name: str
    url: str
    sha256: str
    # Normalised MegaDetector box: (x_min, y_min, width, height).
    bbox: tuple[float, float, float, float]


# The fixed, shared image set. Defined here rather than in the
# expectations file so there is one definition of "what we test on";
# generate_legacy_ground_truth.py imports this list, so the ground truth
# and the test can never disagree about which pixels they mean.
#
# Every model sees the same images, which makes this a fidelity check
# against legacy rather than an accuracy check: an off-region model
# returns nonsense, but deterministic nonsense, and a broken crop or
# normalisation still shows up immediately.
#
# All are public LILA BC images, served from the project's Google Cloud
# mirror. Boxes are full-frame: it is the box a full-image classifier
# gets anyway, it always contains the animal, and it keeps the set
# honest for crop-based models without needing MegaDetector in the loop.
TEST_IMAGES: list[TestImage] = [
    # Ohio drift fence, eastern chipmunk. From osu-small-animals, which
    # is AHDriFT-ID's own training source, so this one doubles as a real
    # accuracy check for AHDRIFT-v1: it should say "eastern chipmunk".
    TestImage(
        name="osu_eastern_chipmunk.jpg",
        url=(
            "https://storage.googleapis.com/public-datasets-lila/"
            "osu-small-animals/Images/Sorted_by_species/Mammalia/"
            "Eastern%20Chipmunk/FCM1__2019-09-22__11-31-52%283%29.JPG"
        ),
        sha256="c650fc3423b254547bb8dff027f106321f2359954c17783f71a4badd34f4cfe9",
        bbox=(0.0, 0.0, 1.0, 1.0),
    ),
]


@dataclass
class Result:
    model_id: str
    env: str = ""
    device: str = ""
    top1: str = ""
    expected: str = ""
    delta: float | None = None
    status: str = "SKIP"
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("PASS", "SKIP")


def _load_expectations() -> dict:
    if not EXPECTATIONS_PATH.exists():
        print(
            f"No expectations file at {EXPECTATIONS_PATH}.\n"
            f"Models will be checked for 'runs' and 'device' only. Generate "
            f"the ground truth with scripts/generate_legacy_ground_truth.py "
            f"to also check outputs against legacy.",
            file=sys.stderr,
        )
        return {}
    with open(EXPECTATIONS_PATH) as f:
        return json.load(f)


def _cache_dir() -> Path:
    path = Path.home() / "AddaxAI" / "test-images"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fetch_image(image: TestImage) -> Path:
    """Download the image once and verify its hash. Cached across runs."""
    dest = _cache_dir() / image.name
    if dest.exists():
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest == image.sha256:
            return dest
        print(f"  cached {image.name} has the wrong hash, refetching")

    print(f"  downloading {image.name}")
    urllib.request.urlretrieve(image.url, dest)
    digest = hashlib.sha256(dest.read_bytes()).hexdigest()
    if digest != image.sha256:
        raise SystemExit(
            f"{image.name} hash mismatch.\n"
            f"  expected {image.sha256}\n"
            f"  got      {digest}\n"
            f"The upstream file changed; ground truth is no longer comparable."
        )
    return dest


def _check_taxonomy_joins(model_dir: Path, class_names: dict[str, str]) -> list[str]:
    """
    Report classes the model emits that taxonomy.csv has no row for.

    Not a failure: taxonomy.csv is optional and a model without one still
    classifies, it just falls back to a flat label list. But a taxonomy
    that is present and *nearly* right is worse than none, because the
    missing species vanish silently.

    Compared case-insensitively, which is what the app does: taxonomy.csv's
    model_class is lowercased into LabelTaxonomy.name, while Detection.label
    keeps the case the model emitted, and the linking matches the two on
    lowercase.
    """
    taxonomy_path = model_dir / "taxonomy.csv"
    if not taxonomy_path.exists():
        return ["no taxonomy.csv (flat label list, no rollup)"]

    with open(taxonomy_path, newline="", encoding="utf-8-sig") as f:
        rows = {(r.get("model_class") or "").strip().lower() for r in csv.DictReader(f)}

    emitted = {name.strip().lower() for name in class_names.values()}
    unjoined = sorted(emitted - rows)
    if not unjoined:
        return []
    shown = ", ".join(unjoined[:5])
    more = f" (+{len(unjoined) - 5} more)" if len(unjoined) > 5 else ""
    return [f"{len(unjoined)}/{len(emitted)} classes missing from taxonomy.csv: {shown}{more}"]


def run_model(
    model_id: str,
    model_dir: Path,
    model_path: Path,
    env_name: str,
    expectation: dict | None,
) -> Result:
    """Classify the shared image set with one model."""
    result = Result(model_id=model_id, env=env_name)

    items = []
    for image in TEST_IMAGES:
        items.append(
            {
                "source": "image",
                "image_path": str(_fetch_image(image)),
                "bbox": list(image.bbox),
            }
        )

    model = CustomClassificationModel(
        model_dir=model_dir,
        model_path=model_path,
        env_name=env_name,
        env_manager=EnvironmentManager(),
    )
    # batch_size=None lets the worker pick its own default, which is the
    # path a real analysis takes.
    results, class_names, device, _best = model.classify_detections(items)

    result.device = device
    result.notes.extend(_check_taxonomy_joins(model_dir, class_names))
    if device == "CPU":
        reason = EXPECTED_CPU.get(model_id)
        if reason and reason == sys.platform:
            result.notes.append("CPU expected on this OS")
        else:
            result.notes.append("no GPU used")

    predictions = []
    for item_result in results:
        if item_result is None:
            result.status = "FAIL"
            result.notes.append("returned no classification")
            return result
        predictions.append((item_result.label, item_result.confidence))

    result.top1 = predictions[0][0]

    if not expectation:
        result.status = "RAN"
        result.notes.append("no ground truth recorded")
        return result

    result.status = "PASS"
    for i, (label, conf) in enumerate(predictions):
        want = expectation["predictions"][i]
        result.expected = want["label"] if i == 0 else result.expected
        delta = abs(conf - want["confidence"])
        if i == 0:
            result.delta = delta
        if label != want["label"]:
            result.status = "FAIL"
            result.notes.append(
                f"image {i}: got {label!r}, legacy said {want['label']!r}"
            )
        elif delta > CONF_TOLERANCE:
            result.status = "FAIL"
            result.notes.append(
                f"image {i}: {label!r} confidence off by {delta:.4f} "
                f"(got {conf:.4f}, legacy {want['confidence']:.4f})"
            )
    return result


def _print_table(results: list[Result]) -> None:
    header = (
        f"{'model':<22} {'env':<14} {'device':<21} {'top-1':<24} "
        f"{'expected':<24} {'Δconf':>7}  status"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in results:
        delta = f"{r.delta:.4f}" if r.delta is not None else "-"
        print(
            f"{r.model_id:<22} {r.env:<14} {r.device:<21} {r.top1[:24]:<24} "
            f"{r.expected[:24]:<24} {delta:>7}  {r.status}"
        )
        for note in r.notes:
            print(f"{'':<22} {'':<14} -> {note}")
    print("-" * len(header))
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    print("  ".join(f"{k}: {v}" for k, v in sorted(counts.items())))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check every classification model runs, uses the right "
        "device, and matches legacy AddaxAI output."
    )
    parser.add_argument(
        "--model", action="append", dest="models", help="Model id. Repeatable."
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip models whose weights are not downloaded.",
    )
    parser.add_argument(
        "--model-dir",
        help="Test a model directory directly, bypassing the catalog.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    args = parser.parse_args(argv)

    expectations = _load_expectations()

    manifest_manager = ManifestManager()
    storage = ModelStorage()
    results: list[Result] = []

    if args.model_dir:
        model_dir = Path(args.model_dir).resolve()
        manifest_path = model_dir / "manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"No manifest.json in {model_dir}")
        manifest_data = json.loads(manifest_path.read_text())
        results.append(
            run_model(
                model_id=manifest_data["model_id"],
                model_dir=model_dir,
                model_path=model_dir / manifest_data["model_fname"],
                env_name=manifest_data["env"],
                expectation=expectations.get("models", {}).get(
                    manifest_data["model_id"]
                ),
            )
        )
        _print_table(results)
        return 0 if all(r.ok for r in results) else 1

    manifests = [
        m
        for m in manifest_manager.load_manifests().values()
        if m.model_category == "classification"
        and (not args.models or m.model_id in args.models)
    ]
    if args.models:
        missing = set(args.models) - {m.model_id for m in manifests}
        if missing:
            raise SystemExit(f"Not in the catalog: {', '.join(sorted(missing))}")

    for manifest in sorted(manifests, key=lambda m: m.model_id):
        result = Result(model_id=manifest.model_id, env=manifest.env)
        try:
            if not storage.check_weights_ready(manifest):
                if args.skip_missing:
                    result.notes.append("weights not downloaded")
                    results.append(result)
                    continue
                print(f"Downloading {manifest.model_id}...")
                storage.download_weights(manifest)

            result = run_model(
                model_id=manifest.model_id,
                model_dir=storage.get_model_path(manifest),
                model_path=storage.get_model_file(manifest),
                env_name=manifest.env,
                expectation=expectations.get("models", {}).get(manifest.model_id),
            )
        except Exception as e:
            result.status = "ERROR"
            result.notes.append(f"{type(e).__name__}: {e}")
        results.append(result)

    if args.json:
        print(json.dumps([r.__dict__ for r in results], indent=2, default=str))
    else:
        _print_table(results)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
