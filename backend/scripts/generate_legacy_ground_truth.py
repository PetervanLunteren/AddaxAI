#!/usr/bin/env python3
"""
Record what the legacy AddaxAI predicts, so the ported models can be
checked against it.

Run this once, on a machine with the legacy AddaxAI installed. It writes
tests/data/model_expectations.json, which scripts/test_models.py then
compares against. You should not need to run it again unless the image
set changes or a new model is ported.

How it gets the numbers
-----------------------
Each legacy model is driven through its own supported entry point: the
per-architecture `classify_detections.py` adapter, called with the same 9
positional arguments the legacy GUI passes. No monkey-patching, no
reaching into internals. The settings are chosen so the output is the
model's raw softmax:

    cls_detec_thresh  0.0    classify every box
    cls_class_thresh  0.0    only used by rollup, which is off
    smooth_bool       False  no sequence smoothing
    cls_tax_fallback  False  no rollup, so no taxon-mapping.csv is read
    selected_classes  = all_classes, so forbidden_classes is empty and
                       remove_forbidden_classes renormalises by 1.0

inference_lib writes the full distribution, sorted, rounded to 5dp, into
`<json>_original.json`. That is the ground truth; we keep the top 5 to
match what the new worker retains (MAX_CLASSIFICATIONS_KEPT).

The legacy install is never modified. Weights are downloaded into a work
directory and the adapter is pointed at that, so /Applications stays
read-only. Only the adapter code and the conda envs are read from it.

Usage
-----
    cd backend
    source venv/bin/activate

    python scripts/generate_legacy_ground_truth.py --model EUR-DF-v1-3
    python scripts/generate_legacy_ground_truth.py            # all mapped models

Options:
    --model         New model id. Repeatable. Default: every model in LEGACY_BY_MODEL_ID.
    --work-dir      Where to download legacy weights. Default ~/AddaxAI/legacy-ground-truth.
    --legacy-root   Default /Applications/AddaxAI_files.
    --keep-going    Carry on when a model fails, instead of stopping.

A model whose legacy env is not installed is reported and skipped: the
legacy app downloads envs on demand, so open it and select the model once
to install the env, then re-run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_models import TEST_IMAGES, _fetch_image  # noqa: E402

EXPECTATIONS_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "model_expectations.json"
)

# Kept to match the new worker's MAX_CLASSIFICATIONS_KEPT, so the two
# sides of the comparison hold the same shape of data.
TOP_K = 5

# New model_id -> the key it has in the legacy registry. The legacy
# registry is a dict keyed by friendly name, and that key doubles as the
# on-disk directory name, so this is the only link between the two apps.
# Models with no legacy equivalent (TKM-ADS-v1 is the Iran model renamed)
# are simply absent.
LEGACY_BY_MODEL_ID: dict[str, str] = {
    # Already ported
    "SPECIESNET-v4-0-2-A": "Global - SpeciesNet - Google",
    "EUR-DF-v1-3": "Europe - DeepFaune v1.3",
    "SAH-DRY-ADS-v1": "Sub-Saharan Drylands - Addax Data Science",
    "TERRAI-NEP-v1": "Terai region Nepal - Alexander Merdian-Tarko",
    "TAS-BB-v1": "Tasmanian vertebrates",
    "NAM-ADS-v1": "Namibian Desert - Addax Data Science",
    "NZI-ADS-v1": "New Zealand Invasives - DOC NZ - Addax Data science",
    "PAM-SDZWA-v1": "Peruvian Amazon - San Diego Zoo Wildlife Alliance",
    "TKM-ADS-v1": "Iran - Addax Data Science",
    "KIR-HEX-v1": "Kirghizistan - Manas v1 - OSI-Panthera - Hex Data",
    "SWUSA-SDZWA-v3": "Southwest USA v3 - San Diego Zoo Wildlife Alliance",
    "GIF-JAP-v0-2": "Gifu region Japan - Gifu University",
    "HWI-ADS-v1": "Hawaiʻi, USA - AI Puaʻa v1.0",
    "VIC-ADS-v1": "Victoria, Australia - Parks Victoria - Addax Data Science",
    "SBUSA-ADS-v1": "Southwestern Borderlands USA",
    "AHDRIFT-v1": "AHDriFT-ID (Midwest US) v1.0",
    # Being ported
    "IND-ADS-v1": "Central Indian Landscapes – Wildlife Conservation Trust",
    "ANT-ADS-v1": "Top End Savanna Vertebrates, Northern Territory Australia",
    "QLD-WOB-v1": "Queensland Wet Tropics - WildObs",
    "AWC135-AWC-v1": "Australian Wildlife Classifier - AWC135",
    "NZS-WEK-v3-03": "New Zealand Species v3.03 - DOC NZ - wekaResearch",
    "NEO-MNCN-v1-0": "Neotropical region - TropiCam-AI v1.0",
    "EUR-DF-v1-4": "Europe - DeepFaune v1.4",
    "EUR-DF-v1-2": "Europe - DeepFaune v1.2",
    "EUR-DF-v1-1": "Europe - DeepFaune v1.1",
    "PAN-SDZWA-v1": "Perivuan Andes - San Diego Zoo Wildlife Alliance",
    "AFR-DFV-v1": "African tropical forests - DeepForestVision",
    "CAM-AI4G-v1": "Colombian Amazon - AI for Good Lab, Microsoft",
}

# platform.system() -> the per-OS override key legacy uses.
_ENV_OVERRIDE_KEY = {"Darwin": "env-macos", "Linux": "env-linux", "Windows": "env-windows"}


def load_registry(legacy_root: Path) -> dict:
    path = legacy_root / "AddaxAI" / "model_info" / "model_info_v5.json"
    if not path.exists():
        raise SystemExit(f"No legacy registry at {path}")
    with open(path) as f:
        return json.load(f)["cls"]


def resolve_env(model_vars: dict) -> str:
    """Pick the conda env, honouring legacy's per-OS override."""
    import platform

    override = _ENV_OVERRIDE_KEY.get(platform.system())
    if override and override in model_vars:
        return model_vars[override]
    return model_vars["env"]


def prepare_model_dir(
    model_vars: dict, work_dir: Path, legacy_key: str, legacy_root: Path
) -> Path:
    """
    Build a self-contained copy of the legacy model directory.

    Weights come from the legacy install when they are already there, and
    from the same URLs the legacy app uses otherwise. The legacy install
    is only ever read: the adapter derives its sibling paths (backbone,
    class list) from the weights path it is given, so pointing it at this
    directory is enough to keep /Applications untouched.
    """
    model_dir = work_dir / legacy_key
    model_dir.mkdir(parents=True, exist_ok=True)
    installed = legacy_root / "models" / "cls" / legacy_key

    for url, fname in model_vars["download_info"]:
        dest = model_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        # Symlink rather than copy: these run to hundreds of MB each and
        # the adapter only reads them.
        already_installed = installed / fname
        if already_installed.exists() and already_installed.stat().st_size > 0:
            print(f"    using installed {fname}")
            dest.symlink_to(already_installed)
            continue
        print(f"    downloading {fname}")
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dest)

    # variables.json is what fetch_forbidden_classes reads. Force
    # selected == all so nothing is zeroed and the renormalisation is a
    # no-op: we want the model's raw softmax, not a user-filtered view.
    variables = dict(model_vars)
    variables["selected_classes"] = list(model_vars.get("all_classes", []))
    with open(model_dir / "variables.json", "w") as f:
        json.dump(variables, f, indent=1)

    return model_dir


def write_synthetic_md_json(image_dir: Path) -> Path:
    """
    Write a MegaDetector-shaped JSON with one full-frame animal box per
    test image, mirroring what full_image_detection.synthesize_full_image_json
    does on the new side.

    inference_lib resolves `file` against the JSON's own directory, so
    the images are linked in next to it.

    Call this once per model, into a fresh directory. The legacy pipeline
    treats this file as single-use: its "rewrite json to be used by
    AddaxAI" step (inference_lib.py:309) runs unconditionally, even with
    smoothing off, and folds the model's class names into
    `detection_categories` while repointing each detection's category at
    its predicted class. A second model reading the same file finds no
    box whose category is "animal" and reports nothing to classify.
    """
    image_dir.mkdir(parents=True, exist_ok=True)
    images = []
    for image in TEST_IMAGES:
        src = _fetch_image(image)
        dest = image_dir / image.name
        if not dest.exists():
            dest.symlink_to(src)
        images.append(
            {
                "file": image.name,
                "detections": [
                    {"category": "1", "conf": 1.0, "bbox": list(image.bbox)}
                ],
            }
        )

    json_path = image_dir / "image_recognition_file.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "info": {"detector": "ground-truth-harness"},
                "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
                "images": images,
            },
            f,
            indent=1,
        )
    return json_path


def _legacy_env(legacy_root: Path) -> dict[str, str]:
    """
    Reproduce the PYTHONPATH the legacy GUI exports before it spawns an
    adapter (AddaxAI_GUI.py:152-163). The adapters import inference_lib,
    which imports `megadetector` from the vendored cameratraps checkout
    rather than from the conda env, so without this every adapter dies on
    ModuleNotFoundError.
    """
    separator = ";" if sys.platform == "win32" else ":"
    paths = [
        str(legacy_root),
        str(legacy_root / "cameratraps"),
        str(legacy_root / "cameratraps" / "megadetector"),
        str(legacy_root / "AddaxAI"),
    ]
    env = {
        "PYTHONPATH": separator.join(paths),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        # Keras picks its backend from the env; legacy's tensorflow envs
        # run on jax. Harmless for the torch adapters.
        "KERAS_BACKEND": "jax",
        # Some adapters read HOME for cache dirs (timm, torch.hub).
        "HOME": str(Path.home()),
    }
    # Legacy exports this before every classification subprocess on macOS
    # and on Linux-with-GPU (AddaxAI_GUI.py:2848-2854), and the new app
    # does the same on Darwin. Without it DeepFaune dies inside timm on
    # aten::_upsample_bicubic2d_aa, which MPS has no kernel for. Both
    # sides must agree here or the comparison is meaningless.
    if sys.platform in ("darwin", "linux"):
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return env


def run_legacy_adapter(
    legacy_root: Path, model_vars: dict, model_dir: Path, json_path: Path
) -> None:
    """Invoke the adapter exactly as the legacy GUI does: 9 positional args."""
    env_name = resolve_env(model_vars)
    python = legacy_root / "envs" / f"env-{env_name}" / "bin" / "python"
    if not python.exists():
        raise FileNotFoundError(
            f"legacy env 'env-{env_name}' is not installed at {python}. "
            f"Open the legacy AddaxAI and select this model once to install it."
        )

    adapter = (
        legacy_root
        / "AddaxAI"
        / "classification_utils"
        / "model_types"
        / model_vars["type"]
        / "classify_detections.py"
    )
    if not adapter.exists():
        raise FileNotFoundError(f"no adapter for type {model_vars['type']!r} at {adapter}")

    cmd = [
        str(python),
        str(adapter),
        str(legacy_root),                          # 1 AddaxAI_files
        str(model_dir / model_vars["model_fname"]),  # 2 cls_model_fpath
        "0.0",                                     # 3 cls_detec_thresh
        "0.0",                                     # 4 cls_class_thresh
        "False",                                   # 5 smooth_bool
        str(json_path),                            # 6 json_path
        "None",                                    # 7 temp_frame_folder
        "False",                                   # 8 cls_tax_fallback
        "0",                                       # 9 cls_tax_levels_idx
    ]
    result = subprocess.run(
        cmd,
        cwd=str(legacy_root),
        env=_legacy_env(legacy_root),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-1500:]
        raise RuntimeError(f"adapter exited {result.returncode}:\n{tail}")


def read_predictions(json_path: Path) -> list[list[dict]]:
    """Read the top-K distribution per image out of `<json>_original.json`."""
    original = json_path.with_name(json_path.stem + "_original" + json_path.suffix)
    if not original.exists():
        raise FileNotFoundError(f"legacy wrote no output at {original}")

    with open(original) as f:
        data = json.load(f)

    id_to_name = data.get("classification_categories", {})
    per_image: list[list[dict]] = []
    for image in data["images"]:
        detections = image.get("detections") or []
        if not detections or "classifications" not in detections[0]:
            raise RuntimeError(f"legacy returned no classifications for {image['file']}")
        ranked = [
            {"label": id_to_name[str(cid)], "confidence": float(conf)}
            for cid, conf in detections[0]["classifications"][:TOP_K]
        ]
        per_image.append(ranked)
    return per_image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record legacy AddaxAI predictions as ground truth."
    )
    parser.add_argument("--model", action="append", dest="models", help="New model id. Repeatable.")
    parser.add_argument(
        "--work-dir",
        default=str(Path.home() / "AddaxAI" / "legacy-ground-truth"),
        help="Where legacy weights are downloaded.",
    )
    parser.add_argument("--legacy-root", default="/Applications/AddaxAI_files")
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args(argv)

    legacy_root = Path(args.legacy_root)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry(legacy_root)
    wanted = args.models or list(LEGACY_BY_MODEL_ID)

    expectations: dict = {"images": [], "models": {}}
    if EXPECTATIONS_PATH.exists():
        with open(EXPECTATIONS_PATH) as f:
            expectations = json.load(f)
    expectations["images"] = [
        {"name": i.name, "url": i.url, "sha256": i.sha256, "bbox": list(i.bbox)}
        for i in TEST_IMAGES
    ]

    failures: list[str] = []

    for model_id in wanted:
        legacy_key = LEGACY_BY_MODEL_ID.get(model_id)
        if legacy_key is None:
            failures.append(f"{model_id}: not in LEGACY_BY_MODEL_ID")
            continue
        if legacy_key not in registry:
            failures.append(f"{model_id}: {legacy_key!r} is not in the legacy registry")
            continue

        print(f"\n{model_id}  ({legacy_key})")
        try:
            model_vars = registry[legacy_key]
            model_dir = prepare_model_dir(model_vars, work_dir, legacy_key, legacy_root)
            # Fresh run directory per model: the legacy pipeline rewrites
            # its input JSON in place, so the file is single-use.
            run_dir = work_dir / "runs" / model_id
            shutil.rmtree(run_dir, ignore_errors=True)
            json_path = write_synthetic_md_json(run_dir)
            run_legacy_adapter(legacy_root, model_vars, model_dir, json_path)
            per_image = read_predictions(json_path)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}")
            failures.append(f"{model_id}: {e}")
            if not args.keep_going:
                break
            continue

        expectations["models"][model_id] = {
            "source": "legacy",
            "legacy_key": legacy_key,
            "legacy_type": model_vars["type"],
            "legacy_env": resolve_env(model_vars),
            # Parallel to TEST_IMAGES. Index 0 is image 0.
            "predictions": [ranked[0] for ranked in per_image],
            "top_k": per_image,
        }
        for image, ranked in zip(TEST_IMAGES, per_image, strict=True):
            top = ranked[0]
            print(f"    {image.name}: {top['label']} ({top['confidence']:.5f})")

    EXPECTATIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPECTATIONS_PATH, "w") as f:
        json.dump(expectations, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\nWrote {EXPECTATIONS_PATH} ({len(expectations['models'])} models)")

    if failures:
        print(f"\n{len(failures)} model(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
