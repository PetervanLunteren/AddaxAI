"""Child-process entrypoint for non-MegaDetector model packages.

This module is intentionally tiny and only accepts the backend names and
RF-DETR classes allow-listed by the manifest schema.  It writes a neutral
intermediate JSON document; the parent process performs final validation and
normalization before anything reaches the database.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_rtdetrv2_config = importlib.import_module(
    ".rtdetrv2_config" if __package__ else "rtdetrv2_config",
    package=__package__ or None,
)
load_rtdetrv2_config = _rtdetrv2_config.load_rtdetrv2_config

_RFDETR_CLASSES = {
    "RFDETRBase",
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRMedium",
    "RFDETRLarge",
    "RFDETRXLarge",
    "RFDETR2XLarge",
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRSegXLarge",
    "RFDETRSeg2XLarge",
    "RFDETRKeypointPreview",
    "RFDETRSegPreview",
}

_RTDETR_VARIANTS = {"MDV6-apa-rtdetr-c", "MDV6-apa-rtdetr-e"}


def _force_local_pywildlife_rtdetr_config() -> None:
    """Prevent the PyWildlife wrapper from downloading a redundant backbone.

    PyTorch-Wildlife 1.2.4.x passes ``resume=weights`` to RT-DETRv2's
    ``YAMLConfig``, but that upstream argument does not disable PResNet's
    ``pretrained: true`` setting. The local MegaDetector checkpoint already
    contains the full model state, so loading a separate ImageNet backbone is
    unnecessary and breaks offline/local-only inference.
    """
    module = importlib.import_module(
        "PytorchWildlife.models.detection.rtdetr_apache.rtdetr_apache_base"
    )
    original_factory = module.YAMLConfig

    def local_factory(config_path: str, **kwargs: Any) -> Any:
        config = original_factory(config_path, **kwargs)
        yaml_config = getattr(config, "yaml_cfg", None)
        backbone = yaml_config.get("PResNet") if isinstance(yaml_config, dict) else None
        if not isinstance(backbone, dict):
            raise RuntimeError(
                "PyTorch-Wildlife RT-DETR config is missing its PResNet settings"
            )
        backbone["pretrained"] = False
        return config

    module.YAMLConfig = local_factory


def _xyxy_to_xywh(box: Any, width: float, height: float) -> list[float] | None:
    try:
        x1, y1, x2, y2 = (float(v) for v in box)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return [x1 / width, y1 / height, max(0.0, x2 - x1) / width, max(0.0, y2 - y1) / height]


def _run_yolo(
    model_path: Path,
    paths: list[Path],
    threshold: float,
    image_size: int | None,
) -> list[dict[str, Any]]:
    from ultralytics import YOLO  # type: ignore[import-not-found]

    model = YOLO(str(model_path))
    output: list[dict[str, Any]] = []
    for path in paths:
        # Keep Ultralytics' input and Results collection to a single image.
        # Some predictor/source combinations retain decoded inputs even when
        # stream=True, so a per-image call is the hard memory bound here.
        predict_kwargs: dict[str, Any] = {
            "source": str(path),
            "conf": threshold,
            "verbose": False,
            "stream": False,
        }
        if image_size is not None:
            predict_kwargs["imgsz"] = image_size
        predictions = model.predict(**predict_kwargs)
        result = predictions[0] if predictions else None
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", None) or getattr(model, "names", None) or {}
        detections: list[dict[str, Any]] = []
        if boxes is not None:
            xyxy = getattr(boxes, "xyxy", [])
            confs = getattr(boxes, "conf", [])
            classes = getattr(boxes, "cls", [])
            orig_shape = getattr(result, "orig_shape", (0, 0))
            width, height = orig_shape[1], orig_shape[0]
            for box, conf, cls in zip(xyxy, confs, classes, strict=False):
                bbox = _xyxy_to_xywh(box, width, height)
                if bbox is None:
                    continue
                detections.append({"bbox": bbox, "conf": float(conf), "class_id": str(int(cls))})
        if isinstance(names, dict):
            class_names = {str(k): str(v) for k, v in names.items()}
        elif isinstance(names, list | tuple):
            class_names = {str(k): str(v) for k, v in enumerate(names)}
        else:
            class_names = {}
        output.append({"file": str(path), "detections": detections, "class_names": class_names})
    return output


def _run_rfdetr(
    model_path: Path,
    paths: list[Path],
    threshold: float,
    image_size: int | None,
    model_class: str,
) -> list[dict[str, Any]]:
    if model_class not in _RFDETR_CLASSES:
        raise ValueError(f"RF-DETR model class is not allow-listed: {model_class}")
    import rfdetr  # type: ignore[import-not-found]

    cls = getattr(rfdetr, model_class, None)
    if cls is None:
        raise RuntimeError(f"RF-DETR package does not expose {model_class}")
    kwargs: dict[str, Any] = {"pretrain_weights": str(model_path)}
    if image_size is not None:
        kwargs["resolution"] = int(image_size)
    model = cls(**kwargs)
    model_names = getattr(model, "classes", getattr(model, "class_names", {}))
    output: list[dict[str, Any]] = []
    for path in paths:
        prediction = model.predict(str(path), threshold=threshold)
        boxes = getattr(prediction, "xyxy", getattr(prediction, "boxes", []))
        confidences = getattr(prediction, "confidence", getattr(prediction, "conf", []))
        class_ids = getattr(prediction, "class_id", getattr(prediction, "classes", []))
        detections: list[dict[str, Any]] = []
        from PIL import Image

        width, height = Image.open(path).size
        for box, confidence, class_id in zip(boxes, confidences, class_ids, strict=False):
            bbox = _xyxy_to_xywh(box, width, height)
            if bbox is None or float(confidence) < threshold:
                continue
            detections.append(
                {
                    "bbox": bbox,
                    "conf": float(confidence),
                    "class_id": str(int(class_id)),
                }
            )
        if isinstance(model_names, dict):
            class_names = {str(k): str(v) for k, v in model_names.items()}
        elif isinstance(model_names, list | tuple):
            class_names = {str(k): str(v) for k, v in enumerate(model_names)}
        else:
            class_names = {}
        output.append({"file": str(path), "detections": detections, "class_names": class_names})
    return output


def _device_name(torch: Any) -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(getattr(torch.backends, "mps", None), "is_available", None)
    if callable(mps) and mps():
        return "mps"
    return "cpu"


def _run_rtdetr(
    model_path: Path,
    paths: list[Path],
    threshold: float,
    model_variant: str,
) -> list[dict[str, Any]]:
    if model_variant not in _RTDETR_VARIANTS:
        raise ValueError(f"RT-DETR model variant is not allow-listed: {model_variant}")
    if not model_path.is_file():
        raise FileNotFoundError(f"RT-DETR weights not found: {model_path}")

    import torch  # type: ignore[import-not-found]
    from PIL import Image
    from PytorchWildlife.models import detection as pw_detection  # type: ignore[import-not-found]

    # The upstream 1.2.4.x Apache wrapper ignores its ``pretrained`` argument
    # for the backbone. Disable that config flag too; the local checkpoint is
    # then the sole source of model weights and no Torch Hub download occurs.
    _force_local_pywildlife_rtdetr_config()
    model = pw_detection.MegaDetectorV6Apache(
        weights=str(model_path),
        device=_device_name(torch),
        pretrained=False,
        version=model_variant,
    )
    raw_names = getattr(model, "CLASS_NAMES", {})
    if isinstance(raw_names, Mapping):
        class_names = {str(key): str(value) for key, value in raw_names.items()}
    else:
        class_names = {}

    output: list[dict[str, Any]] = []
    for path in paths:
        prediction = model.single_image_detection(str(path), det_conf_thres=threshold)
        detections_object = prediction.get("detections") if isinstance(prediction, dict) else None
        boxes = getattr(detections_object, "xyxy", None)
        confidences = getattr(detections_object, "confidence", None)
        class_ids = getattr(detections_object, "class_id", None)
        detections: list[dict[str, Any]] = []
        if boxes is not None and confidences is not None and class_ids is not None:
            width, height = Image.open(path).size
            for box, confidence, class_id in zip(boxes, confidences, class_ids, strict=False):
                score = float(confidence)
                if not math.isfinite(score) or score < threshold:
                    continue
                bbox = _xyxy_to_xywh(box, width, height)
                if bbox is not None:
                    detections.append(
                        {"bbox": bbox, "conf": score, "class_id": str(int(class_id))}
                    )
        output.append(
            {"file": str(path), "detections": detections, "class_names": class_names}
        )
    return output


def _extract_rtdetrv2_state(checkpoint: Any) -> Mapping[str, Any]:
    """Extract only the documented EMA-module or model state dictionaries."""
    if not isinstance(checkpoint, Mapping):
        raise ValueError("RT-DETRv2 checkpoint must contain a mapping")
    ema = checkpoint.get("ema")
    state = ema.get("module") if isinstance(ema, Mapping) else None
    if state is None:
        state = checkpoint.get("model")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("RT-DETRv2 checkpoint must contain ema.module or model state")
    if any(not isinstance(key, str) for key in state):
        raise ValueError("RT-DETRv2 state-dict keys must be strings")
    return state


def _tensor_list(value: Any) -> list[Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _rtdetrv2_detections(
    labels: Any,
    boxes: Any,
    scores: Any,
    *,
    width: int,
    height: int,
    threshold: float,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for label, box, score in zip(
        _tensor_list(labels), _tensor_list(boxes), _tensor_list(scores), strict=False
    ):
        confidence = float(score)
        if not math.isfinite(confidence) or confidence < threshold:
            continue
        bbox = _xyxy_to_xywh(box, width, height)
        if bbox is not None:
            detections.append(
                {"bbox": bbox, "conf": confidence, "class_id": str(int(label))}
            )
    return detections


def _run_rtdetrv2(
    model_path: Path,
    config_path: Path,
    source_root: Path,
    paths: list[Path],
    threshold: float,
    image_size: int | None,
) -> list[dict[str, Any]]:
    if not model_path.is_file():
        raise FileNotFoundError(f"RT-DETRv2 weights not found: {model_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"RT-DETRv2 config not found: {config_path}")
    config = load_rtdetrv2_config(
        config_path,
        model_root=config_path.parent,
        source_root=source_root,
    )

    def reject_pretrained_downloads(value: Any, key_path: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                next_path = f"{key_path}.{key}" if key_path else str(key)
                if key == "pretrained" and item is not False:
                    raise ValueError(
                        f"RT-DETRv2 config must disable implicit pretrained downloads: {next_path}"
                    )
                reject_pretrained_downloads(item, next_path)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                reject_pretrained_downloads(item, f"{key_path}[{index}]")

    reject_pretrained_downloads(config)
    if not isinstance(config.get("model"), str) or not isinstance(
        config.get("postprocessor"), str
    ):
        raise ValueError("RT-DETRv2 config must declare model and postprocessor")

    import torch
    import torchvision.transforms as transforms  # type: ignore[import-not-found]
    import yaml  # type: ignore[import-untyped]
    from PIL import Image

    source_root = source_root.resolve()
    if not (source_root / "src" / "core" / "__init__.py").is_file():
        raise FileNotFoundError(f"Pinned RT-DETRv2 source is incomplete: {source_root}")
    sys.path.insert(0, str(source_root))
    from src.core import YAMLConfig  # type: ignore[import-not-found]

    # The upstream YAMLConfig uses yaml.Loader. Its input is generated from
    # the recursively safe-loaded mapping, with all user include paths removed.
    with tempfile.TemporaryDirectory(prefix="addaxai-rtdetrv2-") as temp_dir:
        flattened_config = Path(temp_dir) / "resolved.yml"
        flattened_config.write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        cfg = YAMLConfig(str(flattened_config))
        try:
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "RT-DETRv2 requires a PyTorch version supporting weights_only checkpoint loading"
            ) from exc
        state = _extract_rtdetrv2_state(checkpoint)
        if any(not isinstance(value, torch.Tensor) for value in state.values()):
            raise ValueError("RT-DETRv2 state dict must contain tensors only")
        cfg.model.load_state_dict(state, strict=True)
        device = torch.device(_device_name(torch))
        model = cfg.model.deploy().to(device).eval()
        postprocessor = cfg.postprocessor.deploy().to(device).eval()

        spatial_size = image_size
        if spatial_size is None:
            eval_size = config.get("eval_spatial_size", [640, 640])
            if (
                isinstance(eval_size, list)
                and len(eval_size) == 2
                and all(isinstance(size, int) and size > 0 for size in eval_size)
            ):
                spatial_size = max(eval_size)
            else:
                spatial_size = 640
        transform = transforms.Compose(
            [transforms.Resize((spatial_size, spatial_size)), transforms.ToTensor()]
        )

        output: list[dict[str, Any]] = []
        with torch.inference_mode():
            for path in paths:
                image = Image.open(path).convert("RGB")
                width, height = image.size
                tensor = transform(image).unsqueeze(0).to(device)
                original_size = torch.tensor([[width, height]], device=device)
                labels, boxes, scores = postprocessor(model(tensor), original_size)
                output.append(
                    {
                        "file": str(path),
                        "detections": _rtdetrv2_detections(
                            labels[0],
                            boxes[0],
                            scores[0],
                            width=width,
                            height=height,
                            threshold=threshold,
                        ),
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="AddaxAI isolated detector child")
    parser.add_argument(
        "--backend", choices=("yolo", "rfdetr", "rtdetr", "rtdetrv2"), required=True
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.005)
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--class-names", default="{}")
    parser.add_argument("--model-class", default="RFDETRMedium")
    parser.add_argument("--model-variant", default="MDV6-apa-rtdetr-c")
    parser.add_argument("--model-config")
    parser.add_argument("--rtdetrv2-source")
    args = parser.parse_args()
    paths = [Path(p) for p in json.loads(Path(args.inputs).read_text(encoding="utf-8"))]
    if args.backend == "yolo":
        images = _run_yolo(Path(args.model), paths, args.threshold, args.image_size)
    elif args.backend == "rfdetr":
        images = _run_rfdetr(
            Path(args.model), paths, args.threshold, args.image_size, args.model_class
        )
    elif args.backend == "rtdetr":
        images = _run_rtdetr(Path(args.model), paths, args.threshold, args.model_variant)
    else:
        if not args.model_config or not args.rtdetrv2_source:
            raise ValueError("rtdetrv2 requires --model-config and --rtdetrv2-source")
        images = _run_rtdetrv2(
            Path(args.model),
            Path(args.model_config),
            Path(args.rtdetrv2_source),
            paths,
            args.threshold,
            args.image_size,
        )
    Path(args.output).write_text(json.dumps({"images": images}), encoding="utf-8")


if __name__ == "__main__":
    main()
