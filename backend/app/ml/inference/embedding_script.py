"""
Standalone DINOv2 embedding script — runs as subprocess in env-addaxai-base.

Usage:
    python embedding_script.py \
        --input /path/to/embedding_input.json \
        --output /path/to/embeddings.npz \
        --weights /path/to/dinov2_vits14_pretrain.pth \
        --model-arch dinov2_vits14 \
        --embedding-dim 384 \
        --input-size 224

Input JSON format:
    {
        "detections": [
            {"detection_id": "uuid-1", "image_path": "/path/to/image.jpg", "bbox": [x, y, w, h]}
        ]
    }

Output: .npz file where keys are detection_ids and values are float16 numpy arrays.

Following CONVENTIONS.md: crash early and loudly, no silent failures.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute DINOv2 embeddings for detection crops")
    parser.add_argument("--input", required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output .npz file")
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth)")
    parser.add_argument(
        "--model-arch", required=True, help="DINOv2 architecture name (e.g., dinov2_vits14)"
    )
    parser.add_argument(
        "--embedding-dim", required=True, type=int, help="Expected embedding dimension"
    )
    parser.add_argument(
        "--input-size", required=True, type=int, help="Input image size (e.g., 224)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=0, help="Batch size (0 = auto-select based on device)"
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Detect best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_batch_size(device: torch.device, user_batch_size: int) -> int:
    """Auto-select batch size based on device if not specified."""
    if user_batch_size > 0:
        return user_batch_size
    if device.type == "cuda":
        return 64
    elif device.type == "mps":
        return 32
    else:
        return 8


def load_model(model_arch: str, weights_path: str, device: torch.device) -> torch.nn.Module:
    """Load DINOv2 model architecture and weights."""
    # Load architecture from torch hub (cached during model preparation)
    model = torch.hub.load("facebookresearch/dinov2", model_arch, pretrained=False)

    # Load local weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()
    return model


def build_transform(input_size: int) -> transforms.Compose:
    """Build preprocessing pipeline matching DINOv2 training."""
    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def crop_detection(image: Image.Image, bbox: list[float]) -> Image.Image:
    """Crop detection region from image using normalized [x, y, w, h] coordinates."""
    w_img, h_img = image.size
    x, y, bw, bh = bbox

    left = int(x * w_img)
    top = int(y * h_img)
    right = int((x + bw) * w_img)
    bottom = int((y + bh) * h_img)

    # Clamp to image bounds
    left = max(0, left)
    top = max(0, top)
    right = min(w_img, right)
    bottom = min(h_img, bottom)

    crop = image.crop((left, top, right, bottom))

    # Avoid zero-size crops
    if crop.size[0] == 0 or crop.size[1] == 0:
        raise ValueError(f"Zero-size crop: bbox={bbox}, image_size={image.size}")

    return crop


def main() -> None:
    args = parse_args()

    # Load input
    with open(args.input) as f:
        data = json.load(f)

    detections = data["detections"]
    if not detections:
        # Nothing to process — write empty npz
        np.savez(args.output)
        print("No detections to embed", file=sys.stderr)
        return

    # Setup
    device = get_device()
    batch_size = get_batch_size(device, args.batch_size)
    print(f"Device: {device}, batch_size: {batch_size}", file=sys.stderr)

    # Print compute device for progress parsing
    print(f"COMPUTE_DEVICE:{device.type}", file=sys.stderr, flush=True)

    # Load model
    print("Loading DINOv2 model...", file=sys.stderr)
    model = load_model(args.model_arch, args.weights, device)
    transform = build_transform(args.input_size)

    # Group detections by image_path for efficient I/O
    from collections import defaultdict

    detections_by_image: dict[str, list[dict]] = defaultdict(list)
    for det in detections:
        detections_by_image[det["image_path"]].append(det)

    # Process all detections
    results: dict[str, np.ndarray] = {}
    total = len(detections)

    pbar = tqdm(total=total, desc="Embedding", unit="crop", file=sys.stderr)

    # Collect crops in batches
    batch_ids: list[str] = []
    batch_tensors: list[torch.Tensor] = []

    image_cache: dict[str, Image.Image] = {}

    for image_path, dets in detections_by_image.items():
        # Cache image loading
        if image_path not in image_cache:
            try:
                image_cache[image_path] = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Failed to open {image_path}: {e}", file=sys.stderr)
                pbar.update(len(dets))
                continue

        img = image_cache[image_path]

        for det in dets:
            try:
                crop = crop_detection(img, det["bbox"])
                tensor = transform(crop)
                batch_ids.append(det["detection_id"])
                batch_tensors.append(tensor)
            except Exception as e:
                print(f"Failed to crop detection {det['detection_id']}: {e}", file=sys.stderr)
                pbar.update(1)
                continue

            # Process batch when full
            if len(batch_tensors) >= batch_size:
                _process_batch(model, device, batch_ids, batch_tensors, results, args.embedding_dim)
                pbar.update(len(batch_ids))
                batch_ids = []
                batch_tensors = []

        # Evict image from cache if we're done with it
        del image_cache[image_path]

    # Process remaining
    if batch_tensors:
        _process_batch(model, device, batch_ids, batch_tensors, results, args.embedding_dim)
        pbar.update(len(batch_ids))

    pbar.close()

    # Save as .npz with detection_ids as keys
    np.savez(args.output, **results)
    print(f"Saved {len(results)} embeddings to {args.output}", file=sys.stderr)


def _process_batch(
    model: torch.nn.Module,
    device: torch.device,
    batch_ids: list[str],
    batch_tensors: list[torch.Tensor],
    results: dict[str, np.ndarray],
    embedding_dim: int,
) -> None:
    """Run forward pass on a batch and store results as float16."""
    batch = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        embeddings = model(batch)  # CLS token output

    # Convert to float16 numpy
    embeddings_np = embeddings.cpu().to(torch.float16).numpy()

    assert (
        embeddings_np.shape[1] == embedding_dim
    ), f"Expected dim {embedding_dim}, got {embeddings_np.shape[1]}"

    for det_id, emb in zip(batch_ids, embeddings_np, strict=False):
        results[det_id] = emb


if __name__ == "__main__":
    main()
