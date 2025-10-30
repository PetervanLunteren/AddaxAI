# Finetuning SpeciesNet

The last few days I've been breaking my head on how to finetune the always-crop PyTorch SpeciesNet on a custom dataset. Together with my two good friends Claude and Codex we managed to figure something out. It hasn't been thoroughly tested, but it seems to work much better than just training a pretrained efficientnet_v2_m. Below I have listed the main findings and a simple script to try it out, so that you don't have to redo the same steps.

## Format
SpeciesNet ships as an ONNX2Torch-exported Keras FX graph, so the layers aren’t arranged in the usual fine-tuning-friendly PyTorch modules. You need to permute the NHWC inputs and adaptively pool the backbone output before the graph behaves like a standard feature extractor.

```python
class FXClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, img_size: int, input_layout: str = "nhwc"):
        super().__init__()
        self.backbone = backbone
        self.input_layout = input_layout.lower()

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            if self.input_layout == "nhwc":
                dummy = dummy.permute(0, 2, 3, 1).contiguous()
            features = self.backbone(dummy)
            if features.ndim == 4:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            elif features.ndim == 3:
                features = features.mean(dim=1)
            else:
                features = features.flatten(1)
            in_features = features.shape[1]

        self.head = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.head.weight, gain=0.5)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_layout == "nhwc":
            x = x.permute(0, 2, 3, 1).contiguous()
        features = self.backbone(x)
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        elif features.ndim == 3:
            features = features.mean(dim=1)
        else:
            features = features.flatten(1)
        return self.head(features)
```

### Stage unfreezing
FX modules don’t expose tidy layer blocks the way timm backbones do. The raw graph contains roughly 490 individual parameter tensors, which is hard to reason about. Bucketing parameters by FX node name reduces that to 59 logical stages—much easier to manage and comparable to timm-style fine-tuning.

```python
def stage_key_from_param_name(name: str) -> str:
    parts = name.split("/")
    if len(parts) < 3:
        return name
    base = parts[2]
    return base.split("_", 1)[0] if "_" in base else base


def freeze_backbone_by_stages(model: nn.Module, unfreeze_layers: int) -> None:
    stage_order: list[str] = []
    stage_to_params: dict[str, list[torch.nn.Parameter]] = {}
    for name, param in model.backbone.named_parameters():
        stage = stage_key_from_param_name(name)
        if stage not in stage_to_params:
            stage_order.append(stage)
            stage_to_params[stage] = []
        stage_to_params[stage].append(param)

    for params in stage_to_params.values():
        for p in params:
            p.requires_grad = False

    if unfreeze_layers == -1:
        selected = stage_order
    elif unfreeze_layers > 0:
        selected = stage_order[-min(unfreeze_layers, len(stage_order)) :]
    else:
        selected = []

    for stage in selected:
        for p in stage_to_params[stage]:
            p.requires_grad = True

    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True
```

### NCHW handling
The SpeciesNet graph takes NHWC tensors and may return 4D or 3D outputs, so we permute inputs to NHWC before the forward pass and use adaptive pooling to collapse whatever shape comes back.

```python
if self.input_layout == "nhwc":
    x = x.permute(0, 2, 3, 1).contiguous()
features = self.backbone(x)
if features.ndim == 4:
    features = F.adaptive_avg_pool2d(features, 1).flatten(1)
elif features.ndim == 3:
    features = features.mean(dim=1)
else:
    features = features.flatten(1)
```

## Working example
Below is a standalone simplified SpeciesNet fine-tune entry point that auto-downloads the crop checkpoint, wraps the FX graph with a fresh linear head, infers classes from train/, and runs a single-phase training loop.

### Requirements
It needs the basic requirements of SpeciesNet (see [docs](https://github.com/google/cameratrapai?tab=readme-ov-file#installing-the-speciesnet-python-package)). 

### Dataset
It expects its dataset in the standard `torchvision.datasets.ImageFolder` format.

```
root/
├── train/
│   ├── cat/
│   │   ├── img001.jpg
│   │   └── ...
│   ├── dog/
│   ├── cow/
│   └── ...
├── val/
│   ├── cat/
│   ├── dog/
│   ├── cow/
│   └── ...
└── test/
    └── ...
```

### Script

This script has the minimal code needed.

<details>
  <summary>finetune_simple.py</summary>

```python
#!/usr/bin/env python3
"""Minimal standalone script for fine-tuning SpeciesNet on project crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SPECIESNET_CROP_FILENAME = "always_crop_99710272_22x8_v12_epoch_00148.pt"
SPECIESNET_CROP_URL = (
    "https://huggingface.co/Addax-Data-Science/SPECIESNET-v4-0-1-A-v1/resolve/main/"
    "always_crop_99710272_22x8_v12_epoch_00148.pt?download=true"
)
DEFAULT_IMG_SIZE = 480


def download_weights(url: str, dest_path: Path) -> None:
    """Download SpeciesNet weights if they are missing."""
    import urllib.request

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading SpeciesNet weights to {dest_path}...")

    def progress(block_count, block_size, total_size):
        if total_size <= 0:
            return
        downloaded = block_count * block_size
        percent = min(100, downloaded * 100 / total_size)
        bar_len = 40
        filled = int(bar_len * percent / 100)
        bar = "=" * filled + "-" * (bar_len - filled)
        mb_downloaded = downloaded / (1024 ** 2)
        mb_total = total_size / (1024 ** 2)
        print(f"\r[{bar}] {percent:5.1f}% ({mb_downloaded:6.1f}/{mb_total:6.1f} MB)", end="")

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=progress)
        print()  # newline after progress bar
    except Exception as exc:  # pragma: no cover - network errors
        if dest_path.exists():
            dest_path.unlink()
        raise RuntimeError(f"Failed to download weights: {exc}")


def ensure_speciesnet_weights(script_dir: Path) -> Path:
    """Return path to speciesnet-crop weights, downloading if necessary."""
    local_path = Path.cwd() / SPECIESNET_CROP_FILENAME
    if local_path.exists():
        return local_path

    fallback = script_dir / SPECIESNET_CROP_FILENAME
    if fallback.exists():
        return fallback

    print("SpeciesNet crop weights not found locally. Initiating download...")
    download_weights(SPECIESNET_CROP_URL, fallback)
    return fallback


def load_fx_checkpoint(weights_path: Path, map_location="cpu") -> torch.nn.Module:
    """Load the exported SpeciesNet GraphModule with PyTorch 2.6 safeguards."""
    try:
        from torch.serialization import add_safe_globals
        from torch.fx.graph_module import reduce_graph_module
        add_safe_globals([reduce_graph_module])
    except Exception:
        pass

    try:
        obj = torch.load(weights_path, map_location=map_location, weights_only=True)
    except Exception:
        obj = torch.load(weights_path, map_location=map_location, weights_only=False)

    if hasattr(obj, "state_dict") and hasattr(obj, "forward"):
        return obj
    raise ValueError("Expected SpeciesNet GraphModule; verify weights file")


def stage_key_from_param_name(name: str) -> str:
    parts = name.split("/")
    if len(parts) < 3:
        return name
    base = parts[2]
    return base.split("_", 1)[0] if "_" in base else base


def freeze_backbone_by_stages(model: nn.Module, unfreeze_layers: int) -> None:
    stage_order: List[str] = []
    stage_to_params: Dict[str, List[torch.nn.Parameter]] = {}
    for name, param in model.backbone.named_parameters():
        stage = stage_key_from_param_name(name)
        if stage not in stage_to_params:
            stage_order.append(stage)
            stage_to_params[stage] = []
        stage_to_params[stage].append(param)

    total_params = sum(len(params) for params in stage_to_params.values())
    print(f"Backbone parameters bucketed into {len(stage_order)} stages (from {total_params} tensors)")

    for params in stage_to_params.values():
        for p in params:
            p.requires_grad = False

    if unfreeze_layers == -1:
        selected = stage_order
    elif unfreeze_layers > 0:
        selected = stage_order[-min(unfreeze_layers, len(stage_order)) :]
    else:
        selected = []

    for stage in selected:
        for p in stage_to_params[stage]:
            p.requires_grad = True

    if selected:
        print(f"Unfreezing last {len(selected)} stage(s): {', '.join(selected)}")
    else:
        print("Keeping backbone frozen (only training head)")

    if hasattr(model, "head"):
        for p in model.head.parameters():
            p.requires_grad = True


class FXClassifier(nn.Module):
    """Wrap SpeciesNet GraphModule with a new classification head."""

    def __init__(self, backbone: nn.Module, num_classes: int, img_size: int, input_layout: str = "nhwc"):
        super().__init__()
        self.backbone = backbone
        self.input_layout = input_layout.lower()

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            if self.input_layout == "nhwc":
                dummy = dummy.permute(0, 2, 3, 1).contiguous()
            features = self.backbone(dummy)
            if features.ndim == 4:
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            elif features.ndim == 3:
                features = features.mean(dim=1)
            else:
                features = features.flatten(1)
            in_features = features.shape[1]

        self.head = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(self.head.weight, gain=0.5)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_layout == "nhwc":
            x = x.permute(0, 2, 3, 1).contiguous()
        features = self.backbone(x)
        if features.ndim == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        elif features.ndim == 3:
            features = features.mean(dim=1)
        else:
            features = features.flatten(1)
        return self.head(features)


class ImageFolderFromFiles(Dataset):
    """Minimal dataset that infers class label from parent folder name."""

    def __init__(self, files: List[Path], class_to_idx: Dict[str, int], transform=None):
        self.files = files
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        label = self.class_to_idx[path.parent.name]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def list_images(directory: Path) -> List[Path]:
    files: List[Path] = []
    for extension in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"):
        files.extend(directory.glob(extension))
    return sorted(files)


def build_datasets(data_root: Path, augment: bool, img_size: int):
    train_root = data_root / "train"
    val_root = data_root / "val"
    if not train_root.exists() or not val_root.exists():
        raise FileNotFoundError("data_root must contain 'train' and 'val' folders")

    classes = sorted([d.name for d in train_root.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError("No class directories found under train/")
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    train_files: List[Path] = []
    for cls in classes:
        cls_dir = train_root / cls
        train_files.extend(list_images(cls_dir))
    if not train_files:
        raise RuntimeError("No training images found")

    val_files: List[Path] = []
    for cls in classes:
        cls_dir = val_root / cls
        val_files.extend(list_images(cls_dir))
    if not val_files:
        raise RuntimeError("No validation images found")

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if augment:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = ImageFolderFromFiles(train_files, class_to_idx, transform=train_transform)
    val_ds = ImageFolderFromFiles(val_files, class_to_idx, transform=eval_transform)
    return train_ds, val_ds, classes


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(1)
    return (preds == targets).float().mean().item()


def save_best_checkpoint(out_dir: Path, state: dict) -> None:
    torch.save(state, out_dir / "best.pt")


def write_metrics(out_dir: Path, history: dict, best_val_loss: float) -> None:
    payload = {"history": history, "best_val_loss": best_val_loss}
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    write_metrics_csv(out_dir, history)
    plot_metrics(out_dir, history)


def write_metrics_csv(out_dir: Path, history: dict) -> None:
    epochs = len(history.get("train_loss", []))
    if epochs == 0:
        return

    rows = [
        [idx + 1, history["train_loss"][idx], history["train_acc"][idx], history["val_loss"][idx], history["val_acc"][idx]]
        for idx in range(epochs)
    ]

    csv_path = out_dir / "metrics.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        writer.writerows(rows)


def plot_metrics(out_dir: Path, history: dict) -> None:
    epochs = len(history.get("train_loss", []))
    if epochs == 0:
        return

    x_axis = list(range(1, epochs + 1))

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(x_axis, history["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(x_axis, history["val_loss"], label="Val Loss", marker="o")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss per Epoch")
    axes[0].grid(True, linestyle="--", alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_axis, [acc * 100 for acc in history["train_acc"]], label="Train Acc", marker="o")
    axes[1].plot(x_axis, [acc * 100 for acc in history["val_acc"]], label="Val Acc", marker="o")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title("Accuracy per Epoch")
    axes[1].grid(True, linestyle="--", alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "metrics.png")
    plt.close(fig)


def train_loop(args) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Using device   : CUDA ({gpu_name})")
    else:
        device = torch.device("cpu")
        print("Using device   : CPU")

    script_dir = Path(__file__).parent
    weights_path = ensure_speciesnet_weights(script_dir)
    fx_backbone = load_fx_checkpoint(weights_path, map_location="cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_size = DEFAULT_IMG_SIZE
    train_ds, val_ds, classes = build_datasets(Path(args.data_root), args.augment, img_size)
    num_classes = len(classes)
    model = FXClassifier(fx_backbone, num_classes=num_classes, img_size=img_size).to(device)

    freeze_backbone_by_stages(model, args.unfreeze_layers)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=len(train_ds) >= args.batch_size * 2,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_loss = math.inf
    best_state = None

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        total_samples = 0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} (train)", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(inputs)
                    loss = F.cross_entropy(logits, targets, label_smoothing=args.label_smoothing)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets, label_smoothing=args.label_smoothing)
                loss.backward()
                optimizer.step()

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_acc += accuracy_from_logits(logits.detach(), targets) * batch_size
            total_samples += batch_size

        train_loss = running_loss / total_samples
        train_acc = running_acc / total_samples

        model.eval()
        val_loss_total = 0.0
        val_acc_total = 0.0
        val_samples = 0
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} (val)", leave=False):
                inputs, targets = inputs.to(device), targets.to(device)
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
                batch_size = inputs.size(0)
                val_loss_total += loss.item() * batch_size
                val_acc_total += accuracy_from_logits(logits, targets) * batch_size
                val_samples += batch_size

        val_loss = val_loss_total / val_samples
        val_acc = val_acc_total / val_samples

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step()

        epoch_time = time.time() - start
        print(
            f"Epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc*100:5.1f}% | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc*100:5.1f}% | "
            f"lr={scheduler.get_last_lr()[0]:.6f}, time={epoch_time:.1f}s"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "epoch": epoch,
                "classes": classes,
                "img_size": img_size,
            }
            save_best_checkpoint(out_dir, best_state)
            print(f"  Saved best checkpoint (epoch {epoch}) to {out_dir / 'best.pt'}")

        write_metrics(out_dir, history, best_val_loss)

    if best_state is not None:
        print(f"Best model selected from epoch {best_state['epoch']} (val_loss={best_val_loss:.4f})")
    else:
        print("No checkpoint saved (validation loss never improved)")

    print(f"Metrics file: {out_dir / 'metrics.json'}")
    print(f"Metrics CSV : {out_dir / 'metrics.csv'}")
    print(f"Metrics plot: {out_dir / 'metrics.png'}")


def parse_args():
    parser = argparse.ArgumentParser("Fine-tune SpeciesNet FX checkpoint directly")
    parser.add_argument("--data_root", type=str, required=True, help="Root with train/ and val/ folders")
    parser.add_argument("--epochs", type=int, default=12, help="Total training epochs (default: 12)")
    parser.add_argument(
        "--unfreeze_layers",
        type=int,
        default=5,
        help="Unfreeze the last N backbone stages (default: 5, 0 keeps backbone frozen, -1 unfreezes all)",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers (default: 2)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for head (default: 5e-5)")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay (default: 1e-4)")
    parser.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing factor (default: 0.05)")
    parser.add_argument("--augment", action="store_true", help="Enable training augmentations (disabled by default)")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store best checkpoint and metrics",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=== SpeciesNet Simple Finetune ===")
    print(f"data_root      : {args.data_root}")
    print(f"output_dir     : {args.output_dir}")
    print(f"epochs         : {args.epochs}")
    print(f"batch_size     : {args.batch_size}")
    print(f"augmentations  : {args.augment}")
    print(f"unfreeze last  : {args.unfreeze_layers} stages")
    print(f"img_size       : {DEFAULT_IMG_SIZE} (fixed)")
    print(f"learning rate  : {args.lr}")
    print(f"weight decay   : {args.weight_decay}")
    print(f"label smoothing: {args.label_smoothing}")
    train_loop(args)


if __name__ == "__main__":
    main()

```

</details>

### Usage

You can use it like this. 

```cmd
python finetune_simple.py --data_root /path/to/dataset --output_dir /tmp/run1
```

### CLI options
This script is intentionally minimal so you can focus on the model’s behavior.

```bash
Usage: Fine-tune SpeciesNet FX checkpoint directly [-h] --data_root DATA_ROOT [--epochs EPOCHS]
                                                   [--unfreeze_layers UNFREEZE_LAYERS] [--batch_size BATCH_SIZE]
                                                   [--num_workers NUM_WORKERS] [--lr LR] [--weight_decay WEIGHT_DECAY]
                                                   [--label_smoothing LABEL_SMOOTHING] [--augment] --output_dir
                                                   OUTPUT_DIR

options:
  -h, --help            show this help message and exit
  --data_root DATA_ROOT
                        Root with train/ and val/ folders
  --epochs EPOCHS       Total training epochs (default: 12)
  --unfreeze_layers UNFREEZE_LAYERS
                        Unfreeze the last N backbone stages (default: 5, 0 keeps backbone frozen, -1 unfreezes all)
  --batch_size BATCH_SIZE
                        Batch size (default: 8)
  --num_workers NUM_WORKERS
                        DataLoader workers (default: 2)
  --lr LR               Learning rate for head (default: 5e-5)
  --weight_decay WEIGHT_DECAY
                        AdamW weight decay (default: 1e-4)
  --label_smoothing LABEL_SMOOTHING
                        Label smoothing factor (default: 0.05)
  --augment             Enable training augmentations (disabled by default)
  --output_dir OUTPUT_DIR
                        Directory to store best checkpoint and metrics
```
