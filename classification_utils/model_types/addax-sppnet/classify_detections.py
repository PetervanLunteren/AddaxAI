# Script to further identify MD animal detections using PT classification models
# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI.

# It is designed for models finetuned on the FXClassifier framework of SpeciesNet
# https://github.com/google/cameratrapai

# Written by Peter van Lunteren
# Latest edit by Peter van Lunteren on 13 May 2025

#############################################
############### MODEL GENERIC ###############
#############################################
# catch shell arguments
import sys
AddaxAI_files = str(sys.argv[1])
cls_model_fpath = str(sys.argv[2])
cls_detec_thresh = float(sys.argv[3])
cls_class_thresh = float(sys.argv[4])
smooth_bool = True if sys.argv[5] == 'True' else False
json_path = str(sys.argv[6])
temp_frame_folder =  None if str(sys.argv[7]) == 'None' else str(sys.argv[7])
cls_tax_fallback = True if sys.argv[8] == 'True' else False
cls_tax_levels_idx = int(sys.argv[9])

# lets not freak out over truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

##############################################
############### MODEL SPECIFIC ###############
##############################################
# imports
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from pathlib import Path

# make sure windows trained models work on unix too
import pathlib
import platform
plt = platform.system()
if plt != 'Windows': pathlib.WindowsPath = pathlib.PosixPath

# check GPU availability
GPU_availability = False
device_str = 'cpu'
try:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        GPU_availability = True
        device_str = 'mps'
except:
    pass
if not GPU_availability:
    if torch.cuda.is_available():
        GPU_availability = True
        device_str = 'cuda'

# ============================================================================
# Model Architecture (copy from finetune.py)
# ============================================================================

def load_fx_checkpoint(weights_path, map_location="cpu"):
    """Load SpeciesNet onnx2torch GraphModule"""
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
    raise ValueError("Expected a torch.nn.Module GraphModule")


class FXClassifier(nn.Module):
    """Wrapper for SpeciesNet backbone + custom head"""

    def __init__(self, backbone: nn.Module, num_classes: int, img_size: int = 480, input_layout: str = "nhwc"):
        super().__init__()
        self.backbone = backbone
        self.input_layout = input_layout.lower()

        for p in self.backbone.parameters():
            p.requires_grad = False

        self.backbone.eval()
        with torch.no_grad():
            x = torch.zeros(1, 3, img_size, img_size)
            if self.input_layout == "nhwc":
                x = x.permute(0, 2, 3, 1).contiguous()
            z = self.backbone(x)

            if z.ndim == 4:
                z = F.adaptive_avg_pool2d(z, 1).flatten(1)
            elif z.ndim == 3:
                z = z.mean(dim=1)
            else:
                z = z.flatten(1)

            in_features = z.shape[1]

        self.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        if self.input_layout == "nhwc":
            x = x.permute(0, 2, 3, 1).contiguous()

        z = self.backbone(x)

        if z.ndim == 4:
            z = F.adaptive_avg_pool2d(z, 1).flatten(1)
        elif z.ndim == 3:
            z = z.mean(dim=1)
        else:
            z = z.flatten(1)

        return self.head(z)


# ============================================================================
# Load Model (run this once at startup)
# ============================================================================

# Paths
BEST_PT = Path(cls_model_fpath)
BACKBONE_WEIGHTS = Path(os.path.join(os.path.dirname(BEST_PT), "always_crop_99710272_22x8_v12_epoch_00148.pt"))

# Device
device = torch.device(device_str)
print(f"Using device: {device}")

# Load checkpoint
print("Loading model...")
try:
    checkpoint = torch.load(BEST_PT, map_location=device, weights_only=True)
except Exception:
    checkpoint = torch.load(BEST_PT, map_location=device, weights_only=False)

# Load backbone
backbone = load_fx_checkpoint(BACKBONE_WEIGHTS, map_location="cpu")

# Create model
model = FXClassifier(
    backbone=backbone,
    num_classes=checkpoint["num_classes"],
    img_size=checkpoint["img_size"],
    input_layout=checkpoint["input_layout"]
)

# Load fine-tuned weights
model.load_state_dict(checkpoint["model"])

# move entire model to device and set eval mode
model = model.to(device)
model.eval()

# Get class names
class_names = checkpoint["class_names"]

# Create transform
norm_params = checkpoint["normalize"]
preprocess = transforms.Compose([
    transforms.Resize((checkpoint["img_size"], checkpoint["img_size"]), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=norm_params["mean"], std=norm_params["std"])
])

print(f"Model loaded with {len(class_names)} classes")

# ============================================================================
# Inference Function
# ============================================================================

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in inference_lib.py
def get_classification(PIL_crop):
    
    # Ensure RGB
    if isinstance(PIL_crop, Image.Image) and PIL_crop.mode != "RGB":
        PIL_crop = PIL_crop.convert("RGB")
    
    # Preprocess image
    input_tensor = preprocess(PIL_crop)
    input_batch = input_tensor.unsqueeze(0)
    input_batch = input_batch.to(device)

    # Run inference
    with torch.no_grad():
        output = model(input_batch)
        probabilities = F.softmax(output, dim=1)

    # Convert to numpy
    probabilities_np = probabilities.cpu().detach().numpy()
    confidence_scores = probabilities_np[0]

    # Format as list of [class_name, probability]
    classifications = []
    for i in range(len(confidence_scores)):
        pred_class = class_names[i]
        pred_conf = float(confidence_scores[i])
        classifications.append([pred_class, pred_conf])

    return classifications

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [x, y, width, height]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
def get_crop(img: Image.Image, bbox_norm):
    W, H = img.size
    x, y, w, h = bbox_norm

    # Convert to pixel coords
    left   = max(0, int(round(x * W)))
    top    = max(0, int(round(y * H)))
    right  = min(W, int(round((x + w) * W)))
    bottom = min(H, int(round((y + h) * H)))

    # Guard against degenerate boxes
    if right <= left or bottom <= top:
        # Fall back to full image
        crop = img
    else:
        crop = img.crop((left, top, right, bottom))

    # Return
    return crop

#############################################
############### MODEL GENERIC ###############
#############################################
# run main function
import AddaxAI.classification_utils.inference_lib as ea
ea.classify_MD_json(json_path = json_path,
                    GPU_availability = GPU_availability,
                    cls_detec_thresh = cls_detec_thresh,
                    cls_class_thresh = cls_class_thresh,
                    smooth_bool = smooth_bool,
                    crop_function = get_crop,
                    inference_function = get_classification,
                    temp_frame_folder = temp_frame_folder,
                    cls_model_fpath = cls_model_fpath,
                    cls_tax_fallback = cls_tax_fallback,
                    cls_tax_levels_idx = cls_tax_levels_idx)
