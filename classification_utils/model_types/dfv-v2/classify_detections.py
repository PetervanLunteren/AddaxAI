# Script to further identify MD animal detections using the DeepForestVision classification model v2.
# https://www.oneforestvision.org
# https://github.com/MNHN-OFVI/DeepForestVisionV2
# It consists of code that is specific for this kind of model architecture, and
# code that is generic for all model architectures that will be run via AddaxAI.

# Script created by Peter van Lunteren
# Some code is created by the DeepForestVision team and is indicated as so
# Latest edit by Peter van Lunteren on 18 Jun 2026

# DeepForestVision is developed under CC BY-NC-SA 4.0 license
# (https://creativecommons.org/licenses/by-nc-sa/4.0) by an academic team from the
# French Muséum National d'Histoire Naturelle (MNHN) as part of the One Forest Vision
# initiative (https://www.oneforestvision.org).

# hugo.magaldi@mnhn.fr; sabrina.krief@mnhn.fr

# NOTE: v2 swaps the dinov2 backbone for a dinov3 ViT-B/16. The backbone architecture
# is not available via pip/transformers, so the dinov3 repo code is downloaded
# alongside the weights (see download_info in model_info) and loaded locally with
# torch.hub. The fine-tuned weights (backbone + head) live in the checkpoint, so the
# backbone is built WITHOUT its own pretrained weights.

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

##############################################
############### MODEL SPECIFIC ###############
##############################################
# imports
import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms as T

# ignore warnings about beta transforms in torchvision
try:
    import torchvision
    torchvision.disable_beta_transforms_warning()
except Exception:
    pass

# check on and on which GPU the process should run
def fetch_device():
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    try:
        if torch.backends.mps.is_built and torch.backends.mps.is_available():
            device = torch.device('mps')
    except AttributeError:
        pass
    return device

################################################
############## CLASSIFTOOLS START ##############
################################################
# Code below adapted from the DeepForestVisionV2 inference script provided by the
# DeepForestVision team (classification.py). Adjustments for AddaxAI are marked.

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def build_val_transform():
    return T.Compose(
        [
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )

class DinoV3Head(nn.Module):
    """
    Backbone = DINOv3 ViT.
    Head input = concat([CLS], mean(patch_tokens)) -> dim = 2 * embed_dim
    """
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        embed_dim = backbone.embed_dim  # DINOv3 ViT exposes embed_dim
        self.classifier = (
            nn.Linear(embed_dim * 2, num_classes) if num_classes > 0 else nn.Identity()
        )

    def forward(self, pixel_values):
        # get_intermediate_layers returns a list; take last requested layer
        seq = self.backbone.get_intermediate_layers(pixel_values, n=1)[0]  # [B, 1+N, D]
        cls_token = seq[:, 0]          # [B, D]
        patch_tokens = seq[:, 1:]      # [B, N, D]
        pooled_patches = patch_tokens.mean(dim=1)  # [B, D]
        x = torch.cat([cls_token, pooled_patches], dim=1)  # [B, 2D]
        logits = self.classifier(x)
        return logits

# ADJUSTMENT (AddaxAI): the dinov3 backbone architecture is not packaged on pip, so the
# dinov3 repo code is shipped next to the checkpoint and loaded locally via torch.hub.
# It may arrive as a zip (dinov3.zip) that we extract on first run.
def locate_dinov3_repo_dir(model_dir):
    # already extracted? find the dir that holds hubconf.py
    for root, _dirs, files in os.walk(model_dir):
        if "hubconf.py" in files:
            return root
    # not yet extracted, try the zip
    zip_path = os.path.join(model_dir, "dinov3.zip")
    if os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(model_dir)
        for root, _dirs, files in os.walk(model_dir):
            if "hubconf.py" in files:
                return root
    raise FileNotFoundError(
        f"Could not find the dinov3 repo code (hubconf.py) in {model_dir}. "
        f"Expected a 'dinov3.zip' or an extracted dinov3 folder next to the checkpoint."
    )

# ADJUSTMENT (AddaxAI): build the dinov3 ViT-B/16 backbone WITHOUT its own pretrained
# weights (the checkpoint provides the full fine-tuned state dict, loaded strict=True).
def build_backbone(dinov3_repo_dir, device):
    for kwargs in ({"pretrained": False}, {"weights": None}, {}):
        try:
            return torch.hub.load(
                str(dinov3_repo_dir), "dinov3_vitb16", source="local", **kwargs
            ).to(device)
        except TypeError:
            continue
    # last resort: let it raise the real error
    return torch.hub.load(
        str(dinov3_repo_dir), "dinov3_vitb16", source="local"
    ).to(device)

class DinoClassifier:
    def __init__(self, model, labels, device, transform):
        self.model = model
        self.labels = labels
        self.device = device
        self.transform = transform

    @torch.inference_mode()
    def predict_proba(self, img):
        x = self.transform(img).unsqueeze(0).to(self.device)
        logits = self.model(x)
        proba = torch.softmax(logits, dim=-1).detach().cpu().tolist()[0]
        return proba

def load_dino_classifier(checkpoint_path, device):
    """
    Loads:
      - checkpoint with 'labels' + 'model_state_dict'
      - wraps with CLS+mean-pool head (DinoV3Head)
    """
    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    labels = list(checkpoint["labels"])
    num_classes = len(labels)

    model_dir = os.path.dirname(str(checkpoint_path))
    dinov3_repo_dir = locate_dinov3_repo_dir(model_dir)
    backbone = build_backbone(dinov3_repo_dir, device)

    model = DinoV3Head(backbone=backbone, num_classes=num_classes).to(device)
    state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return DinoClassifier(
        model=model,
        labels=labels,
        device=device,
        transform=build_val_transform(),
    )

##############################################
############## CLASSIFTOOLS END ##############
##############################################

# load model
device = fetch_device()
classifier = load_dino_classifier(cls_model_fpath, device)

# check GPU availability
GPU_availability = False
try:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        GPU_availability = True
except:
    pass
if not GPU_availability:
    GPU_availability = torch.cuda.is_available()

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardvark', 2.3e-09], ['baboon', 5.6e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in inference_lib.py
# this is also the place to preprocess the image if that need to happen
def get_classification(PIL_crop):
    PIL_crop = PIL_crop.convert('RGB')
    confs = classifier.predict_proba(PIL_crop)
    lbls = classifier.labels
    classifications = []
    for i in range(len(confs)):
        classifications.append([lbls[i], confs[i]])
    return classifications

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
# DeepForestVisionV2 does NOT square-pad: it crops the plain detection bbox clipped to the
# image bounds, and the val transform then does Resize(256) + CenterCrop(224). This matches
# save_cropped_images_from_pw_results() in their deepforestvision/detection_utils.py.
# MD bbox is normalised [xmin, ymin, width, height].
def get_crop(image, bbox):
    width, height = image.size
    xmin = bbox[0] * width
    ymin = bbox[1] * height
    xmax = xmin + bbox[2] * width
    ymax = ymin + bbox[3] * height
    # clip to image bounds (prevents PIL errors / empty crops)
    xmin = max(0.0, min(xmin, width - 1.0))
    ymin = max(0.0, min(ymin, height - 1.0))
    xmax = max(1.0, min(xmax, float(width)))
    ymax = max(1.0, min(ymax, float(height)))
    image_cropped = image.crop((xmin, ymin, xmax, ymax))
    return image_cropped

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
