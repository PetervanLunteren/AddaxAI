# Script to identify MD animal detections using a fine-tuned ResNet18 trained on
# Gorongosa National Park camera trap images. Architecture is the standard
# CV4Ecology CustomResNet18: torchvision resnet18 backbone with a fresh
# Linear classifier head.
#
# Source pipeline: https://github.com/... (kaitlyn_catalyst)
# It consists of code that is specific for this kind of model architecture, and
# code that is generic for all model architectures that will be run via AddaxAI.
#
# Written by Peter van Lunteren
# Latest edit by Peter van Lunteren on 29 Apr 2026

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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18

# CustomResNet18 architecture (mirrors ct_classifier/model.py from training repo)
class CustomResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CustomResNet18, self).__init__()
        # weights=None: ImageNet weights are not needed at inference, the trained
        # checkpoint overwrites them anyway. Avoids a download at startup.
        self.feature_extractor = resnet18(weights=None)
        last_layer = self.feature_extractor.fc
        in_features = last_layer.in_features
        self.feature_extractor.fc = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.feature_extractor(x)
        prediction = self.classifier(features)
        return prediction

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
device = torch.device(device_str)

# load checkpoint and instantiate model
# checkpoint format (from training.py):
#   {"epoch", "model_state", "optimizer_state", "class_names", "config"}
checkpoint = torch.load(cls_model_fpath, map_location=device)
class_names = checkpoint['class_names']
model = CustomResNet18(num_classes=len(class_names))
model.load_state_dict(checkpoint['model_state'])
model.to(device)
model.eval()

# image preprocessing — must match training-time transform exactly
# training pipeline uses ImageNet stats; see speciesnet/inference.py
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['elephant', 0.93], ['buffalo', 0.01], ... ]
# no need to remove forbidden classes from the predictions, that will happen in inference_lib.py
def get_classification(PIL_crop):
    input_tensor = preprocess(PIL_crop)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_batch)
        probabilities = F.softmax(output, dim=1)
    confidence_scores = probabilities.cpu().numpy()[0]
    classifications = []
    for i in range(len(confidence_scores)):
        classifications.append([class_names[i], float(confidence_scores[i])])
    return classifications

# method of removing background
# input: image = full image PIL.Image.open(img_fpath)
# input: bbox_norm = MD bbox as [x, y, w, h] in normalized coordinates
# output: cropped image (PIL.Image)
# 10% padding around the MD bbox, matching speciesnet/detector.py:_crop_with_padding
# used during training (pad_frac=0.10).
def get_crop(img, bbox_norm):
    # match training: crops were loaded with Image.open(...).convert("RGB")
    if img.mode != "RGB":
        img = img.convert("RGB")
    pad_frac = 0.10
    W, H = img.size
    x, y, w, h = bbox_norm

    x1 = int(round(x * W))
    y1 = int(round(y * H))
    x2 = int(round((x + w) * W))
    y2 = int(round((y + h) * H))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(round(bw * pad_frac))
    pad_y = int(round(bh * pad_frac))

    x1 = max(0, min(x1 - pad_x, W))
    y1 = max(0, min(y1 - pad_y, H))
    x2 = max(0, min(x2 + pad_x, W))
    y2 = max(0, min(y2 + pad_y, H))

    if x2 <= x1 or y2 <= y1:
        return img
    return img.crop((x1, y1, x2, y2))

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
