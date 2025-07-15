# Script to further identify MD animal detections using PT classification models
# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI.

# It is designed for models developed by the San Diego Zoo Wildlife Alliance fork 
# https://github.com/PetervanLunteren/animl-py

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
import pandas as pd
import torch.nn as nn
from PIL import ImageOps
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet, convnext_base, ConvNeXt_Base_Weights

# lets not freak out over truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

# init model architectures
class EfficientNetV2M(nn.Module):
    def __init__(self, num_classes, tune=True):
        super(EfficientNetV2M, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model = efficientnet.efficientnet_v2_m(weights=None)
        if tune:
            for params in self.model.parameters():
                params.requires_grad = True
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    def forward(self, x):
        x = self.model.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        prediction = self.model.classifier(x) 
        return prediction
    
class EfficientNetV2S(nn.Module):
    def __init__(self, num_classes, tune=True):
        super(EfficientNetV2S, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model = efficientnet.efficientnet_v2_s(weights=None)
        if tune:
            for params in self.model.parameters():
                params.requires_grad = True
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    def forward(self, x):
        x = self.model.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        prediction = self.model.classifier(x)
        return prediction
    
class ConvNeXtBase(nn.Module):
    def __init__(self, num_classes, tune=True):
        super(ConvNeXtBase, self).__init__()
        self.model = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
        if not tune:
            for param in self.model.parameters():
                param.requires_grad = False
        num_ftrs = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
    def forward(self, x):
        '''
        Forward pass (prediction).
        '''
        return self.model(x)

# load model
checkpoint = torch.load(cls_model_fpath, map_location=torch.device(device_str))
image_size = tuple(checkpoint['image_size'])
architecture = checkpoint['architecture']
categories = checkpoint['categories']
classes = list(categories.keys())
if architecture == 'efficientnet_v2_m':
    model = EfficientNetV2M(len(classes), tune=False)
elif architecture == 'efficientnet_v2_s':
    model = EfficientNetV2S(len(classes), tune=False)
elif architecture == 'convnext_base':
    model = ConvNeXtBase(len(classes), tune=False)
model.load_state_dict(checkpoint['model'])
model.to(torch.device(device_str))
model.eval()
device = torch.device(device_str)

# image preprocessing 
preprocess = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in inference_lib.py
def get_classification(PIL_crop):
    input_tensor = preprocess(PIL_crop)
    input_batch = input_tensor.unsqueeze(0)  
    input_batch = input_batch.to(device)
    output = model(input_batch)
    probabilities = F.softmax(output, dim=1)
    probabilities_np = probabilities.cpu().detach().numpy()
    confidence_scores = probabilities_np[0]
    classifications = []
    for i in range(len(confidence_scores)):
        pred_class = classes[i]
        pred_conf = confidence_scores[i]
        classifications.append([pred_class, pred_conf])
    return classifications

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
def get_crop(img, bbox_norm): # created by Dan Morris
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)
    box_size = max(box_w, box_h)
    box_size = pad_crop(box_size)
    xmin = max(0, min(
        xmin - int((box_size - box_w) / 2),
        img_w - box_w))
    ymin = max(0, min(
        ymin - int((box_size - box_h) / 2),
        img_h - box_h))
    box_w = min(img_w, box_size)
    box_h = min(img_h, box_size)
    if box_w == 0 or box_h == 0:
        return
    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])
    crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)
    return crop

# make sure small animals are not overly enlarged
def pad_crop(box_size):
    input_size_network = 224
    default_padding = 30
    diff_size = input_size_network - box_size
    if box_size >= input_size_network:
        box_size = box_size + default_padding
    else:
        if diff_size < default_padding:
            box_size = box_size + default_padding
        else:
            box_size = input_size_network    
    return box_size

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
