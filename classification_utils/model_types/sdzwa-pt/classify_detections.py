# Script to further identify MD animal detections using PT classification models trained by SDZWA
# SDZWA - San Diego Zoo Wildlife Alliance - The Conservation Technology Lab
# https://github.com/conservationtechlab/animl-py

# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI.

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
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet

# set paths
efficientnet_pth_fpath = os.path.join(os.path.dirname(cls_model_fpath), 'efficientnet_v2_m-dc08266a.pth')
class_csv_fpath = os.path.join(os.path.dirname(cls_model_fpath), 'classes.csv')

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

print("debug 10 sept") # DEBUG
print(device_str)
print(GPU_availability)


# init efficientnet
class EfficientNet(nn.Module):

    def __init__(self, num_classes, tune=True):
        '''
            Construct the model architecture.
        '''
        super(EfficientNet, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model = efficientnet.efficientnet_v2_m(weights=None)
        self.model.load_state_dict(torch.load(efficientnet_pth_fpath, map_location=torch.device(device_str)))
        if tune:
            for params in self.model.parameters():
                params.requires_grad = True
        if tune:
            for params in self.model.parameters():
                params.requires_grad = True
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=num_classes)
        self.model.to(torch.device(device_str))

    def forward(self, x):
        '''
            Forward pass (prediction)
        '''
        x = self.model.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        prediction = self.model.classifier(x)
        return prediction

# load model
classes = pd.read_csv(class_csv_fpath)
model = EfficientNet(len(classes), tune=False)
checkpoint = torch.load(cls_model_fpath, map_location=torch.device(device_str))
model.load_state_dict(checkpoint['model'])
model.to(torch.device(device_str))
model.eval()
model.framework = "EfficientNet"
device = torch.device(device_str)

# image preprocessing 
# according to https://github.com/conservationtechlab/animl-py/blob/a9d1a2d0a40717f1f8346cbf9aca35161edc9a6e/src/animl/generator.py#L175
preprocess = transforms.Compose([
    transforms.Resize((299, 299)),
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
        pred_class = classes.iloc[i].values[1]
        pred_conf = confidence_scores[i]
        classifications.append([pred_class, pred_conf])
    return classifications

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
# I've pulled this crop function from
# https://github.com/conservationtechlab/animl-py/blob/a9d1a2d0a40717f1f8346cbf9aca35161edc9a6e/src/animl/generator.py#L135
def get_crop(img, bbox_norm):
    buffer = 0 
    width, height = img.size
    bbox1, bbox2, bbox3, bbox4 = bbox_norm
    left = width * bbox1
    top = height * bbox2
    right = width * (bbox1 + bbox3)
    bottom = height * (bbox2 + bbox4)
    left = max(0, int(left) - buffer)
    top = max(0, int(top) - buffer)
    right = min(width, int(right) + buffer)
    bottom = min(height, int(bottom) + buffer)
    image_cropped = img.crop((left, top, right, bottom))
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
