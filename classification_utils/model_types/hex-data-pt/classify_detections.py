# Script to further identify MD animal detections using the OSI-Panthera classification model developed by the Hex-Data team
# https://www.hex-data.io/
# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI.

# Script created by Peter van Lunteren
# Some code is created by the Hex-Data team and is indicated as so 
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
import torch
from torchvision import transforms
import pickle
import os

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
        # the model is not compatible with Metal Performance Shaders (MPS)
        # backend, which is used for Apple Silicon GPUs, so we'll set it to CPU
        device_str = 'cpu' 
except:
    pass
if not GPU_availability:
    if torch.cuda.is_available():
        GPU_availability = True
        device_str = 'cuda'
device = torch.device(device_str)

# load model
model = torch.jit.load(cls_model_fpath, map_location=device)
model.eval()

# read label map
class_pickle_fpath = os.path.join(os.path.dirname(cls_model_fpath), 'classes_Fri_Sep__1_18_50_55_2023.pickle')
with open(class_pickle_fpath, "rb") as f:
    class_labels = pickle.load(f)

# define image transforms
img_resize = 316
def get_dev_transform():
    return transforms.Compose([
        transforms.Resize([img_resize, img_resize]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in infrence_lib.py
def get_classification(PIL_crop):
    transform = get_dev_transform()
    PIL_crop = transform(PIL_crop)
    PIL_crop = PIL_crop.unsqueeze(0)
    PIL_crop = PIL_crop.to(device)
    with torch.no_grad():
        output = model(PIL_crop)
    softmax_output = torch.nn.functional.softmax(output, dim=1)
    predictions = []
    for idx, prob in enumerate(softmax_output[0]):
        class_label = class_labels[idx]
        confidence = prob.item()
        predictions.append([class_label, confidence])
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
def get_crop(img, bbox_norm):
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    xmax = xmin + int(bbox_norm[2] * img_w)
    ymax = ymin + int(bbox_norm[3] * img_h)
    crop = img.crop(box=[xmin, ymin, xmax, ymax])
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
