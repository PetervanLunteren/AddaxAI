# Script to further identify MD animal detections using classification models trained by SDZWA
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
import cv2
import yaml
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model

# load model
animal_model = load_model(cls_model_fpath)
img_size = animal_model.get_config()["layers"][0]["config"]["batch_input_shape"][1]

# check GPU availability
GPU_availability = True if len(tf.config.list_logical_devices('GPU')) > 0 else False

# read label map
class_map = {}
with open(os.path.join(os.path.dirname(cls_model_fpath), "Peru-Amazon_0.86.txt"), 'r') as file:
    for line in file:
        parts = line.strip().split('"')
        if len(parts) >= 4:
            identifier = parts[1].strip()
            animal_name = parts[3].strip()
            if identifier.isdigit():
                class_map[str(identifier)] = str(animal_name)

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in infrence_lib.py
def get_classification(PIL_crop):
    img = np.array(PIL_crop)
    img = cv2.resize(img, (img_size, img_size))
    img = np.expand_dims(img, axis=0)
    
    # According to https://github.com/conservationtechlab/animl-py/blob/a9d1a2d0a40717f1f8346cbf9aca35161edc9a6e/src/animl/generator.py#L175
    # there are no particular preprocessing steps or augmentations to handle prior to inference except for horizontal flip.
    
    pred = animal_model.predict(img, verbose=0)[0]
    class_ids = sorted(class_map.values())
    classifications = []
    for i in range(len(pred)):
        classifications.append([class_ids[i], float(pred[i])])
    return classifications

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
# I've pulled this crop function from
# https://github.com/conservationtechlab/animl-py/blob/a9d1a2d0a40717f1f8346cbf9aca35161edc9a6e/src/animl/generator.py#L135
def get_crop(img, bbox): 
    buffer = 0 
    width, height = img.size
    bbox1, bbox2, bbox3, bbox4 = bbox
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
