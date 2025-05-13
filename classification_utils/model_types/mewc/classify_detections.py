# Script to further identify MD animal detections using classification models trained via MEWC
# MEWC - Mega Efficient Wildlife Classifier - University of Tasmania
# https://github.com/zaandahl/mewc

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
animal_model = load_model(cls_model_fpath, custom_objects={'loss': tfa.losses.SigmoidFocalCrossEntropy()})
img_size = animal_model.get_config()["layers"][0]["config"]["batch_input_shape"][1]

# check GPU availability
GPU_availability = True if len(tf.config.list_logical_devices('GPU')) > 0 else False

# read label map
def read_yaml(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)
class_map = read_yaml(os.path.join(os.path.dirname(cls_model_fpath), "class_list.yaml"))
inv_class = {v: k for k, v in class_map.items()}
class_ids = sorted(inv_class.values())

# apparently the class_list.yaml can be formatted differently
def can_all_keys_be_converted_to_int(d):
    for key in d.keys():
        try:
            int(key)
        except ValueError:
            return False
    return True

# take note how it is formatted
if not can_all_keys_be_converted_to_int(class_map):
    formatted_int_label = False
else:
    formatted_int_label = True

# extra processing step if the class_list is formatted as int:label
if formatted_int_label:    
    class_ids = [class_map[i] for i in sorted(inv_class.values())]

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in infrence_lib.py
def get_classification(PIL_crop):
    img = np.array(PIL_crop)
    img = cv2.resize(img, (img_size, img_size))
    img = np.expand_dims(img, axis=0)
    pred = animal_model.predict(img, verbose=0)[0]
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
# thanks Dan Morris: https://github.com/agentmorris/MegaDetector/blob/main/md_visualization/visualization_utils.py
def get_crop(image, bbox): 
    x1, y1, w_box, h_box = bbox
    ymin,xmin,ymax,xmax = y1, x1, y1 + h_box, x1 + w_box
    im_width, im_height = image.size
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                    ymin * im_height, ymax * im_height)
    left = max(left,0); right = max(right,0)
    top = max(top,0); bottom = max(bottom,0)
    left = min(left,im_width-1); right = min(right,im_width-1)
    top = min(top,im_height-1); bottom = min(bottom,im_height-1)
    image_cropped = image.crop((left, top, right, bottom))
    # resizing will be done in get_classification()
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
