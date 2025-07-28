# Script to further identify MD animal detections using classification model
# developed by Andrea Zampetti.

# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI
# Script written by Peter van Lunteren and Andrea Zampetti.
# Latest edit by Peter van Lunteren on 28 Jul 2025.

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
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

# if on macOS Apple Silicon, disable GPU and Metal backend
if os.name == 'posix' and 'darwin' in os.uname().sysname.lower():
    tf.config.set_visible_devices([], 'GPU')

# this is a saved model, so we need the directory, not the file path
cls_model_dir_fpath = os.path.dirname(cls_model_fpath) 

# Load model and taxonomy
model = tf.keras.models.load_model(cls_model_dir_fpath)
taxonomy_path = os.path.join(os.path.dirname(cls_model_fpath), "taxon-mapping.csv")
taxonomy_df = pd.read_csv(taxonomy_path)
class_names = taxonomy_df['model_class'].tolist()

# check GPU availability (tensorflow does support the GPU on Windows Native)
GPU_availability = True if len(tf.config.list_logical_devices('GPU')) > 0 else False

# preprocess function for crops
def preprocess_crop(cropped_img, target_size=(224, 224)):
    img_resized = cropped_img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = tf.keras.applications.convnext.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in infrence_lib.py
def get_classification(PIL_crop):
    img_array = preprocess_crop(PIL_crop)
    prediction = model.predict(img_array, verbose=0)[0]
    classifications = []
    for i in range(len(prediction)):
        classifications.append([class_names[i], float(prediction[i])])
    return classifications

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
# mewc snipping method taken from https://github.com/zaandahl/mewc-snip/blob/main/src/mewc_snip.py#L29
# which points to this MD util https://github.com/agentmorris/MegaDetector/blob/main/megadetector/visualization/visualization_utils.py#L352
# the function below is rewritten for a single image input without expansion
def get_crop(img, bbox):
    w, h = img.size
    left, top = int(bbox[0] * w), int(bbox[1] * h)
    right, bottom = int((bbox[0] + bbox[2]) * w), int((bbox[1] + bbox[3]) * h)
    return img.crop((max(0, left), max(0, top), min(w, right), min(h, bottom)))

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
