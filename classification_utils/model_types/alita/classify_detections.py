# Script to further identify MegaDetector animal detections using the Alita classification model
# The model was created by Olly Powell  https://wekaResearch.com
# The dataset used for training was curated by Jorris Tinnemans at the Department of Conservation, (Aotearoa-New Zealand)
# https://www.doc.govt.nz/
# The dataset is freely available from LILA BC:  https://lila.science/datasets/nz-trailcams
# All source code for this project avaliable at: https://https://github.com/Wologman/Alita

# Script created by Peter van Lunteren
# The Model specific parts created by Olly
# Latest edit by Peter van Lunteren on 3 Jul 2025
# Subsequent edit by Olly on 7 Jul 2025

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
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import timm
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Tuple, List


################################################
############## CLASSIFTOOLS START ##############
################################################


@dataclass
class ImageConfig:
    '''Wrapper class for image processing parameters'''
    EDGE_FADE: bool = False
    BUFFER: int = 0
    RESIZE_METHOD: Literal['md_crop', 'rescale'] = 'md_crop' #MD crop preferred
    MD_RESAMPLE: bool = True #True is preferred
    IMAGE_SIZE: int = 480 #The final size of the second crop, made during the augmentation step
    CROP_SIZE: int = 600 #This is the size that the images will be cropped to as part of the initial bounding box crop
    INPUT_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406) #Values from ImageNet.
    INPUT_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)  #Values from ImageNet.


class ImageTransformer():
    '''Class to handle the various image transformations'''

    def __init__(self, cfg):     
        self.image_size = cfg.IMAGE_SIZE
        self.crop_size = cfg.CROP_SIZE
        self.buffer = cfg.BUFFER
        self.resize_method = cfg.RESIZE_METHOD
        self.downsample = cfg.MD_RESAMPLE
        self.std = cfg.INPUT_STD
        self.mean = cfg.INPUT_MEAN
        self.transform = A.Compose([A.CenterCrop(height=cfg.IMAGE_SIZE, width=cfg.IMAGE_SIZE, p=1),
                                    A.Normalize(mean=self.mean, std=self.std), ToTensorV2()])
    
    def load_image(self, image_path, mode='RGB'):
        try:
            with open(image_path, 'rb') as in_file:
                image = cv2.imread(image_path)
                if image is not None:
                    if mode == 'RGB':
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except:
            print(f"Warning: Unable to load the image at '{image_path}'. Skipping...")
            return None
    
    def crop_to_square(self, image):
        '''Centre-crops a rectangle to a square'''
        height, width = image.shape[:2]
        min_side_length = min(height, width)
        top = (height - min_side_length) // 2
        bottom = top + min_side_length
        left = (width - min_side_length) // 2
        right = left + min_side_length
        return image[top:bottom, left:right]

    def pad_to_square(self, image):
        '''Padds a rectangle to a square by adding zeros to the short side'''
        height, width, channels = image.shape
        dtype = image.dtype
        max_dim = max(height, width, self.crop_size)
        square_image = np.zeros((max_dim, max_dim, channels), dtype=dtype)
        y_offset = (max_dim - height) // 2
        x_offset = (max_dim - width) // 2
        square_image[y_offset:y_offset+height, x_offset:x_offset+width]= image
        return square_image, x_offset, y_offset

    def rescale_image(self, image_arr):
        '''Rescales any large images down to crop_size'''
        size=self.crop_size
        crop = self.crop_to_square(image_arr)
        return cv2.resize(crop, (size, size))
    
    def get_mega_crop_values(self,
                             bbox: List[float],
                             img_w: int,
                             img_h: int,
                             final_size: int):
        '''
        Megadetector bbox is [x_min, y_min, width_of_box, height_of_box] top left (normalised COCO)
        We want to output a square centred on the old box, with width & height = final_size
        Function returns absolute pixel indices in the form: Left, Top, Right, Bottom
        '''
        x_min, y_min, width, height = bbox[0], bbox[1], bbox[2], bbox[3]

        x_centre = (x_min + width/2) * img_w
        y_centre = (y_min + height/2) * img_h
        left = int(x_centre - final_size/2)
        top =  int(y_centre - final_size/2)
        right = left + final_size
        bottom = top + final_size

        # Corrections for when the box is out of the original image dimensions. Shifts by that amount
        if (left < 0) and (right > img_w):
            new_left, new_right = 0, final_size
        else:
            new_left   = left  - (left < 0) * left - (right > img_w)*(right - img_w)
            new_right  = right - (left < 0) * left - (right > img_w)*(right - img_w)
        
        if (top < 0) and (bottom > img_h):
            new_top, new_bottom = 0, final_size
        else:
            new_top    = top    - (top < 0) * top - (bottom > img_h) * (bottom - img_h)
            new_bottom = bottom - (top < 0) * top - (bottom > img_h) * (bottom - img_h)
        
        return new_left, new_top, new_right, new_bottom


    def get_new_scale(self,
                      bbox: List[float],
                      buffer: float,
                      width: int,
                      height: int,
                      final_size: int):
        '''
        Calculates how much to scale down the new image to, 
        so the max(bounding-box) + buffer = the desired crop size.
        Only effects images where the crop box would be greater than the crop size.
        '''
        clamp = lambda n: max(min(1, n), 0)
         # Megadetector output is [x_min, y_min, width_of_box, height_of_box] top left (normalised COCO)
        x_min = clamp(bbox[0] - buffer)
        y_min = clamp(bbox[1] - buffer)
        x_max = clamp(bbox[0] + bbox[2] + buffer)
        y_max = clamp(bbox[1] + bbox[3] + buffer)
        max_dimension = max([(x_max - x_min)*width, (y_max - y_min)*height]) 
        return final_size/max_dimension if max_dimension > final_size else None


    def md_crop_image(self,
                      bbox: List[float],
                      image_arr: np.ndarray):
        '''Arguments: Detections for one file
                      The image as a NumPy array
           1. Select the highest confidence megadetector detection
           2. Downsize the image if the bounding box is > than the expected crop size
           3. Convert the megadetector bbox to absolute pixel locations [left, top, right, bottom]
           4. Crop the numpy array
           '''
        img_buffer = self.buffer
        resample =  self.downsample
        size=self.crop_size
        img_h, img_w = image_arr.shape[:2]

        if resample:
            scale = self.get_new_scale(bbox, img_buffer, img_w, img_h, size)
            if scale is not None:
                img_w, img_h = int(round(img_w * scale)), int(round(img_h * scale))
                image_arr = cv2.resize(image_arr, (img_w, img_h), cv2.INTER_LANCZOS4)

        left, top, right, bottom = self.get_mega_crop_values(bbox, img_w, img_h, size)
        cropped_arr = image_arr[top:bottom, left:right]
        
        crop_h, crop_w = cropped_arr.shape[:2] # both = size, unless one dimension was too small
        if (crop_h < size) or (crop_w < size):
            cropped_arr, _, _ = self.pad_to_square(cropped_arr)

        return cropped_arr


    def make_crop(self, 
            image: Image.Image,
            bbox: List[float],
            ):
        '''Uses the bounding box from MegaDetector to crop the original image to 600 pixels.
        If the bounding box is greater than 600 pixels then it is downsized to 600 pixels.
        The final step normalises the colour channels, and crops to 3x480x480 pixels.'''

        if image is None:
            print(f"Warning: Unable to load the image. Skipping...")
            return None
        
        # Convert PIL Image to NumPy array (RGB)
        image_np = np.array(image)

        # Convert RGB to BGR for OpenCV functions
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        if self.resize_method == 'rescale':
            image_cv = self.rescale_image(image_cv)
        else:
            image_cv = self.md_crop_image(bbox, image_cv)

        # Convert back BGR to RGB
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

        # Albumentations transform expects RGB image
        transformed = self.transform(image=image_rgb)['image']
        
        return transformed


class ClassifierHead(nn.Module):
    def __init__(self, num_features, num_classes, dropout_rate=0.2):
        super().__init__()
        self.linear = nn.Linear(num_features, num_features//2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.output_layer = nn.Linear(num_features//2, num_classes)
        
    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x


class CustomModel(pl.LightningModule):
    def __init__(self,
                 classes,
                 model_name='tf_efficientnetv2_s.in21k',
                 ):
        super().__init__()
        self.custom_head = ClassifierHead
        self.classes = classes
        self.num_classes = len(classes)
        self.backbone = timm.create_model(model_name, pretrained=False)
        self.in_features = self.backbone.classifier.in_features       
        self.backbone.classifier = self.custom_head(self.in_features, self.num_classes)

    def forward(self, images):
        logits = self.backbone(images)
        return logits
    

def get_crop(image: Image.Image,
             bbox: List[float]):
    '''Wrapper to make the form consistent with AddaxAI
       Uses the global instance of ImageTransforms(): transforms'''
    crop =  transforms.make_crop(image, bbox)
    return crop


def get_model(weights, classes, backbone='tf_efficientnetv2_s.in21k'):
    saved_state_dict = torch.load(weights)
    model = CustomModel(classes, backbone)
    model.load_state_dict(saved_state_dict)
    model.eval()
    return model


def predict_one_image(crop):
    '''
    Multi-label predictions for a 3x480x380 image as a NumPy array
    Returns a list of lists [[class_1, pred],[class_2, pred_2]...] 
    Uses the global instance of CustomModel()
    '''

    classifications = []
    classes = model.classes 

    if crop.ndim == 3:
        crop = crop.unsqueeze(0)
    crop = crop.to(model.device)
    
    with torch.no_grad():
        logits = model(crop)
        probs = F.sigmoid(logits)
        cpu_probs = probs.detach().squeeze().cpu().numpy()
    
    for i in range(len(classes)):
        classifications.append([classes[i], cpu_probs[i]])

    return classifications

#############################################
############### MODEL GENERIC ###############
#############################################
# check GPU availability
GPU_availability = False
try:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        GPU_availability = True
except:
    pass
if not GPU_availability:
    GPU_availability = torch.cuda.is_available()

# root = Path(__file__).resolve().parent
# temp_image_pth = str(root / '0AC899F4-9EF7-4ECC-83F8-FD1FF990B77C.JPG') 
# temp_weights = str(root / 'Exp_46_Run_21_best_weights.pt')
# temp_detection = [0.4929, 0.046, 0.507, 0.6269 ]

species_list = ["banded_dotterel", "banded_rail", "bellbird", "black_backed_gull", "black_billed_gull", 
                "black_fronted_tern", "blackbird", "canada_goose", "cat", "chamois", "chicken", "cow", 
                "crake", "deer", "dog", "dunnock", "fantail", "ferret", "finch", "fiordland_crested_penguin", 
                "goat", "grey_warbler", "hare", "harrier", "hedgehog", "horse", "human", "kaka", "kea", 
                "kereru", "kiwi", "little_blue_penguin", "magpie", "mallard", "morepork", "mouse", 
                "myna", "nz_falcon", "oystercatcher", "paradise_duck", "parakeet", "pateke", "pheasant", 
                "pig", "pipit", "possum", "pukeko", "quail", "rabbit", "rat", "redpoll", "rifleman", "robin", 
                "rosella", "sealion", "sheep", "shore_plover", "silvereye", "sparrow", "spurwing_plover", 
                "starling", "stilt", "stoat", "swan", "tahr", "takahe", "thrush", "tieke", "tomtit", "tui",
                "wallaby", "weasel", "weka", "white_faced_heron", "wrybill", "yellow_eyed_penguin", 
                "yellowhammer"]

#Instantiate methods & models
image_config = ImageConfig()
transforms = ImageTransformer(image_config)
model = get_model(classes=species_list,
                    backbone='tf_efficientnetv2_s.in21k',
                    weights=cls_model_fpath)


# run main function
import AddaxAI.classification_utils.inference_lib as ea
ea.classify_MD_json(json_path = json_path,
                    GPU_availability = GPU_availability,
                    cls_detec_thresh = cls_detec_thresh,
                    cls_class_thresh = cls_class_thresh,
                    smooth_bool = smooth_bool,
                    crop_function = get_crop,
                    inference_function = predict_one_image,
                    temp_frame_folder = temp_frame_folder,
                    cls_model_fpath = cls_model_fpath,
                    cls_tax_fallback = cls_tax_fallback,
                    cls_tax_levels_idx = cls_tax_levels_idx)
