# Script to further identify MD animal detections using the DeepFaune classification model v1.4
# https://www.deepfaune.cnrs.fr/en/
# https://plmlab.math.cnrs.fr/deepfaune/software/-/tree/master
# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI.

# Script created by Peter van Lunteren
# Some code is created by the DeepFaune team and is indicated as so 
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
import sys
import numpy as np
import timm
import torch
from torch import tensor
import torch.nn as nn
from torchvision.transforms import InterpolationMode, transforms

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

# The following ClassifTools code snippet is created by the DeepFaune team.
# Original license is shown below.
# Source: https://plmlab.math.cnrs.fr/deepfaune/software/-/blob/master/classifTools.py
# The code is unaltered, except for two minor adjustments:
# 1. Accommodate a non standard location of the model
# 2. Run on Apple Silicon GPU via MPS Metal GPU

################################################
############## CLASSIFTOOLS START ##############
################################################

# Copyright CNRS 2024

# simon.chamaille@cefe.cnrs.fr; vincent.miele@univ-lyon1.fr

# This software is a computer program whose purpose is to identify
# animal species in camera trap images.

#This software is governed by the CeCILL  license under French law and
# abiding by the rules of distribution of free software.  You can  use, 
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info". 

# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability. 

# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or 
# data to be ensured and,  more generally, to use and operate it in the 
# same conditions as regards security. 

# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

CROP_SIZE = 182
BACKBONE = "vit_large_patch14_dinov2.lvd142m"
# weight_path = "deepfaune-vit_large_patch14_dinov2.lvd142m.pt"
weight_path = cls_model_fpath # ADJUSTMENT 1

txt_animalclasses = {
    'en': ['bison', 'badger', 'ibex', 'beaver', 'red deer', 'golden jackal', 'chamois', 'cat', 'goat',
           'roe deer', 'dog', 'raccoon dog', 'fallow deer', 'squirrel', 'moose', 'equid', 'genet',
           'wolverine', 'hedgehog', 'lagomorph', 'wolf', 'otter', 'lynx', 'marmot', 'micromammal', 
           'mouflon', 'sheep', 'mustelid', 'bird', 'bear', 'porcupine', 'nutria', 'muskrat', 'raccoon',
           'fox', 'reindeer', 'wild boar', 'cow']
}


class Classifier:
    def __init__(self):
        self.model = Model()
        self.model.loadWeights(weight_path)
        self.transforms = transforms.Compose([
            transforms.Resize(size=(CROP_SIZE, CROP_SIZE), interpolation=InterpolationMode.BICUBIC, max_size=None, antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))])

    def predictOnBatch(self, batchtensor, withsoftmax=True):
        return self.model.predict(batchtensor, withsoftmax)

    # croppedimage loaded by PIL
    def preprocessImage(self, croppedimage):
        preprocessimage = self.transforms(croppedimage)
        return preprocessimage.unsqueeze(dim=0)

class Model(nn.Module):
    def __init__(self):
        """
        Constructor of model classifier
        """
        super().__init__()
        self.base_model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(txt_animalclasses['fr']),
                                            dynamic_img_size=True)
        print(f"Using {BACKBONE} with weights at {weight_path}, in resolution {CROP_SIZE}x{CROP_SIZE}")
        self.backbone = BACKBONE
        self.nbclasses = len(txt_animalclasses['fr'])

    def forward(self, input):
        x = self.base_model(input)
        return x

    def predict(self, data, withsoftmax=True):
        """
        Predict on test DataLoader
        :param test_loader: test dataloader: torch.utils.data.DataLoader
        :return: numpy array of predictions without soft max
        """
        self.eval()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = fetch_device() # ADJUSTMENT 2
        self.to(device)
        total_output = []
        with torch.no_grad():
            x = data.to(device)
            if withsoftmax:
                output = self.forward(x).softmax(dim=1)
            else:
                output = self.forward(x)
            total_output += output.tolist()

        return np.array(total_output)

    def loadWeights(self, path):
        """
        :param path: path of .pt save of model
        """
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = fetch_device() # ADJUSTMENT 2

        if path[-3:] != ".pt":
            path += ".pt"
        try:
            params = torch.load(path, map_location=device)
            args = params['args']
            if self.nbclasses != args['num_classes']:
                raise Exception("You load a model ({}) that does not have the same number of class"
                                "({})".format(args['num_classes'], self.nbclasses))
            self.backbone = args['backbone']
            self.nbclasses = args['num_classes']
            self.load_state_dict(params['state_dict'])
        except Exception as e:
            print("\n/!\ Can't load checkpoint model /!\ because :\n\n " + str(e), file=sys.stderr)
            raise e

##############################################
############## CLASSIFTOOLS END ##############
##############################################

# load model
classifier = Classifier()

# check GPU availability
GPU_availability = False
try:
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        GPU_availability = True
except:
    pass
if not GPU_availability:
    GPU_availability = torch.cuda.is_available()

# read label map
# not necessary for yolov8 models to retrieve label map exernally, as it is incorporated into the model itself

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in infrence_lib.py
# this is also the place to preprocess the image if that need to happen
def get_classification(PIL_crop):
    PIL_crop = PIL_crop.convert('RGB')
    tensor_cropped = classifier.preprocessImage(PIL_crop)
    confs = classifier.predictOnBatch(tensor_cropped)[0,]
    lbls = txt_animalclasses['en']
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
def get_crop(image, bbox):
    width, height = image.size
    xmin = int(round(bbox[0] * width))
    ymin = int(round(bbox[1] * height))
    xmax = int(round(bbox[2] * width)) + xmin
    ymax = int(round(bbox[3] * height)) + ymin
    xsize = (xmax-xmin)
    ysize = (ymax-ymin)
    if xsize>ysize:
        ymin = ymin-int((xsize-ysize)/2)
        ymax = ymax+int((xsize-ysize)/2)
    if ysize>xsize:
        xmin = xmin-int((ysize-xsize)/2)
        xmax = xmax+int((ysize-xsize)/2)
    image_cropped = image.crop((max(0,xmin), max(0,ymin), min(xmax,image.width), min(ymax,image.height)))
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
