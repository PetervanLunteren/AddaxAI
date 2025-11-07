# Script to further identify MD animal detections using the DeepForestVision classification model v1.
# https://www.oneforestvision.org
# https://github.com/MNHN-OFVI/DeepForestVision
# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI.

# Script created by Peter van Lunteren
# Some code is created by the DeepForestVision team and is indicated as so 
# Latest edit by Peter van Lunteren on 7 Nov 2025

# DeepForestVision is developed under CC BY-NC-SA 4.0 license
# (https://creativecommons.org/licenses/by-nc-sa/4.0) by an academic team from the
# French Muséum National d'Histoire Naturelle (MNHN) as part of the One Forest Vision
# initiative (https://www.oneforestvision.org).

# hugo.magaldi@mnhn.fr; sabrina.krief@mnhn.fr

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
import sys
import numpy as np
import timm
import torch
from torch import tensor
import torch.nn as nn
from torchvision.transforms import InterpolationMode, transforms
from collections import OrderedDict
from transformers import AutoModelForImageClassification

# ignore warnings about beta transforms in torchvision
import torchvision
torchvision.disable_beta_transforms_warning()

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

CROP_SIZE = 224
BACKBONE = "dinov2_large"

txt_animalclasses = {
    'en': ["aardvark","baboon","honey badger","bird","black-and-white colobus","blue duiker","blue monkey","african buffalo","bushbuck",
        "bushpig","chimpanzee","civet_genet","elephant","galago_potto","african golden cat","gorilla","guineafowl","hyrax","side-striped jackal",
        "leopard","l'hoest's monkey","mandrill","mongoose","monkey","pangolin","porcupine","red colobus_red-capped mangabey","red duiker",
        "rodent","serval","spotted hyena","squirrel","water chevrotain","yellow-backed duiker"],
}

class Classifier:
    def __init__(self):
        self.model = Model()
        self.model.loadWeights(cls_model_fpath)
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
        model_checkpoint = 'facebook/dinov2-large'
        self.base_model = AutoModelForImageClassification.from_pretrained(model_checkpoint)
        self.base_model.classifier = torch.nn.Linear(self.base_model.classifier.in_features, len(txt_animalclasses['en']))

        print(f"Using {BACKBONE} with weights at {cls_model_fpath}, in resolution {CROP_SIZE}x{CROP_SIZE}")
        self.backbone = BACKBONE
        self.nbclasses = len(txt_animalclasses['en'])

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
                output = self.forward(x).logits.softmax(dim=1)
            else:
                output = self.forward(x).logits
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
            weights =  torch.load(path, map_location=device)
            weights_new = OrderedDict([(key.replace('dinov2', 'base_model.dinov2').replace('classifier', 'base_model.classifier'), value) for key, value in weights.items()])
            params = {'state_dict': weights_new, 'args':{'num_classes' : len(txt_animalclasses['en']), 'backbone':BACKBONE}}
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
