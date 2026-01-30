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
from PIL import ImageFile, Image
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
from PIL import ImageOps
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode, transforms
import torchvision.transforms.functional as TF

# check on which GPU the process should run
def fetch_device():
    GPU_availability=False
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')
        GPU_availability=True
    try:
        if torch.backends.mps.is_built and torch.backends.mps.is_available():
            device = torch.device('mps')
            GPU_availability=True
    except AttributeError:
        pass
    return device,GPU_availability

################################################
############## AWC CLASSIFICATION STARTS #######
################################################

txt_animalclasses = {
    'en': [
        "spiny-cheeked honeyeater",
        "rufus bettong",
        "dragon sp",
        "australian brush-turkey",
        "frog sp",
        "yellow-footed antechinus",
        "swamp antechinus",
        "australian bustard",
        "diurnal raptor sp",
        "duck sp",
        "finch sp",
        "honeyeater sp",
        "nocturnal predatory bird sp",
        "quail sp",
        "small bird sp",
        "wetland bird sp",
        "wren sp",
        "burrowing bettong",
        "brush-tailed bettong",
        "northern bettong",
        "cattle",
        "swamp buffalo",
        "bush stone-curlew",
        "dingo",
        "goat",
        "pheasant coucal",
        "pygmy possum sp",
        "deer sp",
        "brown-capped emerald-dove",
        "great bowerbird",
        "chestnut quail-thrush",
        "rufous treecreeper",
        "grey shrike-thrush",
        "white-winged chough",
        "corvid sp",
        "butcherbird sp",
        "freshwater crocodile",
        "kookaburra sp",
        "brush-tailed dasyurid sp",
        "quoll sp",
        "emu",
        "horse and donkey sp",
        "domestic cat",
        "diamond dove",
        "bar-shouldered dove",
        "peaceful dove",
        "spinifex pigeon",
        "western partridge pigeon",
        "magpie-lark",
        "australian magpie",
        "water rat",
        "buff-banded rail",
        "musky rat-kangaroo",
        "short-nosed bandicoot sp",
        "hare and rabbit sp",
        "spectacled hare-wallaby",
        "rufous hare-wallaby",
        "banded hare-wallaby",
        "central short-tailed mouse",
        "malleefowl",
        "greater stick-nest rat",
        "kimberley ta-ta lizard",
        "grey kangaroo sp",
        "greater bilby",
        "miner bird sp",
        "grassland melomys",
        "black-footed tree-rat",
        "golden-backed tree-rat",
        "house mouse",
        "numbat",
        "agile wallaby",
        "black-striped wallaby",
        "tammar wallaby",
        "whip-tailed wallaby",
        "red-necked wallaby",
        "hopping mouse",
        "crested pigeon",
        "bridled nailtail wallaby",
        "northern nail-tailed wallaby",
        "crested bellbird",
        "antilopine wallaroo",
        "common wallaroo",
        "red kangaroo",
        "shark bay bandicoot",
        "long-nosed bandicoot sp",
        "glider sp",
        "western short-eared rock-wallaby",
        "monjon",
        "nabarlek",
        "black-footed rock wallaby",
        "mareeba rock-wallaby",
        "mount claro rock-wallaby",
        "eastern short-eared rock-wallaby",
        "white-quilled rock-pigeon",
        "common bronzewing",
        "red-tailed phascogale",
        "brush-tailed phascogale",
        "koala",
        "babbler sp",
        "long-nosed potoroo",
        "pseudantechinus sp",
        "ring-tailed possum sp",
        "delicate mouse",
        "parrot sp",
        "bush rat",
        "black rat",
        "canefield rat",
        "pale field-rat",
        "long-haired rat",
        "cane toad",
        "willie wagtail",
        "skink sp",
        "dunnart sp",
        "currawong sp",
        "pig",
        "short-beaked echidna",
        "red-legged pademelon",
        "bluetongue sp",
        "brushtail possum sp",
        "giant white-tailed rat",
        "ridge-tailed goanna",
        "kimberley rock goanna",
        "black-palmed goanna",
        "mertens' water goanna",
        "heath monitor",
        "black-headed and tree goanna sp",
        "yellow-spotted and sand goanna sp",
        "lace goanna",
        "wombat sp",
        "red fox",
        "swamp wallaby",
        "scaly-tailed possum",
        "common rock-rat",
        "central rock-rat",
        "kimberley rock-rat",
    ]
}

CROP_SIZE = 300
BACKBONE = "tf_efficientnet_b5.ns_jft_in1k"
weight_path = cls_model_fpath


device, GPU_availability = fetch_device()

class Resize:
    """Resize image with ResizeMethod.Crop for validation (center crop)"""
    
    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        """
        Args:
            size: int or tuple (height, width)
            interpolation: PIL interpolation mode
        """
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.interpolation = interpolation
    
    def __call__(self, img):   
        w, h = img.size  # PIL uses (width, height)
        target_h, target_w = self.size
        
        # Resize so LARGER dimension matches, then crop
        # This ensures we never scale up more than necessary
        ratio_w = w / target_w
        ratio_h = h / target_h
        m = min(ratio_w, ratio_h)  # Use smaller ratio
        
        # Calculate intermediate size after resize
        new_w = int(w / m)
        new_h = int(h / m)
        
        # Resize image
        img_resized = TF.resize(img, (new_h, new_w), self.interpolation)
        
        # Center crop to target size
        # Calculate top-left corner for center crop
        left = (new_w - target_w) // 2
        top = (new_h - target_h) // 2
        
        return TF.crop(img_resized, top, left, target_h, target_w)
    
class Classifier():
    def __init__(self):
        """
        Constructor of model classifier
        """
        super().__init__()
        self.train_model = timm.create_model(BACKBONE, pretrained=False, num_classes=len(txt_animalclasses['en']))
        self.load_weights(weight_path)

        self.transforms = transforms.Compose([
            Resize(size=CROP_SIZE, interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        print(f"Using {BACKBONE} with weights at {weight_path}, in resolution {CROP_SIZE}x{CROP_SIZE}")



    # return model
    def load_weights(self, path):
        """
        :param path: path of .pth save of model
        """
        if path[-4:] != ".pth":
            path += ".pth"
        try:
            state_dict = torch.load(path, map_location=device)
            self.train_model.load_state_dict(state_dict, strict=False)
            self.train_model.to(device)
            self.train_model.eval()
        except Exception as e:
            print("\n/!\ Can't load checkpoint model /!\ because :\n\n " + str(e), file=sys.stderr)
            raise e


##############################################
############## CLASSIFTOOLS END ##############
##############################################

# load model
classifier = Classifier()


# read label map
# not neccesary for yolov8 models to retreive label map exernally, as it is incorporated into the model itself

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in infrence_lib.py
# this is also the place to preprocess the image if that need to happen

# save_i=0
def get_classification(PIL_crop):
    input_tensor = classifier.transforms(PIL_crop)

    input_batch = input_tensor.unsqueeze(0)  
    input_batch = input_batch.to(device)
    output = classifier.train_model(input_batch)
    probabilities = F.softmax(output, dim=1)
    probabilities_np = probabilities.cpu().detach().numpy()
    confidence_scores = probabilities_np[0]
    classifications = []
    lbls = txt_animalclasses['en']
    for i in range(len(confidence_scores)):
        pred_class = lbls[i]
        pred_conf = confidence_scores[i]
        classifications.append([pred_class, pred_conf])
    return classifications


# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained

def get_crop(img, bbox_norm, square_crop=True):
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)
    
    if square_crop:
        box_size = max(box_w, box_h)
        xmin = max(0, min(xmin - int((box_size - box_w) / 2), img_w - box_w))
        ymin = max(0, min(ymin - int((box_size - box_h) / 2), img_h - box_h))
        box_w = min(img_w, box_size)
        box_h = min(img_h, box_size)

    crop = img.crop(box=[xmin, ymin, xmin + box_w, ymin + box_h])

    if square_crop and (box_w != box_h):
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)

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
