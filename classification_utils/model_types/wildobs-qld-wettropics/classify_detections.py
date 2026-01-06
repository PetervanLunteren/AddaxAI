# Script to further identify MD animal detections using PT classification models trained by WildObs
# Model developed by Prakash Palanivelu Rajmohan and Renuka Sharma
# https://huggingface.co/WildObs/WildObs_QLD_WetTropics

# It consists of code that is specific for this kind of model architecture, and 
# code that is generic for all model architectures that will be run via AddaxAI.

# AddaxAI integration script by Peter van Lunteren
# Latest edit by Peter van Lunteren on 6 Jan 2026

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
import torch
from torchvision import transforms

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

# load model
classes= ['Alectura_lathami', 'Bos_taurus', 'Canis_familiaris', 'Casuarius_casuarius', 'Felis_catus', 'Heteromyias_cinereifrons', 'Homo_sapiens', 'Hypsiprymnodon_moschatus', 'Megapodius_reinwardt', 'Orthonyx_spaldingii', 'Perameles_nasuta', 'Sus_scrofa', 'Thylogale_stigmatica', 'Uromys_caudimaculatus', 'Wallabia_bicolor']
model = torch.load(cls_model_fpath, map_location=device_str,weights_only=False)
model.eval()
model.to(device_str)

# image preprocessing 
preprocess = transforms.Compose([
    transforms.Resize((480, 480)),
    transforms.ToTensor(),
])

# predict from cropped image
# input: cropped PIL image
# output: unsorted classifications formatted as [['aardwolf', 2.3025326090220233e-09], ['african wild cat', 5.658252888451898e-08], ... ]
# no need to remove forbidden classes from the predictions, that will happen in inference_lib.py
def get_classification(PIL_crop):
    img = preprocess(PIL_crop)     # -> C,H,W
    img = img.unsqueeze(0)         # -> B,C,H,W
    img = img.permute(0,2,3,1)     # -> B,H,W,C
    img = img.to(device_str)
    logits = model(img)
    probs = torch.softmax(logits, dim=1)[0].cpu().detach().numpy()
    classifications = []
    for i in range(len(probs)):
        pred_class = classes[i]
        pred_conf = float(probs[i])
        classifications.append([pred_class, pred_conf])
    return classifications

# method of removing background
# input: image = full image PIL.Image.open(img_fpath) <class 'PIL.JpegImagePlugin.JpegImageFile'>
# input: bbox = the bbox coordinates as read from the MD json - detection['bbox'] - [xmin, ymin, xmax, ymax]
# output: cropped image <class 'PIL.Image.Image'>
# each developer has its own way of padding, squaring, cropping, resizing etc
# it needs to happen exactly the same as on which the model was trained
# I've pulled this crop function from
# https://huggingface.co/WildObs/WildObs_QLD_WetTropics/blob/main/Evaluate_WetTropics_hf.ipynb
def get_crop(img, bbox_norm):
    width, height = img.size
    x, y, w, h = bbox_norm
    left = int(x * width)
    top = int(y * height)
    right = int((x + w) * width)
    bottom = int((y + h) * height)
    crop = img.crop((left, top, right, bottom))
    square_crop = crop.resize((600, 600), Image.BILINEAR)
    return square_crop

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
