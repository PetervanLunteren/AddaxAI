# coding=utf-8

# GUI to simplify camera trap image analysis with species recognition models
# https://addaxdatascience.com/addaxai/
# Created by Peter van Lunteren
# Latest edit by Peter van Lunteren on 10 Jul 2025

# TODO: DEPTH - add depth estimation model: https://pytorch.org/hub/intelisl_midas_v2/
# TODO: CLEAN - if the processing is done, and a image is deleted before the post processing, it crashes and just stops, i think it should just skip the file and then do the rest. I had to manually delete certain entries from the image_recognition_file.json to make it work
# TODO: RESUME DOWNLOAD - make some sort of mechanism that either continues the model download when interrupted, or downloads it to /temp/ folder and only moves it to the correct location after successful download. Otherwise delete from /temp/. That makes sure that users will not be able to continue with half downloaded models. 
# TODO: BUG - when moving files during postprocessing and exporting xlsx on Windows, it errors with an "file is in use". There must be something going on with opening files... does not happen when copying files or on Mac. 
# TODO: PYINSTALLER - Get rid of the PyInstaller apps. Then there wont be the weird histation when opning. While you're at it, remove version number in the execution files. Then you can use the same shortcuts. 
# TODO: WIDGET - make a slider widget for the line width of the bounding box. 
# TODO: Microsoft Amazon is not working on MacOS, and Iran is not working on Windows. 
# TODO: MERGE JSON - for timelapse it is already merged. Would be great to merge the image and video jsons together for AddaxAI too, and process videos and jsons together. See merge_jsons() function.
# TODO: LAT LON 0 0 - filter out the 0,0 coords for map creation
# TODO: JSON - remove the original json if not running AddaxAI in Timelapse mode. No need to keep that anymore. 
# TODO: JSON - remove the part where MD stores its typical threshold values etc in the AddaxAI altered json. It doesn't make sense anymore if the detection caterogies are changed. 
# TODO: VIDEO - create video tutorials of all the steps (simple mode, advanced mode, annotation, postprocessing, etc.)
# TODO: EMPTIES - add a checkbox for folder separation where you can skip the empties from being copied
# TODO: LOG SEQUENCE INFO - add sequence information to JSON, CSV, and XSLX 
# TODO: SEQ SEP - add feature to separate images into sequence subdirs. Something like "treat sequence as detection" or "Include all images in the sequence" while doing the separation step.
# TODO: INFO - add a messagebox when the deployment is done via advanced mode. Now it just says there were errors. Perhaps just one messagebox with extra text if there are errors or warnings. And some counts. 
# TODO: JSON - keep track of confidences for each detection and classification in the JSON. And put that in CSV/XSLX, and visualise it in the images.
# TODO: CSV/XLSX - add frame number and frama rate to the CSV and XLSX files
# TODO: VIS VIDEO - add option to visualise frame with highest confidence
# TODO: N_CORES - add UI "--ncores” option - see email Dan "mambaforge vs. miniforge"
# TODO: REPORTS - add postprocessing reports - see email Dan "mambaforge vs. miniforge"
# TODO: MINOR - By the way, in the AddaxAI UI, I think the frame extraction status popup uses the same wording as the detection popup. They both say something about "frame X of Y". I think for the frame extraction, it should be "video X of Y".
# TODO: JSON - keep track of the original confidence scores whenever it changes (from detection to classification, after human verification, etc.)
# TODO: SMALL FIXES - see list from Saul ('RE: tentative agenda / discussion points') - 12 July 01:11. 
# TODO: ANNOTATION - improve annotation experience
    # - make one progress windows in stead of all separate pbars when using large jsons
    # - I've converted pyqt5 to pyside6 for apple silicon so we don't need to install it via homebrew
    #         the unix install clones a pyside6 branch of my human-in-the-loop fork. Test windows on this
    #         on this version too and make it the default
    # - implement image progress status into main labelimg window, so you don't have two separate windows
    # - apparently you still get images in which a class is found under the annotation threshold,
    #         it should count only the images that have classes above the set annotation threshold,
    #         at this point it only checks whether it should draw an bbox or not, but still shows the image
    # - Add custom shortcuts. See email Grant ('Possible software feature'). 
    # - Add option to order chronological See email Grant ('A few questions I've come up with').
    # - If you press the '?' button in the selection window, it doesn't scroll all the way down anymore. So
    #         adjust the scroll region, of make an option to close the help text
    # - shift forcus on first label. See email Grant ('Another small request').
    # - get rid of the default label pane in the top right. Or at least make it less prominent. 
    # - remove the X cross to remove the box label pane. No need to have an option to remove it. It's difficult to get it back on macOS. 
    # - see if you can add the conf of the bbox in the box label pane too. just for clarification purposes for threshold settings (see email Grant "Showing confidence level")
    # - there should be a setting that shows box labels inside the image. turn this on by default.
    # - remove the messagebox that warns you that you're not completely done with the human verification before postprocess. just do it.
    # - why do I ask if the user is done after verification anyway? why not just take the results as they are and accept it? 
    # - take the annotation confidence ranges the same as the image confidence ranges if the user specified them. Otherwise use 0.6-1.0.
    # - When I zoom in, I always zoom in on the center, and then I can’t manage to move the image.
    # - I figured out when the label becomes greyed out. For me, it happens when I draw a bounding box myself, and then when I go to the next image, "edit label" is greyed out. If I then close the annotation (but not the entire app) and continue, it works again.

#import packages like a very pointy half christmas tree
import os
import re
import io
import sys
import cv2
import csv
import json
import math
import time
import glob
import random
import signal
import shutil
import pickle
import folium
import hashlib 
import argparse
import calendar
import platform
import requests
import tempfile
import datetime
import traceback
import subprocess
import webbrowser
import numpy as np
import PIL.ExifTags
import pandas as pd
import tkinter as tk
import customtkinter
import seaborn as sns
from tqdm import tqdm
from tkinter import *
from pathlib import Path
import plotly.express as px
from subprocess import Popen
from functools import partial
from tkinter.font import Font
from GPSPhoto import gpsphoto
from CTkTable import CTkTable
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import defaultdict
import xml.etree.cElementTree as ET
from PIL import ImageTk, Image, ImageFile
from RangeSlider.RangeSlider import RangeSliderH
from tkinter import filedialog, ttk, messagebox as mb
from folium.plugins import HeatMap, Draw, MarkerCluster
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# check if the script is ran from the macOS installer executable
# if so, don't actually execute the script - it is meant just for installation purposes
if len(sys.argv) > 1:
    if sys.argv[1] == "installer":
        exit()

# set global variables
AddaxAI_files = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ImageFile.LOAD_TRUNCATED_IMAGES = True
CLS_DIR = os.path.join(AddaxAI_files, "models", "cls")
DET_DIR = os.path.join(AddaxAI_files, "models", "det")

# set environment variables
if os.name == 'nt': # windows
    env_dir_fpath = os.path.join(AddaxAI_files, "envs")
elif platform.system() == 'Darwin': # macos
    env_dir_fpath = os.path.join(AddaxAI_files, "envs")
else: # linux
    env_dir_fpath = os.path.join(AddaxAI_files, "envs")

# set versions
with open(os.path.join(AddaxAI_files, 'AddaxAI', 'version.txt'), 'r') as file:
    current_AA_version = file.read().strip()
corresponding_model_info_version = "5"

# colors
# most of the colors are set in the ./themes/addaxai.json file
green_primary = '#0B6065'
green_secondary = '#073d40'
yellow_primary = '#fdfae7'
yellow_secondary = '#F0EEDC'
yellow_tertiary = '#E4E1D0'

# images
PIL_sidebar = PIL.Image.open(os.path.join(AddaxAI_files, "AddaxAI", "imgs", "side-bar.png"))
PIL_logo_incl_text = PIL.Image.open(os.path.join(AddaxAI_files, "AddaxAI", "imgs", "square_logo_incl_text.png"))
PIL_checkmark = PIL.Image.open(os.path.join(AddaxAI_files, "AddaxAI", "imgs", "checkmark.png"))
PIL_dir_image = PIL.Image.open(os.path.join(AddaxAI_files, "AddaxAI", "imgs", "image-gallery.png"))
PIL_mdl_image = PIL.Image.open(os.path.join(AddaxAI_files, "AddaxAI", "imgs", "tech.png"))
PIL_spp_image = PIL.Image.open(os.path.join(AddaxAI_files, "AddaxAI", "imgs", "paw.png"))
PIL_run_image = PIL.Image.open(os.path.join(AddaxAI_files, "AddaxAI", "imgs", "shuttle.png"))
launch_count_file = os.path.join(AddaxAI_files, 'launch_count.json')

# insert dependencies to system variables
cuda_toolkit_path = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
paths_to_add = [
    os.path.join(AddaxAI_files),
    os.path.join(AddaxAI_files, "cameratraps"),
    os.path.join(AddaxAI_files, "cameratraps", "megadetector"),
    os.path.join(AddaxAI_files, "AddaxAI")
]
if cuda_toolkit_path:
    paths_to_add.append(os.path.join(cuda_toolkit_path, "bin"))
for path in paths_to_add:
    sys.path.insert(0, path)
PYTHONPATH_separator = ":" if platform.system() != "Windows" else ";"
os.environ["PYTHONPATH"] = os.environ.get("PYTHONPATH", "") + PYTHONPATH_separator + PYTHONPATH_separator.join(paths_to_add)

# import modules from forked repositories
from visualise_detection.bounding_box import bounding_box as bb
from cameratraps.megadetector.detection.video_utils import frame_results_to_video_results, FrameToVideoOptions, VIDEO_EXTENSIONS
from cameratraps.megadetector.utils.path_utils import IMG_EXTENSIONS

# log pythonpath
print(sys.path)

# set DPI awareness on Windows
scale_factor = 1.0
if platform.system() == "Windows":
    import ctypes
    try:
        # attempt
        ctypes.windll.shcore.SetProcessDpiAwareness(2) 
        scale_factor = ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
    except AttributeError:
        # fallback for older versions of Windows
        ctypes.windll.user32.SetProcessDPIAware()

# load previous settings
def load_global_vars():
    var_file = os.path.join(AddaxAI_files, "AddaxAI", "global_vars.json")
    with open(var_file, 'r') as file:
        variables = json.load(file)
    return variables
global_vars = load_global_vars()

# language settings
languages_available = ['English', 'Español', 'Français']
lang_idx = global_vars["lang_idx"]
step_txt = ['Step', 'Paso', 'Étape']
browse_txt = ['Browse', 'Examinar', 'Parcourir']
cancel_txt = ["Cancel", "Cancelar", "Annuler"]
change_folder_txt = ['Change folder', '¿Cambiar carpeta', 'Changer le dossier']
view_results_txt = ['View results', 'Ver resultados', 'Voir les résultats']
custom_model_txt = ['Custom model', "Otro modelo", "Modèle personnalisé"]
again_txt = ['Again?', '¿Otra vez?', 'Recommencer?']
eg_txt = ['E.g.', 'Ejem.', 'Par. ex.']
show_txt = ["Show", "Mostrar", "Afficher"]
new_project_txt = ["<new project>", "<nuevo proyecto>", "<nouveau projet>"]
warning_txt = ["Warning", "Advertencia", "Avertissement"]
information_txt = ["Information", "Información", "Information"]
error_txt = ["Error", "Error", "Erreur"]
select_txt = ["Select", "Seleccionar", "Sélectionner"]
invalid_value_txt = ["Invalid value", "Valor no válido", "Valeur invalide"]
none_txt = ["None", "Ninguno", "Aucun"]
of_txt = ["of", "de", "de"]
suffixes_for_sim_none = [" - just show me where the animals are",
                         " - muéstrame dónde están los animales",
                         " - montrez-moi uniquement où sont les animaux"]

#############################################
############# BACKEND FUNCTIONS #############
#############################################

# post-process files
def postprocess(src_dir, dst_dir, thresh, sep, keep_series, keep_series_seconds, file_placement, sep_conf, vis, crp, exp, plt, exp_format, data_type, keep_series_species=None):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # update progress window
    progress_window.update_values(process = f"{data_type}_pst", status = "load")

    # plt needs csv files so make sure to produce them, even if the user didn't specify
    # if the user didn't specify to export to csv, make sure to remove them later on
    remove_csv = False
    if plt and not exp:
        # except if the csv are already created ofcourse
        if not (os.path.isfile(os.path.join(dst_dir, "results_detections.csv")) and 
                os.path.isfile(os.path.join(dst_dir, "results_files.csv"))):
            exp = True
            exp_format = dpd_options_exp_format[lang_idx][1] # CSV
            remove_csv = True

    # get correct json file
    if data_type == "img":
        recognition_file = os.path.join(src_dir, "image_recognition_file.json")
    else:
        recognition_file = os.path.join(src_dir, "video_recognition_file.json")

    # check if user is not in the middle of an annotation session
    if data_type == "img" and get_hitl_var_in_json(recognition_file) == "in-progress":
        if not mb.askyesno(["Verification session in progress", "Sesión de verificación en curso", "Session de vérification en cours"][lang_idx],
                           [f"Your verification session is not yet done. You can finish the session by clicking 'Continue' at '{lbl_hitl_main_txt[lang_idx]}', "
                            "or just continue to post-process with the results as they are now.\n\nDo you want to continue to post-process?",
                            f"La sesión de verificación aún no ha finalizado. Puede finalizarla haciendo clic en 'Continuar' en '{lbl_hitl_main_txt[lang_idx]}', "
                            "o simplemente continuar con el posprocesamiento con los resultados tal como están ahora.\n\n¿Quieres continuar con el posprocesamiento?",
                            f"Votre session de vérification n'est pas encore terminée. Vous pouvez la compléter en cliquant sur '{lbl_hitl_main_txt[lang_idx]}', "
                            "ou juste continuer le post-traitement avec les résultats actuels.\n\nSouhaitez-vous continuer le post-traitement?"][lang_idx]):
            return

    # init vars
    global cancel_var
    start_time = time.time()
    nloop = 1

    # warn user
    if data_type == "vid":
        if vis or crp or plt:
            check_json_presence_and_warn_user(["visualize, crop, or plot", "visualizar, recortar o trazar", "visualiser, rogner ou afficher"][lang_idx],
                                              ["visualizing, cropping, or plotting", "visualizando, recortando o trazando", "visualisation, rognage ou affichage"][lang_idx],
                                              ["visualization, cropping, and plotting", "visualización, recorte y trazado", "visualisation, rognage et affichage"][lang_idx])
            vis, crp, plt = [False] * 3

    # fetch label map
    label_map = fetch_label_map_from_json(recognition_file)
    inverted_label_map = {v: k for k, v in label_map.items()}

    # create list with colours for visualisation
    if vis:
        colors = ["fuchsia", "blue", "orange", "yellow", "green", "red", "aqua", "navy", "teal", "olive", "lime", "maroon", "purple"]
        colors = colors * 30

    # make sure json has relative paths
    json_paths_converted = False
    if check_json_paths(recognition_file) != "relative":
        make_json_relative(recognition_file)
        json_paths_converted = True

    # set cancel bool
    cancel_var = False
    
    # open json file
    with open(recognition_file) as image_recognition_file_content:
        data = json.load(image_recognition_file_content)
    n_images = len(data['images'])

    # --- series support: build timestamp index once for efficiency ----
    if keep_series:
        try:
            _file_list_for_index = [img['file'] for img in data.get('images', [])]
            timestamp_index = build_image_timestamp_index(src_dir, _file_list_for_index)
        except Exception:
            timestamp_index = {}
    # used to avoid moving the same original file multiple times
    already_moved_files = set()

    # initialise the csv files
    # csv files are always created, no matter what the user specified as export format
    # these csv files are then converted to the desired format and deleted, if required
    if exp:
        # for files
        csv_for_files = os.path.join(dst_dir, "results_files.csv")
        if not os.path.isfile(csv_for_files):
            df = pd.DataFrame(list(), columns=["absolute_path", "relative_path", "data_type", "n_detections", "file_height", "file_width", "max_confidence", "human_verified",
                                               'DateTimeOriginal', 'DateTime', 'DateTimeDigitized', 'Latitude', 'Longitude', 'GPSLink', 'Altitude', 'Make',
                                               'Model', 'Flash', 'ExifOffset', 'ResolutionUnit', 'YCbCrPositioning', 'XResolution', 'YResolution',
                                               'ExifVersion', 'ComponentsConfiguration', 'FlashPixVersion', 'ColorSpace', 'ExifImageWidth',
                                               'ISOSpeedRatings', 'ExifImageHeight', 'ExposureMode', 'WhiteBalance', 'SceneCaptureType',
                                               'ExposureTime', 'Software', 'Sharpness', 'Saturation', 'ReferenceBlackWhite'])
            df.to_csv(csv_for_files, encoding='utf-8', index=False)
        
        # for detections
        csv_for_detections = os.path.join(dst_dir, "results_detections.csv")
        if not os.path.isfile(csv_for_detections):
            df = pd.DataFrame(list(), columns=["absolute_path", "relative_path", "data_type", "label", "confidence", "human_verified", "bbox_left",
                                               "bbox_top", "bbox_right", "bbox_bottom", "file_height", "file_width", 'DateTimeOriginal', 'DateTime',
                                               'DateTimeDigitized', 'Latitude', 'Longitude', 'GPSLink', 'Altitude', 'Make', 'Model', 'Flash', 'ExifOffset',
                                               'ResolutionUnit', 'YCbCrPositioning', 'XResolution', 'YResolution', 'ExifVersion', 'ComponentsConfiguration',
                                               'FlashPixVersion', 'ColorSpace', 'ExifImageWidth', 'ISOSpeedRatings', 'ExifImageHeight', 'ExposureMode',
                                               'WhiteBalance', 'SceneCaptureType', 'ExposureTime', 'Software', 'Sharpness', 'Saturation', 'ReferenceBlackWhite'])
            df.to_csv(csv_for_detections, encoding='utf-8', index=False)

    # set global vars
    global postprocessing_error_log
    postprocessing_error_log = os.path.join(dst_dir, "postprocessing_error_log.txt")

    # count the number of rows to make sure it doesn't exceed the limit for an excel sheet
    if exp and exp_format == dpd_options_exp_format[lang_idx][0]: # if exp_format is the first option in the dropdown menu -> XLSX
        n_rows_files = 1
        n_rows_detections = 1
        for image in data['images']:
            n_rows_files += 1
            if 'detections' in image:
                for detection in image['detections']:
                    if detection["conf"] >= thresh:
                        n_rows_detections += 1
        if n_rows_detections > 1048576 or n_rows_files > 1048576:
            mb.showerror(["To many rows", "Demasiadas filas", "Trop de lignes"][lang_idx],
                         ["The XLSX file you are trying to create is too large!\n\nThe maximum number of rows in an XSLX file is "
                          f"1048576, while you are trying to create a sheet with {max(n_rows_files, n_rows_detections)} rows.\n\nIf"
                          " you require the results in XLSX format, please run the process on smaller chunks so that it doesn't "
                          f"exceed Microsoft's row limit. Or choose CSV as {lbl_exp_format_txt[lang_idx]} in advanced mode.", 
                          "¡El archivo XLSX que está intentando crear es demasiado grande!\n\nEl número máximo de filas en un archivo"
                          f" XSLX es 1048576, mientras que usted está intentando crear una hoja con {max(n_rows_files, n_rows_detections)}"
                          " filas.\n\nSi necesita los resultados en formato XLSX, ejecute el proceso en trozos más pequeños para que no "
                          f"supere el límite de filas de Microsoft. O elija CSV como {lbl_exp_format_txt[lang_idx]} en modo avanzado.",
                          "Le fichier XLSX que vous tenter de créer est trop long!\n\nLe nombre maximum de lignes dans un fichier XSLX est "
                          f"1048576, alors que vous tenter de créer une feuille avec {max(n_rows_files, n_rows_detections)} lignes.\n\nSi"
                          " vous souhaitez des résultats sous format XLSX, svp exécuter le processus sur de plus petites portions de sorte à ne pas "
                          f"excéder la limite de lignes de Microsoft. Ou choisissez le format CSV comme {lbl_exp_format_txt[lang_idx]} dans le mode avancé."][lang_idx])
            return

    # loop through images
    for image in data['images']:

        # cancel process if required
        if cancel_var:
            break
        
        # check for failure
        if "failure" in image:
            
            # write warnings to log file
            with open(postprocessing_error_log, 'a+') as f:
                f.write(f"File '{image['file']}' was skipped by post processing features because '{image['failure']}'\n")
            f.close()

            # calculate stats
            elapsed_time_sep = str(datetime.timedelta(seconds=round(time.time() - start_time)))
            time_left_sep = str(datetime.timedelta(seconds=round(((time.time() - start_time) * n_images / nloop) - (time.time() - start_time))))
            progress_window.update_values(process = f"{data_type}_pst",
                                            status = "running",
                                            cur_it = nloop,
                                            tot_it = n_images,
                                            time_ela = elapsed_time_sep,
                                            time_rem = time_left_sep,
                                            cancel_func = cancel)

            nloop += 1
            root.update()

            # skip this iteration
            continue
        
        # get image info
        file = image['file']
        detections_list = image['detections']
        n_detections = len(detections_list)

        # check if it has been manually verified
        manually_checked = False
        if 'manually_checked' in image:
            if image['manually_checked']:
                manually_checked = True

        # init vars
        max_detection_conf = 0.0
        unique_labels = []
        bbox_info = []

        # open files
        if vis or crp or exp:
            if data_type == "img":
                im_to_vis = cv2.imread(os.path.normpath(os.path.join(src_dir, file)))

                # check if that image was able to be loaded
                if im_to_vis is None:
                    with open(postprocessing_error_log, 'a+') as f:
                        f.write(f"File '{image['file']}' was skipped by post processing features. This might be due to the file being moved or deleted after analysis, or because of a special character in the file path.\n")
                    f.close()
                    elapsed_time_sep = str(datetime.timedelta(seconds=round(time.time() - start_time)))
                    time_left_sep = str(datetime.timedelta(seconds=round(((time.time() - start_time) * n_images / nloop) - (time.time() - start_time))))
                    progress_window.update_values(process = f"{data_type}_pst",
                                                    status = "running",
                                                    cur_it = nloop,
                                                    tot_it = n_images,
                                                    time_ela = elapsed_time_sep,
                                                    time_rem = time_left_sep,
                                                    cancel_func = cancel)
                    nloop += 1
                    root.update()
                    continue

                im_to_crop_path = os.path.join(src_dir, file)
                
                # load old image and extract EXIF
                origImage = Image.open(os.path.join(src_dir, file))
                try:
                    exif = origImage.info['exif']
                except:
                    exif = None

                origImage.close()
            else:
                vid = cv2.VideoCapture(os.path.join(src_dir, file))

            # read image dates etc
            if exp:

                # try to read metadata
                try:
                    img_for_exif = PIL.Image.open(os.path.join(src_dir, file))
                    metadata = {
                        PIL.ExifTags.TAGS[k]: v
                        for k, v in img_for_exif._getexif().items()
                        if k in PIL.ExifTags.TAGS
                    }
                    img_for_exif.close()
                except:
                    metadata = {'GPSInfo': None,
                                 'ResolutionUnit': None,
                                 'ExifOffset': None,
                                 'Make': None,
                                 'Model': None,
                                 'DateTime': None,
                                 'YCbCrPositioning': None,
                                 'XResolution': None,
                                 'YResolution': None,
                                 'ExifVersion': None,
                                 'ComponentsConfiguration': None,
                                 'ShutterSpeedValue': None,
                                 'DateTimeOriginal': None,
                                 'DateTimeDigitized': None,
                                 'FlashPixVersion': None,
                                 'UserComment': None,
                                 'ColorSpace': None,
                                 'ExifImageWidth': None,
                                 'ExifImageHeight': None}

                # try to add GPS data
                try:
                    gpsinfo = gpsphoto.getGPSData(os.path.join(src_dir, file))
                    if 'Latitude' in gpsinfo and 'Longitude' in gpsinfo:
                        gpsinfo['GPSLink'] = f"https://maps.google.com/?q={gpsinfo['Latitude']},{gpsinfo['Longitude']}"
                except:
                    gpsinfo = {'Latitude': None,
                               'Longitude': None,
                               'GPSLink': None}
                
                # combine metadata and gps data
                exif_data = {**metadata, **gpsinfo} 

                # check if datetime values can be found
                exif_params = []
                for param in ['DateTimeOriginal', 'DateTime', 'DateTimeDigitized', 'Latitude', 'Longitude', 'GPSLink', 'Altitude', 'Make', 'Model',
                              'Flash', 'ExifOffset', 'ResolutionUnit', 'YCbCrPositioning', 'XResolution', 'YResolution', 'ExifVersion',
                              'ComponentsConfiguration', 'FlashPixVersion', 'ColorSpace', 'ExifImageWidth', 'ISOSpeedRatings',
                              'ExifImageHeight', 'ExposureMode', 'WhiteBalance', 'SceneCaptureType', 'ExposureTime', 'Software',
                              'Sharpness', 'Saturation', 'ReferenceBlackWhite']:
                    try:
                        if param.startswith('DateTime'):
                            datetime_raw = str(exif_data[param])
                            param_value = datetime.datetime.strptime(datetime_raw, '%Y:%m:%d %H:%M:%S').strftime('%d/%m/%y %H:%M:%S')
                        else:
                            param_value = str(exif_data[param])
                    except:
                        param_value = "NA"
                    exif_params.append(param_value)

        # loop through detections
        if 'detections' in image:
            for detection in image['detections']:

                # get confidence
                conf = detection["conf"]

                # write max conf
                if manually_checked:
                    max_detection_conf = "NA"
                elif conf > max_detection_conf:
                    max_detection_conf = conf

                # if above user specified thresh
                if conf >= thresh:

                    # change conf to string for verified images
                    if manually_checked:
                        conf = "NA"

                    # get detection info
                    category = detection["category"]
                    label = label_map[category]
                    if sep:
                        unique_labels.append(label)
                        unique_labels = sorted(list(set(unique_labels)))

                    # get bbox info
                    if vis or crp or exp:
                        if data_type == "img":
                            height, width = im_to_vis.shape[:2]
                        else:
                            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))

                        w_box = detection['bbox'][2]
                        h_box = detection['bbox'][3]
                        xo = detection['bbox'][0] + (w_box/2)
                        yo = detection['bbox'][1] + (h_box/2)
                        left = int(round(detection['bbox'][0] * width))
                        top = int(round(detection['bbox'][1] * height))
                        right = int(round(w_box * width)) + left
                        bottom = int(round(h_box * height)) + top

                        # store in list
                        bbox_info.append([label, conf, manually_checked, left, top, right, bottom, height, width, xo, yo, w_box, h_box])
        
        # collect info to append to csv files
        if exp:

            # read shape again - this is a temporary solution
            # read in the height and width of the images and videos, as this was giving a bunch of bugs...
            if data_type == "img":
                img = cv2.imread(os.path.normpath(os.path.join(src_dir, file)))
                height, width = img.shape[:2]
            else:
                cap = cv2.VideoCapture(os.path.join(src_dir, file))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

            # file info CSV
            row = pd.DataFrame([[src_dir, file, data_type, len(bbox_info), height, width, max_detection_conf, manually_checked, *exif_params]])
            row.to_csv(csv_for_files, encoding='utf-8', mode='a', index=False, header=False)

            # detections info CSV
            rows = []
            for bbox in bbox_info:
                row = [src_dir, file, data_type, *bbox[:9], *exif_params]
                rows.append(row)
            rows = pd.DataFrame(rows)
            rows.to_csv(csv_for_detections, encoding='utf-8', mode='a', index=False, header=False)

            # separate files (with optional "keep whole series" feature)
            if sep:
                if n_detections == 0:
                    detection_type = "empty"
                else:
                    if len(unique_labels) > 1:
                        detection_type = "_".join(unique_labels)
                    elif len(unique_labels) == 0:
                        detection_type = "empty"
                    else:
                        detection_type = label

                # keep-series configuration
                should_keep_series = False
                if keep_series:
                    # allow caller to pass the list; otherwise fall back to persisted GUI setting
                    if keep_series_species is None:
                        keep_series_species = global_vars.get('var_keep_series_species', [])

                    labels_in_detection = set(unique_labels) if n_detections > 0 else set()

                    # Interpret selection relative to CURRENT cls model
                    try:
                        cur_model_vars = load_model_vars(model_type="cls")
                        cur_model_classes = set(cur_model_vars.get("all_classes", []) or [])
                    except Exception:
                        cur_model_classes = set()

                    keep_series_species_effective = [c for c in keep_series_species if c in cur_model_classes]
                    keep_series_species_effective_set = set(keep_series_species_effective)

                    # If user selected trigger species *that exist in the current model*: filter by those.
                    # Otherwise (no overlap): treat as "Any animal detection".
                    if keep_series_species_effective:
                        should_keep_series = bool(labels_in_detection.intersection(keep_series_species_effective_set))
                    else:
                        should_keep_series = bool(labels_in_detection.difference({'person', 'vehicle'}))

                if should_keep_series:
                    # move/copy the whole series (files within +/- window_seconds)
                    series_files = find_series_images(file, timestamp_index, window_seconds=keep_series_seconds)
                    for sf in series_files:
                        if sf in already_moved_files:
                            continue
                        moved_rel = move_files(sf, detection_type, file_placement, max_detection_conf, sep_conf,
                                               dst_dir, src_dir, manually_checked)
                        # record original relative path as moved so we don't try to move it again
                        already_moved_files.add(sf)
                        # if this is the current loop item, update the 'file' variable used later for visualization
                        if sf == file:
                            file = moved_rel
                else:
                    # default behaviour: move/copy single file
                    if file not in already_moved_files:
                        orig_file_for_move = file
                        file = move_files(orig_file_for_move, detection_type, file_placement, max_detection_conf,
                                          sep_conf, dst_dir, src_dir, manually_checked)
                        already_moved_files.add(orig_file_for_move)
    
        # visualize images
        if vis and len(bbox_info) > 0:
            
            # blur people
            if var_vis_blur.get():
                for bbox in bbox_info:
                    if bbox[0] == "person":
                        im_to_vis = blur_box(im_to_vis, *bbox[3:7], bbox[8], bbox[7])
                        
            # draw bounding boxes
            if var_vis_bbox.get():
                for bbox in bbox_info:
                    if manually_checked:
                        vis_label = f"{bbox[0]} (verified)"
                    else:
                        conf_label = round(bbox[1], 2) if round(bbox[1], 2) != 1.0 else 0.99
                        vis_label = f"{bbox[0]} {conf_label}"
                    color = colors[int(inverted_label_map[bbox[0]])]
                    bb.add(im_to_vis, *bbox[3:7], vis_label, color, size = dpd_options_vis_size[lang_idx].index(var_vis_size.get())) # convert string to index, e.g. "small" -> 0
            
            im = os.path.join(dst_dir, file)
            Path(os.path.dirname(im)).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(im, im_to_vis)

            # load new image and save exif
            if (exif != None):
                image_new = Image.open(im)
                image_new.save(im, exif=exif)
                image_new.close()
        
        # crop images
        if crp and len(bbox_info) > 0:
            counter = 1
            for bbox in bbox_info:

                # if files have been moved
                if sep:
                    im_to_crp = Image.open(os.path.join(dst_dir,file))                    
                else:
                    im_to_crp = Image.open(im_to_crop_path)
                crp_im = im_to_crp.crop((bbox[3:7]))
                im_to_crp.close()
                filename, file_extension = os.path.splitext(file)
                im_path = os.path.join(dst_dir, filename + '_crop' + str(counter) + '_' + bbox[0] + file_extension)
                Path(os.path.dirname(im_path)).mkdir(parents=True, exist_ok=True)
                crp_im.save(im_path)
                counter += 1

                 # load new image and save exif
                if (exif != None):
                    image_new = Image.open(im_path)
                    image_new.save(im_path, exif=exif)
                    image_new.close()

        # calculate stats
        elapsed_time_sep = str(datetime.timedelta(seconds=round(time.time() - start_time)))
        time_left_sep = str(datetime.timedelta(seconds=round(((time.time() - start_time) * n_images / nloop) - (time.time() - start_time))))
        progress_window.update_values(process = f"{data_type}_pst",
                                        status = "running",
                                        cur_it = nloop,
                                        tot_it = n_images,
                                        time_ela = elapsed_time_sep,
                                        time_rem = time_left_sep,
                                        cancel_func = cancel)

        nloop += 1
        root.update()

    # create summary csv
    if exp:
        csv_for_summary = os.path.join(dst_dir, "results_summary.csv")
        if os.path.exists(csv_for_summary):
            os.remove(csv_for_summary)
        det_info = pd.DataFrame(pd.read_csv(csv_for_detections, dtype=dtypes, low_memory=False))
        summary = pd.DataFrame(det_info.groupby(['label', 'data_type']).size().sort_values(ascending=False).reset_index(name='n_detections'))
        summary.to_csv(csv_for_summary, encoding='utf-8', mode='w', index=False, header=True)

    # convert csv to xlsx if required
    if exp and exp_format == dpd_options_exp_format[lang_idx][0]: # if exp_format is the first option in the dropdown menu -> XLSX
        xlsx_path = os.path.join(dst_dir, "results.xlsx")

        # check if the excel file exists, e.g. when processing both img and vid
        dfs = []
        for result_type in ['detections', 'files', 'summary']:
            csv_path = os.path.join(dst_dir, f"results_{result_type}.csv")
            if os.path.isfile(xlsx_path):

                #  if so, add new rows to existing ones
                df_xlsx = pd.read_excel(xlsx_path, sheet_name=result_type)
                df_csv = pd.read_csv(os.path.join(dst_dir, f"results_{result_type}.csv"), dtype=dtypes, low_memory=False)
                df = pd.concat([df_xlsx, df_csv], ignore_index=True)
            else:
                df = pd.read_csv(os.path.join(dst_dir, f"results_{result_type}.csv"), dtype=dtypes, low_memory=False)
            dfs.append(df)

            # plt needs the csv's, so don't remove just yet
            if not plt:
                if os.path.isfile(csv_path):
                    os.remove(csv_path)

        # overwrite rows to xlsx file
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            for idx, result_type in enumerate(['detections', 'files', 'summary']):
                df = dfs[idx]
                if result_type in ['detections', 'files']:
                    df['DateTimeOriginal'] = pd.to_datetime(df['DateTimeOriginal'], format='%d/%m/%y %H:%M:%S')
                    df['DateTime'] = pd.to_datetime(df['DateTime'], format='%d/%m/%y %H:%M:%S')
                    df['DateTimeDigitized'] = pd.to_datetime(df['DateTimeDigitized'], format='%d/%m/%y %H:%M:%S')
                df.to_excel(writer, sheet_name=result_type, index=None, header=True)

    # convert csv to tsv if required
    if exp and exp_format == dpd_options_exp_format[lang_idx][3]: # if exp_format is the third option in the dropdown menu -> TSV

        # Check if the TSV file exists, e.g., when processing both img and vid
        csv_path = os.path.join(dst_dir, f"results_detections.csv")
        tsv_path = os.path.join(dst_dir, f"results_sensing_clues.tsv")

        if os.path.isfile(tsv_path):  # Append if TSV exists
            with open(csv_path, 'r', newline='') as csv_file, open(tsv_path, 'a', newline='') as tsv_file:
                csv_reader = csv.reader(x.replace('\0', '') for x in csv_file)
                tsv_writer = csv.writer(tsv_file, delimiter='\t')

                csv_header = next(csv_reader)
                idx_date, idx_lat, idx_lon = map(csv_header.index, ["DateTimeOriginal", "Latitude", "Longitude"])

                for row in csv_reader:
                    unique_id = generate_unique_id(row)
                    formatted_date = format_datetime(row[idx_date])
                    new_row = [unique_id, formatted_date, row[idx_lat], row[idx_lon], "AddaxAI"] + row
                    tsv_writer.writerow(new_row)

        else:  # Create new TSV file
            with open(csv_path, 'r', newline='') as csv_file, open(tsv_path, 'w', newline='') as tsv_file:
                csv_reader = csv.reader(x.replace('\0', '') for x in csv_file)
                tsv_writer = csv.writer(tsv_file, delimiter='\t')

                csv_header = next(csv_reader)
                idx_date, idx_lat, idx_lon = map(csv_header.index, ["DateTimeOriginal", "Latitude", "Longitude"])

                tsv_writer.writerow(["ID", "Date", "Lat", "Long", "Method"] + csv_header)

                for row in csv_reader:
                    unique_id = generate_unique_id(row)
                    formatted_date = format_datetime(row[idx_date])
                    new_row = [unique_id, formatted_date, row[idx_lat], row[idx_lon], "AddaxAI"] + row
                    tsv_writer.writerow(new_row)

        # plt needs the CSVs, so don't remove just yet
        if not plt:
            for result_type in ['detections', 'files', 'summary']:
                csv_path = os.path.join(dst_dir, f"results_{result_type}.csv")
                if os.path.isfile(csv_path):
                    os.remove(csv_path)

    # convert csv to coco format if required
    if exp and exp_format == dpd_options_exp_format[lang_idx][2]: # COCO
        
        # init vars
        coco_path = os.path.join(dst_dir, f"results_coco_{data_type}.json")
        detections_df = pd.read_csv(os.path.join(dst_dir, f"results_detections.csv"), dtype=dtypes, low_memory=False)
        files_df = pd.read_csv(os.path.join(dst_dir, f"results_files.csv"), dtype=dtypes, low_memory=False)
        
        # convert csv to coco format
        csv_to_coco(
            detections_df=detections_df,
            files_df=files_df,
            output_path=coco_path
        )
        
        # only plt needs the csv's, so if the user didn't specify plt, remove csvs
        if not plt:
            for result_type in ['detections', 'files', 'summary']:
                csv_path = os.path.join(dst_dir, f"results_{result_type}.csv")
                if os.path.isfile(csv_path):
                    os.remove(csv_path)

    # change json paths back, if converted earlier
    if json_paths_converted:
        make_json_absolute(recognition_file)

    # let the user know it's done
    progress_window.update_values(process = f"{data_type}_pst", status = "done")
    root.update()

    # create graphs
    if plt:
        produce_plots(dst_dir)

        # if user wants XLSX (0), COCO (2), or TSV (3) as output, or if user didn't specify exp all-
        # together but the files were created for plt -> remove CSV files
        if (exp and exp_format == dpd_options_exp_format[lang_idx][0]) or \
            (exp and exp_format == dpd_options_exp_format[lang_idx][2]) or \
            (exp and exp_format == dpd_options_exp_format[lang_idx][3]) or \
            remove_csv:
            for result_type in ['detections', 'files', 'summary']:
                csv_path = os.path.join(dst_dir, f"results_{result_type}.csv")
                if os.path.isfile(csv_path):
                    os.remove(csv_path)

def clean_line(line):
    return line.replace('\0', '')

def generate_unique_id(row):
    """Generate a unique hash for a row based on its contents."""
    row_str = "".join(row).encode('utf-8')
    return hashlib.md5(row_str).hexdigest()

def format_datetime(date_str):
    """Convert 'DD/MM/YY HH:MM:SS' to 'YYYY-MM-DDTHH:MM:SS', handle 'NA' gracefully."""
    try:
        dt = datetime.datetime.strptime(date_str, "%d/%m/%y %H:%M:%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return 'NA'

# convert csv to coco format
def csv_to_coco(detections_df, files_df, output_path):
    
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}\n")
    
    # init coco structure
    coco = {
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": "NA"
            }],
        "info": {
            "description": f"Object detection results exported from AddaxAI (v{str(current_AA_version)}).",
            "url": "https://addaxdatascience.com/addaxai/",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    }

    # prepare categories and category mapping
    category_mapping = {}
    current_category_id = 1

    # assign categories from detections
    for label in detections_df['label'].unique():
        if label not in category_mapping:
            category_mapping[label] = current_category_id
            coco['categories'].append({
                "id": current_category_id,
                "name": label
                })
            current_category_id += 1

    # process each image and its detections
    annotation_id = 1
    for _, file_info in files_df.iterrows():
        
        # create image entry
        image_id = len(coco['images']) + 1
        
        # get date captured
        if type(file_info['DateTimeOriginal']) == float: # means NA value
            date_captured = "NA"
        else:
            date_captured = datetime.datetime.strptime(file_info['DateTimeOriginal'],
                                                        "%d/%m/%y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")   

        # add image to coco
        image_entry = {
            "id": image_id,
            "width": int(file_info['file_width']),
            "height": int(file_info['file_height']),
            "file_name": file_info['relative_path'],
            "license": 1,
            "date_captured": date_captured
        }
        coco['images'].append(image_entry)

        # add annotations for this image
        image_detections = detections_df[detections_df['relative_path'] == file_info['relative_path']]
        for _, detection in image_detections.iterrows():
            bbox_left = int(detection['bbox_left'])
            bbox_top = int(detection['bbox_top'])
            bbox_right = int(detection['bbox_right'])
            bbox_bottom = int(detection['bbox_bottom'])

            bbox_width = bbox_right - bbox_left
            bbox_height = bbox_bottom - bbox_top

            annotation_entry = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_mapping[detection['label']],
                "bbox": [bbox_left, bbox_top, bbox_width, bbox_height],
                "area": float(bbox_width * bbox_height),
                "iscrowd": 0
            }
            coco['annotations'].append(annotation_entry)
            annotation_id += 1

    # save when done
    with open(output_path, 'w') as output_file:
        json.dump(coco, output_file, indent=4)

# set data types for csv import so that the machine doesn't run out of memory with large files (>0.5M rows)
dtypes = {
    'absolute_path': 'str',
    'relative_path': 'str',
    'data_type': 'str',
    'label': 'str',
    'confidence': 'float64',
    'human_verified': 'bool',
    'bbox_left': 'str',
    'bbox_top': 'str',
    'bbox_right': 'str',
    'bbox_bottom': 'str',
    'file_height': 'str',
    'file_width': 'str',
    'DateTimeOriginal': 'str',
    'DateTime': 'str',
    'DateTimeDigitized': 'str',
    'Latitude': 'str',
    'Longitude': 'str',
    'GPSLink': 'str',
    'Altitude': 'str',
    'Make': 'str',
    'Model': 'str',
    'Flash': 'str',
    'ExifOffset': 'str',
    'ResolutionUnit': 'str',
    'YCbCrPositioning': 'str',
    'XResolution': 'str',
    'YResolution': 'str',
    'ExifVersion': 'str',
    'ComponentsConfiguration': 'str',
    'FlashPixVersion': 'str',
    'ColorSpace': 'str',
    'ExifImageWidth': 'str',
    'ISOSpeedRatings': 'str',
    'ExifImageHeight': 'str',
    'ExposureMode': 'str',
    'WhiteBalance': 'str',
    'SceneCaptureType': 'str',
    'ExposureTime': 'str',
    'Software': 'str',
    'Sharpness': 'str',
    'Saturation': 'str',
    'ReferenceBlackWhite': 'str',
    'n_detections': 'int64',
    'max_confidence': 'float64',
}

# create dict with country codes for speciesnet
countries = [
    "NONE   \tDon't filter by regional taxa and allow all species to be present in the results",
    "ABW    \tAruba",
    "AFG    \tAfghanistan",
    "AGO    \tAngola",
    "AIA    \tAnguilla",
    "ALA    \t\u00c5land Islands",
    "ALB    \tAlbania",
    "AND    \tAndorra",
    "ARE    \tUnited Arab Emirates",
    "ARG    \tArgentina",
    "ARM    \tArmenia",
    "ASM    \tAmerican Samoa",
    "ATA    \tAntarctica",
    "ATF    \tFrench Southern Territories",
    "ATG    \tAntigua and Barbuda",
    "AUS    \tAustralia",
    "AUT    \tAustria",
    "AZE    \tAzerbaijan",
    "BDI    \tBurundi",
    "BEL    \tBelgium",
    "BEN    \tBenin",
    "BES    \tBonaire, Sint Eustatius and Saba",
    "BFA    \tBurkina Faso",
    "BGD    \tBangladesh",
    "BGR    \tBulgaria",
    "BHR    \tBahrain",
    "BHS    \tBahamas",
    "BIH    \tBosnia and Herzegovina",
    "BLM    \tSaint Barth\u00e9lemy",
    "BLR    \tBelarus",
    "BLZ    \tBelize",
    "BMU    \tBermuda",
    "BOL    \tBolivia, Plurinational State of",
    "BRA    \tBrazil",
    "BRB    \tBarbados",
    "BRN    \tBrunei Darussalam",
    "BTN    \tBhutan",
    "BVT    \tBouvet Island",
    "BWA    \tBotswana",
    "CAF    \tCentral African Republic",
    "CAN    \tCanada",
    "CCK    \tCocos (Keeling) Islands",
    "CHE    \tSwitzerland",
    "CHL    \tChile",
    "CHN    \tChina",
    "CIV    \tC\u00f4te d'Ivoire",
    "CMR    \tCameroon",
    "COD    \tCongo, Democratic Republic of the",
    "COG    \tCongo",
    "COK    \tCook Islands",
    "COL    \tColombia",
    "COM    \tComoros",
    "CPV    \tCabo Verde",
    "CRI    \tCosta Rica",
    "CUB    \tCuba",
    "CUW    \tCura\u00e7ao",
    "CXR    \tChristmas Island",
    "CYM    \tCayman Islands",
    "CYP    \tCyprus",
    "CZE    \tCzechia",
    "DEU    \tGermany",
    "DJI    \tDjibouti",
    "DMA    \tDominica",
    "DNK    \tDenmark",
    "DOM    \tDominican Republic",
    "DZA    \tAlgeria",
    "ECU    \tEcuador",
    "EGY    \tEgypt",
    "ERI    \tEritrea",
    "ESH    \tWestern Sahara",
    "ESP    \tSpain",
    "EST    \tEstonia",
    "ETH    \tEthiopia",
    "FIN    \tFinland",
    "FJI    \tFiji",
    "FLK    \tFalkland Islands (Malvinas)",
    "FRA    \tFrance",
    "FRO    \tFaroe Islands",
    "FSM    \tMicronesia, Federated States of",
    "GAB    \tGabon",
    "GBR    \tUnited Kingdom of Great Britain and Northern Ireland",
    "GEO    \tGeorgia",
    "GGY    \tGuernsey",
    "GHA    \tGhana",
    "GIB    \tGibraltar",
    "GIN    \tGuinea",
    "GLP    \tGuadeloupe",
    "GMB    \tGambia",
    "GNB    \tGuinea-Bissau",
    "GNQ    \tEquatorial Guinea",
    "GRC    \tGreece",
    "GRD    \tGrenada",
    "GRL    \tGreenland",
    "GTM    \tGuatemala",
    "GUF    \tFrench Guiana",
    "GUM    \tGuam",
    "GUY    \tGuyana",
    "HKG    \tHong Kong",
    "HMD    \tHeard Island and McDonald Islands",
    "HND    \tHonduras",
    "HRV    \tCroatia",
    "HTI    \tHaiti",
    "HUN    \tHungary",
    "IDN    \tIndonesia",
    "IMN    \tIsle of Man",
    "IND    \tIndia",
    "IOT    \tBritish Indian Ocean Territory",
    "IRL    \tIreland",
    "IRN    \tIran, Islamic Republic of",
    "IRQ    \tIraq",
    "ISL    \tIceland",
    "ISR    \tIsrael",
    "ITA    \tItaly",
    "JAM    \tJamaica",
    "JEY    \tJersey",
    "JOR    \tJordan",
    "JPN    \tJapan",
    "KAZ    \tKazakhstan",
    "KEN    \tKenya",
    "KGZ    \tKyrgyzstan",
    "KHM    \tCambodia",
    "KIR    \tKiribati",
    "KNA    \tSaint Kitts and Nevis",
    "KOR    \tKorea, Republic of",
    "KWT    \tKuwait",
    "LAO    \tLao People's Democratic Republic",
    "LBN    \tLebanon",
    "LBR    \tLiberia",
    "LBY    \tLibya",
    "LCA    \tSaint Lucia",
    "LIE    \tLiechtenstein",
    "LKA    \tSri Lanka",
    "LSO    \tLesotho",
    "LTU    \tLithuania",
    "LUX    \tLuxembourg",
    "LVA    \tLatvia",
    "MAC    \tMacao",
    "MAF    \tSaint Martin (French part)",
    "MAR    \tMorocco",
    "MCO    \tMonaco",
    "MDA    \tMoldova, Republic of",
    "MDG    \tMadagascar",
    "MDV    \tMaldives",
    "MEX    \tMexico",
    "MHL    \tMarshall Islands",
    "MKD    \tNorth Macedonia",
    "MLI    \tMali",
    "MLT    \tMalta",
    "MMR    \tMyanmar",
    "MNE    \tMontenegro",
    "MNG    \tMongolia",
    "MNP    \tNorthern Mariana Islands",
    "MOZ    \tMozambique",
    "MRT    \tMauritania",
    "MSR    \tMontserrat",
    "MTQ    \tMartinique",
    "MUS    \tMauritius",
    "MWI    \tMalawi",
    "MYS    \tMalaysia",
    "MYT    \tMayotte",
    "NAM    \tNamibia",
    "NCL    \tNew Caledonia",
    "NER    \tNiger",
    "NFK    \tNorfolk Island",
    "NGA    \tNigeria",
    "NIC    \tNicaragua",
    "NIU    \tNiue",
    "NLD    \tNetherlands, Kingdom of the",
    "NOR    \tNorway",
    "NPL    \tNepal",
    "NRU    \tNauru",
    "NZL    \tNew Zealand",
    "OMN    \tOman",
    "PAK    \tPakistan",
    "PAN    \tPanama",
    "PCN    \tPitcairn",
    "PER    \tPeru",
    "PHL    \tPhilippines",
    "PLW    \tPalau",
    "PNG    \tPapua New Guinea",
    "POL    \tPoland",
    "PRI    \tPuerto Rico",
    "PRK    \tKorea, Democratic People's Republic of",
    "PRT    \tPortugal",
    "PRY    \tParaguay",
    "PSE    \tPalestine, State of",
    "PYF    \tFrench Polynesia",
    "QAT    \tQatar",
    "REU    \tR\u00e9union",
    "ROU    \tRomania",
    "RUS    \tRussian Federation",
    "RWA    \tRwanda",
    "SAU    \tSaudi Arabia",
    "SDN    \tSudan",
    "SEN    \tSenegal",
    "SGP    \tSingapore",
    "SGS    \tSouth Georgia and the South Sandwich Islands",
    "SHN    \tSaint Helena, Ascension and Tristan da Cunha",
    "SJM    \tSvalbard and Jan Mayen",
    "SLB    \tSolomon Islands",
    "SLE    \tSierra Leone",
    "SLV    \tEl Salvador",
    "SMR    \tSan Marino",
    "SOM    \tSomalia",
    "SPM    \tSaint Pierre and Miquelon",
    "SRB    \tSerbia",
    "SSD    \tSouth Sudan",
    "STP    \tSao Tome and Principe",
    "SUR    \tSuriname",
    "SVK    \tSlovakia",
    "SVN    \tSlovenia",
    "SWE    \tSweden",
    "SWZ    \tEswatini",
    "SXM    \tSint Maarten (Dutch part)",
    "SYC    \tSeychelles",
    "SYR    \tSyrian Arab Republic",
    "TCA    \tTurks and Caicos Islands",
    "TCD    \tChad",
    "TGO    \tTogo",
    "THA    \tThailand",
    "TJK    \tTajikistan",
    "TKL    \tTokelau",
    "TKM    \tTurkmenistan",
    "TLS    \tTimor-Leste",
    "TON    \tTonga",
    "TTO    \tTrinidad and Tobago",
    "TUN    \tTunisia",
    "TUR    \tT\u00fcrkiye",
    "TUV    \tTuvalu",
    "TWN    \tTaiwan, Province of China",
    "TZA    \tTanzania, United Republic of",
    "UGA    \tUganda",
    "UKR    \tUkraine",
    "UMI    \tUnited States Minor Outlying Islands",
    "URY    \tUruguay",
    "USA-AL    \tUnited States of America - Alabama",
    "USA-AK    \tUnited States of America - Alaska",
    "USA-AZ    \tUnited States of America - Arizona",
    "USA-AR    \tUnited States of America - Arkansas",
    "USA-CA    \tUnited States of America - California",
    "USA-CO    \tUnited States of America - Colorado",
    "USA-CT    \tUnited States of America - Connecticut",
    "USA-DE    \tUnited States of America - Delaware",
    "USA-FL    \tUnited States of America - Florida",
    "USA-GA    \tUnited States of America - Georgia",
    "USA-HI    \tUnited States of America - Hawaii",
    "USA-ID    \tUnited States of America - Idaho",
    "USA-IL    \tUnited States of America - Illinois",
    "USA-IN    \tUnited States of America - Indiana",
    "USA-IA    \tUnited States of America - Iowa",
    "USA-KS    \tUnited States of America - Kansas",
    "USA-KY    \tUnited States of America - Kentucky",
    "USA-LA    \tUnited States of America - Louisiana",
    "USA-ME    \tUnited States of America - Maine",
    "USA-MD    \tUnited States of America - Maryland",
    "USA-MA    \tUnited States of America - Massachusetts",
    "USA-MI    \tUnited States of America - Michigan",
    "USA-MN    \tUnited States of America - Minnesota",
    "USA-MS    \tUnited States of America - Mississippi",
    "USA-MO    \tUnited States of America - Missouri",
    "USA-MT    \tUnited States of America - Montana",
    "USA-NE    \tUnited States of America - Nebraska",
    "USA-NV    \tUnited States of America - Nevada",
    "USA-NH    \tUnited States of America - New Hampshire",
    "USA-NJ    \tUnited States of America - New Jersey",
    "USA-NM    \tUnited States of America - New Mexico",
    "USA-NY    \tUnited States of America - New York",
    "USA-NC    \tUnited States of America - North Carolina",
    "USA-ND    \tUnited States of America - North Dakota",
    "USA-OH    \tUnited States of America - Ohio",
    "USA-OK    \tUnited States of America - Oklahoma",
    "USA-OR    \tUnited States of America - Oregon",
    "USA-PA    \tUnited States of America - Pennsylvania",
    "USA-RI    \tUnited States of America - Rhode Island",
    "USA-SC    \tUnited States of America - South Carolina",
    "USA-SD    \tUnited States of America - South Dakota",
    "USA-TN    \tUnited States of America - Tennessee",
    "USA-TX    \tUnited States of America - Texas",
    "USA-UT    \tUnited States of America - Utah",
    "USA-VT    \tUnited States of America - Vermont",
    "USA-VA    \tUnited States of America - Virginia",
    "USA-WA    \tUnited States of America - Washington",
    "USA-WV    \tUnited States of America - West Virginia",
    "USA-WI    \tUnited States of America - Wisconsin",
    "USA-WY    \tUnited States of America - Wyoming",
    "UZB    \tUzbekistan",
    "VAT    \tHoly See",
    "VCT    \tSaint Vincent and the Grenadines",
    "VEN    \tVenezuela, Bolivarian Republic of",
    "VGB    \tVirgin Islands (British)",
    "VIR    \tVirgin Islands (U.S.)",
    "VNM    \tViet Nam",
    "VUT    \tVanuatu",
    "WLF    \tWallis and Futuna",
    "WSM    \tSamoa",
    "YEM    \tYemen",
    "ZAF    \tSouth Africa",
    "ZMB    \tZambia",
    "ZWE    \tZimbabwe"
]

# for simplicity, the same list is used for both english, spanish and french. I'll fix everything properly in the new version
dpd_options_sppnet_location = [countries, countries, countries]

# open progress window and initiate the post-process progress window
def start_postprocess():
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # save settings for next time
    write_global_vars({
        "lang_idx": lang_idx,
        "var_separate_files": var_separate_files.get(),
        "var_keep_series": var_keep_series.get(),
        "var_keep_series_seconds": var_keep_series_seconds.get(),
        "var_keep_series_species": global_vars.get('var_keep_series_species', []),
        "var_file_placement": var_file_placement.get(),
        "var_sep_conf": var_sep_conf.get(),
        "var_vis_files": var_vis_files.get(),
        "var_crp_files": var_crp_files.get(),
        "var_exp": var_exp.get(),
        "var_exp_format_idx": dpd_options_exp_format[lang_idx].index(var_exp_format.get()),
        "var_vis_size_idx": dpd_options_vis_size[lang_idx].index(var_vis_size.get()),
        "var_vis_bbox": var_vis_bbox.get(),
        "var_vis_blur": var_vis_blur.get(),
        "var_plt": var_plt.get(),
        "var_thresh": var_thresh.get()
    })
    
    # fix user input
    src_dir = var_choose_folder.get()
    dst_dir = var_output_dir.get()
    thresh = var_thresh.get()
    sep = var_separate_files.get()
    keep_series = var_keep_series.get()
    keep_series_seconds = var_keep_series_seconds.get()
    keep_series_species = global_vars.get('var_keep_series_species', [])
    file_placement = var_file_placement.get()
    sep_conf = var_sep_conf.get()
    vis = var_vis_files.get()
    crp = var_crp_files.get()
    exp = var_exp.get()
    plt = var_plt.get()
    exp_format = var_exp_format.get()

    # init cancel variable
    global cancel_var
    cancel_var = False

    # check which json files are present
    img_json = False
    if os.path.isfile(os.path.join(src_dir, "image_recognition_file.json")):
        img_json = True
    vid_json = False
    if os.path.isfile(os.path.join(src_dir, "video_recognition_file.json")):
        vid_json = True
    if not img_json and not vid_json:
        mb.showerror(error_txt[lang_idx], ["No model output file present. Make sure you run step 2 before post-processing the files.",
                                       "No hay archivo de salida del modelo. Asegúrese de ejecutar el paso 2 antes de postprocesar"
                                       " los archivos.", 
                                       "Aucun fichier de sortie du modèle présent. Assurez-vous d'exécuter l'étape 2 avant le "
                                       "post-traitement des fichiers"][lang_idx])
        return
    
    # check if destination dir is valid and set to input dir if not
    if dst_dir in ["", "/", "\\", ".", "~", ":"] or not os.path.isdir(dst_dir):
        mb.showerror(["Destination folder not set", "Carpeta de destino no establecida.", "Le répertoire de sortie n'est pas spécifié."][lang_idx],
                        ["Destination folder not set.\n\n You have not specified where the post-processing results should be placed or the set "
                        "folder does not exist. This is required.",
                        "Carpeta de destino no establecida. No ha especificado dónde deben colocarse los resultados del postprocesamiento o la "
                        "carpeta establecida no existe. Esto opción es obligatoria.",
                        "Le répertoire de sortie n'est pas spécifié. Vous n'avez pas spécifié l'emplacement où enregistrer les résultats du post-traitement "
                        "ou le répertoire n'existe pas. Ceci est obligatoire."][lang_idx])
        return

    # warn user if the original files will be overwritten with visualized files
    if os.path.normpath(dst_dir) == os.path.normpath(src_dir) and vis and not sep:
        if not mb.askyesno(["Original images will be overwritten", "Las imágenes originales se sobrescribirán.", "Les images originales seront écrasées."][lang_idx], 
                      [f"WARNING! The visualized images will be placed in the folder with the original data: '{src_dir}'. By doing this, you will overwrite the original images"
                      " with the visualized ones. Visualizing is permanent and cannot be undone. Are you sure you want to continue?",
                      f"ATENCIÓN. Las imágenes visualizadas se colocarán en la carpeta con los datos originales: '{src_dir}'. Al hacer esto, se sobrescribirán las imágenes "
                      "originales con las visualizadas. La visualización es permanente y no se puede deshacer. ¿Está seguro de que desea continuar?",
                      f"ATTENTION ! Les images visualisées seront placées dans le dossier contenant les données d'origine : « {src_dir} ». Ce faisant, vous écraserez les images d'origine par celles visualisées. "
                      "La visualisation est définitive et irréversible. Voulez-vous vraiment continuer ? "][lang_idx]):
            return
    
    # warn user if images will be moved and visualized
    if sep and file_placement == 1 and vis:
        if not mb.askyesno(["Original images will be overwritten", "Las imágenes originales se sobrescribirán.", "Les images originales seront écrasées."][lang_idx], 
                      [f"WARNING! You specified to visualize the original images. Visualizing is permanent and cannot be undone. If you don't want to visualize the original "
                      f"images, please select 'Copy' as '{lbl_file_placement_txt}'. Are you sure you want to continue with the current settings?",
                      "ATENCIÓN. Ha especificado visualizar las imágenes originales. La visualización es permanente y no puede deshacerse. Si no desea visualizar las "
                      f"imágenes originales, seleccione 'Copiar' como '{lbl_file_placement_txt}'. ¿Está seguro de que desea continuar con la configuración actual?",
                      "ATTENTION ! Vous avez spécifié de visualiser les images originales. La visualisation est définitive et irréversible. Si vous ne souhaitez pas visualiser les images originales, "
                      f"sélectionnez « Copier » au format « {lbl_file_placement_txt} ». Voulez-vous vraiment conserver les paramètres actuels ?"][lang_idx]):
            return

    # initialise progress window with processes
    processes = []
    if img_json:
        processes.append("img_pst")
    if plt:
        processes.append("plt")
    if vid_json:
        processes.append("vid_pst")
    global progress_window
    progress_window = ProgressWindow(processes = processes)
    progress_window.open()

    try:
        # postprocess images
        if img_json:
            postprocess(src_dir, dst_dir, thresh, sep, keep_series, keep_series_seconds, file_placement, sep_conf, vis, crp, exp, plt, exp_format, data_type = "img", keep_series_species=keep_series_species)

        # postprocess videos
        if vid_json and not cancel_var:
            postprocess(src_dir, dst_dir, thresh, sep, keep_series, keep_series_seconds, file_placement, sep_conf, vis, crp, exp, plt, exp_format, data_type = "vid", keep_series_species=keep_series_species)
            
        # complete
        complete_frame(fth_step)

        # check if there are postprocessing errors written
        if os.path.isfile(postprocessing_error_log): 
            mb.showwarning(warning_txt[lang_idx], [f"One or more files failed to be analysed by the model (e.g., corrupt files) and will be skipped by "
                                                f"post-processing features. See\n\n'{postprocessing_error_log}'\n\nfor more info.",
                                                f"Uno o más archivos no han podido ser analizados por el modelo (por ejemplo, ficheros corruptos) y serán "
                                                f"omitidos por las funciones de post-procesamiento. Para más información, véase\n\n'{postprocessing_error_log}'",
                                                "Un ou plusieurs fichiers n'ont pas pu être analysés par le modèle (par exemple, des fichiers corrompus) et seront"
                                                f"ignorés lors du post-traitement. Voir\n\n'{postprocessing_error_log}'\n\npour plus d'info."][lang_idx])

        # close progress window
        progress_window.close()
    
    except Exception as error:
        # log error
        print("ERROR:\n" + str(error) + "\n\nDETAILS:\n" + str(traceback.format_exc()) + "\n\n")
        
        # show error
        mb.showerror(title=error_txt[lang_idx],
                     message=["An error has occurred", "Ha ocurrido un error", "Une erreur s'est produite"][lang_idx] + " (AddaxAI v" + current_AA_version + "): '" + str(error) + "'.",
                     detail=traceback.format_exc())
        
        # close window
        progress_window.close()

# function to produce graphs and maps
def produce_plots(results_dir):

    # update internal progressbar via a tmdq stats
    def update_pbar_plt():
        pbar.update(1)
        tqdm_stats = pbar.format_dict
        progress_window.update_values(process = "plt",
                                        status = "running",
                                        cur_it = tqdm_stats['n'],
                                        tot_it = tqdm_stats['total'],
                                        time_ela = str(datetime.timedelta(seconds=round(tqdm_stats['elapsed']))),
                                        time_rem = str(datetime.timedelta(seconds=round((tqdm_stats['total'] - tqdm_stats['n']) / tqdm_stats['n'] * tqdm_stats['elapsed'] if tqdm_stats['n'] else 0))),
                                        cancel_func = cancel)

    # create all time plots
    def create_time_plots(data, save_path_base, temporal_units, pbar, counts_df):

        # maximum number of ticks per x axis
        max_n_ticks = 50

        # define specific functions per plot type
        def plot_obs_over_time_total_static(time_unit):
            plt.figure(figsize=(10, 6))
            combined_data = grouped_data.sum(axis=0).resample(time_format_mapping[time_unit]['freq']).sum()
            plt.bar(combined_data.index.strftime(time_format_mapping[time_unit]['time_format']), combined_data, width=0.9)
            plt.suptitle("")
            plt.title(f'Total observations (grouped per {time_unit}, n = {counts_df["count"].sum()})')
            plt.ylabel('Count')
            plt.xlabel(time_unit)
            plt.xticks(rotation=90)
            x_vals = np.arange(len(combined_data))
            tick_step = max(len(combined_data) // max_n_ticks, 1)
            selected_ticks = x_vals[::tick_step]
            while_iteration = 0 
            while len(selected_ticks) >= max_n_ticks:
                tick_step += 1
                while_iteration += 1
                selected_ticks = x_vals[::tick_step]
                if while_iteration > 100:
                    break
            selected_labels = combined_data.index.strftime(time_format_mapping[time_unit]['time_format'])[::tick_step]
            plt.xticks(selected_ticks, selected_labels)
            plt.tight_layout()
            save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "combined-single-layer.png")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            update_pbar_plt()

        def plot_obs_over_time_total_interactive(time_unit):
            combined_data = grouped_data.sum(axis=0).resample(time_format_mapping[time_unit]['freq']).sum()
            hover_text = [f'Period: {date}<br>Count: {count}<extra></extra>' 
                        for date, count in zip(combined_data.index.strftime(time_format_mapping[time_unit]['time_format']), 
                                                combined_data)]
            fig = go.Figure(data=[go.Bar(x=combined_data.index.strftime(time_format_mapping[time_unit]['time_format']), 
                                        y=combined_data,
                                        hovertext=hover_text,
                                        hoverinfo='text')])
            fig.update_traces(hovertemplate='%{hovertext}')
            fig.update_layout(title=f'Total observations (grouped per {time_unit})',
                            xaxis_title='Period',
                            yaxis_title='Count',
                            xaxis_tickangle=90)
            save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "combined-single-layer.html")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            update_pbar_plt()

        def plot_obs_over_time_combined_static(time_unit):
            plt.figure(figsize=(10, 6))
            for label in grouped_data.index:
                grouped_data_indexed = grouped_data.loc[label].resample(time_format_mapping[time_unit]['freq']).sum()
                plt.plot(grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format']), grouped_data_indexed, label=label)
            plt.suptitle("")
            plt.title(f'Observations over time (grouped per {time_unit}, n = {counts_df["count"].sum()})')
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            plt.xlabel(time_unit)
            plt.legend(loc='upper right')
            x_vals = np.arange(len(grouped_data_indexed))
            tick_step = max(len(grouped_data_indexed) // max_n_ticks, 1)
            selected_ticks = x_vals[::tick_step]
            while_iteration = 0 
            while len(selected_ticks) >= max_n_ticks:
                tick_step += 1
                while_iteration += 1
                selected_ticks = x_vals[::tick_step]
                if while_iteration > 100:
                    break
            selected_labels = grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format'])[::tick_step]
            plt.xticks(selected_ticks, selected_labels)
            plt.tight_layout()
            save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "combined-multi-layer.png")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            update_pbar_plt()

        def plot_obs_over_time_combined_interactive(time_unit):
            fig = go.Figure()
            for label in grouped_data.index:
                grouped_data_indexed = grouped_data.loc[label].resample(time_format_mapping[time_unit]['freq']).sum()
                fig.add_trace(go.Scatter(x=grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format']), 
                                        y=grouped_data_indexed,
                                        mode='lines',
                                        name=label))
            fig.update_layout(title=f'Observations over time (grouped per {time_unit})',
                            xaxis_title='Period',
                            yaxis_title='Count',
                            xaxis_tickangle=90,
                            legend=dict(x=0, y=1.0))
            fig.update_layout(hovermode="x unified")
            save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "combined-multi-layer.html")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            update_pbar_plt()

        def plot_obs_over_time_separate_static(label, time_unit):
            plt.figure(figsize=(10, 6))
            grouped_data_indexed = grouped_data.loc[label].resample(time_format_mapping[time_unit]['freq']).sum()
            plt.bar(grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format']), grouped_data_indexed, label=label, width=0.9)
            plt.suptitle("")
            plt.title(f'Observations over time for {label} (grouped per {time_unit}, n = {counts_df[counts_df["label"] == label]["count"].values[0]})')
            plt.ylabel('Count')
            plt.xticks(rotation=90)
            plt.xlabel(time_unit)
            x_vals = np.arange(len(grouped_data_indexed))
            tick_step = max(len(grouped_data_indexed) // max_n_ticks, 1)
            selected_ticks = x_vals[::tick_step]
            while_iteration = 0 
            while len(selected_ticks) >= max_n_ticks:
                tick_step += 1
                while_iteration += 1
                selected_ticks = x_vals[::tick_step]
                if while_iteration > 100:
                    break
            selected_labels = grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format'])[::tick_step]
            plt.xticks(selected_ticks, selected_labels)
            plt.tight_layout()
            save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "class-specific", f"{label}.png")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            update_pbar_plt()

        def plot_obs_over_time_separate_interactive(label, time_unit):
            grouped_data_indexed = grouped_data.loc[label].resample(time_format_mapping[time_unit]['freq']).sum()            
            hover_text = [f'Period: {date}<br>Count: {count}<extra></extra>' 
                        for date, count in zip(grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format']), 
                                                grouped_data_indexed)]
            fig = go.Figure(go.Bar(x=grouped_data_indexed.index.strftime(time_format_mapping[time_unit]['time_format']), 
                                    y=grouped_data_indexed,
                                    hovertext=hover_text,
                                    hoverinfo='text'))
            fig.update_traces(hovertemplate='%{hovertext}')
            fig.update_layout(title=f'Observations over time for {label} (grouped per {time_unit})',
                            xaxis_title='Period',
                            yaxis_title='Count',
                            xaxis_tickangle=90)
            save_path = os.path.join(save_path_base, "graphs", "bar-charts", time_format_mapping[time_unit]['dir'], "class-specific", f"{label}.html")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            update_pbar_plt()

        def plot_obs_over_time_heatmap_static_absolute(time_unit):
            data['Period'] = data['DateTimeOriginal'].dt.strftime(time_format_mapping[time_unit]['time_format'])
            time_range = pd.Series(pd.date_range(data['DateTimeOriginal'].min(), data['DateTimeOriginal'].max(), freq=time_format_mapping[time_unit]['freq']))
            df_time = pd.DataFrame({time_unit: time_range.dt.strftime(time_format_mapping[time_unit]['time_format'])})
            heatmap_data = data.groupby(['Period', 'label']).size().unstack(fill_value=0)
            merged_data = pd.merge(df_time, heatmap_data, left_on=time_unit, right_index=True, how='left').fillna(0)
            merged_data.set_index(time_unit, inplace=True)
            merged_data = merged_data.sort_index()
            plt.figure(figsize=(14, 8))
            ax = sns.heatmap(merged_data, cmap="Blues")
            sorted_labels = sorted(merged_data.columns)
            ax.set_xticks([i + 0.5 for i in range(len(sorted_labels))])
            ax.set_xticklabels(sorted_labels)
            plt.title(f'Temporal heatmap (absolute values, grouped per {time_unit}, n = {counts_df["count"].sum()})')
            plt.tight_layout()
            legend_text = 'Number of observations'
            ax.collections[0].colorbar.set_label(legend_text)
            save_path = os.path.join(save_path_base, "graphs", "temporal-heatmaps", time_format_mapping[time_unit]['dir'], "absolute", "temporal-heatmap.png")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            update_pbar_plt()

        def plot_obs_over_time_heatmap_static_relative(time_unit):
            data['Period'] = data['DateTimeOriginal'].dt.strftime(time_format_mapping[time_unit]['time_format'])
            time_range = pd.Series(pd.date_range(data['DateTimeOriginal'].min(), data['DateTimeOriginal'].max(), freq=time_format_mapping[time_unit]['freq']))
            df_time = pd.DataFrame({time_unit: time_range.dt.strftime(time_format_mapping[time_unit]['time_format'])})
            heatmap_data = data.groupby(['Period', 'label']).size().unstack(fill_value=0)
            normalized_data = heatmap_data.div(heatmap_data.sum(axis=0), axis=1)
            merged_data = pd.merge(df_time, normalized_data, left_on=time_unit, right_index=True, how='left').fillna(0)
            merged_data.set_index(time_unit, inplace=True)
            merged_data = merged_data.sort_index()
            plt.figure(figsize=(14, 8))
            ax = sns.heatmap(merged_data, cmap="Blues")
            sorted_labels = sorted(normalized_data.columns)
            ax.set_xticks([i + 0.5 for i in range(len(sorted_labels))])
            ax.set_xticklabels(sorted_labels)
            plt.title(f'Temporal heatmap (relative values, grouped per {time_unit}, n = {counts_df["count"].sum()})')
            plt.tight_layout()
            legend_text = 'Number of observations normalized per label'
            ax.collections[0].colorbar.set_label(legend_text)
            save_path = os.path.join(save_path_base, "graphs", "temporal-heatmaps", time_format_mapping[time_unit]['dir'], "relative", "temporal-heatmap.png")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            update_pbar_plt()

        def plot_obs_over_time_heatmap_interactive_absolute(time_unit):
            data['Period'] = data['DateTimeOriginal'].dt.strftime(time_format_mapping[time_unit]['time_format'])
            time_range = pd.Series(pd.date_range(data['DateTimeOriginal'].min(), data['DateTimeOriginal'].max(), freq=time_format_mapping[time_unit]['freq']))
            df_time = pd.DataFrame({time_unit: time_range.dt.strftime(time_format_mapping[time_unit]['time_format'])})
            heatmap_data = data.groupby(['Period', 'label']).size().unstack(fill_value=0)
            merged_data = pd.merge(df_time, heatmap_data, left_on=time_unit, right_index=True, how='left').fillna(0)
            merged_data.set_index(time_unit, inplace=True)
            heatmap_trace = go.Heatmap(z=merged_data.values,
                                    x=merged_data.columns,
                                    y=merged_data.index,
                                    customdata=merged_data.stack().reset_index().values.tolist(),
                                    colorscale='Blues',
                                    hovertemplate='Class: %{x}<br>Period: %{y}<br>Count: %{z}<extra></extra>',
                                    colorbar=dict(title='Number of<br>observations'))
            fig = go.Figure(data=heatmap_trace)
            fig.update_layout(title=f'Temporal heatmap (absolute values, grouped per {time_unit}, n = {counts_df["count"].sum()})',
                            xaxis_title='Label',
                            yaxis_title='Period',
                            yaxis={'autorange': 'reversed'})
            save_path = os.path.join(save_path_base, "graphs", "temporal-heatmaps", time_format_mapping[time_unit]['dir'], "absolute", "temporal-heatmap.html")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            update_pbar_plt()

        def plot_obs_over_time_heatmap_interactive_relative(time_unit):
            data['Period'] = data['DateTimeOriginal'].dt.strftime(time_format_mapping[time_unit]['time_format'])
            time_range = pd.date_range(data['DateTimeOriginal'].min(), data['DateTimeOriginal'].max(), freq=time_format_mapping[time_unit]['freq'])
            df_time = pd.DataFrame({time_unit: time_range.strftime(time_format_mapping[time_unit]['time_format'])})
            heatmap_data = data.groupby(['Period', 'label']).size().unstack(fill_value=0)
            merged_data = pd.merge(df_time, heatmap_data, left_on=time_unit, right_index=True, how='left').fillna(0)
            merged_data.set_index(time_unit, inplace=True)
            normalized_data = merged_data.div(merged_data.sum(axis=0), axis=1)
            heatmap_trace = go.Heatmap(
                z=normalized_data.values,
                x=normalized_data.columns,
                y=normalized_data.index,
                customdata=normalized_data.stack().reset_index().values.tolist(),
                colorscale='Blues',
                hovertemplate='Class: %{x}<br>Period: %{y}<br>Normalized count: %{z}<extra></extra>',
                colorbar=dict(title='Number of<br>observations<br>normalized<br>per label'))
            fig = go.Figure(data=heatmap_trace)
            fig.update_layout(
                title=f'Temporal heatmap (relative values, grouped per {time_unit}, n = {counts_df["count"].sum()}))',
                xaxis_title='Label',
                yaxis_title='Period',
                yaxis={'autorange': 'reversed'})
            save_path = os.path.join(save_path_base, "graphs", "temporal-heatmaps", time_format_mapping[time_unit]['dir'], "relative", "temporal-heatmap.html")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            update_pbar_plt()

        # init vars
        time_format_mapping = {
            "year": {'freq': 'Y', 'time_format': '%Y', 'dir': "grouped-by-year"},
            "month": {'freq': 'M', 'time_format': '%Y-%m', 'dir': "grouped-by-month"},
            "week": {'freq': 'W', 'time_format': '%Y-wk %U', 'dir': "grouped-by-week"},
            "day": {'freq': 'D', 'time_format': '%Y-%m-%d', 'dir': "grouped-by-day"}
        }

        # group data per label
        grouped_data = data.groupby(['label', pd.Grouper(key='DateTimeOriginal', freq=f'1D')]).size().unstack(fill_value=0)

        # create plots
        for time_unit in temporal_units:
            plot_obs_over_time_total_static(time_unit);plt.close('all')
            plot_obs_over_time_total_interactive(time_unit);plt.close('all')
            plot_obs_over_time_combined_static(time_unit);plt.close('all')
            plot_obs_over_time_combined_interactive(time_unit);plt.close('all')
            plot_obs_over_time_heatmap_static_absolute(time_unit);plt.close('all')
            plot_obs_over_time_heatmap_static_relative(time_unit);plt.close('all')
            plot_obs_over_time_heatmap_interactive_absolute(time_unit);plt.close('all')
            plot_obs_over_time_heatmap_interactive_relative(time_unit);plt.close('all')
            for label in grouped_data.index:
                plot_obs_over_time_separate_static(label, time_unit);plt.close('all')
                plot_obs_over_time_separate_interactive(label, time_unit);plt.close('all')

    # activity plots
    def create_activity_patterns(df, save_path_base, pbar):
        
        # format df
        df['DateTimeOriginal'] = pd.to_datetime(df['DateTimeOriginal'])
        grouped_data = df.groupby(['label', pd.Grouper(key='DateTimeOriginal', freq=f'1D')]).size().unstack(fill_value=0)
        df['Hour'] = df['DateTimeOriginal'].dt.hour
        hourly_df = df.groupby(['label', 'Hour']).size().reset_index(name='count')
        df['Month'] = df['DateTimeOriginal'].dt.month
        monthly_df = df.groupby(['label', 'Month']).size().reset_index(name='count')

        # for static activity plots
        def plot_static_activity_pattern(df, unit, label=''):
            if label != '':
                df = df[df['label'] == label]
            total_observations = df['count'].sum()
            plt.figure(figsize=(10, 6))

            if unit == "Hour":
                x_ticks = range(24)
                x_tick_labels = [f'{x:02}-{(x + 1) % 24:02}' for x in x_ticks]
            else:
                x_ticks = range(1, 13)
                x_tick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            plt.bar(df[unit], df['count'], width=0.9, align='center')
            plt.xlabel(unit)
            plt.ylabel('Number of observations')
            plt.title(f'Activity pattern of {label if label != "" else "all animals combined"} by {"hour of the day" if unit == "Hour" else "month of the year"} (n = {total_observations})')
            plt.xticks(x_ticks, x_tick_labels, rotation=90)
            plt.tight_layout()
            if label != '':
                save_path = os.path.join(save_path_base, "graphs", "activity-patterns", "hour-of-day" if unit == "Hour" else "month-of-year", "class-specific", f"{label}.png")
            else:
                save_path = os.path.join(save_path_base, "graphs", "activity-patterns", "hour-of-day" if unit == "Hour" else "month-of-year", f"combined.png")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
            plt.close()
            update_pbar_plt()

        # for dynamic activity plots
        def plot_dynamic_activity_pattern(df, unit, label=''):
            if label != '':
                df = df[df['label'] == label]
            n_ticks = 24 if unit == "Hour" else 12
            if unit == "Hour":
                x_ticks = list(range(24))
                x_tick_labels = [f'{x:02}-{(x + 1) % 24:02}' for x in x_ticks]
            else:
                x_ticks = list(range(12))
                x_tick_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                df.loc[:, 'Month'] = df['Month'].map({i: calendar.month_abbr[i] for i in range(1, 13)})
            df = df.groupby(unit, as_index=False)['count'].sum()
            if unit == "Month":
                all_months = pd.DataFrame({
                    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                })
                merged_df = all_months.merge(df, on='Month', how='left')
                merged_df['count'] = merged_df['count'].fillna(0)
                merged_df['count'] = merged_df['count'].astype(int)
                df = merged_df
            else:
                df = df.set_index(unit).reindex(range(n_ticks), fill_value=0).reset_index()
            total_observations = df['count'].sum()            
            fig = px.bar(df, x=unit, y='count', title=f'Activity pattern of {label if label != "" else "all animals combined"} by {"hour of the day" if unit == "Hour" else "month of the year"} (n = {total_observations})').update_traces(width = 0.7)
            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=x_ticks,
                    ticktext=x_tick_labels
                ),
                xaxis_title=unit,
                yaxis_title='Count',
                bargap=0.1
            )
            if label != '':
                save_path = os.path.join(save_path_base, "graphs", "activity-patterns", "hour-of-day" if unit == "Hour" else "month-of-year", "class-specific", f"{label}.html")
            else:
                save_path = os.path.join(save_path_base, "graphs", "activity-patterns", "hour-of-day" if unit == "Hour" else "month-of-year", f"combined.html")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            pbar.update(1)

        # run class-specific
        for label in grouped_data.index:
            plot_static_activity_pattern(hourly_df, "Hour", label);plt.close('all')
            plot_static_activity_pattern(monthly_df, "Month", label);plt.close('all')
            plot_dynamic_activity_pattern(hourly_df, "Hour", label);plt.close('all')
            plot_dynamic_activity_pattern(monthly_df, "Month", label);plt.close('all')
        
        # run combined
        plot_static_activity_pattern(hourly_df, "Hour", "");plt.close('all')
        plot_static_activity_pattern(monthly_df, "Month", "");plt.close('all')
        plot_dynamic_activity_pattern(hourly_df, "Hour", "");plt.close('all')
        plot_dynamic_activity_pattern(monthly_df, "Month", "");plt.close('all')

    # heatmaps and markers
    def create_geo_plots(data, save_path_base, pbar):

        # define specific functions per plot type
        def create_combined_multi_layer_clustermap(data, save_path_base):
            if len(data) == 0:
                return
            map_path = os.path.join(save_path_base, "graphs", "maps")
            unique_labels = data['label'].unique()
            checkboxes = {label: folium.plugins.MarkerCluster(name=label) for label in unique_labels}
            for label in unique_labels:
                label_data = data[data['label'] == label]
                max_lat, min_lat = label_data['Latitude'].max(), label_data['Latitude'].min()
                max_lon, min_lon = label_data['Longitude'].max(), label_data['Longitude'].min()
                center_lat, center_lon = label_data['Latitude'].mean(), label_data['Longitude'].mean()
                m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
                m.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])
                for _, row in label_data.iterrows():
                    folium.Marker(location=[row['Latitude'], row['Longitude']]).add_to(checkboxes[label])
                folium.TileLayer('openstreetmap').add_to(m)
                folium.LayerControl().add_to(m)
                Draw(export=True).add_to(m)
            max_lat, min_lat = data['Latitude'].max(), data['Latitude'].min()
            max_lon, min_lon = data['Longitude'].max(), data['Longitude'].min()
            center_lat, center_lon = data['Latitude'].mean(), data['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            m.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])
            for label, marker_cluster in checkboxes.items():
                marker_cluster.add_to(m)
            folium.LayerControl(collapsed=False).add_to(m)
            Draw(export=True).add_to(m)
            combined_multi_layer_file = os.path.join(map_path, "combined-multi-layer.html")
            Path(os.path.dirname(combined_multi_layer_file)).mkdir(parents=True, exist_ok=True)
            m.save(combined_multi_layer_file)
            update_pbar_plt()

        # this creates a heatmap layer and a clustermarker layer which can be dynamically enabled
        def create_obs_over_geo_both_heat_and_mark(data, save_path_base, category = ''):
            if category != '':
                data = data[data['label'] == category]
            data = data.dropna(subset=['Latitude', 'Longitude'])
            if len(data) == 0:
                return
            map_path = os.path.join(save_path_base, "graphs", "maps")
            max_lat, min_lat = data['Latitude'].max(), data['Latitude'].min()
            max_lon, min_lon = data['Longitude'].max(), data['Longitude'].min()
            center_lat, center_lon = data['Latitude'].mean(), data['Longitude'].mean()
            m = folium.Map(location=[center_lat, center_lon], zoom_start=10)
            m.fit_bounds([(min_lat, min_lon), (max_lat, max_lon)])
            folium.TileLayer('OpenStreetMap', overlay=False).add_to(m)
            Draw(export=True).add_to(m)
            heatmap_layer = folium.FeatureGroup(name='Heatmap', show=True, overlay=True).add_to(m)
            cluster_layer = MarkerCluster(name='Markers', show=False, overlay=True).add_to(m)
            HeatMap(data[['Latitude', 'Longitude']]).add_to(heatmap_layer)
            for _, row in data.iterrows():
                folium.Marker(location=[row['Latitude'], row['Longitude']]).add_to(cluster_layer)
            folium.LayerControl(collapsed=False).add_to(m)
            if category != '':
                map_file = os.path.join(map_path, "class-specific", f"{category}.html")
            else:
                map_file = os.path.join(map_path, 'combined-single-layer.html')
            Path(os.path.dirname(map_file)).mkdir(parents=True, exist_ok=True)
            m.save(map_file)
            update_pbar_plt()
        
        # create plots 
        create_obs_over_geo_both_heat_and_mark(data, save_path_base);plt.close('all')
        create_combined_multi_layer_clustermap(data, save_path_base);plt.close('all')
        for label in data['label'].unique():
            create_obs_over_geo_both_heat_and_mark(data, save_path_base, label);plt.close('all')

    # create pie charts with distributions
    def create_pie_plots_detections(df, results_dir, pbar):

        # def nested function
        def create_pie_chart_detections_static():
            label_counts = df['label'].value_counts()
            total_count = len(df['label'])
            percentages = label_counts / total_count * 100
            hidden_categories = list(percentages[percentages < 0].index)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            wedges, _, autotexts = ax1.pie(label_counts, autopct='', startangle=140)
            ax1.axis('equal')
            for i, autotext in enumerate(autotexts):
                if label_counts.index[i] in hidden_categories:
                    autotext.set_visible(False)
            legend_labels = ['%s (n = %s, %.1f%%)' % (label, count, (float(count) / len(df['label'])) * 100) for label, count in zip(label_counts.index, label_counts)]
            ax2.legend(wedges, legend_labels, loc="center", fontsize='medium')
            ax2.axis('off')
            for autotext in autotexts:
                autotext.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'))
            fig.suptitle(f"Distribution of detections (n = {total_count})", fontsize=16, y=0.95)
            plt.subplots_adjust(wspace=0.1)
            plt.tight_layout()
            save_path = os.path.join(results_dir, "graphs", "pie-charts", "distribution-detections.png")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            update_pbar_plt()

        def create_pie_chart_detections_interactive():
            grouped_df = df.groupby('label').size().reset_index(name='count')
            total_count = grouped_df['count'].sum()
            grouped_df['percentage'] = (grouped_df['count'] / total_count) * 100
            grouped_df['percentage'] = grouped_df['percentage'].round(2).astype(str) + '%'
            fig = px.pie(grouped_df, names='label', values='count', title=f"Distribution of detections (n = {total_count})", hover_data={'percentage'})
            fig.update_traces(textinfo='label')
            save_path = os.path.join(results_dir, "graphs", "pie-charts", "distribution-detections.html")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            update_pbar_plt()

        # run
        create_pie_chart_detections_static();plt.close('all')
        create_pie_chart_detections_interactive();plt.close('all')

    # create pie charts with distributions
    def create_pie_plots_files(df, results_dir, pbar):

        # def nested function
        def create_pie_chart_files_static():
            df['label'] = df['n_detections'].apply(lambda x: 'detection' if x >= 1 else 'empty')
            label_counts = df['label'].value_counts()
            total_count = len(df['label'])
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            def autopct_func(pct):
                if pct > 0:
                    return f'{pct:.1f}%'
                else:
                    return ''
            labels = [label for label in label_counts.index]
            wedges, texts, autotexts = ax1.pie(label_counts, labels=labels, autopct=autopct_func, startangle=140)
            ax1.axis('equal')
            legend_labels = ['%s (n = %s, %.1f%%)' % (label, count, (float(count) / len(df['label'])) * 100) for label, count in zip(label_counts.index, label_counts)]
            ax2.legend(wedges, legend_labels, loc="center", fontsize='medium')
            ax2.axis('off')
            for autotext in autotexts:
                autotext.set_bbox(dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'))
            fig.suptitle(f"Distribution of files (n = {total_count})", fontsize=16, y=0.95)
            plt.subplots_adjust(wspace=0.5)
            save_path = os.path.join(results_dir, "graphs", "pie-charts", "distribution-files.png")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path)
            update_pbar_plt()

        def create_pie_chart_files_interactive():
            df['label'] = df['n_detections'].apply(lambda x: 'detection' if x >= 1 else 'empty')
            grouped_df = df.groupby('label').size().reset_index(name='count')
            total_count = grouped_df['count'].sum()
            grouped_df['percentage'] = (grouped_df['count'] / total_count) * 100
            grouped_df['percentage'] = grouped_df['percentage'].round(2).astype(str) + '%'
            fig = px.pie(grouped_df, names='label', values='count', title=f"Distribution of files (n = {total_count})", hover_data={'percentage'})
            fig.update_traces(textinfo='label')
            save_path = os.path.join(results_dir, "graphs", "pie-charts", "distribution-files.html")
            Path(os.path.dirname(save_path)).mkdir(parents=True, exist_ok=True)
            fig.write_html(save_path)
            update_pbar_plt()

        # run
        create_pie_chart_files_static();plt.close('all')
        create_pie_chart_files_interactive();plt.close('all')

    # overlay logo
    def overlay_logo(image_path, logo):
        main_image = Image.open(image_path)
        main_width, main_height = main_image.size
        logo_width, logo_height = logo.size
        position = (main_width - logo_width - 10, 10)
        main_image.paste(logo, position, logo)
        main_image.save(image_path)

    # check the time difference in the dataset
    def calculate_time_span(df):
        any_dates_present = df['DateTimeOriginal'].notnull().any()
        if not any_dates_present:
            return 0, 0, 0, 0
        first_date = df['DateTimeOriginal'].min()
        last_date = df['DateTimeOriginal'].max()
        time_difference = last_date - first_date
        days = time_difference.days
        years = int(days / 365)
        months = int(days / 30)
        weeks = int(days / 7)
        return years, months, weeks, days

    # main code to plot graphs
    results_dir = os.path.normpath(results_dir)
    plots_dir = os.path.join(results_dir, "graphs")
    det_df = pd.read_csv(os.path.join(results_dir, "results_detections.csv"))
    fil_df = pd.read_csv(os.path.join(results_dir, "results_files.csv"))

    # for the temporal plots we need to check the number of units
    det_df['DateTimeOriginal'] = pd.to_datetime(det_df['DateTimeOriginal'], format='%d/%m/%y %H:%M:%S')
    n_years, n_months, n_weeks, n_days = calculate_time_span(det_df)

    # to limit unnecessary computing only plot units if they have a minimum of 2 and a maximum of *max_units* units
    temporal_units = []
    max_units = 100
    if n_years > 1:
        temporal_units.append("year")
    if 1 < n_months <= max_units:
        temporal_units.append("month")
    if 1 < n_weeks <= max_units:
        temporal_units.append("week")
    if 1 < n_days <= max_units:
        temporal_units.append("day")
    print(f"Years: {n_years}, Months: {n_months}, Weeks: {n_weeks}, Days: {n_days}")
    print(f"temporal_units : {temporal_units}")

    # check if we have geo tags in the data
    det_df_geo = det_df[(det_df['Latitude'].notnull()) & (det_df['Longitude'].notnull())]
    if len(det_df_geo) > 0:
        data_permits_map_creation = True
        n_categories_geo = len(det_df_geo['label'].unique())
    else:
        data_permits_map_creation = False
        n_categories_geo = 0

    # calculate the number of plots to be created
    any_dates_present = det_df['DateTimeOriginal'].notnull().any()
    n_categories_with_timestamps = len(det_df[det_df['DateTimeOriginal'].notnull()]['label'].unique())
    n_obs_per_label_with_timestamps = det_df[det_df['DateTimeOriginal'].notnull()] .groupby('label').size().reset_index(name='count')
    activity_patterns_n_plots = (((n_categories_with_timestamps * 2) + 2) * 2) if any_dates_present else 0 
    bar_charts_n_plots = (((n_categories_with_timestamps * 2) + 4) * len(temporal_units)) if any_dates_present else 0 
    maps_n_plots = (n_categories_geo + 2) if data_permits_map_creation else 0
    pie_charts_n_plots = 4 
    temporal_heatmaps_n_plots = (4 * len(temporal_units)) if any_dates_present else 0 
    n_plots = (activity_patterns_n_plots + bar_charts_n_plots + maps_n_plots + pie_charts_n_plots + temporal_heatmaps_n_plots)

    # create plots
    with tqdm(total=n_plots, disable=False) as pbar:
        progress_window.update_values(process = f"plt", status = "load")
        if any_dates_present: create_time_plots(det_df, results_dir, temporal_units, pbar, n_obs_per_label_with_timestamps);plt.close('all')
        if cancel_var: return
        if data_permits_map_creation:
            create_geo_plots(det_df_geo, results_dir, pbar);plt.close('all')
        if cancel_var: return
        create_pie_plots_detections(det_df, results_dir, pbar);plt.close('all')
        if cancel_var: return
        create_pie_plots_files(fil_df, results_dir, pbar);plt.close('all')
        if cancel_var: return
        if any_dates_present: create_activity_patterns(det_df, results_dir, pbar);plt.close('all')
        if cancel_var: return

    # add addaxai logo
    logo_for_graphs = PIL_logo_incl_text.resize((int(LOGO_WIDTH/1.2), int(LOGO_HEIGHT/1.2)))
    for root, dirs, files in os.walk(plots_dir):
        for file in files:
            if file.endswith(".png"):
                image_path = os.path.join(root, file)
                overlay_logo(image_path, logo_for_graphs)
    
    # end pbar
    progress_window.update_values(process = f"plt", status = "done")

# open human-in-the-loop verification windows
def open_annotation_windows(recognition_file, class_list_txt, file_list_txt, label_map):

    # check if file list exists
    if not os.path.isfile(file_list_txt):
        mb.showerror(["No images to verify", "No hay imágenes para verificar", "Aucune image à vérifier"][lang_idx],
                     ["There are no images to verify with the selected criteria. Use the 'Update counts' button to see how many "
                     "images you need to verify with the selected criteria.", "No hay imágenes para verificar con los criterios "
                     "seleccionados. Utilice el botón 'Actualizar recuentos' para ver cuántas imágenes necesita verificar con "
                     "los criterios seleccionados.",
                     "Il n'y a aucune image à vérifier selon les critères sélectionnés. Utilisez le bouton « Mettre à jour le nombre » pour " 
                     "voir le nombre d'images à vérifier selon les critères sélectionnés."][lang_idx])
        return

    # check number of images to verify
    total_n_files = 0
    with open(file_list_txt) as f:
        for line in f:
            total_n_files += 1
    if total_n_files == 0:
        mb.showerror(["No images to verify", "No hay imágenes para verificar", "Aucune image à vérifier"][lang_idx],
                     ["There are no images to verify with the selected criteria. Use the 'Update counts' button to see how many "
                     "images you need to verify with the selected criteria.", "No hay imágenes para verificar con los criterios "
                     "seleccionados. Utilice el botón 'Actualizar recuentos' para ver cuántas imágenes necesita verificar con "
                     "los criterios seleccionados.",
                     "Il n'y a aucune image à vérifier selon les critères sélectionnés. Utilisez le bouton « Mettre à jour le nombre » pour " 
                     "voir le nombre d'images à vérifier selon les critères sélectionnés."][lang_idx])
        return
    
    # TODO: progressbars are not in front of other windows
    # check corrupted images # TODO: this needs to be included in the progressbar
    corrupted_images = check_images(file_list_txt)

    # fix images # TODO: this needs to be included in the progressbar
    if len(corrupted_images) > 0:
            if mb.askyesno(["Corrupted images found", "Imágenes corruptas encontradas", "Des images corrompues ont été trouvées"][lang_idx],
                            [f"There are {len(corrupted_images)} images corrupted. Do you want to repair?",
                            f"Hay {len(corrupted_images)} imágenes corruptas. Quieres repararlas?",
                            f"{len(corrupted_images)} images sont corrompues. Voulez-vous les réparer?"][lang_idx]):
                fix_images(corrupted_images)

    # read label map from json
    label_map = fetch_label_map_from_json(recognition_file)
    inverted_label_map = {v: k for k, v in label_map.items()}

    # count n verified files and locate images that need converting
    n_verified_files = 0
    if get_hitl_var_in_json(recognition_file) != "never-started":
        init_dialog = PatienceDialog(total = total_n_files, text = ["Initializing...", "Inicializando...", "Initialisation..."][lang_idx])
        init_dialog.open()
        init_current = 1
        imgs_needing_converting = []
        with open(file_list_txt) as f:
            for line in f:
                img = line.rstrip()
                annotation = return_xml_path(img)

                # check which need converting to json
                if check_if_img_needs_converting(img):
                    imgs_needing_converting.append(img)

                # check how many are verified
                if verification_status(annotation):
                    n_verified_files += 1

                # update progress window
                init_dialog.update_progress(current = init_current, percentage = True)
                init_current += 1
        init_dialog.close()

    # track hitl progress in json
    change_hitl_var_in_json(recognition_file, "in-progress")

    # close settings window if open
    try:
        hitl_settings_window.destroy()
    except NameError:
        print("hitl_settings_window not defined -> nothing to destroy()")
        
    # init window
    hitl_progress_window = customtkinter.CTkToplevel(root)
    hitl_progress_window.title(["Manual check overview", "Verificación manual", "Vérification manuelle"][lang_idx])
    hitl_progress_window.geometry("+10+10")

    # explanation frame
    hitl_explanation_frame = LabelFrame(hitl_progress_window, text=[" Explanation ", " Explicación ", "Explication"][lang_idx],
                                            pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg=green_primary)
    hitl_explanation_frame.configure(font=(text_font, 15, "bold"))
    hitl_explanation_frame.grid(column=0, row=1, columnspan=2, sticky='ew')
    hitl_explanation_frame.columnconfigure(0, weight=3, minsize=115)
    hitl_explanation_frame.columnconfigure(1, weight=1, minsize=115)

    # explanation text
    text_hitl_explanation_frame = Text(master=hitl_explanation_frame, wrap=WORD, width=1, height=15 * explanation_text_box_height_factor) 
    text_hitl_explanation_frame.grid(column=0, row=0, columnspan=5, padx=5, pady=5, sticky='ew')
    text_hitl_explanation_frame.tag_config('explanation', font=f'{text_font} {int(13 * text_size_adjustment_factor)} normal', lmargin1=10, lmargin2=10)
    text_hitl_explanation_frame.insert(END, ["This is where you do the actual verification. You'll have to make sure that all objects in all images are correctly "
                                            "labeled. That also includes classes that you did not select but are on the image by chance. If an image is verified, "
                                            "you'll have to let AddaxAI know by pressing the space bar. If all images are verified and up-to-date, you can close "
                                            "the window. AddaxAI will prompt you for the final step. You can also close the window and continue at a later moment.", 
                                            "Deberá asegurarse de que todos los objetos en todas las imágenes estén "
                                            "etiquetados correctamente. Eso también incluye clases que no seleccionaste pero que están en la imagen por casualidad. "
                                            "Si se verifica una imagen, deberá informar a AddaxAI presionando la barra espaciadora. Si todas las imágenes están "
                                            "verificadas y actualizadas, puede cerrar la ventana. AddaxAI le indicará el paso final. También puedes cerrar la "
                                            "ventana y continuar en otro momento.",
                                            "C'est ici que vous effectuez la vérification proprement dite. Vous devrez vous assurer que tous les objets de toutes "
                                            "les images sont correctement étiquetées. Cela inclut également les classes que vous n'avez pas sélectionnées, mais qui "
                                            "se trouvent sur l'image par hasard. Si une image est vérifiée, vous devrez en informer AddaxAI en appuyant sur la barre "
                                            "d'espacement. Si toutes les images sont vérifiées et à jour, vous pouvez fermer la fenêtre. AddaxAI vous invitera à effectuer "
                                            "la dernière étape. Vous pouvez également fermer la fenêtre et continuer ultérieurement."][lang_idx])
    text_hitl_explanation_frame.tag_add('explanation', '1.0', '1.end')

    # shortcuts frame
    hitl_shortcuts_frame = LabelFrame(hitl_progress_window, text=[" Shortcuts ", " Atajos ", "Raccourci "][lang_idx],
                                        pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg=green_primary)
    hitl_shortcuts_frame.configure(font=(text_font, 15, "bold"))
    hitl_shortcuts_frame.grid(column=0, row=2, columnspan=2, sticky='ew')
    hitl_shortcuts_frame.columnconfigure(0, weight=3, minsize=115)
    hitl_shortcuts_frame.columnconfigure(1, weight=1, minsize=115)

    # shortcuts label
    shortcut_labels = [["Next image:", "Previous image:", "Create box:", "Edit box:", "Delete box:", "Verify, save, and next image:"],
                       ["Imagen siguiente:", "Imagen anterior:", "Crear cuadro:", "Editar cuadro:", "Eliminar cuadro:", "Verificar, guardar, y siguiente imagen:"],
                       ["Image suivante :", "Image précédente :", "Créer une zone :", "Modifier la zone :", "Supprimer la zone :", "Vérifier, enregistrer et image suivante :" ]][lang_idx]
    shortcut_values = ["d", "a", "w", "s", "del", ["space", "espacio", "espace"][lang_idx]]
    for i in range(len(shortcut_labels)):
        ttk.Label(master=hitl_shortcuts_frame, text=shortcut_labels[i]).grid(column=0, row=i, columnspan=1, sticky='w')
        ttk.Label(master=hitl_shortcuts_frame, text=shortcut_values[i]).grid(column=1, row=i, columnspan=1, sticky='e')

    # numbers frame
    hitl_stats_frame = LabelFrame(hitl_progress_window, text=[" Progress ", " Progreso ", " Progression "][lang_idx],
                                    pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg=green_primary)
    hitl_stats_frame.configure(font=(text_font, 15, "bold"))
    hitl_stats_frame.grid(column=0, row=3, columnspan=2, sticky='ew')
    hitl_stats_frame.columnconfigure(0, weight=3, minsize=115)
    hitl_stats_frame.columnconfigure(1, weight=1, minsize=115)

    # progress bar 
    hitl_progbar = ttk.Progressbar(master=hitl_stats_frame, orient='horizontal', mode='determinate', length=280)
    hitl_progbar.grid(column=0, row=0, columnspan=2, padx=5, pady=(3,0))

    # percentage done
    lbl_hitl_stats_percentage = ttk.Label(master=hitl_stats_frame, text=["Percentage done:", "Porcentaje realizado:", "Pourcentage complété:"][lang_idx])
    lbl_hitl_stats_percentage.grid(column=0, row=1, columnspan=1, sticky='w')
    value_hitl_stats_percentage = ttk.Label(master=hitl_stats_frame, text="")
    value_hitl_stats_percentage.grid(column=1, row=1, columnspan=1, sticky='e')

    # total n images to verify
    lbl_hitl_stats_verified = ttk.Label(master=hitl_stats_frame, text=["Files verified:", "Archivos verificados:", "Fichiers vérifiés"][lang_idx])
    lbl_hitl_stats_verified.grid(column=0, row=2, columnspan=1, sticky='w')
    value_hitl_stats_verified = ttk.Label(master=hitl_stats_frame, text="")
    value_hitl_stats_verified.grid(column=1, row=2, columnspan=1, sticky='e')

    # show window
    percentage = round((n_verified_files/total_n_files)*100)
    hitl_progbar['value'] = percentage
    value_hitl_stats_percentage.configure(text = f"{percentage}%")
    value_hitl_stats_verified.configure(text = f"{n_verified_files}/{total_n_files}")
    hitl_progress_window.update_idletasks()
    hitl_progress_window.update()

    # init paths
    labelImg_dir = os.path.join(AddaxAI_files, "Human-in-the-loop")
    labelImg_script = os.path.join(labelImg_dir, "labelImg.py")
    python_executable = get_python_interprator("base")

    # create command
    command_args = []
    command_args.append(python_executable)
    command_args.append(labelImg_script)
    command_args.append(class_list_txt)
    command_args.append(file_list_txt)

    # adjust command for unix OS
    if os.name != 'nt':
        command_args = "'" + "' '".join(command_args) + "'"

    # prepend os-specific commands
    platform_name = platform.system().lower()
    if platform_name == 'darwin' and 'arm64' in platform.machine():
        print("This is an Apple Silicon system.")
        command_args =  "arch -arm64 " + command_args

    # log command
    print(command_args)

    # run command
    p = Popen(command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                shell=True,
                universal_newlines=True)

    # read the output
    for line in p.stdout:
        print(line, end='')

        if "<EA>" in line:
            ver_diff = re.search('<EA>(.)<EA>', line).group().replace('<EA>', '')

            # adjust verification count
            if ver_diff == '+':
                n_verified_files += 1
            elif ver_diff == '-':
                n_verified_files -= 1

            # update labels
            percentage = round((n_verified_files/total_n_files)*100)
            hitl_progbar['value'] = percentage
            value_hitl_stats_percentage.configure(text = f"{percentage}%")
            value_hitl_stats_verified.configure(text = f"{n_verified_files}/{total_n_files}")

            # show window
            hitl_progress_window.update()
        
        # set save status
        try:
            hitl_progress_window.update_idletasks()
            hitl_progress_window.update()
        
        # python can throw a TclError if user closes the window because the widgets are destroyed - nothing to worry about
        except Exception as error:
            print("\nWhen closing the annotation window, there was an error. python can throw a TclError if user closes "
                                                "the window because the widgets are destroyed - nothing to worry about.")
            print("ERROR:\n" + str(error) + "\n\nDETAILS:\n" + str(traceback.format_exc()) + "\n\n")

    # close accompanying window
    hitl_progress_window.destroy()
    bind_scroll_to_deploy_canvas()

    # update frames of root
    update_frame_states()

    # check if the json has relative paths
    if check_json_paths(recognition_file) == "relative":
        json_paths_are_relative = True
    else:
        json_paths_are_relative = False

    # open patience window
    # TODO: dit moet een progresswindow worden die heen en weer gaat. Maar daar heb ik een grote json voor nodig.
    converting_patience_dialog = PatienceDialog(total = 1,
                                                text = ["Running verification...", "Verificación de funcionamiento...", "Vérification en cours..."][lang_idx])
    converting_patience_dialog.open()

    # check which images need converting
    imgs_needing_converting = []
    with open(file_list_txt) as f:
        for line in f:
            img = line.rstrip()
            annotation = return_xml_path(img)
            if check_if_img_needs_converting(img):
                imgs_needing_converting.append(img)
    converting_patience_dialog.update_progress(current = 1)
    converting_patience_dialog.close()

    # open json
    with open(recognition_file, "r") as image_recognition_file_content:
        n_img_in_json = len(json.load(image_recognition_file_content)['images'])

    # open patience window
    patience_dialog = PatienceDialog(total = len(imgs_needing_converting) + n_img_in_json, text = ["Checking results...", "Comprobando resultados...", "Vérification des résultats..."][lang_idx])
    patience_dialog.open()
    current = 1

    # convert
    update_json_from_img_list(imgs_needing_converting, inverted_label_map, recognition_file, patience_dialog, current)
    current += len(imgs_needing_converting)

    # open json
    with open(recognition_file, "r") as image_recognition_file_content:
        data = json.load(image_recognition_file_content)

    # check if there are images that the user first verified and then un-verified
    for image in data['images']:
        image_path = image['file']
        patience_dialog.update_progress(current = current, percentage = True)
        current += 1
        if json_paths_are_relative:
            image_path = os.path.join(os.path.dirname(recognition_file), image_path)
        if 'manually_checked' in image:
            if image['manually_checked']:
                # image has been manually checked in json ...
                xml_path = return_xml_path(image_path)
                if os.path.isfile(xml_path):
                    # ... but not anymore in xml
                    if not verification_status(xml_path):
                        # set check flag in json
                        image['manually_checked'] = False
                        # reset confidence from 1.0 to arbitrary value 
                        if 'detections' in image:
                            for detection in image['detections']:
                                detection['conf'] = 0.7
    
    # write json
    image_recognition_file_content.close()
    with open(recognition_file, "w") as json_file:
        json.dump(data, json_file, indent=1)
    image_recognition_file_content.close()
    patience_dialog.close()

    # finalise things if all images are verified
    if n_verified_files == total_n_files:
        if mb.askyesno(title=["Are you done?", "¿Ya terminaste?", "Avez-vous terminé?"][lang_idx],
                       message=["All images are verified and the 'image_recognition_file.json' is up-to-date.\n\nDo you want to close this "
                                "verification session and proceed to the final step?", "Todas las imágenes están verificadas y "
                                "'image_recognition_file.json' está actualizado.\n\n¿Quieres cerrar esta sesión de verificación"
                                " y continuar con el paso final?", "Toutes les images ont été vérifiées et le fichier 'image_recognition_file.json' "
                                "est à jour.\n\nVoulez-vous quitter cette session de vérification et procéder à l'étape finale?"][lang_idx]):
            # close window
            hitl_progress_window.destroy()
            bind_scroll_to_deploy_canvas()

            # get plot from xml files
            fig = produce_graph(file_list_txt = file_list_txt)

            # init window
            hitl_final_window = customtkinter.CTkToplevel(root)
            hitl_final_window.title("Overview")
            hitl_final_window.geometry("+10+10")

            # add plot
            chart_type = FigureCanvasTkAgg(fig, hitl_final_window)
            chart_type.get_tk_widget().grid(row = 0, column = 0)

            # button frame
            hitl_final_actions_frame = LabelFrame(hitl_final_window, text=[" Do you want to export these verified images as training data? ",
                                                                           " ¿Quieres exportar estas imágenes verificadas como datos de entrenamiento? ",
                                                                           " Voulez-vous exporter ces images vérifiées à titre de données d'entraînement?"][lang_idx],
                                                                           pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg=green_primary, labelanchor = 'n')
            hitl_final_actions_frame.configure(font=(text_font, 15, "bold"))
            hitl_final_actions_frame.grid(column=0, row=3, columnspan=2, sticky='ew')
            hitl_final_actions_frame.columnconfigure(0, weight=1, minsize=115)
            hitl_final_actions_frame.columnconfigure(1, weight=1, minsize=115)

            # buttons
            btn_hitl_final_export_y = Button(master=hitl_final_actions_frame, text=["Yes - choose folder and create training data",
                                                                                    "Sí - elija la carpeta y crear datos de entrenamiento",
                                                                                    "Oui - choisir un dossier et créer données d'entraînement"][lang_idx], 
                                    width=1, command = lambda: [uniquify_and_move_img_and_xml_from_filelist(file_list_txt = file_list_txt, recognition_file = recognition_file, hitl_final_window = hitl_final_window),
                                                                update_frame_states()])
            btn_hitl_final_export_y.grid(row=0, column=0, rowspan=1, sticky='nesw', padx=5)

            btn_hitl_final_export_n = Button(master=hitl_final_actions_frame, text=["No - go back to the main AddaxAI window",
                                                                                    "No - regrese a la ventana principal de AddaxAI",
                                                                                    "Non - retourner à la fenêtre principale AddaxAI"][lang_idx], 
                                    width=1, command = lambda: [delete_temp_folder(file_list_txt),
                                                                hitl_final_window.destroy(),
                                                                change_hitl_var_in_json(recognition_file, "done"),
                                                                update_frame_states()])
            btn_hitl_final_export_n.grid(row=0, column=1, rowspan=1, sticky='nesw', padx=5)

# os dependent python executables
def get_python_interprator(env_name):
    if platform.system() == 'Windows':
        return os.path.join(AddaxAI_files, "envs", f"env-{env_name}", "python.exe")
    else:
        return os.path.join(AddaxAI_files, "envs", f"env-{env_name}", "bin", "python")

# get the images and xmls from annotation session and store them with unique filename
def uniquify_and_move_img_and_xml_from_filelist(file_list_txt, recognition_file, hitl_final_window):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # choose destination
    dst_dir = filedialog.askdirectory()

    # ask to move or copy
    window = TextButtonWindow(["Method of file placement", "Método de colocación de archivos", "Méthode de classement de fichiers"][lang_idx],
                              [f"Do you want to copy or move the images to\n'{dst_dir}'?",
                              f"¿Quieres copiar o mover las imágenes a\n'{dst_dir}'?",
                              f"Voulez-vous COPIER ou DÉPLACER les images vers\n'{dst_dir}'"][lang_idx],
                              [["Move", "Mover", "Déplacer"][lang_idx], ["Copy", "Copiar", "Copier"][lang_idx], ["Cancel", "Cancelar", "Annuler"][lang_idx]])
    user_input = window.run()
    if user_input == "Cancel" or user_input == "Cancelar" or user_input == "Annuler":
        return
    else:
        if user_input == "Move" or user_input == "Mover" or user_input == "Déplacer":
            copy_or_move = "Move"
        if user_input == "Copy" or user_input == "Copiar" or user_input == "Copier":
            copy_or_move = "Copy"

    # init vars
    src_dir = os.path.normpath(var_choose_folder.get())
    
    # loop through the images
    with open(file_list_txt) as f:

        # count total number of images without loading to memory
        n_imgs = 0
        for i in f:
            n_imgs += 1
        
        # reset file index
        f.seek(0)

        # open patience window
        patience_dialog = PatienceDialog(total = n_imgs, text = ["Writing files...", "Escribir archivos...", "Écriture des fichiers..."][lang_idx])
        patience_dialog.open()
        current = 1

        # loop
        for img in f:

            # get relative path
            img_rel_path = os.path.relpath(img.rstrip(), src_dir)

            # uniquify image
            src_img = os.path.join(src_dir, img_rel_path)
            dst_img = os.path.join(dst_dir, img_rel_path)
            Path(os.path.dirname(dst_img)).mkdir(parents=True, exist_ok=True)
            if copy_or_move == "Move":
                shutil.move(src_img, dst_img)
            elif copy_or_move == "Copy":
                shutil.copy2(src_img, dst_img)

            # uniquify annotation
            ann_rel_path = os.path.splitext(img_rel_path)[0] + ".xml"
            src_ann = return_xml_path(os.path.join(src_dir, img_rel_path))
            dst_ann = os.path.join(dst_dir, ann_rel_path)
            Path(os.path.dirname(dst_ann)).mkdir(parents=True, exist_ok=True)
            shutil.move(src_ann, dst_ann)

            # update dialog
            patience_dialog.update_progress(current)
            current += 1
        f.close()
    
    # finalize
    patience_dialog.close()
    delete_temp_folder(file_list_txt)
    hitl_final_window.destroy()
    change_hitl_var_in_json(recognition_file, "done")

# check if input can be converted to float
def is_valid_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# get size of file in appropriate unit
def get_size(path):
    size = os.path.getsize(path)
    if size < 1024:
        return f"{size} bytes"
    elif size < pow(1024,2):
        return f"{round(size/1024, 2)} KB"
    elif size < pow(1024,3):
        return f"{round(size/(pow(1024,2)), 2)} MB"
    elif size < pow(1024,4):
        return f"{round(size/(pow(1024,3)), 2)} GB"

# check if the user is already in progress of verifying, otherwise start new session
def start_or_continue_hitl():

    # early exit if only video json
    selected_dir = var_choose_folder.get()
    path_to_image_json = os.path.join(selected_dir, "image_recognition_file.json")

    # warn user if the json file is very large
    json_size = os.path.getsize(path_to_image_json)
    if json_size > 500000:
        mb.showwarning(warning_txt[lang_idx], [f"The JSON file is very large ({get_size(path_to_image_json)}). This can cause the verification"
                                            " step to perform very slow. It will work, but you'll have to be patient. ", "El archivo "
                                            f"JSON es muy grande ({get_size(path_to_image_json)}). Esto puede hacer que el paso de verificación"
                                            " funcione muy lentamente. Funcionará, pero tendrás que tener paciencia. ",
                                            f"Le fichier JSON est très volumineux ({get_size(path_to_image_json)}). Ceci peut causer un ralentissement"
                                            " de l'étape de vérification. Cela devrait toutefois fonctionner. SVP veuillez patienter. "][lang_idx])

    # check requirements
    check_json_presence_and_warn_user(["verify", "verificar", "vérifier"][lang_idx],
                                      ["verifying", "verificando", "en vérification"][lang_idx],
                                      ["verification", "verificación", "vérification"][lang_idx])
    if not os.path.isfile(path_to_image_json):
        return

    # check hitl status
    status = get_hitl_var_in_json(path_to_image_json)

    # start first session
    if status == "never-started":
        # open window to select criteria
        open_hitl_settings_window()
    
    # continue previous session
    elif status == "in-progress":

        # read selection criteria from last time
        annotation_arguments_pkl = os.path.join(selected_dir, 'temp-folder', 'annotation_information.pkl')
        with open(annotation_arguments_pkl, 'rb') as fp:
            annotation_arguments = pickle.load(fp)
        
        # update class_txt_file from json in case user added classes last time
        class_list_txt = annotation_arguments['class_list_txt']
        label_map = fetch_label_map_from_json(os.path.join(var_choose_folder.get(), 'image_recognition_file.json'))
        if os.path.isfile(class_list_txt):
            os.remove(class_list_txt)
        with open(class_list_txt, 'a') as f:
            for k, v in label_map.items():
                f.write(f"{v}\n")
            f.close()

        # ask user 
        if not mb.askyesno(["Verification session in progress", "Sesión de verificación en curso", "Vérification en cours"][lang_idx],
                            ["Do you want to continue with the previous verification session? If you press 'No', you will start a new session.", 
                            "¿Quieres continuar con la sesión de verificación anterior? Si presiona 'No', iniciará una nueva sesión.",
                            "Voulez-vous reprendre la dernière session de vérification? Si vous choisissez 'Non', une nouvelle session démarrera."][lang_idx]):
            delete_temp_folder(annotation_arguments['file_list_txt'])
            change_hitl_var_in_json(path_to_image_json, "never-started") # if user closes window, it can start fresh next time
            open_hitl_settings_window()

        # start human in the loop process and skip selection window
        else:
            try:
                open_annotation_windows(recognition_file = annotation_arguments['recognition_file'],
                                        class_list_txt = annotation_arguments['class_list_txt'],
                                        file_list_txt = annotation_arguments['file_list_txt'],
                                        label_map = annotation_arguments['label_map'])
            except Exception as error:
                # log error
                print("ERROR:\n" + str(error) + "\n\nDETAILS:\n" + str(traceback.format_exc()) + "\n\n")
                
                # show error
                mb.showerror(title=error_txt[lang_idx],
                            message=["An error has occurred", "Ha ocurrido un error", "Une erreur s'est produite"][lang_idx] + " (AddaxAI v" + current_AA_version + "): '" + str(error) + "'.",
                            detail=traceback.format_exc())
    
    # start new session
    elif status == "done":
        if mb.askyesno(["Previous session is done", "Sesión anterior terminada.", "Dernière session complétée."][lang_idx], ["It seems like you have completed the previous manual "
                        "verification session. Do you want to start a new session?", "Parece que has completado la sesión de verificación manual "
                        "anterior. ¿Quieres iniciar una nueva sesión?",
                        "Il semble que vous ayez déjà complété la dernière session de vérification. Souhaitez-vous démarrer un nouvelle session?"][lang_idx]):
            open_hitl_settings_window()

# open xml and check if the data is already in the json
def check_if_img_needs_converting(img_file): 
    # open xml
    root = ET.parse(return_xml_path(img_file)).getroot()

    # read verification status
    try:
        verification_status = True if root.attrib['verified'] == 'yes' else False
    except:
        verification_status = False
    
    # read json update status
    try:
        json_update_status = True if root.attrib['json_updated'] == 'yes' else False
    except:
        json_update_status = False

    # return whether or not it needs converting to json
    if verification_status == True and json_update_status == False: 
        return True
    else:
        return False

# converts individual xml to coco
def convert_xml_to_coco(xml_path, inverted_label_map):
    # open
    tree = ET.parse(xml_path)
    root = tree.getroot()
    try:
        verification_status = True if root.attrib['verified'] == 'yes' else False
    except:
        verification_status = False
    path = root.findtext('path')
    size = root.find('size')
    im_width = int(size.findtext('width'))
    im_height = int(size.findtext('height'))

    # fetch objects
    verified_detections = []
    new_class = False
    for obj in root.findall('object'):
        name = obj.findtext('name')

        # check if new class
        if name not in inverted_label_map:
            new_class = True
            highest_index = 0
            for key, value in inverted_label_map.items():
                value = int(value)
                if value > highest_index:
                    highest_index = value
            inverted_label_map[name] = str(highest_index + 1)
        category = inverted_label_map[name]

        # read 
        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.findtext('xmin')))
        ymin = int(float(bndbox.findtext('ymin')))
        xmax = int(float(bndbox.findtext('xmax')))
        ymax = int(float(bndbox.findtext('ymax')))

        # convert
        w_box = round(abs(xmax - xmin) / im_width, 5)
        h_box = round(abs(ymax - ymin) / im_height, 5)
        xo = round(xmin / im_width, 5)
        yo = round(ymin / im_height, 5)
        bbox = [xo, yo, w_box, h_box]

        # compile
        verified_detection = {'category' : category,
                              'conf' : 1.0,
                              'bbox' : bbox}
        verified_detections.append(verified_detection)

    verified_image = {'file' : path,
                      'detections' : verified_detections}
    
    # return
    return [verified_image, verification_status, new_class, inverted_label_map]

# update json from list with verified images
def update_json_from_img_list(verified_images, inverted_label_map, recognition_file, patience_dialog, current): 

        # check if the json has relative paths
        if check_json_paths(recognition_file) == "relative":
            json_paths_are_relative = True
        else:
            json_paths_are_relative = False

        # open
        with open(recognition_file, "r") as image_recognition_file_content:
            data = json.load(image_recognition_file_content)

        # adjust
        for image in data['images']:
            image_path = image['file']
            if json_paths_are_relative:
                image_path = os.path.normpath(os.path.join(os.path.dirname(recognition_file), image_path))
            if image_path in verified_images:

                # update progress
                patience_dialog.update_progress(current = current, percentage = True)
                current += 1

                # read
                xml = return_xml_path(image_path)
                coco, verification_status, new_class, inverted_label_map = convert_xml_to_coco(xml, inverted_label_map)
                image['manually_checked'] = verification_status
                if new_class:
                    data['detection_categories'] = {v: k for k, v in inverted_label_map.items()}
                if verification_status:
                    image['detections'] = coco['detections']

                    # adjust xml file
                    tree = ET.parse(xml)
                    root = tree.getroot()
                    root.set('json_updated', 'yes')
                    indent(root)
                    tree.write(xml)
        image_recognition_file_content.close()

        # write
        print(recognition_file)
        with open(recognition_file, "w") as json_file:
            json.dump(data, json_file, indent=1)
        image_recognition_file_content.close()

# write model specific variables to file 
def write_model_vars(model_type="cls", new_values = None):
        
    # exit if no cls is selected
    if var_cls_model.get() in none_txt:
        return

    # adjust
    variables = load_model_vars(model_type)
    if new_values is not None:
        for key, value in new_values.items():
            if key in variables:
                variables[key] = value
            else:
                print(f"Warning: Variable {key} not found in the loaded model variables.")

    # write
    model_dir = var_cls_model.get() if model_type == "cls" else var_det_model.get()
    var_file = os.path.join(AddaxAI_files, "models", model_type, model_dir, "variables.json")
    with open(var_file, 'w') as file:
        json.dump(variables, file, indent=4)
        
# check if there is a taxonomic csv file
def taxon_mapping_csv_present():   
    return os.path.isfile(os.path.join(AddaxAI_files, "models", "cls", var_cls_model.get(), "taxon-mapping.csv"))

# return the dataframe if there is a taxonomic csv file
def fetch_taxon_mapping_df():
    taxon_mapping_csv = os.path.join(AddaxAI_files, "models", "cls", var_cls_model.get(), "taxon-mapping.csv")
    if os.path.isfile(taxon_mapping_csv):
        return pd.read_csv(taxon_mapping_csv)

# take MD json and classify detections
def classify_detections(json_fpath, data_type, simple_mode = False):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # show user it's loading
    progress_window.update_values(process = f"{data_type}_cls", status = "load")
    root.update()
        
    # load model specific variables
    model_vars = load_model_vars() 
    cls_model_fname = model_vars["model_fname"]
    cls_model_type = model_vars["type"]
    cls_model_fpath = os.path.join(AddaxAI_files, "models", "cls", var_cls_model.get(), cls_model_fname)

    # check if taxonomic fallback should be the default
    taxon_mapping_csv_is_present = taxon_mapping_csv_present()
    taxon_mapping_is_default = model_vars.get("var_tax_fallback_default", False)

    # if present take os-specific env else take general env
    if os.name == 'nt': # windows
        cls_model_env = model_vars.get("env-windows", model_vars["env"])
    elif platform.system() == 'Darwin': # macos
        cls_model_env = model_vars.get("env-macos", model_vars["env"])
    else: # linux
        cls_model_env = model_vars.get("env-linux", model_vars["env"])

    # get param values
    cls_tax_fallback = False
    cls_tax_levels_idx = 0
    if simple_mode:
        cls_disable_GPU = False
        cls_detec_thresh = model_vars["var_cls_detec_thresh_default"]
        cls_class_thresh = model_vars["var_cls_class_thresh_default"]
        cls_animal_smooth = False
        if taxon_mapping_csv_is_present:
            if taxon_mapping_is_default:
                cls_tax_fallback =  True
                # leave cls_tax_levels_idx at default 0 to let the model decide
    else:
        cls_disable_GPU = var_disable_GPU.get()
        cls_detec_thresh = var_cls_detec_thresh.get() 
        cls_class_thresh = var_cls_class_thresh.get()
        cls_animal_smooth = var_smooth_cls_animal.get()
        if taxon_mapping_csv_is_present:
            cls_tax_fallback = var_tax_fallback.get() # the users choice
            cls_tax_levels_idx = model_vars["var_tax_levels_idx"] # take idx from model vars
        
    # init paths
    python_executable = get_python_interprator(cls_model_env)
    inference_script = os.path.join(AddaxAI_files, "AddaxAI", "classification_utils", "model_types", cls_model_type, "classify_detections.py")

    # create command
    command_args = []
    command_args.append(python_executable)
    command_args.append(inference_script)
    command_args.append(AddaxAI_files)
    command_args.append(cls_model_fpath)
    command_args.append(str(cls_detec_thresh))
    command_args.append(str(cls_class_thresh))
    command_args.append(str(cls_animal_smooth))
    command_args.append(json_fpath)
    try:
        command_args.append(temp_frame_folder)
    except NameError:
        command_args.append("None")
        pass
    command_args.append(str(cls_tax_fallback))
    command_args.append(str(cls_tax_levels_idx))
    
    # adjust command for unix OS
    if os.name != 'nt':
        command_args = "'" + "' '".join(command_args) + "'"

    # prepend with os-specific commands
    if os.name == 'nt': # windows
        if cls_disable_GPU:
            command_args = ['set CUDA_VISIBLE_DEVICES="" &'] + command_args
    elif platform.system() == 'Darwin': # macos
        command_args = "export PYTORCH_ENABLE_MPS_FALLBACK=1 && " + command_args
    else: # linux
        if cls_disable_GPU:
            command_args =  "CUDA_VISIBLE_DEVICES='' " + command_args
        else:
            command_args = "export PYTORCH_ENABLE_MPS_FALLBACK=1 && " + command_args

    # log command
    print(command_args)

    # prepare process and cancel method per OS
    if os.name == 'nt':
        # run windows command
        p = Popen(command_args,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT,
                  bufsize=1,
                  shell=True,
                  universal_newlines=True)

    else:
        # run unix command
        p = Popen(command_args,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.STDOUT,
                  bufsize=1,
                  shell=True,
                  universal_newlines=True,
                  preexec_fn=os.setsid)

    # set global vars
    global subprocess_output
    subprocess_output = ""

    # calculate metrics while running
    status_setting = 'running'
    for line in p.stdout:

        # save output if something goes wrong
        subprocess_output = subprocess_output + line
        subprocess_output = subprocess_output[-1000:]

        # log
        print(line, end='')

        # catch early exit if there are no detections that meet the requirmentents to classify
        if line.startswith("n_crops_to_classify is zero. Nothing to classify."):
            mb.showinfo(information_txt[lang_idx], ["There are no animal detections that meet the criteria. You either "
                                                "have selected images without any animals present, or you have set "
                                                "your detection confidence threshold to high.", "No hay detecciones"
                                                " de animales que cumplan los criterios. O bien ha seleccionado "
                                                "imágenes sin presencia de animales, o bien ha establecido el umbral"
                                                " de confianza de detección en alto.",
                                                "Aucune détection d'animal ne rencontre les critères. Vous avez soit sélectionner "
                                                "des images sans animaux présents, ou vous avez régler le seuil de confiance "
                                                "de détection trop haut."][lang_idx])
            elapsed_time = "00:00",
            time_left = "00:00",
            current_im = "0",
            total_im = "0",
            processing_speed = "0it/s",
            percentage = "100",
            GPU_param = "Unknown",
            data_type = data_type,
            break

        # catch smoothening info lines
        if "<EA>" in line:
            smooth_output_line = re.search('<EA>(.+)<EA>', line).group().replace('<EA>', '')
            smooth_output_file = os.path.join(os.path.dirname(json_fpath), "smooth-output.txt")
            with open(smooth_output_file, 'a+') as f:
                f.write(f"{smooth_output_line}\n")
            f.close()

        # if smoothing, the pbar should change description
        if "<EA-status-change>" in line:
            status_setting = re.search('<EA-status-change>(.+)<EA-status-change>', line).group().replace('<EA-status-change>', '')

        # get process stats and send them to tkinter
        if line.startswith("GPU available: False"):
            GPU_param = "CPU"
        elif line.startswith("GPU available: True"):
            GPU_param = "GPU"
        elif '%' in line[0:4]:
            
            # read stats
            times = re.search("(\[.*?\])", line)[1]
            progress_bar = re.search("^[^\/]*[^[^ ]*", line.replace(times, ""))[0]
            percentage = re.search("\d*%", progress_bar)[0][:-1]
            current_im = re.search("\d*\/", progress_bar)[0][:-1]
            total_im = re.search("\/\d*", progress_bar)[0][1:]
            elapsed_time = re.search("(?<=\[)(.*)(?=<)", times)[1]
            time_left = re.search("(?<=<)(.*)(?=,)", times)[1]
            processing_speed = re.search("(?<=,)(.*)(?=])", times)[1].strip()

            # print stats
            progress_window.update_values(process = f"{data_type}_cls",
                                            status = status_setting,
                                            cur_it = int(current_im),
                                            tot_it = int(total_im),
                                            time_ela = elapsed_time,
                                            time_rem = time_left,
                                            speed = processing_speed,
                                            hware = GPU_param,
                                            cancel_func = lambda: cancel_subprocess(p))
        root.update()

    # process is done
    progress_window.update_values(process = f"{data_type}_cls",
                                       status = "done",
                                       time_ela = elapsed_time,
                                       speed = processing_speed)

    root.update()

# quit popen process
def cancel_subprocess(process):
    global cancel_deploy_model_pressed
    global btn_start_deploy
    global sim_run_btn
    if os.name == 'nt':
        Popen(f"TASKKILL /F /PID {process.pid} /T")
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    btn_start_deploy.configure(state=NORMAL)
    sim_run_btn.configure(state=NORMAL)
    cancel_deploy_model_pressed = True
    progress_window.close()

# deploy model and create json output files 
warn_smooth_vid = True
def deploy_model(path_to_image_folder, selected_options, data_type, simple_mode = False):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
    
    # note if user is video analysing without smoothing
    global warn_smooth_vid
    if (var_cls_model.get() not in none_txt) and \
        (var_smooth_cls_animal.get() == False) and \
            data_type == 'vid' and \
                simple_mode == False and \
                    warn_smooth_vid == True:
                        warn_smooth_vid = False
                        if not mb.askyesno(information_txt[lang_idx], ["You are about to analyze videos without smoothing the confidence scores. "
                            "Typically, a video may contain many frames of the same animal, increasing the likelihood that at least "
                            f"one of the labels could be a false prediction. With '{lbl_smooth_cls_animal_txt[lang_idx]}' enabled, all"
                            " predictions from a single video will be averaged, resulting in only one label per video. Do you wish to"
                            " continue without smoothing?\n\nPress 'No' to go back.", "Estás a punto de analizar videos sin suavizado "
                            "habilitado. Normalmente, un video puede contener muchos cuadros del mismo animal, lo que aumenta la "
                            "probabilidad de que al menos una de las etiquetas pueda ser una predicción falsa. Con "
                            f"'{lbl_smooth_cls_animal_txt[lang_idx]}' habilitado, todas las predicciones de un solo video se promediarán,"
                            " lo que resultará en una sola etiqueta por video. ¿Deseas continuar sin suavizado habilitado?\n\nPresiona "
                            "'No' para regresar.",
                            "Vous êtes sur le point d'analyser des vidéos sans lisser les scores de confiance. "
                            "Typiquement, un vidéo peut contenir plusieurs images d'un même animal, ce qui augmente les chances qu'au moins un "
                            f"des labels puisse être une fausse prédiction. Avec '{lbl_smooth_cls_animal_txt[lang_idx]}' activé, toute"
                            " les prédictions d'un seul vidéo seront moyennées, résultant en un seul label par vidéo. Souhaitez-vous"
                            " continuer sans lissage?\n\nAppuyer sur 'Non' pour revenir en arrière."][lang_idx]):
                            return
    
    # display loading window
    # try to update progress window, if AttributeError, it means it tries to update the img_det and we're working with a full image classifier
    try:
        progress_window.update_values(process = f"{data_type}_det", status = "load")
    except AttributeError:
        pass 

    # prepare variables
    chosen_folder = str(Path(path_to_image_folder))
    run_detector_batch_py = os.path.join(AddaxAI_files, "cameratraps", "megadetector", "detection", "run_detector_batch.py")
    image_recognition_file = os.path.join(chosen_folder, "image_recognition_file.json")
    process_video_py = os.path.join(AddaxAI_files, "cameratraps", "megadetector", "detection", "process_video.py")
    video_recognition_file = "--output_json_file=" + os.path.join(chosen_folder, "video_recognition_file.json")
    GPU_param = "Unknown"
    python_executable = get_python_interprator("base")

    # select model based on user input via dropdown menu, or take MDv5a for simple mode 
    custom_model_bool = False
    if simple_mode:
        det_model_fpath = os.path.join(DET_DIR, "MegaDetector 5a", "md_v5a.0.0.pt")
        switch_yolov5_version("old models")
    elif var_det_model.get() != dpd_options_model[lang_idx][-1]: # if not chosen the last option, which is "custom model"
        det_model_fname = load_model_vars("det")["model_fname"]
        det_model_fpath = os.path.join(DET_DIR, var_det_model.get(), det_model_fname)
        switch_yolov5_version("old models")
    else:
        # set model file
        det_model_fpath = var_det_model_path.get()
        custom_model_bool = True

        # set yolov5 git to accommodate new models (checkout depending on how you retrain MD)
        switch_yolov5_version("new models") 
        
        # extract classes
        label_map = extract_label_map_from_model(det_model_fpath)

        # write labelmap to separate json
        json_object = json.dumps(label_map, indent=1)
        native_model_classes_json_file = os.path.join(chosen_folder, "native_model_classes.json")
        with open(native_model_classes_json_file, "w") as outfile:
            outfile.write(json_object)
        
        # add argument to command call
        selected_options.append("--class_mapping_filename=" + native_model_classes_json_file)

    # set global cancel bool
    global cancel_deploy_model_pressed
    cancel_deploy_model_pressed = False

    # if a full image classifier is selected, imitate object detection to get full bboxes
    full_image_cls = load_model_vars("cls").get("full_image_cls", False)
    if full_image_cls:
        imitate_object_detection_for_full_image_classifier(chosen_folder)
    
    # for crop classifiers we need to run the detection first
    else:

        # create commands for Windows
        if os.name == 'nt':
            if selected_options == []:
                img_command = [python_executable, run_detector_batch_py, det_model_fpath, '--threshold=0.01', chosen_folder, image_recognition_file]
                vid_command = [python_executable, process_video_py, '--max_width=1280', '--verbose', '--quality=85', '--allow_empty_video', video_recognition_file, det_model_fpath, chosen_folder]
            else:
                img_command = [python_executable, run_detector_batch_py, det_model_fpath, *selected_options, '--threshold=0.01', chosen_folder, image_recognition_file]
                vid_command = [python_executable, process_video_py, *selected_options, '--max_width=1280', '--verbose', '--quality=85', '--allow_empty_video', video_recognition_file, det_model_fpath, chosen_folder]

        # create command for MacOS and Linux
        else:
            if selected_options == []:
                img_command = [f"'{python_executable}' '{run_detector_batch_py}' '{det_model_fpath}' '--threshold=0.01' '{chosen_folder}' '{image_recognition_file}'"]
                vid_command = [f"'{python_executable}' '{process_video_py}' '--max_width=1280' '--verbose' '--quality=85' '--allow_empty_video' '{video_recognition_file}' '{det_model_fpath}' '{chosen_folder}'"]
            else:
                selected_options = "' '".join(selected_options)
                img_command = [f"'{python_executable}' '{run_detector_batch_py}' '{det_model_fpath}' '{selected_options}' '--threshold=0.01' '{chosen_folder}' '{image_recognition_file}'"]
                vid_command = [f"'{python_executable}' '{process_video_py}' '{selected_options}' '--max_width=1280' '--verbose' '--quality=85' '--allow_empty_video' '{video_recognition_file}' '{det_model_fpath}' '{chosen_folder}'"]

        # pick one command
        if data_type == "img":
            command = img_command
        else:
            command = vid_command

        # if user specified to disable GPU, prepend and set system variable
        if var_disable_GPU.get() and not simple_mode:
            if os.name == 'nt': # windows
                command[:0] = ['set', 'CUDA_VISIBLE_DEVICES=""', '&']
            elif platform.system() == 'Darwin': # macos
                mb.showwarning(warning_txt[lang_idx],
                            ["Disabling GPU processing is currently only supported for CUDA devices on Linux and Windows "
                                "machines, not on macOS. Proceeding without GPU disabled.", "Deshabilitar el procesamiento de "
                                "la GPU actualmente sólo es compatible con dispositivos CUDA en máquinas Linux y Windows, no en"
                                " macOS. Proceder sin GPU desactivada.",
                                "La désactivation du traitement par GPU est uniquement supportée sur les dispositifs CUDA sous "
                                "Linux et Windows, pas sous MacOS. Poursuite du traitement sans désactiver le GPU."
                                ""][lang_idx])
                var_disable_GPU.set(False)
            else: # linux
                command = "CUDA_VISIBLE_DEVICES='' " + command

        # log
        print(f"command:\n\n{command}\n\n")
            
        # prepare process and cancel method per OS
        if os.name == 'nt':
            # run windows command
            p = Popen(command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    shell=True,
                    universal_newlines=True)

        else:
            # run unix command
            p = Popen(command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    shell=True,
                    universal_newlines=True,
                    preexec_fn=os.setsid)
        
        # set global vars
        global subprocess_output
        subprocess_output = ""
        previous_processed_img = ["There is no previously processed image. The problematic character is in the first image to analyse.",
                                "No hay ninguna imagen previamente procesada. El personaje problemático está en la primera imagen a analizar.",
                                "Il n'y a aucune image traitée précédemment. Le caractère problématique est dans la première image à analyser."][lang_idx]
        extracting_frames_mode = False
        
        # check if the unit shown should be frame or video
        if data_type == "vid" and var_cls_model.get() in none_txt:
            frame_video_choice = "video"
        elif data_type == "vid" and var_cls_model.get() not in none_txt:
            frame_video_choice = "frame"
        else:
            frame_video_choice = None
        
        # read output
        for line in p.stdout:
            
            # save output if something goes wrong
            subprocess_output = subprocess_output + line
            subprocess_output = subprocess_output[-1000:]

            # log
            print(line, end='')
            
            # catch model errors
            if line.startswith("No image files found"):
                mb.showerror(["No images found", "No se han encontrado imágenes", "Aucune image trouvée"][lang_idx],
                            [f"There are no images found in '{chosen_folder}'. \n\nAre you sure you specified the correct folder?"
                            f" If the files are in subdirectories, make sure you don't tick '{lbl_exclude_subs_txt[lang_idx]}'.",
                            f"No se han encontrado imágenes en '{chosen_folder}'. \n\n¿Está seguro de haber especificado la carpeta correcta?"
                            f" Si los archivos están en subdirectorios, asegúrese de no marcar la casilla '{lbl_exclude_subs_txt[lang_idx]}'.",
                            f"Aucune image trouvée dans '{chosen_folder}'. \n\nAvez-vous spécifié le bon dossier?"
                            f" Si les fichiers sont dans des sous-dossiers, assurez-vous ne pas avoir coché '{lbl_exclude_subs_txt[lang_idx]}'."][lang_idx])
                return
            if line.startswith("No videos found"):
                mb.showerror(["No videos found", "No se han encontrado vídeos", "Aucun vidéo trouvé"][lang_idx],
                            line + [f"\n\nAre you sure you specified the correct folder? If the files are in subdirectories, make sure you don't tick '{lbl_exclude_subs_txt[lang_idx]}'.",
                                    f"\n\n¿Está seguro de haber especificado la carpeta correcta? Si los archivos están en subdirectorios, asegúrese de no marcar la casilla '{lbl_exclude_subs_txt[lang_idx]}'.",
                                    f"\n\nAvez-vous spécifié le bon dossier? Si les fichiers sont dans des sous-dossiers, assurez-vous ne pas avoir coché '{lbl_exclude_subs_txt[lang_idx]}'."][lang_idx])
                return
            if line.startswith("No frames extracted"):
                mb.showerror(["Could not extract frames", "No se pueden extraer fotogramas", "Impossible d'extraire les images"][lang_idx],
                            line + ["\n\nConverting the videos to .mp4 might fix the issue.",
                                    "\n\nConvertir los vídeos a .mp4 podría solucionar el problema.",
                                    "\n\nConvertir les vidéos au format .mp4 pourrait régler le problème."][lang_idx])
                return
            if line.startswith("UnicodeEncodeError:"):
                mb.showerror("Unparsable special character",
                            [f"{line}\n\nThere seems to be a special character in a filename that cannot be parsed. Unfortunately, it's not"
                            " possible to point you to the problematic file directly, but I can tell you that the last successfully analysed"
                            f" image was\n\n{previous_processed_img}\n\nThe problematic character should be in the file or folder name of "
                            "the next image, alphabetically. Please remove any special characters from the path and try again.", 
                            f"{line}\n\nParece que hay un carácter especial en un nombre de archivo que no se puede analizar. Lamentablemente,"
                            " no es posible indicarle directamente el archivo problemático, pero puedo decirle que la última imagen analizada "
                            f"con éxito fue\n\n{previous_processed_img}\n\nEl carácter problemático debe estar en el nombre del archivo o "
                            "carpeta de la siguiente imagen, alfabéticamente. Elimine los caracteres especiales de la ruta e inténtelo de "
                            "nuevo.",
                            f"{line}\n\nIl semble y avoir un caractère spécial non-reconnu dans le nom d'un fichier. Malheureusement, il est"
                            " impossible d'identifier le fichier directement, cependant la dernière images correctement analysée était "
                            f" \n\n{previous_processed_img}\n\nLe caractère problématique devrait être dans le nom de fichier ou de dossier de "
                            "la prochaine image, alphabetiquement. SVP remplacer tout caractère spécial du chemin et du nom de fichier et réessayer."][lang_idx])
                return
            if line.startswith("Processing image "):
                previous_processed_img = line.replace("Processing image ", "")

            # write errors to log file
            if "Exception:" in line:
                with open(model_error_log, 'a+') as f:
                    f.write(f"{line}\n")
                f.close()

            # write warnings to log file
            if "Warning:" in line:
                if not "could not determine MegaDetector version" in line \
                    and not "no metadata for unknown detector version" in line \
                    and not "using user-supplied image size" in line \
                    and not "already exists and will be overwritten" in line:
                    with open(model_warning_log, 'a+') as f:
                        f.write(f"{line}\n")
                    f.close()
                    
            # print frame extraction progress and dont continue until done
            if "Extracting frames for folder " in line and \
                data_type == "vid":
                progress_window.update_values(process = f"{data_type}_det",
                                            status = "extracting frames")
                extracting_frames_mode = True
            if extracting_frames_mode:
                if '%' in line[0:4]:
                    progress_window.update_values(process = f"{data_type}_det",
                                                status = "extracting frames",
                                                extracting_frames_txt = [f"Extracting frames... {line[:3]}%",
                                                                        f"Extrayendo fotogramas... {line[:3]}%"])
            if "Extracted frames for" in line and \
                data_type == "vid":
                    extracting_frames_mode = False
            if extracting_frames_mode:
                continue
            
            # get process stats and send them to tkinter
            if line.startswith("GPU available: False"):
                GPU_param = "CPU"
            elif line.startswith("GPU available: True"):
                GPU_param = "GPU"
            elif '%' in line[0:4]:
                
                # read stats
                times = re.search("(\[.*?\])", line)[1]
                progress_bar = re.search("^[^\/]*[^[^ ]*", line.replace(times, ""))[0]
                percentage = re.search("\d*%", progress_bar)[0][:-1]
                current_im = re.search("\d*\/", progress_bar)[0][:-1]
                total_im = re.search("\/\d*", progress_bar)[0][1:]
                elapsed_time = re.search("(?<=\[)(.*)(?=<)", times)[1]
                time_left = re.search("(?<=<)(.*)(?=,)", times)[1]
                processing_speed = re.search("(?<=,)(.*)(?=])", times)[1].strip()

                # show progress
                progress_window.update_values(process = f"{data_type}_det",
                                                status = "running",
                                                cur_it = int(current_im),
                                                tot_it = int(total_im),
                                                time_ela = elapsed_time,
                                                time_rem = time_left,
                                                speed = processing_speed,
                                                hware = GPU_param,
                                                cancel_func = lambda: cancel_subprocess(p),
                                                frame_video_choice = frame_video_choice)
            root.update()
        
        # process is done
        progress_window.update_values(process = f"{data_type}_det", status = "done")
        root.update()
    
    # create addaxai metadata
    addaxai_metadata = {"addaxai_metadata" : {"version" : current_AA_version,
                                                  "custom_model" : custom_model_bool,
                                                  "custom_model_info" : {}}}
    if custom_model_bool:
        addaxai_metadata["addaxai_metadata"]["custom_model_info"] = {"model_name" : os.path.basename(os.path.normpath(det_model_fpath)),
                                                                         "label_map" : label_map}
    
    # write metadata to json and make absolute if specified
    image_recognition_file = os.path.join(chosen_folder, "image_recognition_file.json")
    video_recognition_file = os.path.join(chosen_folder, "video_recognition_file.json")
    if data_type == "img" and os.path.isfile(image_recognition_file):
        append_to_json(image_recognition_file, addaxai_metadata)
        if var_abs_paths.get():
            make_json_absolute(image_recognition_file)
    if data_type == "vid" and os.path.isfile(video_recognition_file):
        append_to_json(video_recognition_file, addaxai_metadata)
        if var_abs_paths.get():
            make_json_absolute(video_recognition_file)
    
    # classify detections if specified by user
    if not cancel_deploy_model_pressed:
        if var_cls_model.get() not in none_txt:
            if data_type == "img":
                classify_detections(os.path.join(chosen_folder, "image_recognition_file.json"), data_type, simple_mode = simple_mode)
            else:
                classify_detections(os.path.join(chosen_folder, "video_recognition_file.json"), data_type, simple_mode = simple_mode)

# merge image and video jsons together
def merge_jsons(image_json, video_json, output_file_path):

    # Load the image recognition JSON file
    if image_json:
        with open(image_json, 'r') as image_file:
            image_data = json.load(image_file)
            
    # Load the video recognition JSON file
    if video_json:
        with open(video_json, 'r') as video_file:
            video_data = json.load(video_file)

    # Merge the "images" lists
    if image_json and video_json:
        merged_images = image_data['images'] + video_data['images']
        detection_categories = image_data['detection_categories']
        info = image_data['info']
        classification_categories = image_data['classification_categories'] if 'classification_categories' in image_data else {}
        forbidden_classes = image_data['forbidden_classes'] if 'forbidden_classes' in image_data else {}
    elif image_json:
        merged_images = image_data['images']
        detection_categories = image_data['detection_categories']
        info = image_data['info']
        classification_categories = image_data['classification_categories'] if 'classification_categories' in image_data else {}
        forbidden_classes = image_data['forbidden_classes'] if 'forbidden_classes' in image_data else {}
    elif video_json:
        merged_images = video_data['images']
        detection_categories = video_data['detection_categories']
        info = video_data['info']
        classification_categories = video_data['classification_categories'] if 'classification_categories' in video_data else {}
        forbidden_classes = video_data['forbidden_classes'] if 'forbidden_classes' in video_data else {}
        
    # Create the merged data
    merged_data = {
        "images": merged_images,
        "detection_categories": detection_categories,
        "info": info,
        "classification_categories": classification_categories,
        "forbidden_classes": forbidden_classes
    }

    # Save the merged data to a new JSON file
    with open(output_file_path, 'w') as output_file:
        json.dump(merged_data, output_file, indent=1)

    print(f'merged json file saved to {output_file_path}')


# pop up window showing the user that an AddaxAI update is required for a particular model
def show_update_info(model_vars, model_name):

    # create window
    su_root = customtkinter.CTkToplevel(root)
    su_root.title("Update required")
    su_root.geometry("+10+10")
    su_root.columnconfigure(0, weight=1, minsize=300)
    su_root.columnconfigure(1, weight=1, minsize=300)
    lbl1 = customtkinter.CTkLabel(su_root, text=[f"Update required for model {model_name}", f"Actualización requerida para el modelo {model_name}",
                                                 f"Mise-à-jour requise pour le modèle {model_name}"][lang_idx], font = main_label_font)
    lbl1.grid(row=0, column=0, padx=PADX, pady=(PADY, PADY/2), columnspan = 2, sticky="nsew")
    lbl2 = customtkinter.CTkLabel(su_root, text=[f"Minimum AddaxAI version required is v{model_vars['min_version']}, while your current version is v{current_AA_version}.",
                                                 f"La versión mínima de AddaxAI requerida es v{model_vars['min_version']}, mientras que su versión actual es v{current_AA_version}.",
                                                 f"La version minimale d'AddaxAI requise est v{model_vars['min_version']}, tandis que la version courante est v{current_AA_version}."][lang_idx])
    lbl2.grid(row=1, column=0, padx=PADX, pady=(0, PADY), columnspan = 2, sticky="nsew")

    # define functions
    def close():
        su_root.destroy()
    def read_more():
        webbrowser.open("https://addaxdatascience.com/addaxai/")
        su_root.destroy()

    # buttons frame
    btns_frm = customtkinter.CTkFrame(master=su_root)
    btns_frm.columnconfigure(0, weight=1, minsize=10)
    btns_frm.columnconfigure(1, weight=1, minsize=10)
    btns_frm.grid(row=5, column=0, padx=PADX, pady=(0, PADY), columnspan = 2,sticky="nswe")
    close_btn = customtkinter.CTkButton(btns_frm, text="Cancel", command=close)
    close_btn.grid(row=2, column=0, padx=PADX, pady=PADY, sticky="nswe")
    lmore_btn = customtkinter.CTkButton(btns_frm, text="Update", command=read_more)
    lmore_btn.grid(row=2, column=1, padx=(0, PADX), pady=PADY, sticky="nwse")

# check if a particular model needs downloading
def model_needs_downloading(model_vars, model_type):
    model_name = var_cls_model.get() if model_type == "cls" else var_det_model.get()
    if model_name not in none_txt:
        model_fpath = os.path.join(AddaxAI_files, "models", model_type, model_name, load_model_vars(model_type)["model_fname"])
        if os.path.isfile(model_fpath):
            # the model file is already present
            return [False, ""]
        else:
            # the model is not present yet
            min_version = model_vars["min_version"]

            # let's check if the model works with the current EA version
            if needs_EA_update(min_version):
                show_update_info(model_vars, model_name)
                return [None, ""]
            else:
                return [True, os.path.dirname(model_fpath)]
    else:
        # user selected none
        return [False, ""]

# check if a particular environment needs downloading
def environment_needs_downloading(model_vars):
       
    # find out which env is required
    # if present take os-specific env else take general env
    if os.name == 'nt': # windows
        env_name = model_vars.get("env-windows", model_vars.get("env", "base"))
    elif platform.system() == 'Darwin': # macos
        env_name = model_vars.get("env-macos", model_vars.get("env", "base"))
    else: # linux
        env_name = model_vars.get("env-linux", model_vars.get("env", "base"))
    
    # check if that env is already present
    if os.path.isdir(os.path.join(AddaxAI_files, "envs", f'env-{env_name}')):
        return [False, env_name]
    else:
        return [True, env_name]

# check if path contains special characters
def contains_special_characters(path):
    allowed_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-./ +\:'()")
    for char in path:
        if char not in allowed_characters:
            return [True, char]
    return [False, ""]

# we run this instead of a detection model for full image classification
def imitate_object_detection_for_full_image_classifier(chosen_folder):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
    
    # List all images in the chosen folder
    image_files = [f for f in os.listdir(chosen_folder) if f.lower().endswith(('jpg', 'jpeg', 'png'))]

    # Initialize the JSON structure
    result = {
        "images": [],
        "detection_categories": {
            "1": "animal",
            "2": "person",
            "3": "vehicle"
        },
        "info": {
            "detection_completion_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "format_version": "",
            "detector": None,
            "detector_metadata": {
            }
        }
    }
    
    # Loop through each image in the folder and add it to the result
    for image_file in image_files:
        # Prepare the image's detections with the full image bounding box
        image_data = {
            "file": image_file,
            "detections": [
                {
                    "category": "1",  # Assuming all detections are animals
                    "conf": 1.0,  # High confidence for full-image detection
                    "bbox": [0.0, 0.0, 1.0, 1.0]  # Full image bounding box
                }
            ]
        }
        
        # Add image data to the images list
        result["images"].append(image_data)
    
    # Save the result as a JSON file
    json_filename = os.path.join(chosen_folder, "image_recognition_file.json")
    with open(json_filename, "w") as json_file:
        json.dump(result, json_file, indent=4)
    
    print(f"JSON file created: {json_filename}")

# open progress window and initiate the model deployment
def start_deploy(simple_mode = False):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # check if there are any images or videos in the folder
    chosen_folder = var_choose_folder.get()
    if simple_mode:
        check_img_presence = True
        check_vid_presence = True
    else:
        check_img_presence = var_process_img.get()
        check_vid_presence = var_process_vid.get()
    img_present = False
    vid_present = False
    if var_exclude_subs.get():
        # non recursive
        for f in os.listdir(chosen_folder):
            if check_img_presence:
                if f.lower().endswith(IMG_EXTENSIONS):
                    img_present = True
            if check_vid_presence:
                if f.lower().endswith(VIDEO_EXTENSIONS):
                    vid_present = True
            if (img_present and vid_present) or \
                (img_present and not check_vid_presence) or \
                    (vid_present and not check_img_presence) or \
                        (not check_img_presence and not check_vid_presence):
                break
    else:
        # recursive
        for main_dir, _, files in os.walk(chosen_folder):
            for file in files:
                if check_img_presence and file.lower().endswith(IMG_EXTENSIONS):
                    img_present = True
                if check_vid_presence and file.lower().endswith(VIDEO_EXTENSIONS):
                    vid_present = True
            if (img_present and vid_present) or \
                (img_present and not check_vid_presence) or \
                    (vid_present and not check_img_presence) or \
                        (not check_img_presence and not check_vid_presence):
                    break

    # check if user selected to process either images or videos
    if not img_present and not vid_present:
        if simple_mode:
            mb.showerror(["No data found", "No se han encontrado datos", "Aucune donnée trouvée"][lang_idx],
                            message=[f"There are no images nor videos found.\n\nAddaxAI accepts images in the format {IMG_EXTENSIONS}."
                                     f"\n\nIt accepts videos in the format {VIDEO_EXTENSIONS}.",
                                     f"No se han encontrado imágenes ni vídeos.\n\nAddaxAI acepta imágenes en formato {IMG_EXTENSIONS}."
                                     f"\n\nAcepta vídeos en formato {VIDEO_EXTENSIONS}.",
                                     f"Aucune image ou vidéo trouvé.\n\nAddaxAI accepte des images au format {IMG_EXTENSIONS}."
                                     f"\n\nLes vidéos au format {VIDEO_EXTENSIONS} sont également acceptés."][lang_idx])
        else:
            mb.showerror(["No data found", "No se han encontrado datos", "Aucune donnée trouvée"][lang_idx],
                            message=[f"There are no images nor videos found, or you selected not to search for them. If there is indeed data to be "
                                    f"processed, make sure the '{lbl_process_img_txt[lang_idx]}' and/or '{lbl_process_vid_txt[lang_idx]}' options "
                                    f"are selected. You must select at least one of these.\n\nAddaxAI accepts images in the format {IMG_EXTENSIONS}."
                                    f"\n\nIt accepts videos in the format {VIDEO_EXTENSIONS}.",
                                    f"No se han encontrado imágenes ni vídeos, o ha seleccionado no buscarlos. Si efectivamente hay datos para procesar,"
                                    f" asegúrese de que las opciones '{lbl_process_img_txt[lang_idx]}' y/o '{lbl_process_vid_txt[lang_idx]}' están seleccionadas."
                                    f" Debe seleccionar al menos una de ellas.\n\nAddaxAI acepta imágenes en formato {IMG_EXTENSIONS}."
                                    f"\n\nAcepta vídeos en formato {VIDEO_EXTENSIONS}.",
                                    f"Aucune image ou vidéo trouvé, ou vous avez sélectionné de ne pas faire de recherche pour ces derniers. Si des données à traiter "
                                    f"existent, assurez-vous que les options'{lbl_process_img_txt[lang_idx]}' et/ou '{lbl_process_vid_txt[lang_idx]}' "
                                    f"sont sélectionnées. Vous devez sélectionner au moins l'une d'entre elles.\n\nAddaxAI accepte des images au format {IMG_EXTENSIONS}."
                                    f"\n\nLes vidéos au format {VIDEO_EXTENSIONS} sont également acceptés."][lang_idx])
        btn_start_deploy.configure(state=NORMAL)
        sim_run_btn.configure(state=NORMAL)
        return

    # run species net
    if var_cls_model.get() == "Global - SpeciesNet - Google":
        
        # if simple mode, tell user to use the advanced mode
        if simple_mode:
            mb.showerror(["SpeciesNet not available", "SpeciesNet no disponible", "SpeciesNet non-disponible"][lang_idx],
                            message=[f"SpeciesNet is not available in simple mode. Please switch to advanced mode to use SpeciesNet.",
                                        f"SpeciesNet no está disponible en modo simple. Cambie al modo avanzado para usar SpeciesNet.",
                                        f"SpeciesNet n'est pas disponible en mode simple. SVP choisir le mode avancé pour utiliser SpeciesNet."][lang_idx])
            
            # reset
            btn_start_deploy.configure(state=NORMAL)
            sim_run_btn.configure(state=NORMAL)
            return
        
        # if videos present, tell users that Species net cannot process them
        if vid_present:
            mb.showerror(["SpeciesNet not available", "SpeciesNet no disponible", "SpeciesNet non-disponible"][lang_idx],
                            message=[f"Video support for SpeciesNet will be available in a future AddaxAI release, please uncheck 'process videos'.",
                                        f"El soporte de video para SpeciesNet estará disponible en una futura versión de AddaxAI, por favor desmarque 'procesar videos'.",
                                        f"Le support pour vidéo avec SpeciesNet sera disponible dans une version future d'AddaxAI, svp décocher la case 'traiter les vidéos'."][lang_idx])
            # reset
            btn_start_deploy.configure(state=NORMAL)
            sim_run_btn.configure(state=NORMAL)
            return
        
        # check if env-speciesnet needs to be downloaded
        model_vars = load_model_vars(model_type = "cls")
        bool, env_name = environment_needs_downloading(model_vars)
        if bool: # env needs be downloaded, ask user 
            user_wants_to_download = download_environment(env_name, model_vars)
            if not user_wants_to_download:
                btn_start_deploy.configure(state=NORMAL)
                sim_run_btn.configure(state=NORMAL)
                return  # user doesn't want to download

        # open progress window
        sppnet_output_window = SpeciesNetOutputWindow()
        sppnet_output_window.add_string("SpeciesNet is starting up...\n\n")

        # deploy speciesnet
        try:
            return_value = deploy_speciesnet(chosen_folder, sppnet_output_window)
        
            # due to a package conflict on macos there might need to be a restart
            if return_value == "restart":
                sppnet_output_window.add_string("\n\nRestarting SpeciesNet...\n\n")
                deploy_speciesnet(chosen_folder, sppnet_output_window)
                
            # enable stuff
            btn_start_deploy.configure(state=NORMAL)
            sim_run_btn.configure(state=NORMAL)
            sppnet_output_window.close()
            return
    
        except Exception as error:
            # log error
            print("\n\nERROR:\n" + str(error) + "\n\nTRACEBACK:\n" + traceback.format_exc() + "\n\n")
            
            # show error
            mb.showerror(title=error_txt[lang_idx],
                        message=["An error has occurred", "Ha ocurrido un error", "Une erreur s'est produite"][lang_idx] + " (AddaxAI v" + current_AA_version + "): '" + str(error) + "'.",
                        detail= traceback.format_exc())
            
            # enable stuff
            btn_start_deploy.configure(state=NORMAL)
            sim_run_btn.configure(state=NORMAL)
            sppnet_output_window.close()
            return

    # note if user is video analysing without smoothing
    global warn_smooth_vid
    if (var_cls_model.get() not in none_txt) and \
        (var_smooth_cls_animal.get() == False) and \
            vid_present and \
                simple_mode == False and \
                    warn_smooth_vid == True:
                        warn_smooth_vid = False
                        if not mb.askyesno(information_txt[lang_idx], ["You are about to analyze videos without smoothing the confidence scores. "
                            "Typically, a video may contain many frames of the same animal, increasing the likelihood that at least "
                            f"one of the labels could be a false prediction. With '{lbl_smooth_cls_animal_txt[lang_idx]}' enabled, all"
                            " predictions from a single video will be averaged, resulting in only one label per video. Do you wish to"
                            " continue without smoothing?\n\nPress 'No' to go back.", "Estás a punto de analizar videos sin suavizado "
                            "habilitado. Normalmente, un video puede contener muchos cuadros del mismo animal, lo que aumenta la "
                            "probabilidad de que al menos una de las etiquetas pueda ser una predicción falsa. Con "
                            f"'{lbl_smooth_cls_animal_txt[lang_idx]}' habilitado, todas las predicciones de un solo video se promediarán,"
                            " lo que resultará en una sola etiqueta por video. ¿Deseas continuar sin suavizado habilitado?\n\nPresiona "
                            "'No' para regresar.",
                            "Vous êtes sur le point d'analyser des vidéos sans lisser les scores de confiance. "
                            "Typiquement, un vidéo peut contenir plusieurs images d'un même animal, ce qui augmente les chances qu'au moins un "
                            f"des labels puisse être une fausse prédiction. Avec '{lbl_smooth_cls_animal_txt[lang_idx]}' activé, toute"
                            " les prédictions d'un seul vidéo seront moyennées, résultant en un seul label par vidéo. Souhaitez-vous"
                            " continuer sans lissage?\n\nAppuyer sur 'Non' pour revenir en arrière."][lang_idx]):
                            return
    
    # de not allow full image classifier to process videos
    full_image_cls = load_model_vars("cls").get("full_image_cls", False)
    if full_image_cls:
        vid_present = False
    
    # check which processes need to be listed on the progress window
    if simple_mode:
        processes = []
        if img_present:
            processes.append("img_det")
            if var_cls_model.get() not in none_txt:
                processes.append("img_cls")
        if vid_present:
            processes.append("vid_det")
            if var_cls_model.get() not in none_txt:
                processes.append("vid_cls")
        if not timelapse_mode and img_present:
            processes.append("img_pst")
        if not timelapse_mode and vid_present:
            processes.append("vid_pst")
        if not timelapse_mode:
            processes.append("plt")
    else:
        processes = []
        if img_present:
            processes.append("img_det")
            if var_cls_model.get() not in none_txt:
                processes.append("img_cls")
        if vid_present:
            processes.append("vid_det")
            if var_cls_model.get() not in none_txt:
                processes.append("vid_cls")
    
    # if working with a full image classifier is selected, remove the detection processes and video stuff
    full_image_cls = load_model_vars("cls").get("full_image_cls", False)
    if full_image_cls:
        if "img_det" in processes:
            processes.remove("img_det")
        if "vid_det" in processes:
            processes.remove("vid_det")
        if "vid_cls" in processes:
            processes.remove("vid_cls")
        if "vid_pst" in processes:
            processes.remove("vid_pst")
    
    # redirect warnings and error to log files
    global model_error_log
    model_error_log = os.path.join(chosen_folder, "model_error_log.txt")
    global model_warning_log
    model_warning_log = os.path.join(chosen_folder, "model_warning_log.txt")
    global model_special_char_log
    model_special_char_log = os.path.join(chosen_folder, "model_special_char_log.txt")

    # set global variable
    temp_frame_folder_created = False

    # make sure user doesn't press the button twice
    btn_start_deploy.configure(state=DISABLED)
    sim_run_btn.configure(state=DISABLED)
    root.update()

    # check if models need to be downloaded
    if simple_mode:
        var_det_model.set("MegaDetector 5a")
    for model_type in ["cls", "det"]:
        model_vars = load_model_vars(model_type = model_type)
        if model_vars == {}: # if selected model is None
            continue
        bool, dirpath = model_needs_downloading(model_vars, model_type)
        if bool is None: # EA needs updating, return to window
            btn_start_deploy.configure(state=NORMAL)
            sim_run_btn.configure(state=NORMAL)
            return 
        elif bool: # model can be downloaded, ask user 
            user_wants_to_download = download_model(dirpath)
            if not user_wants_to_download:
                btn_start_deploy.configure(state=NORMAL)
                sim_run_btn.configure(state=NORMAL)
                return  # user doesn't want to download

    # check if environment need to be downloaded
    if simple_mode:
        var_det_model.set("MegaDetector 5a")
    for model_type in ["cls", "det"]:
        model_vars = load_model_vars(model_type = model_type)
        if model_vars == {}: # if selected model is None
            continue
        bool, env_name = environment_needs_downloading(model_vars)
        if bool: # env needs be downloaded, ask user 
            user_wants_to_download = download_environment(env_name, model_vars)
            if not user_wants_to_download:
                btn_start_deploy.configure(state=NORMAL)
                sim_run_btn.configure(state=NORMAL)
                return  # user doesn't want to download

    # run some checks that make sense for both simple and advanced mode
    # check if chosen folder is valid
    if chosen_folder in ["", "/", "\\", ".", "~", ":"] or not os.path.isdir(chosen_folder):
        mb.showerror(error_txt[lang_idx],
            message=["Please specify a directory with data to be processed.",
                     "Por favor, especifique un directorio con los datos a procesar.",
                     "SVP spécifier un répertoire avec des données à traiter."][lang_idx])
        btn_start_deploy.configure(state=NORMAL)
        sim_run_btn.configure(state=NORMAL)
        return

    # save simple settings for next time
    write_global_vars({
        "lang_idx": lang_idx,
        "var_cls_model_idx": dpd_options_cls_model[lang_idx].index(var_cls_model.get()),
        "var_sppnet_location_idx": dpd_options_sppnet_location[lang_idx].index(var_sppnet_location.get()),
    })

    # simple_mode and advanced mode shared image settings
    additional_img_options = ["--output_relative_filenames"]
    
    # simple_mode and advanced mode shared video settings
    additional_vid_options = ["--json_confidence_threshold=0.01"]
    if timelapse_mode:
        additional_vid_options.append("--include_all_processed_frames")
    temp_frame_folder_created = False
    if vid_present:
        if var_cls_model.get() not in none_txt:
            global temp_frame_folder
            temp_frame_folder_obj = tempfile.TemporaryDirectory()
            temp_frame_folder_created = True
            temp_frame_folder = temp_frame_folder_obj.name
            additional_vid_options.append("--frame_folder=" + temp_frame_folder)
            additional_vid_options.append("--keep_extracted_frames")


    # if user deployed from simple mode everything will be default, so easy
    if simple_mode:
        
        # simple mode specific image options
        additional_img_options.append("--recursive")
        
        # simple mode specific video options
        additional_vid_options.append("--recursive")
        additional_vid_options.append("--time_sample=1")

    # if the user comes from the advanced mode, there are more settings to be checked
    else:
        # save advanced settings for next time
        write_global_vars({
            "var_det_model_idx": dpd_options_model[lang_idx].index(var_det_model.get()),
            "var_det_model_path": var_det_model_path.get(),
            "var_det_model_short": var_det_model_short.get(),
            "var_exclude_subs": var_exclude_subs.get(),
            "var_use_custom_img_size_for_deploy": var_use_custom_img_size_for_deploy.get(),
            "var_image_size_for_deploy": var_image_size_for_deploy.get() if var_image_size_for_deploy.get().isdigit() else "",
            "var_abs_paths": var_abs_paths.get(),
            "var_disable_GPU": var_disable_GPU.get(),
            "var_process_img": var_process_img.get(),
            "var_use_checkpnts": var_use_checkpnts.get(),
            "var_checkpoint_freq": var_checkpoint_freq.get() if var_checkpoint_freq.get().isdecimal() else "",
            "var_cont_checkpnt": var_cont_checkpnt.get(),
            "var_process_vid": var_process_vid.get(),
            "var_not_all_frames": var_not_all_frames.get(),
            "var_nth_frame": var_nth_frame.get() if var_nth_frame.get().isdecimal() else ""
        })
        
        # check if checkpoint entry is valid
        if var_use_custom_img_size_for_deploy.get() and not var_image_size_for_deploy.get().isdecimal():
            mb.showerror(invalid_value_txt[lang_idx],
                        ["You either entered an invalid value for the image size, or none at all. You can only "
                        "enter numeric characters.",
                        "Ha introducido un valor no válido para el tamaño de la imagen o no ha introducido ninguno. "
                        "Sólo puede introducir caracteres numéricos.",
                        "Vous avez saisi une valeur invalide pour les dimensions de l'image, ou aucune valeur du tout.. Vous ne pouvez "
                        "que saisir des caractères numériques."][lang_idx])
            btn_start_deploy.configure(state=NORMAL)
            sim_run_btn.configure(state=NORMAL)
            return

        # check if checkpoint entry is valid
        if var_use_checkpnts.get() and not var_checkpoint_freq.get().isdecimal():
            if mb.askyesno(invalid_value_txt[lang_idx],
                            ["You either entered an invalid value for the checkpoint frequency, or none at all. You can only "
                            "enter numeric characters.\n\nDo you want to proceed with the default value 500?",
                            "Ha introducido un valor no válido para la frecuencia del punto de control o no ha introducido ninguno. "
                            "Sólo puede introducir caracteres numéricos.\n\n¿Desea continuar con el valor por defecto 500?",
                            "Vous avez saisi une valeur invalide pour la fréquence des points de contrôle, ou aucune valeur du tout. Vous ne pouvez "
                            "que saisir des caractères numériques.\n\nSouhaitez-vous utiliser la valeur par défaut de 500?"][lang_idx]):
                var_checkpoint_freq.set("500")
                ent_checkpoint_freq.configure(fg='black')
            else:
                btn_start_deploy.configure(state=NORMAL)
                sim_run_btn.configure(state=NORMAL)
                
                return
        
        # check if the nth frame entry is valid
        if var_not_all_frames.get() and not is_valid_float(var_nth_frame.get()):
            if mb.askyesno(invalid_value_txt[lang_idx],
                           [f"Invalid input for '{lbl_nth_frame_txt[lang_idx]}'. Please enter a numeric value (e.g., '1', '1.5', '0.3', '7')."
                            " Non-numeric values like 'two' or '1,2' are not allowed.\n\nWould you like to proceed with the default value"
                            " of 1?\n\nThis means the program will only process 1 frame every second.", "Entrada no válida para "
                            f"'{lbl_nth_frame_txt[lang_idx]}'. Introduzca un valor numérico (por ejemplo, 1, 1.5, 0.3). Valores no numéricos como"
                            " 'dos' o '1,2' no están permitidos.\n\n¿Desea continuar con el valor predeterminado de 1?\n\nEsto significa que"
                            " el programa solo procesará 1 fotograma cada segundo.",
                            f"Entrée invalide pour '{lbl_nth_frame_txt[lang_idx]}'. SVP entrer une valeur numérique (par ex.: '1', '1.5', '0.3', '7')."
                            " Les valeurs non-numérique comme 'deux' ou '1,2' ne sont pas permises.\n\nVoulez-vous continuer avec la valeur par défaut "
                            " de 1?\n\nCela signifie que le programme ne traitera qu'une seule image par seconde."][lang_idx]):
                var_nth_frame.set("1")
                ent_nth_frame.configure(fg='black')
            else:
                btn_start_deploy.configure(state=NORMAL)
                sim_run_btn.configure(state=NORMAL)
                return

        # create command for the image process to be passed on to run_detector_batch.py
        if not var_exclude_subs.get():
            additional_img_options.append("--recursive")
        if var_use_checkpnts.get():
            additional_img_options.append("--checkpoint_frequency=" + var_checkpoint_freq.get())
        if var_cont_checkpnt.get() and check_checkpnt():
            additional_img_options.append("--resume_from_checkpoint=" + loc_chkpnt_file)
        if var_use_custom_img_size_for_deploy.get():
            additional_img_options.append("--image_size=" + var_image_size_for_deploy.get())

        # create command for the video process to be passed on to process_video.py
        if not var_exclude_subs.get():
            additional_vid_options.append("--recursive")
        if var_not_all_frames.get():
            additional_vid_options.append("--time_sample=" + var_nth_frame.get())

    
    # open progress window with frames for each process that needs to be done
    global progress_window
    progress_window = ProgressWindow(processes = processes)
    progress_window.open()

    # check the chosen folder of special characters and alert the user is there are any
    isolated_special_fpaths = {"total_saved_images": 0}
    for main_dir, _, files in os.walk(chosen_folder):
        for file in files:
            file_path = os.path.join(main_dir, file)
            if os.path.splitext(file_path)[1].lower() in ['.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mpeg', '.mpg']:
                bool, char = contains_special_characters(file_path)
                if bool:
                    drive, rest_of_path = os.path.splitdrive(file_path)
                    path_components = rest_of_path.split(os.path.sep)
                    isolated_special_fpath = drive
                    for path_component in path_components: # check the largest dir that is faulty
                        isolated_special_fpath = os.path.join(isolated_special_fpath, path_component)
                        if contains_special_characters(path_component)[0]:
                            isolated_special_fpaths["total_saved_images"] += 1
                            if isolated_special_fpath in isolated_special_fpaths:
                                isolated_special_fpaths[isolated_special_fpath][0] += 1
                            else:
                                isolated_special_fpaths[isolated_special_fpath] = [1, char]
    n_special_chars = len(isolated_special_fpaths) - 1
    total_saved_images = isolated_special_fpaths['total_saved_images'];del isolated_special_fpaths['total_saved_images']

    if total_saved_images > 0:
        # write to log file 
        if os.path.isfile(model_special_char_log):
            os.remove(model_special_char_log)            
        for k, v in isolated_special_fpaths.items():
            line = f"There are {str(v[0]).ljust(4)} files hidden behind the {str(v[1])} character in folder '{k}'"
            if not line.isprintable():
                line = repr(line)
                print(f"\nSPECIAL CHARACTER LOG: This special character is going to give an error : {line}\n")  # log
            with open(model_special_char_log, 'a+', encoding='utf-8') as f:
                f.write(f"{line}\n")
        
        # log to console
        print(f"\nSPECIAL CHARACTER LOG: There are {total_saved_images} files hidden behind {n_special_chars} special characters.\n")

        # prompt user
        special_char_popup_btns = [["Continue with filepaths as they are now",
                                    "Open log file and review the problematic filepaths"],
                                ["Continuar con las rutas de archivo tal y como están ahora",
                                    "Abrir el archivo de registro y revisar las rutas de archivo probelmáticas"],
                                ["Continuer avec les chemins de fichiers tels quels",
                                "Ouvrez le fichier journal et examinez les fichiers problématiques"]][lang_idx]
        special_char_popup = TextButtonWindow(title = ["Special characters found", "Caracteres especiales encontrados", "Caractères spéciaux détectés"][lang_idx],
                                            text = ["Special characters can be problematic during analysis, resulting in files being skipped.\n"
                                                    f"With your current folder structure, there are a total of {total_saved_images} files that will be potentially skipped.\n"
                                                    f"If you want to make sure these images will be analysed, you would need to manually adjust the names of {n_special_chars} folders.\n"
                                                    "You can find an overview of the probelematic characters and filepaths in the log file:\n\n"
                                                    f"'{model_special_char_log}'\n\n"
                                                    f"You can also decide to continue with the filepaths as they are now, with the risk of excluding {total_saved_images} files.", 
                                                    "Los caracteres especiales pueden ser problemáticos durante el análisis, haciendo que se omitan archivos.\n"
                                                    f"Con su actual estructura de carpetas, hay un total de {total_saved_images} archivos que serán potencialmente omitidos.\n"
                                                    f"Si desea asegurarse de que estas imágenes se analizarán, deberá ajustar manualmente los nombres de las carpetas {n_special_chars}.\n"
                                                    "Puede encontrar un resumen de los caracteres problemáticos y las rutas de los archivos en el archivo de registro:\n\n"
                                                    f"'{model_special_char_log}'\n\n"
                                                    f"También puede decidir continuar con las rutas de archivo tal y como están ahora, con el riesgo de excluir archivos {total_saved_images}",
                                                    "Les caractères spéciaux peuvent être problématiques lors de l'analyse, ce qui entraîne l'omission de fichiers.\n"
                                                    f"Avec votre structure de dossiers actuelle, il y a un total de {total_saved_images} fichiers qui seront potentiellement ignorés.\n"
                                                    f"Si vous souhaitez vous assurer que ces images seront analysées, vous devrez ajuster manuellement les noms de {n_special_chars} dossiers.\n"
                                                    "Vous pouvez trouver un aperçu des caractères problématiques et des chemins de fichiers dans le fichier journal :\n\n"
                                                    f"'{model_special_char_log}'\n\n"
                                                    f"Vous pouvez également décider de continuer avec les chemins de fichiers tels qu'ils sont actuellement, avec le risque d'exclure "
                                                    "{total_saved_images} fichiers."][lang_idx],
                                            buttons = special_char_popup_btns)
        
        # run option window and check user input
        user_input = special_char_popup.run()
        if user_input != special_char_popup_btns[0]:
            # user does not want to continue as is
            if user_input == special_char_popup_btns[1]:
                # user chose to review paths, so open log file
                open_file_or_folder(model_special_char_log)
            # close progressbar and fix deploy buttuns
            btn_start_deploy.configure(state=NORMAL)
            sim_run_btn.configure(state=NORMAL)
            progress_window.close()
            return

    try:

        # process images and/or videos
        if img_present:
            deploy_model(chosen_folder, additional_img_options, data_type = "img", simple_mode = simple_mode)
        if vid_present:
            deploy_model(chosen_folder, additional_vid_options, data_type = "vid", simple_mode = simple_mode)
        
        # if deployed through simple mode, add predefined postprocess directly after deployment and classification
        if simple_mode and not timelapse_mode:

                # FIX: For videos in simple mode, convert frame results to video results BEFORE postprocess
                # This ensures the classification updates from .frames.json are applied to .json
                video_recognition_file = os.path.join(chosen_folder, "video_recognition_file.json")
                video_recognition_file_original = os.path.join(chosen_folder, "video_recognition_file_original.json")
                video_recognition_file_frame = os.path.join(chosen_folder, "video_recognition_file.frames.json")
                video_recognition_file_frame_original = os.path.join(chosen_folder, "video_recognition_file.frames_original.json")

                if os.path.isfile(video_recognition_file) and\
                   os.path.isfile(video_recognition_file_frame) and\
                   os.path.isfile(video_recognition_file_frame_original):

                    # get the frame_rates from the video_recognition_file.json
                    frame_rates = {}
                    with open(video_recognition_file) as f:
                        data = json.load(f)
                        images = data['images']
                        for image in images:
                            file = image['file']
                            frame_rate = image['frame_rate']
                            frame_rates[file] = frame_rate

                    # convert frame results to video results
                    options = FrameToVideoOptions()
                    options.include_all_processed_frames = False
                    frame_results_to_video_results(input_file = video_recognition_file_frame,
                                                output_file = video_recognition_file,
                                                options = options,
                                                video_filename_to_frame_rate = frame_rates)
                    frame_results_to_video_results(input_file = video_recognition_file_frame_original,
                                                output_file = video_recognition_file_original,
                                                options = options,
                                                video_filename_to_frame_rate = frame_rates)

                # if only analysing images, postprocess images with plots
                if "img_pst" in processes and not "vid_pst" in processes:
                    postprocess(src_dir = chosen_folder,
                                dst_dir = chosen_folder,
                                thresh = global_vars["var_thresh_default"],
                                sep = False,
                                file_placement = 1,
                                sep_conf = False,
                                vis = False,
                                crp = False,
                                exp = True,
                                plt = True,
                                exp_format = "XLSX",
                                data_type = "img")
                
                # if only analysing videos, postprocess videos with plots
                elif "vid_pst" in processes and not "img_pst" in processes:
                    postprocess(src_dir = chosen_folder,
                                dst_dir = chosen_folder,
                                thresh = global_vars["var_thresh_default"],
                                sep = False,
                                file_placement = 1,
                                sep_conf = False,
                                vis = False,
                                crp = False,
                                exp = True,
                                plt = True,
                                exp_format = "XLSX",
                                data_type = "vid")
                
                # otherwise postprocess first images without plots, and then videos with plots
                else:
                    postprocess(src_dir = chosen_folder,
                                dst_dir = chosen_folder,
                                thresh = global_vars["var_thresh_default"],
                                sep = False,
                                file_placement = 1,
                                sep_conf = False,
                                vis = False,
                                crp = False,
                                exp = True,
                                plt = False,
                                exp_format = "XLSX",
                                data_type = "img")
                    postprocess(src_dir = chosen_folder,
                                dst_dir = chosen_folder,
                                thresh = global_vars["var_thresh_default"],
                                sep = False,
                                file_placement = 1,
                                sep_conf = False,
                                vis = False,
                                crp = False,
                                exp = True,
                                plt = True,
                                exp_format = "XLSX",
                                data_type = "vid")

        # let's organise all the json files and check their presence
        image_recognition_file = os.path.join(chosen_folder, "image_recognition_file.json")
        image_recognition_file_original = os.path.join(chosen_folder, "image_recognition_file_original.json")
        video_recognition_file = os.path.join(chosen_folder, "video_recognition_file.json")
        video_recognition_file_original = os.path.join(chosen_folder, "video_recognition_file_original.json")
        video_recognition_file_frame = os.path.join(chosen_folder, "video_recognition_file.frames.json") 
        video_recognition_file_frame_original = os.path.join(chosen_folder, "video_recognition_file.frames_original.json")
        timelapse_json = os.path.join(chosen_folder, "timelapse_recognition_file.json")
        exif_data_json = os.path.join(chosen_folder, "exif_data.json")

        # convert to frame jsons to video jsons if frames are classified 
        if os.path.isfile(video_recognition_file) and\
            os.path.isfile(video_recognition_file_frame) and\
                os.path.isfile(video_recognition_file_frame_original):
                    
            # get the frame_rates from the video_recognition_file.json
            frame_rates = {}
            with open(video_recognition_file) as f:
                data = json.load(f)
                images = data['images']
                for image in images:
                    file = image['file']
                    frame_rate = image['frame_rate']
                    frame_rates[file] = frame_rate
        
            # convert frame results to video results
            options = FrameToVideoOptions()
            if timelapse_mode:
                options.include_all_processed_frames = True
            else:
                options.include_all_processed_frames = False
            frame_results_to_video_results(input_file = video_recognition_file_frame,
                                        output_file = video_recognition_file,
                                        options = options,
                                        video_filename_to_frame_rate = frame_rates)
            frame_results_to_video_results(input_file = video_recognition_file_frame_original,
                                        output_file = video_recognition_file_original,
                                        options = options,
                                        video_filename_to_frame_rate = frame_rates)

        # remove unnecessary jsons after conversion
        if os.path.isfile(video_recognition_file_frame_original):
            os.remove(video_recognition_file_frame_original)
        if os.path.isfile(video_recognition_file_frame):
            os.remove(video_recognition_file_frame)
        if os.path.isfile(exif_data_json):
            os.remove(exif_data_json)
        
        # prepare for Timelapse use
        if timelapse_mode:
            # merge json
            if var_cls_model.get() not in none_txt:
                # if a classification model is selected
                merge_jsons(image_recognition_file_original if os.path.isfile(image_recognition_file_original) else None,
                            video_recognition_file_original if os.path.isfile(video_recognition_file_original) else None,
                            timelapse_json)
            else:
                # if no classification model is selected
                merge_jsons(image_recognition_file if os.path.isfile(image_recognition_file) else None,
                            video_recognition_file if os.path.isfile(video_recognition_file) else None,
                            timelapse_json)
            
            # remove unnecessary jsons
            if os.path.isfile(image_recognition_file_original):
                os.remove(image_recognition_file_original)
            if os.path.isfile(image_recognition_file):
                os.remove(image_recognition_file)
            if os.path.isfile(video_recognition_file_original):
                os.remove(video_recognition_file_original)
            if os.path.isfile(video_recognition_file):
                os.remove(video_recognition_file)
        
        # prepare for AddaxAI use
        else:
            
            # # If at a later stage I want a merged json for AddaxAI too - this is the code
            # merge_jsons(image_recognition_file if os.path.isfile(image_recognition_file) else None,
            #             video_recognition_file if os.path.isfile(video_recognition_file) else None,
            #             os.path.join(chosen_folder, "merged_recognition_file.json"))
            
            # remove unnecessary jsons
            if os.path.isfile(image_recognition_file_original):
                os.remove(image_recognition_file_original)
            if os.path.isfile(video_recognition_file_original):
                os.remove(video_recognition_file_original)

        # reset window
        update_frame_states()
        
        # close progress window
        progress_window.close()

        # clean up temp folder with frames
        if temp_frame_folder_created:
            temp_frame_folder_obj.cleanup()

        # show model error pop up window
        if os.path.isfile(model_error_log):
            mb.showerror(error_txt[lang_idx], [f"There were one or more model errors. See\n\n'{model_error_log}'\n\nfor more information.",
                                            f"Se han producido uno o más errores de modelo. Consulte\n\n'{model_error_log}'\n\npara obtener más información.",
                                            f"Une ou plusieurs erreurs ont été générées par le modèle. Voir\n\n'{model_error_log}'\n\npour plus d'informations."][lang_idx])

        # show model warning pop up window
        if os.path.isfile(model_warning_log):
            mb.showerror(error_txt[lang_idx], [f"There were one or more model warnings. See\n\n'{model_warning_log}'\n\nfor more information.",
                                        f"Se han producido uno o más advertencias de modelo. Consulte\n\n'{model_warning_log}'\n\npara obtener más información.",
                                        f"Un ou plusieurs avertissements ont été générés par le modèle. Voir\n\n'{model_error_log}'\n\npour plus d'informations."][lang_idx])

        # show postprocessing warning log
        global postprocessing_error_log
        postprocessing_error_log = os.path.join(chosen_folder, "postprocessing_error_log.txt")
        if os.path.isfile(postprocessing_error_log): 
            mb.showwarning(warning_txt[lang_idx], [f"One or more files failed to be analysed by the model (e.g., corrupt files) and will be skipped by "
                                                f"post-processing features. See\n\n'{postprocessing_error_log}'\n\nfor more info.",
                                                f"Uno o más archivos no han podido ser analizados por el modelo (por ejemplo, ficheros corruptos) y serán "
                                                f"omitidos por las funciones de post-procesamiento. Para más información, véase\n\n'{postprocessing_error_log}'",
                                                f"Un ou plusieurs fichiers n'ont pas pu être analysés par le modèle (par exemple, des fichiers corrompus) et seront ignorés "
                                                f"lors du post-traitement. Voir\n\n'{postprocessing_error_log}'\n\npour plus d'informations."][lang_idx])

        # enable button
        btn_start_deploy.configure(state=NORMAL)
        sim_run_btn.configure(state=NORMAL)
        root.update()

        # show results
        if timelapse_mode:
            mb.showinfo("Analysis done!", f"Recognition file created at \n\n{timelapse_json}\n\nTo use it in Timelapse, return to "
                                            "Timelapse with the relevant image set open, select the menu item 'Recognition > Import "
                                            "recognition data for this image set' and navigate to the file above.")
            open_file_or_folder(os.path.dirname(timelapse_json))
        elif simple_mode:
            show_result_info(os.path.join(chosen_folder, "results.xlsx"))

    except Exception as error:

        # log error
        print("\n\nERROR:\n" + str(error) + "\n\nSUBPROCESS OUTPUT:\n" + subprocess_output + "\n\nTRACEBACK:\n" + traceback.format_exc() + "\n\n")
        print(f"cancel_deploy_model_pressed : {cancel_deploy_model_pressed}")

        if cancel_deploy_model_pressed:
            pass
        
        else:
            # show error
            mb.showerror(title=error_txt[lang_idx],
                        message=["An error has occurred", "Ha ocurrido un error", "Une erreur est survenue"][lang_idx] + " (AddaxAI v" + current_AA_version + "): '" + str(error) + "'.",
                        detail=subprocess_output + "\n" + traceback.format_exc())
            
            # close window
            progress_window.close()

            # enable button
            btn_start_deploy.configure(state=NORMAL)
            sim_run_btn.configure(state=NORMAL)

# get data from file list and create graph
def produce_graph(file_list_txt = None, dir = None):
    
    # if a list with images is specified
    if file_list_txt:
        count_dict = {}

        # loop through the files
        with open(file_list_txt) as f:
            for line in f:

                # open xml 
                img = line.rstrip()
                annotation = return_xml_path(img)
                tree = ET.parse(annotation)
                root = tree.getroot()

                # loop through detections
                for obj in root.findall('object'):

                    # add detection to dict
                    name = obj.findtext('name')
                    if name not in count_dict:
                        count_dict[name] = 0
                    count_dict[name] += 1
            f.close()

        # create plot
        classes = list(count_dict.keys())
        counts = list(count_dict.values())
        fig = plt.figure(figsize = (10, 5))
        plt.bar(classes, counts, width = 0.4, color=green_primary)
        plt.ylabel(["No. of instances verified", "No de instancias verificadas", "No. de l'instance vérifiée"][lang_idx])
        plt.close()

        # return results
        return fig

# create pascal voc annotation files from a list of detections
def create_pascal_voc_annotation(image_path, annotation_list, human_verified):

    # init vars
    image_path = Path(image_path)
    img = np.array(Image.open(image_path).convert('RGB'))
    annotation = ET.Element('annotation')

    # set verified flag if been verified in a previous session
    if human_verified:
        annotation.set('verified', 'yes')

    ET.SubElement(annotation, 'folder').text = str(image_path.parent.name)
    ET.SubElement(annotation, 'filename').text = str(image_path.name)
    ET.SubElement(annotation, 'path').text = str(image_path)

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(img.shape[1])
    ET.SubElement(size, 'height').text = str(img.shape[0])
    ET.SubElement(size, 'depth').text = str(img.shape[2])

    ET.SubElement(annotation, 'segmented').text = '0'

    for annot in annotation_list:
        tmp_annot = annot.split(',')
        cords, label = tmp_annot[0:-2], tmp_annot[-1]
        xmin, ymin, xmax, ymax = cords[0], cords[1], cords[4], cords[5] # left, top, right, bottom

        object = ET.SubElement(annotation, 'object')
        ET.SubElement(object, 'name').text = label
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = '0'
        ET.SubElement(object, 'difficult').text = '0'

        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    indent(annotation)
    tree = ET.ElementTree(annotation)
    xml_file_name = return_xml_path(image_path)
    Path(os.path.dirname(xml_file_name)).mkdir(parents=True, exist_ok=True)
    tree.write(xml_file_name)

# loop json and see which images and annotations fall in user-specified catgegory
def select_detections(selection_dict, prepare_files):

    # open patience window
    steps_progress = PatienceDialog(total = 8, text = [f"Loading...", f"Cargando...", f"Chargement..."][lang_idx])
    steps_progress.open()
    current_step = 1
    steps_progress.update_progress(current_step);current_step += 1

    # init vars
    selected_dir = var_choose_folder.get()
    recognition_file = os.path.join(selected_dir, 'image_recognition_file.json')
    temp_folder = os.path.join(selected_dir, 'temp-folder')
    Path(temp_folder).mkdir(parents=True, exist_ok=True)
    file_list_txt = os.path.join(temp_folder, 'hitl_file_list.txt')
    class_list_txt = os.path.join(temp_folder, 'hitl_class_list.txt')
    steps_progress.update_progress(current_step);current_step += 1

    # make sure json has relative paths
    json_paths_converted = False
    if check_json_paths(recognition_file) != "relative":
        make_json_relative(recognition_file)
        json_paths_converted = True
    steps_progress.update_progress(current_step);current_step += 1

    # list selection criteria
    selected_categories = []
    min_confs = []
    max_confs = []
    ann_min_confs_specific = {}
    selected_files = {}
    rad_ann_val = rad_ann_var.get()
    ann_min_confs_generic = None
    steps_progress.update_progress(current_step);current_step += 1

    # class specific values
    for key, values in selection_dict.items():
        category = values['class']
        chb_val = values['chb_var'].get()
        min_conf = round(values['min_conf_var'].get(), 2)
        max_conf = round(values['max_conf_var'].get(), 2)
        ann_min_conf_specific = values['scl_ann_var_specific'].get()
        ann_min_confs_generic = values['scl_ann_var_generic'].get()
        ann_min_confs_specific[category] = ann_min_conf_specific
        
        # if class is selected
        if chb_val:
            selected_categories.append(category)
            min_confs.append(min_conf)
            max_confs.append(max_conf)
            selected_files[category] = []
    steps_progress.update_progress(current_step);current_step += 1

    # remove old file list if present
    if prepare_files:
        if os.path.isfile(file_list_txt):
            os.remove(file_list_txt)
    steps_progress.update_progress(current_step);current_step += 1

    # loop though images and list those which pass the criteria
    img_and_detections_dict = {}
    with open(recognition_file, "r") as image_recognition_file_content:
        data = json.load(image_recognition_file_content)
        label_map = fetch_label_map_from_json(recognition_file)

        # check all images...
        for image in data['images']:

            # set vars
            image_path = os.path.join(selected_dir, image['file'])
            annotations = []
            image_already_added = False

            # check if the image has already been human verified
            try:
                human_verified = image['manually_checked']
            except:
                human_verified = False
            
            # check all detections ...
            if 'detections' in image:
                for detection in image['detections']:
                    category_id = detection['category']
                    category = label_map[category_id]
                    conf = detection['conf']

                    # ... if they pass any of the criteria
                    for i in range(len(selected_categories)):
                        if category == selected_categories[i] and conf >= min_confs[i] and conf <= max_confs[i]:
                            
                            # this image contains one or more detections which pass
                            if not image_already_added:
                                selected_files[selected_categories[i]].append(image_path)
                                image_already_added = True

                    # prepare annotations
                    if prepare_files:
                        display_annotation = False

                        # if one annotation threshold for all classes is specified
                        if rad_ann_val == 1 and conf >= ann_min_confs_generic:
                            display_annotation = True
                        
                        # if class-specific annotation thresholds are specified
                        elif rad_ann_val == 2 and conf >= ann_min_confs_specific[category]: 
                            display_annotation = True

                        # add this detection to the list
                        if display_annotation:
                            im = Image.open(image_path)
                            width, height = im.size
                            left = int(round(detection['bbox'][0] * width)) # xmin
                            top = int(round(detection['bbox'][1] * height)) # ymin 
                            right = int(round(detection['bbox'][2] * width)) + left # width
                            bottom = int(round(detection['bbox'][3] * height)) + top # height
                            list = [left, top, None, None, right, bottom, None, category]
                            string = ','.join(map(str, list))
                            annotations.append(string)
                                    
            # create pascal voc annotation file for this image
            if prepare_files:
                img_and_detections_dict[image_path] = {"annotations": annotations, "human_verified": human_verified}
    steps_progress.update_progress(current_step);current_step += 1

    # update count widget
    total_imgs = 0
    for category, files in selected_files.items():
        label_map = fetch_label_map_from_json(recognition_file)
        classes_list = [v for k, v in label_map.items()]
        row = classes_list.index(category) + 2
        frame = selection_dict[row]['frame']
        lbl_n_img = selection_dict[row]['lbl_n_img']
        chb_var = selection_dict[row]['chb_var'].get()
        rad_var = selection_dict[row]['rad_var'].get()

        # if user specified a percentage of total images
        if chb_var and rad_var == 2:

            # check if entry is valid
            ent_per_var = selection_dict[row]['ent_per_var'].get()
            try:
                ent_per_var = float(ent_per_var)
            except:
                invalid_value_warning([f"percentage of images for class '{category}'", f"porcentaje de imágenes para la clase '{category}'",
                                       f"pourcentage d'images pour la classe '{category}'"][lang_idx])
                return
            if ent_per_var == "" or ent_per_var < 0 or ent_per_var > 100:
                invalid_value_warning([f"percentage of images for class '{category}'", f"porcentaje de imágenes para la clase '{category}'",
                                       f"pourcentage d'images pour la classe '{category}'"][lang_idx])
                return
            
            # randomly select percentage of images
            total_n = len(files)
            n_selected = int(total_n * (ent_per_var / 100))
            random.shuffle(files)
            files = files[:n_selected]

        # user specified a max number of images
        elif chb_var and rad_var == 3: 

            # check if entry is valid
            ent_amt_var = selection_dict[row]['ent_amt_var'].get()
            try:
                ent_amt_var = float(ent_amt_var)
            except:
                invalid_value_warning([f"number of images for class '{category}'", f"número de imágenes para la clase '{category}'",
                                       f"nombre d'images pour la classe '{category}'"][lang_idx])
                return
            if ent_amt_var == "":
                invalid_value_warning([f"number of images for class '{category}'", f"número de imágenes para la clase '{category}'",
                                       f"nombre d'images pour la classe '{category}'"][lang_idx])
                return

            # randomly select specified number of images
            total_n = len(files)
            n_selected = int(ent_amt_var)
            random.shuffle(files)
            files = files[:n_selected]

        # update label text 
        n_imgs = len(files)
        lbl_n_img.configure(text = str(n_imgs))
        total_imgs += n_imgs

        # loop through the ultimately selected images and create files
        if prepare_files and len(files) > 0:

            # open patience window
            patience_dialog = PatienceDialog(total = n_imgs, text = [f"Preparing files for {category}...", f"Preparando archivos para {category}...",
                                                                     f"Préparation des fichiers pour {category}..."][lang_idx])
            patience_dialog.open()
            current = 1
            
            # human sort images per class
            def atoi(text):
                return int(text) if text.isdigit() else text
            def natural_keys(text):
                return [atoi(c) for c in re.split('(\d+)', text)]
            files.sort(key=natural_keys)

            for img in files:

                # update patience window
                patience_dialog.update_progress(current)
                current += 1

                # create text file with images
                file_list_txt = os.path.normpath(file_list_txt)
                with open(file_list_txt, 'a') as f:
                    f.write(f"{os.path.normpath(img)}\n")
                    f.close()
                
                # # list annotations 
                annotation_path = return_xml_path(img)

                # create xml file if not already present
                if not os.path.isfile(annotation_path):
                    create_pascal_voc_annotation(img, img_and_detections_dict[img]['annotations'], img_and_detections_dict[img]['human_verified'])

            # close patience window
            patience_dialog.close()      
    steps_progress.update_progress(current_step);current_step += 1
    steps_progress.close()
    
    # if the user want to sort the files alphabetically
    global_vars = load_global_vars()
    if global_vars["var_hitl_file_order"] == 1:
        
        # read all lines of the file list
        if os.path.isfile(file_list_txt):
            with open(file_list_txt) as f:
                previous_lines = f.readlines()
            
            # remove old file list
            os.remove(file_list_txt)
            
            # Apply natural sort using custom key
            sorted_lines = sorted(previous_lines, key=natural_sort_key)
        
            # and write them back in aphabetical order
            with open(file_list_txt, 'w') as f:
                for line in sorted_lines:
                    f.write(line + '\n')
        
    # update total number of images
    lbl_n_total_imgs.configure(text = [f"TOTAL: {total_imgs}", f"TOTAL: {total_imgs}", f"TOTAL: {total_imgs}"][lang_idx])
    
    if prepare_files:

        # TODO: hier moet ook een progress window komen als het een grote file is

        # create file with classes
        with open(class_list_txt, 'a') as f:
            for k, v in label_map.items():
                f.write(f"{v}\n")
            f.close()

        # write arguments to file in case user quits and continues later
        annotation_arguments = {"recognition_file" : recognition_file,
                                "class_list_txt" : class_list_txt,
                                "file_list_txt" : file_list_txt,
                                "label_map" : label_map,
                                "img_and_detections_dict" : img_and_detections_dict}

        annotation_arguments_pkl = os.path.join(selected_dir, 'temp-folder', 'annotation_information.pkl')
        with open(annotation_arguments_pkl, 'wb') as fp:
            pickle.dump(annotation_arguments, fp)
            fp.close()

        # start human in the loop process
        try:
            open_annotation_windows(recognition_file = recognition_file,
                                    class_list_txt = class_list_txt,
                                    file_list_txt = file_list_txt,
                                    label_map = label_map)
        except Exception as error:
            # log error
            print("ERROR:\n" + str(error) + "\n\nDETAILS:\n" + str(traceback.format_exc()) + "\n\n")
            
            # show error
            mb.showerror(title=error_txt[lang_idx],
                        message=["An error has occurred", "Ha ocurrido un error", "Une erreur est survenue"][lang_idx] + " (AddaxAI v" + current_AA_version + "): '" + str(error) + "'.",
                        detail=traceback.format_exc())

    # change json paths back, if converted earlier
    if json_paths_converted:
        make_json_absolute(recognition_file)

# Split string into a list of text and number chunks for natural sort
def natural_sort_key(s):
    # Remove leading/trailing whitespace
    s = s.strip()
    # Split into parts: e.g., "IMG_100.JPG" → ["IMG_", 100, ".JPG"]
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# count confidence values per class for histograms
def fetch_confs_per_class(json_fpath):
    label_map = fetch_label_map_from_json(os.path.join(var_choose_folder.get(), 'image_recognition_file.json'))
    confs = {}
    for key in label_map:
        confs[key] = []
    with open(json_fpath) as content:
        data = json.load(content)
        for image in data['images']:
            if 'detections' in image:
                for detection in image['detections']:
                    conf = detection["conf"]
                    category = detection["category"]
                    confs[category].append(conf)
    return confs

# open the human-in-the-loop settings window
def open_hitl_settings_window():
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # TODO: this window pops up behind the main AddaxAI window on windows OS. place in front, or hide AddaxAI frame.

    # fetch confs for histograms
    confs = fetch_confs_per_class(os.path.join(var_choose_folder.get(), 'image_recognition_file.json'))

    # set global vars
    global selection_dict
    global rad_ann_var
    global hitl_ann_selection_frame
    global hitl_settings_canvas
    global hitl_settings_window
    global lbl_n_total_imgs

    # init vars
    selected_dir = var_choose_folder.get()
    recognition_file = os.path.join(selected_dir, 'image_recognition_file.json')

    # init window
    hitl_settings_window = customtkinter.CTkToplevel(root)
    hitl_settings_window.title(["Verification selection settings", "Configuración de selección de verificación", "Vérification des paramètres de configuration"][lang_idx])
    hitl_settings_window.geometry("+10+10")
    hitl_settings_window.maxsize(width=ADV_WINDOW_WIDTH, height=800)

    # set scrollable frame
    hitl_settings_scroll_frame = Frame(hitl_settings_window)
    hitl_settings_scroll_frame.pack(fill=BOTH, expand=1)

    # set canvas
    hitl_settings_canvas = Canvas(hitl_settings_scroll_frame)
    hitl_settings_canvas.pack(side=LEFT, fill=BOTH, expand=1)

    # set scrollbar
    hitl_settings_scrollbar = tk.Scrollbar(hitl_settings_scroll_frame, orient=VERTICAL, command=hitl_settings_canvas.yview)
    hitl_settings_scrollbar.pack(side=RIGHT, fill=Y)

    # enable scroll on mousewheel 
    def hitl_settings_canvas_mousewheel(event):
        if os.name == 'nt':
            hitl_settings_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        else:
            hitl_settings_canvas.yview_scroll(int(-1 * (event.delta / 2)), 'units')

    # configure canvas and bind scroll events
    hitl_settings_canvas.configure(yscrollcommand=hitl_settings_scrollbar.set)
    hitl_settings_canvas.bind('<Configure>', lambda e: hitl_settings_canvas.configure(scrollregion=hitl_settings_canvas.bbox("all")))
    hitl_settings_canvas.bind_all("<MouseWheel>", hitl_settings_canvas_mousewheel)
    hitl_settings_canvas.bind_all("<Button-4>", hitl_settings_canvas_mousewheel) 
    hitl_settings_canvas.bind_all("<Button-5>", hitl_settings_canvas_mousewheel)

    # set labelframe to fill with widgets
    hitl_settings_main_frame = LabelFrame(hitl_settings_canvas)

    # img selection frame
    hitl_img_selection_frame = LabelFrame(hitl_settings_main_frame, text=[" Image selection criteria ", " Criterios de selección de imágenes ", "Critères de sélection d'images "][lang_idx],
                                            pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg=green_primary, labelanchor = 'n')
    hitl_img_selection_frame.configure(font=(text_font, 15, "bold"))
    hitl_img_selection_frame.grid(column=0, row=1, columnspan=2, sticky='ew')
    hitl_img_selection_frame.columnconfigure(0, weight=1, minsize=50 * scale_factor)
    hitl_img_selection_frame.columnconfigure(1, weight=1, minsize=200 * scale_factor)
    hitl_img_selection_frame.columnconfigure(2, weight=1, minsize=200 * scale_factor)
    hitl_img_selection_frame.columnconfigure(3, weight=1, minsize=200 * scale_factor)
    hitl_img_selection_frame.columnconfigure(4, weight=1, minsize=200 * scale_factor)

    # show explanation and resize window
    def show_text_hitl_img_selection_explanation():
        text_hitl_img_selection_explanation.grid(column=0, row=0, columnspan=5, padx=5, pady=5, sticky='ew')
        hitl_settings_window.update()
        w = hitl_settings_main_frame.winfo_width() + 30
        h = hitl_settings_main_frame.winfo_height() + 10
        hitl_settings_window.geometry(f'{w}x{h}')
        hitl_settings_window.update()

    # img explanation
    Button(master=hitl_img_selection_frame, text="?", width=1, command=show_text_hitl_img_selection_explanation).grid(column=0, row=0, columnspan=1, padx=5, pady=5, sticky='ew')
    text_hitl_img_selection_explanation = Text(master=hitl_img_selection_frame, wrap=WORD, width=1, height=12 * explanation_text_box_height_factor) 
    text_hitl_img_selection_explanation.tag_config('explanation', font=f'{text_font} {int(13 * text_size_adjustment_factor)} normal', lmargin1=10, lmargin2=10)
    text_hitl_img_selection_explanation.insert(END, ["Here, you can specify which images you wish to review. If a detection aligns with the chosen criteria, the image will be "
                                                    "chosen for the verification process. In the review process, you’ll need to make sure all detections in the image are correct. "
                                                    "You have the option to select a subset of your images based on specific classes, confidence ranges, and selection methods. For "
                                                    "instance, the default settings will enable you to verify images with detections that the model is medium-sure about (with"
                                                    " confidences between 0.2 and 0.8). This means that you don’t review high-confidence detections of more than 0.8 confidence and "
                                                    "avoid wasting time on low-confidence detections of less than 0.2. Feel free to adjust these settings to suit your data. To "
                                                    "determine the number of images that will require verification based on the selected criteria, press the “Update counts” button "
                                                    "below. If required, you can specify a selection method that will randomly choose a subset based on a percentage or an absolute "
                                                    "number. Verification will adjust the results in the JSON file. This means that you can continue to use AddaxAI with verified "
                                                    "results and post-process as usual.", "Aquí puede especificar qué imágenes desea revisar. Si una detección se alinea con los "
                                                    "criterios elegidos, la imagen será elegida para el proceso de verificación. Tiene la opción de seleccionar un subconjunto de "
                                                    "sus imágenes según clases específicas, rangos de confianza y métodos de selección. Por ejemplo, la configuración"
                                                    " predeterminada le permitirá verificar imágenes con detecciones de las que el modelo está medio seguro "
                                                    "(con confianzas entre 0,2 y 0,8). Esto significa que no revisa las detecciones de alta confianza con "
                                                    "más de 0,8 de confianza y evita perder tiempo en detecciones de baja confianza de menos de 0,2. Siéntase"
                                                    " libre de ajustar estas configuraciones para adaptarlas a sus datos. Para determinar la cantidad de imágenes "
                                                    "que requerirán verificación según los criterios seleccionados, presione el botón 'Actualizar recuentos' a continuación. Si es "
                                                    "necesario, puede especificar un método de selección que elegirá aleatoriamente un subconjunto en función de un porcentaje o un "
                                                    "número absoluto. La verificación ajustará los resultados en el archivo JSON. Esto significa que puede continuar usando AddaxAI"
                                                    " con resultados verificados y realizar el posprocesamiento como de costumbre.",
                                                    "Ici, vous pouvez spécifier les images que vous souhaitez examiner. Si une détection correspond aux critères choisis, l'image " 
                                                    "sera retenue pour le processus de vérification. Lors de la révision, vous devrez vous assurer que toutes les détections dans "
                                                    "l’image sont correctes. Vous avez la possibilité de sélectionner un sous-ensemble de vos images en fonction de classes "
                                                    "spécifiques, de plages de confiance et de méthodes de sélection. Par exemple, les paramètres par défaut vous permettront "
                                                    "de vérifier les images avec des détections dont le modèle est moyennement certain (niveau de confiance entre 0.2 and 0.8). "
                                                    "Cela signifie que vous n'aurez pas à réviser les détection avec un niveau de confiance de plus de 0.8 ou de moins de 0.2)."
                                                    "Vous pouvez ajuster ces réglages en fonction de vos données. Pour déterminer le nombre d'images qui requerront une "
                                                    "vérification basé sur les critères choisis, cliquer sur le bouton “M-à-j. des comptes” ci-dessous. Au besoin, vous pouvez spéficier "
                                                    "une méthode de sélection aléatoire qui choisira un sous-ensemble basé sur un pourcentage ou une valeur absolue. Le résultat "
                                                    "sera ajusté dans le fichier JSON de sortie. Vous pourrez par la suite poursuivre l'utilisation d'AddaxAI avec les résultats "
                                                    "vérifiés et effectuer le post-traitement comme à l'habitude."][lang_idx])
    text_hitl_img_selection_explanation.tag_add('explanation', '1.0', '1.end')

    # img table headers
    ttk.Label(master=hitl_img_selection_frame, text="").grid(column=0, row=1)
    ttk.Label(master=hitl_img_selection_frame, text=["Class", "Clases", "Classes"][lang_idx], font=f'{text_font} 13 bold').grid(column=1, row=1)
    ttk.Label(master=hitl_img_selection_frame, text=["Confidence range", "Rango de confianza", "Plage de confiance"][lang_idx], font=f'{text_font} 13 bold').grid(column=2, row=1)
    ttk.Label(master=hitl_img_selection_frame, text=["Selection method", "Método de selección", "Méthode de sélection"][lang_idx], font=f'{text_font} 13 bold').grid(column=3, row=1)
    ttk.Label(master=hitl_img_selection_frame, text=["Number of images", "Número de imagenes", "Nombre d'images"][lang_idx], font=f'{text_font} 13 bold').grid(column=4, row=1)

    # ann selection frame
    hitl_ann_selection_frame = LabelFrame(hitl_settings_main_frame, text=[" Annotation selection criteria ", " Criterios de selección de anotaciones ", " Critères de sélection d'annotations "][lang_idx],
                                            pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg=green_primary, labelanchor = 'n')
    hitl_ann_selection_frame.configure(font=(text_font, 15, "bold"))
    hitl_ann_selection_frame.grid(column=0, row=2, columnspan=2, sticky='ew')
    hitl_ann_selection_frame.columnconfigure(0, weight=1, minsize=50)
    hitl_ann_selection_frame.columnconfigure(1, weight=1, minsize=200)
    hitl_ann_selection_frame.columnconfigure(2, weight=1, minsize=200)
    hitl_ann_selection_frame.columnconfigure(3, weight=1, minsize=200)
    hitl_ann_selection_frame.columnconfigure(4, weight=1, minsize=200)

    # ann explanation
    text_hitl_ann_selection_explanation = Text(master=hitl_ann_selection_frame, wrap=WORD, width=1, height=5 * explanation_text_box_height_factor) 
    text_hitl_ann_selection_explanation.grid(column=0, row=0, columnspan=5, padx=5, pady=5, sticky='ew')
    text_hitl_ann_selection_explanation.tag_config('explanation', font=f'{text_font} {int(13 * text_size_adjustment_factor)} normal', lmargin1=10, lmargin2=10)
    text_hitl_ann_selection_explanation.insert(END, ["In the previous step, you selected which images to verify. In this frame, you can specify which annotations to display "
                                              "on these images. During the verification process, all instances of all classes need to be labeled. That is why you want to display "
                                              "all annotations above a reasonable confidence threshold. You can select generic or class-specific confidence thresholds. If you are"
                                              " uncertain, just stick with the default value. A threshold of 0.2 is probably a conservative threshold for most projects.",
                                              "En el paso anterior, seleccionó qué imágenes verificar. En este marco, puede especificar qué anotaciones mostrar en estas imágenes."
                                              " Durante el proceso de verificación, se deben etiquetar todas las instancias de todas las clases. Es por eso que desea mostrar todas"
                                              " las anotaciones por encima de un umbral de confianza razonable. Puede seleccionar umbrales de confianza genéricos o específicos de"
                                              " clase. Si no está seguro, siga con el valor predeterminado. Un umbral de 0,2 es un umbral conservador para la mayoría"
                                              " de los proyectos.",
                                              "À l’étape précédente, vous avez sélectionné les images à vérifier. Ici, vous pouvez spécifier quelles annotations afficher sur ces "
                                              "images. Pendant le processus de vérifications, toutes les instances de toutes les classes doivent être identifiées. Pour cette "
                                              "raison, vous voudrez afficher toutes les annotations au-dessus d'un seuil de confiance raisonnable. Vous pouvez sélectionner un "
                                              "seuil générique ou un seuil basé sur la classe. En cas de doute, conserver la valeur par défaut. Un seuil de 0.2 est probablement "
                                              "conservateur pour la majorité des projets."][lang_idx])
    text_hitl_ann_selection_explanation.tag_add('explanation', '1.0', '1.end')

    # ann same thresh
    rad_ann_var = IntVar()
    rad_ann_var.set(1)
    rad_ann_same = Radiobutton(hitl_ann_selection_frame, text=["Same annotation confidence threshold for all classes",
                                                               "Mismo umbral de confianza para todas las clases",
                                                               "Même seuil d'annotation pour toutes les classes"][lang_idx],
                                variable=rad_ann_var, value=1, command=lambda: toggle_hitl_ann_selection(rad_ann_var, hitl_ann_selection_frame))
    rad_ann_same.grid(row=1, column=1, columnspan=2, sticky='w')
    frame_ann_same = LabelFrame(hitl_ann_selection_frame, text="", pady=2, padx=5, relief=RAISED)
    frame_ann_same.grid(column=3, row=1, columnspan=2, sticky='ew')
    frame_ann_same.columnconfigure(0, weight=1, minsize=200)
    frame_ann_same.columnconfigure(1, weight=1, minsize=200)
    lbl_ann_same = ttk.Label(master=frame_ann_same, text=["All classes", "Todas las clases", "Toutes les classes"][lang_idx])
    lbl_ann_same.grid(row=0, column=0, sticky='w')
    scl_ann_var_generic = DoubleVar()
    scl_ann_var_generic.set(0.60)
    scl_ann = Scale(frame_ann_same, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, variable=scl_ann_var_generic, width=10, length=1, showvalue=0)
    scl_ann.grid(row=0, column=1, sticky='we')
    dsp_scl_ann = Label(frame_ann_same, textvariable=scl_ann_var_generic)
    dsp_scl_ann.grid(row=0, column=0, sticky='e', padx=5)

    # ann specific thresh
    rad_ann_gene = Radiobutton(hitl_ann_selection_frame, text=["Class-specific annotation confidence thresholds",
                                                               "Umbrales de confianza específicas de clase",
                                                               "Seuils de confiance pour chacune des classes"][lang_idx],
                                variable=rad_ann_var, value=2, command=lambda: toggle_hitl_ann_selection(rad_ann_var, hitl_ann_selection_frame))
    rad_ann_gene.grid(row=2, column=1, columnspan=2, sticky='w')

    # create widgets and vars for each class
    label_map = fetch_label_map_from_json(recognition_file)
    selection_dict = {}
    for i, [k, v] in enumerate(label_map.items()):
        
        # image selection frame
        row = i + 2
        frame = LabelFrame(hitl_img_selection_frame, text="", pady=2, padx=5, relief=RAISED)
        frame.grid(column=0, row=1, columnspan=2, sticky='ew')
        frame.columnconfigure(0, weight=1, minsize=50)
        frame.columnconfigure(1, weight=1, minsize=200)
        frame.columnconfigure(2, weight=1, minsize=200)
        frame.columnconfigure(3, weight=1, minsize=200)
        frame.columnconfigure(4, weight=1, minsize=200)
        chb_var = BooleanVar()
        chb_var.set(False)
        chb = tk.Checkbutton(frame, variable=chb_var, command=lambda e=row:enable_selection_widgets(e))
        lbl_class = ttk.Label(master=frame, text=v, state=DISABLED)
        min_conf = DoubleVar(value = 0.2)
        max_conf = DoubleVar(value = 1.0)
        fig = plt.figure(figsize = (2, 0.3))
        plt.hist(confs[k], bins = 10, range = (0,1), color=green_primary, rwidth=0.8)
        plt.xticks([])
        plt.yticks([])
        dist_graph = FigureCanvasTkAgg(fig, frame)
        plt.close()
        rsl = RangeSliderH(frame, [min_conf, max_conf], padX=11, digit_precision='.2f', bgColor = '#ececec', Width = 180, font_size = 10, font_family = text_font)
        rad_var = IntVar()
        rad_var.set(1)
        rad_all = Radiobutton(frame, text=["All images in range", "Todo dentro del rango", "Toutes les images de la plage"][lang_idx],
                                variable=rad_var, value=1, state=DISABLED, command=lambda e=row:enable_amt_per_ent(e))
        rad_per = Radiobutton(frame, text=["Subset percentage", "Subconjunto %", "Pourcentage du sous-ensemble"][lang_idx],
                                variable=rad_var, value=2, state=DISABLED, command=lambda e=row:enable_amt_per_ent(e))
        rad_amt = Radiobutton(frame, text=["Subset number", "Subconjunto no.", "No. du sous-ensemble"][lang_idx],
                                variable=rad_var, value=3, state=DISABLED, command=lambda e=row:enable_amt_per_ent(e))
        ent_per_var = StringVar()
        ent_per = tk.Entry(frame, textvariable=ent_per_var, width=4, state=DISABLED)
        ent_amt_var = StringVar()
        ent_amt = tk.Entry(frame, textvariable=ent_amt_var, width=4, state=DISABLED)
        lbl_n_img = ttk.Label(master=frame, text="0", state=DISABLED)

        # annotation selection frame
        frame_ann = LabelFrame(hitl_ann_selection_frame, text="", pady=2, padx=5, relief=SUNKEN)
        frame_ann.grid(column=3, row=row, columnspan=2, sticky='ew')
        frame_ann.columnconfigure(0, weight=1, minsize=200)
        frame_ann.columnconfigure(1, weight=1, minsize=200)
        lbl_ann_gene = ttk.Label(master=frame_ann, text=v, state = DISABLED)
        lbl_ann_gene.grid(row=0, column=0, sticky='w')
        scl_ann_var_specific = DoubleVar()
        scl_ann_var_specific.set(0.60)
        scl_ann_gene = Scale(frame_ann, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, variable=scl_ann_var_specific, width=10, length=1, showvalue=0, state = DISABLED)
        scl_ann_gene.grid(row=0, column=1, sticky='we')
        dsp_scl_ann_gene = Label(frame_ann, textvariable=scl_ann_var_specific, state = DISABLED)
        dsp_scl_ann_gene.grid(row=0, column=0, sticky='e', padx=5)
        
        # store info in a dictionary 
        item = {'row': row,
                'label_map_id': k,
                'class': v,
                'frame': frame,
                'min_conf_var': min_conf,
                'max_conf_var': max_conf,
                'chb_var': chb_var,
                'lbl_class': lbl_class,
                'range_slider_widget': rsl,
                'lbl_n_img': lbl_n_img,
                'rad_all': rad_all,
                'rad_per': rad_per,
                'rad_amt': rad_amt,
                'rad_var': rad_var,
                'ent_per_var': ent_per_var,
                'ent_per': ent_per,
                'ent_amt_var': ent_amt_var,
                'ent_amt': ent_amt,
                'scl_ann_var_specific': scl_ann_var_specific,
                'scl_ann_var_generic': scl_ann_var_generic}
        selection_dict[row] = item

        # place widgets
        frame.grid(row = row, column = 0, columnspan = 5)
        chb.grid(row = 1, column = 0)
        lbl_class.grid(row = 1, column = 1)
        rsl.lower()
        dist_graph.get_tk_widget().grid(row = 0, rowspan= 3, column = 2, sticky = 'n')
        rad_all.grid(row=0, column=3, sticky='w')
        rad_per.grid(row=1, column=3, sticky='w')
        ent_per.grid(row=1, column=3, sticky='e')
        rad_amt.grid(row=2, column=3, sticky='w')
        ent_amt.grid(row=2, column=3, sticky='e')
        lbl_n_img.grid(row = 1, column = 4)

        # set row minsize
        set_minsize_rows(frame)

        # update window
        hitl_settings_window.update_idletasks()

        # place in front
        bring_window_to_top_but_not_for_ever(hitl_settings_window)

    # set minsize for rows
    row_count = hitl_img_selection_frame.grid_size()[1]
    for row in range(row_count):
        hitl_img_selection_frame.grid_rowconfigure(row, minsize=minsize_rows)

    # add row with total number of images to review
    total_imgs_frame = LabelFrame(hitl_img_selection_frame, text="", pady=2, padx=5, relief=RAISED)
    total_imgs_frame.columnconfigure(0, weight=1, minsize=50)
    total_imgs_frame.columnconfigure(1, weight=1, minsize=200)
    total_imgs_frame.columnconfigure(2, weight=1, minsize=200)
    total_imgs_frame.columnconfigure(3, weight=1, minsize=200)
    total_imgs_frame.columnconfigure(4, weight=1, minsize=200)
    total_imgs_frame.grid(row = row_count, column = 0, columnspan = 5)
    lbl_n_total_imgs = ttk.Label(master=total_imgs_frame, text="TOTAL: 0", state=NORMAL)
    lbl_n_total_imgs.grid(row = 1, column = 4)

    # button frame
    hitl_test_frame = LabelFrame(hitl_settings_main_frame, text=[" Actions ", " Acciones ", "Actions"][lang_idx],
                                    pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg=green_primary, labelanchor = 'n')
    hitl_test_frame.configure(font=(text_font, 15, "bold"))
    hitl_test_frame.grid(column=0, row=3, columnspan=2, sticky='ew')
    hitl_test_frame.columnconfigure(0, weight=1, minsize=115)
    hitl_test_frame.columnconfigure(1, weight=1, minsize=115)
    hitl_test_frame.columnconfigure(2, weight=1, minsize=115)

    # shorten texts for linux
    if sys.platform == "linux" or sys.platform == "linux2":
        btn_hitl_update_txt = ["Update counts", "La actualización cuenta", "M-à-j. des comptes"][lang_idx]
        btn_hitl_show_txt = ["Show / hide annotation", "Mostrar / ocultar anotaciones", "Afficher / cacher annotations"][lang_idx]
        btn_hitl_start_txt = ["Start review process", "Iniciar proceso de revisión", "Démarrer la révision"][lang_idx]
    else:
        btn_hitl_update_txt = ["Update counts", "La actualización cuenta", "Mise-à-jour des comptes"][lang_idx]
        btn_hitl_show_txt = ["Show / hide annotation selection criteria", "Mostrar / ocultar criterios de anotaciones", "Afficher / cacher les critères d'annotations"][lang_idx]
        btn_hitl_start_txt = ["Start review process with selected criteria", "Iniciar proceso de revisión", "Démarrer le processus de révision"][lang_idx]

    # buttons
    btn_hitl_update = Button(master=hitl_test_frame, text=btn_hitl_update_txt, width=1, command=lambda: select_detections(selection_dict = selection_dict, prepare_files = False))
    btn_hitl_update.grid(row=0, column=0, rowspan=1, sticky='nesw', padx=5)
    btn_hitl_show = Button(master=hitl_test_frame, text=btn_hitl_show_txt, width=1, command = toggle_hitl_ann_selection_frame)
    btn_hitl_show.grid(row=0, column=1, rowspan=1, sticky='nesw', padx=5)
    btn_hitl_start = Button(master=hitl_test_frame, text=btn_hitl_start_txt, width=1, command=lambda: select_detections(selection_dict = selection_dict, prepare_files = True))
    btn_hitl_start.grid(row=0, column=2, rowspan=1, sticky='nesw', padx=5)
    
    # radio options for ordering
    lbl_hitl_file_order_txt = ["\n   During validation, how would you like the files to be sorted?", "\n   Durante la validación, ¿cómo desea que se ordenen los archivos?",
                               "\n   Lors de la validation, comment souhaitez-vous que les fichiers soient triés ?"]
    row_hitl_file_order = 1
    lbl_hitl_file_order = Label(hitl_test_frame, text="     " + lbl_hitl_file_order_txt[lang_idx], pady=2, width=1, anchor="w")
    lbl_hitl_file_order.grid(row=row_hitl_file_order, columnspan=3, sticky='nesw')
    var_hitl_file_order = IntVar()
    var_hitl_file_order.set(global_vars.get("var_hitl_file_order", 2))
    rad_hitl_file_order_class = Radiobutton(hitl_test_frame, text=["Group by class: first all images of class A, then B, etc.", "Agrupar por clases: primero todas las imágenes de la especie A, luego B, etc.",
                                                                   "Regrouper par classe: d'abord toutes les images de la classe A, puis B, etc."][lang_idx], variable=var_hitl_file_order, value=2)
    rad_hitl_file_order_class.grid(row=row_hitl_file_order+2, columnspan=3, sticky='nsw', padx=25)
    rad_hitl_file_order_alpha = Radiobutton(hitl_test_frame, text=["Alphabetical by file name: keeps sequences and locations together.", "Alfabético por nombre de archivo: mantiene juntas las secuencias o ubicaciones.",
                                                                   "Alphabétique par nom de fichier: conserve les séquences et les emplacements ensemble."][lang_idx], variable=var_hitl_file_order, value=1)
    rad_hitl_file_order_alpha.grid(row=row_hitl_file_order+1, columnspan=3, sticky='nsw', padx=25)
    def trace_callback(*args): # no idea why this is needed, but if not, the value is not saved
        write_global_vars({"var_hitl_file_order": var_hitl_file_order.get()})
    var_hitl_file_order.trace_add("write", trace_callback)
    
    # create scrollable canvas window
    hitl_settings_canvas.create_window((0, 0), window=hitl_settings_main_frame, anchor="nw")

    # hide annotation selection frame
    toggle_hitl_ann_selection_frame(cmd = "hide")

    # update counts after the window is created
    select_detections(selection_dict = selection_dict, prepare_files = False)

    # adjust window size to widgets
    w = hitl_settings_main_frame.winfo_width() + 30
    h = hitl_settings_main_frame.winfo_height() + 10
    hitl_settings_window.geometry(f'{w}x{h}')

# helper function to quickly check the verification status of xml
def verification_status(xml):
    tree = ET.parse(xml)
    root = tree.getroot()
    try:
        verification_status = True if root.attrib['verified'] == 'yes' else False
    except:
        verification_status = False
    return verification_status

# helper function to blur person bbox
def blur_box(image, bbox_left, bbox_top, bbox_right, bbox_bottom, image_width, image_height):
    x1, y1, x2, y2 = map(int, [bbox_left, bbox_top, bbox_right, bbox_bottom])
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
        raise ValueError(f"Invalid bounding box: ({x1}, {y1}, {x2}, {y2})")
    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        raise ValueError("Extracted ROI is empty. Check the bounding box coordinates.")
    blurred_roi = cv2.GaussianBlur(roi, (71, 71), 0)
    image[y1:y2, x1:x2] = blurred_roi
    return image

# helper function to correctly indent pascal voc annotation files
def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

# make sure the program quits when simple or advanced window is closed
def on_toplevel_close():
    write_global_vars({
        "lang_idx": lang_idx,
        "var_cls_model_idx": dpd_options_cls_model[lang_idx].index(var_cls_model.get()),
        "var_sppnet_location_idx": dpd_options_sppnet_location[lang_idx].index(var_sppnet_location.get())
        })
    root.destroy()

# check if image is corrupted by attempting to load them 
def is_image_corrupted(image_path):
    try:
        ImageFile.LOAD_TRUNCATED_IMAGES = False
        with Image.open(image_path) as img:
            img.load()
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        return False
    except:
        return True 

# read file list and check all images if they are corrupted
def check_images(image_list_file):
    corrupted_images = []
    with open(image_list_file, 'r') as file:
        image_paths = file.read().splitlines()
    for image_path in image_paths:
        if os.path.exists(image_path):
            if is_image_corrupted(image_path):
                corrupted_images.append(image_path)
    return corrupted_images

# try to fix truncated file by opening and saving it again
def fix_images(image_paths):
    for image_path in image_paths:
        if os.path.exists(image_path):
            try:
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                with Image.open(image_path) as img:
                    img_copy = img.copy()
                    img_copy.save(image_path, format=img.format, exif=img.info.get('exif'))
            except Exception as e:
                print(f"Could not fix image: {e}")

# remove non ansi characters from text
def remove_ansi_escape_sequences(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

# classes to open window with speciesnet output
class SpeciesNetOutputWindow:
    def __init__(self):
        self.sppnet_output_window_root = customtkinter.CTkToplevel(root)
        self.sppnet_output_window_root.title("SpeciesNet output")
        self.text_area = tk.Text(self.sppnet_output_window_root, wrap=tk.WORD, height=7, width=85)
        self.text_area.pack(padx=10, pady=10)
        self.close_button = tk.Button(self.sppnet_output_window_root, text="Cancel", command=self.cancel)
        self.close_button.pack(pady=5)
        self.sppnet_output_window_root.protocol("WM_DELETE_WINDOW", self.close)  # Handle window close
        bring_window_to_top_but_not_for_ever(self.sppnet_output_window_root)
    
    def add_string(self, text, process=None):
        if process is not None:
            self.process = process
        if text.strip():
            print(text)
            
            clean_text = remove_ansi_escape_sequences(text)

            # Check if this is a progress bar update
            is_pbar = "%" in clean_text

            if not is_pbar:
                # Insert non-progress-bar messages above the progress section
                self.text_area.insert(tk.END, clean_text + "\n")  # Insert at the top
                self.text_area.see(tk.END)
                self.sppnet_output_window_root.update()
                return  # Exit function early, don't process as a progress bar

            # Ensure attributes exist before updating
            if not hasattr(self, "detector_preprocess_line"):
                self.detector_preprocess_line =   " Detector preprocess:   0%\n"
            if not hasattr(self, "detector_predict_line"):
                self.detector_predict_line =      " Detector predict:      0%\n"
            if not hasattr(self, "classifier_preprocess_line"):
                self.classifier_preprocess_line = " Classifier preprocess: 0%\n"
            if not hasattr(self, "classifier_predict_line"):
                self.classifier_predict_line =    " Classifier predict:    0%\n"
            if not hasattr(self, "geolocation_line"):
                self.geolocation_line =           " Geolocation:           0%\n"

            # Update progress bar lines based on prefixes
            if clean_text.startswith("Detector preprocess"):
                self.detector_preprocess_line = clean_text
            elif clean_text.startswith("Detector predict"):
                self.detector_predict_line = clean_text
            elif clean_text.startswith("Classifier preprocess"):
                self.classifier_preprocess_line = clean_text
            elif clean_text.startswith("Classifier predict"):
                self.classifier_predict_line = clean_text
            elif clean_text.startswith("Geolocation"):
                self.geolocation_line = clean_text

            # Insert all progress bars together to maintain order
            self.text_area.insert(tk.END, f"\n {self.detector_preprocess_line}", "progress")
            self.text_area.insert(tk.END, f" {self.detector_predict_line}", "progress")
            self.text_area.insert(tk.END, f" {self.classifier_preprocess_line}", "progress")
            self.text_area.insert(tk.END, f" {self.classifier_predict_line}", "progress")
            self.text_area.insert(tk.END, f" {self.geolocation_line}", "progress")

            # Ensure scrolling to the latest update
            self.text_area.see(tk.END)
            self.sppnet_output_window_root.update()
    
    def close(self):
        self.sppnet_output_window_root.destroy()
        
    def cancel(self):
        global cancel_speciesnet_deploy_pressed
        global btn_start_deploy
        global sim_run_btn
        if os.name == 'nt':
            Popen(f"TASKKILL /F /PID {self.process.pid} /T")
        else:
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        btn_start_deploy.configure(state=NORMAL)
        sim_run_btn.configure(state=NORMAL)
        cancel_speciesnet_deploy_pressed = True
        self.sppnet_output_window_root.destroy()

# temporary function to deploy speciesnet
def deploy_speciesnet(chosen_folder, sppnet_output_window, simple_mode = False):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
    
    # prepare variables
    chosen_folder = str(Path(chosen_folder))
    python_executable = get_python_interprator("speciesnet")
    sppnet_output_file = os.path.join(chosen_folder, "sppnet_output_file.json")

    # save settings for next time
    write_global_vars({
        "lang_idx": lang_idx,
        "var_cls_model_idx": dpd_options_cls_model[lang_idx].index(var_cls_model.get()),
        "var_sppnet_location_idx": dpd_options_sppnet_location[lang_idx].index(var_sppnet_location.get())
    })
    
    # save advanced settings for next time
    if not simple_mode:
        write_global_vars({
            "var_det_model_idx": dpd_options_model[lang_idx].index(var_det_model.get()),
            "var_det_model_path": var_det_model_path.get(),
            "var_det_model_short": var_det_model_short.get(),
            "var_exclude_subs": var_exclude_subs.get(),
            "var_use_custom_img_size_for_deploy": var_use_custom_img_size_for_deploy.get(),
            "var_image_size_for_deploy": var_image_size_for_deploy.get() if var_image_size_for_deploy.get().isdigit() else "",
            "var_abs_paths": var_abs_paths.get(),
            "var_disable_GPU": var_disable_GPU.get(),
            "var_process_img": var_process_img.get(),
            "var_use_checkpnts": var_use_checkpnts.get(),
            "var_checkpoint_freq": var_checkpoint_freq.get() if var_checkpoint_freq.get().isdecimal() else "",
            "var_cont_checkpnt": var_cont_checkpnt.get(),
            "var_process_vid": var_process_vid.get(),
            "var_not_all_frames": var_not_all_frames.get(),
            "var_nth_frame": var_nth_frame.get() if var_nth_frame.get().isdecimal() else ""
        })
    
    # get param values
    model_vars = load_model_vars()
    if simple_mode:
        cls_detec_thresh = model_vars["var_cls_detec_thresh_default"]
    else:
        cls_detec_thresh = var_cls_detec_thresh.get()

    # get location information
    location_args = []
    country_code = var_sppnet_location.get()[:3]
    if country_code != "NON": # only add location args if not selected 'NONE' option
        location_args.append(f"--country={country_code}")
        if country_code == "USA":
            state_code = var_sppnet_location.get()[4:6]
            location_args.append(f"--admin1_region={state_code}")
    write_global_vars({
        "var_sppnet_location_idx": dpd_options_sppnet_location[lang_idx].index(var_sppnet_location.get())
    })

    # create commands for Windows
    if os.name == 'nt':
        if location_args == []:
            command = [python_executable, "-m", "speciesnet.scripts.run_model", f"--folders={chosen_folder}", f"--predictions_json={sppnet_output_file}"]
        else:
            command = [python_executable, "-m", "speciesnet.scripts.run_model", f"--folders={chosen_folder}", f"--predictions_json={sppnet_output_file}", *location_args]

     # create command for MacOS and Linux
    else:
        if location_args == []:
            command = [f"'{python_executable}' -m speciesnet.scripts.run_model --folders='{chosen_folder}' --predictions_json='{sppnet_output_file}'"]
        else:
            location_args = "' '".join(location_args)
            command = [f"'{python_executable}' -m speciesnet.scripts.run_model --folders='{chosen_folder}' --predictions_json='{sppnet_output_file}' '{location_args}'"]
    
    # log command
    print("command:")
    print(json.dumps(command, indent=4))

    # prepare process and cancel method per OS
    if os.name == 'nt':
        # run windows command
        p = Popen(command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                shell=True,
                universal_newlines=True)

    else:
        # run unix command
        p = Popen(command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                shell=True,
                universal_newlines=True,
                preexec_fn=os.setsid)

    global cancel_speciesnet_deploy_pressed
    cancel_speciesnet_deploy_pressed = False

    # read output
    for line in p.stdout:
        
        # log
        sppnet_output_window.add_string(line, p)
        
        # early exit if cancel button is pressed
        if cancel_speciesnet_deploy_pressed:
            sppnet_output_window.add_string("\n\nCancel button pressed!")
            time.sleep(2)
            return
        
        # temporary fix for macOS package conflict
        # since the env is compiled on macOS 10.15, scipy is not compatible with macOS 10.14
        if line.startswith("ImportError: "):
            sppnet_output_window.add_string(f"\n\nThere seems to be a mismatch between macOS versions: {line}\n\n")
            sppnet_output_window.add_string("Attempting to solve conflict automatically...\n\n")
            
            # uninstall scipy
            p = Popen(f"{python_executable} -m pip uninstall -y scipy",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                shell=True,
                universal_newlines=True,
                preexec_fn=os.setsid)
            for line in p.stdout:
                sppnet_output_window.add_string(line)
            
            # install scipy again
            p = Popen(f"{python_executable} -m pip install --no-cache-dir scipy",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                shell=True,
                universal_newlines=True,
                preexec_fn=os.setsid)
            for line in p.stdout:
                sppnet_output_window.add_string(line)
            
            # retry
            return "restart"
    
    # convert json to AddaxAI format
    sppnet_output_window.add_string("\n\nConverting SpeciesNet output to AddaxAI format...")
    speciesnet_to_md_py = os.path.join(AddaxAI_files, "AddaxAI", "classification_utils", "model_types", "speciesnet_to_md.py")
    recognition_file = os.path.join(chosen_folder, "image_recognition_file.json")
    
    # cmd for windows
    if os.name == 'nt': 
        p = Popen([f"{python_executable}", f"{speciesnet_to_md_py}", f"{sppnet_output_file}", f"{recognition_file}"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                shell=True,
                universal_newlines=True)
    
    # cmd for macos and linux
    else: 
        p = Popen([f'"{python_executable}" "{speciesnet_to_md_py}" "{sppnet_output_file}" "{recognition_file}"'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                shell=True,
                universal_newlines=True,
                preexec_fn=os.setsid)
    
    # log output
    for line in p.stdout:
        sppnet_output_window.add_string(line, p)
    sppnet_output_window.add_string("\n\nConverting Done!")
    
    # if that is done, remove the speciesnet output file
    if os.path.exists(sppnet_output_file):
        os.remove(sppnet_output_file)
    
    # create addaxai metadata
    sppnet_output_window.add_string("\n\nAdding AddaxAI metadata...")
    addaxai_metadata = {"addaxai_metadata" : {"version" : current_AA_version,
                                                  "custom_model" : False,
                                                  "custom_model_info" : {}}}
    
    # write metadata to json and make absolute if specified
    append_to_json(recognition_file, addaxai_metadata)
    
    # get rid of absolute paths if specified
    if check_json_paths(recognition_file) == "absolute":
        make_json_relative(recognition_file)
    
    # if in timelapse mode, change name of recognition file
    if timelapse_mode:
        timelapse_json = os.path.join(chosen_folder, "timelapse_recognition_file.json")
        os.rename(recognition_file, timelapse_json)
        mb.showinfo("Analysis done!", f"Recognition file created at \n\n{timelapse_json}\n\nTo use it in Timelapse, return to "
                                        "Timelapse with the relevant image set open, select the menu item 'Recognition > Import "
                                        "recognition data for this image set' and navigate to the file above.")
        open_file_or_folder(os.path.dirname(timelapse_json))
        
    # convert JSON to AddaxAI format if not in timelapse mode
    else:
        with open(recognition_file) as image_recognition_file_content:
            data = json.load(image_recognition_file_content)

            # fetch and invert label maps
            cls_label_map = data['classification_categories']
            det_label_map = data['detection_categories']
            inverted_cls_label_map = {v: k for k, v in cls_label_map.items()}
            inverted_det_label_map = {v: k for k, v in det_label_map.items()}

            # add cls classes to det label map
            # if a model shares category names with MD, add to existing value
            for k, _ in inverted_cls_label_map.items():
                if k in inverted_det_label_map.keys(): 
                    value = str(inverted_det_label_map[k])
                    inverted_det_label_map[k] = value
                else:              
                    inverted_det_label_map[k] = str(len(inverted_det_label_map) + 1)

            # loop and adjust
            for image in data['images']:
                if 'detections' in image and image['detections'] is not None:
                    for detection in image['detections']:
                        category_id = detection['category']
                        category_conf = detection['conf']
                        if category_conf >= cls_detec_thresh and det_label_map[category_id] == "animal":
                            if 'classifications' in detection:
                                highest_classification = detection['classifications'][0]
                                class_idx = highest_classification[0]
                                class_name = cls_label_map[class_idx]
                                detec_idx = inverted_det_label_map[class_name]
                                detection['prev_conf'] = detection["conf"]
                                detection['prev_category'] = detection['category']
                                detection["conf"] = highest_classification[1]
                                detection['category'] = str(detec_idx)

        # write json to be used by AddaxAI
        data['detection_categories_original'] = data['detection_categories']
        data['detection_categories'] = {v: k for k, v in inverted_det_label_map.items()}
        
        # overwrite the file wit adjusted data
        with open(recognition_file, "w") as json_file:
            json.dump(data, json_file, indent=1)

    # reset window
    update_frame_states()
    root.update()

# convert pascal bbox to yolo
def convert_bbox_pascal_to_yolo(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1 
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

# special function because the sim dpd has a different value for 'None'
def sim_mdl_dpd_callback(self):
    
    # this means the user chose SpeciesNet in simple mode, so tell user to use the advanced mode
    if self == "Global - SpeciesNet - Google":
        mb.showerror(["SpeciesNet not available", "SpeciesNet no disponible", "SpeciesNet non-disponible"][lang_idx],
                        message=[f"'Global - SpeciesNet - Google' is not available in simple mode. Please switch to advanced mode to use SpeciesNet.",
                                    f"'Global - SpeciesNet - Google' no está disponible en modo simple. Cambie al modo avanzado para usar SpeciesNet.",
                                    f"'Global - SpeciesNet - Google' n'est pas disponible en mode simple. SVP utilisez le mode avancé pour utiliser SpeciesNet."][lang_idx])
    
    var_cls_model.set(dpd_options_cls_model[lang_idx][sim_dpd_options_cls_model[lang_idx].index(self)])
    model_cls_animal_options(var_cls_model.get())

# return xml path with temp-folder squeezed in
def return_xml_path(img_path):
    head_path = var_choose_folder.get()
    tail_path = os.path.splitext(os.path.relpath(img_path, head_path))
    temp_xml_path = os.path.join(head_path, "temp-folder", tail_path[0] + ".xml")
    return os.path.normpath(temp_xml_path)

# temporary file which labelImg writes to notify AddaxAI that it should convert xml to coco
class LabelImgExchangeDir:
    def __init__(self, dir):
        self.dir = dir
        Path(self.dir).mkdir(parents=True, exist_ok=True)

    def create_file(self, content, idx):
        timestamp_milliseconds = str(str(datetime.date.today()) + str(datetime.datetime.now().strftime('%H%M%S%f'))).replace('-', '')
        temp_file = os.path.normpath(os.path.join(self.dir, f"{timestamp_milliseconds}-{idx}.txt"))
        with open(temp_file, 'w') as f:
            f.write(content)
    
    def read_file(self, fp):
        with open(fp, 'r') as f:
            content = f.read()
            return content

    def delete_file(self, fp):
        if os.path.exists(fp):
            os.remove(fp)

    def exist_file(self):
        filelist = glob.glob(os.path.normpath(os.path.join(self.dir, '*.txt')))
        for fn in sorted(filelist):
            return [True, fn]
        return [False, '']

# delete temp folder
def delete_temp_folder(file_list_txt):
    temp_folder = os.path.dirname(file_list_txt)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)

# browse file and display result
def browse_file(var, var_short, var_path, dsp, filetype, cut_off_length, options, nrow):
    # choose file
    file = filedialog.askopenfilename(filetypes=filetype)
    
    # shorten if needed
    dsp_file = os.path.basename(file)
    if len(dsp_file) > cut_off_length:
        dsp_file = "..." + dsp_file[0 - cut_off_length + 3:]
    
    # set variables
    var_short.set(dsp_file)

    # reset to default if faulty
    if file != "":
        dsp.grid(column=0, row=nrow, sticky='e')
        var_path.set(file)
    else:
        var.set(options[0])

# switches the yolov5 version by modifying the python import path
def switch_yolov5_version(model_type):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({model_type})\n")
    
    # set the path to the desired version
    base_path = os.path.join(AddaxAI_files, "yolov5_versions")
    if model_type == "old models":
        version_path = os.path.join(base_path, "yolov5_old", "yolov5")
    elif model_type == "new models":
        version_path = os.path.join(base_path, "yolov5_new", "yolov5")
    else:
        raise ValueError("Invalid model_type")
        
    # add yolov5 checkout to PATH if not already there
    if version_path not in sys.path:
        sys.path.insert(0, version_path)
    
    # add yolov5 checkout to PYTHONPATH if not already there
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    PYTHONPATH_to_add = version_path + PYTHONPATH_separator
    if not current_pythonpath.startswith(PYTHONPATH_to_add):
        os.environ["PYTHONPATH"] = PYTHONPATH_to_add + current_pythonpath
        
# extract label map from custom model
def extract_label_map_from_model(model_file):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})")

    # import module from cameratraps dir
    from cameratraps.megadetector.detection.pytorch_detector import PTDetector

    # load model
    label_map_detector = PTDetector(model_file, force_cpu = True)
    
    # fetch classes
    try:
        CUSTOM_DETECTOR_LABEL_MAP = {}
        for id in label_map_detector.model.names:
            CUSTOM_DETECTOR_LABEL_MAP[id] = label_map_detector.model.names[id]
    except Exception as error:
        # log error
        print("ERROR:\n" + str(error) + "\n\nDETAILS:\n" + str(traceback.format_exc()) + "\n\n")
        
        # show error
        mb.showerror(title=error_txt[lang_idx],
                     message=["An error has occurred when trying to extract classes", "Se ha producido un error al intentar extraer las clases",
                              "Une erreur est survenue lors de l'extraction des classes"][lang_idx] +
                                " (AddaxAI v" + current_AA_version + "): '" + str(error) + "'" +
                                [".\n\nWill try to proceed and produce the output json file, but post-processing features of AddaxAI will not work.",
                                 ".\n\nIntentará continuar y producir el archivo json de salida, pero las características de post-procesamiento de AddaxAI no funcionarán.",
                                 ".\n\nUne tentative de poursuivre et de générer le fichier de sortie JSON sera effecutée, mais les fonctionnalités de post-traitement d'AddaxAI ne fonctionneront pas."][lang_idx],
                     detail=traceback.format_exc())
    
    # delete and free up memory
    del label_map_detector
    
    # log
    print(f"Label map: {CUSTOM_DETECTOR_LABEL_MAP})\n")

    # return label map
    return CUSTOM_DETECTOR_LABEL_MAP

# fetch label map from json
def fetch_label_map_from_json(path_to_json):
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    label_map = data['detection_categories']
    return label_map

# check if json paths are relative or absolute
def check_json_paths(path_to_json): 
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    path = os.path.normpath(data['images'][0]['file'])
    if path.startswith(os.path.normpath(var_choose_folder.get())):
        return "absolute"
    else:
        return "relative"

# make json paths relative
def make_json_relative(path_to_json):
    if check_json_paths(path_to_json) == "absolute":
        # open
        with open(path_to_json, "r") as json_file:
            data = json.load(json_file)
        
        # adjust
        for image in data['images']:
            absolute_path =  os.path.normpath(image['file'])
            relative_path = absolute_path.replace(os.path.normpath(var_choose_folder.get()), "")[1:]
            image['file'] = relative_path
        
        # write
        with open(path_to_json, "w") as json_file:
            json.dump(data, json_file, indent=1)
            
# make json paths absolute
def make_json_absolute(path_to_json):
    if check_json_paths(path_to_json) == "relative":
        # open
        with open(path_to_json, "r") as json_file:
            data = json.load(json_file)
        
        # adjust
        for image in data['images']:
            relative_path = image['file']
            absolute_path = os.path.normpath(os.path.join(var_choose_folder.get(), relative_path))
            image['file'] = absolute_path
        
        # write
        with open(path_to_json, "w") as json_file:
            json.dump(data, json_file, indent=1)

# add information to json file
def append_to_json(path_to_json, object_to_be_appended):
    # open
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    
    # adjust
    data['info'].update(object_to_be_appended)

    # write
    with open(path_to_json, "w") as json_file:
        json.dump(data, json_file, indent=1)

# change human-in-the-loop prgress variable
def change_hitl_var_in_json(path_to_json, status):
    # open
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    
    # adjust
    data['info']["addaxai_metadata"]["hitl_status"] = status

    # write
    with open(path_to_json, "w") as json_file:
        json.dump(data, json_file, indent=1)

# get human-in-the-loop prgress variable
def get_hitl_var_in_json(path_to_json):
    # open
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
        addaxai_metadata = data['info'].get("addaxai_metadata") or data['info'].get("ecoassist_metadata") # include old name 'EcoAssist' for backwards compatibility
    
    # get status
    if "hitl_status" in addaxai_metadata:
        status = addaxai_metadata["hitl_status"]
    else:
        status = "never-started"

    # return
    return status

# show warning for video post-processing
def check_json_presence_and_warn_user(infinitive, continuous, noun):
    # check json presence
    img_json = False
    if os.path.isfile(os.path.join(var_choose_folder.get(), "image_recognition_file.json")):
        img_json = True
    vid_json = False
    if os.path.isfile(os.path.join(var_choose_folder.get(), "video_recognition_file.json")):
        vid_json = True
    
    # show warning
    if not img_json:
        if vid_json:
            mb.showerror(error_txt[lang_idx], [f"{noun.capitalize()} is not supported for videos.",
                                           f"{noun.capitalize()} no es compatible con vídeos.",
                                           f"{noun.capitalize()} n'est pas supporté avec les vidéos."][lang_idx])
            return True
        if not vid_json:
            mb.showerror(error_txt[lang_idx], [f"No model output file present. Make sure you run step 2 before {continuous} the files. {noun.capitalize()} "
                                            "is only supported for images.",
                                           f"No hay archivos de salida del modelo. Asegúrese de ejecutar el paso 2 antes de {continuous} los archivos. "
                                           f"{noun.capitalize()} sólo es compatible con imágenes",
                                           f"Aucun fichier de sortie du modèle présent. Assurez-vous d'exécuter l'étape 2 avant de {continuous} les fichiers."
                                           f"{noun.capitalize()} est supporté uniquement pour les images.",][lang_idx])
            return True
    if img_json:
        if vid_json:
            mb.showinfo(warning_txt[lang_idx], [f"{noun.capitalize()} is not supported for videos. Will continue to only {infinitive} the images...",
                                            f"No se admiten {noun.capitalize()} para los vídeos. Continuará sólo {infinitive} las imágenes...",
                                            f"{noun.capitalize()} n'est pas supporté pour les vidéos. AddaxAI continuera de {infinitive} les images uniquement..."][lang_idx])

# dir names for when separating on confidence
conf_dirs = {0.0 : "conf_0.0",
             0.1 : "conf_0.0-0.1",
             0.2 : "conf_0.1-0.2",
             0.3 : "conf_0.2-0.3",
             0.4 : "conf_0.3-0.4",
             0.5 : "conf_0.4-0.5",
             0.6 : "conf_0.5-0.6",
             0.7 : "conf_0.6-0.7",
             0.8 : "conf_0.7-0.8",
             0.9 : "conf_0.8-0.9",
             1.0 : "conf_0.9-1.0"}

def _parse_timestamp_from_filename(filename):
    """
    Try common filename timestamp patterns (YYYYMMDDHHMMSS, YYYYMMDD_HHMMSS, etc.)
    Return a datetime or None.
    """
    patterns = [
        r'(?P<ts>\d{14})',                # 20220101123045
        r'(?P<ts>\d{8}[_-]\d{6})',        # 20220101_123045 or 20220101-123045
        r'(?P<ts>\d{4}-\d{2}-\d{2}[_-]\d{6})'
    ]
    for pat in patterns:
        m = re.search(pat, filename)
        if not m:
            continue
        ts = re.sub(r'[_-]', '', m.group('ts'))
        try:
            return datetime.datetime.strptime(ts, '%Y%m%d%H%M%S')
        except Exception:
            continue
    return None

def get_image_timestamp(src_dir, rel_path):
    """
    Return datetime for given file (src_dir + rel_path).
    1) Try EXIF DateTimeOriginal / DateTime
    2) Try filename patterns
    3) Fallback to filesystem mtime
    """
    abs_path = os.path.join(src_dir, rel_path)
    # (1) EXIF
    try:
        img = Image.open(abs_path)
        exif = img.getexif()
        if exif:
            # map tag name -> id
            tag_map = {v:k for k,v in PIL.ExifTags.TAGS.items()}
            for tag_name in ("DateTimeOriginal", "DateTime"):
                if tag_name in tag_map:
                    tid = tag_map[tag_name]
                    if tid in exif:
                        val = exif.get(tid)
                        if isinstance(val, bytes):
                            val = val.decode(errors="ignore")
                        if val:
                            val = val.replace('\x00','').strip()
                            try:
                                return datetime.datetime.strptime(val, "%Y:%m:%d %H:%M:%S")
                            except Exception:
                                try:
                                    return datetime.datetime.strptime(val, "%Y-%m-%d %H:%M:%S")
                                except Exception:
                                    pass
    except Exception:
        # ignore EXIF-read errors and fall back
        pass

    # (2) filename
    try:
        ts = _parse_timestamp_from_filename(os.path.basename(rel_path))
        if ts:
            return ts
    except Exception:
        pass

    # (3) filesystem mtime
    try:
        return datetime.datetime.fromtimestamp(os.path.getmtime(abs_path))
    except Exception:
        return None

def build_image_timestamp_index(src_dir, file_list):
    """
    Build dict {rel_path: datetime or None} for the given files.
    Call this once before the main loop (expensive to open images; we cache).
    """
    idx = {}
    for rel in file_list:
        try:
            idx[rel] = get_image_timestamp(src_dir, rel)
        except Exception:
            idx[rel] = None
    return idx

def _camera_prefix_of_filename(filename):
    """
    Heuristic: return the filename part before the first timestamp pattern.
    If no timestamp found, return None.
    """
    m = re.search(r'\d{14}', filename)
    if m:
        return filename[:m.start()]
    m = re.search(r'\d{8}[_-]\d{6}', filename)
    if m:
        return filename[:m.start()]
    return None

def find_series_images(target_rel_path, timestamp_index, window_seconds=10, require_same_camera=True):
    """
    Return list of rel_paths that belong to same 'burst' as target_rel_path.
    - Uses timestamp_index (dict from build_image_timestamp_index).
    - window_seconds: include files within +/- window_seconds of the target timestamp.
    - require_same_camera: only include files with same filename prefix (heuristic).
    """
    if target_rel_path not in timestamp_index:
        return [target_rel_path]
    t0 = timestamp_index.get(target_rel_path)
    if t0 is None:
        return [target_rel_path]

    prefix0 = _camera_prefix_of_filename(os.path.basename(target_rel_path)) if require_same_camera else None
    candidates = []
    for rel, t in timestamp_index.items():
        if t is None:
            continue
        if require_same_camera:
            prefix = _camera_prefix_of_filename(os.path.basename(rel))
            if prefix0 is not None and prefix is not None and prefix0 != prefix:
                continue
        if abs((t - t0).total_seconds()) <= window_seconds:
            candidates.append((rel, t))
    # sort ascending (optional — we keep chronological)
    candidates.sort(key=lambda x: x[1] or datetime.datetime.max)
    return [c[0] for c in candidates]

# move files into subdirectories
def move_files(file, detection_type, var_file_placement, max_detection_conf, var_sep_conf, dst_root, src_dir, manually_checked):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # squeeze in extra dir if sorting on confidence
    if var_sep_conf and detection_type != "empty":
        global conf_dirs
        if manually_checked:
            confidence_dir = "verified"
        else:
            ceiled_confidence = math.ceil(max_detection_conf * 10) / 10.0
            confidence_dir = conf_dirs[ceiled_confidence]
        new_file = os.path.join(detection_type, confidence_dir, file)
    else:
        new_file = os.path.join(detection_type, file)
    
    # set paths
    src = os.path.join(src_dir, file)
    dst = os.path.join(dst_root, new_file)
    
    # create subfolder
    Path(os.path.dirname(dst)).mkdir(parents=True, exist_ok=True)
    
    # place image or video in subfolder
    if var_file_placement == 1: # move
        shutil.move(src, dst)
    elif var_file_placement == 2: # copy
        shutil.copy2(src, dst)
        
    # return new relative file path
    return(new_file)

# sort multiple checkpoint in order from recent to last
def sort_checkpoint_files(files):
    def get_timestamp(file):
        timestamp_str = file.split('_')[1].split('.')[0]
        return datetime.datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
    sorted_files = sorted(files, key=get_timestamp, reverse=True)
    return sorted_files

# check if checkpoint file is present and assign global variable
def check_checkpnt():
    global loc_chkpnt_file
    loc_chkpnt_files = []
    for filename in os.listdir(var_choose_folder.get()):
        if re.search('^md_checkpoint_\d+\.json$', filename):
            loc_chkpnt_files.append(filename)
    if len(loc_chkpnt_files) == 0:
        mb.showinfo(["No checkpoint file found", "No se ha encontrado ningún archivo de puntos de control", "Aucun point de contrôle trouvé"][lang_idx],
                        ["There is no checkpoint file found. Cannot continue from checkpoint file...",
                        "No se ha encontrado ningún archivo de punto de control. No se puede continuar desde el archivo de punto de control...",
                        "Aucun fichier de point de contrôle trouvé. La poursuite à partir d'un point de contrôle est impossible."][lang_idx])
        return False
    if len(loc_chkpnt_files) == 1:
        loc_chkpnt_file = os.path.join(var_choose_folder.get(), loc_chkpnt_files[0])
    elif len(loc_chkpnt_files) > 1:
        loc_chkpnt_file = os.path.join(var_choose_folder.get(), sort_checkpoint_files(loc_chkpnt_files)[0])
    return True

# cut off string if it is too long
def shorten_path(path, length):
    if len(path) > length:
        path = "..." + path[0 - length + 3:]
    return path

# browse directory
def browse_dir(var, var_short, dsp, cut_off_length, n_row, n_column, str_sticky, source_dir = False):    
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # choose directory
    chosen_dir = filedialog.askdirectory()

    # early exit
    if chosen_dir in ["", "/", "\\", ".", "~", ":"] or not os.path.isdir(chosen_dir):
        return

    # set choice to variable
    var.set(chosen_dir)
    
    # shorten, set and grid display
    dsp_chosen_dir = chosen_dir
    dsp_chosen_dir = shorten_path(dsp_chosen_dir, cut_off_length)
    var_short.set(dsp_chosen_dir)
    dsp.grid(column=n_column, row=n_row, sticky=str_sticky)
    
    # also update simple mode if it regards the source dir
    if source_dir:
        global sim_dir_pth
        sim_dir_pth.configure(text = dsp_chosen_dir, text_color = "black")

# choose a custom classifier for animals
def model_cls_animal_options(self):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # set simple mode cls dropdown to the same index for its own dpd list
    sim_mdl_dpd.set(sim_dpd_options_cls_model[lang_idx][dpd_options_cls_model[lang_idx].index(self)])

    # remove or show widgets
    if self not in none_txt:
        cls_frame.grid(row=cls_frame_row, column=0, columnspan=2, sticky = 'ew')
    else:
        cls_frame.grid_forget()
    
    # get model specific variable values
    global sim_spp_scr
    model_vars = load_model_vars()
    if self not in none_txt and self != "Global - SpeciesNet - Google": # normal procedure for all classifiers other than speciesnet
        
        dsp_choose_classes.configure(text = f"{len(model_vars['selected_classes'])} of {len(model_vars['all_classes'])}")
        var_cls_detec_thresh.set(model_vars["var_cls_detec_thresh"])
        var_cls_class_thresh.set(model_vars["var_cls_class_thresh"])
        var_smooth_cls_animal.set(model_vars["var_smooth_cls_animal"])

        # remove widgets of species net
        lbl_sppnet_location.grid_remove()
        dpd_sppnet_location.grid_remove()

        # make sure detection model selection is shown
        lbl_model.grid(row=row_model, sticky='nesw', pady=2)
        dpd_model.grid(row=row_model, column=1, sticky='nesw', padx=5)
        if var_det_model_short.get() != "":
            dsp_model.grid(column=0, row=row_model, sticky='e')

        # show widgets for other classifiers
        lbl_choose_classes.grid(row=row_choose_classes, sticky='nesw', pady=2)
        btn_choose_classes.grid(row=row_choose_classes, column=1, sticky='nesw', padx=5)
        dsp_choose_classes.grid(row=row_choose_classes, column=0, sticky='e', padx=0)
        lbl_cls_class_thresh.grid(row=row_cls_class_thresh, sticky='nesw', pady=2)
        scl_cls_class_thresh.grid(row=row_cls_class_thresh, column=1, sticky='ew', padx=10)
        dsp_cls_class_thresh.grid(row=row_cls_class_thresh, column=0, sticky='e', padx=0)
        lbl_smooth_cls_animal.grid(row=row_smooth_cls_animal, sticky='nesw', pady=2)
        chb_smooth_cls_animal.grid(row=row_smooth_cls_animal, column=1, sticky='nesw', padx=5)
        
        # set rowsize
        set_minsize_rows(cls_frame)

        # adjust simple_mode window
        sim_spp_lbl.configure(text_color = "black")
        sim_spp_scr.grid_forget()
        sim_spp_scr = SpeciesSelectionFrame(master=sim_spp_frm,
                                            height=sim_spp_scr_height,
                                            all_classes=model_vars['all_classes'],
                                            selected_classes=model_vars['selected_classes'],
                                            command = on_spp_selection)
        sim_spp_scr._scrollbar.configure(height=0)
        sim_spp_scr.grid(row=1, column=0, padx=PADX, pady=(PADY/4, PADY), sticky="ew", columnspan = 2)

    elif self == "Global - SpeciesNet - Google": # special procedure for speciesnet
        
        dsp_choose_classes.configure(text = f"{len(model_vars['selected_classes'])} of {len(model_vars['all_classes'])}")
        var_cls_detec_thresh.set(model_vars["var_cls_detec_thresh"])
        var_cls_class_thresh.set(model_vars["var_cls_class_thresh"])
        var_smooth_cls_animal.set(model_vars["var_smooth_cls_animal"])

        # remove detection model selection
        lbl_model.grid_remove()
        print(f"Removing detection model selection...")
        dpd_model.grid_remove()
        print(f"Removing detection model selection...")
        dsp_model.grid_remove()
        
        
        # remove widgets for other classifiers
        lbl_choose_classes.grid_remove()
        btn_choose_classes.grid_remove()
        dsp_choose_classes.grid_remove()
        lbl_cls_class_thresh.grid_remove()
        scl_cls_class_thresh.grid_remove()
        dsp_cls_class_thresh.grid_remove()
        lbl_smooth_cls_animal.grid_remove()
        chb_smooth_cls_animal.grid_remove()
        
        # set rowsize to 0
        cls_frame.grid_rowconfigure(2, minsize=0)
        cls_frame.grid_rowconfigure(3, minsize=0)
        cls_frame.grid_rowconfigure(4, minsize=0)
        
        # place widgets for speciesnet
        lbl_sppnet_location.grid(row=row_sppnet_location, sticky='nesw', pady=2)
        dpd_sppnet_location.grid(row=row_sppnet_location, column=1, sticky='nesw', padx=5, pady=2)
        
        # set selection frame to dummy spp again
        sim_spp_lbl.configure(text_color = "grey")
        sim_spp_scr.grid_forget()
        sim_spp_scr = SpeciesSelectionFrame(master=sim_spp_frm, height=sim_spp_scr_height, dummy_spp = True)
        sim_spp_scr._scrollbar.configure(height=0)
        sim_spp_scr.grid(row=1, column=0, padx=PADX, pady=(PADY/4, PADY), sticky="ew", columnspan = 2)

    else:
        # set selection frame to dummy spp again
        sim_spp_lbl.configure(text_color = "grey")
        sim_spp_scr.grid_forget()
        sim_spp_scr = SpeciesSelectionFrame(master=sim_spp_frm, height=sim_spp_scr_height, dummy_spp = True)
        sim_spp_scr._scrollbar.configure(height=0)
        sim_spp_scr.grid(row=1, column=0, padx=PADX, pady=(PADY/4, PADY), sticky="ew", columnspan = 2)

    # save settings
    write_global_vars({
        "var_cls_model_idx": dpd_options_cls_model[lang_idx].index(var_cls_model.get()),  # write index instead of value
        "var_sppnet_location_idx": dpd_options_sppnet_location[lang_idx].index(var_sppnet_location.get()),  # write index instead of value
        })
 
    # show/hide taxonomic level widgets
    if taxon_mapping_csv_present():
        lbl_tax_fallback.grid(row=row_tax_fallback, sticky='nesw', pady=2)
        chb_tax_fallback.grid(row=row_tax_fallback, column=1, sticky='nesw', padx=5)
        var_tax_fallback.set(model_vars.get('var_tax_fallback', False))
        toggle_tax_levels_dpd_options()
        toggle_tax_levels()        
    else:
        lbl_tax_fallback.grid_forget()
        chb_tax_fallback.grid_forget()
        lbl_tax_levels.grid_forget()
        dpd_tax_levels.grid_forget()
        cls_frame.grid_rowconfigure(row_tax_levels, minsize=0)
        cls_frame.grid_rowconfigure(row_tax_fallback, minsize=0)
 
    # finish up
    toggle_cls_frame()
    resize_canvas_to_content()

def fetch_taxon_dpd_options():
    # read model vars
    model_vars = load_model_vars("cls")
        
    # read taxon options from csv
    if taxon_mapping_csv_present():
        taxon_mapping_df = fetch_taxon_mapping_df()
        level_cols = [col.replace('level_', '') for col in taxon_mapping_df.columns if col.startswith('level_')]
        only_above_cols = [col.replace('only_above_', '') for col in taxon_mapping_df.columns if col.startswith('only_above_')]
        
        # set the options for the dropdown
        dpd_options_tax_levels_en = []
        dpd_options_tax_levels_es = []
        for level_col in level_cols:
            dpd_options_tax_levels_en.append("Fix predictions at the " + level_col + " level (if available)")
            dpd_options_tax_levels_es.append("Fijar predicciones en el nivel " + level_col + " (si está disponible)")
        for only_above_col in only_above_cols:
            dpd_options_tax_levels_en.append(f"Only predict categories with ≥ {int(only_above_col):,} training samples")
            dpd_options_tax_levels_es.append(f"Predecir sólo categorías con ≥ {int(only_above_col):,} muestras de entrenamiento")
        dpd_options_tax_levels = [["Let the model decide the best prediction level (recommended)"] + dpd_options_tax_levels_en,
                                ["Dejar que el modelo decida el mejor nivel de predicción (recomendado)"] + dpd_options_tax_levels_es]
    else:
        dpd_options_tax_levels = [['dummy'], ['dummy']] # set dummy value to avoid error

    return dpd_options_tax_levels

# function to update the dropdown options for taxonomic levels
def toggle_tax_levels_dpd_options():
    
    # fetch the taxon options
    dpd_options_tax_levels = fetch_taxon_dpd_options()

    # delte the old options
    menu = dpd_tax_levels["menu"]
    menu.delete(0, "end")

    # Add new options
    for option in dpd_options_tax_levels[lang_idx]:
        menu.add_command(
            label=option,
            command=lambda value=option: (
                var_tax_levels.set(value),
                write_model_vars(new_values = {"var_tax_levels_idx": dpd_options_tax_levels[lang_idx].index(value)})
            )
        )

    # set to the previously chosen value
    model_vars = load_model_vars("cls")
    var_tax_levels.set(dpd_options_tax_levels[lang_idx][model_vars["var_tax_levels_idx"]]) # take idx instead of string

# load a custom yolov5 model
def model_options(self):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
   
    # if custom model is selected
    if var_det_model.get() in custom_model_txt:
        
        # choose, display and set global var
        browse_file(var_det_model,
                    var_det_model_short,
                    var_det_model_path,
                    dsp_model,
                    [("Yolov5 model","*.pt")],
                    30,
                    dpd_options_model[lang_idx],
                    row_model)

    else:
        var_det_model_short.set("")
        var_det_model_path.set("")

    # save settings
    write_global_vars({"var_det_model_idx": dpd_options_model[lang_idx].index(var_det_model.get()), # write index instead of value
                        "var_det_model_short": var_det_model_short.get(),
                        "var_det_model_path": var_det_model_path.get()})

# view results after processing
def view_results(frame):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})")
    print(f"frame text: {frame.cget('text')}\n")
    
    # convert path separators
    chosen_folder = os.path.normpath(var_choose_folder.get())
    
    # set json paths
    image_recognition_file = os.path.join(chosen_folder, "image_recognition_file.json")
    video_recognition_file = os.path.join(chosen_folder, "video_recognition_file.json")

    # open json files at step 2
    if frame.cget('text').startswith(f' {step_txt[lang_idx]} 2'):
        if os.path.isfile(image_recognition_file):
            open_file_or_folder(image_recognition_file)
        if os.path.isfile(video_recognition_file):
            open_file_or_folder(video_recognition_file)
    
    # open destination folder at step 4
    if frame.cget('text').startswith(f' {step_txt[lang_idx]} 4'):
        open_file_or_folder(var_output_dir.get())

# open file or folder
def open_file_or_folder(path, show_error = True):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
    
    # set language var
    error_opening_results_txt = ["Error opening results", "Error al abrir los resultados"]

    # open file
    if platform.system() == 'Darwin': # mac  
        try:
            subprocess.call(('open', path))
        except:
            if show_error:
                mb.showerror(error_opening_results_txt[lang_idx], [f"Could not open '{path}'. You'll have to find it yourself...",
                                                            f"No se ha podido abrir '{path}'. Tendrás que encontrarlo tú mismo...",
                                                            f"Échec de l'ouverture de '{path}'. Vous devrez le spécifier manuellement..."][lang_idx])
    elif platform.system() == 'Windows': # windows
        try:
            os.startfile(path)
        except:
            if show_error:
                mb.showerror(error_opening_results_txt[lang_idx], [f"Could not open '{path}'. You'll have to find it yourself...",
                                                            f"No se ha podido abrir '{path}'. Tendrás que encontrarlo tú mismo...",
                                                            f"Échec de l'ouverture de '{path}'. Vous devrez le spécifier manuellement..."][lang_idx])
    else: # linux
        try:
            subprocess.call(('xdg-open', path))
        except:
            try:
                subprocess.call(('gnome-open', path))
            except:
                if show_error:
                    mb.showerror(error_opening_results_txt[lang_idx], [f"Could not open '{path}'. Neither the 'xdg-open' nor 'gnome-open' command worked. "
                                                                "You'll have to find it yourself...",
                                                                f"No se ha podido abrir '{path}'. Ni el comando 'xdg-open' ni el 'gnome-open' funcionaron. "
                                                                "Tendrá que encontrarlo usted mismo...",
                                                                f"Échec de l'ouverture de '{path}'. Ni la commande 'xdg-open' ni 'gnome-open' n'a fonctionné. "
                                                                "Vous devrez le chercher manuellement..."][lang_idx])

# retrieve model specific variables from file 
def load_model_vars(model_type = "cls"):
    if var_cls_model.get() in none_txt and model_type == "cls":
        return {}
    model_dir = var_cls_model.get() if model_type == "cls" else var_det_model.get()
    var_file = os.path.join(AddaxAI_files, "models", model_type, model_dir, "variables.json")
    try:
        with open(var_file, 'r', encoding='utf-8') as file:
            variables = json.load(file)
            return variables
    except Exception as e:
        print("DEBUG – load_model_vars failed:", e)
        return {}

# read variables.json for an arbitrary model (without touching the UI dropdowns)
def load_model_vars_for(model_type: str, model_dir: str) -> dict:
    var_file = os.path.join(AddaxAI_files, "models", model_type, model_dir, "variables.json")
    try:
        with open(var_file, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception:
        return {}

# union of all classes across all supported models (det + cls)
_ALL_SUPPORTED_MODEL_CLASSES_CACHE = None
def get_all_supported_model_classes(force_refresh: bool = False):
    global _ALL_SUPPORTED_MODEL_CLASSES_CACHE
    if _ALL_SUPPORTED_MODEL_CLASSES_CACHE is not None and not force_refresh:
        return _ALL_SUPPORTED_MODEL_CLASSES_CACHE

    all_classes = set()
    for model_type in ["cls"]:
        root_dir = os.path.join(AddaxAI_files, "models", model_type)
        if not os.path.isdir(root_dir):
            continue
        for model_dir in fetch_known_models(root_dir):
            mv = load_model_vars_for(model_type, model_dir)
            for c in mv.get("all_classes", []) or []:
                if c is None:
                    continue

                c = str(c).strip()

                if "SpeciesNet" in c:
                    continue
                if c:
                    all_classes.add(c)

    _ALL_SUPPORTED_MODEL_CLASSES_CACHE = sorted(all_classes, key=lambda s: s.lower())
    return _ALL_SUPPORTED_MODEL_CLASSES_CACHE

# write global variables to file
def write_global_vars(new_values = None):
    # adjust
    variables = load_global_vars()
    if new_values is not None:
        for key, value in new_values.items():
            if key in variables:
                variables[key] = value
            else:
                print(f"Warning: Variable {key} not found in the loaded model variables.")

    # write
    var_file = os.path.join(AddaxAI_files, "AddaxAI", "global_vars.json")
    with open(var_file, 'w') as file:
        json.dump(variables, file, indent=4)

# check which models are known and should be listed in the dpd
def fetch_known_models(root_dir):
    return sorted([subdir for subdir in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, subdir))])

# convert plt graph to img via in-memory buffer
def fig2img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

# make piechart from results.xlsx
def create_pie_chart(file_path, looks, st_angle = 45):

    # log
    print(f"EXECUTED : {sys._getframe().f_code.co_name}({locals()})\n")

    df = pd.read_excel(file_path, sheet_name='summary')
    labels = df['label']
    detections = df['n_detections']
    total_detections = sum(detections)
    percentages = (detections / total_detections) * 100
    rows = []
    for i in range(len(labels.values.tolist())):
        rows.append([labels.values.tolist()[i],
                     detections.values.tolist()[i],
                     f"{round(percentages.values.tolist()[i], 1)}%"])
    _, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"), facecolor="#CFCFCF")
    wedges, _ = ax.pie(detections, startangle=st_angle, colors=sns.color_palette('Set2'))
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    if looks != "no-lines":
        kw = dict(arrowprops=dict(arrowstyle="-"),
                bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = f"angle,angleA=0,angleB={ang}"
        if looks == "nice":
            kw["arrowprops"].update({"connectionstyle": connectionstyle}) # nicer, but sometimes raises bug: https://github.com/matplotlib/matplotlib/issues/12820
        elif looks == "simple":
            kw["arrowprops"].update({"arrowstyle": '-'})
        if looks != "no-lines":
            ax.annotate(labels[i], xy=(x, y), xytext=(1.35*np.sign(x), 1.4*y),
                        horizontalalignment=horizontalalignment, **kw)
    img = fig2img(plt)
    plt.close()
    return [img, rows]

# format the appropriate size unit
def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{round(size)} {unit}"
        size /= 1024.0

# function to create a dir and create a model_vars.json
# it does not yet download the model, but it will show up in the dropdown
def set_up_unknown_model(title, model_dict, model_type):
    model_dir = os.path.join(AddaxAI_files, "models", model_type, title)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    var_file = os.path.join(model_dir, "variables.json")
    with open(var_file, "w") as vars:
        json.dump(model_dict, vars, indent=2)
        
    # download taxonomy mapping csv if it is present
    if model_dict.get("taxon_mapping_csv", None):
        taxon_mapping_csv_url = model_dict["taxon_mapping_csv"]
        taxon_mapping_csv_path = os.path.join(model_dir, "taxon-mapping.csv")
        if not os.path.exists(taxon_mapping_csv_path):
            try:
                response = requests.get(taxon_mapping_csv_url, timeout=1)
                if response.status_code == 200:
                    with open(taxon_mapping_csv_path, 'wb') as file:
                        file.write(response.content)
                    print(f"Downloaded taxonomy mapping CSV for {title} to {taxon_mapping_csv_path}")
                else:
                    print(f"Failed to download taxonomy mapping CSV for {title}. Status code: {response.status_code}")
            except requests.exceptions.RequestException as e:   
                print(f"Error downloading taxonomy mapping CSV for {title}: {e}")

# check if this is the first startup since install 
def is_first_startup():
    return os.path.exists(os.path.join(AddaxAI_files, "first-startup.txt"))

# remove the first startup file
def remove_first_startup_file():
    first_startup_file = os.path.join(AddaxAI_files, "first-startup.txt")
    os.remove(first_startup_file)

# read existing model info and distribute separate jsons to all models
# will only be executed once: at first startup
def distribute_individual_model_jsons(model_info_fpath):
    # log
    print(f"EXECUTED : {sys._getframe().f_code.co_name}({locals()})\n")

    model_info = json.load(open(model_info_fpath))
    for typ in ["det", "cls"]:
        model_dicts = model_info[typ] 
        all_models = list(model_dicts.keys())
        for model_id in all_models:
            model_dict = model_dicts[model_id]
            set_up_unknown_model(title = model_id, model_dict = model_dict, model_type = typ)

# this function downloads a json with model info and tells the user is there is a new model
def fetch_latest_model_info():
    # log
    print(f"EXECUTED : {sys._getframe().f_code.co_name}({locals()})\n")

    # if this is the first time starting, take the existing model info file in the repo and use that
    # no need to download th same file again
    model_info_fpath = os.path.join(AddaxAI_files, "AddaxAI", "model_info", f"model_info_v{corresponding_model_info_version}.json")
    if is_first_startup():
        distribute_individual_model_jsons(model_info_fpath)
        remove_first_startup_file()
        update_model_dropdowns()

    # if this is not the first startup, it should try to download the latest model json version
    # and check if there are any new models the user should know about
    else:
        start_time = time.time()
        release_info_url = "https://api.github.com/repos/PetervanLunteren/AddaxAI/releases"
        model_info_url = f"https://raw.githubusercontent.com/PetervanLunteren/AddaxAI/main/model_info/model_info_v{corresponding_model_info_version}.json"
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
                "Accept-Encoding": "*",
                "Connection": "keep-alive"
            }
            model_info_response = requests.get(model_info_url, timeout=1, headers=headers)
            release_info_response = requests.get(release_info_url, timeout=1, headers=headers)

            # model info
            if model_info_response.status_code == 200:
                with open(model_info_fpath, 'wb') as file:
                    file.write(model_info_response.content)
                print(f"Updated model_info.json successfully.")

                # check if there is a new model available
                model_info = json.load(open(model_info_fpath))
                for typ in ["det", "cls"]:
                    model_dicts = model_info[typ] 
                    all_models = list(model_dicts.keys())
                    known_models = fetch_known_models(CLS_DIR if typ == "cls" else DET_DIR)
                    unknown_models = [e for e in all_models if e not in known_models]

                    # show a description of all the unknown models, except if first startup
                    if unknown_models != []:
                        for model_id in unknown_models:
                            model_dict = model_dicts[model_id]
                            show_model_info(title = model_id, model_dict = model_dict, new_model = True)
                            set_up_unknown_model(title = model_id, model_dict = model_dict, model_type = typ)

            # release info
            if release_info_response.status_code == 200:
                print("Checking release info")

                # check which releases are already shown
                release_shown_json = os.path.join(AddaxAI_files, "AddaxAI", "releases_shown.json")
                if os.path.exists(release_shown_json):
                    with open(release_shown_json, 'r') as f:
                        already_shown_releases = json.load(f)
                else:
                    already_shown_releases = []
                    with open(release_shown_json, 'w') as f:
                        json.dump([], f)

                # check internet
                releases = release_info_response.json()
                release_info_list = []
                for release in releases:

                    # clean tag
                    release_str = release.get("tag_name")
                    if "v." in release_str:
                        release_str = release_str.replace("v.", "")
                    elif "v" in release_str:
                        release_str = release_str.replace("v", "")

                    # collect newer versions
                    newer_version = needs_EA_update(release_str)
                    already_shown = release_str in already_shown_releases
                    if newer_version and not already_shown:
                        print(f"Found newer version: {release_str}")
                        release_info = {
                            "tag_name_raw": release.get("tag_name"),
                            "tag_name_clean": release_str,
                            "newer_version": newer_version,
                            "name": release.get("name"),
                            "body": release.get("body"),
                            "created_at": release.get("created_at"),
                            "published_at": release.get("published_at")
                        }
                        release_info_list.append(release_info)

                # show user
                for release_info in release_info_list:
                    show_release_info(release_info)
                    already_shown_releases.append(release_info["tag_name_clean"])
                
                # remember shown releases
                with open(release_shown_json, 'w') as f:
                    json.dump(already_shown_releases, f)

        except requests.exceptions.Timeout:
            print("Request timed out. File download stopped.")

        except Exception as e:
            print(f"Could not update model and version info: {e}")

        # update root so that the new models show up in the dropdown menu, 
        # but also the correct species for the existing models
        update_model_dropdowns()
        print(f"model info updated in {round(time.time() - start_time, 2)} seconds")

# open window with release info
def show_release_info(release):
    
    # define functions
    def close():
        rl_root.destroy()
    def update():
        webbrowser.open("https://addaxdatascience.com/addaxai/#install")

    # catch vars
    name_var = release["name"]
    body_var_raw = release["body"]
    date_var = datetime.datetime.strptime(release["published_at"], "%Y-%m-%dT%H:%M:%SZ").strftime("%B %d, %Y")

    # tidy body
    filtered_lines = [line for line in body_var_raw.split('\r\n') if "Full Changelog" not in line]
    body_var = '\n'.join(filtered_lines)

    # create window
    rl_root = customtkinter.CTkToplevel(root)
    rl_root.title("Release information")
    rl_root.geometry("+10+10")
    bring_window_to_top_but_not_for_ever(rl_root)

    # new version label
    lbl = customtkinter.CTkLabel(rl_root, text="New version available!", font = main_label_font)
    lbl.grid(row=0, column=0, padx=PADX, pady=(PADY, PADY/4), columnspan = 2, sticky="nswe")

    # name frame
    row_idx = 1
    name_frm_1 = model_info_frame(master=rl_root)
    name_frm_1.grid(row=row_idx, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    name_frm_2 = model_info_frame(master=name_frm_1)
    name_frm_2.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
    name_lbl_1 = customtkinter.CTkLabel(name_frm_1, text="Name", font = main_label_font)
    name_lbl_1.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), sticky="nse")
    name_lbl_2 = customtkinter.CTkLabel(name_frm_2, text=name_var)
    name_lbl_2.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), columnspan = 2, sticky="nsw")

    # date frame
    row_idx += 1
    date_frm_1 = model_info_frame(master=rl_root)
    date_frm_1.grid(row=row_idx, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    date_frm_2 = model_info_frame(master=date_frm_1)
    date_frm_2.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
    date_lbl_1 = customtkinter.CTkLabel(date_frm_1, text="Release date", font = main_label_font)
    date_lbl_1.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), sticky="nse")
    date_lbl_2 = customtkinter.CTkLabel(date_frm_2, text=date_var)
    date_lbl_2.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), columnspan = 2, sticky="nsw")

    # body frame
    row_idx += 1
    body_frm_1 = model_info_frame(master=rl_root)
    body_frm_1.grid(row=row_idx, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    body_frm_2 = model_info_frame(master=body_frm_1)
    body_frm_2.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
    body_lbl_1 = customtkinter.CTkLabel(body_frm_1, text="Description", font = main_label_font)
    body_lbl_1.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), sticky="nse")
    body_txt_1 = customtkinter.CTkTextbox(master=body_frm_2, corner_radius=10, height = 150, wrap = "word", fg_color = "transparent")
    body_txt_1.grid(row=0, column=0, padx=PADX/4, pady=(0, PADY/4), columnspan = 2, sticky="nswe")
    body_txt_1.insert("0.0", body_var)
    body_txt_1.configure(state="disabled")

    # buttons frame
    row_idx += 1
    btns_frm = customtkinter.CTkFrame(master=rl_root)
    btns_frm.columnconfigure(0, weight=1, minsize=10)
    btns_frm.columnconfigure(1, weight=1, minsize=10)
    btns_frm.grid(row=row_idx, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    close_btn = customtkinter.CTkButton(btns_frm, text="Close", command=close)
    close_btn.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="nswe")
    updat_btn = customtkinter.CTkButton(btns_frm, text="Update", command=update)
    updat_btn.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nwse")

# check if the user needs an update
def needs_EA_update(required_version):
    current_parts = list(map(int, current_AA_version.split('.')))
    required_parts = list(map(int, required_version.split('.')))

    # Pad the shorter version with zeros
    while len(current_parts) < len(required_parts):
        current_parts.append(0)
    while len(required_parts) < len(current_parts):
        required_parts.append(0)

    # Compare each part of the version
    for current, required in zip(current_parts, required_parts):
        if current < required:
            return True  # current_version is lower than required_version
        elif current > required:
            return False  # current_version is higher than required_version

    # All parts are equal, consider versions equal
    return False

# download required files for a particular model
def download_model(model_dir, skip_ask=False):
    # init vars
    model_title = os.path.basename(model_dir)
    model_type = os.path.basename(os.path.dirname(model_dir))
    model_vars = load_model_vars(model_type = model_type)
    download_info = model_vars["download_info"]
    total_download_size = model_vars["total_download_size"]

    # download
    try:          
        # check if the user wants to download
        if not skip_ask:
            if not mb.askyesno(["Download required", "Descarga necesaria", "Téléchargement requis"][lang_idx],
                            [f"The model {model_title} is not downloaded yet. It will take {total_download_size}"
                            f" of storage. Do you want to download?", f"El modelo {model_title} aún no se ha descargado."
                            f" Ocupará {total_download_size} de almacenamiento. ¿Desea descargarlo?",
                            f"Le modèle {model_title} n'a pas encore été téléchargé. Il occupera {total_download_size}"
                            f" d'espace disque. Souhaitez-vous le télécharger?"][lang_idx]):
                return False

        # set headers to trick host to thinking we are a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept-Encoding": "*",
            "Connection": "keep-alive"
        }

        # some models have multiple files to be downloaded
        # check the total size first
        total_size = 0
        for download_url, _ in download_info:
            response = requests.get(download_url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()
            total_size += int(response.headers.get('content-length', 0))

        # if yes, initiate download and show progress
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        download_popup = ModelDownloadProgressWindow(model_title = model_title, total_size_str = format_size(total_size))
        download_popup.open()
        for download_url, fname in download_info:
            file_path = os.path.normpath(os.path.join(model_dir, fname))
            dir_name = os.path.dirname(file_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            response = requests.get(download_url, stream=True, timeout=30, headers=headers)
            response.raise_for_status()
            
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=65536):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
                        percentage_done = progress_bar.n / total_size
                        download_popup.update_progress(percentage_done)
        progress_bar.close()
        download_popup.close()
        print(f"Download successful. File saved at: {file_path}")
        return True

    # catch errors
    except Exception as error:
        print("ERROR:\n" + str(error) + "\n\nDETAILS:\n" + str(traceback.format_exc()) + "\n\n")
        try:
            # remove incomplete download
            if os.path.isfile(file_path): 
                os.remove(file_path)
        except UnboundLocalError:
            # file_path is not set, meaning there is no incomplete download
            pass
        show_download_error_window(model_title, model_dir, model_vars)

# download environment
def download_environment(env_name, model_vars, skip_ask=False):
    
    # download
    try:        
        
        env_dir = os.path.join(AddaxAI_files, "envs")
        # set environment variables
        if os.name == 'nt': # windows
            download_pinned_url = f"https://addaxaipremiumstorage.blob.core.windows.net/github-zips/v{current_AA_version}/windows/envs/env-{env_name}.zip"
            download_latest_url = f"https://addaxaipremiumstorage.blob.core.windows.net/github-zips/latest/windows/envs/env-{env_name}.zip"
            filename = f"{env_name}.zip"
        elif platform.system() == 'Darwin': # macos
            import tarfile
            download_pinned_url = f"https://addaxaipremiumstorage.blob.core.windows.net/github-zips/v{current_AA_version}/macos/envs/env-{env_name}.tar.xz"
            download_latest_url = f"https://addaxaipremiumstorage.blob.core.windows.net/github-zips/latest/macos/envs/env-{env_name}.tar.xz"
            filename = f"{env_name}.tar.xz"
        else: # linux
            return False # linux install this during setup 

        # set headers to trick host to thinking we are a browser
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
            "Accept-Encoding": "*",
            "Connection": "keep-alive"
        }

        # check the total size first
        try:
            # first try the pinned version
            total_size = 0
            response = requests.get(download_pinned_url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()
            total_size += int(response.headers.get('content-length', 0))
            download_url = download_pinned_url
        except:
            # if that link doesn't woprk anymore, revert back to the latest version
            total_size = 0
            response = requests.get(download_latest_url, stream=True, timeout=60, headers=headers)
            response.raise_for_status()
            total_size += int(response.headers.get('content-length', 0))
            download_url = download_latest_url

        # check if the user wants to download
        if not skip_ask:
            if not mb.askyesno(["Download required", "Descarga necesaria", "Téléchargement requis"][lang_idx],
                            [f"The model you selected needs the virtual environment '{env_name}', which is not downloaded yet. It will take {format_size(total_size)}"
                            f" of storage. Do you want to download?", f"El envo {env_name} aún no se ha descargado."
                            f" Ocupará {format_size(total_size)} de almacenamiento. ¿Desea descargarlo?",
                            f"Le modèle sélectionné requiert un environnement virtuel '{env_name}', qui n'est pas encore téléchargé. Il occupera {format_size(total_size)}"
                            f" d'espace disque. Souhaitez-vous le télécharger?"][lang_idx]):
                return False

        # if yes, initiate download and show progress
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        download_popup = EnvDownloadProgressWindow(env_title = env_name, total_size_str = format_size(total_size))
        download_popup.open()
        file_path = os.path.join(env_dir, filename)
        response = requests.get(download_url, stream=True, timeout=60, headers=headers)
        response.raise_for_status()
        with open(file_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=65536):
                if chunk:
                    file.write(chunk)
                    progress_bar.update(len(chunk))
                    percentage_done = progress_bar.n / total_size
                    download_popup.update_download_progress(percentage_done)
        progress_bar.close()
        print(f"Download successful. File saved at: {file_path}")
        
        # After download, begin extraction
        if filename.endswith(".tar.xz"):
            # Extract the .tar.xz file
            with tarfile.open(file_path, "r:xz") as tar:
                # Get the total number of files to be extracted
                total_files = len(tar.getnames())
                extraction_progress_bar = tqdm(total=total_files, unit='file', desc="Extracting")
                
                # Extract each file and update the extraction progress
                for member in tar:
                    tar.extract(member, path=env_dir)
                    extraction_progress_bar.update(1)
                    extraction_progress_percentage = extraction_progress_bar.n / total_files
                    download_popup.update_extraction_progress(extraction_progress_percentage)
                extraction_progress_bar.close()
            download_popup.close()
            print(f"Extraction successful. Files extracted to: {env_dir}")

            # Remove the .tar.xz file after extraction
            try:
                os.remove(file_path)
                print(f"Removed the .tar.xz file: {file_path}")
            except Exception as e:
                print(f"Error removing file: {e}")

        if filename.endswith(".zip"):
            import zipfile
            # open the zip file for extraction
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                # get the total number of files to be extracted
                total_files = len(zip_ref.namelist())
                extraction_progress_bar = tqdm(total=total_files, unit='file', desc="Extracting")
                
                # extract each file and update the extraction progress
                for member in zip_ref.namelist():
                    zip_ref.extract(member, path=env_dir)
                    extraction_progress_bar.update(1)
                    extraction_progress_percentage = extraction_progress_bar.n / total_files
                    download_popup.update_extraction_progress(extraction_progress_percentage)
                extraction_progress_bar.close()
            download_popup.close()
            print(f"Extraction successful. Files extracted to: {env_dir}")
            
            # Remove the zip file after extraction
            try:
                os.remove(file_path)
                print(f"Removed the zip file: {file_path}")
            except Exception as e:
                print(f"Error removing file: {e}")

        # return success
        return True

    # catch errors
    except Exception as error:
        print("ERROR:\n" + str(error) + "\n\nDETAILS:\n" + str(traceback.format_exc()) + "\n\n")
        try:
            # remove incomplete archive
            if os.path.isfile(file_path): 
                os.remove(file_path)
                
            # remove incomplete extracted dir
            extracted_dir = os.path.join(env_dir, f"env-{env_name}")
            if os.path.isdir(extracted_dir): 
                shutil.rmtree(extracted_dir)

            # close popup
            download_popup.close()
                
        except UnboundLocalError:
            # file_path is not set, meaning there is no incomplete download
            pass

        # show internet options
        show_download_error_window_env(env_name, env_dir, model_vars)


##############################################
############# FRONTEND FUNCTIONS #############
##############################################

# open window with model info
def show_download_error_window(model_title, model_dir, model_vars):
    
    # get dwonload info
    download_info = model_vars["download_info"]
    total_download_size = model_vars["total_download_size"]
    
    # define functions
    def try_again():
        de_root.destroy()
        download_model(model_dir, skip_ask = True)
    
    # create window
    de_root = customtkinter.CTkToplevel(root)
    de_root.title(["Download error", "Error de descarga", "Erreur de téléchargement"][lang_idx])
    de_root.geometry("+10+10")
    bring_window_to_top_but_not_for_ever(de_root)

    # main label
    lbl2 = customtkinter.CTkLabel(de_root, text=f"{model_title} ({total_download_size})", font = main_label_font)
    lbl2.grid(row=0, column=0, padx=PADX, pady=(PADY, 0), columnspan = 2, sticky="nswe")
    lbl2 = customtkinter.CTkLabel(de_root, text=["Something went wrong while trying to download the model. This can have "
                                                 "several causes.", "Algo salió mal al intentar descargar el modelo. Esto "
                                                 "puede tener varias causas.",
                                                 "Quelque chose a mal tourné lors du téléchargement du modèle. Plusieurs "
                                                 "causes sont possibles."][lang_idx])
    lbl2.grid(row=1, column=0, padx=PADX, pady=(0, PADY/2), columnspan = 2, sticky="nswe")

    # internet connection frame
    int_frm_1 = customtkinter.CTkFrame(master=de_root)
    int_frm_1.grid(row=2, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    int_frm_1.columnconfigure(0, weight=1, minsize=700)
    int_frm_2 = customtkinter.CTkFrame(master=int_frm_1)
    int_frm_2.grid(row=2, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    int_frm_2.columnconfigure(0, weight=1, minsize=700)
    int_lbl = customtkinter.CTkLabel(int_frm_1, text=[" 1. Internet connection", " 1. Conexión a Internet", " 1. Connexion Internet"][lang_idx], font = main_label_font)
    int_lbl.grid(row=0, column=0, padx=PADX, pady=(PADY, PADY/2), sticky="nsw")
    int_txt_1 = customtkinter.CTkTextbox(master=int_frm_2, corner_radius=10, height = 55, wrap = "word", fg_color = "transparent")
    int_txt_1.grid(row=0, column=0, padx=PADX/4, pady=(0, PADY/4), sticky="nswe")
    int_txt_1.insert(END, ["Check if you have a stable internet connection. If possible, try again on a fibre internet "
                           "connection, or perhaps on a different, stronger, Wi-Fi network. Sometimes connecting to an "
                           "open network such as a mobile hotspot can do the trick.", "Comprueba si tienes una conexión "
                           "a Internet estable. Si es posible, inténtalo de nuevo con una conexión de fibra o quizás con "
                           "otra red Wi-Fi más potente. A veces, conectarse a una red abierta, como un hotspot móvil, "
                           "puede funcionar.",
                           "Vérifiez si votre connexion Internet est stable. Si possible, réessayer avec une connexion "
                           "sur fibre optique ou sur un réseau Wi-Fi plus puissant. Parfois, la connexion à un réseau ouvert "
                           "tel qu'un point d'accès mobile peut solutionner le problème."][lang_idx])

    # protection software frame
    pro_frm_1 = customtkinter.CTkFrame(master=de_root)
    pro_frm_1.grid(row=3, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    pro_frm_1.columnconfigure(0, weight=1, minsize=700)
    pro_frm_2 = customtkinter.CTkFrame(master=pro_frm_1)
    pro_frm_2.grid(row=2, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    pro_frm_2.columnconfigure(0, weight=1, minsize=700)
    pro_lbl = customtkinter.CTkLabel(pro_frm_1, text=[" 2. Protection software", " 2. Software de protección", " 2. Logiciel de sécurité"][lang_idx], font = main_label_font)
    pro_lbl.grid(row=0, column=0, padx=PADX, pady=(PADY, PADY/2), sticky="nsw")
    pro_txt_1 = customtkinter.CTkTextbox(master=pro_frm_2, corner_radius=10, height = 55, wrap = "word", fg_color = "transparent")
    pro_txt_1.grid(row=0, column=0, padx=PADX/4, pady=(0, PADY/4), sticky="nswe")
    pro_txt_1.insert(END, ["Some firewall, proxy or VPN settings might block the internet connection. Try again with this "
                           "protection software disabled.", "Algunas configuraciones de cortafuegos, proxy o VPN podrían "
                           "bloquear la conexión a Internet. Inténtalo de nuevo con este software de protección "
                           "desactivado.",
                           "Certains réglages de parefeux, de serveurs mandataires (proxy) ou de VPN peuvent bloquer la"
                           "connexion à Internet. Réessayer après avoir désactivé le logiciel de sécurité."][lang_idx])

    # try internet connection again 
    btns_frm1 = customtkinter.CTkFrame(master=de_root)
    btns_frm1.columnconfigure(0, weight=1, minsize=10)
    btns_frm1.grid(row=4, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    tryag_btn = customtkinter.CTkButton(btns_frm1, text=["Try internet connection again", "Prueba de nuevo la conexión a Internet", "Ré-essayer de vous connecter à Internet"][lang_idx], command=try_again)
    tryag_btn.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="nswe")

    # manual download frame
    pro_frm_1 = customtkinter.CTkFrame(master=de_root)
    pro_frm_1.grid(row=5, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    pro_frm_1.columnconfigure(0, weight=1, minsize=700)
    pro_frm_2 = customtkinter.CTkFrame(master=pro_frm_1)
    pro_frm_2.grid(row=2, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    pro_frm_2.columnconfigure(0, weight=1, minsize=700)
    pro_lbl1 = customtkinter.CTkLabel(pro_frm_1, text=[" 3. Manual download", " 3. Descarga manual", " 3. Téléchargement manuel"][lang_idx], font = main_label_font)
    pro_lbl1.grid(row=0, column=0, padx=PADX, pady=(PADY, PADY/2), sticky="nsw")
    pro_lbl2 = customtkinter.CTkLabel(pro_frm_2, text=["If the above suggestions don't work, it might be easiest to manually"
                                                       " download the file(s) and place them in the appropriate folder.", 
                                                       "Si las sugerencias anteriores no funcionan, puede que lo más fácil "
                                                       "sea descargar manualmente el archivo o archivos y colocarlos en "
                                                       "la carpeta adecuada.",
                                                       "Si les suggestions ci-dessus ne fonctionnent pas, il peut être plus "
                                                       "facile de télécharger le(s) fichier(s) manuellement et de le placer "
                                                       "dans le dossier approprié."][lang_idx])
    pro_lbl2.grid(row=0, column=0, padx=PADX, pady=(PADY, 0), sticky="nsw")

    # download instructions are dependent on their host
    step_n = 1
    show_next_steps = True
    pro_lbl5_row = 4
    if model_title == "Namibian Desert - Addax Data Science":
        main_url = download_info[0][0].replace("/resolve/main/namib_desert_v1.pt?download=true", "/tree/main")
        pro_lbl3 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Go to website:",
                                                           f" {step_n}. Ir al sitio web:",
                                                           f" {step_n}. Visiter le site web:"][lang_idx]);step_n += 1
        pro_lbl3.grid(row=2, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
        pro_lbl4 = customtkinter.CTkLabel(pro_frm_2, text=main_url, cursor="hand2", font = url_label_font)
        pro_lbl4.grid(row=3, column=0, padx=(PADX * 4, PADX), pady=(PADY/8, PADY/8), sticky="nsw")
        pro_lbl4.bind("<Button-1>", lambda e: callback(main_url))
        pro_lbl5 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Download file '{download_info[0][1]}'.",
                                                           f" {step_n}. Descarga el archivo '{download_info[0][1]}'.",
                                                           f" {step_n}. Télécharger le fichier '{download_info[0][1]}'."][lang_idx]);step_n += 1
        pro_lbl5.grid(row=4, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
    elif download_info[0][0].startswith("https://huggingface.co/Addax-Data-Science/"):
        main_url = download_info[0][0].replace(f"/resolve/main/{download_info[0][1]}?download=true", "/tree/main")
        pro_lbl3 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Go to website:",
                                                           f" {step_n}. Ir al sitio web:",
                                                           f" {step_n}. Visiter le site web:"][lang_idx]);step_n += 1
        pro_lbl3.grid(row=2, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
        pro_lbl4 = customtkinter.CTkLabel(pro_frm_2, text=main_url, cursor="hand2", font = url_label_font)
        pro_lbl4.grid(row=3, column=0, padx=(PADX * 4, PADX), pady=(PADY/8, PADY/8), sticky="nsw")
        pro_lbl4.bind("<Button-1>", lambda e: callback(main_url))
        for download_file in download_info:
            pro_lbl5 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Download file '{download_file[1]}'.",
                                                            f" {step_n}. Descarga el archivo '{download_file[1]}'.",
                                                            f" {step_n}. Télécharger le fichier '{download_file[1]}'."][lang_idx]);step_n += 1
            pro_lbl5.grid(row=pro_lbl5_row, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
            pro_lbl5_row += 1
    elif download_info[0][0].startswith("https://zenodo.org/records/"):
        main_url = download_info[0][0].replace(f"/files/{download_info[0][1]}?download=1", "")
        pro_lbl3 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Go to website:",
                                                           f" {step_n}. Ir al sitio web:",
                                                           f" {step_n}. Visiter le site web:"][lang_idx]);step_n += 1
        pro_lbl3.grid(row=2, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
        pro_lbl4 = customtkinter.CTkLabel(pro_frm_2, text=main_url, cursor="hand2", font = url_label_font)
        pro_lbl4.grid(row=3, column=0, padx=(PADX * 4, PADX), pady=(PADY/8, PADY/8), sticky="nsw")
        pro_lbl4.bind("<Button-1>", lambda e: callback(main_url))
        pro_lbl5 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Download file '{download_info[0][1]}'.",
                                                           f" {step_n}. Descarga el archivo '{download_info[0][1]}'.",
                                                           f" {step_n}. Télécharger le fichier '{download_file[0][1]}'."][lang_idx]);step_n += 1
        pro_lbl5.grid(row=4, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
    elif model_title == "Tasmania - University of Tasmania":
        main_url = download_info[1][0].replace("/resolve/main/class_list.yaml?download=true", "/tree/main")
        pro_lbl3 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Go to website:",
                                                           f" {step_n}. Ir al sitio web:",
                                                           f" {step_n}. Visiter le site web:"][lang_idx]);step_n += 1
        pro_lbl3.grid(row=2, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
        pro_lbl4 = customtkinter.CTkLabel(pro_frm_2, text=main_url, cursor="hand2", font = url_label_font)
        pro_lbl4.grid(row=3, column=0, padx=(PADX * 4, PADX), pady=(PADY/8, PADY/8), sticky="nsw")
        pro_lbl4.bind("<Button-1>", lambda e: callback(main_url))
        pro_lbl5 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Download file '{download_info[0][1]}'.",
                                                           f" {step_n}. Descarga el archivo '{download_info[0][1]}'.",
                                                           f" {step_n}. Télécharger le fichier '{download_file[0][1]}'."][lang_idx]);step_n += 1
        pro_lbl5.grid(row=4, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
        pro_lbl6 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Download file '{download_info[1][1]}'.",
                                                           f" {step_n}. Descarga el archivo '{download_info[1][1]}'.",
                                                           f" {step_n}. Télécharger le fichier '{download_file[1][1]}'."][lang_idx]);step_n += 1
        pro_lbl6.grid(row=5, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
    elif model_title == "MegaDetector 5a" or model_title == "MegaDetector 5b":
        main_url = "https://github.com/agentmorris/MegaDetector/releases/tag/v5.0"
        pro_lbl3 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Go to website:",
                                                           f" {step_n}. Ir al sitio web:",
                                                           f" {step_n}. Visiter le site web:"][lang_idx]);step_n += 1
        pro_lbl3.grid(row=2, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
        pro_lbl4 = customtkinter.CTkLabel(pro_frm_2, text=main_url, cursor="hand2", font = url_label_font)
        pro_lbl4.grid(row=3, column=0, padx=(PADX * 4, PADX), pady=(PADY/8, PADY/8), sticky="nsw")
        pro_lbl4.bind("<Button-1>", lambda e: callback(main_url))
        pro_lbl5 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Download file '{download_info[0][1]}'.",
                                                           f" {step_n}. Descarga el archivo '{download_info[0][1]}'.",
                                                           f" {step_n}. Télécharger le fichier '{download_file[0][1]}'."][lang_idx]);step_n += 1
        pro_lbl5.grid(row=4, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
    elif model_title == "Europe - DeepFaune v1.1":
        main_url = "https://pbil.univ-lyon1.fr/software/download/deepfaune/v1.1"
        pro_lbl3 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Go to website:",
                                                           f" {step_n}. Ir al sitio web:",
                                                           f" {step_n}. Visiter le site web:"][lang_idx]);step_n += 1
        pro_lbl3.grid(row=2, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
        pro_lbl4 = customtkinter.CTkLabel(pro_frm_2, text=main_url, cursor="hand2", font = url_label_font)
        pro_lbl4.grid(row=3, column=0, padx=(PADX * 4, PADX), pady=(PADY/8, PADY/8), sticky="nsw")
        pro_lbl4.bind("<Button-1>", lambda e: callback(main_url))
        pro_lbl5 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Download file '{download_info[0][1]}'.",
                                                           f" {step_n}. Descarga el archivo '{download_info[0][1]}'.",
                                                           f" {step_n}. Télécharger le fichier '{download_file[0][1]}'."][lang_idx]);step_n += 1
        pro_lbl5.grid(row=4, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
    else:
        pro_lbl3 = customtkinter.CTkLabel(pro_frm_2, text=[f" (!) No manual steps provided. Please take a screenshot of this"
                                                           " window and send an email to", f" (!) No se proporcionan pasos "
                                                           "manuales. Por favor, tome una captura de pantalla de esta ventana"
                                                           " y enviar un correo electrónico a",
                                                           f" (!) Aucun étape manuelle fournie. SVP faire une capture d'écran de "
                                                           "cette fenêtre et l'envoyer par courriel à"][lang_idx])
        pro_lbl3.grid(row=2, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
        pro_lbl4 = customtkinter.CTkLabel(pro_frm_2, text="peter@addaxdatascience.com", cursor="hand2", font = url_label_font)
        pro_lbl4.grid(row=3, column=0, padx=(PADX * 4, PADX), pady=(PADY/8, PADY/8), sticky="nsw")
        pro_lbl4.bind("<Button-1>", lambda e: callback("mailto:peter@addaxdatascience.com"))
        show_next_steps = False

    if show_next_steps:
        # general steps
        pro_lbl7 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Make sure you can view hidden files in your file explorer.", 
                                                           f" {step_n}. Asegúrate de que puedes ver los archivos ocultos en tu explorador de archivos.",
                                                           f" {step_n}. Assurez-vous de faire afficher les fichiers cachés dans l'explorateur de fichiers."][lang_idx]);step_n += 1
        pro_lbl7.grid(row=pro_lbl5_row + 1, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
        pro_lbl8 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Move the downloaded file(s) into the folder:",
                                                           f" {step_n}. Mueva los archivos descargados a la carpeta:",
                                                           f" {step_n}. Déplacer le(s) fichier(s) téléchargé(s) dans le dossier:"][lang_idx]);step_n += 1
        pro_lbl8.grid(row=pro_lbl5_row + 2, column=0, padx=PADX, pady=(0, 0), sticky="nsw")
        pro_lbl9 = customtkinter.CTkLabel(pro_frm_2, text=f"'{model_dir}'")
        pro_lbl9.grid(row=pro_lbl5_row + 3, column=0, padx=(PADX * 4, PADX), pady=(PADY/8, PADY/8), sticky="nsw")
        pro_lbl10 = customtkinter.CTkLabel(pro_frm_2, text=[f" {step_n}. Close AddaxAI and try again.",
                                                            f" {step_n}. Cierre AddaxAI e inténtelo de nuevo.",
                                                            f" {step_n}. Fermer AddaxAI et réessayer."][lang_idx]);step_n += 1
        pro_lbl10.grid(row=pro_lbl5_row + 4, column=0, padx=PADX, pady=(PADY/8, PADY/8), sticky="nsw")

        # close AddaxAI
        btns_frm2 = customtkinter.CTkFrame(master=de_root)
        btns_frm2.columnconfigure(0, weight=1, minsize=10)
        btns_frm2.grid(row=6, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
        close_btn = customtkinter.CTkButton(btns_frm2, text=["Close AddaxAI", "Cerrar AddaxAI", "Fermer AddaxAI"][lang_idx], command=on_toplevel_close)
        close_btn.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="nswe")

# open window with env info
def show_download_error_window_env(model_title, model_dir, model_vars):
    
    # create window
    de_root = customtkinter.CTkToplevel(root)
    de_root.title(["Download error", "Error de descarga", "Erreur de téléchargement"][lang_idx])
    de_root.geometry("+10+10")
    bring_window_to_top_but_not_for_ever(de_root)

    # main label
    lbl2 = customtkinter.CTkLabel(de_root, text=f"{model_title.capitalize()} download error", font = main_label_font)
    lbl2.grid(row=0, column=0, padx=PADX, pady=(PADY, 0), columnspan = 2, sticky="nswe")
    lbl2 = customtkinter.CTkLabel(de_root, text=["Something went wrong while trying to download the virtual environment. This can have "
                                                 "several causes.", "Algo salió mal al intentar descargar el modelo. Esto "
                                                 "puede tener varias causas.",
                                                 "Quelque chose a mal tourné lors du téléchargement de l'environnement virtuel. Cela "
                                                 "peut avoir plusieurs causes."][lang_idx])
    lbl2.grid(row=1, column=0, padx=PADX, pady=(0, PADY/2), columnspan = 2, sticky="nswe")

    # internet connection frame
    int_frm_1 = customtkinter.CTkFrame(master=de_root)
    int_frm_1.grid(row=2, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    int_frm_1.columnconfigure(0, weight=1, minsize=700)
    int_frm_2 = customtkinter.CTkFrame(master=int_frm_1)
    int_frm_2.grid(row=2, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    int_frm_2.columnconfigure(0, weight=1, minsize=700)
    int_lbl = customtkinter.CTkLabel(int_frm_1, text=[" 1. Internet connection", " 1. Conexión a Internet", " 1. Connexion Internet"][lang_idx], font = main_label_font)
    int_lbl.grid(row=0, column=0, padx=PADX, pady=(PADY, PADY/2), sticky="nsw")
    int_txt_1 = customtkinter.CTkTextbox(master=int_frm_2, corner_radius=10, height = 55, wrap = "word", fg_color = "transparent")
    int_txt_1.grid(row=0, column=0, padx=PADX/4, pady=(0, PADY/4), sticky="nswe")
    int_txt_1.insert(END, ["Check if you have a stable internet connection. If possible, try again on a fibre internet "
                           "connection, or perhaps on a different, stronger, Wi-Fi network. Sometimes connecting to an "
                           "open network such as a mobile hotspot can do the trick.", "Comprueba si tienes una conexión "
                           "a Internet estable. Si es posible, inténtalo de nuevo con una conexión de fibra o quizás con "
                           "otra red Wi-Fi más potente. A veces, conectarse a una red abierta, como un hotspot móvil, "
                           "puede funcionar.",
                           "Vérifiez si votre connexion Internet est stable. Si possible, réessayer avec une connexion "
                           "sur fibre optique ou sur un réseau Wi-Fi plus puissant. Parfois, la connexion à un réseau ouvert "
                           "tel qu'un point d'accès mobile peut solutionner le problème."][lang_idx])

    # protection software frame
    pro_frm_1 = customtkinter.CTkFrame(master=de_root)
    pro_frm_1.grid(row=3, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    pro_frm_1.columnconfigure(0, weight=1, minsize=700)
    pro_frm_2 = customtkinter.CTkFrame(master=pro_frm_1)
    pro_frm_2.grid(row=2, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    pro_frm_2.columnconfigure(0, weight=1, minsize=700)
    pro_lbl = customtkinter.CTkLabel(pro_frm_1, text=[" 2. Protection software", " 2. Software de protección" ," 2. Logiciel de sécurité"][lang_idx], font = main_label_font)
    pro_lbl.grid(row=0, column=0, padx=PADX, pady=(PADY, PADY/2), sticky="nsw")
    pro_txt_1 = customtkinter.CTkTextbox(master=pro_frm_2, corner_radius=10, height = 55, wrap = "word", fg_color = "transparent")
    pro_txt_1.grid(row=0, column=0, padx=PADX/4, pady=(0, PADY/4), sticky="nswe")
    pro_txt_1.insert(END, ["Some firewall, proxy or VPN settings might block the internet connection. Try again with this "
                           "protection software disabled.", "Algunas configuraciones de cortafuegos, proxy o VPN podrían "
                           "bloquear la conexión a Internet. Inténtalo de nuevo con este software de protección "
                           "desactivado.",
                           "Certains réglages de parefeux, de serveurs mandataires (proxy) ou de VPN peuvent bloquer la"
                           "connexion à Internet. Réessayer après avoir désactivé le logiciel de sécurité."][lang_idx])

# open frame to select species for advanc mode
def open_species_selection():

    # retrieve model specific variable values
    model_vars = load_model_vars()
    all_classes = model_vars['all_classes']
    selected_classes = model_vars['selected_classes']

    # on window closing
    def save():
        selected_classes = scrollable_checkbox_frame.get_checked_items()
        dsp_choose_classes.configure(text = f"{len(selected_classes)} of {len(all_classes)}")
        write_model_vars(new_values = {"selected_classes": selected_classes})
        model_cls_animal_options(var_cls_model.get())
        ss_root.withdraw()
    
    # on seleciton change
    def on_selection():
        selected_classes = scrollable_checkbox_frame.get_checked_items()
        lbl2.configure(text = f"{['Selected', 'Seleccionadas', 'Sélection de'][lang_idx]} {len(selected_classes)} {['of', 'de', 'de'][lang_idx]} {len(all_classes)}")

    # create window
    ss_root = customtkinter.CTkToplevel(root)
    ss_root.title("Species selection")
    ss_root.geometry("+10+10")
    bring_window_to_top_but_not_for_ever(ss_root)
    spp_frm_1 = customtkinter.CTkFrame(master=ss_root)
    spp_frm_1.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="nswe")
    spp_frm = customtkinter.CTkFrame(master=spp_frm_1, width=1000)
    spp_frm.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="nswe")
    lbl1 = customtkinter.CTkLabel(spp_frm, text=["Which species are present in your project area?", 
                                                 "¿Qué especies están presentes en la zona de su proyecto?",
                                                 "Quelles espèces sont présentes dans la zone de votre projet"][lang_idx],
                                                 font = main_label_font)
    lbl1.grid(row=0, column=0, padx=2*PADX, pady=PADY, columnspan = 2, sticky="nsw")
    lbl2 = customtkinter.CTkLabel(spp_frm, text="")
    lbl2.grid(row=1, column=0, padx=2*PADX, pady=0, columnspan = 2, sticky="nsw")
    scrollable_checkbox_frame = SpeciesSelectionFrame(master=spp_frm, command=on_selection,
                                                      height=400, width=500,
                                                      all_classes=all_classes,
                                                      selected_classes=selected_classes)
    scrollable_checkbox_frame._scrollbar.configure(height=0)
    scrollable_checkbox_frame.grid(row=2, column=0, padx=PADX, pady=PADY, sticky="ew")

    # toggle selection count with dummy event
    on_selection()

    # catch window close events
    close_button = customtkinter.CTkButton(ss_root, text="OK", command=save)
    close_button.grid(row=3, column=0, padx=PADX, pady=(0, PADY), columnspan = 2, sticky="nswe")
    ss_root.protocol("WM_DELETE_WINDOW", save)

# open frame to select trigger species/classes for 'keep series images'
def open_keep_series_species_selection():

    def t(en, es, fr):
        return [en, es, fr][lang_idx]

    # Full universe of valid classes across all installed cls models
    all_supported = get_all_supported_model_classes()  # sorted list :contentReference[oaicite:2]{index=2}

    # Current persisted selection (can include classes from other models)
    selected_global = global_vars.get("var_keep_series_species", []) or []
    # Drop selections that no longer exist anywhere
    selected_global = [c for c in selected_global if c in all_supported]

    # Only show classes of the currently selected classifier model
    model_vars = load_model_vars(model_type="cls")  # uses var_cls_model :contentReference[oaicite:3]{index=3}
    visible_classes = model_vars.get("all_classes", []) or []
    visible_set = set(visible_classes)

    # Only pre-check boxes that are visible in this model
    selected_visible = [c for c in selected_global if c in visible_set]

    def update_count_label():
        cur_visible = scrollable_checkbox_frame.get_checked_items()
        hidden_n = sum(1 for c in selected_global if c not in visible_set)

        if len(cur_visible) == 0 and hidden_n == 0:
            lbl2.configure(
                text=t(
                    "Trigger: any animal detection",
                    "Activador: cualquier detección de animal",
                    "Déclencheur : toute détection d'animal",
                )
            )
            return

        base = t(
            f"Selected {len(cur_visible)} of {len(visible_classes)}",
            f"Seleccionadas {len(cur_visible)} de {len(visible_classes)}",
            f"Sélection de {len(cur_visible)} de {len(visible_classes)}",
        )

        if hidden_n > 0:
            base += t(
                f" (+{hidden_n} stored from other models)",
                f" (+{hidden_n} guardadas de otros modelos)",
                f" (+{hidden_n} enregistrées d'autres modèles)",
            )

        lbl2.configure(text=base)

    def save():
        chosen_visible = scrollable_checkbox_frame.get_checked_items()

        # Keep selections from other models (not visible in the current list)
        new_set = (set(selected_global) - visible_set) | set(chosen_visible)

        # Persist in a stable order (based on all_supported order)
        chosen_all = [c for c in all_supported if c in new_set]

        global_vars["var_keep_series_species"] = chosen_all
        write_global_vars({"var_keep_series_species": chosen_all})

        # update the small counter in the keep-series frame (if it exists)
        try:
            if len(chosen_all) == 0:
                dsp_keep_series_species.configure(text=t("Any", "Cualquiera", "Toutes"))
            else:
                dsp_keep_series_species.configure(text=str(len(chosen_all)))
        except Exception:
            pass

        ss_root.withdraw()

    def clear_selection():
        nonlocal selected_global
        selected_global = []  # clears also hidden selections from other models
        for cb in scrollable_checkbox_frame.checkbox_list:
            try:
                cb.deselect()
            except Exception:
                pass
        update_count_label()

    # create window
    ss_root = customtkinter.CTkToplevel(root)
    ss_root.title(
        t(
            "Keep-series trigger species",
            "Especies activadoras para conservar series",
            "Espèces déclencheuses pour conserver les séries",
        )
    )
    ss_root.geometry("+10+10")
    bring_window_to_top_but_not_for_ever(ss_root)

    spp_frm_1 = customtkinter.CTkFrame(master=ss_root)
    spp_frm_1.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="nswe")
    spp_frm = customtkinter.CTkFrame(master=spp_frm_1, width=1000)
    spp_frm.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="nswe")

    model_name = var_cls_model.get()

    lbl1 = customtkinter.CTkLabel(
        spp_frm,
        text=t(
            "Keep empty images from the same trigger series when one of these species is detected."
            + "\n"
            + "Shown: current classifier model classes (" + model_name + ")",
            "Conservar las imágenes vacías de la misma serie cuando se detecte una de estas especies."
            + "\n"
            + "Mostradas: clases del modelo clasificador actual (" + model_name + ")",
            "Conserver les images vides de la même série lorsqu'une de ces espèces est détectée."
            + "\n"
            + "Affichées : classes du modèle classificateur actuel (" + model_name + ")",
        ),
        font=main_label_font,
    )
    lbl1.grid(row=0, column=0, padx=2 * PADX, pady=PADY, columnspan=2, sticky="nsw")

    lbl2 = customtkinter.CTkLabel(spp_frm, text="")
    lbl2.grid(row=1, column=0, padx=2 * PADX, pady=0, columnspan=2, sticky="nsw")

    scrollable_checkbox_frame = SpeciesSelectionFrame(
        master=spp_frm,
        command=update_count_label,
        height=400,
        width=500,
        all_classes=visible_classes,
        selected_classes=selected_visible,
    )
    scrollable_checkbox_frame._scrollbar.configure(height=0)
    scrollable_checkbox_frame.grid(row=2, column=0, padx=PADX, pady=PADY, sticky="ew")

    update_count_label()

    btn_row = customtkinter.CTkFrame(ss_root)
    btn_row.grid(row=3, column=0, padx=PADX, pady=(0, PADY), sticky="ew")
    btn_row.grid_columnconfigure(0, weight=1)
    btn_row.grid_columnconfigure(1, weight=1)

    clear_button = customtkinter.CTkButton(btn_row, text=t("Clear", "Borrar", "Effacer"), command=clear_selection)
    clear_button.grid(row=0, column=0, padx=(0, PADX), sticky="ew")

    close_button = customtkinter.CTkButton(btn_row, text="OK", command=save)
    close_button.grid(row=0, column=1, sticky="ew")

    ss_root.protocol("WM_DELETE_WINDOW", save)

class MyMainFrame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        if scale_factor != 1.0:
            self.columnconfigure(0, weight=1, minsize=70 * round(scale_factor * 1.35, 2))
            self.columnconfigure(1, weight=1, minsize=350 * round(scale_factor * 1.35, 2))
        else:
            self.columnconfigure(0, weight=1, minsize=70)
            self.columnconfigure(1, weight=1, minsize=350)

class MySubFrame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1, minsize=250)
        self.columnconfigure(1, weight=1, minsize=250)

class MySubSubFrame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)

class InfoButton(customtkinter.CTkButton):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color = ("#ebebeb", "#333333"),
                       hover = False,
                       text_color = ("grey", "grey"),
                       height = 1,
                       width = 1)

def sim_dir_show_info():
    mb.showinfo(title = information_txt[lang_idx],
                message = ["Select the images to analyse", "Seleccionar las imágenes a analizar", "Sélectionner les images à analyser"][lang_idx],
                detail = ["Here you can select a folder containing camera trap images. It will process all images it can find, also in subfolders."
                          " Switch to advanced mode for more options.", " Aquí puede seleccionar una carpeta que contenga imágenes de cámaras trampa."
                          " Procesará todas las imágenes que encuentre, también en las subcarpetas. Cambia al modo avanzado para más opciones.",
                          "Ici vous pouvez choisir un dossier contenant les images des pièges photographiques. Toutes les images trouvées, incluant celles "
                          "dans les sous-dossiers seront traitées. Utiliser le mode avancé pour plus d'options."][lang_idx])

def callback(url):
    webbrowser.open_new(url)

def sim_spp_show_info():
    mb.showinfo(title = information_txt[lang_idx],
                message = ["Select the species that are present", "Seleccione las especies presentes", "Sélection des espèces présentes"][lang_idx],
                detail = ["Here, you can select and deselect the animals categories that are present in your project"
                          " area. If the animal category is not selected, it will be excluded from the results. The "
                          "category list will update according to the model selected.", "Aquí puede seleccionar y anular"
                          " la selección de las categorías de animales presentes en la zona de su proyecto. Si la "
                          "categoría de animales no está seleccionada, quedará excluida de los resultados. La lista de "
                          "categorías se actualizará según el modelo seleccionado.",
                          "Ici, vous pouvez sélectionner/désélectionner les catégories d'espèces présentes dans la zone de votre projet."
                          " Si une catégorie n'est pas sélectionnée, elle sera exclue des résultats. La liste de catégories sera mise-à-jour en "
                          "fonction du modèle sélectionné."][lang_idx])

def on_spp_selection():
    selected_classes = sim_spp_scr.get_checked_items()
    all_classes = sim_spp_scr.get_all_items()
    write_model_vars(new_values = {"selected_classes": selected_classes})
    dsp_choose_classes.configure(text = f"{len(selected_classes)} of {len(all_classes)}")

def sim_mdl_show_info():
    if var_cls_model.get() in none_txt:
        mb.showinfo(title = information_txt[lang_idx],
                    message = ["Select the model to identify animals", "Seleccione el modelo para identificar animales", "Sélectionner le modèle d'identification d'animaux"][lang_idx],
                    detail = ["Here, you can choose a model that can identify your target species. If you select ‘None’, it will find vehicles,"
                              " people, and animals, but will not further identify them. When a model is selected, press this button again to "
                              "read more about the model in question.", "Aquí, puede elegir un modelo que pueda identificar su especie objetivo."
                              " Si selecciona 'Ninguno', encontrará vehículos, personas y animales, pero no los identificará más. Cuando haya "
                              "seleccionado un modelo, vuelva a pulsar este botón para obtener más información sobre el modelo en cuestión.",
                              "Ici, vous pouvez choisir un modèle qui identifie les espèces cibles. Si vous sélectionnez ‘Aucun’, il trouvera les "
                              "véhicules, les personnes et les animaux, mais sans les identifier. Lorsqu'un modèle est sélectionné, cliquer sur ce "
                              "bouton à nouveau pour obtenir plus d'information sur ce dernier."][lang_idx])
    else:
        show_model_info()


def checkbox_frame_event():
    print(f"checkbox frame modified: {sim_spp_scr.get_checked_items()}")

# class to list species with checkboxes
class SpeciesSelectionFrame(customtkinter.CTkScrollableFrame):
    def __init__(self, master, all_classes=[], selected_classes=[], command=None, dummy_spp=False, **kwargs):
        super().__init__(master, **kwargs)
        self.dummy_spp = dummy_spp
        if dummy_spp:
            all_classes = [f"{['Species', 'Especies', 'Espèces'][lang_idx]} {i + 1}" for i in range(10)]
        self.command = command
        self.checkbox_list = []
        self.selected_classes = selected_classes
        for item in all_classes:
            self.add_item(item)

    def add_item(self, item):
        checkbox = customtkinter.CTkCheckBox(self, text=item)
        if self.dummy_spp:
            checkbox.configure(state="disabled")
        if item in self.selected_classes:
            checkbox.select()
        if self.command is not None:
            checkbox.configure(command=self.command)
        checkbox.grid(row=len(self.checkbox_list), column=0, pady=PADY, sticky="nsw")
        self.checkbox_list.append(checkbox)

    def get_checked_items(self):
        return [checkbox.cget("text") for checkbox in self.checkbox_list if checkbox.get() == 1]

    def get_all_items(self):
        return [checkbox.cget("text") for checkbox in self.checkbox_list]

def open_nosleep_page():
    webbrowser.open("https://nosleep.page")

# show download and extract progress for environments
class EnvDownloadProgressWindow:
    def __init__(self, env_title, total_size_str):
        self.dm_root = customtkinter.CTkToplevel(root)
        self.dm_root.title("Download progress")
        self.dm_root.geometry("+10+10")
        self.frm = customtkinter.CTkFrame(master=self.dm_root)
        self.frm.grid(row=3, column=0, padx=PADX, pady=(PADY, PADY/2), sticky="nswe")
        self.frm.columnconfigure(0, weight=1, minsize=500 * scale_factor)
        
        self.lbl = customtkinter.CTkLabel(self.dm_root, text=[f"Downloading environment '{env_title}' ({total_size_str})",
                                                              f"Descargar entorno '{env_title}' ({total_size_str})",
                                                              f"Téléchargement de l'environnement '{env_title}' ({total_size_str})"][lang_idx], 
                                          font = customtkinter.CTkFont(family='CTkFont', size=14, weight = 'bold'))
        self.lbl.grid(row=0, column=0, padx=PADX, pady=(0, 0), sticky="nsew")
        
        self.war = customtkinter.CTkLabel(self.dm_root, text=["Please prevent computer from sleeping during the download.",
                                                              "Por favor, evite que el ordenador se duerma durante la descarga.",
                                                              "SVP empêcher l'ordinateur de tomber en mode veille lors du téléchargement."][lang_idx])
        self.war.grid(row=1, column=0, padx=PADX, pady=0, sticky="nswe")
        
        self.but = CancelButton(self.dm_root, text=["  Prevent sleep with online tool ", "  Usar prevención de sueño en línea  ", "  Prévenir la mise en veille avec un outil en ligne "][lang_idx], command=open_nosleep_page)
        self.but.grid(row=2, column=0, padx=PADX, pady=(PADY/2, 0), sticky="")
        
        # Label for Downloading Progress
        self.lbl_download = customtkinter.CTkLabel(self.frm, text=["Downloading...", "Descargando...", "Téléchargement..."][lang_idx])
        self.lbl_download.grid(row=1, column=0, padx=PADX, pady=(0, 0), sticky="nsew")

        # Progress bar for downloading
        self.pbr_download = customtkinter.CTkProgressBar(self.frm, orientation="horizontal", height=22, corner_radius=5, width=1)
        self.pbr_download.set(0)
        self.pbr_download.grid(row=2, column=0, padx=PADX, pady=PADY, sticky="nsew")

        self.per_download = customtkinter.CTkLabel(self.frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
        self.per_download.grid(row=2, column=0, padx=PADX, pady=PADY, sticky="")

        # Label for Extraction Progress
        self.lbl_extraction = customtkinter.CTkLabel(self.frm, text=["Extracting...", "Extrayendo...", "Extraction..."][lang_idx])
        self.lbl_extraction.grid(row=3, column=0, padx=PADX, pady=(0, 0), sticky="nsew")

        # Progress bar for extraction
        self.pbr_extraction = customtkinter.CTkProgressBar(self.frm, orientation="horizontal", height=22, corner_radius=5, width=1)
        self.pbr_extraction.set(0)
        self.pbr_extraction.grid(row=4, column=0, padx=PADX, pady=PADY, sticky="nsew")

        self.per_extraction = customtkinter.CTkLabel(self.frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
        self.per_extraction.grid(row=4, column=0, padx=PADX, pady=PADY, sticky="")

        self.dm_root.withdraw()

    def open(self):
        self.dm_root.update()
        self.dm_root.deiconify()

    def update_download_progress(self, percentage):
        self.pbr_download.set(percentage)
        self.per_download.configure(text=f" {round(percentage * 100)}% ")
        if percentage > 0.5:
            self.per_download.configure(fg_color=(green_primary, "#1F6BA5"))
        else:
            self.per_download.configure(fg_color=("#949BA2", "#4B4D50"))
        self.dm_root.update()

    def update_extraction_progress(self, percentage):
        self.pbr_extraction.set(percentage)
        self.per_extraction.configure(text=f" {round(percentage * 100)}% ")
        if percentage > 0.5:
            self.per_extraction.configure(fg_color=(green_primary, "#1F6BA5"))
        else:
            self.per_extraction.configure(fg_color=("#949BA2", "#4B4D50"))
        self.dm_root.update()

    def close(self):
        self.dm_root.destroy()

# show download progress for model files
class ModelDownloadProgressWindow:
    def __init__(self, model_title, total_size_str):
        self.dm_root = customtkinter.CTkToplevel(root)
        self.dm_root.title("Download progress")
        self.dm_root.geometry("+10+10")
        self.frm = customtkinter.CTkFrame(master=self.dm_root)
        self.frm.grid(row=3, column=0, padx=PADX, pady=(PADY, PADY/2), sticky="nswe")
        self.frm.columnconfigure(0, weight=1, minsize=500 * scale_factor)
        self.lbl = customtkinter.CTkLabel(self.dm_root, text=[f"Downloading model '{model_title}' ({total_size_str})",
                                                              f"Descargar modelo '{model_title}' ({total_size_str})",
                                                              f"Téléchargement du modèle '{model_title}' ({total_size_str})"][lang_idx], 
                                          font = customtkinter.CTkFont(family='CTkFont', size=14, weight = 'bold'))
        self.lbl.grid(row=0, column=0, padx=PADX, pady=(0, 0), sticky="nsew")
        self.war = customtkinter.CTkLabel(self.dm_root, text=["Please prevent computer from sleeping during the download.",
                                                          "Por favor, evite que el ordenador se duerma durante la descarga.",
                                                          "SVP empêcher l'ordinateur de tomber en mode veille pendant le téléchargement."][lang_idx])
        self.war.grid(row=1, column=0, padx=PADX, pady=0, sticky="nswe")
        self.but = CancelButton(self.dm_root, text=["  Prevent sleep with online tool ", "  Usar prevención de sueño en línea  ", "  Prévenir la mise en veille avec un outil en ligne "][lang_idx], command=open_nosleep_page)
        self.but.grid(row=2, column=0, padx=PADX, pady=(PADY/2, 0), sticky="")
        self.pbr = customtkinter.CTkProgressBar(self.frm, orientation="horizontal", height=22, corner_radius=5, width=1)
        self.pbr.set(0)
        self.pbr.grid(row=1, column=0, padx=PADX, pady=PADY, sticky="nsew")
        self.per = customtkinter.CTkLabel(self.frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
        self.per.grid(row=1, column=0, padx=PADX, pady=PADY, sticky="")
        self.dm_root.withdraw()

    def open(self):
        self.dm_root.update()
        self.dm_root.deiconify()

    def update_progress(self, percentage):
        self.pbr.set(percentage)
        self.per.configure(text = f" {round(percentage * 100)}% ")
        if percentage > 0.5:
            self.per.configure(fg_color=(green_primary, "#1F6BA5"))
        else:
            self.per.configure(fg_color=("#949BA2", "#4B4D50"))
        self.dm_root.update()

    def close(self):
        self.dm_root.destroy()

# make sure the window pops up in front initially, but does not stay on top if the users selects an other window
def bring_window_to_top_but_not_for_ever(master):
    def lift_toplevel():
        master.lift()
        master.attributes('-topmost', False)
    master.attributes('-topmost', True)
    master.after(1000, lift_toplevel)

# bunch of functions to keep track of the number of times the application has been launched
# the donation popup window will show every 5th launch
def load_launch_count():
    if not os.path.exists(launch_count_file):
        with open(launch_count_file, 'w') as f:
            json.dump({'count': 0}, f)
    with open(launch_count_file, 'r') as f:
        data = json.load(f)
        count = data.get('count', 0)
        print(f"Launch count: {count}")
        return count
def save_launch_count(count):
    with open(launch_count_file, 'w') as f:
        json.dump({'count': count}, f)
def check_donation_window_popup():
    launch_count = load_launch_count()
    launch_count += 1
    save_launch_count(launch_count)
    if launch_count % 5 == 0:
        show_donation_popup()

# show donation window
def show_donation_popup():
    
    # define functions
    def open_link(url):
        webbrowser.open(url)
    
    # define text variables
    donation_text = [
        "AddaxAI is free and open-source because we believe conservation technology should be available to everyone, regardless of budget. But keeping it that way takes time, effort, and resources—all contributed by volunteers. If you’re using AddaxAI, consider chipping in. Think of it as an honesty box: if every user contributed just $3 per month, we could sustain development, improve features, and keep expanding the model zoo.",
        "AddaxAI es gratuita y de código abierto porque creemos que la tecnología de conservación debe ser accesible para todos. Mantenerlo requiere tiempo y recursos de voluntarios. Si la usas, considera contribuir: con solo $3 al mes, podríamos seguir mejorando y ampliando el modelo de zoológico.",
        "AddaxAI est gratuit et en code ouvert car nous pensons que les technologies de conservation devraient être accessibles à tous, quel que soit le budget. Mais maintenir ce statut exige du temps, des efforts et des ressources, tous fournis par des bénévoles. Si vous utilisez AddaxAI, pensez à contribuer. Voyez-le comme une panier à dons, si chaque utilisateur contribuait ne serait-ce que 3 $ par mois, nous pourrions soutenir le développement, améliorer les fonctionnalités et continuer à développer le zoo de modèles."
    ]
    title_text = [
        "Open-source honesty box",
        "Caja de la honradez de código abierto",
        "Panier à dons pour le code ouvert"
    ]
    subtitle_text = [
        "Let's keep AddaxAI free and available for everybody!",
        "¡Mantengamos AddaxAI libre y disponible para todos!",
        "Gardons AddaxAI gratuit et disponible pour tous!"
    ]
    questions_text = [
        "Let us know if you have any questions or want to receive an invoice for tax-deduction purposes.",
        "Háganos saber si tiene alguna pregunta o desea recibir una factura para fines de deducción de impuestos.",
        "Faites-nous savoir si vous avez des questions ou souhaitez recevoir une facture pour motif de déduction d'impôts."
    ]
    email_text = "peter@addaxdatascience.com"
    btn_1_txt = [
        "$3 per month per user",
        "3$ al mes por usuario",
        "3$ par mois par utilisateur"
    ]
    btn_2_txt = [
        "Choose your own amount",
        "Elige tu propia cantidad",
        "Chosissez un montant"
    ]

    # create window
    do_root = customtkinter.CTkToplevel(root)
    do_root.title(["Model information", "Información del modelo", "Informations sur le modèle"][lang_idx])
    do_root.geometry("+10+10")
    bring_window_to_top_but_not_for_ever(do_root)

    # title frame
    row_idx = 1
    frm_1 = donation_popup_frame(master=do_root)
    frm_1.grid(row=row_idx, padx=PADX, pady=PADY, sticky="nswe")
    title_lbl_1 = customtkinter.CTkLabel(frm_1, text=title_text[lang_idx], font=customtkinter.CTkFont(family='CTkFont', size=18, weight = 'bold'))
    title_lbl_1.grid(row=0, padx=PADX, pady=(PADY, PADY/2), sticky="nswe")
    descr_txt_1 = customtkinter.CTkTextbox(master=frm_1, corner_radius=10, height=90, wrap="word", fg_color="transparent")
    descr_txt_1.grid(row=1, padx=PADX, pady=(0, 0), sticky="nswe")
    descr_txt_1.tag_config("center", justify="center")
    descr_txt_1.insert("0.0", donation_text[lang_idx], "center")
    descr_txt_1.configure(state="disabled")
    title_lbl_2 = customtkinter.CTkLabel(frm_1, text=subtitle_text[lang_idx], font=main_label_font)
    title_lbl_2.grid(row=2, padx=PADX, pady=(0, PADY), sticky="nswe")

    # buttons frame
    btns_frm = customtkinter.CTkFrame(master=do_root)
    btns_frm.columnconfigure(0, weight=1, minsize=400)
    btns_frm.columnconfigure(1, weight=1, minsize=400)
    btns_frm.grid(row=3, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    btn_1 = customtkinter.CTkButton(btns_frm, text=btn_1_txt[lang_idx], command=lambda: open_link("https://buy.stripe.com/00g8xx3aI93lb4c9AI"))
    btn_1.grid(row=1, column=0, padx=PADX, pady=(PADY, PADY/2), sticky="we")
    btn_2 = customtkinter.CTkButton(btns_frm, text=btn_2_txt[lang_idx], command=lambda: open_link("https://paymentlink.mollie.com/payment/al7x0Z6k2XWvEcdTwB5c7/"))
    btn_2.grid(row=1, column=1, padx=(0, PADX), pady=PADY, sticky="we")
    btn_lbl_2 = customtkinter.CTkLabel(btns_frm, text=questions_text[lang_idx], font=italic_label_font)
    btn_lbl_2.grid(row=2, columnspan=4, padx=PADX, pady=(PADY/2, 0), sticky="nswe")
    btn_lbl_4 = customtkinter.CTkLabel(btns_frm, text=email_text, cursor="hand2", font=url_label_font)
    btn_lbl_4.grid(row=3, columnspan=4, padx=PADX, pady=(0, PADY/2), sticky="nswe")
    btn_lbl_4.bind("<Button-1>", lambda e: callback("mailto:peter@addaxdatascience.com"))

# open window with model info
def show_model_info(title = None, model_dict = None, new_model = False):
    
    # fetch current selected model if title and model_dict are not supplied
    if title is None:
        title = var_cls_model.get()
    if model_dict is None:
        model_dict = load_model_vars()

    # read vars from json
    description_var = model_dict.get("description", "")
    developer_var = model_dict.get("developer", "")
    owner_var = model_dict.get("owner", "")
    classes_list = model_dict.get("all_classes", [])
    url_var = model_dict.get("info_url", "")
    min_version = model_dict.get("min_version", "1000.1")
    citation = model_dict.get("citation", "")
    citation_present = False if citation == "" else True
    license = model_dict.get("license", "")
    license_present = False if license == "" else True
    needs_EA_update_bool = needs_EA_update(min_version)
    if needs_EA_update_bool:
        update_var = [f"Your current AddaxAI version (v{current_AA_version}) will not be able to run this model. An update is required.",
                      f"La versión actual de AddaxAI (v{current_AA_version}) no podrá ejecutar este modelo. Se requiere una actualización.",
                      f"La version courante d'AddaxAI (v{current_AA_version}) ne pourra pas utiliser ce modèle. Une mise-à-jour est requise."][lang_idx]
    else:
        update_var = [f"Current version of AddaxAI (v{current_AA_version}) is able to use this model. No update required.",
                      f"La versión actual de AddaxAI (v{current_AA_version}) puede usar este modelo. No requiere actualización.",
                      f"La version courante d'AddaxAI (v{current_AA_version}) est en mesure d'utiliser ce modèle. Aucune mise-à-jour requise."][lang_idx]
    
    # define functions
    def close():
        nm_root.destroy()
    def read_more():
        webbrowser.open(url_var)
    def update():
        webbrowser.open("https://addaxdatascience.com/addaxai/#install")
    def cite():
        webbrowser.open(citation)
    def see_license():
        webbrowser.open(license)

    # create window
    nm_root = customtkinter.CTkToplevel(root)
    nm_root.title(["Model information", "Información sobre el modelo", "Informations sur le modèle"][lang_idx])
    nm_root.geometry("+10+10")
    bring_window_to_top_but_not_for_ever(nm_root)

    # new model label
    if new_model:
        lbl = customtkinter.CTkLabel(nm_root, text=["New model available!", "¡Nuevo modelo disponible!", "Nouveau modèle disponible!"][lang_idx], font = main_label_font)
        lbl.grid(row=0, column=0, padx=PADX, pady=(PADY, PADY/4), columnspan = 2, sticky="nswe")

    # title frame
    row_idx = 1
    title_frm_1 = model_info_frame(master=nm_root)
    title_frm_1.grid(row=row_idx, column=0, padx=PADX, pady=PADY, sticky="nswe")
    title_frm_2 = model_info_frame(master=title_frm_1)
    title_frm_2.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
    title_lbl_1 = customtkinter.CTkLabel(title_frm_1, text=["Title", "Título", "Titre"][lang_idx], font = main_label_font)
    title_lbl_1.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), sticky="nse")
    title_lbl_2 = customtkinter.CTkLabel(title_frm_2, text=title)
    title_lbl_2.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), columnspan = 2, sticky="nsw")

    # owner frame
    if owner_var != "":
        row_idx += 1
        owner_frm_1 = model_info_frame(master=nm_root)
        owner_frm_1.grid(row=row_idx, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
        owner_frm_2 = model_info_frame(master=owner_frm_1)
        owner_frm_2.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
        owner_lbl_1 = customtkinter.CTkLabel(owner_frm_1, text=["Owner", "Dueño", "Propriétaire"][lang_idx], font = main_label_font)
        owner_lbl_1.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), sticky="nse")
        owner_lbl_2 = customtkinter.CTkLabel(owner_frm_2, text=owner_var)
        owner_lbl_2.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), columnspan = 2, sticky="nsw")

    # developer frame
    row_idx += 1
    devop_frm_1 = model_info_frame(master=nm_root)
    devop_frm_1.grid(row=row_idx, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    devop_frm_2 = model_info_frame(master=devop_frm_1)
    devop_frm_2.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
    devop_lbl_1 = customtkinter.CTkLabel(devop_frm_1, text=["Developer", "Desarrollador", "Développeur"][lang_idx], font = main_label_font)
    devop_lbl_1.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), sticky="nse")
    devop_lbl_2 = customtkinter.CTkLabel(devop_frm_2, text=developer_var)
    devop_lbl_2.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), columnspan = 2, sticky="nsw")

    # description frame
    row_idx += 1
    descr_frm_1 = model_info_frame(master=nm_root)
    descr_frm_1.grid(row=row_idx, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    descr_frm_2 = model_info_frame(master=descr_frm_1)
    descr_frm_2.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
    descr_lbl_1 = customtkinter.CTkLabel(descr_frm_1, text=["Description", "Descripción", "Description"][lang_idx], font = main_label_font)
    descr_lbl_1.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), sticky="nse")
    descr_txt_1 = customtkinter.CTkTextbox(master=descr_frm_2, corner_radius=10, height = 150, wrap = "word", fg_color = "transparent")
    descr_txt_1.grid(row=0, column=0, padx=PADX/4, pady=(0, PADY/4), columnspan = 2, sticky="nswe")
    descr_txt_1.insert("0.0", description_var)
    descr_txt_1.configure(state="disabled")

    # classes frame
    row_idx += 1
    class_frm_1 = model_info_frame(master=nm_root)
    class_frm_1.grid(row=row_idx, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    class_frm_2 = model_info_frame(master=class_frm_1)
    class_frm_2.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
    class_lbl_1 = customtkinter.CTkLabel(class_frm_1, text=["Classes", "Clases", "Classes"][lang_idx], font = main_label_font)
    class_lbl_1.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), sticky="nse")
    class_txt_1 = customtkinter.CTkTextbox(master=class_frm_2, corner_radius=10, height = 150, wrap = "word", fg_color = "transparent")
    class_txt_1.grid(row=0, column=0, padx=PADX/4, pady=(0, PADY/4), columnspan = 2, sticky="nswe")
    for spp_class in classes_list:
        class_txt_1.insert(tk.END, f"• {spp_class}\n")
    class_txt_1.configure(state="disabled")

    # update frame
    row_idx += 1
    updat_frm_1 = model_info_frame(master=nm_root)
    updat_frm_1.grid(row=row_idx, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    updat_frm_2 = model_info_frame(master=updat_frm_1)
    updat_frm_2.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
    updat_lbl_1 = customtkinter.CTkLabel(updat_frm_1, text=["Update", "Actualizar", "Mise-à-jour"][lang_idx], font = main_label_font)
    updat_lbl_1.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), sticky="nse")
    updat_lbl_2 = customtkinter.CTkLabel(updat_frm_2, text=update_var)
    updat_lbl_2.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), columnspan = 2, sticky="nsw")

    # buttons frame
    row_idx += 1
    n_btns = 2
    if needs_EA_update_bool: n_btns += 1
    if citation_present: n_btns += 1
    if license_present: n_btns += 1
    btns_frm = customtkinter.CTkFrame(master=nm_root)
    for col in range(0, n_btns):
        btns_frm.columnconfigure(col, weight=1, minsize=10)
    btns_frm.grid(row=row_idx, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    close_btn = customtkinter.CTkButton(btns_frm, text=["Close", "Cerca", "Fermer"][lang_idx], command=close)
    close_btn.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="nswe")
    lmore_btn = customtkinter.CTkButton(btns_frm, text=["Learn more", "Más información", "En savoir plus"][lang_idx], command=read_more)
    lmore_btn.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nwse")
    ncol = 2
    if needs_EA_update_bool:
        updat_btn = customtkinter.CTkButton(btns_frm, text=["Update", "Actualizar", "Mise-à-jour"][lang_idx], command=update)
        updat_btn.grid(row=0, column=ncol, padx=(0, PADX), pady=PADY, sticky="nwse")
        ncol += 1
    if citation_present:
        citat_btn = customtkinter.CTkButton(btns_frm, text=["Cite", "Citar", "Citer"][lang_idx], command=cite)
        citat_btn.grid(row=0, column=ncol, padx=(0, PADX), pady=PADY, sticky="nwse")
        ncol += 1
    if license_present:
        licen_btn = customtkinter.CTkButton(btns_frm, text=["License", "Licencia", "Licence"][lang_idx], command=see_license)
        licen_btn.grid(row=0, column=ncol, padx=(0, PADX), pady=PADY, sticky="nwse")
        ncol += 1

# class frame for model window
class model_info_frame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1, minsize=120 * scale_factor)
        self.columnconfigure(1, weight=1, minsize=500 * scale_factor)

# class frame for donation window
class donation_popup_frame(customtkinter.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1, minsize=500 * scale_factor)

# make sure the latest updated models also are listed in the dpd menu
def update_model_dropdowns():
    global dpd_options_cls_model
    cls_models = fetch_known_models(CLS_DIR)
    dpd_options_cls_model = [["None"] + cls_models, ["Ninguno"] + cls_models, ["Aucun"] + cls_models]
    update_dpd_options(dpd_cls_model, snd_step, var_cls_model, dpd_options_cls_model, model_cls_animal_options, row_cls_model, lbl_cls_model, lang_idx)
    global dpd_options_model
    det_models = fetch_known_models(DET_DIR)
    dpd_options_model = [det_models + ["Custom model"], det_models + ["Otro modelo"], det_models + ["Modèle personnalisé"]]
    update_dpd_options(dpd_model, snd_step, var_det_model, dpd_options_model, model_options, row_model, lbl_model, lang_idx)
    model_cls_animal_options(var_cls_model.get())
    global sim_dpd_options_cls_model
    sim_dpd_options_cls_model = [[item[0] + suffixes_for_sim_none[i], *item[1:]] for i, item in enumerate(dpd_options_cls_model)]
    update_sim_mdl_dpd()
    root.update_idletasks()

# window for quick results info while running simple mode
def show_result_info(file_path):
    
    # define functions
    def close():
        rs_root.withdraw()
    def more_options():
        switch_mode()
        rs_root.withdraw()

    file_path = os.path.normpath(file_path)

    # read results for xlsx file
    # some combinations of percentages raises a bug: https://github.com/matplotlib/matplotlib/issues/12820
    # hence we're going to try the nicest layout with some different angles, then an OK layout, and no
    # lines as last resort
    try:
        graph_img, table_rows = create_pie_chart(file_path, looks = "nice", st_angle = 0)
    except ValueError:
        print("ValueError - trying again with different params.")
        try:
            graph_img, table_rows = create_pie_chart(file_path, looks = "nice", st_angle = 23)
        except ValueError:
            print("ValueError - trying again with different params.")
            try:
                graph_img, table_rows = create_pie_chart(file_path, looks = "nice", st_angle = 45)
            except ValueError:
                print("ValueError - trying again with different params.")
                try:
                    graph_img, table_rows = create_pie_chart(file_path, looks = "nice", st_angle = 90)
                except ValueError:
                    print("ValueError - trying again with different params.")
                    try:
                        graph_img, table_rows = create_pie_chart(file_path, looks = "simple")
                    except ValueError:
                        print("ValueError - trying again with different params.")
                        graph_img, table_rows = create_pie_chart(file_path, looks = "no-lines")

    # create window
    rs_root = customtkinter.CTkToplevel(root)
    rs_root.title("Results - quick view")
    rs_root.geometry("+10+10")
    result_bg_image = customtkinter.CTkImage(PIL_sidebar, size=(RESULTS_WINDOW_WIDTH, RESULTS_WINDOW_HEIGHT))
    result_bg_image_label = customtkinter.CTkLabel(rs_root, image=result_bg_image)
    result_bg_image_label.grid(row=0, column=0)
    result_main_frame = customtkinter.CTkFrame(rs_root, corner_radius=0, fg_color = 'transparent')
    result_main_frame.grid(row=0, column=0, sticky="ns")

    # label
    lbl1 = customtkinter.CTkLabel(result_main_frame, text=["The images are processed!", "¡Las imágenes están procesadas!", "Les images ont été traitées!"][lang_idx], font = main_label_font, height=20)
    lbl1.grid(row=0, column=0, padx=PADX, pady=(PADY, PADY/4), columnspan = 2, sticky="nswe")
    lbl2 = customtkinter.CTkLabel(result_main_frame, text=[f"The results and graphs are saved at '{os.path.dirname(file_path)}'.", f"Los resultados y gráficos se guardan en '{os.path.dirname(file_path)}'.",
                                                           f"Les résultats et graphiques sont enregistrés sous '{os.path.dirname(file_path)}'."][lang_idx], height=20)
    lbl2.grid(row=1, column=0, padx=PADX, pady=(PADY/4, PADY/4), columnspan = 2, sticky="nswe")
    lbl3 = customtkinter.CTkLabel(result_main_frame, text=[f"You can find a quick overview of the results below.", f"A continuación encontrará un resumen de los resultados.",
                                                           f"Un aperçu des résultats est présenté ci-dessous."][lang_idx], height=20)
    lbl3.grid(row=2, column=0, padx=PADX, pady=(PADY/4, PADY/4), columnspan = 2, sticky="nswe")

    # graph frame
    graph_frm_1 = model_info_frame(master=result_main_frame)
    graph_frm_1.grid(row=3, column=0, padx=PADX, pady=PADY, sticky="nswe")
    graph_frm_2 = model_info_frame(master=graph_frm_1)
    graph_frm_2.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
    graph_lbl_1 = customtkinter.CTkLabel(graph_frm_1, text=["Graph", "Gráfico", "Graphique"][lang_idx], font = main_label_font)
    graph_lbl_1.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), sticky="nse")
    graph_img = customtkinter.CTkImage(graph_img, size=(600, 300))
    graph_lbl_2 = customtkinter.CTkLabel(graph_frm_2, text="", image = graph_img)
    graph_lbl_2.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), columnspan = 2, sticky="nsw")

    # table frame
    table_frm_1 = model_info_frame(master=result_main_frame)
    table_frm_1.grid(row=4, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    table_lbl_1 = customtkinter.CTkLabel(table_frm_1, text=["Table", "Tabla", "Table"][lang_idx], font = main_label_font)
    table_lbl_1.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), sticky="nse")
    table_scr_frm = customtkinter.CTkScrollableFrame(table_frm_1, width = RESULTS_TABLE_WIDTH)
    table_scr_frm.grid(row=0, column=1, columnspan = 3, padx=(0, PADX), pady=PADY, sticky="nesw")
    table_header = CTkTable(master=table_scr_frm,
                      column=3,
                      values=[[["Species", "Especie", "Espèces"][lang_idx], ["Count", "Cuenta", "Compte"][lang_idx], ["Percentage", "Porcentaje", "Pourcentage"][lang_idx]]],
                      font = main_label_font,
                      color_phase = "horizontal",
                      header_color = customtkinter.ThemeManager.theme["CTkFrame"]["top_fg_color"],
                      wraplength = 130)
    table_header.grid(row=0, column=0, padx=PADX, pady=(PADY/4, 0), columnspan = 2, sticky="nesw")
    table_values = CTkTable(master=table_scr_frm,
                      column=3, 
                      values=table_rows,
                      color_phase = "horizontal",
                      wraplength = 130,
                      width = RESULTS_TABLE_WIDTH / 3 - PADX)
    table_values.grid(row=1, column=0, padx=PADX, pady=(0, PADY/4), columnspan = 2, sticky="nesw")

    # buttons frame
    btns_frm = customtkinter.CTkFrame(master=result_main_frame)
    btns_frm.grid(row=5, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
    btns_frm.columnconfigure(0, weight=1, minsize=10)
    btns_frm.columnconfigure(1, weight=1, minsize=10)
    btns_frm.columnconfigure(2, weight=1, minsize=10)
    btns_frm.columnconfigure(3, weight=1, minsize=10)
    close_btn = customtkinter.CTkButton(btns_frm, text=["Close window", "Cerrar ventana", "Fermer la fenêtre"][lang_idx], command=close)
    close_btn.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="nswe")
    openf_btn = customtkinter.CTkButton(btns_frm, text=["See results", "Ver resultados", "Voir les résultats"][lang_idx], command=lambda: open_file_or_folder(file_path))
    openf_btn.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nwse")
    seegr_dir_path = os.path.join(os.path.dirname(file_path), "graphs")
    seegr_btn = customtkinter.CTkButton(btns_frm, text=["See graphs", "Ver gráficos", "Voir les graphiques"][lang_idx], command=lambda: open_file_or_folder(seegr_dir_path))
    seegr_btn.grid(row=0, column=2, padx=(0, PADX), pady=PADY, sticky="nwse")
    moreo_btn = customtkinter.CTkButton(btns_frm, text=["More options", "Otras opciones", "Autres options (mode avancé)"][lang_idx], command=more_options)
    moreo_btn.grid(row=0, column=3, padx=(0, PADX), pady=PADY, sticky="nwse")

    # place in front
    bring_window_to_top_but_not_for_ever(rs_root)

# class for simple question with buttons
class TextButtonWindow:
    def __init__(self, title, text, buttons):
        self.root = customtkinter.CTkToplevel(root)
        self.root.title(title)
        self.root.geometry("+10+10")
        bring_window_to_top_but_not_for_ever(self.root)
        self.root.protocol("WM_DELETE_WINDOW", self.user_close)
        
        self.text_label = tk.Label(self.root, text=text)
        self.text_label.pack(padx=10, pady=10)
        
        self.selected_button = None
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(padx=10, pady=10)
        
        for button_text in buttons:
            button = tk.Button(self.button_frame, text=button_text, command=lambda btn=button_text: self._button_click(btn))
            button.pack(side=tk.LEFT, padx=5)
        
    def _button_click(self, button_text):
        self.selected_button = button_text
        self.root.quit()
        
    def open(self):
        self.root.mainloop()
        
    def user_close(self):
        self.selected_button = "EXIT"
        self.root.quit()
        self.root.destroy()

    def run(self):
        self.open()
        self.root.destroy()
        return self.selected_button

# simple window to show progressbar
class PatienceDialog:
    def __init__(self, total, text):
        self.root = customtkinter.CTkToplevel(root)
        self.root.title(["Be patient", "Esperar", "Patientez"][lang_idx])
        self.root.geometry("+10+10")
        self.total = total
        self.text = text
        self.label = tk.Label(self.root, text=text)
        self.label.pack(pady=10)
        self.progress = ttk.Progressbar(self.root, mode='determinate', length=200)
        self.progress.pack(pady=10, padx = 10)
        self.root.withdraw()

    def open(self):
        self.root.update()
        self.root.deiconify()

    def update_progress(self, current, percentage = False):
        # updating takes considerable time - only do this 100 times
        if current % math.ceil(self.total / 100) == 0:
            self.progress['value'] = (current / self.total) * 100
            if percentage:
                percentage_value = round((current/self.total) * 100)
                self.label.configure(text = f"{self.text}\n{percentage_value}%")
            else:
                self.label.configure(text = f"{self.text}\n{current} of {self.total}")
            self.root.update()

    def close(self):
        self.root.destroy()

# simple window class to pop up and be closed
class CustomWindow:
    def __init__(self, title="", text=""):
        self.title = title
        self.text = text
        self.root = None

    def open(self):
        self.root = customtkinter.CTkToplevel(root)
        self.root.title(self.title)
        self.root.geometry("+10+10")

        label = tk.Label(self.root, text=self.text)
        label.pack(padx=10, pady=10)

        self.root.update_idletasks()
        self.root.update()

    def close(self):
        self.root.destroy()

# disable annotation frame
def disable_ann_frame(row, hitl_ann_selection_frame):
    labelframe = hitl_ann_selection_frame.grid_slaves(row=row, column=3)[0]
    labelframe.configure(relief=SUNKEN)
    for widget in labelframe.winfo_children():
        widget.configure(state = DISABLED)

# enable annotation frame
def enable_ann_frame(row, hitl_ann_selection_frame):
    labelframe = hitl_ann_selection_frame.grid_slaves(row=row, column=3)[0]
    labelframe.configure(relief=RAISED)
    for widget in labelframe.winfo_children():
        widget.configure(state = NORMAL)


# toggle taxonomic fallback widgets
def toggle_tax_levels():
    write_model_vars(new_values={"var_tax_fallback": var_tax_fallback.get()})
    if var_tax_fallback.get():
        lbl_tax_levels.grid(row=row_tax_levels, sticky='nesw', pady=2)
        dpd_tax_levels.grid(row=row_tax_levels, column=1, sticky='nesw', padx=5, pady=2)
        set_minsize_rows(cls_frame)
    else:
        lbl_tax_levels.grid_forget()
        dpd_tax_levels.grid_forget()
        cls_frame.grid_rowconfigure(6, minsize=0)

# show hide the annotation selection frame in the human-in-the-loop settings window
def toggle_hitl_ann_selection_frame(cmd = None):
    is_vis = hitl_ann_selection_frame.grid_info()
    if cmd == "hide":
        hitl_ann_selection_frame.grid_remove()
    else:
        if is_vis != {}:
            hitl_ann_selection_frame.grid_remove()
        else:
            hitl_ann_selection_frame.grid(column=0, row=2, columnspan=2, sticky='ew')
    hitl_settings_window.update()
    hitl_settings_canvas.configure(scrollregion=hitl_settings_canvas.bbox("all"))

# enable or disable the options in the human-in-the-loop annotation selection frame
def toggle_hitl_ann_selection(rad_ann_var, hitl_ann_selection_frame):
    rad_ann_var = rad_ann_var.get()    
    cols, rows = hitl_ann_selection_frame.grid_size()
    if rad_ann_var == 1:
        enable_ann_frame(1, hitl_ann_selection_frame)
        for row in range(2, rows):
            disable_ann_frame(row, hitl_ann_selection_frame)
    elif rad_ann_var == 2:
        disable_ann_frame(1, hitl_ann_selection_frame)
        for row in range(2, rows):
            enable_ann_frame(row, hitl_ann_selection_frame)


# update counts of the subset functions of the human-in-the-loop image selection frame
def enable_amt_per_ent(row):
    global selection_dict
    rad_var = selection_dict[row]['rad_var'].get()
    ent_per = selection_dict[row]['ent_per']
    ent_amt = selection_dict[row]['ent_amt']
    if rad_var == 1:
        ent_per.configure(state = DISABLED)
        ent_amt.configure(state = DISABLED)      
    if rad_var == 2:
        ent_per.configure(state = NORMAL)
        ent_amt.configure(state = DISABLED)
    if rad_var == 3:
        ent_per.configure(state = DISABLED)
        ent_amt.configure(state = NORMAL)

# show or hide widgets in the human-in-the-loop image selection frame
def enable_selection_widgets(row):
    global selection_dict
    frame = selection_dict[row]['frame']
    chb_var = selection_dict[row]['chb_var'].get()
    lbl_class = selection_dict[row]['lbl_class']
    rsl = selection_dict[row]['range_slider_widget']
    rad_all = selection_dict[row]['rad_all']
    rad_per = selection_dict[row]['rad_per']
    rad_amt = selection_dict[row]['rad_amt']
    lbl_n_img = selection_dict[row]['lbl_n_img']
    if chb_var:
        frame.configure(relief = RAISED)
        lbl_class.configure(state = NORMAL)
        rsl.grid(row = 0, rowspan= 3, column = 2)
        rad_all.configure(state = NORMAL)
        rad_per.configure(state = NORMAL)
        rad_amt.configure(state = NORMAL)
        lbl_n_img.configure(state = NORMAL)
    else:
        frame.configure(relief = SUNKEN)
        lbl_class.configure(state = DISABLED)
        rsl.grid_remove()
        rad_all.configure(state = DISABLED)
        rad_per.configure(state = DISABLED)
        rad_amt.configure(state = DISABLED)
        lbl_n_img.configure(state = DISABLED)

# front end class for cancel button
class CancelButton(customtkinter.CTkButton):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color = ("#ebeaea", "#4B4D50"),
                       hover_color = ("#939aa2", "#2B2B2B"),
                       text_color = ("black", "white"),
                       height = 10,
                       width = 120)

# open progress window for deploy and postprocess
class ProgressWindow:
    def __init__(self, processes):
        self.progress_top_level_window = customtkinter.CTkToplevel()
        self.progress_top_level_window.title(["Analysis progress", "Progreso del análisis", "Progression de l'analyse"][lang_idx])
        self.progress_top_level_window.geometry("+10+10")
        lbl_height = 12
        pbr_height = 22
        ttl_font = customtkinter.CTkFont(family='CTkFont', size=13, weight = 'bold')
        self.pady_progress_window = PADY/1.5
        self.padx_progress_window = PADX/1.5
        
        # language settings
        in_queue_txt = ['In queue', 'En cola', 'Mise en file d\'attente']
        checking_fpaths_txt = ['Checking file paths', 'Comprobación de rutas de archivos', 'Vérification des chemins de fichiers']
        processing_image_txt = ['Processing image', 'Procesamiento de imágenes', 'Traitement des images']
        processing_animal_txt = ['Processing animal', 'Procesamiento de animales', 'Traitement des animaux']
        processing_unknown_txt = ['Processing', 'Procesamiento', 'Traitement']
        images_per_second_txt = ['Images per second', 'Imágenes por segundo', 'Images par seconde']
        animals_per_second_txt = ['Animals per second', 'Animales por segundo', 'Animaux par seconde']
        frames_per_second_txt = ['Frames per second', 'Fotogramas por segundo', 'Trames par seconde']
        elapsed_time_txt = ["Elapsed time", "Tiempo transcurrido", "Temps écoulé"]
        remaining_time_txt = ["Remaining time", "Tiempo restante", "Temps restant"]
        running_on_txt = ["Running on", "Funcionando en", "S'exécute sur"]

        # clarify titles if both images and videos are being processed
        if "img_det" in processes and "vid_det" in processes:
            img_det_extra_string = [" in images", " en imágenes", " dans les images"][lang_idx]
            vid_det_extra_string = [" in videos", " en vídeos", " dans les vidéos"][lang_idx]
        else:
            img_det_extra_string = ""
            vid_det_extra_string = ""
        if "img_pst" in processes and "vid_pst" in processes:
            img_pst_extra_string = [" images", " de imágenes", " images"][lang_idx]
            vid_pst_extra_string = [" videos", " de vídeos", " vidéos"][lang_idx]
        else:
            img_pst_extra_string = ""
            vid_pst_extra_string = ""

        # initialise image detection process
        if "img_det" in processes:
            self.img_det_frm = customtkinter.CTkFrame(master=self.progress_top_level_window)
            self.img_det_frm.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            img_det_ttl_txt = [f'Locating animals{img_det_extra_string}...', f'Localización de animales{img_det_extra_string}...', f'Localisation des animaux{img_det_extra_string}...']
            self.img_det_ttl = customtkinter.CTkLabel(self.img_det_frm, text=img_det_ttl_txt[lang_idx], font = ttl_font)
            self.img_det_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.img_det_sub_frm = customtkinter.CTkFrame(master=self.img_det_frm)
            self.img_det_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.img_det_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.img_det_pbr = customtkinter.CTkProgressBar(self.img_det_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.img_det_pbr.set(0)
            self.img_det_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.img_det_per = customtkinter.CTkLabel(self.img_det_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.img_det_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.img_det_wai_lbl = customtkinter.CTkLabel(self.img_det_sub_frm, height = lbl_height, text=checking_fpaths_txt[lang_idx])
            self.img_det_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.img_det_num_lbl = customtkinter.CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"{processing_image_txt[lang_idx]}:")
            self.img_det_num_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_det_num_lbl.grid_remove()
            self.img_det_num_val = customtkinter.CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"")
            self.img_det_num_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_det_num_val.grid_remove()
            self.img_det_ela_lbl = customtkinter.CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[lang_idx]}:")
            self.img_det_ela_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_det_ela_lbl.grid_remove()
            self.img_det_ela_val = customtkinter.CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"")
            self.img_det_ela_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.img_det_ela_val.grid_remove()
            self.img_det_rem_lbl = customtkinter.CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"{remaining_time_txt[lang_idx]}:")
            self.img_det_rem_lbl.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_det_rem_lbl.grid_remove()
            self.img_det_rem_val = customtkinter.CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"")
            self.img_det_rem_val.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_det_rem_val.grid_remove()
            self.img_det_spe_lbl = customtkinter.CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"{images_per_second_txt[lang_idx]}:")
            self.img_det_spe_lbl.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_det_spe_lbl.grid_remove()
            self.img_det_spe_val = customtkinter.CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"")
            self.img_det_spe_val.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_det_spe_val.grid_remove()
            self.img_det_hwa_lbl = customtkinter.CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"{running_on_txt[lang_idx]}:")
            self.img_det_hwa_lbl.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_det_hwa_lbl.grid_remove()
            self.img_det_hwa_val = customtkinter.CTkLabel(self.img_det_sub_frm, height = lbl_height, text=f"")
            self.img_det_hwa_val.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_det_hwa_val.grid_remove()
            self.img_det_can_btn = CancelButton(master = self.img_det_sub_frm, text = "Cancel", command = lambda: print(""))
            self.img_det_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.img_det_can_btn.grid_remove()

        # initialise image classification process
        if "img_cls" in processes:
            self.img_cls_frm = customtkinter.CTkFrame(master=self.progress_top_level_window)
            self.img_cls_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            img_cls_ttl_txt = [f'Identifying animals{img_det_extra_string}...', f'Identificación de animales{img_det_extra_string}...', f'Identification des animaux{img_det_extra_string}...']
            self.img_cls_ttl = customtkinter.CTkLabel(self.img_cls_frm, text=img_cls_ttl_txt[lang_idx], font = ttl_font)
            self.img_cls_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.img_cls_sub_frm = customtkinter.CTkFrame(master=self.img_cls_frm)
            self.img_cls_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.img_cls_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.img_cls_pbr = customtkinter.CTkProgressBar(self.img_cls_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.img_cls_pbr.set(0)
            self.img_cls_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.img_cls_per = customtkinter.CTkLabel(self.img_cls_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.img_cls_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.img_cls_wai_lbl = customtkinter.CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=in_queue_txt[lang_idx])
            self.img_cls_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.img_cls_num_lbl = customtkinter.CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"{processing_animal_txt[lang_idx]}:")
            self.img_cls_num_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_cls_num_lbl.grid_remove()
            self.img_cls_num_val = customtkinter.CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"")
            self.img_cls_num_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_cls_num_val.grid_remove()
            self.img_cls_ela_lbl = customtkinter.CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[lang_idx]}:")
            self.img_cls_ela_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_cls_ela_lbl.grid_remove()
            self.img_cls_ela_val = customtkinter.CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"")
            self.img_cls_ela_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.img_cls_ela_val.grid_remove()
            self.img_cls_rem_lbl = customtkinter.CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"{remaining_time_txt[lang_idx]}:")
            self.img_cls_rem_lbl.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_cls_rem_lbl.grid_remove()
            self.img_cls_rem_val = customtkinter.CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"")
            self.img_cls_rem_val.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_cls_rem_val.grid_remove()
            self.img_cls_spe_lbl = customtkinter.CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"{animals_per_second_txt[lang_idx]}:")
            self.img_cls_spe_lbl.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_cls_spe_lbl.grid_remove()
            self.img_cls_spe_val = customtkinter.CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"")
            self.img_cls_spe_val.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_cls_spe_val.grid_remove()
            self.img_cls_hwa_lbl = customtkinter.CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"{running_on_txt[lang_idx]}:")
            self.img_cls_hwa_lbl.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_cls_hwa_lbl.grid_remove()
            self.img_cls_hwa_val = customtkinter.CTkLabel(self.img_cls_sub_frm, height = lbl_height, text=f"")
            self.img_cls_hwa_val.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.img_cls_hwa_val.grid_remove()
            self.img_cls_can_btn = CancelButton(master = self.img_cls_sub_frm, text = "Cancel", command = lambda: print(""))
            self.img_cls_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.img_cls_can_btn.grid_remove()

        # initialise video detection process
        if "vid_det" in processes:
            self.vid_det_frm = customtkinter.CTkFrame(master=self.progress_top_level_window)
            self.vid_det_frm.grid(row=2, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            vid_det_ttl_txt = [f'Locating animals{vid_det_extra_string}...', f'Localización de animales{vid_det_extra_string}...', f'Localisation des animaux{vid_det_extra_string}...']
            self.vid_det_ttl = customtkinter.CTkLabel(self.vid_det_frm, text=vid_det_ttl_txt[lang_idx], font = ttl_font)
            self.vid_det_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.vid_det_sub_frm = customtkinter.CTkFrame(master=self.vid_det_frm)
            self.vid_det_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.vid_det_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.vid_det_pbr = customtkinter.CTkProgressBar(self.vid_det_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.vid_det_pbr.set(0)
            self.vid_det_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.vid_det_per = customtkinter.CTkLabel(self.vid_det_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.vid_det_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.vid_det_wai_lbl = customtkinter.CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=in_queue_txt[lang_idx])
            self.vid_det_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.vid_det_num_lbl = customtkinter.CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"{processing_unknown_txt[lang_idx]}:")
            self.vid_det_num_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_det_num_lbl.grid_remove()
            self.vid_det_num_val = customtkinter.CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"")
            self.vid_det_num_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_det_num_val.grid_remove()
            self.vid_det_ela_lbl = customtkinter.CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[lang_idx]}:")
            self.vid_det_ela_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_det_ela_lbl.grid_remove()
            self.vid_det_ela_val = customtkinter.CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"")
            self.vid_det_ela_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.vid_det_ela_val.grid_remove()
            self.vid_det_rem_lbl = customtkinter.CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"{remaining_time_txt[lang_idx]}:")
            self.vid_det_rem_lbl.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_det_rem_lbl.grid_remove()
            self.vid_det_rem_val = customtkinter.CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"")
            self.vid_det_rem_val.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_det_rem_val.grid_remove()
            self.vid_det_spe_lbl = customtkinter.CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"{frames_per_second_txt[lang_idx]}:")
            self.vid_det_spe_lbl.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_det_spe_lbl.grid_remove()
            self.vid_det_spe_val = customtkinter.CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"")
            self.vid_det_spe_val.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_det_spe_val.grid_remove()
            self.vid_det_hwa_lbl = customtkinter.CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"{running_on_txt[lang_idx]}:")
            self.vid_det_hwa_lbl.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_det_hwa_lbl.grid_remove()
            self.vid_det_hwa_val = customtkinter.CTkLabel(self.vid_det_sub_frm, height = lbl_height, text=f"")
            self.vid_det_hwa_val.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_det_hwa_val.grid_remove()
            self.vid_det_can_btn = CancelButton(master = self.vid_det_sub_frm, text = "Cancel", command = lambda: print(""))
            self.vid_det_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.vid_det_can_btn.grid_remove()

        # initialise video classification process
        if "vid_cls" in processes:
            self.vid_cls_frm = customtkinter.CTkFrame(master=self.progress_top_level_window)
            self.vid_cls_frm.grid(row=3, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            vid_cls_ttl_txt = [f'Identifying animals{vid_det_extra_string}...', f'Identificación de animales{vid_det_extra_string}...', f'Identification des animaux{vid_det_extra_string}...']
            self.vid_cls_ttl = customtkinter.CTkLabel(self.vid_cls_frm, text=vid_cls_ttl_txt[lang_idx], font = ttl_font)
            self.vid_cls_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.vid_cls_sub_frm = customtkinter.CTkFrame(master=self.vid_cls_frm)
            self.vid_cls_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.vid_cls_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.vid_cls_pbr = customtkinter.CTkProgressBar(self.vid_cls_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.vid_cls_pbr.set(0)
            self.vid_cls_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.vid_cls_per = customtkinter.CTkLabel(self.vid_cls_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.vid_cls_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.vid_cls_wai_lbl = customtkinter.CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=in_queue_txt[lang_idx])
            self.vid_cls_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.vid_cls_num_lbl = customtkinter.CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"{processing_animal_txt[lang_idx]}:")
            self.vid_cls_num_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_cls_num_lbl.grid_remove()
            self.vid_cls_num_val = customtkinter.CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"")
            self.vid_cls_num_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_cls_num_val.grid_remove()
            self.vid_cls_ela_lbl = customtkinter.CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[lang_idx]}:")
            self.vid_cls_ela_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_cls_ela_lbl.grid_remove()
            self.vid_cls_ela_val = customtkinter.CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"")
            self.vid_cls_ela_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.vid_cls_ela_val.grid_remove()
            self.vid_cls_rem_lbl = customtkinter.CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"{remaining_time_txt[lang_idx]}:")
            self.vid_cls_rem_lbl.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_cls_rem_lbl.grid_remove()
            self.vid_cls_rem_val = customtkinter.CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"")
            self.vid_cls_rem_val.grid(row=4, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_cls_rem_val.grid_remove()
            self.vid_cls_spe_lbl = customtkinter.CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"{animals_per_second_txt[lang_idx]}:")
            self.vid_cls_spe_lbl.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_cls_spe_lbl.grid_remove()
            self.vid_cls_spe_val = customtkinter.CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"")
            self.vid_cls_spe_val.grid(row=5, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_cls_spe_val.grid_remove()
            self.vid_cls_hwa_lbl = customtkinter.CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"{running_on_txt[lang_idx]}:")
            self.vid_cls_hwa_lbl.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_cls_hwa_lbl.grid_remove()
            self.vid_cls_hwa_val = customtkinter.CTkLabel(self.vid_cls_sub_frm, height = lbl_height, text=f"")
            self.vid_cls_hwa_val.grid(row=6, padx=self.padx_progress_window, pady=0, sticky="nse")
            self.vid_cls_hwa_val.grid_remove()
            self.vid_cls_can_btn = CancelButton(master = self.vid_cls_sub_frm, text = "Cancel", command = lambda: print(""))
            self.vid_cls_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.vid_cls_can_btn.grid_remove()

        # postprocessing for images
        if "img_pst" in processes:
            self.img_pst_frm = customtkinter.CTkFrame(master=self.progress_top_level_window)
            self.img_pst_frm.grid(row=4, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            img_pst_ttl_txt = [f'Postprocessing{img_pst_extra_string}...', f'Postprocesado{img_pst_extra_string}...', f'Post-traitement des{img_pst_extra_string}...']
            self.img_pst_ttl = customtkinter.CTkLabel(self.img_pst_frm, text=img_pst_ttl_txt[lang_idx], font = ttl_font)
            self.img_pst_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.img_pst_sub_frm = customtkinter.CTkFrame(master=self.img_pst_frm)
            self.img_pst_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.img_pst_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.img_pst_pbr = customtkinter.CTkProgressBar(self.img_pst_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.img_pst_pbr.set(0)
            self.img_pst_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.img_pst_per = customtkinter.CTkLabel(self.img_pst_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.img_pst_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.img_pst_wai_lbl = customtkinter.CTkLabel(self.img_pst_sub_frm, height = lbl_height, text=in_queue_txt[lang_idx])
            self.img_pst_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.img_pst_ela_lbl = customtkinter.CTkLabel(self.img_pst_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[lang_idx]}:")
            self.img_pst_ela_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_pst_ela_lbl.grid_remove()
            self.img_pst_ela_val = customtkinter.CTkLabel(self.img_pst_sub_frm, height = lbl_height, text=f"")
            self.img_pst_ela_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.img_pst_ela_val.grid_remove()
            self.img_pst_rem_lbl = customtkinter.CTkLabel(self.img_pst_sub_frm, height = lbl_height, text=f"{remaining_time_txt[lang_idx]}:")
            self.img_pst_rem_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.img_pst_rem_lbl.grid_remove()
            self.img_pst_rem_val = customtkinter.CTkLabel(self.img_pst_sub_frm, height = lbl_height, text=f"")
            self.img_pst_rem_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.img_pst_rem_val.grid_remove()
            self.img_pst_can_btn = CancelButton(master = self.img_pst_sub_frm, text = "Cancel", command = lambda: print(""))
            self.img_pst_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.img_pst_can_btn.grid_remove()

        # postprocessing for videos
        if "vid_pst" in processes:
            self.vid_pst_frm = customtkinter.CTkFrame(master=self.progress_top_level_window)
            self.vid_pst_frm.grid(row=5, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            vid_pst_ttl_txt = [f'Postprocessing{vid_pst_extra_string}...', f'Postprocesado{vid_pst_extra_string}...', f'Post-traitement des{vid_pst_extra_string}...']
            self.vid_pst_ttl = customtkinter.CTkLabel(self.vid_pst_frm, text=vid_pst_ttl_txt[lang_idx], font = ttl_font)
            self.vid_pst_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.vid_pst_sub_frm = customtkinter.CTkFrame(master=self.vid_pst_frm)
            self.vid_pst_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.vid_pst_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.vid_pst_pbr = customtkinter.CTkProgressBar(self.vid_pst_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.vid_pst_pbr.set(0)
            self.vid_pst_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.vid_pst_per = customtkinter.CTkLabel(self.vid_pst_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.vid_pst_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.vid_pst_wai_lbl = customtkinter.CTkLabel(self.vid_pst_sub_frm, height = lbl_height, text=in_queue_txt[lang_idx])
            self.vid_pst_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.vid_pst_ela_lbl = customtkinter.CTkLabel(self.vid_pst_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[lang_idx]}:")
            self.vid_pst_ela_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_pst_ela_lbl.grid_remove()
            self.vid_pst_ela_val = customtkinter.CTkLabel(self.vid_pst_sub_frm, height = lbl_height, text=f"")
            self.vid_pst_ela_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.vid_pst_ela_val.grid_remove()
            self.vid_pst_rem_lbl = customtkinter.CTkLabel(self.vid_pst_sub_frm, height = lbl_height, text=f"{remaining_time_txt[lang_idx]}:")
            self.vid_pst_rem_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.vid_pst_rem_lbl.grid_remove()
            self.vid_pst_rem_val = customtkinter.CTkLabel(self.vid_pst_sub_frm, height = lbl_height, text=f"")
            self.vid_pst_rem_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.vid_pst_rem_val.grid_remove()
            self.vid_pst_can_btn = CancelButton(master = self.vid_pst_sub_frm, text = "Cancel", command = lambda: print(""))
            self.vid_pst_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.vid_pst_can_btn.grid_remove()

        # plotting can only be done for images
        if "plt" in processes:
            self.plt_frm = customtkinter.CTkFrame(master=self.progress_top_level_window)
            self.plt_frm.grid(row=6, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe")
            plt_ttl_txt = [f'Creating graphs...', f'Creando gráficos...', f'Création des graphiques...']
            self.plt_ttl = customtkinter.CTkLabel(self.plt_frm, text=plt_ttl_txt[lang_idx], font = ttl_font)
            self.plt_ttl.grid(row=0, padx=self.padx_progress_window * 2, pady=(self.pady_progress_window, 0), columnspan = 2, sticky="nsw")
            self.plt_sub_frm = customtkinter.CTkFrame(master=self.plt_frm)
            self.plt_sub_frm.grid(row=1, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nswe", ipady=self.pady_progress_window/2)
            self.plt_sub_frm.columnconfigure(0, weight=1, minsize=300 * scale_factor)
            self.plt_pbr = customtkinter.CTkProgressBar(self.plt_sub_frm, orientation="horizontal", height=pbr_height, corner_radius=5, width=1)
            self.plt_pbr.set(0)
            self.plt_pbr.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="nsew")
            self.plt_per = customtkinter.CTkLabel(self.plt_sub_frm, text=f" 0% ", height=5, fg_color=("#949BA2", "#4B4D50"), text_color="white")
            self.plt_per.grid(row=0, padx=self.padx_progress_window, pady=self.pady_progress_window, sticky="")
            self.plt_wai_lbl = customtkinter.CTkLabel(self.plt_sub_frm, height = lbl_height, text=in_queue_txt[lang_idx])
            self.plt_wai_lbl.grid(row=1, padx=self.padx_progress_window, pady=0, sticky="nsew")
            self.plt_ela_lbl = customtkinter.CTkLabel(self.plt_sub_frm, height = lbl_height, text=f"{elapsed_time_txt[lang_idx]}:")
            self.plt_ela_lbl.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.plt_ela_lbl.grid_remove()
            self.plt_ela_val = customtkinter.CTkLabel(self.plt_sub_frm, height = lbl_height, text=f"")
            self.plt_ela_val.grid(row=2, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.plt_ela_val.grid_remove()
            self.plt_rem_lbl = customtkinter.CTkLabel(self.plt_sub_frm, height = lbl_height, text=f"{remaining_time_txt[lang_idx]}:")
            self.plt_rem_lbl.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nsw")
            self.plt_rem_lbl.grid_remove()
            self.plt_rem_val = customtkinter.CTkLabel(self.plt_sub_frm, height = lbl_height, text=f"")
            self.plt_rem_val.grid(row=3, padx=self.padx_progress_window, pady=0, sticky="nse")     
            self.plt_rem_val.grid_remove()
            self.plt_can_btn = CancelButton(master = self.plt_sub_frm, text = "Cancel", command = lambda: print(""))
            self.plt_can_btn.grid(row=7, padx=self.padx_progress_window, pady=(self.pady_progress_window, 0), sticky="ns")
            self.plt_can_btn.grid_remove()

        self.progress_top_level_window.update()

    def update_values(self,
                      process,
                      status,
                      cur_it = 1,
                      tot_it = 1,
                      time_ela = "",
                      time_rem = "",
                      speed = "",
                      hware = "",
                      cancel_func = lambda: print(""),
                      extracting_frames_txt = ["Extracting frames...     ",
                                               "Extrayendo fotogramas...     ",
                                               "Extraction des trames..."],
                      frame_video_choice = "frame"):

        # language settings
        algorithm_starting_txt = ["Algorithm is starting up...", 'El algoritmo está arrancando...', 'Algorithme en démarrage...']
        smoothing_txt = ["Smoothing predictions...", 'Suavizar las predicciones...', 'Lissage de prédictions...']
        image_per_second_txt = ["Images per second:", "Imágenes por segundo:", "Images par seconde:"]
        seconds_per_image_txt = ["Seconds per image:", "Segundos por imagen:", "Secondes par image:"]
        animals_per_second_txt = ["Animals per second:", "Animales por segundo:", "Animaux par seconde:"]
        seconds_per_animal_txt = ["Seconds per animal:", "Segundos por animal:", "Secondes par animal:"]
        frames_per_second_txt = ["Frames per second:", "Fotogramas por segundo:"], "Trames par seconde:"
        seconds_per_frame_txt = ["Seconds per frame:", "Segundos por fotograma:", "Secondes par trame:"]
        videos_per_second_txt = ["Videos per second:", "Vídeos por segundo:", "Vidéos par seconde:"]
        seconds_per_video_txt = ["Seconds per video:", "Segundos por vídeo:", "Secondes par vidéo:"]
        processing_videos_txt = ["Processing video:", "Procesando vídeo:", "Traitement du vidéo:"]
        processing_frames_txt = ["Processing frame:", "Procesando fotograma:", "Traitement de la trame:"]
        starting_up_txt = ["Starting up...", "Arrancando...", "Démarrage..."]

        # detection of images
        if process == "img_det":
            if status == "load":
                self.img_det_wai_lbl.configure(text = algorithm_starting_txt[lang_idx])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.img_det_wai_lbl.grid_remove()
                    self.img_det_num_lbl.grid()
                    self.img_det_num_val.grid()
                    self.img_det_ela_lbl.grid()
                    self.img_det_ela_val.grid()
                    self.img_det_rem_lbl.grid()
                    self.img_det_rem_val.grid()
                    self.img_det_spe_lbl.grid()
                    self.img_det_spe_val.grid()
                    self.img_det_hwa_lbl.grid()
                    self.img_det_hwa_val.grid()
                    self.img_det_can_btn.grid()
                    self.img_det_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.img_det_pbr.set(percentage)
                self.img_det_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.img_det_per.configure(fg_color=(green_primary, "#1F6BA5"))
                else:
                    self.img_det_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.img_det_num_val.configure(text = f"{cur_it} of {tot_it}")
                self.img_det_ela_val.configure(text = time_ela)
                self.img_det_rem_val.configure(text = time_rem)
                self.img_det_spe_lbl.configure(text = image_per_second_txt[lang_idx] if "it/s" in speed else seconds_per_image_txt[lang_idx])
                parsed_speed = speed.replace("it/s", "").replace("s/it", "")
                self.img_det_spe_val.configure(text = parsed_speed)
                self.img_det_hwa_val.configure(text = hware)
            elif status == "done":
                self.img_det_num_lbl.grid_remove()
                self.img_det_num_val.grid_remove()
                self.img_det_rem_lbl.grid_remove()
                self.img_det_rem_val.grid_remove()
                self.img_det_hwa_lbl.grid_remove()
                self.img_det_hwa_val.grid_remove()
                self.img_det_can_btn.grid_remove()
                self.img_det_ela_val.grid_remove()
                self.img_det_ela_lbl.grid_remove()
                self.img_det_spe_lbl.grid_remove()
                self.img_det_spe_val.grid_remove()
                self.img_det_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.img_det_per.grid_configure(pady=(self.pady_progress_window, 0))
                
        # classification of images
        elif process == "img_cls":
            if status == "load":
                self.img_cls_wai_lbl.configure(text = algorithm_starting_txt[lang_idx])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.img_cls_wai_lbl.grid_remove()
                    self.img_cls_num_lbl.grid()
                    self.img_cls_num_val.grid()
                    self.img_cls_ela_lbl.grid()
                    self.img_cls_ela_val.grid()
                    self.img_cls_rem_lbl.grid()
                    self.img_cls_rem_val.grid()
                    self.img_cls_spe_lbl.grid()
                    self.img_cls_spe_val.grid()
                    self.img_cls_hwa_lbl.grid()
                    self.img_cls_hwa_val.grid()
                    self.img_cls_can_btn.grid()
                    self.img_cls_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.img_cls_pbr.set(percentage)
                self.img_cls_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.img_cls_per.configure(fg_color=(green_primary, "#1F6BA5"))
                else:
                    self.img_cls_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.img_cls_num_val.configure(text = f"{cur_it} of {tot_it}")
                self.img_cls_ela_val.configure(text = time_ela)
                self.img_cls_rem_val.configure(text = time_rem)
                self.img_cls_spe_lbl.configure(text = animals_per_second_txt[lang_idx] if "it/s" in speed else seconds_per_animal_txt[lang_idx])
                parsed_speed = speed.replace("it/s", "").replace("s/it", "")
                self.img_cls_spe_val.configure(text = parsed_speed)
                self.img_cls_hwa_val.configure(text = hware)
            elif status == "smoothing":
                self.img_cls_num_lbl.grid_remove()
                self.img_cls_num_val.grid_remove()
                self.img_cls_rem_lbl.grid_remove()
                self.img_cls_rem_val.grid_remove()
                self.img_cls_hwa_lbl.grid_remove()
                self.img_cls_hwa_val.grid_remove()
                self.img_cls_can_btn.grid_remove()
                self.img_cls_ela_val.grid_remove()
                self.img_cls_ela_lbl.grid_remove()
                self.img_cls_spe_lbl.grid_remove()
                self.img_cls_spe_val.grid_remove()
                self.img_cls_wai_lbl.grid()
                self.img_cls_wai_lbl.configure(text = smoothing_txt[lang_idx])
            elif status == "done":
                self.img_cls_num_lbl.grid_remove()
                self.img_cls_num_val.grid_remove()
                self.img_cls_rem_lbl.grid_remove()
                self.img_cls_rem_val.grid_remove()
                self.img_cls_hwa_lbl.grid_remove()
                self.img_cls_hwa_val.grid_remove()
                self.img_cls_can_btn.grid_remove()
                self.img_cls_ela_val.grid_remove()
                self.img_cls_ela_lbl.grid_remove()
                self.img_cls_spe_lbl.grid_remove()
                self.img_cls_spe_val.grid_remove()
                self.img_cls_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.img_cls_per.grid_configure(pady=(self.pady_progress_window, 0))

        # detection of videos
        if process == "vid_det":
            if status == "load":
                self.vid_det_wai_lbl.configure(text = algorithm_starting_txt[lang_idx])
                self.just_shown_load_screen = True
            elif status == "extracting frames":
                self.vid_det_wai_lbl.configure(text = extracting_frames_txt[lang_idx])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.vid_det_wai_lbl.grid_remove()
                    self.vid_det_num_lbl.grid()
                    self.vid_det_num_val.grid()
                    self.vid_det_ela_lbl.grid()
                    self.vid_det_ela_val.grid()
                    self.vid_det_rem_lbl.grid()
                    self.vid_det_rem_val.grid()
                    self.vid_det_spe_lbl.grid()
                    self.vid_det_spe_val.grid()
                    self.vid_det_hwa_lbl.grid()
                    self.vid_det_hwa_val.grid()
                    self.vid_det_can_btn.grid()
                    self.vid_det_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.vid_det_pbr.set(percentage)
                self.vid_det_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.vid_det_per.configure(fg_color=(green_primary, "#1F6BA5"))
                else:
                    self.vid_det_per.configure(fg_color=("#949BA2", "#4B4D50"))
                if frame_video_choice == "frame":
                    self.vid_det_num_lbl.configure(text = processing_frames_txt[lang_idx])
                else:
                    self.vid_det_num_lbl.configure(text = processing_videos_txt[lang_idx])
                self.vid_det_num_val.configure(text = f"{cur_it} of {tot_it}")
                self.vid_det_ela_val.configure(text = time_ela)
                self.vid_det_rem_val.configure(text = time_rem)
                if frame_video_choice == "frame":
                    self.vid_det_spe_lbl.configure(text = frames_per_second_txt[lang_idx] if "it/s" in speed else seconds_per_frame_txt[lang_idx])
                else:
                    self.vid_det_spe_lbl.configure(text = videos_per_second_txt[lang_idx] if "it/s" in speed else seconds_per_video_txt[lang_idx])
                parsed_speed = speed.replace("it/s", "").replace("s/it", "")
                self.vid_det_spe_val.configure(text = parsed_speed)
                self.vid_det_hwa_val.configure(text = hware)
            elif status == "done":
                self.vid_det_num_lbl.grid_remove()
                self.vid_det_num_val.grid_remove()
                self.vid_det_rem_lbl.grid_remove()
                self.vid_det_rem_val.grid_remove()
                self.vid_det_hwa_lbl.grid_remove()
                self.vid_det_hwa_val.grid_remove()
                self.vid_det_ela_val.grid_remove()
                self.vid_det_ela_lbl.grid_remove()
                self.vid_det_spe_lbl.grid_remove()
                self.vid_det_spe_val.grid_remove()
                self.vid_det_can_btn.grid_remove()
                self.vid_det_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.vid_det_per.grid_configure(pady=(self.pady_progress_window, 0))

        # classification of videos
        elif process == "vid_cls":
            if status == "load":
                self.vid_cls_wai_lbl.configure(text = algorithm_starting_txt[lang_idx])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.vid_cls_wai_lbl.grid_remove()
                    self.vid_cls_num_lbl.grid()
                    self.vid_cls_num_val.grid()
                    self.vid_cls_ela_lbl.grid()
                    self.vid_cls_ela_val.grid()
                    self.vid_cls_rem_lbl.grid()
                    self.vid_cls_rem_val.grid()
                    self.vid_cls_spe_lbl.grid()
                    self.vid_cls_spe_val.grid()
                    self.vid_cls_hwa_lbl.grid()
                    self.vid_cls_hwa_val.grid()
                    self.vid_cls_can_btn.grid()
                    self.vid_cls_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.vid_cls_pbr.set(percentage)
                self.vid_cls_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.vid_cls_per.configure(fg_color=(green_primary, "#1F6BA5"))
                else:
                    self.vid_cls_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.vid_cls_num_val.configure(text = f"{cur_it} of {tot_it}")
                self.vid_cls_ela_val.configure(text = time_ela)
                self.vid_cls_rem_val.configure(text = time_rem)
                self.vid_cls_spe_lbl.configure(text = animals_per_second_txt[lang_idx] if "it/s" in speed else seconds_per_animal_txt[lang_idx])
                parsed_speed = speed.replace("it/s", "").replace("s/it", "")
                self.vid_cls_spe_val.configure(text = parsed_speed)
                self.vid_cls_hwa_val.configure(text = hware)
            elif status == "smoothing":
                self.vid_cls_num_lbl.grid_remove()
                self.vid_cls_num_val.grid_remove()
                self.vid_cls_rem_lbl.grid_remove()
                self.vid_cls_rem_val.grid_remove()
                self.vid_cls_hwa_lbl.grid_remove()
                self.vid_cls_hwa_val.grid_remove()
                self.vid_cls_can_btn.grid_remove()
                self.vid_cls_ela_val.grid_remove()
                self.vid_cls_ela_lbl.grid_remove()
                self.vid_cls_spe_lbl.grid_remove()
                self.vid_cls_spe_val.grid_remove()
                self.vid_cls_wai_lbl.grid()
                self.vid_cls_wai_lbl.configure(text = smoothing_txt[lang_idx])
            elif status == "done":
                self.vid_cls_num_lbl.grid_remove()
                self.vid_cls_num_val.grid_remove()
                self.vid_cls_rem_lbl.grid_remove()
                self.vid_cls_rem_val.grid_remove()
                self.vid_cls_hwa_lbl.grid_remove()
                self.vid_cls_hwa_val.grid_remove()
                self.vid_cls_ela_val.grid_remove()
                self.vid_cls_ela_lbl.grid_remove()
                self.vid_cls_spe_lbl.grid_remove()
                self.vid_cls_spe_val.grid_remove()
                self.vid_cls_can_btn.grid_remove()
                self.vid_cls_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.vid_cls_per.grid_configure(pady=(self.pady_progress_window, 0))

        # postprocessing of images
        elif process == "img_pst":
            if status == "load":
                self.img_pst_wai_lbl.configure(text = starting_up_txt[lang_idx])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.img_pst_wai_lbl.grid_remove()
                    self.img_pst_ela_lbl.grid()
                    self.img_pst_ela_val.grid()
                    self.img_pst_rem_lbl.grid()
                    self.img_pst_rem_val.grid()
                    self.img_pst_can_btn.grid()
                    self.img_pst_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.img_pst_pbr.set(percentage)
                self.img_pst_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.img_pst_per.configure(fg_color=(green_primary, "#1F6BA5"))
                else:
                    self.img_pst_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.img_pst_ela_val.configure(text = time_ela)
                self.img_pst_rem_val.configure(text = time_rem)
            elif status == "done":
                self.img_pst_rem_lbl.grid_remove()
                self.img_pst_rem_val.grid_remove()
                self.img_pst_ela_val.grid_remove()
                self.img_pst_ela_lbl.grid_remove()
                self.img_pst_can_btn.grid_remove()
                self.img_pst_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.img_pst_per.grid_configure(pady=(self.pady_progress_window, 0))

        # postprocessing of videos
        elif process == "vid_pst":
            if status == "load":
                self.vid_pst_wai_lbl.configure(text = starting_up_txt[lang_idx])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.vid_pst_wai_lbl.grid_remove()
                    self.vid_pst_ela_lbl.grid()
                    self.vid_pst_ela_val.grid()
                    self.vid_pst_rem_lbl.grid()
                    self.vid_pst_rem_val.grid()
                    self.vid_pst_can_btn.grid()
                    self.vid_pst_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.vid_pst_pbr.set(percentage)
                self.vid_pst_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.vid_pst_per.configure(fg_color=(green_primary, "#1F6BA5"))
                else:
                    self.vid_pst_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.vid_pst_ela_val.configure(text = time_ela)
                self.vid_pst_rem_val.configure(text = time_rem)
            elif status == "done":
                self.vid_pst_rem_lbl.grid_remove()
                self.vid_pst_rem_val.grid_remove()
                self.vid_pst_ela_val.grid_remove()
                self.vid_pst_ela_lbl.grid_remove()
                self.vid_pst_can_btn.grid_remove()
                self.vid_pst_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.vid_pst_per.grid_configure(pady=(self.pady_progress_window, 0))

        # postprocessing of videos
        elif process == "plt":
            if status == "load":
                self.plt_wai_lbl.configure(text = starting_up_txt[lang_idx])
                self.just_shown_load_screen = True
            elif status == "running":
                if self.just_shown_load_screen:
                    self.plt_wai_lbl.grid_remove()
                    self.plt_ela_lbl.grid()
                    self.plt_ela_val.grid()
                    self.plt_rem_lbl.grid()
                    self.plt_rem_val.grid()
                    self.plt_can_btn.grid()
                    self.plt_can_btn.configure(command = cancel_func)
                    self.just_shown_load_screen = False
                percentage = (cur_it / tot_it)
                self.plt_pbr.set(percentage)
                self.plt_per.configure(text = f" {round(percentage * 100)}% ")
                if percentage > 0.5:
                    self.plt_per.configure(fg_color=(green_primary, "#1F6BA5"))
                else:
                    self.plt_per.configure(fg_color=("#949BA2", "#4B4D50"))
                self.plt_ela_val.configure(text = time_ela)
                self.plt_rem_val.configure(text = time_rem)
            elif status == "done":
                self.plt_rem_lbl.grid_remove()
                self.plt_rem_val.grid_remove()
                self.plt_ela_val.grid_remove()
                self.plt_ela_lbl.grid_remove()
                self.plt_can_btn.grid_remove()
                self.plt_pbr.grid_configure(pady=(self.pady_progress_window, 0))
                self.plt_per.grid_configure(pady=(self.pady_progress_window, 0))

        # update screen
        self.progress_top_level_window.update()

    def open(self):
        self.progress_top_level_window.deiconify()

    def close(self):
        self.progress_top_level_window.destroy()

# refresh dropdown menu options
def update_dpd_options(dpd, master, var, options, cmd, row, lbl, from_lang_idx):

    # recreate new option menu with updated options
    dpd.grid_forget()
    index = options[from_lang_idx].index(var.get()) # get dpd index
    var.set(options[lang_idx][index]) # set to previous index
    if cmd:
        dpd = OptionMenu(master, var, *options[lang_idx], command=cmd)
    else:
        dpd = OptionMenu(master, var, *options[lang_idx])
    dpd.configure(width=1)
    dpd.grid(row=row, column=1, sticky='nesw', padx=5)

    # remove detection dropdown for speciesnet
    if (var_cls_model.get() == "Global - SpeciesNet - Google") and \
         ("detec" in lbl['text']):
        dpd.grid_remove()

    # give it same state as its label
    dpd.configure(state = str(lbl['state']))

# special refresh function for the model seleciton dropdown in simple mode because customtkinter works a bit different
def update_sim_mdl_dpd():
    global sim_mdl_dpd
    sim_mdl_dpd.grid_forget()
    sim_mdl_dpd = customtkinter.CTkOptionMenu(sim_mdl_frm, values=sim_dpd_options_cls_model[lang_idx], command=sim_mdl_dpd_callback, width = 1)
    sim_mdl_dpd.set(sim_dpd_options_cls_model[lang_idx][dpd_options_cls_model[lang_idx].index(var_cls_model.get())])
    sim_mdl_dpd.grid(row=1, column=0, padx=PADX, pady=(PADY/4, PADY), sticky="nswe", columnspan = 2)

# refresh ent texts
def update_ent_text(var, string):
    if var.get() == "":
        return
    if no_user_input(var):
        original_state = str(var['state'])
        var.configure(state=NORMAL, fg='grey')
        var.delete(0, tk.END)
        var.insert(0, string)
        var.configure(state=original_state)

# check next language to print on button when program starts
def set_lang_buttons(lang_idx):
    from_lang_idx = lang_idx
    to_lang_idx = 0 if from_lang_idx + 1 >= len(languages_available) else from_lang_idx + 1
    to_lang = languages_available[to_lang_idx]
    sim_btn_switch_lang.configure(text = f"{to_lang}")
    adv_btn_switch_lang.configure(text = f"{to_lang}")

# change language
def set_language():
    # calculate indeces
    global lang_idx
    from_lang_idx = lang_idx
    to_lang_idx = 0 if from_lang_idx + 1 >= len(languages_available) else from_lang_idx + 1
    next_lang_idx = 0 if to_lang_idx + 1 >= len(languages_available) else to_lang_idx + 1

    # log
    print(f"EXECUTED : {sys._getframe().f_code.co_name}({locals()})\n")

    # set the global variable to the new language
    lang_idx = to_lang_idx
    write_global_vars({"lang_idx": lang_idx})

    # update tab texts
    tabControl.tab(deploy_tab, text=deploy_tab_text[lang_idx])
    tabControl.tab(help_tab, text=help_tab_text[lang_idx])
    tabControl.tab(about_tab, text=about_tab_text[lang_idx])

    # update texts of deploy tab
    fst_step.configure(text=" " + fst_step_txt[lang_idx] + " ")
    lbl_choose_folder.configure(text=lbl_choose_folder_txt[lang_idx])
    btn_choose_folder.configure(text=browse_txt[lang_idx])
    snd_step.configure(text=" " + snd_step_txt[lang_idx] + " ")
    lbl_model.configure(text=lbl_model_txt[lang_idx])
    update_dpd_options(dpd_model, snd_step, var_det_model, dpd_options_model, model_options, row_model, lbl_model, from_lang_idx)
    lbl_exclude_subs.configure(text=lbl_exclude_subs_txt[lang_idx])
    lbl_use_custom_img_size_for_deploy.configure(text=lbl_use_custom_img_size_for_deploy_txt[lang_idx])
    lbl_image_size_for_deploy.configure(text=lbl_image_size_for_deploy_txt[lang_idx])
    update_ent_text(ent_image_size_for_deploy, f"{eg_txt[lang_idx]}: 640")
    lbl_abs_paths.configure(text=lbl_abs_paths_txt[lang_idx])
    lbl_disable_GPU.configure(text=lbl_disable_GPU_txt[lang_idx])
    lbl_process_img.configure(text=lbl_process_img_txt[lang_idx])
    lbl_cls_model.configure(text=lbl_cls_model_txt[lang_idx])
    update_dpd_options(dpd_cls_model, snd_step, var_cls_model, dpd_options_cls_model, model_cls_animal_options, row_cls_model, lbl_cls_model, from_lang_idx)
    cls_frame.configure(text=" ↳ " + cls_frame_txt[lang_idx] + " ")
    lbl_model_info.configure(text = "     " + lbl_model_info_txt[lang_idx])
    btn_model_info.configure(text=show_txt[lang_idx])
    lbl_choose_classes.configure(text = "     " + lbl_choose_classes_txt[lang_idx])
    btn_choose_classes.configure(text = select_txt[lang_idx])
    lbl_cls_detec_thresh.configure(text="     " + lbl_cls_detec_thresh_txt[lang_idx])
    lbl_cls_class_thresh.configure(text="     " + lbl_cls_class_thresh_txt[lang_idx])
    lbl_smooth_cls_animal.configure(text="     " + lbl_smooth_cls_animal_txt[lang_idx])
    img_frame.configure(text=" ↳ " + img_frame_txt[lang_idx] + " ")
    lbl_use_checkpnts.configure(text="     " + lbl_use_checkpnts_txt[lang_idx])
    lbl_checkpoint_freq.configure(text="        ↳ " + lbl_checkpoint_freq_txt[lang_idx])
    update_ent_text(ent_checkpoint_freq, f"{eg_txt[lang_idx]}: 500")
    lbl_cont_checkpnt.configure(text="     " + lbl_cont_checkpnt_txt[lang_idx])
    lbl_process_vid.configure(text=lbl_process_vid_txt[lang_idx])
    vid_frame.configure(text=" ↳ " + vid_frame_txt[lang_idx] + " ")
    lbl_not_all_frames.configure(text="     " + lbl_not_all_frames_txt[lang_idx])
    lbl_nth_frame.configure(text="        ↳ " + lbl_nth_frame_txt[lang_idx])
    update_ent_text(ent_nth_frame, f"{eg_txt[lang_idx]}: 1")
    btn_start_deploy.configure(text=btn_start_deploy_txt[lang_idx])
    trd_step.configure(text=" " + trd_step_txt[lang_idx] + " ")
    lbl_hitl_main.configure(text=lbl_hitl_main_txt[lang_idx])
    btn_hitl_main.configure(text=["Start", "Iniciar", "Démarrer"][lang_idx])
    fth_step.configure(text=" " + fth_step_txt[lang_idx] + " ")
    lbl_output_dir.configure(text=lbl_output_dir_txt[lang_idx])
    btn_output_dir.configure(text=browse_txt[lang_idx])
    lbl_separate_files.configure(text=lbl_separate_files_txt[lang_idx])
    sep_frame.configure(text=" ↳ " + sep_frame_txt[lang_idx] + " ")
    lbl_file_placement.configure(text="     " + lbl_file_placement_txt[lang_idx])
    rad_file_placement_move.configure(text=["Copy", "Copiar", "Copier"][lang_idx])
    rad_file_placement_copy.configure(text=["Move", "Mover", "Déplacer"][lang_idx])
    lbl_sep_conf.configure(text="     " + lbl_sep_conf_txt[lang_idx])
    lbl_keep_series.configure(text=lbl_keep_series_txt[lang_idx])
    keep_series_frame.configure(text=" ↳ " + keep_series_frame_txt[lang_idx] + " ")
    lbl_keep_series_seconds.configure(text="     " + lbl_keep_series_seconds_txt[lang_idx])
    lbl_keep_series_species.configure(text="     " + lbl_keep_series_species_txt[lang_idx])
    btn_keep_series_species.configure(text=select_txt[lang_idx])
    try:
        if len(global_vars.get('var_keep_series_species', []) or []) == 0:
            dsp_keep_series_species.configure(text=["Any", "Cualquiera", "Toutes"][lang_idx])
        else:
            dsp_keep_series_species.configure(text=str(len(global_vars.get('var_keep_series_species', []))))
    except Exception:
        pass
    lbl_vis_files.configure(text=lbl_vis_files_txt[lang_idx])
    lbl_crp_files.configure(text=lbl_crp_files_txt[lang_idx])
    lbl_exp.configure(text=lbl_exp_txt[lang_idx])
    exp_frame.configure(text=" ↳ " + exp_frame_txt[lang_idx] + " ")
    vis_frame.configure(text=" ↳ " + vis_frame_txt[lang_idx] + " ")
    lbl_exp_format.configure(text="     " + lbl_exp_format_txt[lang_idx])
    lbl_plt.configure(text=lbl_plt_txt[lang_idx])
    lbl_thresh.configure(text=lbl_thresh_txt[lang_idx])
    btn_start_postprocess.configure(text=btn_start_postprocess_txt[lang_idx])
    lbl_vis_size.configure(text="        ↳ " + lbl_vis_size_txt[lang_idx])
    lbl_vis_bbox.configure(text="     " + lbl_vis_bbox_txt[lang_idx])
    lbl_vis_blur.configure(text="     " + lbl_vis_blur_txt[lang_idx])
    var_vis_size.set(dpd_options_vis_size[lang_idx][global_vars['var_vis_size_idx']])

    # update texts of help tab
    help_text.configure(state=NORMAL)
    help_text.delete('1.0', END)
    write_help_tab()

    # update texts of about tab
    about_text.configure(state=NORMAL)
    about_text.delete('1.0', END)
    write_about_tab()

    # top buttons
    adv_btn_switch_mode.configure(text = adv_btn_switch_mode_txt[lang_idx])
    sim_btn_switch_mode.configure(text = sim_btn_switch_mode_txt[lang_idx])
    sim_btn_switch_lang.configure(text = languages_available[next_lang_idx])
    adv_btn_switch_lang.configure(text = languages_available[next_lang_idx])
    adv_btn_sponsor.configure(text = adv_btn_sponsor_txt[lang_idx])
    sim_btn_sponsor.configure(text = adv_btn_sponsor_txt[lang_idx])
    adv_btn_reset_values.configure(text = adv_btn_reset_values_txt[lang_idx])
    sim_btn_reset_values.configure(text = adv_btn_reset_values_txt[lang_idx])

    # by addax text
    adv_abo_lbl.configure(text=adv_abo_lbl_txt[lang_idx])
    sim_abo_lbl.configure(text=adv_abo_lbl_txt[lang_idx])

    # simple mode
    sim_dir_lbl.configure(text = sim_dir_lbl_txt[lang_idx])
    sim_dir_btn.configure(text = browse_txt[lang_idx])
    sim_dir_pth.configure(text = sim_dir_pth_txt[lang_idx])
    sim_mdl_lbl.configure(text = sim_mdl_lbl_txt[lang_idx])
    update_sim_mdl_dpd()
    sim_spp_lbl.configure(text = sim_spp_lbl_txt[lang_idx])
    sim_run_btn.configure(text = sim_run_btn_txt[lang_idx])

# update frame states
def update_frame_states():
    # check dir validity
    if var_choose_folder.get() in ["", "/", "\\", ".", "~", ":"] or not os.path.isdir(var_choose_folder.get()):
        return
    if var_choose_folder.get() not in ["", "/", "\\", ".", "~", ":"] and os.path.isdir(var_choose_folder.get()):
        complete_frame(fst_step)
    else:
        enable_frame(fst_step)

    # check json files
    img_json = False
    path_to_image_json = os.path.join(var_choose_folder.get(), "image_recognition_file.json")
    if os.path.isfile(path_to_image_json):
        img_json = True
    vid_json = False
    if os.path.isfile(os.path.join(var_choose_folder.get(), "video_recognition_file.json")):
        vid_json = True
    
    # check if dir is already processed
    if img_json or vid_json:
        complete_frame(snd_step)
        enable_frame(fth_step)
    else:
        enable_frame(snd_step)
        disable_frame(fth_step)

    # check hitl status
    if img_json:
        status = get_hitl_var_in_json(path_to_image_json)
        if status == "never-started":
            enable_frame(trd_step)
            btn_hitl_main.configure(text = ["Start", "Iniciar", "Démarrer"][lang_idx])
        elif status == "in-progress":
            enable_frame(trd_step)
            btn_hitl_main.configure(text = ["Continue", "Continuar", "Continuer"][lang_idx])
        elif status == "done":
            complete_frame(trd_step)
    else:
        disable_frame(trd_step)
    
    # if in timelapse mode, always disable trd and fth step
    if timelapse_mode:
        disable_frame(trd_step)
        disable_frame(fth_step)    

# check if user entered text in entry widget
def no_user_input(var):
    if var.get() == "" or var.get().startswith("E.g.:") or var.get().startswith("Ejem.:"):
        return True
    else:
        return False

# show warning if not valid input
def invalid_value_warning(str, numeric = True):
    string = [f"You either entered an invalid value for the {str}, or none at all.", f"Ingresó un valor no válido para {str} o ninguno.",
              f"Vous avez soit saisi un valeur invalide pour {str}, ou aucune valeur du tout."][lang_idx] 
    if numeric:
        string += [" You can only enter numeric characters.", " Solo puede ingresar caracteres numéricos.", "Vous pouvez uniquement saisir des caractères numériques."][lang_idx]
    mb.showerror(invalid_value_txt[lang_idx], string)

# disable widgets based on row and col indeces
def disable_widgets_based_on_location(master, rows, cols):
    # list widgets to be removed
    widgets = []
    for row in rows:
        for col in cols:
            l = master.grid_slaves(row, col)
            for i in l:
                widgets.append(i)

    # remove widgets
    for widget in widgets:
        widget.configure(state=DISABLED)

# remove widgets based on row and col indexes
def remove_widgets_based_on_location(master, rows, cols):
    # list widgets to be removed
    widgets = []
    for row in rows:
        for col in cols:
            l = master.grid_slaves(row, col)
            for i in l:
                widgets.append(i)

    # remove widgets
    for widget in widgets:
        widget.grid_forget()

# create hyperlinks (thanks marvin from GitHub) 
class HyperlinkManager:
    def __init__(self, text):
        self.text = text
        self.text.tag_config("hyper", foreground=green_primary, underline=1)
        self.text.tag_bind("hyper", "<Enter>", self._enter)
        self.text.tag_bind("hyper", "<Leave>", self._leave)
        self.text.tag_bind("hyper", "<Button-1>", self._click)
        self.reset()

    def reset(self):
        self.links = {}

    def add(self, action):
        tag = "hyper-%d" % len(self.links)
        self.links[tag] = action
        return "hyper", tag

    def _enter(self, event):
        self.text.configure(cursor="hand2")

    def _leave(self, event):
        self.text.configure(cursor="")

    def _click(self, event):
        for tag in self.text.tag_names(CURRENT):
            if tag[:6] == "hyper-":
                self.links[tag]()
                return

# set cancel variable to true
def cancel():
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
    
    global cancel_var
    cancel_var = True

# set all children of frame to disabled state
def disable_widgets(frame):
    children = frame.winfo_children()
    for child in children:
        # labelframes have no state
        if child.winfo_class() != "Labelframe":
            child.configure(state=DISABLED)

# set all children of frame to normal state
def enable_widgets(frame):
    children = frame.winfo_children()
    for child in children:
        # labelframes have no state
        if child.winfo_class() != "Labelframe":
            child.configure(state=NORMAL)

# show warning for absolute paths option
shown_abs_paths_warning = True
def abs_paths_warning():
    global shown_abs_paths_warning
    if var_abs_paths.get() and shown_abs_paths_warning:
        mb.showinfo(warning_txt[lang_idx], ["It is not recommended to use absolute paths in the output file. Third party software (such "
                    "as Timelapse) will not be able to read the json file if the paths are absolute. Only enable"
                    " this option if you know what you are doing.",
                    "No se recomienda utilizar rutas absolutas en el archivo de salida. Software de terceros (como Timelapse"
                    ") no podrán leer el archivo json si las rutas son absolutas. Sólo active esta opción si sabe lo"
                    " que está haciendo.",
                    "Il n'est pas recommandé d'utiliser des chemin absolus dans le fichier de sortie. Des logiciels tiers (tel "
                    "que Timelapse) ne seront pas en mesure de lire le fichier JSON si les chemins sont absolus. N'activer cette "
                    "option que si vous savez ce que vous faites."][lang_idx])
        shown_abs_paths_warning = False

# toggle image size entry box
def toggle_image_size_for_deploy():
    if var_use_custom_img_size_for_deploy.get():
        lbl_image_size_for_deploy.grid(row=row_image_size_for_deploy, sticky='nesw', pady=2)
        ent_image_size_for_deploy.grid(row=row_image_size_for_deploy, column=1, sticky='nesw', padx=5)
    else:
        lbl_image_size_for_deploy.grid_remove()
        ent_image_size_for_deploy.grid_remove()
    resize_canvas_to_content()

# toggle separation subframe
def toggle_sep_frame():
    if var_separate_files.get():
        sep_frame.grid(row=sep_frame_row, column=0, columnspan=2, sticky = 'ew')
        enable_widgets(sep_frame)
        sep_frame.configure(fg='black')
        # ensure nested keep-series options reflect current checkbox state
        try:
            toggle_keep_series_frame()
        except Exception:
            pass
    else:
        # hide nested keep-series options too
        try:
            keep_series_frame.grid_forget()
        except Exception:
            pass
        disable_widgets(sep_frame)
        sep_frame.configure(fg='grey80')
        sep_frame.grid_forget()
    resize_canvas_to_content()

# toggle keep series subframe
def toggle_keep_series_frame():
    # only relevant when separation is enabled
    if var_separate_files.get() and var_keep_series.get():
        keep_series_frame.grid(row=keep_series_frame_row, column=0, columnspan=2, sticky = 'ew')
        enable_widgets(keep_series_frame)
        keep_series_frame.configure(fg='black')
    else:
        disable_widgets(keep_series_frame)
        keep_series_frame.configure(fg='grey80')
        keep_series_frame.grid_forget()
    resize_canvas_to_content()

# toggle export subframe
def toggle_exp_frame():
    if var_exp.get() and lbl_exp.cget("state") == "normal":
        exp_frame.grid(row=exp_frame_row, column=0, columnspan=2, sticky = 'ew')
        enable_widgets(exp_frame)
        exp_frame.configure(fg='black')
    else:
        disable_widgets(exp_frame)
        exp_frame.configure(fg='grey80')
        exp_frame.grid_forget()
    resize_canvas_to_content()

# toggle visualization subframe
def toggle_vis_frame():
    if var_vis_files.get() and lbl_vis_files.cget("state") == "normal":
        vis_frame.grid(row=vis_frame_row, column=0, columnspan=2, sticky = 'ew')
        enable_widgets(vis_frame)
        vis_frame.configure(fg='black')
    else:
        disable_widgets(vis_frame)
        vis_frame.configure(fg='grey80')
        vis_frame.grid_forget()
    resize_canvas_to_content()

# on checkbox change
def on_chb_smooth_cls_animal_change():
    write_model_vars(new_values={"var_smooth_cls_animal": var_smooth_cls_animal.get()})
    if var_smooth_cls_animal.get():
        mb.showinfo(information_txt[lang_idx], ["This feature averages confidence scores to avoid noise. Note that it assumes a single species per "
                                               "sequence or video and should therefore only be used if multi-species sequences are rare. It does not"
                                               " affect detections of vehicles or people alongside animals.", "Esta función promedia las puntuaciones "
                                               "de confianza para evitar el ruido. Tenga en cuenta que asume una única especie por secuencia o vídeo "
                                               "y, por lo tanto, sólo debe utilizarse si las secuencias multiespecíficas son poco frecuentes. No afecta"
                                               " a las detecciones de vehículos o personas junto a animales.",
                                               "Cette fonctionnalité fait la moyenne les scores de confiance pour éliminer du aberrations statistiques. "
                                               "Notez qu'il est considéré qu'une seule espèce par séquence ou vidéo est présente et devrait donc uniquement "
                                               "être utilisé si les séquences multi-espèces sont rares. Cela n'affecte pas les détections de véhicules ou "
                                               "de personnnes à côté des animaux."][lang_idx])

# toggle classification subframe
def toggle_cls_frame(): 
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # check the state of snd_step
    snd_step_enabled = False if snd_step.cget('fg') == 'grey80' else True

    # only enable cls_frame if snd_step is also enabled and user didn't choose None
    if var_cls_model.get() not in none_txt and snd_step_enabled:
        cls_frame.grid(row=cls_frame_row, column=0, columnspan=2, sticky = 'ew')
        enable_widgets(cls_frame)
        toggle_checkpoint_freq()
        cls_frame.configure(fg='black')
    else:
        disable_widgets(cls_frame)
        cls_frame.configure(fg='grey80')
        cls_frame.grid_forget()
    resize_canvas_to_content()

# toggle image subframe
def toggle_img_frame():
    if var_process_img.get():
        img_frame.grid(row=img_frame_row, column=0, columnspan=2, sticky = 'ew')
        enable_widgets(img_frame)
        toggle_checkpoint_freq()
        img_frame.configure(fg='black')
    else:
        disable_widgets(img_frame)
        img_frame.configure(fg='grey80')
        img_frame.grid_forget()
    resize_canvas_to_content()

# toggle video subframe
def toggle_vid_frame():
    if var_process_vid.get():
        vid_frame.grid(row=vid_frame_row, column=0, columnspan=2, sticky='ew')
        enable_widgets(vid_frame)
        toggle_nth_frame()
        vid_frame.configure(fg='black')
    else:
        disable_widgets(vid_frame)
        vid_frame.configure(fg='grey80')
        vid_frame.grid_forget()
    resize_canvas_to_content()

# convert frame to completed
def complete_frame(frame):
    global check_mark_one_row
    global check_mark_two_rows

    # check which frame 
    any_step = frame.cget('text').startswith(f' {step_txt[lang_idx]}')
    fst_step = frame.cget('text').startswith(f' {step_txt[lang_idx]} 1')
    snd_step = frame.cget('text').startswith(f' {step_txt[lang_idx]} 2')
    trd_step = frame.cget('text').startswith(f' {step_txt[lang_idx]} 3')
    fth_step = frame.cget('text').startswith(f' {step_txt[lang_idx]} 4')

    # adjust frames
    frame.configure(relief = 'groove')
    if any_step:
        frame.configure(fg=green_primary)
    if snd_step:
        cls_frame.configure(relief = 'groove')
        img_frame.configure(relief = 'groove')
        vid_frame.configure(relief = 'groove')

    if trd_step or fst_step:
        # add check mark
        lbl_check_mark = Label(frame, image=check_mark_one_row)
        lbl_check_mark.image = check_mark_one_row
        lbl_check_mark.grid(row=0, column=0, rowspan=15, columnspan=2, sticky='nesw')
        if trd_step:
            btn_hitl_main.configure(text=["New session?", "¿Nueva sesión?", "Nouvelle session?"][lang_idx], state = NORMAL)
            btn_hitl_main.lift()
        if fst_step:
            btn_choose_folder.configure(text=f"{change_folder_txt[lang_idx]}?", state = NORMAL)
            btn_choose_folder.lift()
            dsp_choose_folder.lift()
    
    else:
        # the rest
        if not any_step:
            # sub frames of fth_step only
            frame.configure(fg=green_primary)

        # add check mark
        lbl_check_mark = Label(frame, image=check_mark_two_rows)
        lbl_check_mark.image = check_mark_two_rows
        lbl_check_mark.grid(row=0, column=0, rowspan=15, columnspan=2, sticky='nesw')

        # add buttons
        btn_view_results = Button(master=frame, text=view_results_txt[lang_idx], width=1, command=lambda: view_results(frame))
        btn_view_results.grid(row=0, column=1, sticky='nesw', padx = 5)
        btn_uncomplete = Button(master=frame, text=again_txt[lang_idx], width=1, command=lambda: enable_frame(frame))
        btn_uncomplete.grid(row=1, column=1, sticky='nesw', padx = 5)

# enable a frame
def enable_frame(frame):
    uncomplete_frame(frame)
    enable_widgets(frame)

    # check which frame 
    any_step = frame.cget('text').startswith(f' {step_txt[lang_idx]}')
    fst_step = frame.cget('text').startswith(f' {step_txt[lang_idx]} 1')
    snd_step = frame.cget('text').startswith(f' {step_txt[lang_idx]} 2')
    trd_step = frame.cget('text').startswith(f' {step_txt[lang_idx]} 3')
    fth_step = frame.cget('text').startswith(f' {step_txt[lang_idx]} 4')

    # all frames
    frame.configure(relief = 'solid')
    if any_step:
        frame.configure(fg=green_primary)
    if snd_step:
        toggle_cls_frame()
        cls_frame.configure(relief = 'solid')
        toggle_img_frame()
        img_frame.configure(relief = 'solid')
        toggle_vid_frame()
        vid_frame.configure(relief = 'solid')
        toggle_image_size_for_deploy()
    if fth_step:
        toggle_sep_frame()
        toggle_keep_series_frame()
        toggle_exp_frame()
        toggle_vis_frame()
        sep_frame.configure(relief = 'solid')
        keep_series_frame.configure(relief = 'solid')
        exp_frame.configure(relief = 'solid')
        vis_frame.configure(relief = 'solid')

# remove checkmarks and complete buttons
def uncomplete_frame(frame):
    if not frame.cget('text').startswith(f' {step_txt[lang_idx]}'):
        # subframes in fth_step only
        frame.configure(fg='black')
    children = frame.winfo_children()
    for child in children:
        if child.winfo_class() == "Button" or child.winfo_class() == "Label":
            if child.cget('text') == again_txt[lang_idx] or child.cget('text') == view_results_txt[lang_idx] or child.cget('image') != "":
                child.grid_remove()

# disable a frame
def disable_frame(frame):
    uncomplete_frame(frame)
    disable_widgets(frame)
    # all frames
    frame.configure(fg='grey80')
    frame.configure(relief = 'flat')
    if frame.cget('text').startswith(f' {step_txt[lang_idx]} 2'):
        # snd_step only
        disable_widgets(cls_frame)
        cls_frame.configure(fg='grey80')
        cls_frame.configure(relief = 'flat')
        disable_widgets(img_frame)
        img_frame.configure(fg='grey80')
        img_frame.configure(relief = 'flat')
        disable_widgets(vid_frame)
        vid_frame.configure(fg='grey80')
        vid_frame.configure(relief = 'flat')
    if frame.cget('text').startswith(f' {step_txt[lang_idx]} 4'):
        # fth_step only
        disable_widgets(sep_frame)
        sep_frame.configure(fg='grey80')
        sep_frame.configure(relief = 'flat')
        disable_widgets(keep_series_frame)
        keep_series_frame.configure(fg='grey80')
        keep_series_frame.configure(relief='flat')
        disable_widgets(exp_frame)
        exp_frame.configure(fg='grey80')
        exp_frame.configure(relief = 'flat')
        disable_widgets(vis_frame)
        vis_frame.configure(fg='grey80')
        vis_frame.configure(relief = 'flat')

    
# check if checkpoint is present and set checkbox accordingly
def disable_chb_cont_checkpnt():
    if var_cont_checkpnt.get():
        var_cont_checkpnt.set(check_checkpnt())

# set minimum row size for all rows in a frame
def set_minsize_rows(frame):
    row_count = frame.grid_size()[1]
    for row in range(row_count):
        frame.grid_rowconfigure(row, minsize=minsize_rows)

# toggle state of checkpoint frequency
def toggle_checkpoint_freq():
    if var_use_checkpnts.get():
        lbl_checkpoint_freq.configure(state=NORMAL)
        ent_checkpoint_freq.configure(state=NORMAL)
    else:
        lbl_checkpoint_freq.configure(state=DISABLED)
        ent_checkpoint_freq.configure(state=DISABLED)

# toggle state of nth frame
def toggle_nth_frame():
    if var_not_all_frames.get():
        lbl_nth_frame.configure(state=NORMAL)
        ent_nth_frame.configure(state=NORMAL)
    else:
        lbl_nth_frame.configure(state=DISABLED)
        ent_nth_frame.configure(state=DISABLED)

# check required and maximum size of canvas and resize accordingly
def resize_canvas_to_content():
    root.update_idletasks()
    _, _, _, height_logo = advanc_main_frame.grid_bbox(0, 0)
    _, _, width_step1, height_step1 = deploy_scrollable_frame.grid_bbox(0, 1)
    _, _, _, height_step2 = deploy_scrollable_frame.grid_bbox(0, 2)
    _, _, width_step3, _ = deploy_scrollable_frame.grid_bbox(1, 1)
    canvas_required_height = height_step1 + height_step2
    canvas_required_width  = width_step1 + width_step3
    max_screen_height = root.winfo_screenheight()
    canvas_max_height = max_screen_height - height_logo - 300
    canvas_height = min(canvas_required_height, canvas_max_height, 800)
    deploy_canvas.configure(width = canvas_required_width, height = canvas_height)
    bg_height = (canvas_height + height_logo + ADV_EXTRA_GRADIENT_HEIGHT) * (1 / scale_factor)
    new_advanc_bg_image = customtkinter.CTkImage(PIL_sidebar, size=(ADV_WINDOW_WIDTH, bg_height))
    advanc_bg_image_label.configure(image=new_advanc_bg_image)

# functions to delete the grey text in the entry boxes for the...
# ... image size for deploy
image_size_for_deploy_init = True
def image_size_for_deploy_focus_in(_):
    global image_size_for_deploy_init
    if image_size_for_deploy_init and not var_image_size_for_deploy.get().isdigit():
        ent_image_size_for_deploy.delete(0, tk.END)
        ent_image_size_for_deploy.configure(fg='black')
    image_size_for_deploy_init = False

# ... checkpoint frequency
checkpoint_freq_init = True
def checkpoint_freq_focus_in(_):
    global checkpoint_freq_init
    if checkpoint_freq_init and not var_checkpoint_freq.get().isdigit():
        ent_checkpoint_freq.delete(0, tk.END)
        ent_checkpoint_freq.configure(fg='black')
    checkpoint_freq_init = False

# ... nth frame
nth_frame_init = True
def nth_frame_focus_in(_):
    global nth_frame_init
    if nth_frame_init and not var_nth_frame.get().isdigit():
        ent_nth_frame.delete(0, tk.END)
        ent_nth_frame.configure(fg='black')
    nth_frame_init = False

# check current status, switch to opposite and save
def switch_mode():

    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # load
    advanced_mode = load_global_vars()["advanced_mode"]

    # switch
    if advanced_mode:
        advanc_mode_win.withdraw()
        simple_mode_win.deiconify()         
    else:
        advanc_mode_win.deiconify()
        simple_mode_win.withdraw()

    # save
    write_global_vars({
        "advanced_mode": not advanced_mode
    })

def sponsor_project():
    webbrowser.open("https://addaxdatascience.com/addaxai/#donate")

class GreyTopButton(customtkinter.CTkButton):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color = (yellow_secondary, "#333333"),
                       hover_color = (yellow_tertiary, "#2B2B2B"),
                       text_color = ("black", "white"),
                       height = 10,
                       width = 140,
                       border_width=GREY_BUTTON_BORDER_WIDTH)

def reset_values():

    # set values
    var_thresh.set(global_vars['var_thresh_default'])
    var_det_model_path.set("")
    var_det_model_short.set("")
    var_exclude_subs.set(False)
    var_use_custom_img_size_for_deploy.set(False)
    var_image_size_for_deploy.set("")
    var_abs_paths.set(False)
    var_disable_GPU.set(False)
    var_process_img.set(False)
    var_use_checkpnts.set(False)
    var_checkpoint_freq.set("")
    var_cont_checkpnt.set(False)
    var_process_vid.set(False)
    var_not_all_frames.set(True)
    var_nth_frame.set("1")
    var_separate_files.set(False)
    var_keep_series.set(False)
    var_keep_series_seconds.set(5)
    global_vars['var_keep_series_species'] = []
    var_file_placement.set(2)
    var_sep_conf.set(False)
    var_vis_files.set(False)
    var_vis_size.set(dpd_options_vis_size[lang_idx][global_vars['var_vis_size_idx']])
    var_vis_bbox.set(False)
    var_vis_blur.set(False)
    var_crp_files.set(False)
    var_exp.set(True)
    var_exp_format.set(dpd_options_exp_format[lang_idx][global_vars['var_exp_format_idx']])
    
    write_global_vars({
        "var_det_model_idx": dpd_options_model[lang_idx].index(var_det_model.get()),
        "var_det_model_path": var_det_model_path.get(),
        "var_det_model_short": var_det_model_short.get(),
        "var_exclude_subs": var_exclude_subs.get(),
        "var_use_custom_img_size_for_deploy": var_use_custom_img_size_for_deploy.get(),
        "var_image_size_for_deploy": var_image_size_for_deploy.get() if var_image_size_for_deploy.get().isdigit() else "",
        "var_abs_paths": var_abs_paths.get(),
        "var_disable_GPU": var_disable_GPU.get(),
        "var_process_img": var_process_img.get(),
        "var_use_checkpnts": var_use_checkpnts.get(),
        "var_checkpoint_freq": var_checkpoint_freq.get() if var_checkpoint_freq.get().isdecimal() else "",
        "var_cont_checkpnt": var_cont_checkpnt.get(),
        "var_process_vid": var_process_vid.get(),
        "var_not_all_frames": var_not_all_frames.get(),
        "var_nth_frame": var_nth_frame.get() if var_nth_frame.get().isdecimal() else "",
        "var_separate_files": var_separate_files.get(),
        "var_keep_series": var_keep_series.get(),
        "var_keep_series_seconds": var_keep_series_seconds.get(),
        "var_keep_series_species": [],
        "var_file_placement": var_file_placement.get(),
        "var_sep_conf": var_sep_conf.get(),
        "var_vis_files": var_vis_files.get(),
        "var_vis_size_idx": dpd_options_vis_size[lang_idx].index(var_vis_size.get()),
        "var_vis_bbox": var_vis_bbox.get(),
        "var_vis_blur": var_vis_blur.get(),
        "var_crp_files": var_crp_files.get(),
        "var_exp": var_exp.get(),
        "var_exp_format_idx": dpd_options_exp_format[lang_idx].index(var_exp_format.get())
    })

    # update keep-series trigger display
    try:
        dsp_keep_series_species.configure(text=["Any", "Cualquiera", "Toutes"][lang_idx])
    except Exception:
        pass

    # reset model specific variables
    model_vars = load_model_vars()
    if model_vars != {}:

        # select all classes
        selected_classes = model_vars["all_classes"]
        write_model_vars(new_values = {"selected_classes": selected_classes})
        model_cls_animal_options(var_cls_model.get())

        # set model specific thresholds
        var_cls_detec_thresh.set(model_vars["var_cls_detec_thresh_default"])
        var_cls_class_thresh.set(model_vars["var_cls_class_thresh_default"])
        write_model_vars(new_values = {"var_cls_detec_thresh": str(var_cls_detec_thresh.get()),
                                    "var_cls_class_thresh": str(var_cls_class_thresh.get())})

    # update window
    toggle_cls_frame()
    toggle_img_frame()
    toggle_nth_frame()
    toggle_vid_frame()
    toggle_exp_frame()
    toggle_vis_frame()
    toggle_sep_frame()
    toggle_keep_series_frame()
    toggle_image_size_for_deploy()
    resize_canvas_to_content()

##########################################
############# TKINTER WINDOW #############
##########################################

# make it look similar on different systems
if os.name == "nt": # windows
    text_font = "TkDefaultFont"
    resize_img_factor = 0.95
    text_size_adjustment_factor = 0.83
    first_level_frame_font_size = 13
    second_level_frame_font_size = 10
    label_width = 320 * scale_factor
    widget_width = 225 * scale_factor
    frame_width = label_width + widget_width + 60
    subframe_correction_factor = 15
    minsize_rows = 28 * scale_factor
    explanation_text_box_height_factor = 0.8
    PADY = 8
    PADX = 10
    ICON_SIZE = 35
    LOGO_WIDTH = 135
    LOGO_HEIGHT = 50
    ADV_WINDOW_WIDTH = 1194
    SIM_WINDOW_WIDTH = 630
    SIM_WINDOW_HEIGHT = 699
    ADV_EXTRA_GRADIENT_HEIGHT = 98 * scale_factor
    ADV_TOP_BANNER_WIDTH_FACTOR = 17.4
    SIM_TOP_BANNER_WIDTH_FACTOR = 6
    RESULTS_TABLE_WIDTH = 600
    RESULTS_WINDOW_WIDTH = 803
    RESULTS_WINDOW_HEIGHT = 700
    ADDAX_TXT_SIZE = 8
    GREY_BUTTON_BORDER_WIDTH = 0
elif sys.platform == "linux" or sys.platform == "linux2": # linux
    text_font = "Times"
    resize_img_factor = 1
    text_size_adjustment_factor = 0.7
    first_level_frame_font_size = 13
    second_level_frame_font_size = 10
    label_width = 320
    widget_width = 225
    frame_width = label_width + widget_width + 60
    subframe_correction_factor = 15
    minsize_rows = 28
    explanation_text_box_height_factor = 1
    PADY = 8
    PADX = 10
    ICON_SIZE = 35
    LOGO_WIDTH = 135
    LOGO_HEIGHT = 50
    ADV_WINDOW_WIDTH = 1194
    SIM_WINDOW_WIDTH = 630
    SIM_WINDOW_HEIGHT = 683
    ADV_EXTRA_GRADIENT_HEIGHT = 90
    ADV_TOP_BANNER_WIDTH_FACTOR = 17.4
    SIM_TOP_BANNER_WIDTH_FACTOR = 6
    RESULTS_TABLE_WIDTH = 600
    RESULTS_WINDOW_WIDTH = 803
    RESULTS_WINDOW_HEIGHT = 700
    ADDAX_TXT_SIZE = 8
    GREY_BUTTON_BORDER_WIDTH = 1
else: # macOS
    text_font = "TkDefaultFont"
    resize_img_factor = 1
    text_size_adjustment_factor = 1
    first_level_frame_font_size = 15
    second_level_frame_font_size = 13
    label_width = 350
    widget_width = 170
    frame_width = label_width + widget_width + 50
    subframe_correction_factor = 15
    minsize_rows = 28
    explanation_text_box_height_factor = 1
    PADY = 8
    PADX = 10
    ICON_SIZE = 35
    LOGO_WIDTH = 135
    LOGO_HEIGHT = 50
    ADV_WINDOW_WIDTH = 1194
    SIM_WINDOW_WIDTH = 630
    SIM_WINDOW_HEIGHT = 696
    ADV_EXTRA_GRADIENT_HEIGHT = 130
    ADV_TOP_BANNER_WIDTH_FACTOR = 23.2
    SIM_TOP_BANNER_WIDTH_FACTOR = 6
    RESULTS_TABLE_WIDTH = 600
    RESULTS_WINDOW_WIDTH = 803
    RESULTS_WINDOW_HEIGHT = 700
    ADDAX_TXT_SIZE = 9
    GREY_BUTTON_BORDER_WIDTH = 0

# TKINTER MAIN WINDOW 
root = customtkinter.CTk()
AddaxAI_icon_image = tk.PhotoImage(file=os.path.join(AddaxAI_files, "AddaxAI", "imgs", "square_logo_excl_text.png"))
root.iconphoto(True, AddaxAI_icon_image)
root.withdraw()
main_label_font = customtkinter.CTkFont(family='CTkFont', size=14, weight = 'bold')
url_label_font = customtkinter.CTkFont(family='CTkFont', underline = True)
italic_label_font = customtkinter.CTkFont(family='CTkFont', size=14, slant='italic')

# set the global appearance for the app
customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme(os.path.join(AddaxAI_files, "AddaxAI", "themes", "addaxai.json"))

# ADVANCED MODE WINDOW 
advanc_mode_win = customtkinter.CTkToplevel(root)
advanc_mode_win.title(f"AddaxAI v{current_AA_version} - "+["Advanced mode", "Modo avanzado", "Mode avancé"][lang_idx])
advanc_mode_win.geometry("+20+20")
advanc_mode_win.protocol("WM_DELETE_WINDOW", on_toplevel_close)
advanc_bg_image = customtkinter.CTkImage(PIL_sidebar, size=(ADV_WINDOW_WIDTH, 10))
advanc_bg_image_label = customtkinter.CTkLabel(advanc_mode_win, image=advanc_bg_image)
advanc_bg_image_label.grid(row=0, column=0)
advanc_main_frame = customtkinter.CTkFrame(advanc_mode_win, corner_radius=0, fg_color = 'transparent', bg_color = yellow_primary)
advanc_main_frame.grid(row=0, column=0, sticky="ns")
if scale_factor != 1.0: # set fixed width for when scaling is applied
    tabControl = ttk.Notebook(advanc_main_frame, width = int(1150 * scale_factor))
else:
    tabControl = ttk.Notebook(advanc_main_frame)
advanc_mode_win.withdraw() # only show when all widgets are loaded

# logo
logoImage = customtkinter.CTkImage(PIL_logo_incl_text, size=(LOGO_WIDTH, LOGO_HEIGHT))
customtkinter.CTkLabel(advanc_main_frame, text="", image = logoImage).grid(column=0, row=0, columnspan=2, sticky='', pady=(PADY, 0), padx=0)
adv_top_banner = customtkinter.CTkImage(PIL_logo_incl_text, size=(LOGO_WIDTH, LOGO_HEIGHT))
customtkinter.CTkLabel(advanc_main_frame, text="", image = adv_top_banner).grid(column=0, row=0, columnspan=2, sticky='ew', pady=(PADY, 0), padx=0)
adv_spacer_top = customtkinter.CTkFrame(advanc_main_frame, height=PADY, fg_color=yellow_primary)
adv_spacer_top.grid(column=0, row=1, columnspan=2, sticky='ew')
adv_spacer_bottom = customtkinter.CTkFrame(advanc_main_frame, height=PADY, fg_color=yellow_primary)
adv_spacer_bottom.grid(column=0, row=5, columnspan=2, sticky='ew')

# prepare check mark for later use
check_mark_one_row = PIL_checkmark.resize((20, 20), Image.Resampling.LANCZOS)
check_mark_one_row = ImageTk.PhotoImage(check_mark_one_row)
check_mark_two_rows = PIL_checkmark.resize((45, 45), Image.Resampling.LANCZOS)
check_mark_two_rows = ImageTk.PhotoImage(check_mark_two_rows)

# grey top buttons
adv_btn_switch_mode_txt = ["To simple mode", 'Al modo simple', 'Mode simple']
adv_btn_switch_mode = GreyTopButton(master = advanc_main_frame, text = adv_btn_switch_mode_txt[lang_idx], command = switch_mode)
adv_btn_switch_mode.grid(row=0, column=0, padx=PADX, pady=(PADY, 0), columnspan = 2, sticky="nw")
adv_btn_switch_lang = GreyTopButton(master = advanc_main_frame, text = "Switch language", command = set_language)
adv_btn_switch_lang.grid(row=0, column=0, padx=PADX, pady=(0, 0), columnspan = 2, sticky="sw")
adv_btn_sponsor_txt = ["Sponsor project", "Patrocine proyecto", "Projet de financement"]
adv_btn_sponsor = GreyTopButton(master = advanc_main_frame, text = adv_btn_sponsor_txt[lang_idx], command = sponsor_project)
adv_btn_sponsor.grid(row=0, column=0, padx=PADX, pady=(PADY, 0), columnspan = 2, sticky="ne")
adv_btn_reset_values_txt = ["Reset values", 'Restablecer valores', 'Réinitialiser les valeurs']
adv_btn_reset_values = GreyTopButton(master = advanc_main_frame, text = adv_btn_reset_values_txt[lang_idx], command = reset_values)
adv_btn_reset_values.grid(row=0, column=0, padx=PADX, pady=(0, 0), columnspan = 2, sticky="se")

# about
adv_abo_lbl_txt = ["By Addax Data Science. More conservation technology? Visit", "Creado por Addax Data Science. ¿Más tecnología de conservación? Visite",
                   "Par Addax Data Science. Vous désirez plus de technologies de conservation? Visitez"]
adv_abo_lbl = tk.Label(advanc_main_frame, text=adv_abo_lbl_txt[lang_idx], font = Font(size = ADDAX_TXT_SIZE), fg="black", bg = yellow_primary)
adv_abo_lbl.grid(row=6, column=0, columnspan = 2, sticky="")
adv_abo_lbl_link = tk.Label(advanc_main_frame, text="addaxdatascience.com", cursor="hand2", font = Font(size = ADDAX_TXT_SIZE, underline=1), fg=green_primary, bg =yellow_primary)
adv_abo_lbl_link.grid(row=7, column=0, columnspan = 2, sticky="", pady=(0, PADY))
adv_abo_lbl_link.bind("<Button-1>", lambda e: callback("http://addaxdatascience.com"))

# deploy tab
deploy_tab = ttk.Frame(tabControl)
deploy_tab.columnconfigure(0, weight=1, minsize=frame_width)
deploy_tab.columnconfigure(1, weight=1, minsize=frame_width)
deploy_tab_text = ['Deploy', 'Despliegue', 'Traitement']
tabControl.add(deploy_tab, text=deploy_tab_text[lang_idx])
deploy_canvas = tk.Canvas(deploy_tab)
deploy_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
deploy_y_scrollbar = ttk.Scrollbar(deploy_tab, orient=tk.VERTICAL, command=deploy_canvas.yview)
deploy_y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
deploy_canvas.configure(yscrollcommand=deploy_y_scrollbar.set)
deploy_scrollable_frame = ttk.Frame(deploy_canvas)
deploy_canvas.create_window((0, 0), window=deploy_scrollable_frame, anchor="nw")
deploy_scrollable_frame.bind("<Configure>", lambda event: deploy_canvas.configure(scrollregion=deploy_canvas.bbox("all")))

# help tab
help_tab = ttk.Frame(tabControl)
help_tab_text = ['Help', 'Ayuda', 'Aide']
tabControl.add(help_tab, text=help_tab_text[lang_idx])

# about tab
about_tab = ttk.Frame(tabControl)
about_tab_text = ['About', 'Acerca de', 'À propos']
tabControl.add(about_tab, text=about_tab_text[lang_idx])

# grid
tabControl.grid(column=0, row=2, sticky="ns", pady = 0)

#### deploy tab
### first step
fst_step_txt = ['Step 1: Select folder', 'Paso 1: Seleccione carpeta', 'Étape 1: Sélectionner le dossier']
row_fst_step = 1
fst_step = LabelFrame(deploy_scrollable_frame, text=" " + fst_step_txt[lang_idx] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg=green_primary, borderwidth=2)
fst_step.configure(font=(text_font, first_level_frame_font_size, "bold"))
fst_step.grid(column=0, row=row_fst_step, columnspan=1, sticky='ew')
fst_step.columnconfigure(0, weight=1, minsize=label_width)
fst_step.columnconfigure(1, weight=1, minsize=widget_width)

# choose folder
lbl_choose_folder_txt = ["Source folder", "Carpeta de origen", "Dossier source"]
row_choose_folder = 0
lbl_choose_folder = Label(master=fst_step, text=lbl_choose_folder_txt[lang_idx], width=1, anchor="w")
lbl_choose_folder.grid(row=row_choose_folder, sticky='nesw', pady=2)
var_choose_folder = StringVar()
var_choose_folder.set("")
var_choose_folder_short = StringVar()
dsp_choose_folder = Label(master=fst_step, textvariable=var_choose_folder_short, fg='grey', padx = 5)
btn_choose_folder = Button(master=fst_step, text=browse_txt[lang_idx], width=1, command=lambda: [browse_dir(var_choose_folder, var_choose_folder_short, dsp_choose_folder, 25, row_choose_folder, 0, 'w', source_dir = True), update_frame_states()])
btn_choose_folder.grid(row=row_choose_folder, column=1, sticky='nesw', padx=5)

### second step
snd_step_txt = ['Step 2: Analysis', 'Paso 2: Análisis', 'Étape 2: Analyse']
row_snd_step = 2
snd_step = LabelFrame(deploy_scrollable_frame, text=" " + snd_step_txt[lang_idx] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg=green_primary, borderwidth=2)
snd_step.configure(font=(text_font, first_level_frame_font_size, "bold"))
snd_step.grid(column=0, row=row_snd_step, sticky='nesw')
snd_step.columnconfigure(0, weight=1, minsize=label_width)
snd_step.columnconfigure(1, weight=1, minsize=widget_width)

# check which detectors are installed
det_models = fetch_known_models(DET_DIR)
dpd_options_model = [det_models + ["Custom model"], det_models + ["Otro modelo"], det_models + ["Modèle personnalisé"]]

# choose detector
lbl_model_txt = ['Model to detect animals, vehicles, and persons', 'Modelo de detección de animales, vehículos y personas', 'Modèle de détection d\'animaux, véhicules et personnes']
row_model = 0
lbl_model = Label(master=snd_step, text=lbl_model_txt[lang_idx], width=1, anchor="w")
lbl_model.grid(row=row_model, sticky='nesw', pady=2)
var_det_model = StringVar(snd_step)
var_det_model.set(dpd_options_model[lang_idx][global_vars["var_det_model_idx"]]) # take idx instead of string
var_det_model_short = StringVar()
var_det_model_short.set(global_vars["var_det_model_short"])
var_det_model_path = StringVar()
var_det_model_path.set(global_vars["var_det_model_path"])
dpd_model = OptionMenu(snd_step, var_det_model, *dpd_options_model[lang_idx], command=model_options)
dpd_model.configure(width=1)
dpd_model.grid(row=row_model, column=1, sticky='nesw', padx=5)
dsp_model = Label(master=snd_step, textvariable=var_det_model_short, fg=green_primary)
if var_det_model_short.get() != "":
    dsp_model.grid(column=0, row=row_model, sticky='e')

# check if user has classifiers installed
cls_models = fetch_known_models(CLS_DIR)
dpd_options_cls_model = [["None"] + cls_models, ["Ninguno"] + cls_models, ["Aucun"] + cls_models]

# use classifier
lbl_cls_model_txt = ["Model to identify animals", "Modelo para identificar mejor a los animales", "Modèle pour l'identification d'animaux"]
row_cls_model = 1
lbl_cls_model = Label(snd_step, text=lbl_cls_model_txt[lang_idx], width=1, anchor="w")
lbl_cls_model.grid(row=row_cls_model, sticky='nesw', pady=2)
var_cls_model = StringVar(snd_step)
var_cls_model.set(dpd_options_cls_model[lang_idx][global_vars["var_cls_model_idx"]]) # take idx instead of string
dpd_cls_model = OptionMenu(snd_step, var_cls_model, *dpd_options_cls_model[lang_idx], command=model_cls_animal_options)
dpd_cls_model.configure(width=1, state=DISABLED)
dpd_cls_model.grid(row=row_cls_model, column=1, sticky='nesw', padx=5, pady=2)

# set global model vars for startup
model_vars = load_model_vars()

## classification option frame (hidden by default)
cls_frame_txt = ["Identification options", "Opciones de identificación", "Options d'identification"]
cls_frame_row = 2
cls_frame = LabelFrame(snd_step, text=" ↳ " + cls_frame_txt[lang_idx] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="black")
cls_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
cls_frame.grid(row=cls_frame_row, column=0, columnspan=2, sticky = 'ew')
cls_frame.columnconfigure(0, weight=1, minsize= label_width - subframe_correction_factor)
cls_frame.columnconfigure(1, weight=1, minsize= widget_width - subframe_correction_factor)
cls_frame.grid_forget()

# show model info
row_btn_model_info = 1
lbl_model_info_txt = ["Show model information", "Mostrar información del modelo (inglés)", "Afficher les informations du modèle"]
row_model_info = 0
lbl_model_info = Label(master=cls_frame, text="     " + lbl_model_info_txt[lang_idx], width=1, anchor="w")
lbl_model_info.grid(row=row_model_info, sticky='nesw', pady=2)
btn_model_info = Button(master=cls_frame, text=show_txt[lang_idx], width=1, command=show_model_info)
btn_model_info.grid(row=row_model_info, column=1, sticky='nesw', padx=5)

# choose classes
lbl_choose_classes_txt = ["Select species", "Seleccionar especies", "Sélectionner les espèces à détecter"]
row_choose_classes = 1
lbl_choose_classes = Label(master=cls_frame, text="     " + lbl_choose_classes_txt[lang_idx], width=1, anchor="w")
lbl_choose_classes.grid(row=row_choose_classes, sticky='nesw', pady=2)
btn_choose_classes = Button(master=cls_frame, text=select_txt[lang_idx], width=1, command=open_species_selection)
btn_choose_classes.grid(row=row_choose_classes, column=1, sticky='nesw', padx=5)
if var_cls_model.get() not in none_txt:
    dsp_choose_classes = Label(cls_frame, text = f"{len(model_vars.get('selected_classes', []))} of {len(model_vars.get('all_classes', []))}")
else:
    dsp_choose_classes = Label(cls_frame, text= "")
dsp_choose_classes.grid(row=row_choose_classes, column=0, sticky='e', padx=0)
dsp_choose_classes.configure(fg=green_primary)

# threshold to classify detections
lbl_cls_detec_thresh_txt = ["Detection confidence threshold", "Umbral de confianza de detección", "Seuil de confiance des détections"]
row_cls_detec_thresh = 2
lbl_cls_detec_thresh = Label(cls_frame, text="     " + lbl_cls_detec_thresh_txt[lang_idx], width=1, anchor="w")
lbl_cls_detec_thresh.grid(row=row_cls_detec_thresh, sticky='nesw', pady=2)
var_cls_detec_thresh = DoubleVar()

var_cls_detec_thresh.set(model_vars.get('var_cls_detec_thresh', 0.6))
scl_cls_detec_thresh = Scale(cls_frame, from_=0.01, to=1, resolution=0.01, orient=HORIZONTAL,
                             variable=var_cls_detec_thresh, showvalue=0, width=10, length=1, state=DISABLED,
                             command=lambda value: write_model_vars(new_values = {"var_cls_detec_thresh": str(value)}))
scl_cls_detec_thresh.grid(row=row_cls_detec_thresh, column=1, sticky='ew', padx=10)
dsp_cls_detec_thresh = Label(cls_frame, textvariable=var_cls_detec_thresh)
dsp_cls_detec_thresh.grid(row=row_cls_detec_thresh, column=0, sticky='e', padx=0)
dsp_cls_detec_thresh.configure(fg=green_primary)

# threshold accept identifications
lbl_cls_class_thresh_txt = ["Classification confidence threshold", "Umbral de confianza de la clasificación", "Seuil de confiance des classifications"]
row_cls_class_thresh = 3
lbl_cls_class_thresh = Label(cls_frame, text="     " + lbl_cls_class_thresh_txt[lang_idx], width=1, anchor="w")
lbl_cls_class_thresh.grid(row=row_cls_class_thresh, sticky='nesw', pady=2)
var_cls_class_thresh = DoubleVar()
var_cls_class_thresh.set(model_vars.get('var_cls_class_thresh', 0.5))
scl_cls_class_thresh = Scale(cls_frame, from_=0.01, to=1, resolution=0.01, orient=HORIZONTAL,
                             variable=var_cls_class_thresh, showvalue=0, width=10, length=1, state=DISABLED,
                             command=lambda value: write_model_vars(new_values = {"var_cls_class_thresh": value}))
scl_cls_class_thresh.grid(row=row_cls_class_thresh, column=1, sticky='ew', padx=10)
dsp_cls_class_thresh = Label(cls_frame, textvariable=var_cls_class_thresh)
dsp_cls_class_thresh.grid(row=row_cls_class_thresh, column=0, sticky='e', padx=0)
dsp_cls_class_thresh.configure(fg=green_primary)

# Smoothen results
lbl_smooth_cls_animal_txt = ["Smooth confidence scores per sequence", "Suavizar puntuaciones por secuencia", "Lissage de scores de confiance par séquence"]
row_smooth_cls_animal = 4
lbl_smooth_cls_animal = Label(cls_frame, text="     " + lbl_smooth_cls_animal_txt[lang_idx], width=1, anchor="w")
lbl_smooth_cls_animal.grid(row=row_smooth_cls_animal, sticky='nesw', pady=2)
var_smooth_cls_animal = BooleanVar()
var_smooth_cls_animal.set(model_vars.get('var_smooth_cls_animal', False))
chb_smooth_cls_animal = Checkbutton(cls_frame, variable=var_smooth_cls_animal, anchor="w", command = on_chb_smooth_cls_animal_change)
chb_smooth_cls_animal.grid(row=row_smooth_cls_animal, column=1, sticky='nesw', padx=5)

# taxonomic fallback checkbox (only visible if taxon mapping is present)
lbl_tax_fallback_txt = ["Enable taxonomic confidence aggregation", "Activar agregación de confianza taxonómica", "Activer l'aggrégation taxonomique"]
row_tax_fallback = 5
lbl_tax_fallback = Label(cls_frame, text="     " + lbl_tax_fallback_txt[lang_idx], width=1, anchor="w")
var_tax_fallback = BooleanVar()
var_tax_fallback.set(model_vars.get('var_tax_fallback', False))
chb_tax_fallback = Checkbutton(cls_frame, variable=var_tax_fallback, anchor="w", command = toggle_tax_levels)

# taxonomic fallbaock dropdown (only visible if taxon mapping is present)
lbl_tax_levels_txt = ["Prediction level", "Nivel de predicción", "Niveau de prédiction"]
row_tax_levels = 6
lbl_tax_levels = Label(cls_frame, text="     " + lbl_tax_levels_txt[lang_idx], width=1, anchor="w")
var_tax_levels = StringVar(cls_frame)
var_tax_levels.set("dummy") # set dummy value to avoid error
dpd_tax_levels = OptionMenu(cls_frame, var_tax_levels, ["dummy"])
dpd_tax_levels.configure(width=1, state=DISABLED)

# make taxonomic fallback widgets visible if taxon mapping is present
if taxon_mapping_csv_present():
    toggle_tax_levels_dpd_options()
    lbl_tax_fallback.grid(row=row_tax_fallback, sticky='nesw', pady=2)
    chb_tax_fallback.grid(row=row_tax_fallback, column=1, sticky='nesw', padx=5)
    toggle_tax_levels()

# choose location for species net
lbl_sppnet_location_txt = ["Location", "Ubicación", "Emplacement"]
row_sppnet_location = 1
lbl_sppnet_location = Label(master=cls_frame, text="     " + lbl_sppnet_location_txt[lang_idx], width=1, anchor="w")
lbl_sppnet_location.grid(row=row_sppnet_location, sticky='nesw', pady=2)
var_sppnet_location = StringVar(cls_frame)
var_sppnet_location.set(dpd_options_sppnet_location[lang_idx][global_vars["var_sppnet_location_idx"]]) # take idx instead of string
dpd_sppnet_location = OptionMenu(cls_frame, var_sppnet_location, *dpd_options_sppnet_location[lang_idx])
dpd_sppnet_location.configure(width=1, state=DISABLED)
# dpd_sppnet_location.grid(row=row_sppnet_location, column=1, sticky='nesw', padx=5, pady=2) # dont grid this by default

# include subdirectories
lbl_exclude_subs_txt = ["Don't process subdirectories", "No procesar subcarpetas", "Ne pas traiter les sous-dossiers"]
row_exclude_subs = 3
lbl_exclude_subs = Label(snd_step, text=lbl_exclude_subs_txt[lang_idx], width=1, anchor="w")
lbl_exclude_subs.grid(row=row_exclude_subs, sticky='nesw', pady=2)
var_exclude_subs = BooleanVar()
var_exclude_subs.set(global_vars['var_exclude_subs'])
chb_exclude_subs = Checkbutton(snd_step, variable=var_exclude_subs, anchor="w")
chb_exclude_subs.grid(row=row_exclude_subs, column=1, sticky='nesw', padx=5)

# use custom image size
lbl_use_custom_img_size_for_deploy_txt = ["Use custom image size", "Usar tamaño de imagen personalizado", "Utiliser une dimension d'image personnalisée"]
row_use_custom_img_size_for_deploy = 4
lbl_use_custom_img_size_for_deploy = Label(snd_step, text=lbl_use_custom_img_size_for_deploy_txt[lang_idx], width=1, anchor="w")
lbl_use_custom_img_size_for_deploy.grid(row=row_use_custom_img_size_for_deploy, sticky='nesw', pady=2)
var_use_custom_img_size_for_deploy = BooleanVar()
var_use_custom_img_size_for_deploy.set(global_vars['var_use_custom_img_size_for_deploy'])
chb_use_custom_img_size_for_deploy = Checkbutton(snd_step, variable=var_use_custom_img_size_for_deploy, command=toggle_image_size_for_deploy, anchor="w")
chb_use_custom_img_size_for_deploy.grid(row=row_use_custom_img_size_for_deploy, column=1, sticky='nesw', padx=5)

# specify custom image size (not grid by default)
lbl_image_size_for_deploy_txt = ["Image size", "Tamaño imagen", "Dimension d'image"]
row_image_size_for_deploy = 5
lbl_image_size_for_deploy = Label(snd_step, text=" ↳ " + lbl_image_size_for_deploy_txt[lang_idx], width=1, anchor="w")
var_image_size_for_deploy = StringVar()
var_image_size_for_deploy.set(global_vars['var_image_size_for_deploy'])
ent_image_size_for_deploy = tk.Entry(snd_step, textvariable=var_image_size_for_deploy, fg='grey', state=NORMAL, width=1)
if var_image_size_for_deploy.get() == "":
    ent_image_size_for_deploy.insert(0, f"{eg_txt[lang_idx]}: 640")
else:
    ent_image_size_for_deploy.configure(fg='black')
ent_image_size_for_deploy.bind("<FocusIn>", image_size_for_deploy_focus_in)
ent_image_size_for_deploy.configure(state=DISABLED)

# use absolute paths
lbl_abs_paths_txt = ["Use absolute paths in output file", "Usar rutas absolutas en archivo de salida", "Utiliser chemins absolus dans le fichier de sortie"]
row_abs_path = 6
lbl_abs_paths = Label(snd_step, text=lbl_abs_paths_txt[lang_idx], width=1, anchor="w")
lbl_abs_paths.grid(row=row_abs_path, sticky='nesw', pady=2)
var_abs_paths = BooleanVar()
var_abs_paths.set(global_vars['var_abs_paths'])
chb_abs_paths = Checkbutton(snd_step, variable=var_abs_paths, command=abs_paths_warning, anchor="w")
chb_abs_paths.grid(row=row_abs_path, column=1, sticky='nesw', padx=5)

# use absolute paths
lbl_disable_GPU_txt = ["Disable GPU processing", "Desactivar el procesamiento en la GPU", "Désactiver le traitement par GPU"]
row_disable_GPU = 7
lbl_disable_GPU = Label(snd_step, text=lbl_disable_GPU_txt[lang_idx], width=1, anchor="w")
lbl_disable_GPU.grid(row=row_disable_GPU, sticky='nesw', pady=2)
var_disable_GPU = BooleanVar()
var_disable_GPU.set(global_vars['var_disable_GPU'])
chb_disable_GPU = Checkbutton(snd_step, variable=var_disable_GPU, anchor="w")
chb_disable_GPU.grid(row=row_disable_GPU, column=1, sticky='nesw', padx=5)

# process images
lbl_process_img_txt = ["Process images, if present", "Si está presente, procesa todas las imágenes", "Traiter les images, si présentes"]
row_process_img = 8
lbl_process_img = Label(snd_step, text=lbl_process_img_txt[lang_idx], width=1, anchor="w")
lbl_process_img.grid(row=row_process_img, sticky='nesw', pady=2)
var_process_img = BooleanVar()
var_process_img.set(global_vars['var_process_img'])
chb_process_img = Checkbutton(snd_step, variable=var_process_img, command=toggle_img_frame, anchor="w")
chb_process_img.grid(row=row_process_img, column=1, sticky='nesw', padx=5)

## image option frame (hidden by default)
img_frame_txt = ["Image options", "Opciones de imagen", "Options d'images"]
img_frame_row = 9
img_frame = LabelFrame(snd_step, text=" ↳ " + img_frame_txt[lang_idx] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")
img_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
img_frame.grid(row=img_frame_row, column=0, columnspan=2, sticky = 'ew')
img_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
img_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
img_frame.grid_forget()

# use checkpoints
lbl_use_checkpnts_txt = ["Use checkpoints while running", "Usar puntos de control mientras se ejecuta", "Utiliser des points de contrôle pendant l'exécution"]
row_use_checkpnts = 0
lbl_use_checkpnts = Label(img_frame, text="     " + lbl_use_checkpnts_txt[lang_idx], pady=2, state=DISABLED, width=1, anchor="w")
lbl_use_checkpnts.grid(row=row_use_checkpnts, sticky='nesw')
var_use_checkpnts = BooleanVar()
var_use_checkpnts.set(global_vars['var_use_checkpnts'])
chb_use_checkpnts = Checkbutton(img_frame, variable=var_use_checkpnts, command=toggle_checkpoint_freq, state=DISABLED, anchor="w")
chb_use_checkpnts.grid(row=row_use_checkpnts, column=1, sticky='nesw', padx=5)

# checkpoint frequency
lbl_checkpoint_freq_txt = ["Checkpoint frequency", "Frecuencia puntos de control", "Fréquence des points de contrôle"]
row_checkpoint_freq = 1
lbl_checkpoint_freq = tk.Label(img_frame, text="        ↳ " + lbl_checkpoint_freq_txt[lang_idx], pady=2, state=DISABLED, width=1, anchor="w")
lbl_checkpoint_freq.grid(row=row_checkpoint_freq, sticky='nesw')
var_checkpoint_freq = StringVar()
var_checkpoint_freq.set(global_vars['var_checkpoint_freq'])
ent_checkpoint_freq = tk.Entry(img_frame, textvariable=var_checkpoint_freq, fg='grey', state=NORMAL, width=1)
ent_checkpoint_freq.grid(row=row_checkpoint_freq, column=1, sticky='nesw', padx=5)
if var_checkpoint_freq.get() == "":
    ent_checkpoint_freq.insert(0, f"{eg_txt[lang_idx]}: 10000")
else:
    ent_checkpoint_freq.configure(fg='black')
ent_checkpoint_freq.bind("<FocusIn>", checkpoint_freq_focus_in)
ent_checkpoint_freq.configure(state=DISABLED)

# continue from checkpoint file
lbl_cont_checkpnt_txt = ["Continue from last checkpoint file", "Continuar desde el último punto de control", "Poursuivre à partir du dernier point de contrôle"]
row_cont_checkpnt = 2
lbl_cont_checkpnt = Label(img_frame, text="     " + lbl_cont_checkpnt_txt[lang_idx], pady=2, state=DISABLED, width=1, anchor="w")
lbl_cont_checkpnt.grid(row=row_cont_checkpnt, sticky='nesw')
var_cont_checkpnt = BooleanVar()
var_cont_checkpnt.set(global_vars['var_cont_checkpnt'])
chb_cont_checkpnt = Checkbutton(img_frame, variable=var_cont_checkpnt, state=DISABLED, command=disable_chb_cont_checkpnt, anchor="w")
chb_cont_checkpnt.grid(row=row_cont_checkpnt, column=1, sticky='nesw', padx=5)

# process videos
lbl_process_vid_txt = ["Process videos, if present", "Si está presente, procesa todos los vídeos", "Traiter les vidéos, si présent"]
row_process_vid = 10
lbl_process_vid = Label(snd_step, text=lbl_process_vid_txt[lang_idx], width=1, anchor="w")
lbl_process_vid.grid(row=row_process_vid, sticky='nesw', pady=2)
var_process_vid = BooleanVar()
var_process_vid.set(global_vars['var_process_vid'])
chb_process_vid = Checkbutton(snd_step, variable=var_process_vid, command=toggle_vid_frame, anchor="w")
chb_process_vid.grid(row=row_process_vid, column=1, sticky='nesw', padx=5)

## video option frame (disabled by default)
vid_frame_txt = ["Video options", "Opciones de vídeo", "Options des vidéos"]
vid_frame_row = 11
vid_frame = LabelFrame(snd_step, text=" ↳ " + vid_frame_txt[lang_idx] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")
vid_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
vid_frame.grid(row=vid_frame_row, column=0, columnspan=2, sticky='ew')
vid_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
vid_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
vid_frame.grid_forget()

# dont process all frames
lbl_not_all_frames_txt = ["Don't process every frame", "No procesar cada fotograma", "Ne pas traiter toutes les trames"]
row_not_all_frames = 0
lbl_not_all_frames = Label(vid_frame, text="     " + lbl_not_all_frames_txt[lang_idx], pady=2, state=DISABLED, width=1, anchor="w")
lbl_not_all_frames.grid(row=row_not_all_frames, sticky='nesw')
var_not_all_frames = BooleanVar()
var_not_all_frames.set(global_vars['var_not_all_frames'])
chb_not_all_frames = Checkbutton(vid_frame, variable=var_not_all_frames, command=toggle_nth_frame, state=DISABLED, anchor="w")
chb_not_all_frames.grid(row=row_not_all_frames, column=1, sticky='nesw', padx=5)

# process every nth frame
lbl_nth_frame_txt = ["Sample frames every N seconds", "Muestreo de tramas cada N segundos", "Échantillonner les trames à chaque N secondes"]
row_nth_frame = 1
lbl_nth_frame = tk.Label(vid_frame, text="        ↳ " + lbl_nth_frame_txt[lang_idx], pady=2, state=DISABLED, width=1, anchor="w")
lbl_nth_frame.grid(row=row_nth_frame, sticky='nesw')
var_nth_frame = StringVar()
var_nth_frame.set(global_vars['var_nth_frame'])
ent_nth_frame = tk.Entry(vid_frame, textvariable=var_nth_frame, fg='grey' if var_nth_frame.get().isdecimal() else 'black', state=NORMAL, width=1)
ent_nth_frame.grid(row=row_nth_frame, column=1, sticky='nesw', padx=5)
if var_nth_frame.get() == "":
    ent_nth_frame.insert(0, f"{eg_txt[lang_idx]}: 1")
    ent_nth_frame.configure(fg='grey')
else:
    ent_nth_frame.configure(fg='black')
ent_nth_frame.bind("<FocusIn>", nth_frame_focus_in)
ent_nth_frame.configure(state=DISABLED)

# button start deploy
btn_start_deploy_txt = ["Start processing", "Empezar a procesar", "Démarrer le traitement"]
row_btn_start_deploy = 12
btn_start_deploy = Button(snd_step, text=btn_start_deploy_txt[lang_idx], command=start_deploy)
btn_start_deploy.grid(row=row_btn_start_deploy, column=0, columnspan=2, sticky='ew')

### human-in-the-loop step
trd_step_txt = ["Step 3: Annotation (optional)", "Paso 3: Anotación (opcional)", "Étape 3: Annotation (optionnel)"]
trd_step_row = 1
trd_step = LabelFrame(deploy_scrollable_frame, text=" " + trd_step_txt[lang_idx] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg=green_primary, borderwidth=2)
trd_step.configure(font=(text_font, first_level_frame_font_size, "bold"))
trd_step.grid(column=1, row=trd_step_row, sticky='nesw')
trd_step.columnconfigure(0, weight=1, minsize=label_width)
trd_step.columnconfigure(1, weight=1, minsize=widget_width)

# human-in-the-loop 
lbl_hitl_main_txt = ["Manually verify results", "Verificar manualmente los resultados", "Vérifier les résultats manuellement"]
row_hitl_main = 0
lbl_hitl_main = Label(master=trd_step, text=lbl_hitl_main_txt[lang_idx], width=1, anchor="w")
lbl_hitl_main.grid(row=row_hitl_main, sticky='nesw', pady=2)
btn_hitl_main = Button(master=trd_step, text=["Start", "Iniciar", "Démarrer"][lang_idx], width=1, command = start_or_continue_hitl)
btn_hitl_main.grid(row=row_hitl_main, column=1, sticky='nesw', padx=5)

### fourth step
fth_step_txt = ["Step 4: Post-processing (optional)", "Paso 4: Post-Procesado (opcional)", "Étape 4: Post-traitement (optionnel)"]
fth_step_row = 2
fth_step = LabelFrame(deploy_scrollable_frame, text=" " + fth_step_txt[lang_idx] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg=green_primary, borderwidth=2)
fth_step.configure(font=(text_font, first_level_frame_font_size, "bold"))
fth_step.grid(column=1, row=fth_step_row, sticky='nesw')
fth_step.columnconfigure(0, weight=1, minsize=label_width)
fth_step.columnconfigure(1, weight=1, minsize=widget_width)

# folder for results
lbl_output_dir_txt = ["Destination folder", "Carpeta de destino", "Dossier de destination"]
row_output_dir = 0
lbl_output_dir = Label(master=fth_step, text=lbl_output_dir_txt[lang_idx], width=1, anchor="w")
lbl_output_dir.grid(row=row_output_dir, sticky='nesw', pady=2)
var_output_dir = StringVar()
var_output_dir.set("")
var_output_dir_short = StringVar()
dsp_output_dir = Label(master=fth_step, textvariable=var_output_dir_short, fg=green_primary)
btn_output_dir = Button(master=fth_step, text=browse_txt[lang_idx], width=1, command=lambda: browse_dir(var_output_dir, var_output_dir_short, dsp_output_dir, 25, row_output_dir, 0, 'e'))
btn_output_dir.grid(row=row_output_dir, column=1, sticky='nesw', padx=5)

# separate files
lbl_separate_files_txt = ["Separate files into subdirectories", "Separar archivos en subcarpetas", "Classer les fichiers dans des sous-dossiers"]
row_separate_files = 1
lbl_separate_files = Label(fth_step, text=lbl_separate_files_txt[lang_idx], width=1, anchor="w")
lbl_separate_files.grid(row=row_separate_files, sticky='nesw', pady=2)
var_separate_files = BooleanVar()
var_separate_files.set(global_vars['var_separate_files'])
chb_separate_files = Checkbutton(fth_step, variable=var_separate_files, command=toggle_sep_frame, anchor="w")
chb_separate_files.grid(row=row_separate_files, column=1, sticky='nesw', padx=5)

## separation frame
sep_frame_txt = ["Separation options", "Opciones de separación", "Options de classement"]
sep_frame_row = 2
sep_frame = LabelFrame(fth_step, text=" ↳ " + sep_frame_txt[lang_idx] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")
sep_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
sep_frame.grid(row=sep_frame_row, column=0, columnspan=2, sticky = 'ew')
sep_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
sep_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
sep_frame.grid_forget()

# method of file placement
lbl_file_placement_txt = ["Method of file placement", "Método de desplazamiento de archivo", "Type de classement"]
row_file_placement = 0
lbl_file_placement = Label(sep_frame, text="     " + lbl_file_placement_txt[lang_idx], pady=2, width=1, anchor="w")
lbl_file_placement.grid(row=row_file_placement, sticky='nesw')
var_file_placement = IntVar()
var_file_placement.set(global_vars['var_file_placement'])
rad_file_placement_move = Radiobutton(sep_frame, text=["Copy", "Copiar", "Copie"][lang_idx], variable=var_file_placement, value=2)
rad_file_placement_move.grid(row=row_file_placement, column=1, sticky='w', padx=5)
rad_file_placement_copy = Radiobutton(sep_frame, text=["Move", "Mover", "Déplacement"][lang_idx], variable=var_file_placement, value=1)
rad_file_placement_copy.grid(row=row_file_placement, column=1, sticky='e', padx=5)

# separate per confidence
lbl_sep_conf_txt = ["Sort results based on confidence", "Clasificar resultados basados en confianza", "Trier les résultats à partir du score de confiance"]
row_sep_conf = 1
lbl_sep_conf = Label(sep_frame, text="     " + lbl_sep_conf_txt[lang_idx], width=1, anchor="w")
lbl_sep_conf.grid(row=row_sep_conf, sticky='nesw', pady=2)
var_sep_conf = BooleanVar()
var_sep_conf.set(global_vars['var_sep_conf'])
chb_sep_conf = Checkbutton(sep_frame, variable=var_sep_conf, anchor="w")
chb_sep_conf.grid(row=row_sep_conf, column=1, sticky='nesw', padx=5)


# keep series files (only affects separation)
lbl_keep_series_txt = ["Keep series (if animal detected in one image)", "Conservar series (si se detecta un animal en una imagen)", "Conserver les séries (si un animal est détecté dans une image)"]
row_keep_series = 2
lbl_keep_series = Label(sep_frame, text="     " + lbl_keep_series_txt[lang_idx], width=1, anchor="w")
lbl_keep_series.grid(row=row_keep_series, sticky='nesw', pady=2)
var_keep_series = BooleanVar()
var_keep_series.set(global_vars['var_keep_series'])
chb_keep_series = Checkbutton(sep_frame, variable=var_keep_series, command=toggle_keep_series_frame, anchor="w")
chb_keep_series.grid(row=row_keep_series, column=1, sticky='nesw', padx=5)

## keep_series frame (nested under separation options)
keep_series_frame_txt = ["Keep series options", "Opciones de conservación de series", "Options de conservation des séries"]
keep_series_frame_row = 3
keep_series_frame = LabelFrame(sep_frame, text="        ↳ " + keep_series_frame_txt[lang_idx] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")
keep_series_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
keep_series_frame.grid(row=keep_series_frame_row, column=0, columnspan=2, sticky = 'ew')
keep_series_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
keep_series_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
keep_series_frame.grid_forget()

# keep series seconds - only visible if keep series is checked
lbl_keep_series_seconds_txt = ["Seconds", "Segundos", "Secondes"]
row_keep_series_seconds = 0
lbl_keep_series_seconds = Label(keep_series_frame, text="     " + lbl_keep_series_seconds_txt[lang_idx], width=1, anchor="w")
lbl_keep_series_seconds.grid(row=row_keep_series_seconds, sticky='nesw', pady=2)
var_keep_series_seconds = DoubleVar()
var_keep_series_seconds.set(global_vars['var_keep_series_seconds'])
chb_keep_series_seconds = Scale(keep_series_frame, from_=0.1, to=10, resolution=0.1, orient=HORIZONTAL, variable=var_keep_series_seconds, showvalue=0, width=10, length=1)
chb_keep_series_seconds.grid(row=row_keep_series_seconds, column=1, sticky='ew', padx=10)
dsp_keep_series_seconds = Label(keep_series_frame, textvariable=var_keep_series_seconds)
dsp_keep_series_seconds.configure(fg=green_primary)
dsp_keep_series_seconds.grid(row=row_keep_series_seconds, column=0, sticky='e', padx=0)

# keep series species - optional trigger filter
lbl_keep_series_species_txt = ["Select species", "Seleccionar especies", "Sélectionner des espèces"]
row_keep_series_species = 1
lbl_keep_series_species = Label(keep_series_frame, text="     " + lbl_keep_series_species_txt[lang_idx], width=1, anchor="w")
lbl_keep_series_species.grid(row=row_keep_series_species, sticky='nesw', pady=2)

# display: show how many triggers are selected (empty = any)
dsp_keep_series_species = Label(keep_series_frame, fg=green_primary)
if len(global_vars.get('var_keep_series_species', []) or []) == 0:
    dsp_keep_series_species.configure(text=["Any", "Cualquiera", "Toutes"][lang_idx])
else:
    dsp_keep_series_species.configure(text=str(len(global_vars.get('var_keep_series_species', []))))
dsp_keep_series_species.grid(row=row_keep_series_species, column=0, sticky='e', padx=0)

btn_keep_series_species = Button(keep_series_frame, text=select_txt[lang_idx], command=open_keep_series_species_selection)
btn_keep_series_species.grid(row=row_keep_series_species, column=1, sticky='w', padx=10)


## visualize images
lbl_vis_files_txt = ["Visualise detections and blur people", "Mostrar detecciones y difuminar personas", "Visualiser les détections et anonymiser les personnes"]
row_vis_files = 3
lbl_vis_files = Label(fth_step, text=lbl_vis_files_txt[lang_idx], width=1, anchor="w")
lbl_vis_files.grid(row=row_vis_files, sticky='nesw', pady=2)
var_vis_files = BooleanVar()
var_vis_files.set(global_vars['var_vis_files'])
chb_vis_files = Checkbutton(fth_step, variable=var_vis_files, anchor="w", command=toggle_vis_frame)
chb_vis_files.grid(row=row_vis_files, column=1, sticky='nesw', padx=5)

## visualization options
vis_frame_txt = ["Visualization options", "Opciones de visualización", "Options de visualisation"]
vis_frame_row = 4
vis_frame = LabelFrame(fth_step, text=" ↳ " + vis_frame_txt[lang_idx] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")
vis_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
vis_frame.grid(row=vis_frame_row, column=0, columnspan=2, sticky = 'ew')
vis_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
vis_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
vis_frame.grid_forget()

## draw bboxes 
lbl_vis_bbox_txt = ["Draw bounding boxes and confidences", "Dibujar contornos y confianzas", "Dessiner les coutours des détections et les scores "] 
row_vis_bbox = 0
lbl_vis_bbox = Label(vis_frame, text="     " + lbl_vis_bbox_txt[lang_idx], width=1, anchor="w")
lbl_vis_bbox.grid(row=row_vis_bbox, sticky='nesw', pady=2)
var_vis_bbox = BooleanVar()
var_vis_bbox.set(global_vars['var_vis_bbox'])
chb_vis_bbox = Checkbutton(vis_frame, variable=var_vis_bbox, anchor="w") 
chb_vis_bbox.grid(row=row_vis_bbox, column=1, sticky='nesw', padx=5)

# line size
lbl_vis_size_txt = ["Select line width and font size", "Ancho de línea y tamaño de fuente", "Sélectionner la largeur de ligne et la taille de police"]
row_vis_size = 1
lbl_vis_size = Label(vis_frame, text="        ↳ " + lbl_vis_size_txt[lang_idx], pady=2, width=1, anchor="w")
lbl_vis_size.grid(row=row_vis_size, sticky='nesw')
dpd_options_vis_size = [["Extra small", "Small", "Medium", "Large", "Extra large"],
                        ["Extra pequeño", "Pequeño", "Mediano", "Grande", "Extra grande"],
                        ["Extra petit", "Petit", "Moyen", "Grand", "Extra grand"]]
var_vis_size = StringVar(vis_frame)
var_vis_size.set(dpd_options_vis_size[lang_idx][global_vars['var_vis_size_idx']])
dpd_vis_size = OptionMenu(vis_frame, var_vis_size, *dpd_options_vis_size[lang_idx])
dpd_vis_size.configure(width=1)
dpd_vis_size.grid(row=row_vis_size, column=1, sticky='nesw', padx=5)

## blur people
lbl_vis_blur_txt = ["Blur people", "Desenfocar a la gente", "Anonymiser (appliquer un flou) les personnes"]
row_vis_blur = 2
lbl_vis_blur = Label(vis_frame, text="     " + lbl_vis_blur_txt[lang_idx], width=1, anchor="w")
lbl_vis_blur.grid(row=row_vis_blur, sticky='nesw', pady=2)
var_vis_blur = BooleanVar()
var_vis_blur.set(global_vars['var_vis_blur'])
chb_vis_blur = Checkbutton(vis_frame, variable=var_vis_blur, anchor="w") 
chb_vis_blur.grid(row=row_vis_blur, column=1, sticky='nesw', padx=5)

## crop images
lbl_crp_files_txt = ["Crop detections", "Recortar detecciones", "Rogner les détection"]
row_crp_files = 5
lbl_crp_files = Label(fth_step, text=lbl_crp_files_txt[lang_idx], width=1, anchor="w")
lbl_crp_files.grid(row=row_crp_files, sticky='nesw', pady=2)
var_crp_files = BooleanVar()
var_crp_files.set(global_vars['var_crp_files'])
chb_crp_files = Checkbutton(fth_step, variable=var_crp_files, anchor="w")
chb_crp_files.grid(row=row_crp_files, column=1, sticky='nesw', padx=5)

# plot
lbl_plt_txt = ["Create maps and graphs", "Crear mapas y gráficos", "Créer les cartes et graphiques"]
row_plt = 6
lbl_plt = Label(fth_step, text=lbl_plt_txt[lang_idx], width=1, anchor="w")
lbl_plt.grid(row=row_plt, sticky='nesw', pady=2)
var_plt = BooleanVar()
var_plt.set(global_vars['var_plt'])
chb_plt = Checkbutton(fth_step, variable=var_plt, anchor="w")
chb_plt.grid(row=row_plt, column=1, sticky='nesw', padx=5)

# export results
lbl_exp_txt = ["Export results and retrieve metadata", "Exportar resultados y recuperar metadatos", "Exporter les résultats and récupérer les métadonnées"]
row_exp = 7
lbl_exp = Label(fth_step, text=lbl_exp_txt[lang_idx], width=1, anchor="w")
lbl_exp.grid(row=row_exp, sticky='nesw', pady=2)
var_exp = BooleanVar()
var_exp.set(global_vars['var_exp'])
chb_exp = Checkbutton(fth_step, variable=var_exp, anchor="w", command=toggle_exp_frame)
chb_exp.grid(row=row_exp, column=1, sticky='nesw', padx=5)

## exportation options
exp_frame_txt = ["Export options", "Opciones de exportación", "Options d'exportation"]
exp_frame_row = 8
exp_frame = LabelFrame(fth_step, text=" ↳ " + exp_frame_txt[lang_idx] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")
exp_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
exp_frame.grid(row=exp_frame_row, column=0, columnspan=2, sticky = 'ew')
exp_frame.columnconfigure(0, weight=1, minsize=label_width - subframe_correction_factor)
exp_frame.columnconfigure(1, weight=1, minsize=widget_width - subframe_correction_factor)
exp_frame.grid_forget()

# export format
lbl_exp_format_txt = ["Output file format", "Formato del archivo de salida", "Format du fichier de sortie"]
row_exp_format = 0
lbl_exp_format = Label(exp_frame, text="     " + lbl_exp_format_txt[lang_idx], pady=2, width=1, anchor="w")
lbl_exp_format.grid(row=row_exp_format, sticky='nesw')
dpd_options_exp_format = [["XLSX", "CSV", "COCO", "Sensing Clues (TSV)"], ["XLSX", "CSV", "COCO", "Sensing Clues (TSV)"], ["XLSX", "CSV", "COCO", "Sensing Clues (TSV)"]]
var_exp_format = StringVar(exp_frame)
var_exp_format.set(dpd_options_exp_format[lang_idx][global_vars['var_exp_format_idx']])
dpd_exp_format = OptionMenu(exp_frame, var_exp_format, *dpd_options_exp_format[lang_idx])
dpd_exp_format.configure(width=1)
dpd_exp_format.grid(row=row_exp_format, column=1, sticky='nesw', padx=5)

# threshold
lbl_thresh_txt = ["Confidence threshold", "Umbral de confianza", "Seuil de confiance"]
row_lbl_thresh = 9
lbl_thresh = Label(fth_step, text=lbl_thresh_txt[lang_idx], width=1, anchor="w")
lbl_thresh.grid(row=row_lbl_thresh, sticky='nesw', pady=2)
var_thresh = DoubleVar()
var_thresh.set(global_vars['var_thresh'])
scl_thresh = Scale(fth_step, from_=0.01, to=1, resolution=0.01, orient=HORIZONTAL, variable=var_thresh, showvalue=0, width=10, length=1)
scl_thresh.grid(row=row_lbl_thresh, column=1, sticky='ew', padx=10)
dsp_thresh = Label(fth_step, textvariable=var_thresh)
dsp_thresh.configure(fg=green_primary)
dsp_thresh.grid(row=row_lbl_thresh, column=0, sticky='e', padx=0)

# postprocessing button
btn_start_postprocess_txt = ["Start post-processing", "Iniciar el postprocesamiento", "Démarrer le post-traitement"]
row_start_postprocess = 10
btn_start_postprocess = Button(fth_step, text=btn_start_postprocess_txt[lang_idx], command=start_postprocess)
btn_start_postprocess.grid(row=row_start_postprocess, column=0, columnspan = 2, sticky='ew')

# set minsize for all rows inside labelframes...
for frame in [fst_step, snd_step, cls_frame, img_frame, vid_frame, fth_step, sep_frame, keep_series_frame, exp_frame, vis_frame]:
    set_minsize_rows(frame)

# ... but not for the hidden rows
snd_step.grid_rowconfigure(row_cls_detec_thresh, minsize=0) # model thresh
snd_step.grid_rowconfigure(row_image_size_for_deploy, minsize=0) # image size for deploy
snd_step.grid_rowconfigure(cls_frame_row, minsize=0) # cls options
snd_step.grid_rowconfigure(img_frame_row, minsize=0) # image options
snd_step.grid_rowconfigure(vid_frame_row, minsize=0) # video options
cls_frame.grid_rowconfigure(row_cls_detec_thresh, minsize=0) # cls animal thresh
# cls_frame.grid_rowconfigure(row_smooth_cls_animal, minsize=0) # cls animal smooth
fth_step.grid_rowconfigure(sep_frame_row, minsize=0) # sep options
sep_frame.grid_rowconfigure(keep_series_frame_row, minsize=0) # keep series options (inside sep_frame)
fth_step.grid_rowconfigure(exp_frame_row, minsize=0) # exp options
fth_step.grid_rowconfigure(vis_frame_row, minsize=0) # vis options

# enable scroll on mousewheel
def deploy_canvas_mousewheel(event):
    if os.name == 'nt':
        deploy_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    else:
        deploy_canvas.yview_scroll(int(-1 * (event.delta / 2)), 'units')

# make deploy_tab scrollable
def bind_scroll_to_deploy_canvas():
    deploy_canvas.update_idletasks()
    deploy_canvas.configure(scrollregion=deploy_canvas.bbox("all"))
    deploy_canvas.bind_all("<MouseWheel>", deploy_canvas_mousewheel)
    deploy_canvas.bind_all("<Button-4>", deploy_canvas_mousewheel)
    deploy_canvas.bind_all("<Button-5>", deploy_canvas_mousewheel)
bind_scroll_to_deploy_canvas()

# help tab
scroll = Scrollbar(help_tab)
help_text = Text(help_tab, width=1, height=1, wrap=WORD, yscrollcommand=scroll.set) 
help_text.configure(spacing1=2, spacing2=3, spacing3=2)
help_text.tag_config('intro', font=f'{text_font} {int(13 * text_size_adjustment_factor)} italic', foreground='black', lmargin1=10, lmargin2=10, underline = False) 
help_text.tag_config('tab', font=f'{text_font} {int(16 * text_size_adjustment_factor)} bold', foreground='black', lmargin1=10, lmargin2=10, underline = True) 
help_text.tag_config('frame', font=f'{text_font} {int(15 * text_size_adjustment_factor)} bold', foreground=green_primary, lmargin1=15, lmargin2=15) 
help_text.tag_config('feature', font=f'{text_font} {int(14 * text_size_adjustment_factor)} normal', foreground='black', lmargin1=20, lmargin2=20, underline = True) 
help_text.tag_config('explanation', font=f'{text_font} {int(13 * text_size_adjustment_factor)} normal', lmargin1=25, lmargin2=25)
hyperlink1 = HyperlinkManager(help_text)

# function to write text which can be called when user changes language settings
def write_help_tab():
    global help_text
    line_number = 1 

    # intro sentence
    help_text.insert(END, ["Below you can find detailed documentation for each setting. If you have any questions, feel free to contact me on ",
                           "A continuación encontrarás documentación detallada sobre cada ajuste. Si tienes alguna pregunta, no dudes en ponerte en contacto conmigo en ",
                           "Ci-dessous, vous trouverez la documentation détaillée pour chaque paramètre. Si vous avez des questions, n'hésitez pas à me contacter (anglais) à l'adresse "][lang_idx])
    help_text.insert(INSERT, "peter@addaxdatascience.com", hyperlink1.add(partial(webbrowser.open, "mailto:peter@addaxdatascience.com")))
    help_text.insert(END, [" or raise an issue on the ", " o plantear una incidencia en ", " ou à rapporter un incident sur la "][lang_idx])
    help_text.insert(INSERT, ["GitHub page", "la página de GitHub", "page GitHub"][lang_idx], hyperlink1.add(partial(webbrowser.open, "https://github.com/PetervanLunteren/AddaxAI/issues")))
    help_text.insert(END, ".\n\n")
    help_text.tag_add('intro', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # first step
    help_text.insert(END, f"{fst_step_txt[lang_idx]}\n")
    help_text.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.insert(END, f"{browse_txt[lang_idx]}\n")
    help_text.insert(END, ["Here you can browse for a folder which contains images and/or video\'s. The model will be deployed on this directory, as well as the post-processing analyses.\n\n",
                           "Aquí puede buscar una carpeta que contenga imágenes y/o vídeos. El modelo se desplegará en este directorio, así como los análisis de post-procesamiento.\n\n",
                           "Ici vous pouvez spécifier un dossier contenant des images et/ou des vidéos. La détection, l'identifications et les analyses de post-traitement s'effectueront sur le contenu de ce répertoire.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # second step
    help_text.insert(END, f"{snd_step_txt[lang_idx]}\n")
    help_text.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1

    # det model
    help_text.insert(END, f"{lbl_model_txt[lang_idx]}\n")
    help_text.insert(END, [
        "AddaxAI uses a combination of a detection model and a classification model to identify animals. The detection model will locate the animal, whereas the "
        "classification model will identify which species the animal belongs to. Here, you can select the detection model that you want to use. If the dropdown "
        "option 'Custom model' is selected, you will be prompted to select a custom YOLOv5 model file. The preloaded 'MegaDetector' models detect animals, people, "
        "and vehicles in camera trap imagery. It does not identify the animals; it just finds them. Version 5a and 5b differ only in their training data. Each may perform "
        "slightly better depending on your specific dataset, so feel free to compare. If you don’t know where to start, stick with the default 'MegaDetector 5a'. "
        "Newer detection models are also available. 'MDv1000-redwood' is a newer, larger model designed to improve detection accuracy and shows very high recall, but "
        "it still needs community testing. Use it if you're willing to explore cutting-edge performance. 'MDv1000-spruce' is a smaller, faster model, but only use it if speed is a key requirement "
        "and you're confident about its trade-offs. In general, speed is rarely a bottleneck anymore, since even modest hardware can process tens of thousands of images per day. "
        "More info about MegaDetector models ",
        
        "AddaxAI utiliza una combinación de un modelo de detección y un modelo de clasificación para identificar animales. El modelo de detección localizará al animal, mientras que el modelo de "
        "clasificación identificará a qué especie pertenece. Aquí puede seleccionar el modelo de detección que desea utilizar. Si selecciona la opción desplegable 'Modelo personalizado', se le "
        "pedirá que seleccione un archivo de modelo YOLOv5 personalizado. Los modelos 'MegaDetector' precargados detectan animales, personas y vehículos en imágenes de cámaras trampa. "
        "No identifican a los animales; solo los encuentran. Las versiones 5a y 5b difieren únicamente en sus datos de entrenamiento. Uno puede funcionar ligeramente mejor que el otro dependiendo de sus datos. "
        "Si no está seguro, utilice el modelo predeterminado 'MegaDetector 5a'. "
        "También hay modelos nuevos disponibles. 'MDv1000-redwood' es un modelo más grande diseñado para mejorar la precisión y tiene un excelente recall, aunque aún requiere más pruebas por la comunidad. "
        "Úselo si quiere probar lo más avanzado. 'MDv1000-spruce' es un modelo más pequeño y rápido, pero sólo se recomienda si la velocidad es una prioridad clara y conoce sus limitaciones. En general, la velocidad rara vez es un cuello de botella, "
        "ya que incluso hardware modesto puede procesar decenas de miles de imágenes por día. "
        "Más información sobre los modelos MegaDetector ",

        "AddaxAI utilise une combinaison de modèles de détection et de classification pour identifier les animaux. Le modèle de détection localise l'animal, tandis que le modèle de classification "
        "détermine à quelle espèce appartient l'animal. Vous pouvez sélectionner ici le modèle de détection souhaité. Si vous sélectionnez l’option « Modèle personnalisé », vous devrez choisir un fichier YOLOv5 personnalisé. "
        "Les modèles 'MegaDetector' préchargés détectent les animaux, les personnes et les véhicules dans les images de pièges photographiques. Ils ne reconnaissent pas les espèces. Les versions 5a et 5b ne diffèrent que par leurs données d'entraînement. "
        "L’un peut être légèrement meilleur que l’autre selon vos données. Si vous hésitez, choisissez le modèle par défaut 'MegaDetector 5a'. "
        "Des modèles plus récents sont également disponibles. 'MDv1000-redwood' est un modèle plus grand conçu pour une meilleure précision et montre un très bon rappel, mais il doit encore être testé par la communauté. "
        "Utilisez-le si vous souhaitez expérimenter des performances de pointe. 'MDv1000-spruce' est plus petit et plus rapide, mais n’est recommandé que si la vitesse est une exigence importante et que vous en acceptez les compromis. "
        "De manière générale, la vitesse n'est presque jamais un facteur limitant, car même du matériel modeste peut traiter des dizaines de milliers d'images par jour. "
        "Plus d'informations sur les modèles MegaDetector "
    ][lang_idx])
    help_text.insert(INSERT, ["here", "aquí", "ici"][lang_idx], hyperlink1.add(partial(webbrowser.open, "https://github.com/ecologize/CameraTraps/blob/main/megadetector.md#megadetector-v50-20220615")))
    help_text.insert(END, ".\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 2

    # cls model
    help_text.insert(END, f"{lbl_cls_model_txt[lang_idx]}\n")
    help_text.insert(END, ["AddaxAI uses a combination of a detection model and a classification model to identify animals. The detection model will locate the animal, whereas the "
                           "classification model will identify which species the animal belongs to. Here, you can select the classification model that you want to use. Each "
                           "classification model is developed for a specific area. Explore which model suits your data best, but please note that models developed for other biomes "
                           "or projects do not necessarily perform equally well in other ecosystems. Always investigate the model’s accuracy on your data before accepting any results.", 
                           "AddaxAI utiliza una combinación de un modelo de detección y un modelo de clasificación para identificar animales. El modelo de detección localizará al "
                           "animal, mientras que el modelo de clasificación identificará a qué especie pertenece el animal. Aquí puede seleccionar el modelo de clasificación que desea "
                           "utilizar. Cada modelo de clasificación se desarrolla para un área específica. Explore qué modelo se adapta mejor a sus datos, pero tenga en cuenta que los "
                           "modelos desarrollados para otros biomas o proyectos no funcionan necesariamente igual de bien en otros ecosistemas. Investiga siempre la precisión del modelo"
                           " en tus datos antes de aceptar cualquier resultado.",
                           "AddaxAI utilise une combinaison de modèles de détection et de classification pour identifier les animaux. Le modèle de détection localise l'animal, tandis "
                           "que le modèle de classification identifie l'espèce à laquelle il appartient. Vous pouvez sélectionner ici le modèle de classification que vous souhaitez "
                           "utiliser. Chaque modèle de classification est développé pour une zone spécifique. Découvrez quel modèle est le plus adapté à vos données, mais veuillez "
                           "noter que les modèles développés pour d'autres biomes ou projets ne sont pas nécessairement aussi performants dans d'autres écosystèmes. Vérifiez toujours "
                           "la précision du modèle sur vos données avant d'accepter les résultats."][lang_idx])
    help_text.insert(END, "\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # cls model info
    help_text.insert(END, f"{lbl_model_info_txt[lang_idx]}\n")
    help_text.insert(END, ["This will open a window with model information.", "Esto abrirá una ventana con información sobre el modelo.", "Ceci ouvrira une fenêtre avec des informations sur le modèle."][lang_idx])
    help_text.insert(END, "\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # cls spp selection
    help_text.insert(END, f"{lbl_choose_classes_txt[lang_idx]}\n")
    help_text.insert(END, ["Here, you can select and deselect the animals categories that are present in your project"
                          " area. If the animal category is not selected, it will be excluded from the results. The "
                          "category list will update according to the model selected.", "Aquí puede seleccionar y anular"
                          " la selección de las categorías de animales presentes en la zona de su proyecto. Si la "
                          "categoría de animales no está seleccionada, quedará excluida de los resultados. La lista de "
                          "categorías se actualizará según el modelo seleccionado.",
                          "Ici, vous pouvez sélectionner et désélectionner les catégories d'animaux présentes dans "
                          "la zone d'étude de votre projet. Si la catégorie d'animal n'est pas sélectionnée, elle sera "
                          "exclue des résultats. La liste des catégories sera mise à jour en fonction du modèle sélectionné."][lang_idx])
    help_text.insert(END, "\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # threshold to classify detections
    help_text.insert(END, f"{lbl_cls_detec_thresh_txt[lang_idx]}\n")
    help_text.insert(END, ["AddaxAI uses a combination of a detection model and a classification model to identify animals. The detection model will locate "
                           "the animal, whereas the classification model will identify which species the animal belongs to. This confidence threshold defines "
                           "which animal detections will be passed on to the classification model for further identification.", "AddaxAI utiliza una "
                           "combinación de un modelo de detección y un modelo de clasificación para identificar a los animales. El modelo de detección "
                           "localizará al animal, mientras que el modelo de clasificación identificará a qué especie pertenece el animal. Este umbral de "
                           "confianza define qué animales detectados se pasarán al modelo de clasificación para su posterior identificación.",
                           "AddaxAI utilise une combinaison de modèles de détection et de classification pour identifier les animaux. Le modèle de détection "
                           "localise l'animal, tandis que le modèle de classification identifie son espèce. Ce seuil de confiance définit les détections "
                           "d'animaux qui seront transmises au modèle de classification pour une identification plus approfondie."][lang_idx])
    help_text.insert(END, "\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # threshold to classify detections
    help_text.insert(END, f"{lbl_cls_class_thresh_txt[lang_idx]}\n")
    help_text.insert(END, ["AddaxAI uses a combination of a detection model and a classification model to identify animals. The detection model will locate "
                           "the animal, whereas the classification model will identify which species the animal belongs to. This confidence threshold defines "
                           "which animal identifications will be accepted.", "AddaxAI utiliza una combinación de un modelo de detección y un modelo de "
                           "clasificación para identificar a los animales. El modelo de detección localizará al animal, mientras que el modelo de clasificación"
                           " identificará a qué especie pertenece el animal. Este umbral de confianza define qué identificaciones de animales se aceptarán.",
                           "AddaxAI utilise une combinaison de modèles de détection et de classification pour identifier les animaux. Le modèle de détection "
                           "localise l'animal, tandis que le modèle de classification identifie son espèce. Ce seuil de confiance définit les classifications "
                           "d'animaux qui seront considérées valides."][lang_idx])
    help_text.insert(END, "\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # smooth results
    help_text.insert(END, f"{lbl_smooth_cls_animal_txt[lang_idx]}\n")
    help_text.insert(END, ["Sequence smoothing averages confidence scores across detections within a sequence to reduce noise. This improves accuracy by "
                           "providing more stable results by combining information over multiple images. Note that it assumes a single species per "
                           "sequence and should therefore only be used if multi-species sequences are rare. It does not affect detections of vehicles or "
                           "people alongside animals.", "El suavizado de secuencias promedia las puntuaciones de confianza entre detecciones dentro de "
                           "una secuencia para reducir el ruido. Esto mejora la precisión al proporcionar resultados más estables mediante la combinación"
                           " de información de múltiples imágenes. Tenga en cuenta que supone una única especie por secuencia y, por lo tanto, sólo debe "
                           "utilizarse si las secuencias multiespecie son poco frecuentes. No afecta a las detecciones de vehículos o personas junto a "
                           "animales.",
                           "Le lissage fait la moyenne des scores de confiance des détections au sein d'une séquence afin de réduire les "
                           "aberrations statistiques. Cela améliore la précision en fournissant des résultats plus stables grâce à la combinaison des "
                           "informations de plusieurs images. Notez que ce lissage suppose une seule espèce par séquence et ne doit donc être utilisé "
                           "que si les séquences multi-espèces sont rares. Il n'affecte pas les détections de véhicules ou de personnes à proximité "
                           "d'animaux."][lang_idx])
    help_text.insert(END, "\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # fallback to higher taxonomy if uncertain
    help_text.insert(END, f"{lbl_tax_fallback_txt[lang_idx]}\n")
    help_text.insert(END, [
        "If enabled, the model will automatically fall back to a higher taxonomic level (e.g., genus or family) when its confidence at the species level is low. "
        "This can improve overall prediction accuracy by avoiding uncertain species-level classifications. Note that some categories may not have species-level predictions at all — "
        "for example, if the model was trained on a broader class like 'bird', it will never predict individual bird species. "
        "This option is only available in models that have been adjusted to support taxonomic fallback.",
        
        "Si está activado, el modelo recurrirá automáticamente a un nivel taxonómico superior (por ejemplo, género o familia) cuando su confianza en el nivel de especie sea baja. "
        "Esto puede mejorar la precisión general al evitar clasificaciones inciertas a nivel de especie. Tenga en cuenta que algunas categorías no tienen predicciones a nivel de especie — "
        "por ejemplo, si el modelo fue entrenado en una clase amplia como 'ave', nunca predecirá especies individuales de aves. "
        "Esta opción solo está disponible en modelos que han sido ajustados para admitir el retroceso taxonómico.",

        "Si cette option est activée, le modèle reviendra automatiquement à un niveau taxonomique supérieur (par exemple, genre ou famille) lorsque son niveau de confiance"
        "de l'espèce est faible. Cela peut améliorer la précision globale des prédictions en évitant les classifications incertaines au niveau de l'espèce. Notez que certaines "
        "catégories peuvent ne pas avoir de prédictions au niveau de l'espèce du tout — par exemple, si le modèle a été entraîné sur une classe plus large comme « oiseau », "
        "il ne prédira jamais les espèces d'oiseaux individuelles. Cette option est uniquement disponible dans les modèles qui ont été ajustés pour prendre en charge l'aggrégation "
        "taxonomique."
    ][lang_idx])
    help_text.insert(END, "\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 2

    # prediction level
    help_text.insert(END, f"{lbl_tax_levels_txt[lang_idx]}\n")
    help_text.insert(END, [
        "This setting allows you to control the granularity of the model's predictions. By default, the model will automatically choose the most appropriate "
        "taxonomic level (e.g., species, genus, or family) depending on its confidence. You can also choose to force predictions to a specific level, "
        "such as always predicting at the family or genus level, if available. This can be useful for applications where fine-grained classification is not needed or where high-level consistency is preferred. "
        "You can also restrict predictions to only those categories that had a minimum number of training samples (e.g., ≥ 10,000). This helps reduce errors caused by underrepresented categories, "
        "but may result in some predictions being skipped entirely if they don’t meet the threshold. "
        "Note that the availability of certain taxonomic levels (e.g., class, genus, species) depends on how the model was trained. If a level is not available for a given detection, "
        "the model will fall back to the closest broader category.",
        
        "Esta opción le permite controlar el nivel de detalle de las predicciones del modelo. Por defecto, el modelo elegirá automáticamente el nivel taxonómico más adecuado "
        "(por ejemplo, especie, género o familia), en función de su confianza y de cómo se entrenó el modelo. También puede optar por forzar las predicciones a un nivel específico, "
        "como predecir siempre a nivel de familia o género, si están disponibles. Esto puede ser útil en aplicaciones en las que no se necesita una clasificación precisa "
        "o se prefiere una consistencia a un nivel superior. "
        "También puede restringir las predicciones a categorías que hayan tenido un número mínimo de muestras de entrenamiento (por ejemplo, ≥ 10,000). Esto ayuda a reducir errores "
        "causados por categorías poco representadas, pero puede dar lugar a que algunas predicciones se omitan si no cumplen ese umbral. "
        "Tenga en cuenta que la disponibilidad de ciertos niveles taxonómicos (por ejemplo, clase, género, especie o grupo de edad y sexo) depende de cómo se entrenó el modelo. "
        "Si un nivel no está disponible para una detección determinada, el modelo recurrirá al nivel más amplio disponible.",

        "Ce paramètre vous permet de contrôler la granularité des prédictions du modèle. Par défaut, le modèle choisit automatiquement le niveau taxonomique le plus approprié (par exemple, espèce, genre "
        "ou famille) en fonction de sa fiabilité. Vous pouvez également forcer les prédictions à un niveau spécifique, par exemple en prédisant toujours au niveau de la famille ou du genre, si disponible."
        " Cela peut être utile pour les applications où une classification fine n'est pas nécessaire ou où une cohérence de haut niveau est privilégiée. Vous pouvez également restreindre les prédictions aux "
        "catégories ayant un nombre minimum d'échantillons d'apprentissage (par exemple, ≥ 10 000). Cela permet de réduire les erreurs causées par des catégories sous-représentées, mais peut entraîner "
        "l'abandon complet de certaines prédictions si elles n'atteignent pas le seuil. Notez que la disponibilité de certains niveaux taxonomiques (par exemple, classe, genre, espèce) dépend de la manière "
        "dont le modèle a été entraîné. Si un niveau n'est pas disponible pour une détection donnée, le modèle reviendra à la catégorie la plus proche."
    ][lang_idx])
    help_text.insert(END, "\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end"); line_number += 2


    # exclude subs
    help_text.insert(END, f"{lbl_exclude_subs_txt[lang_idx]}\n")
    help_text.insert(END, ["By default, AddaxAI will recurse into subdirectories. Select this option if you want to ignore the subdirectories and process only"
                           " the files directly in the chosen folder.\n\n", "Por defecto, AddaxAI buscará en los subdirectorios. Seleccione esta opción si "
                           "desea ignorar los subdirectorios y procesar sólo los archivos directamente en la carpeta elegida.\n\n",
                           "Par défaut, AddaxAI effectue une récursion dans les sous-répertoires. Sélectionnez cette option si vous souhaitez ignorer les "
                           "sous-répertoires et traiter uniquement les fichiers directement dans le dossier choisi.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # exclude detections
    help_text.insert(END, f"{lbl_use_custom_img_size_for_deploy_txt[lang_idx]} / {lbl_image_size_for_deploy_txt[lang_idx]}\n")
    help_text.insert(END, ["AddaxAI will resize the images before they get processed. AddaxAI will by default resize the images to 1280 pixels. "
                    "Deploying a model with a lower image size will reduce the processing time, but also the detection accuracy. Best results are obtained if you use the"
                    " same image size as the model was trained on. If you trained a model in AddaxAI using the default image size, you should set this value to 640 for "
                    "the YOLOv5 models. Use the default for the MegaDetector models.\n\n",
                    "AddaxAI redimensionará las imágenes antes de procesarlas. Por defecto, AddaxAI redimensionará las imágenes a 1280 píxeles. Desplegar un modelo "
                    "con un tamaño de imagen inferior reducirá el tiempo de procesamiento, pero también la precisión de la detección. Los mejores resultados se obtienen "
                    "si se utiliza el mismo tamaño de imagen con el que se entrenó el modelo. Si ha entrenado un modelo en AddaxAI utilizando el tamaño de imagen por "
                    "defecto, debe establecer este valor en 640 para los modelos YOLOv5. Utilice el valor por defecto para los modelos MegaDetector.\n\n",
                    "AddaxAI redimensionne les images avant leur traitement. Par défaut, AddaxAI les redimensionne à 1280 pixels. Déployer un modèle avec une taille "
                    "d'image inférieure réduira le temps de traitement, mais aussi la précision de la détection. De meilleurs résultats sont obtenus en utilisant la "
                    "même taille d'image que celle utilisée pour l'entraînement du modèle. Si vous avez entraîné un modèle dans AddaxAI avec la taille d'image par "
                    "défaut, vous devez définir cette valeur sur 640 pour les modèles YOLOv5. Utilisez la valeur par défaut pour les modèles MegaDetector.\n\n"
                    ][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # use absolute paths
    help_text.insert(END, f"{lbl_abs_paths_txt[lang_idx]}\n")
    help_text.insert(END, ["By default, the paths in the output file are relative (i.e. 'image.jpg') instead of absolute (i.e. '/path/to/some/folder/image.jpg'). This "
                    "option will make sure the output file contains absolute paths, but it is not recommended. Third party software (such as ",
                    "Por defecto, las rutas en el archivo de salida son relativas (es decir, 'imagen.jpg') en lugar de absolutas (es decir, '/ruta/a/alguna/carpeta/"
                    "imagen.jpg'). Esta opción se asegurará de que el archivo de salida contenga rutas absolutas, pero no se recomienda. Software de terceros (como ",
                    "Par défaut, les chemins du fichier de sortie sont relatifs (par exemple, « image.jpg ») et non absolus (par exemple, "
                    "« /chemin/vers/un/dossier/image.jpg »). Cette option garantit que le fichier de sortie contient des chemins absolus, mais elle n'est pas "
                    "recommandée. Les logiciels tiers (tel que « "][lang_idx])
    help_text.insert(INSERT, "Timelapse", hyperlink1.add(partial(webbrowser.open, "https://timelapse.ucalgary.ca/")))
    help_text.insert(END, [") will not be able to read the output file if the paths are absolute. Only enable this option if you know what you are doing. More information"
                    " how to use Timelapse in conjunction with MegaDetector, see the ",
                    ") no serán capaces de leer el archivo de salida si las rutas son absolutas. Solo active esta opción si sabe lo que está haciendo. Para más información"
                    " sobre cómo utilizar Timelapse junto con MegaDetector, consulte ",
                    " », de Saul Greenberg, University of Calgary) ne pourra pas lire le fichier de sortie si les chemins sont absolus. N'activez cette option que si vous savez ce que vous faites. Pour plus "
                    "d'informations, consultez la section « Comment utiliser Timelapse avec MegaDetector ? ». "][lang_idx])
    help_text.insert(INSERT, ["Timelapse Image Recognition Guide", "la Guía de Reconocimiento de Imágenes de Timelapse", "Guide de reconnaissance avec Timelapse (anglais)"
    ][lang_idx], hyperlink1.add(partial(webbrowser.open, "https://timelapse.ucalgary.ca/wp-content/uploads/Guides/TimelapseImageRecognitionGuide.pdf")))
    help_text.insert(END, ".\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # use checkpoints
    help_text.insert(END, f"{lbl_use_checkpnts_txt[lang_idx]}\n")
    help_text.insert(END, ["This is a functionality to save results to checkpoints intermittently, in case a technical hiccup arises. That way, you won't have to restart"
                    " the entire process again when the process is interrupted.\n\n",
                    "Se trata de una funcionalidad para guardar los resultados en puntos de control de forma intermitente, en caso de que surja un contratiempo técnico. "
                    "De esta forma, no tendrás que reiniciar todo el proceso de nuevo cuando éste se interrumpa.\n\n",
                    "Cette fonctionnalité permet d'enregistrer les résultats dans des points de contrôle de manière intermittente, en cas de problème technique. Ainsi, "
                    "vous n'aurez pas à redémarrer le processus en cas d'interruption.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # checkpoint frequency
    help_text.insert(END, f"{lbl_checkpoint_freq_txt[lang_idx]}\n")
    help_text.insert(END, ["Fill in how often you want to save the results to checkpoints. The number indicates the number of images after which checkpoints will be saved."
                    " The entry must contain only numeric characters.\n\n",
                    "Introduzca la frecuencia con la que desea guardar los resultados en los puntos de control. El número indica el número de imágenes tras las cuales se "
                    "guardarán los puntos de control. La entrada debe contener sólo caracteres numéricos.\n\n",
                    "Indiquez la fréquence à laquelle vous souhaitez enregistrer les résultats dans les points de contrôle. Le nombre indique le nombre d'images après "
                    "lequel les points de contrôle seront enregistrés. L'entrée doit contenir uniquement des caractères numériques.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # continue from checkpoint
    help_text.insert(END, f"{lbl_cont_checkpnt_txt[lang_idx]}\n")
    help_text.insert(END, ["Here you can choose to continue from the last saved checkpoint onwards so that the algorithm can continue where it left off. Checkpoints are"
                    " saved into the main folder and look like 'checkpoint_<timestamp>.json'. When choosing this option, it will search for a valid"
                    " checkpoint file and prompt you if it can't find it.\n\n",
                    "Aquí puede elegir continuar desde el último punto de control guardado para que el algoritmo pueda continuar donde lo dejó. Los puntos de control se "
                    "guardan en la carpeta principal y tienen el aspecto 'checkpoint_<fecha y hora>.json'. Al elegir esta opción, se buscará un archivo de punto de control "
                    "válido y se le preguntará si no puede encontrarlo.\n\n",
                    "Ici, vous pouvez choisir de continuer à partir du dernier point de contrôle enregistré afin que l'algorithme puisse reprendre là où il s'est arrêté. "
                    "Les points de contrôle sont enregistrés dans le dossier principal et ressemblent à « checkpoint_<horodatage>.json ». Lorsque vous choisissez cette "
                    "option, l'algorithme recherche un fichier de point de contrôle valide et vous avertit s'il ne le trouve pas.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # don't process every frame
    help_text.insert(END, f"{lbl_not_all_frames_txt[lang_idx]}\n")
    help_text.insert(END,["When processing every frame of a video, it can take a long time to finish. Here, you can specify whether you want to analyse only a selection of frames."
                    f" At '{lbl_nth_frame_txt[lang_idx]}' you can specify how many frames you want to be analysed.\n\n",
                     "Procesar todos los fotogramas de un vídeo puede llevar mucho tiempo. Aquí puede especificar si desea analizar sólo una selección de fotogramas. "
                    f"En '{lbl_nth_frame_txt[lang_idx]}' puedes especificar cuántos fotogramas quieres que se analicen.\n\n",
                    "Le traitement de chaque image d'une vidéo peut prendre un certain temps. Vous pouvez ici spécifier si vous souhaitez analyser uniquement une sélection d'images."
                    f" L'option « {lbl_nth_frame_txt[lang_idx]} » permet de spécifier combien de trames (images) vous souhaitez analyser.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # analyse every nth frame
    help_text.insert(END, f"{lbl_nth_frame_txt[lang_idx]}\n")
    help_text.insert(END, ["Specify the frame sampling rate you'd like to use. For example, entering '1' will process one frame per second. Typically, sampling one frame per second is sufficient and can significantly reduce processing time. The exact time savings depend on the video's frame rate. Most camera traps record at 30 frames per second, meaning this approach can reduce processing time by 97% compared to processing every frame.\n\n",
                    "Especifica la tasa de muestreo de fotogramas que deseas utilizar. Por ejemplo, ingresar '1' procesará un fotograma por segundo. Generalmente, muestrear un fotograma por segundo es suficiente y puede reducir significativamente el tiempo de procesamiento. El ahorro exacto de tiempo depende de la tasa de fotogramas del video. La mayoría de las cámaras trampa graban a 30 fotogramas por segundo, lo que significa que este enfoque puede reducir el tiempo de procesamiento aproximadamente en un 97% en comparación con procesar todos los fotogramas.\n\n",
                    "Spécifiez la fréquence d'échantillonnage d'images que vous souhaitez utiliser. Par exemple, saisir « 1 » traitera une image par seconde. En général, un échantillonnage d'une image par seconde est suffisant et peut réduire considérablement le temps de traitement. Le gain de temps exact dépend de la fréquence d'images de la vidéo. La plupart des pièges photographiques enregistrent à 30 images par seconde, ce qui signifie que cette approche peut réduire le temps de traitement de 97 % par rapport au traitement de chaque image.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # third step
    help_text.insert(END, f"{trd_step_txt[lang_idx]}\n")
    help_text.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1

    # human verification
    help_text.insert(END, f"{lbl_hitl_main_txt[lang_idx]}\n")
    help_text.insert(END, ["This feature lets you verify the results of the model. You can use it to create training data or to double-check the results. When starting a new "
                           "session, you will first be directed to a window where you can select which images you would like to verify. For instance, someone might be only "
                           "interested in creating training data for 'class A' to unbalance his training dataset or only want to double-check detections with medium-sure "
                           "confidences. After you have selected the images, you will be able to verify them. After having verified all selected images, you will be prompted"
                           " if you want to create training data. If you do, the selected images and their associated annotation files will get a unique name and be either "
                           "moved or copied to a folder of your choice. This is particularly handy when you want to create training data since, for training, all files must "
                           "be in one folder. This way, the files will be unique, and you won't have replacement problems when adding the files to your existing training data. "
                           "You can also skip the training data and just continue to post-process the verified results. Not applicable to videos.\n\n",
                           "Esta característica le permite verificar los resultados del modelo. Puedes usarlo para crear datos de entrenamiento o para verificar los resultados. "
                           "Al iniciar una nueva sesión, primero se le dirigirá a una ventana donde podrá seleccionar qué imágenes desea verificar. Por ejemplo, alguien podría "
                           "estar interesado únicamente en crear datos de entrenamiento para la 'clase A' para desequilibrar su conjunto de datos de entrenamiento o simplemente "
                           "querer verificar las detecciones con confianzas medias-seguras. Una vez que hayas seleccionado las imágenes, podrás verificarlas. Después de haber "
                           "verificado todas las imágenes seleccionadas, se le preguntará si desea crear datos de entrenamiento. Si lo hace, las imágenes seleccionadas y sus "
                           "archivos de anotaciones asociados obtendrán un nombre único y se moverán o copiarán a una carpeta de su elección. Esto es particularmente útil cuando"
                           " desea crear datos de entrenamiento ya que, para el entrenamiento, todos los archivos deben estar en una carpeta. De esta manera, los archivos serán "
                           "únicos y no tendrás problemas de reemplazo al agregar los archivos a tus datos de entrenamiento existentes. También puedes omitir los datos de "
                           "entrenamiento y simplemente continuar con el posprocesamiento de los resultados verificados. No aplicable a vídeos.\n\n",
                           "Cette fonctionnalité vous permet de vérifier les résultats du modèle. Vous pouvez l'utiliser pour créer des données d'entraînement ou pour revérifier "
                           "les résultats. Lorsque vous démarrez une nouvelle session, vous serez d'abord redirigé vers une fenêtre vous permettant de sélectionner les images à "
                           "vérifier. Par exemple, quelqu'un pourrait souhaiter créer uniquement des données d'entraînement pour la « classe A » afin de déséquilibrer son "
                           "ensemble de données d'entraînement, ou seulement revérifier les détections avec un niveau de confiance moyen. Après avoir sélectionné les images, "
                           "vous pourrez les vérifier. Après avoir vérifié toutes les images sélectionnées, vous serez invité à créer des données d'entraînement. Dans ce cas, "
                           "les images sélectionnées et leurs fichiers d'annotation associés recevront un nom unique et seront déplacés ou copiés dans le dossier de votre choix."
                           " Ceci est particulièrement pratique pour créer des données d'entraînement, car pour l'entraînement, tous les fichiers doivent se trouver dans un seul "
                           "dossier. Ainsi, les fichiers seront uniques et vous n'aurez pas de problèmes de remplacement lors de l'ajout de fichiers à vos données d'entraînement "
                           "existantes. Vous pouvez également ignorer les données d'entraînement et simplement continuer à post-traiter les résultats vérifiés. Non applicable "
                           "aux vidéos.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # forth step
    help_text.insert(END, f"{fth_step_txt[lang_idx]}\n")
    help_text.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1

    # destination folder
    help_text.insert(END, f"{lbl_output_dir_txt[lang_idx]}\n")
    help_text.insert(END, ["Here you can browse for a folder in which the results of the post-processing features will be placed. If nothing is selected, the folder "
                    "chosen at step one will be used as the destination folder.\n\n",
                    "Aquí puede buscar una carpeta en la que se colocarán los resultados de las funciones de postprocesamiento. Si no se selecciona nada, la carpeta "
                    "elegida en el primer paso se utilizará como carpeta de destino.\n\n",
                    "Vous pouvez ici sélectionner le dossier dans lequel seront placés les résultats des fonctions de post-traitement. Si rien n'est sélectionné, le "
                    "dossier choisi à l'étape 1 sera utilisé comme dossier de destination.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # separate files
    help_text.insert(END, f"{lbl_separate_files_txt[lang_idx]}\n")
    help_text.insert(END, ["This function divides the files into subdirectories based on their detections. Please be warned that this will be done automatically. "
                    "There will not be an option to review and adjust the detections before the images will be moved. If you want that (a human in the loop), take a look at ",
                    "Esta función divide los archivos en subdirectorios en función de sus detecciones. Tenga en cuenta que esto se hará automáticamente. No habrá opción de "
                    "revisar y ajustar las detecciones antes de mover las imágenes. Si quieres eso (una humano en el bucle), echa un vistazo a ",
                    "Cette fonction divise les fichiers en sous-répertoires en fonction de leurs détections. Veuillez noter que cette opération sera effectuée "
                    "automatiquement. Il n'y aura pas d'option permettant de vérifier et d'ajuster les détections avant le déplacement des images. Si vous souhaitez une "
                    "intervention humaine, consultez la section "][lang_idx])
    help_text.insert(INSERT, "Timelapse", hyperlink1.add(partial(webbrowser.open, "https://saul.cpsc.ucalgary.ca/timelapse/")))
    help_text.insert(END, [", which offers such a feature. More information about that ",
                           ", que ofrece tal característica. Más información al respecto ",
                           ", qui offre une telle fonctionnalité. Plus d'informations à ce sujet "][lang_idx])
    help_text.insert(INSERT, ["here", "aquí", "ici"][lang_idx], hyperlink1.add(partial(webbrowser.open, "https://saul.cpsc.ucalgary.ca/timelapse/uploads/Guides/TimelapseImageRecognitionGuide.pdf")))
    help_text.insert(END,[" (starting on page 9). The process of importing the output file produced by AddaxAI into Timelapse is described ",
                          " (a partir de la página 9). El proceso de importación del archivo de salida producido por AddaxAI en Timelapse se describe ",
                          " (à partir de la page 9). Le processus d'importation du fichier de sortie produit par AddaxAI dans Timelapse est décrit "][lang_idx])
    help_text.insert(INSERT, ["here", "aquí", "ici"][lang_idx], hyperlink1.add(partial(webbrowser.open, "https://saul.cpsc.ucalgary.ca/timelapse/pmwiki.php?n=Main.DownloadMegadetector")))
    help_text.insert(END,".\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # method of file placement
    help_text.insert(END, f"{lbl_file_placement_txt[lang_idx]}\n")
    help_text.insert(END, ["Here you can choose whether to move the files into subdirectories, or copy them so that the originals remain untouched.\n\n",
                           "Aquí puedes elegir si quieres mover los archivos a subdirectorios o copiarlos de forma que los originales permanezcan intactos.\n\n",
                           "Ici, vous pouvez choisir de DÉPLACER les fichiers dans des sous-répertoires ou de les COPIER afin que les originaux restent intacts.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # sort results based on confidence
    help_text.insert(END, f"{lbl_sep_conf_txt[lang_idx]}\n")
    help_text.insert(END, ["This feature will further separate the files based on its confidence value (in tenth decimal intervals). That means that each class will"
                        " have subdirectories like e.g. 'conf_0.6-0.7', 'conf_0.7-0.8', 'conf_0.8-0.9', etc.\n\n",
                        "Esta función separará aún más los archivos en función de su valor de confianza (en intervalos decimales). Esto significa que cada clase tendrá"
                        " subdirectorios como, por ejemplo, 'conf_0.6-0.7', 'conf_0.7-0.8', 'conf_0.8-0.9', etc.\n\n",
                        "Cette fonctionnalité permet de séparer les fichiers selon leur niveau de confiance (au dixième près). Cela signifie que chaque classe aura "
                        "des sous-répertoires tels que « conf_0.6-0.7 », « conf_0.7-0.8 », « conf_0.8-0.9 », etc.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # visualize files
    help_text.insert(END, f"{lbl_vis_files_txt[lang_idx]}\n")
    help_text.insert(END, ["This functionality draws boxes around the detections and prints their confidence values. This can be useful to visually check the results."
                    " Videos can't be visualized using this tool. Please be aware that this action is permanent and cannot be undone. Be wary when using this on original images.\n\n",
                    "Esta funcionalidad dibuja recuadros alrededor de las detecciones e imprime sus valores de confianza. Esto puede ser útil para comprobar visualmente los "
                    "resultados. Los vídeos no pueden visualizarse con esta herramienta.\n\n",
                    "Cette fonctionnalité encadre les détections et affiche leurs scores de confiance. Cela peut être utile pour vérifier visuellement les résultats. Les vidéos "
                    "ne peuvent pas être visualisées avec cet outil. Veuillez noter que cette action est permanente et irréversible. Soyez prudent lorsque vous l'utilisez sur "
                    "des images originales.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # crop files
    help_text.insert(END, f"{lbl_crp_files_txt[lang_idx]}\n")
    help_text.insert(END, ["This feature will crop the detections and save them as separate images. Not applicable for videos.\n\n",
                           "Esta función recortará las detecciones y las guardará como imágenes separadas. No es aplicable a los vídeos.\n\n",
                           "Cette fonctionnalité recadre les détections et les enregistre sous forme d'images distinctes. Non applicable aux vidéos.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # plot graphs
    help_text.insert(END, f"{lbl_plt_txt[lang_idx]}\n")
    help_text.insert(END, ["Here you can select to create activity patterns, bar charts, pie charts and temporal heatmaps. The time unit (year, month, "
                           "week or day) will be chosen automatically based on the time period of your data. If more than 100 units are needed to "
                           "visualize, they will be skipped due to long processing times. Each visualization results in a static PNG file and a dynamic"
                           " HTML file to explore the data further. Additional interactive maps will be produced when geotags can be retrieved from the "
                           "image metadata.", "Aquí puede seleccionar la creación de patrones de actividad, gráficos de barras, gráficos circulares y "
                           "mapas térmicos temporales. La unidad temporal (año, mes, semana o día) se elegirá automáticamente en función del periodo de"
                           " tiempo de sus datos. Si se necesitan más de 100 unidades para visualizar, se omitirán debido a los largos tiempos de "
                           "procesamiento. Cada visualización da como resultado un archivo PNG estático y un archivo HTML dinámico para explorar más a "
                           "fondo los datos. Se producirán mapas interactivos adicionales cuando se puedan recuperar geoetiquetas de los metadatos de "
                           "las imágenes.",
                           "Vous pouvez ici créer des modèles d'activité, des graphiques à barres, des diagrammes circulaires et des cartes thermiques "
                           "temporelles. L'unité de temps (année, mois, semaine ou jour) sera automatiquement choisie en fonction de la période de vos "
                           "données. Si plus de 100 unités sont nécessaires à la visualisation, elles seront ignorées en raison des longs délais de "
                           "traitement. Chaque visualisation génère un fichier PNG statique et un fichier HTML dynamique pour une exploration plus "
                           "approfondie des données. Des cartes interactives supplémentaires seront générées lorsque les géotags pourront être "
                           "récupérés à partir des métadonnées de l'image."][lang_idx])
    help_text.insert(END, "\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # export results
    help_text.insert(END, f"{lbl_exp_txt[lang_idx]}\n")
    help_text.insert(END, ["Here you can select whether you want to export the results to other file formats. It will additionally try to fetch image metadata, like "
                           "timestamps, locations, and more.\n\n", "Aquí puede seleccionar si desea exportar los resultados a otros formatos de archivo. Además, "
                           "intentará obtener metadatos de la imagen, como marcas de tiempo, ubicaciones, etc. \n\n",
                           "Ici, vous pouvez choisir d'exporter les résultats vers d'autres formats de fichier. L'outil essaiera également de récupérer les "
                           "métadonnées de l'image, comme les horodatages, les emplacements, etc.\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # postprocess confidence threshold
    help_text.insert(END, f"{lbl_thresh_txt[lang_idx]}\n")
    help_text.insert(END, ["Detections below this value will not be post-processed. To adjust the threshold value, you can drag the slider or press either sides next to "
                    "the slider for a 0.01 reduction or increment. Confidence values are within the [0.01, 1] interval. If you set the confidence threshold too high, "
                    "you will miss some detections. On the other hand, if you set the threshold too low, you will get false positives. When choosing a threshold for your "
                    f"project, it is important to choose a threshold based on your own data. My advice is to first visualize your data ('{lbl_vis_files_txt[lang_idx]}') with a low "
                    "threshold to get a feeling of the confidence values in your data. This will show you how sure the model is about its detections and will give you an "
                    "insight into which threshold will work best for you. If you really don't know, 0.2 is probably a conservative threshold for most projects.\n\n",
                    "Las detecciones por debajo de este valor no se postprocesarán. Para ajustar el valor del umbral, puede arrastrar el control deslizante o pulsar "
                    "cualquiera de los lados junto al control deslizante para una reducción o incremento de 0,01. Los valores de confianza están dentro del intervalo "
                    "[0,01, 1]. Si ajusta el umbral de confianza demasiado alto, pasará por alto algunas detecciones. Por otro lado, si fija el umbral demasiado bajo, "
                    "obtendrá falsos positivos. Al elegir un umbral para su proyecto, es importante elegir un umbral basado en sus propios datos. Mi consejo es que primero"
                    f" visualice sus datos ('{lbl_vis_files_txt[lang_idx]}') con un umbral bajo para hacerse una idea de los valores de confianza de sus datos. Esto le mostrará lo "
                    "seguro que está el modelo sobre sus detecciones y le dará una idea de qué umbral funcionará mejor para usted. Si realmente no lo sabe, 0,2 es "
                    "probablemente un umbral conservador para la mayoría de los proyectos.\n\n",
                    "Les détections inférieures à cette valeur ne seront pas traitées ultérieurement. Pour ajuster la valeur seuil, faites glisser le curseur ou appuyez sur l'un "
                    "des côtés de la barre de défilement par incrément ou décrément de 0.01. Les valeurs de confiance sont comprises entre [0,01 et 1]. Si le seuil de confiance "
                    "est trop élevé, vous manquerez certaines détections. En revanche, si vous définissez un seuil trop bas, vous obtiendrez des faux positifs. Lorsque vous "
                    "choisissez un seuil pour votre projet, il est important de choisir un seuil en fonction de vos propres données. Je vous conseille de commencer par visualiser "
                    f"vos données ('{lbl_vis_files_txt[lang_idx]}') avec un seuil bas pour obtenir une idée du niveau de confiance dans vos données. Cela vous montrera à quel "
                    "point le modèle est sûr de ses détections et vous donnera une intuition du seuil le plus adapté à vos besoins. Si vous ne le savez pas vraiment, 0,2 est "
                    "probablement un seuil prudent pour la plupart des projets..\n\n"][lang_idx])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # config help_text
    help_text.pack(fill="both", expand=True)
    help_text.configure(font=(text_font, 11, "bold"), state=DISABLED)
    scroll.configure(command=help_text.yview)
write_help_tab()

# about tab
about_scroll = Scrollbar(about_tab)
about_text = Text(about_tab, width=1, height=1, wrap=WORD, yscrollcommand=scroll.set)
about_text.configure(spacing1=2, spacing2=3, spacing3=2)
about_text.tag_config('title', font=f'{text_font} {int(15 * text_size_adjustment_factor)} bold', foreground=green_primary, lmargin1=10, lmargin2=10) 
about_text.tag_config('info', font=f'{text_font} {int(13 * text_size_adjustment_factor)} normal', lmargin1=20, lmargin2=20)
about_text.tag_config('citation', font=f'{text_font} {int(13 * text_size_adjustment_factor)} normal', lmargin1=30, lmargin2=50)
hyperlink = HyperlinkManager(about_text)

# function to write text which can be called when user changes language settings
def write_about_tab():
    global about_text
    text_line_number=1

    # contact
    about_text.insert(END, ["Contact\n", "Contacto\n", "Contact\n"][lang_idx])
    about_text.insert(END, ["Please also help me to keep improving AddaxAI and let me know about any improvements, bugs, or new features so that I can keep it up-to-date. You can "
                           "contact me at ",
                           "Por favor, ayúdame también a seguir mejorando AddaxAI e infórmame de cualquier mejora, error o nueva función para que pueda mantenerlo actualizado. "
                           "Puedes ponerte en contacto conmigo en ",
                           "Merci de m'aider à améliorer AddaxAI et de me signaler toute amélioration, bogue ou nouvelle fonctionnalité afin que je puisse le maintenir à jour."
                           " Vous pouvez me contacter à l'adresse suivante: "][lang_idx])
    about_text.insert(INSERT, "peter@addaxdatascience.com", hyperlink.add(partial(webbrowser.open, "mailto:peter@addaxdatascience.com")))
    about_text.insert(END, [" or raise an issue on the ", " o plantear un problema en ", " ou rapporter un incident sur la "][lang_idx])
    about_text.insert(INSERT, ["GitHub page", "la página de GitHub", "page GitHub"][lang_idx], hyperlink.add(partial(webbrowser.open, "https://github.com/PetervanLunteren/AddaxAI/issues")))
    about_text.insert(END, ".\n\n")
    about_text.tag_add('title', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('info', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=2

    # addaxai citation
    about_text.insert(END, ["AddaxAI citation\n", "Citar AddaxAI\n", "Citation AddaxAI\n"][lang_idx])
    about_text.insert(END, ["If you used AddaxAI in your research, please use the following citations. The AddaxAI software was previously called 'EcoAssist'.\n",
                            "Si ha utilizado AddaxAI en su investigación, utilice la siguiente citas. AddaxAI se llamaba antes 'EcoAssist'.\n",
                            "Si vous avez utilisé AddaxAI dans vos recherches, veuillez utiliser les citations suivantes. Le logiciel AddaxAI s'appelait auparavant « EcoAssist ».\n"][lang_idx])
    about_text.insert(END, "- van Lunteren, P., (2023). AddaxAI: A no-code platform to train and deploy custom YOLOv5 object detection models. Journal of Open Source Software, 8(88), 5581, https://doi.org/10.21105/joss.05581")
    about_text.insert(INSERT, "https://doi.org/10.21105/joss.05581", hyperlink.add(partial(webbrowser.open, "https://doi.org/10.21105/joss.05581")))
    about_text.insert(END, ".\n")
    about_text.insert(END, ["- Plus the citation of the models used.\n\n", "- Más la cita de los modelos utilizados.\n\n", "et ajoutez la citation du modèle utilisé.\n\n"][lang_idx]    )
    about_text.tag_add('title', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('info', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('citation', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('citation', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=2

    # development credits
    about_text.insert(END, ["Development\n", "Desarrollo\n", "Développement\n"][lang_idx])
    about_text.insert(END, ["AddaxAI is developed by ",
                            "AddaxAI ha sido desarrollado por ",
                            "AddaxAI est développé par "][lang_idx])
    about_text.insert(INSERT, "Addax Data Science", hyperlink.add(partial(webbrowser.open, "https://addaxdatascience.com/")))
    about_text.insert(END, [" in collaboration with ",
                            " en colaboración con ",
                            " en collaboration avec "][lang_idx])
    about_text.insert(INSERT, "Smart Parks", hyperlink.add(partial(webbrowser.open, "https://www.smartparks.org/")))
    about_text.insert(END, ".\n\n")
    about_text.tag_add('title', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('info', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=2

    # config about_text
    about_text.pack(fill="both", expand=True)
    about_text.configure(font=(text_font, 11, "bold"), state=DISABLED)
    scroll.configure(command=about_text.yview)
write_about_tab()

# SIMPLE MODE WINDOW 
dir_image = customtkinter.CTkImage(PIL_dir_image, size=(ICON_SIZE, ICON_SIZE))
mdl_image = customtkinter.CTkImage(PIL_mdl_image, size=(ICON_SIZE, ICON_SIZE))
spp_image = customtkinter.CTkImage(PIL_spp_image, size=(ICON_SIZE, ICON_SIZE))
run_image = customtkinter.CTkImage(PIL_run_image, size=(ICON_SIZE, ICON_SIZE))

# set the global appearance for the app
customtkinter.set_appearance_mode("light")
customtkinter.set_default_color_theme(os.path.join(AddaxAI_files, "AddaxAI", "themes", "addaxai.json"))

# set up window
simple_mode_win = customtkinter.CTkToplevel(root)
simple_mode_win.title(f"AddaxAI v{current_AA_version} - "+["Simple mode", "Modo simple", "Mode simple"][lang_idx])
simple_mode_win.geometry("+20+20")
simple_mode_win.protocol("WM_DELETE_WINDOW", on_toplevel_close)
simple_mode_win.columnconfigure(0, weight=1, minsize=500)
main_label_font = customtkinter.CTkFont(family='CTkFont', size=14, weight = 'bold')
simple_bg_image = customtkinter.CTkImage(PIL_sidebar, size=(SIM_WINDOW_WIDTH, SIM_WINDOW_HEIGHT))
simple_bg_image_label = customtkinter.CTkLabel(simple_mode_win, image=simple_bg_image)
simple_bg_image_label.grid(row=0, column=0)
simple_main_frame = customtkinter.CTkFrame(simple_mode_win, corner_radius=0, fg_color = 'transparent')
simple_main_frame.grid(row=0, column=0, sticky="ns")
simple_mode_win.withdraw() # only show when all widgets are loaded

# logo
sim_top_banner = customtkinter.CTkImage(PIL_logo_incl_text, size=(LOGO_WIDTH, LOGO_HEIGHT))
customtkinter.CTkLabel(simple_main_frame, text="", image = sim_top_banner).grid(column=0, row=0, columnspan=2, sticky='ew', pady=(PADY, 0), padx=0)

# top buttons
sim_btn_switch_mode_txt = ["To advanced mode", "Al modo avanzado", "Mode avancé"]
sim_btn_switch_mode = GreyTopButton(master = simple_main_frame, text = sim_btn_switch_mode_txt[lang_idx], command = switch_mode)
sim_btn_switch_mode.grid(row=0, column=0, padx=PADX, pady=(PADY, 0), columnspan = 2, sticky="nw")
sim_btn_switch_lang = GreyTopButton(master = simple_main_frame, text = "Switch language", command = set_language)
sim_btn_switch_lang.grid(row=0, column=0, padx=PADX, pady=(0, 0), columnspan = 2, sticky="sw")
sim_btn_sponsor = GreyTopButton(master = simple_main_frame, text = adv_btn_sponsor_txt[lang_idx], command = sponsor_project)
sim_btn_sponsor.grid(row=0, column=0, padx=PADX, pady=(PADY, 0), columnspan = 2, sticky="ne")
sim_btn_reset_values = GreyTopButton(master = simple_main_frame, text = adv_btn_reset_values_txt[lang_idx], command = reset_values)
sim_btn_reset_values.grid(row=0, column=0, padx=PADX, pady=(0, 0), columnspan = 2, sticky="se")

# choose folder
sim_dir_frm_1 = MyMainFrame(master=simple_main_frame)
sim_dir_frm_1.grid(row=2, column=0, padx=PADX, pady=PADY, sticky="nswe")
sim_dir_img_widget = customtkinter.CTkLabel(sim_dir_frm_1, text="", image = dir_image, compound = 'left')
sim_dir_img_widget.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="nswe")
sim_dir_frm = MySubFrame(master=sim_dir_frm_1)
sim_dir_frm.grid(row=0, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
sim_dir_lbl_txt = ["Which folder do you want to analyse?", "¿Qué carpeta quieres analizar?", "Quel dossier voulez-vous analyser?"]
sim_dir_lbl = customtkinter.CTkLabel(sim_dir_frm, text=sim_dir_lbl_txt[lang_idx], font = main_label_font)
sim_dir_lbl.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), columnspan = 2, sticky="nsw")
sim_dir_inf = InfoButton(master = sim_dir_frm, text = "?", command = sim_dir_show_info)
sim_dir_inf.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="e", columnspan = 2)
sim_dir_btn = customtkinter.CTkButton(sim_dir_frm, text=browse_txt[lang_idx], width = 1, command = lambda: [browse_dir(var_choose_folder, var_choose_folder_short, dsp_choose_folder, 25, row_choose_folder, 0, 'w', source_dir = True), update_frame_states()])
sim_dir_btn.grid(row=1, column=0, padx=(PADX, PADX/2), pady=(0, PADY), sticky="nswe")
sim_dir_pth_frm = MySubSubFrame(master=sim_dir_frm)
sim_dir_pth_frm.grid(row=1, column=1, padx=(PADX/2, PADX), pady=(0, PADY), sticky="nesw")
sim_dir_pth_txt = ["no folder selected", "no hay carpeta seleccionada", "aucun dossier sélectionné"]
sim_dir_pth = customtkinter.CTkLabel(sim_dir_pth_frm, text=sim_dir_pth_txt[lang_idx], text_color = "grey")
sim_dir_pth.pack()

# choose model
sim_mdl_frm_1 = MyMainFrame(master=simple_main_frame)
sim_mdl_frm_1.grid(row=3, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
sim_mdl_img_widget = customtkinter.CTkLabel(sim_mdl_frm_1, text="", image = mdl_image, compound = 'left')
sim_mdl_img_widget.grid(row=1, column=0, padx=PADX, pady=PADY, sticky="nswe")
sim_mdl_frm = MySubFrame(master=sim_mdl_frm_1)
sim_mdl_frm.grid(row=1, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
sim_mdl_lbl_txt = ["Which species identification model do you want to use?", "¿Qué modelo de identificación de especies quiere utilizar?", "Quel modèle de classification d'espèces souhaitez-vous utiliser?"]
sim_mdl_lbl = customtkinter.CTkLabel(sim_mdl_frm, text=sim_mdl_lbl_txt[lang_idx], font = main_label_font)
sim_mdl_lbl.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), columnspan = 2, sticky="nsw")
sim_mdl_inf = InfoButton(master = sim_mdl_frm, text = "?", command = sim_mdl_show_info)
sim_mdl_inf.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="e", columnspan = 2)
# convert to more elaborate dpd value for the 'None' simple mode option
sim_dpd_options_cls_model = [[item[0] + suffixes_for_sim_none[i], *item[1:]] for i, item in enumerate(dpd_options_cls_model)]
sim_mdl_dpd = customtkinter.CTkOptionMenu(sim_mdl_frm, values=sim_dpd_options_cls_model[lang_idx], command=sim_mdl_dpd_callback, width = 1)
sim_mdl_dpd.set(sim_dpd_options_cls_model[lang_idx][global_vars["var_cls_model_idx"]]) # take idx instead of string
sim_mdl_dpd.grid(row=1, column=0, padx=PADX, pady=(PADY/4, PADY), sticky="nswe", columnspan = 2)

# select animals
sim_spp_frm_1 = MyMainFrame(master=simple_main_frame)
sim_spp_frm_1.grid(row=4, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
sim_spp_img_widget = customtkinter.CTkLabel(sim_spp_frm_1, text="", image = spp_image, compound = 'left')
sim_spp_img_widget.grid(row=2, column=0, padx=PADX, pady=PADY, sticky="nswe")
sim_spp_frm = MySubFrame(master=sim_spp_frm_1, width=1000)
sim_spp_frm.grid(row=2, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
sim_spp_lbl_txt = ["Which species are present in your project area?", "¿Qué especies están presentes en la zona de su proyecto?", "Quelles espèces sont présentes dans la zone d'étude de votre projet?"]
sim_spp_lbl = customtkinter.CTkLabel(sim_spp_frm, text=sim_spp_lbl_txt[lang_idx], font = main_label_font, text_color = 'grey')
sim_spp_lbl.grid(row=0, column=0, padx=PADX, pady=(0, PADY/4), columnspan = 2, sticky="nsw")
sim_spp_inf = InfoButton(master = sim_spp_frm, text = "?", command = sim_spp_show_info)
sim_spp_inf.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="e", columnspan = 2)
sim_spp_scr_height = 238
sim_spp_scr = SpeciesSelectionFrame(master=sim_spp_frm, height=sim_spp_scr_height, dummy_spp = True)
sim_spp_scr._scrollbar.configure(height=0)
sim_spp_scr.grid(row=1, column=0, padx=PADX, pady=(PADY/4, PADY), sticky="ew", columnspan = 2)

# deploy button
sim_run_frm_1 = MyMainFrame(master=simple_main_frame)
sim_run_frm_1.grid(row=5, column=0, padx=PADX, pady=(0, PADY), sticky="nswe")
sim_run_img_widget = customtkinter.CTkLabel(sim_run_frm_1, text="", image = run_image, compound = 'left')
sim_run_img_widget.grid(row=3, column=0, padx=PADX, pady=PADY, sticky="nswe")
sim_run_frm = MySubFrame(master=sim_run_frm_1, width=1000)
sim_run_frm.grid(row=3, column=1, padx=(0, PADX), pady=PADY, sticky="nswe")
sim_run_btn_txt = ["Start processing", "Empezar a procesar", "Démarrer le traitement"]
sim_run_btn = customtkinter.CTkButton(sim_run_frm, text=sim_run_btn_txt[lang_idx], command=lambda: start_deploy(simple_mode = True))
sim_run_btn.grid(row=0, column=0, padx=PADX, pady=PADY, sticky="nswe", columnspan = 2)

# about
sim_abo_lbl = tk.Label(simple_main_frame, text=adv_abo_lbl_txt[lang_idx], font = Font(size = ADDAX_TXT_SIZE), fg="black", bg = yellow_primary)
sim_abo_lbl.grid(row=6, column=0, columnspan = 2, sticky="")
sim_abo_lbl_link = tk.Label(simple_main_frame, text="addaxdatascience.com", cursor="hand2", font = Font(size = ADDAX_TXT_SIZE, underline=1), fg=green_primary, bg =yellow_primary)
sim_abo_lbl_link.grid(row=7, column=0, columnspan = 2, sticky="", pady=(0, PADY))
sim_abo_lbl_link.bind("<Button-1>", lambda e: callback("http://addaxdatascience.com"))

# resize deploy tab to content
resize_canvas_to_content()

# main function
def main():

    # check if user calls this script from Timelapse
    parser = argparse.ArgumentParser(description="AddaxAI GUI")
    parser.add_argument('--timelapse-path', type=str, help="Path to the timelapse folder")
    args = parser.parse_args()
    global timelapse_mode
    global timelapse_path
    timelapse_mode = False
    timelapse_path = ""
    if args.timelapse_path:
        timelapse_mode = True
        timelapse_path = os.path.normpath(args.timelapse_path)
        var_choose_folder.set(timelapse_path)
        dsp_timelapse_path = shorten_path(timelapse_path, 25)
        sim_dir_pth.configure(text = dsp_timelapse_path, text_color = "black")
        var_choose_folder_short.set(dsp_timelapse_path)
        dsp_choose_folder.grid(column=0, row=row_choose_folder, sticky="w")

    # try to download the model info json to check if there are new models
    fetch_latest_model_info()

    # show donation popup if user has launched the app a certain number of times
    check_donation_window_popup()

    # initialise start screen
    enable_frame(fst_step)
    disable_frame(snd_step)
    disable_frame(trd_step)
    disable_frame(fth_step)
    set_lang_buttons(lang_idx)

    # super weird but apparently necessary, otherwise script halts at first root.update()
    switch_mode()
    switch_mode()

    # update frame states if we already have a timelapse path
    if timelapse_mode:
        update_frame_states()

    if scale_factor != 1.0:
        if not global_vars['var_scale_warning_shown']:
            mb.showwarning(
                [f"Scale set to {int(scale_factor * 100)}%", f"Escala fijada en {int(scale_factor * 100)}%", f"Échelle réglée à {int(scale_factor * 100)}%"][lang_idx],
                [f"The user interface of AddaxAI is designed for a scale setting of 100%. However, your screen settings are set to {int(scale_factor * 100)}%. We've worked to maintain a consistent look across different scale settings, but it may still affect the appearance of the application, causing some elements (like checkboxes or windows) to appear disproportionately large or small. Note that these visual differences won't impact the functionality of the application.\n\nThis warning will only appear once.",
                 f"La interfaz de usuario de AddaxAI está diseñada para un ajuste de escala del 100%. Sin embargo, su configuración de pantalla está establecida en {int(scale_factor * 100)}%. Hemos trabajado para mantener una apariencia consistente a través de diferentes configuraciones de escala, pero aún puede afectar la apariencia de la aplicación, causando que algunos elementos (como casillas de verificación o ventanas) aparezcan desproporcionadamente grandes o pequeñas. Tenga en cuenta que estas diferencias visuales no afectarán a la funcionalidad de la aplicación.\n\nEste aviso sólo aparecerá una vez.",
                 f"L'interface utilisateur d'AddaxAI est conçue pour une échelle de 100 %. Cependant, les paramètres de votre écran sont définis sur {int(scale_factor * 100)}%. Nous avons veillé à maintenir une apparence cohérente entre les différents paramètres d'échelle, mais cela peut néanmoins affecter l'apparence de l'application, entraînant une taille disproportionnée de certains éléments (comme les cases à cocher ou les fenêtres). Notez que ces différences visuelles n'affectent pas les fonctionnalités de l'application.\n\nCet avertissement n'apparaîtra qu'une seule fois."][lang_idx]
            )
        write_global_vars({"var_scale_warning_shown": True})

    # run
    root.mainloop()
    
# executable as script
if __name__ == "__main__":
    main()



