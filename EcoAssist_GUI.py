# Non-code GUI platform for training and deploying object detection models: https://github.com/PetervanLunteren/EcoAssist
# Written by Peter van Lunteren
# Latest edit by Peter van Lunteren on 8 Aug 2025

# import packages like a christmas tree
import os
import re
import sys
import cv2
import git
import json
import math
import time
import glob
import torch
import random
import signal
import shutil
import pickle
import platform
import tempfile
import datetime
import traceback
import subprocess
import webbrowser
import numpy as np
import PIL.ExifTags
import pandas as pd
import tkinter as tk
from tkinter import *
from pathlib import Path
from random import randint
from functools import partial
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import xml.etree.cElementTree as ET
from bounding_box import bounding_box as bb
from RangeSlider.RangeSlider import RangeSliderH
from tkinter import filedialog, ttk, messagebox as mb
from PIL import ImageTk, Image, ImageFilter, ImageFile
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# set global variables
version = "4.3"
EcoAssist_files = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
ImageFile.LOAD_TRUNCATED_IMAGES = True

# insert pythonpath
sys.path.insert(0, os.path.join(EcoAssist_files))
sys.path.insert(0, os.path.join(EcoAssist_files, "ai4eutils"))
sys.path.insert(0, os.path.join(EcoAssist_files, "yolov5"))
sys.path.insert(0, os.path.join(EcoAssist_files, "cameratraps"))

# log pythonpath
print(sys.path)

# language settings
lang = 0
step_txt = ['Step', 'Paso']
browse_txt = ['Browse', 'Examinar']
cancel_txt = ["Cancel", "Cancelar"]
change_folder_txt = ['Change folder', '¿Cambiar carpeta']
view_results_txt = ['View results', 'Ver resultados']
again_txt = ['Again?', '¿Otra vez?']
eg_txt = ['E.g.', 'Ejem.']
new_project_txt = ["<new project>", "<nuevo proyecto>"]
warning_txt = ["Warning", "Advertencia"]
error_txt = ["Error", "Error"]
invalid_value_txt = ["Invalid value", "Valor no válido"]
perc_done_txt = ["Percentage done", "Porcentaje hecho"]
processing_txt = ["Processing", "Procesando"]
elapsed_time_txt = ["Elapsed time", "Tiempo transcurrido"]
remaining_time_txt = ["Remaining time", "Tiempo restante"]
running_on_txt = ["Running on", "Funcionando en"]
none_txt = ["None", "Ninguno"]
of_txt = ["of", "de"]

##########################################
############# MAIN FUNCTIONS #############
##########################################

# post-process files
def postprocess(src_dir, dst_dir, thresh, sep, file_placement, sep_conf, vis, crp, csv, data_type):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # prepare data specific vars
    if data_type == "img":
        recognition_file = os.path.join(src_dir, "image_recognition_file.json")
        progress_postprocess_frame = img_progress_postprocess_frame
        progress_postprocess_progbar = img_progress_postprocess_progbar
        progress_postprocess_stats = img_progress_postprocess_stats
    else:
        recognition_file = os.path.join(src_dir, "video_recognition_file.json")
        progress_postprocess_frame = vid_progress_postprocess_frame
        progress_postprocess_progbar = vid_progress_postprocess_progbar
        progress_postprocess_stats = vid_progress_postprocess_stats

    # check if user is not in the middle of an annotation session
    if data_type == "img" and get_hitl_var_in_json(recognition_file) == "in-progress":
        if not mb.askyesno("Verification session in progress", f"Your verification session is not yet done. You can finish the session "
                                                               f"by clicking 'Continue' at '{lbl_hitl_main_txt[lang]}', or just continue to post-process "
                                                               "with the results as they are now.\n\nDo you want to continue to post-process?"):
            return

    # init vars
    global cancel_var
    start_time = time.time()
    nloop = 1

    # warn user
    if data_type == "vid":
        if vis or crp:
            check_json_presence_and_warn_user(["visualize or crop", "visualizar 0 recortar"][lang],
                                              ["visualizing or cropping", "visualizando o recortando"][lang],
                                              ["visualization and cropping", "visualización y recorte"][lang])
            vis, crp = [False] * 2

    # fetch label map
    label_map = fetch_label_map_from_json(recognition_file)
    inverted_label_map = {v: k for k, v in label_map.items()}

    # create list with colours for visualisation
    if vis:
        colors = ["fuchsia", "blue", "orange", "yellow", "green", "red", "aqua", "navy", "teal", "olive", "lime", "maroon", "purple"]
        colors = colors * 30 # colors for 390 classes
    
    # make sure json has relative paths
    json_paths_converted = False
    if check_json_paths(recognition_file) != "relative":
        make_json_relative(recognition_file)
        json_paths_converted = True
    
    # add cancel button
    cancel_var = False
    btn_cancel = Button(progress_postprocess_frame, text=cancel_txt[lang], command=cancel)
    btn_cancel.grid(row=9, column=0, columnspan=2)
    
    # open json file
    with open(recognition_file) as image_recognition_file_content:
        data = json.load(image_recognition_file_content)
    n_images = len(data['images'])

    # initialise the csv files
    if csv:
        # for files
        csv_for_files = os.path.join(dst_dir, "results_files.csv")
        if not os.path.isfile(csv_for_files):
            df = pd.DataFrame(list(), columns=["absolute_path", "relative_path", "data_type", "n_detections", "max_confidence", "human_verified",
                                               'datetime', 'datetime_original', 'datetime_digitized', 'make', 'shutter_speed_value',
                                               'aperture_value', 'exposure_bias_value', 'max_aperture_value', 'GPSInfo'])
            df.to_csv(csv_for_files, encoding='utf-8', index=False)
        
        # for detections
        csv_for_detections = os.path.join(dst_dir, "results_detections.csv")
        if not os.path.isfile(csv_for_detections):
            df = pd.DataFrame(list(), columns=["absolute_path", "relative_path", "data_type", "label", "confidence", "human_verified", "bbox_left",
                                               "bbox_top", "bbox_right", "bbox_bottom", "file_height", "file_width", 'datetime',
                                               'datetime_original', 'datetime_digitized', 'make', 'shutter_speed_value', 'aperture_value',
                                               'exposure_bias_value', 'max_aperture_value', 'GPSInfo'])
            df.to_csv(csv_for_detections, encoding='utf-8', index=False)

    # loop through images
    failure_warning_shown = False
    failure_warning_log = os.path.join(dst_dir, "failure_warning_log.txt")
    for image in data['images']:

        # cancel process if required
        if cancel_var:
            break
        
        # check for failure
        if "failure" in image:
            if not failure_warning_shown:
                mb.showwarning(warning_txt[lang], [f"One or more files failed to be analysed by the model (e.g., corrupt files) and will be skipped by "
                                                  f"post-processing features. See\n\n'{failure_warning_log}'\n\nfor more info.",
                                                  f"Uno o más archivos no han podido ser analizados por el modelo (por ejemplo, ficheros corruptos) y serán "
                                                  f"omitidos por las funciones de post-procesamiento. Para más información, véase\n\n'{failure_warning_log}'"][lang])
                failure_warning_shown = True
            
            # write warnings to log file
            with open(failure_warning_log, 'a+') as f:
                f.write(f"File '{image['file']}' was skipped by post processing features because '{image['failure']}'\n")
            f.close()

            # skip this iteration
            continue
        
        # get image info
        file = image['file']
        detections_list = image['detections']
        n_detections = len(detections_list)
        progress_postprocess_progbar['value'] += 100 / n_images

        # check if it has been manually verified
        manually_checked = False
        if 'manually_checked' in image:
            if image['manually_checked']:
                manually_checked = True

        # init vars
        max_detection_conf = 0.0
        unique_labels = []
        bbox_info = []
        csv_detectons = []
        csv_files = []

        # open files
        if vis or crp or csv:
            if data_type == "img":
                im_to_vis = cv2.imread(os.path.join(src_dir, file))
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

            # try to read exif data
            if csv:
                try:
                    img_for_exif = PIL.Image.open(os.path.join(src_dir, file))
                    exif_data = {
                        PIL.ExifTags.TAGS[k]: v
                        for k, v in img_for_exif._getexif().items()
                        if k in PIL.ExifTags.TAGS
                    }
                    img_for_exif.close()
                except:
                    exif_data = None

                # check if datetime values can be found
                exif_params = []
                for param in ['DateTime', 'DateTimeOriginal', 'DateTimeDigitized', 'Make', 'ShutterSpeedValue', 'ApertureValue', 'ExposureBiasValue', 'MaxApertureValue', 'GPSInfo']:
                    try:
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
                    if vis or crp or csv:
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

        # separate files
        if sep:
            if n_detections == 0:
                file = move_files(file, "empty", file_placement, max_detection_conf, sep_conf, dst_dir, src_dir, manually_checked)
            else:
                if len(unique_labels) > 1:
                    labels_str = "_".join(unique_labels)
                    file = move_files(file, labels_str, file_placement, max_detection_conf, sep_conf, dst_dir, src_dir, manually_checked)
                elif len(unique_labels) == 0:
                    file = move_files(file, "empty", file_placement, max_detection_conf, sep_conf, dst_dir, src_dir, manually_checked)
                else:
                    file = move_files(file, label, file_placement, max_detection_conf, sep_conf, dst_dir, src_dir, manually_checked)
        
        # collect info to append to csv files
        if csv:
            # file info
            row = pd.DataFrame([[src_dir, file, data_type, len(bbox_info), max_detection_conf, manually_checked, *exif_params]])
            row.to_csv(csv_for_files, encoding='utf-8', mode='a', index=False, header=False)

            # detections info
            rows = []
            for bbox in bbox_info:
                row = [src_dir, file, data_type, *bbox[:9], *exif_params]
                rows.append(row)
            rows = pd.DataFrame(rows)
            rows.to_csv(csv_for_detections, encoding='utf-8', mode='a', index=False, header=False)
    
        # visualize images
        if vis and len(bbox_info) > 0:
            for bbox in bbox_info:
                if manually_checked:
                    vis_label = f"{bbox[0]} (verified)"
                else:
                    vis_label = f"{bbox[0]} {bbox[1]}"
                color = colors[int(inverted_label_map[bbox[0]])]
                bb.add(im_to_vis, *bbox[3:7], vis_label, color)
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
        progress_postprocess_stats['text'] = create_postprocess_lbl(elapsed_time_sep, time_left_sep, command="running")
        nloop += 1
        root.update()

    # create summary csv
    if csv:
        csv_for_summary = os.path.join(dst_dir, "results_summary.csv")
        if os.path.exists(csv_for_summary):
            os.remove(csv_for_summary)
        det_info = pd.DataFrame(pd.read_csv(csv_for_detections))
        summary = pd.DataFrame(det_info.groupby(['label', 'data_type']).size().sort_values(ascending=False).reset_index(name='n_detections'))
        summary.to_csv(csv_for_summary, encoding='utf-8', mode='w', index=False, header=True)

    # remove cancel button
    btn_cancel.grid_remove()
    
    # change json paths back, if converted earlier
    if json_paths_converted:
        make_json_absolute(recognition_file)
    
    # let the user know it's done
    progress_postprocess_stats['text'] = create_postprocess_lbl(elapsed_time_sep, time_left_sep, command="done")
    root.update()

# open progress window and initiate the post-process
def start_postprocess():
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
        
    # set global variables
    global img_progress_postprocess_frame
    global vid_progress_postprocess_frame
    
    # fix user input
    src_dir = var_choose_folder.get()
    dst_dir = var_output_dir.get()
    thresh = var_thresh.get()
    sep = var_separate_files.get()
    file_placement = var_file_placement.get()
    sep_conf = var_sep_conf.get()
    vis = var_vis_files.get()
    crp = var_crp_files.get()
    csv = var_csv.get()

    # check which json files are present
    img_json = False
    if os.path.isfile(os.path.join(src_dir, "image_recognition_file.json")):
        img_json = True
    vid_json = False
    if os.path.isfile(os.path.join(src_dir, "video_recognition_file.json")):
        vid_json = True
    if not img_json and not vid_json:
        mb.showerror(error_txt[lang], ["No model output file present. Make sure you run step 2 before post-processing the files.",
                                       "No hay archivo de salida del modelo. Asegúrese de ejecutar el paso 2 antes de postprocesar"
                                       " los archivos."][lang])
        return
    
    # check if destination dir is valid and set to input dir if not
    if dst_dir in ["", "/", "\\", ".", "~", ":"] or not os.path.isdir(dst_dir):
        mb.showerror(["Destination folder not set", "Carpeta de destino no establecida."][lang],
                        ["Destination folder not set.\n\n You have not specified where the post-processing results should be placed or the set "
                        "folder does not exist. This is required.",
                        "Carpeta de destino no establecida. No ha especificado dónde deben colocarse los resultados del postprocesamiento o la "
                        "carpeta establecida no existe. Esto opción es obligatoria."][lang])
        return

    # warn user if the original files will be overwritten with visualized files
    if os.path.normpath(dst_dir) == os.path.normpath(src_dir) and vis and not sep:
        if not mb.askyesno(["Original images will be overwritten", "Las imágenes originales se sobrescribirán."][lang], 
                      [f"WARNING! The visualized images will be placed in the folder with the original data: '{src_dir}'. By doing this, you will overwrite the original images"
                      " with the visualized ones. Visualizing is permanent and cannot be undone. Are you sure you want to continue?",
                      f"ATENCIÓN. Las imágenes visualizadas se colocarán en la carpeta con los datos originales: '{src_dir}'. Al hacer esto, se sobrescribirán las imágenes "
                      "originales con las visualizadas. La visualización es permanente y no se puede deshacer. ¿Está seguro de que desea continuar?"][lang]):
            return
    
    # warn user if images will be moved and visualized
    if sep and file_placement == 1 and vis:
        if not mb.askyesno(["Original images will be overwritten", "Las imágenes originales se sobrescribirán."][lang], 
                      [f"WARNING! You specified to visualize the original images. Visualizing is permanent and cannot be undone. If you don't want to visualize the original "
                      f"images, please select 'Copy' as '{lbl_file_placement_txt}'. Are you sure you want to continue with the current settings?",
                      "ATENCIÓN. Ha especificado visualizar las imágenes originales. La visualización es permanente y no puede deshacerse. Si no desea visualizar las "
                      f"imágenes originales, seleccione 'Copiar' como '{lbl_file_placement_txt}'. ¿Está seguro de que desea continuar con la configuración actual?"][lang]):
            return

    # open new window with progress bar and stats
    pp_process_window = Toplevel(root)
    pp_process_window.title("Post-process progress")
    pp_process_window.geometry()

    # logo
    logo = tk.Label(pp_process_window, image=grey_bg_logo)
    logo.grid(column=0, row=0, columnspan=2, sticky='ew', pady=(5, 0))

    # add image progress
    if img_json:
        img_progress_postprocess_frame = LabelFrame(pp_process_window, text=[" Postprocessing images ", " Postprocesamiento de imágenes "][lang], pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue')
        img_progress_postprocess_frame.configure(font=(text_font, 15, "bold"))
        img_progress_postprocess_frame.grid(column=0, row=1, columnspan=2, sticky='ew')
        img_progress_postprocess_frame.columnconfigure(0, weight=3, minsize=115)
        img_progress_postprocess_frame.columnconfigure(1, weight=1, minsize=115)
        global img_progress_postprocess_progbar
        img_progress_postprocess_progbar = ttk.Progressbar(master=img_progress_postprocess_frame, orient='horizontal', mode='determinate', length=280)
        img_progress_postprocess_progbar.grid(column=0, row=0, columnspan=2, padx=5, pady=(3,0))
        global img_progress_postprocess_stats
        img_progress_postprocess_stats = ttk.Label(master=img_progress_postprocess_frame, text=create_postprocess_lbl())
        img_progress_postprocess_stats.grid(column=0, row=1, padx=5, pady=(0,3), columnspan=2)

    # add video progress
    if vid_json:
        vid_progress_postprocess_frame = LabelFrame(pp_process_window, text=[" Postprocessing videos ", " Postprocesamiento de vídeos "][lang], pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue')
        vid_progress_postprocess_frame.configure(font=(text_font, 15, "bold"))
        vid_progress_postprocess_frame.grid(column=0, row=2, columnspan=2, sticky='ew')
        vid_progress_postprocess_frame.columnconfigure(0, weight=3, minsize=115)
        vid_progress_postprocess_frame.columnconfigure(1, weight=1, minsize=115)
        global vid_progress_postprocess_progbar
        vid_progress_postprocess_progbar = ttk.Progressbar(master=vid_progress_postprocess_frame, orient='horizontal', mode='determinate', length=280)
        vid_progress_postprocess_progbar.grid(column=0, row=0, columnspan=2, padx=5, pady=(3,0))
        global vid_progress_postprocess_stats
        vid_progress_postprocess_stats = ttk.Label(master=vid_progress_postprocess_frame, text=create_postprocess_lbl())
        vid_progress_postprocess_stats.grid(column=0, row=1, padx=5, pady=(0,3), columnspan=2)
    
    try:
        # postprocess images
        if img_json:
            postprocess(src_dir, dst_dir, thresh, sep, file_placement, sep_conf, vis, crp, csv, data_type = "img")

        # postprocess videos
        if vid_json:
            postprocess(src_dir, dst_dir, thresh, sep, file_placement, sep_conf, vis, crp, csv, data_type = "vid")
        
        # complete
        complete_frame(fth_step)

        # close progress window
        pp_process_window.destroy()
    
    except Exception as error:
        # log error
        print("ERROR:\n" + str(error) + "\n\nDETAILS:\n" + str(traceback.format_exc()) + "\n\n")
        
        # show error
        mb.showerror(title=error_txt[lang],
                     message=["An error has occurred", "Ha ocurrido un error"][lang] + " (EcoAssist v" + version + "): '" + str(error) + "'.",
                     detail=traceback.format_exc())
        
        # close window
        pp_process_window.destroy()

# This function preserves the hierarchy of parent and child folders while sorting all file paths. It ensures that:
# Files and subfolders within the same parent folder are sorted together.
# Files and folders from different parent folders are kept separate and sorted relative to their full path.
def sort_paths_per_dir_layer(file_paths):
    # Split each path into components
    split_paths = [os.path.normpath(path).split(os.sep) for path in file_paths]
    
    # Sort paths alphabetically by each directory layer
    sorted_paths = sorted(split_paths, key=lambda x: tuple(x))
    
    # Reconstruct the file paths
    sorted_file_paths = []
    for path in sorted_paths:
        drive, tail = os.path.splitdrive(os.path.join(*path))
        if drive:
            sorted_file_paths.append(drive + os.sep + tail)
        else:
            sorted_file_paths.append(tail)
            
    # return
    return sorted_file_paths

from PIL.ExifTags import TAGS
from datetime import datetime

def extract_timestamp_from_exif(filepath):
    """
    Extract timestamp from image EXIF DateTimeOriginal metadata.
    """
    try:
        with Image.open(filepath) as img:
            exif_data = img._getexif()
            if exif_data is not None:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "DateTimeOriginal":
                        # DateTimeOriginal format: "YYYY:MM:DD HH:MM:SS"
                        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                        
                # Fallback to DateTime if DateTimeOriginal not found
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == "DateTime":
                        return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
        
        return None
    except Exception as e:
        print(f"Error reading EXIF from {filepath}: {e}")
        return None

def group_files_into_sequences(files, data_folder, max_gap_seconds=60):
    """
    Group files into sequences based on EXIF timestamp proximity.
    Files taken within max_gap_seconds of each other belong to the same sequence.
    """
    # Extract timestamps and sort files by timestamp
    files_with_timestamps = []
    no_timestamp_files = []
    
    for filename in files:
        filepath = os.path.join(data_folder, filename)
        timestamp = extract_timestamp_from_exif(filepath)
        if timestamp:
            files_with_timestamps.append((filename, timestamp))
        else:
            print(f"Warning: Could not extract timestamp from {filename}")
            no_timestamp_files.append(filename)
    
    # Sort by timestamp
    files_with_timestamps.sort(key=lambda x: x[1])
    
    if not files_with_timestamps:
        # If no timestamps found, treat each file as its own sequence
        return [[f] for f in files]
    
    sequences = []
    current_sequence = [files_with_timestamps[0][0]]
    last_timestamp = files_with_timestamps[0][1]
    
    for filename, timestamp in files_with_timestamps[1:]:
        time_diff = (timestamp - last_timestamp).total_seconds()
        
        if time_diff <= max_gap_seconds:
            # Same sequence
            current_sequence.append(filename)
        else:
            # New sequence
            sequences.append(current_sequence)
            current_sequence = [filename]
        
        last_timestamp = timestamp
    
    # Don't forget the last sequence
    sequences.append(current_sequence)
    
    # Add files without timestamps as individual sequences at the end
    for filename in no_timestamp_files:
        sequences.append([filename])
    
    return sequences

# check data and prepare for training
def prepare_data_for_training(data_folder, prop_to_test, prop_to_val):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
    # convert pascal voc to yolo
    pascal_voc_to_yolo(data_folder)
    # get list of all images in dir
    data_folder = os.path.normpath(data_folder)
    files = [f for f in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder, f)) and not f.endswith(".DS_Store") and f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    # calculate amounts
    total_n = len(files)
    n_test = int(total_n * prop_to_test)
    n_val = int(total_n * prop_to_val)
    
    # Group files into sequences based on EXIF timestamps
    sequences = group_files_into_sequences(files, data_folder, max_gap_seconds=60)
    print(f"Found {len(sequences)} sequences from {total_n} files")
    
    # Show sequence size distribution
    seq_sizes = [len(seq) for seq in sequences]
    print(f"Sequence sizes: min={min(seq_sizes)}, max={max(seq_sizes)}, avg={sum(seq_sizes)/len(seq_sizes):.1f}")
    
    # Randomly shuffle the sequences
    random.shuffle(sequences)
    
    # Flatten sequences back into a single list (maintaining sequence order within each sequence)
    files = []
    for sequence in sequences:
        files.extend(sequence)
    
    # Normal split (at most one sequence will be split)
    test_files = files[:n_test]
    val_files = files[n_test:n_test+n_val]
    train_files = files[n_test+n_val:]
    
    print(f"Split: {len(test_files)} test, {len(val_files)} val, {len(train_files)} train")

    # remove files for previous training
    old_files = ["dataset.yaml", "train_selection.txt", "train_selection.cache", "train_selection.cache.npy", "val_selection.txt", "val_selection.cache",
                 "val_selection.cache.npy", "test_selection.txt", "test_selection.cache", "test_selection.cache.npy"]
    for filename in old_files:
        old_file = os.path.join(data_folder, filename)
        if os.path.isfile(old_file):
            os.remove(old_file)

    # write text files with images
    for elem in [[train_files, "train"], [val_files, "val"], [test_files, "test"]]:
        counter = 0
        with open(os.path.join(data_folder, elem[1] + "_selection.txt"), 'w') as f:
            for file in elem[0]:
                f.write("./" + file + "\n")
                counter += 1
        send_to_output_window(f"\nWill use {counter} images as {elem[1]}");root.update()

    # read class names
    with open(os.path.join(data_folder, "classes.txt")) as f:
        lines = f.readlines()
        names = [line.rstrip('\n') for line in lines]
    nc = len(names)

    # create dataset.yaml
    if prop_to_test == 0:
        yaml_content = f"# set paths\npath: '{data_folder}'\ntrain: ./train_selection.txt\nval: ./val_selection.txt\n\n# n classes\nnc: {nc}\n\n# class names\nnames: {names}\n"
    else:
        yaml_content = f"# set paths\npath: '{data_folder}'\ntrain: ./train_selection.txt\nval: ./val_selection.txt\ntest: ./test_selection.txt\n\n# n classes\nnc: {nc}\n\n# class names\nnames: {names}\n"
    yaml_file = os.path.join(data_folder, "dataset.yaml")
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
        send_to_output_window(f"\nWritten {yaml_file} with content:\n\n{yaml_content}\n");root.update()

def is_on_c_drive(path):
    drive, _ = os.path.splitdrive(os.path.abspath(path))
    return drive.upper() == 'C:'

# check input and execute train command
def start_training():
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # set button states
    cancel_training_bool.set(False)
    set_buttons_to_training()

    # checkout yolov5 repo to new models
    switch_yolov5_git_to("new models")

    # build command
    send_to_output_window("Building train command...");root.update()
    command_args = []
    command_args.append(sys.executable)
    command_args.append(os.path.join(EcoAssist_files, "yolov5", "train.py"))

    # resume existing training
    if var_train_type.get() == dpd_train_type_options[lang][1]: 
        # resume from file
        checkpoint_file = var_resume_checkpoint_path.get()
        if not os.path.isfile(checkpoint_file):
            mb.showerror(error_txt[lang], message=["Please specify the last checkpoint file to resume from.",
                                                   "Por favor, especifique el último archivo de punto de control desde el que reanudar."][lang])
            set_buttons_to_idle()
            return
        command_args.append(f"--resume={checkpoint_file}")
    
        # extract output folder from checkpoint file
        results_dir =  os.path.dirname(
                            os.path.dirname(
                                os.path.dirname(
                                    os.path.dirname(
                                        os.path.normpath(
                                            checkpoint_file)))))
        
    # start new training
    elif var_train_type.get() == dpd_train_type_options[lang][0]: 
        # set retrain from
        command_args.append(f"--weights={var_learning_model_path.get()}")

        # prepare data for training
        send_to_output_window("Preparing data training set...");root.update()
        data_dir = var_annotated_data.get()
        if data_dir in ["", "/", "\\", ".", "~", ":"] or not os.path.isdir(data_dir):
            mb.showerror(error_txt[lang], message=["Please specify a directory with annotated data to train on.",
                                                   "Por favor, especifique un directorio con datos anotados para entrenar."][lang])
            set_buttons_to_idle()
            return
        prepare_data_for_training(data_dir, var_test_prop.get(), var_val_prop.get())

        # add data argument
        command_args.append(f"--data={os.path.normpath(os.path.join(data_dir, 'dataset.yaml'))}")
        
        # select gpu if available
        if var_train_gpu.get():
            send_to_output_window("Searching for GPU...");root.update()

            # for windows machines
            if torch.cuda.is_available():
                send_to_output_window("\tCUDA is available.");root.update()
                command_args.append(f"--device=0")
            else:
                send_to_output_window("\tCUDA is not available.");root.update()
                
            # for apple silicon machines
            try:
                if torch.backends.mps.is_built() and torch.backends.mps.is_available():
                    send_to_output_window("\tGPU (MPS) is available.");root.update()
                    command_args.append(f"--device=mps")
                else:
                    send_to_output_window("\tGPU (MPS) is not available.");root.update()
            except AttributeError:
                pass
        
        # number of epochs
        if not var_n_epochs.get().isdecimal() or var_n_epochs.get() == "":
            invalid_value_warning("number of epochs")
            set_buttons_to_idle()
            return
        command_args.append(f"--epochs={var_n_epochs.get()}")

        # batch size
        if no_user_input(var_batch_size):
            command_args.append("--batch-size=-1")
        elif not var_batch_size.get().isdecimal():
            invalid_value_warning("batch size")
            set_buttons_to_idle()
            return
        else:
            command_args.append(f"--batch-size={var_batch_size.get()}")

        # number of dataloader workers
        if no_user_input(var_n_workers):
            command_args.append("--workers=4")
        elif not var_n_workers.get().isdecimal():
            invalid_value_warning("number of workers")
            set_buttons_to_idle()
            return
        else:
            command_args.append(f"--workers={var_n_workers.get()}")

        # image size
        if no_user_input(var_image_size_for_training) == False:
            if not var_image_size_for_training.get().isdecimal():
                invalid_value_warning("image size")
                set_buttons_to_idle()
                return
            else:
                command_args.append(f"--img={var_image_size_for_training.get()}")
        elif var_learning_model.get() == dpd_learning_model_options[lang][0] or var_learning_model.get() == dpd_learning_model_options[lang][1]:
            # megadetector models
            command_args.append(f"--img=1280")
        
        # frozen layers
        if var_learning_model.get() == dpd_learning_model_options[lang][0] or var_learning_model.get() == dpd_learning_model_options[lang][1]:
            # megadetector models
            command_args.append(f"--freeze=12")
        elif var_learning_model.get() == dpd_learning_model_options[lang][7]:
            # custom model
            if no_user_input(var_n_freeze_layers) == False:
                if not var_n_freeze_layers.get().isdecimal():
                    invalid_value_warning("number of frozen layers")
                    set_buttons_to_idle()
                    return
                else:
                    command_args.append(f"--freeze={var_n_freeze_layers.get()}")
        
        # model architecture
        if var_learning_model.get() == dpd_learning_model_options[lang][8]:
            # from scratch
            if not var_model_architecture.get() == dpd_model_architecture_options[lang][6]:
                # not "none" selected
                command_args.append(f"--cfg={var_model_architecture_path.get()}")

        # check user input for destination folder
        results_dir = var_results_dir.get()
        if results_dir in ["", "/", "\\", ".", "~", ":"] or not os.path.isdir(results_dir):
            mb.showerror(error_txt[lang], message=["Please specify the destination directory.",
                                                   "Por favor, especifique una carpeta de destino."][lang])
            set_buttons_to_idle()
            return

        # project name
        if no_user_input(var_project_name):
            invalid_value_warning("project name", numeric = False)
            set_buttons_to_idle()
            return
        project_name = var_project_name.get()
        command_args.append(f"--project={project_name}")
            
        # name of the run
        if no_user_input(var_run_name) == False:
            command_args.append(f"--name={var_run_name.get()}")

        # hyperparameter file
        if var_hyper_file_path.get() != "":
            command_args.append(f"--hyp={var_hyper_file_path.get()}")

        # cache images
        if var_cache_imgs.get():
            command_args.append(f"--cache")

        # evolve
        if var_evolve.get():
            # get n generations
            if no_user_input(var_n_generations) == False:
                if not var_n_generations.get().isdecimal():
                    invalid_value_warning("number of generations")
                    set_buttons_to_idle()
                    return
                command_args.append(f"--evolve={var_n_generations.get()}")
            else:
                command_args.append(f"--evolve=300")
    
    # change directory to the destination folder, if on C drive, else error
    if is_on_c_drive(results_dir):
        os.chdir(results_dir)
    else:
        send_to_output_window(f"\nDestination folder '{results_dir} is on {os.path.splitdrive(results_dir)[0]} drive. You need to choose a folder on your C: drive!")
        clean_training_dir(data_dir)
        cancel_training_bool.set(False)
        set_buttons_to_idle()
        return
    
    # adjust command for unix OS
    if os.name != 'nt':
        command_args = "'" + "' '".join(command_args) + "'"

    # log command
    send_to_output_window(f"\ncommand_args : {command_args}\n");root.update()
    send_to_output_window("\nStarting training process...\n");root.update()

    # run command
    p = Popen(command_args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                shell=True,
                universal_newlines=True)
    
    # read the output
    skip = False
    for line in p.stdout:

        # send to console but only update the window for the lines that matter
        send_to_output_window(line)

        # skip the model summary 
        if line.split() == ["from", "n", "params", "module", "arguments"]:
            skip = True
        if line.startswith("Model summary: "):
            skip = False

        # skip the freezing process
        if not line.startswith("freezing ") and not skip:

            # pause process for unix OS
            if os.name != 'nt':
                p.send_signal(signal.SIGSTOP)
            
            # check if user cancelled the training
            if cancel_training_bool.get():
                send_to_output_window("TRAINING CANCELLED BY USER...")
                set_buttons_to_idle()
                return
            
            # update root so that console output will be updated for the user
            root.update()

            # continue process for unix OS
            if os.name != 'nt':
                p.send_signal(signal.SIGCONT)

    # remove temporary files
    clean_training_dir(data_dir)
    print("\nTraining has finished.")

    # set button states
    cancel_training_bool.set(False)
    set_buttons_to_idle()

# open human-in-the-loop verification windows
def open_annotation_windows(recognition_file, class_list_txt, file_list_txt, label_map):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # check if file list exists
    if not os.path.isfile(file_list_txt):
        mb.showerror(["No images to verify", "No hay imágenes para verificar"][lang],
                     ["There are no images to verify with the selected criteria. Use the 'Update counts' button to see how many "
                     "images you need to verify with the selected criteria.", "No hay imágenes para verificar con los criterios "
                     "seleccionados. Utilice el botón 'Actualizar recuentos' para ver cuántas imágenes necesita verificar con "
                     "los criterios seleccionados."][lang])
        return

    # check number of images to verify
    total_n_files = 0
    with open(file_list_txt) as f:
        for line in f:
            total_n_files += 1
    if total_n_files == 0:
        mb.showerror(["No images to verify", "No hay imágenes para verificar"][lang],
                     ["There are no images to verify with the selected criteria. Use the 'Update counts' button to see how many "
                     "images you need to verify with the selected criteria.", "No hay imágenes para verificar con los criterios "
                     "seleccionados. Utilice el botón 'Actualizar recuentos' para ver cuántas imágenes necesita verificar con "
                     "los criterios seleccionados."][lang])
        return

    # read label map from json
    label_map = fetch_label_map_from_json(recognition_file)
    inverted_label_map = {v: k for k, v in label_map.items()}

    # count n verified files and locate images that need converting
    n_verified_files = 0
    if get_hitl_var_in_json(recognition_file) != "never-started":
        init_dialog = PatienceDialog(total = total_n_files, text = ["Initializing...", "Inicializando..."][lang])
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
    hitl_progress_window = Toplevel(root)
    hitl_progress_window.title(["Manual check overview", "Verificación manual"][lang])
    hitl_progress_window.geometry()

    # logo
    logo = tk.Label(hitl_progress_window, image=grey_bg_logo)
    logo.grid(column=0, row=0, columnspan=2, sticky='ew', pady=(5, 0))

    # explenation frame
    hitl_explenation_frame = LabelFrame(hitl_progress_window, text=[" Explanation ", " Explicación "][lang],
                                            pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue')
    hitl_explenation_frame.configure(font=(text_font, 15, "bold"))
    hitl_explenation_frame.grid(column=0, row=1, columnspan=2, sticky='ew')
    hitl_explenation_frame.columnconfigure(0, weight=3, minsize=115)
    hitl_explenation_frame.columnconfigure(1, weight=1, minsize=115)

    # explenation text
    text_hitl_explenation_frame = Text(master=hitl_explenation_frame, wrap=WORD, width=1, height=15 * explanation_text_box_height_factor) 
    text_hitl_explenation_frame.grid(column=0, row=0, columnspan=5, padx=5, pady=5, sticky='ew')
    text_hitl_explenation_frame.tag_config('explanation', font=f'{text_font} {int(13 * text_size_adjustment_factor)} normal', lmargin1=10, lmargin2=10)
    text_hitl_explenation_frame.insert(END, ["This is where you do the actual verification. You'll have to make sure that all objects in all images are correctly "
                                            "labeled. That also includes classes that you did not select but are on the image by chance. If an image is verified, "
                                            "you'll have to let EcoAssist know by pressing the space bar. If all images are verified and up-to-date, you can close "
                                            "the window. EcoAssist will prompt you for the final step. You can also close the window and continue at a later moment.", 
                                            "Deberá asegurarse de que todos los objetos en todas las imágenes estén "
                                            "etiquetados correctamente. Eso también incluye clases que no seleccionaste pero que están en la imagen por casualidad. "
                                            "Si se verifica una imagen, deberá informar a EcoAssist presionando la barra espaciadora. Si todas las imágenes están "
                                            "verificadas y actualizadas, puede cerrar la ventana. EcoAssist le indicará el paso final. También puedes cerrar la "
                                            "ventana y continuar en otro momento."][lang])
    text_hitl_explenation_frame.tag_add('explanation', '1.0', '1.end')

    # shortcuts frame
    hitl_shortcuts_frame = LabelFrame(hitl_progress_window, text=[" Shortcuts ", " Atajos "][lang],
                                        pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue')
    hitl_shortcuts_frame.configure(font=(text_font, 15, "bold"))
    hitl_shortcuts_frame.grid(column=0, row=2, columnspan=2, sticky='ew')
    hitl_shortcuts_frame.columnconfigure(0, weight=3, minsize=115)
    hitl_shortcuts_frame.columnconfigure(1, weight=1, minsize=115)

    # shortcuts label
    shortcut_labels = [["Next image:", "Previous image:", "Create box:", "Edit box:", "Delete box:", "Verify, save, and next image:"],
                       ["Imagen siguiente:", "Imagen anterior:", "Crear cuadro:", "Editar cuadro:", "Eliminar cuadro:", "Verificar, guardar, y siguiente imagen:"]][lang]
    shortcut_values = ["d", "a", "w", "s", "del", ["space", "espacio"][lang]]
    for i in range(len(shortcut_labels)):
        ttk.Label(master=hitl_shortcuts_frame, text=shortcut_labels[i]).grid(column=0, row=i, columnspan=1, sticky='w')
        ttk.Label(master=hitl_shortcuts_frame, text=shortcut_values[i]).grid(column=1, row=i, columnspan=1, sticky='e')

    # numbers frame
    hitl_stats_frame = LabelFrame(hitl_progress_window, text=[" Progress ", " Progreso "][lang],
                                    pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue')
    hitl_stats_frame.configure(font=(text_font, 15, "bold"))
    hitl_stats_frame.grid(column=0, row=3, columnspan=2, sticky='ew')
    hitl_stats_frame.columnconfigure(0, weight=3, minsize=115)
    hitl_stats_frame.columnconfigure(1, weight=1, minsize=115)

    # progress bar 
    hitl_progbar = ttk.Progressbar(master=hitl_stats_frame, orient='horizontal', mode='determinate', length=280)
    hitl_progbar.grid(column=0, row=0, columnspan=2, padx=5, pady=(3,0))

    # percentage done
    lbl_hitl_stats_percentage = ttk.Label(master=hitl_stats_frame, text=["Percentage done:", "Porcentaje realizado:"][lang])
    lbl_hitl_stats_percentage.grid(column=0, row=1, columnspan=1, sticky='w')
    value_hitl_stats_percentage = ttk.Label(master=hitl_stats_frame, text="")
    value_hitl_stats_percentage.grid(column=1, row=1, columnspan=1, sticky='e')

    # total n images to verify
    lbl_hitl_stats_verified = ttk.Label(master=hitl_stats_frame, text=["Files verified:", "Archivos verificados:"][lang])
    lbl_hitl_stats_verified.grid(column=0, row=2, columnspan=1, sticky='w')
    value_hitl_stats_verified = ttk.Label(master=hitl_stats_frame, text="")
    value_hitl_stats_verified.grid(column=1, row=2, columnspan=1, sticky='e')

    # show window
    percentage = round((n_verified_files/total_n_files)*100)
    hitl_progbar['value'] = percentage
    value_hitl_stats_percentage.config(text = f"{percentage}%")
    value_hitl_stats_verified.config(text = f"{n_verified_files}/{total_n_files}")
    hitl_progress_window.update_idletasks()
    
    # locate open script
    if os.name == 'nt':
        labelImg_script = os.path.join(EcoAssist_files, "EcoAssist", "label.bat")
    else:
        labelImg_script = os.path.join(EcoAssist_files, "EcoAssist", "label.command")

    # create command
    command_args = []
    command_args.append(labelImg_script)
    command_args.append(class_list_txt)
    command_args.append(file_list_txt)

    # adjust command for unix OS
    if os.name != 'nt':
        command_args = "'" + "' '".join(command_args) + "'"

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
            value_hitl_stats_percentage.config(text = f"{percentage}%")
            value_hitl_stats_verified.config(text = f"{n_verified_files}/{total_n_files}")

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
                                                text = ["Running verification...", "Verificación de funcionamiento..."][lang])
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
    patience_dialog = PatienceDialog(total = len(imgs_needing_converting) + n_img_in_json, text = ["Checking results...", "Comprobando resultados"][lang])
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
        if mb.askyesno(title=["Are you done?", "¿Ya terminaste?"][lang],
                       message=["All images are verified and the 'image_recognition_file.json' is up-to-date.\n\nDo you want to close this "
                                "verification session and proceed to the final step?", "Todas las imágenes están verificadas y "
                                "'image_recognition_file.json' está actualizado.\n\n¿Quieres cerrar esta sesión de verificación"
                                " y continuar con el paso final?"][lang]):
            # close window
            hitl_progress_window.destroy()

            # get plot from xml files
            fig = produce_graph(file_list_txt = file_list_txt)

            # init window
            hitl_final_window = Toplevel(root)
            hitl_final_window.title("Overview")
            hitl_final_window.geometry()

            # add plot
            chart_type = FigureCanvasTkAgg(fig, hitl_final_window)
            chart_type.get_tk_widget().grid(row = 0, column = 0)

            # button frame
            hitl_final_actions_frame = LabelFrame(hitl_final_window, text=[" Do you want to export these verified images as training data? ",
                                                                           " ¿Quieres exportar estas imágenes verificadas como datos de entrenamiento? "][lang],
                                                                           pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue', labelanchor = 'n')
            hitl_final_actions_frame.configure(font=(text_font, 15, "bold"))
            hitl_final_actions_frame.grid(column=0, row=3, columnspan=2, sticky='ew')
            hitl_final_actions_frame.columnconfigure(0, weight=1, minsize=115)
            hitl_final_actions_frame.columnconfigure(1, weight=1, minsize=115)

            # buttons
            btn_hitl_final_export_y = Button(master=hitl_final_actions_frame, text=["Yes - choose folder and create training data",
                                                                                    "Sí - elija la carpeta y crear datos de entrenamiento"][lang], 
                                    width=1, command = lambda: [uniquify_and_move_img_and_xml_from_filelist(file_list_txt = file_list_txt, recognition_file = recognition_file, hitl_final_window = hitl_final_window),
                                                                update_frame_states()])
            btn_hitl_final_export_y.grid(row=0, column=0, rowspan=1, sticky='nesw', padx=5)

            btn_hitl_final_export_n = Button(master=hitl_final_actions_frame, text=["No - go back to the main EcoAssist window",
                                                                                    "No - regrese a la ventana principal de EcoAssist"][lang], 
                                    width=1, command = lambda: [delete_temp_folder(file_list_txt),
                                                                hitl_final_window.destroy(),
                                                                change_hitl_var_in_json(recognition_file, "done"),
                                                                update_frame_states()])
            btn_hitl_final_export_n.grid(row=0, column=1, rowspan=1, sticky='nesw', padx=5)

# get the images and xmls from annotation session and store them with unique filename
def uniquify_and_move_img_and_xml_from_filelist(file_list_txt, recognition_file, hitl_final_window):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # choose destination
    dst_dir = filedialog.askdirectory()

    # ask to move or copy
    window = TextButtonWindow(["Method of file placement", "Método de colocación de archivos"][lang],
                              [f"Do you want to copy or move the images to\n'{dst_dir}'?",
                              f"¿Quieres copiar o mover las imágenes a\n'{dst_dir}'?"][lang],
                              [["Move", "Mover"][lang], ["Copy", "Copiar"][lang], ["Cancel", "Cancelar"][lang]])
    user_input = window.run()
    if user_input == "Cancel" or user_input == "Cancelar":
        return
    else:
        if user_input == "Move" or user_input == "Mover":
            copy_or_move = "Move"
        if user_input == "Copy" or user_input == "Copiar":
            copy_or_move = "Copy"

    # init vars
    timestamp = str(datetime.date.today()) + str(datetime.datetime.now().strftime("%H%M%S"))
    timestamp = timestamp.replace('-', '')
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
        patience_dialog = PatienceDialog(total = n_imgs, text = ["Writing files...", "Escribir archivos..."][lang])
        patience_dialog.open()
        current = 1

        # loop
        for img in f:

            # get relative path
            img = os.path.relpath(img.rstrip(), src_dir)

            # uniquify image
            img_filename_dst = f"{timestamp}-{'-'.join([x for x in img.split(os.sep) if x != ''])}"
            src_img = os.path.join(src_dir, img)
            dst_img = os.path.join(dst_dir, img_filename_dst)
            if copy_or_move == "Move":
                shutil.move(src_img, dst_img)
            elif copy_or_move == "Copy":
                shutil.copy2(src_img, dst_img)

            # uniquify annotation
            ann_filename_dst = os.path.splitext(img_filename_dst)[0] + ".xml"
            src_ann = return_xml_path(os.path.join(src_dir, img))
            dst_ann = os.path.join(dst_dir, ann_filename_dst)
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
    if json_size > 10000:
        mb.showwarning(warning_txt[lang], [f"The JSON file is very large ({get_size(path_to_image_json)}). This can cause the verification"
                                            " step to perform very slow. It will work, but you'll have to have patience. ", "El archivo "
                                            f"JSON es muy grande ({get_size(path_to_image_json)}). Esto puede hacer que el paso de verificación"
                                            " funcione muy lentamente. Funcionará, pero tendrás que tener paciencia. "][lang])

    # check requirements
    check_json_presence_and_warn_user(["verify", "verificar"][lang],
                                      ["verifying", "verificando"][lang],
                                      ["verification", "verificación"][lang])
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
        if not mb.askyesno(["Verification session in progress", "Sesión de verificación en curso"][lang],
                            ["Do you want to continue with the previous verification session? If you press 'No', you will start a new session.", 
                            "¿Quieres continuar con la sesión de verificación anterior? Si presiona 'No', iniciará una nueva sesión."][lang]):
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
                mb.showerror(title=error_txt[lang],
                            message=["An error has occurred", "Ha ocurrido un error"][lang] + " (EcoAssist v" + version + "): '" + str(error) + "'.",
                            detail=traceback.format_exc())
    
    # start new session
    elif status == "done":
        if mb.askyesno(["Previous session is done", "Sesión anterior terminada."][lang], ["It seems like you have completed the previous manual "
                        "verification session. Do you want to start a new session?", "Parece que has completado la sesión de verificación manual "
                        "anterior. ¿Quieres iniciar una nueva sesión?"][lang]):
            open_hitl_settings_window()

# convert the pascal voc annotations to yolo so that training can start
def pascal_voc_to_yolo(folder_path):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # begin with a clean slate
    clean_training_dir(folder_path)

    # keep track of the files written
    yolo_written_file = os.path.join(folder_path, "yolo-files-written.txt")
    if os.path.exists(yolo_written_file):
        os.remove(yolo_written_file)

    # loop trough xml files
    send_to_output_window("\nConverting PASCAL VOC to YOLO annotation files...")
    classes_list = []
    counts = {'background' : 0, 'images': 0}
    classes_counts = {}
    index_d = 0
    with open(yolo_written_file, 'w') as yolo_written:
        for file in os.listdir(folder_path):
            
            # init vars
            is_background = None
            file_name, file_ext = os.path.splitext(file)

            # for all images
            if file_ext.lower() in ['.jpg', '.jpeg', '.gif', '.png']:
                counts['images'] += 1

                # show progress per 1000 files
                if index_d % 1000 == 0:
                    send_to_output_window(f"   currently at number {index_d}...")
                index_d += 1

                # an image without an xml is a background
                if not os.path.isfile(os.path.join(folder_path, f"{file_name}.xml")):
                    is_background = True

            #  loop through xmls
            if file.endswith('.xml'): 
                xml_path = os.path.join(folder_path, file)
                yolo_txt_path = os.path.join(folder_path, f"{file_name}.txt")
                tree = ET.parse(xml_path)
                root = tree.getroot()
                size = root.find('size')
                w = int(size.find('width').text)
                h = int(size.find('height').text)
                with open(yolo_txt_path, 'w') as yolo_file:
                    yolo_written.write(yolo_txt_path + '\n')

                    # a xml without any objects is also a background
                    is_background = True
                    for obj in root.findall('object'):
                        is_background = False
                        class_name = obj.find('name').text

                        # check if it is a known class
                        if class_name not in classes_list:
                            classes_list.append(class_name)

                        # keep count
                        if class_name in classes_counts:
                            classes_counts[class_name] += 1
                        else:
                            classes_counts[class_name] = 1
                        
                        # fetch and convert
                        class_id = classes_list.index(class_name)
                        bbox = obj.find('bndbox')
                        x_min = float(bbox.find('xmin').text)
                        y_min = float(bbox.find('ymin').text)
                        x_max = float(bbox.find('xmax').text)
                        y_max = float(bbox.find('ymax').text)
                        b = (float(bbox.find('xmin').text), float(bbox.find('xmax').text), float(bbox.find('ymin').text), float(bbox.find('ymax').text))
                        bbox = convert_bbox_pascal_to_yolo((w,h), b)

                        # write
                        yolo_line = str(class_id) + " " + " ".join([str(round(a, 6)) for a in bbox]) + '\n'
                        yolo_file.write(yolo_line)
            
            # count background
            if is_background == True:
                counts['background'] += 1
        
    # show progres
    send_to_output_window(f"   currently at number {index_d}...")
    send_to_output_window(f"   done!")

    # create classes.txt
    classes_txt = os.path.join(folder_path, "classes.txt")
    if not os.path.isfile(classes_txt):
        with open(classes_txt, 'w') as fp:
            for elem in classes_list:
                fp.write(f"{elem}\n")
    
    # count instances
    total_instances = 0
    for key, value in classes_counts.items():
        total_instances += value

    # show counts and proportions
    send_to_output_window("\nThe dataset constists of:")
    for key, value in classes_counts.items():
        send_to_output_window(f"   {value} instances of class {key} ({round(value / total_instances * 100, 1)}% of total n instances)")
    if counts['background'] == 0:
        send_to_output_window(f"   {counts['background']} background images (0.0% of total n images)")
    else:
        send_to_output_window(f"   {counts['background']} background images ({round(counts['background'] / counts['images'] * 100, 1)}% of total n images)")

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
        if name in inverted_label_map:
            new_class = False
        else:
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
        print(recognition_file)
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

# take MD json and classify detections
def classify_detections(json_fpath, data_type):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # adjust variables for images or videos
    if data_type == "img":
        progress_stats = progress_img_stats_cls
        progress_frame = progress_img_frame
        progress_progbar = progress_img_progbar_cls
    else:
        progress_stats = progress_vid_stats_cls
        progress_frame = progress_vid_frame
        progress_progbar = progress_vid_progbar_cls

    # show user it's loading
    progress_stats['text'] = create_md_progress_lbl(command = "load")
    root.update()

    # locate script
    if os.name == 'nt':
        classify_detections_script = os.path.join(EcoAssist_files, "EcoAssist", "classify_detections.bat")
    else:
        classify_detections_script = os.path.join(EcoAssist_files, "EcoAssist", "classify_detections.command")

    # get classification params
    if var_cls_animal_model.get() not in none_txt: 
        cls_animal_model_fpath = os.path.join(EcoAssist_files, "classification_models", "cls_animals", var_cls_animal_model.get())
        cls_animal_thresh = var_cls_animal_thresh.get()
        cls_animal_smooth = var_smooth_cls_animal.get()
    else:
        cls_animal_model_fpath = 'None'
        cls_animal_thresh = 'None'
        cls_animal_smooth = 'None'
    if var_cls_vehicle_model.get() not in none_txt: 
        cls_vehicle_model_fpath = os.path.join(EcoAssist_files, "classification_models", "cls_vehicles", var_cls_vehicle_model.get())
        cls_vehicle_thresh = var_cls_vehicle_thresh.get()
    else:
        cls_vehicle_model_fpath = 'None'
        cls_vehicle_thresh = 'None'
    if var_cls_person_model.get() not in none_txt: 
        cls_person_model_fpath = os.path.join(EcoAssist_files, "classification_models", "cls_persons", var_cls_person_model.get())
        cls_person_thresh = var_cls_person_thresh.get()
    else:
        cls_person_model_fpath = 'None'
        cls_person_thresh = 'None'

    # create command
    command_args = []
    command_args.append(classify_detections_script)
    command_args.append(EcoAssist_files)
    command_args.append(cls_animal_model_fpath)
    command_args.append(str(cls_animal_thresh))
    command_args.append(str(cls_animal_smooth))
    command_args.append(cls_vehicle_model_fpath)
    command_args.append(str(cls_vehicle_thresh))
    command_args.append(cls_person_model_fpath)
    command_args.append(str(cls_person_thresh))
    command_args.append(json_fpath)
    try:
        command_args.append(temp_frame_folder)
    except NameError:
        pass

    # adjust command for unix OS
    if os.name != 'nt':
        command_args = "'" + "' '".join(command_args) + "'"

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

    # cancel button
    btn_cancel_cls = Button(progress_frame, text=cancel_txt[lang], command=lambda: cancel_subprocess(p))
    btn_cancel_cls.grid(row=8, column=0, columnspan=2)

    # calculate metrics while running
    for line in p.stdout:
        print(line, end='')

        # catch smoothening info lines
        if "<EA>" in line:
            smooth_output_line = re.search('<EA>(.+)<EA>', line).group().replace('<EA>', '')
            smooth_output_file = os.path.join(os.path.dirname(json_fpath), "smooth-output.txt")
            with open(smooth_output_file, 'a+') as f:
                f.write(f"{smooth_output_line}\n")
            f.close()

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
            
            # order stats
            stats = create_md_progress_lbl(elapsed_time = elapsed_time,
                                            time_left = time_left,
                                            current_im = current_im,
                                            total_im = total_im,
                                            processing_speed = processing_speed,
                                            percentage = percentage,
                                            GPU_param = GPU_param,
                                            data_type = data_type,
                                            command = "running")
            
            # print stats
            progress_progbar['value'] = percentage
            progress_stats['text'] = stats
        root.update()

    # repeat when process is done
    progress_stats['text'] = create_md_progress_lbl(elapsed_time = elapsed_time,
                                                    time_left = time_left,
                                                    current_im = current_im,
                                                    total_im = total_im,
                                                    processing_speed = processing_speed,
                                                    percentage = percentage,
                                                    GPU_param = GPU_param,
                                                    data_type = data_type,
                                                    command = "done")
    root.update()



    # remove button after process is done
    btn_cancel_cls.grid_remove()

# quit popen process
def cancel_subprocess(process, process_killed = ""):
    global cancel_deploy_model_pressed
    if os.name == 'nt':
        Popen(f"TASKKILL /F /PID {process.pid} /T")
    else:
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    if process_killed == "deploy_model":
        cancel_deploy_model_pressed = True

# delpoy model and create json output files 
def deploy_model(path_to_image_folder, selected_options, data_type):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
    
    # adjust variables for images or videos
    if data_type == "img":
        progress_stats = progress_img_stats
        progress_frame = progress_img_frame
        progress_progbar = progress_img_progbar
    else:
        progress_stats = progress_vid_stats
        progress_frame = progress_vid_frame
        progress_progbar = progress_vid_progbar
    
    # display loading window
    progress_stats['text'] = create_md_progress_lbl(command="load", data_type = data_type)

    # prepare variables
    chosen_folder = str(Path(path_to_image_folder))
    run_detector_batch_py = os.path.join(EcoAssist_files, "cameratraps", "detection", "run_detector_batch.py")
    image_recognition_file = os.path.join(chosen_folder, "image_recognition_file.json")
    process_video_py = os.path.join(EcoAssist_files, "cameratraps", "detection", "process_video.py")
    video_recognition_file = "--output_json_file=" + os.path.join(chosen_folder, "video_recognition_file.json")
    GPU_param = "Unknown"

    # select model based on user input via dropdown menu
    custom_model_bool = False
    if var_model.get() == dpd_options_model[lang][0]: 
        # set model file
        model_file = os.path.join(EcoAssist_files, "pretrained_models", "md_v5a.0.0.pt")
        
        # set yolov5 git to accommodate old models
        switch_yolov5_git_to("old models")
        
    elif var_model.get() == dpd_options_model[lang][1]:
        # set model file
        model_file = os.path.join(EcoAssist_files, "pretrained_models", "md_v5b.0.0.pt")
        
        # set yolov5 git to accommodate old models
        switch_yolov5_git_to("old models")
    else:
        # set model file
        model_file = var_model_path.get()
        custom_model_bool = True

        # set yolov5 git to accommodate new models
        switch_yolov5_git_to("new models")
        
        # extract classes
        label_map = extract_label_map_from_model(model_file)

        # write labelmap to separate json
        json_object = json.dumps(label_map, indent=1)
        native_model_classes_json_file = os.path.join(chosen_folder, "native_model_classes.json")
        with open(native_model_classes_json_file, "w") as outfile:
            outfile.write(json_object)
        
        # add argument to command call
        selected_options.append("--class_mapping_filename=" + native_model_classes_json_file)
            
    # create commands for Windows
    if os.name == 'nt':
        if selected_options == []:
            img_command = [sys.executable, run_detector_batch_py, model_file, chosen_folder, image_recognition_file]
            vid_command = [sys.executable, process_video_py, video_recognition_file, model_file, chosen_folder]
        else:
            img_command = [sys.executable, run_detector_batch_py, model_file, *selected_options, chosen_folder, image_recognition_file]
            vid_command = [sys.executable, process_video_py, *selected_options, video_recognition_file, model_file, chosen_folder]

     # create command for MacOS and Linux
    else:
        if selected_options == []:
            img_command = [f"'{sys.executable}' '{run_detector_batch_py}' '{model_file}' '{chosen_folder}' '{image_recognition_file}'"]
            vid_command = [f"'{sys.executable}' '{process_video_py}' '{video_recognition_file}' '{model_file}' '{chosen_folder}'"]
        else:
            selected_options = "' '".join(selected_options)
            img_command = [f"'{sys.executable}' '{run_detector_batch_py}' '{model_file}' '{selected_options}' '{chosen_folder}' '{image_recognition_file}'"]
            vid_command = [f"'{sys.executable}' '{process_video_py}' '{selected_options}' '{video_recognition_file}' '{model_file}' '{chosen_folder}'"]

    # pick one command
    if data_type == "img":
        command = img_command
    else:
        command = vid_command
    
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
    
    # cancel button
    global cancel_deploy_model_pressed
    cancel_deploy_model_pressed = False
    btn_cancel = Button(progress_frame, text=cancel_txt[lang], command=lambda: cancel_subprocess(p, "deploy_model"))
    btn_cancel.grid(row=3, column=0, columnspan=2)
    
    # read output and direct to tkinter
    model_error_present = False
    model_error_log = os.path.join(chosen_folder, "model_error_log.txt")
    model_warning_present = False
    model_warning_log = os.path.join(chosen_folder, "model_warning_log.txt")
    for line in p.stdout:
        print(line, end='')
        
        # catch model errors
        if line.startswith("No image files found"):
            mb.showerror(["No images found", "No se han encontrado imágenes"][lang],
                        [f"There are no images found in '{chosen_folder}'. \n\nAre you sure you specified the correct folder?"
                        f" If the files are in subdirectories, make sure you don't tick '{lbl_exclude_subs_txt[lang]}'.",
                        f"No se han encontrado imágenes en '{chosen_folder}'. \n\n¿Está seguro de haber especificado la carpeta correcta?"][lang])
            return
        if line.startswith("No videos found"):
            mb.showerror(["No videos found", "No se han encontrado vídeos"][lang],
                        line + [f"\n\nAre you sure you specified the correct folder? If the files are in subdirectories, make sure you don't tick '{lbl_exclude_subs_txt}'.",
                                "\n\n¿Está seguro de haber especificado la carpeta correcta?"][lang])
            return
        if line.startswith("No frames extracted"):
            mb.showerror(["Could not extract frames", "No se pueden extraer fotogramas"][lang],
                        line + ["\n\nConverting the videos to .mp4 might fix the issue.",
                                "\n\nConvertir los vídeos a .mp4 podría solucionar el problema."][lang])
            return

        # write errors to log file
        if "Exception:" in line:
            model_error_present = True
            with open(model_error_log, 'a+') as f:
                f.write(f"{line}\n")
            f.close()

        # write warnings to log file
        if "Warning:" in line:
            if not "could not determine MegaDetector version" in line \
                and not "no metadata for unknown detector version" in line \
                and not "using user-supplied image size" in line:
                model_warning_present = True
                with open(model_warning_log, 'a+') as f:
                    f.write(f"{line}\n")
                f.close()
        
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
            
            # order stats
            stats = create_md_progress_lbl(elapsed_time = elapsed_time,
                                            time_left = time_left,
                                            current_im = current_im,
                                            total_im = total_im,
                                            processing_speed = processing_speed,
                                            percentage = percentage,
                                            GPU_param = GPU_param,
                                            data_type = data_type,
                                            command = "running")
            
            # print stats
            progress_progbar['value'] = percentage
            progress_stats['text'] = stats
        root.update()
    
    # repeat when process is done
    progress_stats['text'] = create_md_progress_lbl(elapsed_time = elapsed_time,
                                                    time_left = time_left,
                                                    current_im = current_im,
                                                    total_im = total_im,
                                                    processing_speed = processing_speed,
                                                    percentage = percentage,
                                                    GPU_param = GPU_param,
                                                    data_type = data_type,
                                                    command = "done")
    root.update()

    # show model error pop up window
    if model_error_present:
        mb.showerror(error_txt[lang], [f"There are one or more model errors. See\n\n'{model_error_log}'\n\nfor more information.",
                                        f"Hay uno o más errores de modelo. Consulte\n\n'{model_error_log}'\n\npara obtener más información."][lang])

    # show model warning pop up window
    if model_warning_present:
        mb.showerror(error_txt[lang], [f"There are one or more model warnings. See\n\n'{model_warning_log}'\n\nfor more information.",
                                    f"Hay uno o más advertencias de modelo. Consulte\n\n'{model_warning_log}'\n\npara obtener más información."][lang])

    # remove button after process is done
    btn_cancel.grid_remove()
    
    # create ecoassist metadata
    ecoassist_metadata = {"ecoassist_metadata" : {"version" : version,
                                                  "custom_model" : custom_model_bool,
                                                  "custom_model_info" : {}}}
    if custom_model_bool:
        ecoassist_metadata["ecoassist_metadata"]["custom_model_info"] = {"model_name" : os.path.basename(os.path.normpath(model_file)),
                                                                         "label_map" : label_map}
    
    # write metadata to json and make abosulte if specified
    image_recognition_file = os.path.join(chosen_folder, "image_recognition_file.json")
    video_recognition_file = os.path.join(chosen_folder, "video_recognition_file.json")
    if data_type == "img" and os.path.isfile(image_recognition_file):
        append_to_json(image_recognition_file, ecoassist_metadata)
        if var_abs_paths.get():
            make_json_absolute(image_recognition_file)
    if data_type == "vid" and os.path.isfile(video_recognition_file):
        append_to_json(video_recognition_file, ecoassist_metadata)
        if var_abs_paths.get():
            make_json_absolute(video_recognition_file)
    
    # classify detections if specified by user
    if not cancel_deploy_model_pressed:
        if var_cls_further.get() and var_cls_animal_model.get() not in none_txt or \
            var_cls_further.get() and var_cls_vehicle_model.get() not in none_txt or \
            var_cls_further.get() and var_cls_person_model.get() not in none_txt:
            if data_type == "img":
                classify_detections(os.path.join(chosen_folder, "image_recognition_file.json"), data_type)
            else:
                classify_detections(os.path.join(chosen_folder, "video_recognition_file.json"), data_type)

    # remove frames.json file
    frames_video_recognition_file = os.path.join(chosen_folder, "video_recognition_file.frames.json")
    if os.path.isfile(frames_video_recognition_file):
        os.remove(frames_video_recognition_file)

# open progress window and initiate the model deployment
def start_deploy():
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
    
    # fetch global variables
    global progress_img_frame
    global progress_vid_frame
    
    # check if user selected to process either images or videos
    if not var_process_img.get() and not var_process_vid.get():
        mb.showerror(["Nothing selected to be processed", "No se ha seleccionado nada para procesar"][lang],
                        message=["You selected neither images nor videos to be processed.",
                                 "No ha seleccionado ni imágenes ni vídeos para procesar."][lang])
        return
    
    # check if chosen folder is valid
    if var_choose_folder.get() in ["", "/", "\\", ".", "~", ":"] or not os.path.isdir(var_choose_folder.get()):
        mb.showerror(error_txt[lang],
            message=["Please specify a directory with data to be processed.",
                     "Por favor, especifique un directorio con los datos a procesar."][lang])
        return
    
    # check if checkpoint entry is valid
    if var_use_custom_img_size_for_deploy.get() and not var_image_size_for_deploy.get().isdecimal():
        mb.showerror(invalid_value_txt[lang],
                    ["You either entered an invalid value for the image size, or none at all. You can only "
                    "enter numberic characters.",
                    "Ha introducido un valor no válido para el tamaño de la imagen o no ha introducido ninguno. "
                    "Sólo puede introducir caracteres numéricos."][lang])
        return

    # check if checkpoint entry is valid
    if var_use_checkpnts.get() and not var_checkpoint_freq.get().isdecimal():
        if mb.askyesno(invalid_value_txt[lang],
                        ["You either entered an invalid value for the checkpoint frequency, or none at all. You can only "
                        "enter numberic characters.\n\nDo you want to proceed with the default value 500?",
                        "Ha introducido un valor no válido para la frecuencia del punto de control o no ha introducido ninguno. "
                        "Sólo puede introducir caracteres numéricos.\n\n¿Desea continuar con el valor por defecto 500?"][lang]):
            var_checkpoint_freq.set("500")
            ent_checkpoint_freq.config(fg='black')
        else:
            return
    
    # check if the nth frame entry is valid
    if var_not_all_frames.get() and not var_nth_frame.get().isdecimal():
        if mb.askyesno(invalid_value_txt[lang],
                        [f"You either entered an invalid value for '{lbl_nth_frame_txt[lang]}', or none at all. You can only "
                        "enter numberic characters.\n\nDo you want to proceed with the default value 10?\n\n"
                        "That means you process only 1 out of 10 frames, making the process time 10 times faster.",
                        f"Ha introducido un valor no válido para '{lbl_nth_frame_txt[lang]}', o no ha introducido ninguno. Sólo "
                        "puede introducir caracteres numéricos.\n\n¿Desea continuar con el valor por defecto 10?. Eso significa "
                        "que sólo se procesa 1 de cada 10 fotogramas, con lo que el tiempo de proceso es 10 veces más rápido."][lang]):
            var_nth_frame.set("10")
            ent_nth_frame.config(fg='black')
        else:
            return
        
    # create command for the image process to be passed on to run_detector_batch.py
    additional_img_options = ["--output_relative_filenames"]
    if not var_exclude_subs.get():
        additional_img_options.append("--recursive")
    if var_use_checkpnts.get():
        additional_img_options.append("--checkpoint_frequency=" + var_checkpoint_freq.get())
    if var_cont_checkpnt.get():
        additional_img_options.append("--resume_from_checkpoint=" + loc_chkpnt_file)
    if var_use_custom_img_size_for_deploy.get():
        additional_img_options.append("--image_size=" + var_image_size_for_deploy.get())

    # create command for the video process to be passed on to process_video.py
    additional_vid_options = []
    additional_vid_options.append("--json_confidence_threshold=0.01")
    if not var_exclude_subs.get():
        additional_vid_options.append("--recursive")
    if var_not_all_frames.get():
        additional_vid_options.append("--frame_sample=" + var_nth_frame.get())
    temp_frame_folder_created = False
    if var_process_vid.get():
        if var_cls_further.get() and var_cls_animal_model.get() not in none_txt or \
            var_cls_further.get() and var_cls_vehicle_model.get() not in none_txt or \
            var_cls_further.get() and var_cls_person_model.get() not in none_txt:
            global temp_frame_folder
            temp_frame_folder_obj = tempfile.TemporaryDirectory()
            temp_frame_folder_created = True
            temp_frame_folder = temp_frame_folder_obj.name
            additional_vid_options.append("--frame_folder=" + temp_frame_folder)
            additional_vid_options.append("--keep_extracted_frames")
    
    # open new window with progress bar and stats
    md_progress_window = Toplevel(root)
    md_progress_window.title("Deploy progress")
    md_progress_window.geometry()

    # logo
    logo = tk.Label(md_progress_window, image=grey_bg_logo)
    logo.grid(column=0, row=0, columnspan=2, sticky='ew', pady=(5, 0))

    # check if user has selected any cls models
    if var_cls_further.get() and var_cls_animal_model.get() not in none_txt or \
        var_cls_further.get() and var_cls_vehicle_model.get() not in none_txt or \
        var_cls_further.get() and var_cls_person_model.get() not in none_txt:
        any_cls_model_selected = True
    else:
        any_cls_model_selected = False

    # add image progress
    if var_process_img.get():
        progress_img_frame = LabelFrame(md_progress_window, text=[" Images ", " Imágenes "][lang], pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue')
        progress_img_frame.configure(font=(text_font, 15, "bold"))
        progress_img_frame.grid(column=0, row=1, columnspan=2, sticky='ew')
        progress_img_frame.columnconfigure(0, weight=3, minsize=115)
        progress_img_frame.columnconfigure(1, weight=1, minsize=115)
        ttk.Label(master=progress_img_frame, text="Detection progress", font=f'{text_font} {int(13 * text_size_adjustment_factor)} bold').grid(column=0, row=0, sticky = 'w')
        global progress_img_progbar
        progress_img_progbar = ttk.Progressbar(master=progress_img_frame, orient='horizontal', mode='determinate', length=280)
        progress_img_progbar.grid(column=0, row=1, columnspan=2, padx=5, pady=(3,0))
        global progress_img_stats
        progress_img_stats = ttk.Label(master=progress_img_frame, text=create_postprocess_lbl())
        progress_img_stats.grid(column=0, row=2, padx=5, pady=(0,3), columnspan=2)

        # progressbar for classification
        if any_cls_model_selected:
            ttk.Label(master=progress_img_frame, text="").grid(column=0, row=4, sticky = 'w')
            ttk.Label(master=progress_img_frame, text="Classification progress", font=f'{text_font} {int(13 * text_size_adjustment_factor)} bold').grid(column=0, row=5, sticky = 'w')
            global progress_img_progbar_cls
            progress_img_progbar_cls = ttk.Progressbar(master=progress_img_frame, orient='horizontal', mode='determinate', length=280)
            progress_img_progbar_cls.grid(column=0, row=6, columnspan=2, padx=5, pady=(3,0))
            global progress_img_stats_cls
            progress_img_stats_cls = ttk.Label(master=progress_img_frame, text=create_postprocess_lbl())
            progress_img_stats_cls.grid(column=0, row=7, padx=5, pady=(0,3), columnspan=2)

    # add video progress
    if var_process_vid.get():
        progress_vid_frame = LabelFrame(md_progress_window, text=[" Videos ", " Vídeos "][lang], pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue')
        progress_vid_frame.configure(font=(text_font, 15, "bold"))
        progress_vid_frame.grid(column=0, row=2, columnspan=2, sticky='ew')
        progress_vid_frame.columnconfigure(0, weight=3, minsize=115)
        progress_vid_frame.columnconfigure(1, weight=1, minsize=115)
        ttk.Label(master=progress_vid_frame, text="Detection progress", font=f'{text_font} {int(13 * text_size_adjustment_factor)} bold').grid(column=0, row=0, sticky = 'w')
        global progress_vid_progbar
        progress_vid_progbar = ttk.Progressbar(master=progress_vid_frame, orient='horizontal', mode='determinate', length=280)
        progress_vid_progbar.grid(column=0, row=1, columnspan=2, padx=10, pady=2)
        global progress_vid_stats
        progress_vid_stats = ttk.Label(master=progress_vid_frame, text=create_postprocess_lbl())
        progress_vid_stats.grid(column=0, row=2, columnspan=2)

        # progressbar for classification
        if any_cls_model_selected:
            ttk.Label(master=progress_vid_frame, text="").grid(column=0, row=4, sticky = 'w')
            ttk.Label(master=progress_vid_frame, text="Classification progress", font=f'{text_font} {int(13 * text_size_adjustment_factor)} bold').grid(column=0, row=5, sticky = 'w')
            global progress_vid_progbar_cls
            progress_vid_progbar_cls = ttk.Progressbar(master=progress_vid_frame, orient='horizontal', mode='determinate', length=280)
            progress_vid_progbar_cls.grid(column=0, row=6, columnspan=2, padx=5, pady=(3,0))
            global progress_vid_stats_cls
            progress_vid_stats_cls = ttk.Label(master=progress_vid_frame, text=create_postprocess_lbl())
            progress_vid_stats_cls.grid(column=0, row=7, padx=5, pady=(0,3), columnspan=2)
    
    try:
        # detect images ...
        if var_process_img.get():
            deploy_model(var_choose_folder.get(), additional_img_options, data_type = "img")

        # ... and/or videos
        if var_process_vid.get():
            deploy_model(var_choose_folder.get(), additional_vid_options, data_type = "vid")
        
        # reset window
        update_frame_states()
        
        # close progress window
        md_progress_window.destroy()

        # clean up temp folder with frames
        if temp_frame_folder_created:
            temp_frame_folder_obj.cleanup()

    except Exception as error:
        # log error
        print("ERROR:\n" + str(error) + "\n\nDETAILS:\n" + str(traceback.format_exc()) + "\n\n")
        
        # show error
        mb.showerror(title=error_txt[lang],
                     message=["An error has occurred", "Ha ocurrido un error"][lang] + " (EcoAssist v" + version + "): '" + str(error) + "'.",
                     detail=traceback.format_exc())
        
        # reset root with new states
        reset_frame_states()
        
        # close window
        md_progress_window.destroy()

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
        plt.bar(classes, counts, width = 0.4)
        plt.ylabel(["No. of instances verified", "No de instancias verificadas"][lang])
        plt.close()

        # return results
        return fig

# remove the temporary files created by the training
def clean_training_dir(folder_path):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # check if there were temp files
    yolo_written_file = os.path.join(folder_path, "yolo-files-written.txt")
    yolo_written_file_present = os.path.exists(yolo_written_file)
    if yolo_written_file_present:
        yolo_written_file_empty = os.stat(yolo_written_file).st_size == 0
    else:
        yolo_written_file_empty = True

    # remove yolo annotation files
    index = 0
    if yolo_written_file_present and not yolo_written_file_empty:
        send_to_output_window("\nCleaning up temporary files...")
        with open(yolo_written_file) as f:
            for txt_file in [line.rstrip() for line in f]:
                if index % 1000 == 0:
                    send_to_output_window(f"   currently at number {index}...")
                index += 1
                os.remove(txt_file)
        f.close()
        os.remove(yolo_written_file)
        send_to_output_window(f"   currently at number {index}...")
        send_to_output_window(f"   done!")
        classes_txt = os.path.join(folder_path, "classes.txt")
        if os.path.isfile(classes_txt):
            os.remove(classes_txt)
    else:
        send_to_output_window("\nNo temporary files. Nothing to clean...")

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
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # open patience window
    steps_progress = PatienceDialog(total = 8, text = [f"Loading...", f"Cargando..."][lang])
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
        counter = 0

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

                        # if one annotation treshold for all classes is specified
                        if rad_ann_val == 1 and conf >= ann_min_confs_generic:
                            display_annotation = True
                        
                        # if class-specific annotation tresholds are specified
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
    class_index = 1
    for category, files in selected_files.items():
        label_map = fetch_label_map_from_json(recognition_file)
        classes_list = [v for k, v in label_map.items()]
        row = classes_list.index(category) + 2
        frame = selection_dict[row]['frame']
        lbl_n_img = selection_dict[row]['lbl_n_img']
        chb_var = selection_dict[row]['chb_var'].get()
        rad_var = selection_dict[row]['rad_var'].get()

        # if user specified a precentage of total images
        if chb_var and rad_var == 2:

            # check if entry is valid
            ent_per_var = selection_dict[row]['ent_per_var'].get()
            try:
                ent_per_var = float(ent_per_var)
            except:
                invalid_value_warning([f"percentage of images for class '{category}'", f"porcentaje de imágenes para la clase '{category}'"][lang])
                return
            if ent_per_var == "" or ent_per_var < 0 or ent_per_var > 100:
                invalid_value_warning([f"percentage of images for class '{category}'", f"porcentaje de imágenes para la clase '{category}'"][lang])
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
                invalid_value_warning([f"number of images for class '{category}'", f"número de imágenes para la clase '{category}'"][lang])
                return
            if ent_amt_var == "":
                invalid_value_warning([f"number of images for class '{category}'", f"número de imágenes para la clase '{category}'"][lang])
                return

            # randomly select specified number of images
            total_n = len(files)
            n_selected = int(ent_amt_var)
            random.shuffle(files)
            files = files[:n_selected]

        # update label text 
        n_imgs = len(files)
        lbl_n_img.config(text = str(n_imgs))
        total_imgs += n_imgs

        # loop through the ultimately selected images and create files
        if prepare_files and len(files) > 0:

            # open patience window
            patience_dialog = PatienceDialog(total = n_imgs, text = [f"Preparing files for {category}...", f"Preparando archivos para {category}..."][lang])
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
                with open(file_list_txt, 'a') as f:
                    f.write(f"{os.path.normpath(img)}\n")
                    f.close()
                
                # # list annotaions 
                annotation_path = return_xml_path(img)

                # create xml file if not already present
                if not os.path.isfile(annotation_path):
                    create_pascal_voc_annotation(img, img_and_detections_dict[img]['annotations'], img_and_detections_dict[img]['human_verified'])

            # close patience window
            patience_dialog.close()      
    steps_progress.update_progress(current_step);current_step += 1
    steps_progress.close()

    # update total number of images
    lbl_n_total_imgs.config(text = [f"TOTAL: {total_imgs}", f"TOTAL: {total_imgs}"][lang])
    
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
            mb.showerror(title=error_txt[lang],
                        message=["An error has occurred", "Ha ocurrido un error"][lang] + " (EcoAssist v" + version + "): '" + str(error) + "'.",
                        detail=traceback.format_exc())

    # change json paths back, if converted earlier
    if json_paths_converted:
        make_json_absolute(recognition_file)

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
    hitl_settings_window = Toplevel(root)
    hitl_settings_window.title(["Verification selection settings", "Configuración de selección de verificación"][lang])
    hitl_settings_window.geometry()

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
    hitl_img_selection_frame = LabelFrame(hitl_settings_main_frame, text=[" Image selection criteria ", " Criterios de selección de imágenes "][lang],
                                            pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue', labelanchor = 'n')
    hitl_img_selection_frame.configure(font=(text_font, 15, "bold"))
    hitl_img_selection_frame.grid(column=0, row=1, columnspan=2, sticky='ew')
    hitl_img_selection_frame.columnconfigure(0, weight=1, minsize=50)
    hitl_img_selection_frame.columnconfigure(1, weight=1, minsize=200)
    hitl_img_selection_frame.columnconfigure(2, weight=1, minsize=200)
    hitl_img_selection_frame.columnconfigure(3, weight=1, minsize=200)
    hitl_img_selection_frame.columnconfigure(4, weight=1, minsize=200)

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
                                                    "number. Verification will adjust the results in the JSON file. This means that you can continue to use EcoAssist with verified "
                                                    "results and post-process as usual.", "Aquí puede especificar qué imágenes desea revisar. Si una detección se alinea con los "
                                                    "criterios elegidos, la imagen será elegida para el proceso de verificación. Tiene la opción de seleccionar un subconjunto de "
                                                    "sus imágenes según clases específicas, rangos de confianza y métodos de selección. Por ejemplo, la configuración"
                                                    " predeterminada le permitirá verificar imágenes con detecciones de las que el modelo está medio seguro "
                                                    "(con confianzas entre 0,2 y 0,8). Esto significa que no revisa las detecciones de alta confianza con "
                                                    "más de 0,8 de confianza y evita perder tiempo en detecciones de baja confianza de menos de 0,2. Siéntase"
                                                    " libre de ajustar estas configuraciones para adaptarlas a sus datos. Para determinar la cantidad de imágenes "
                                                    "que requerirán verificación según los criterios seleccionados, presione el botón 'Actualizar recuentos' a continuación. Si es "
                                                    "necesario, puede especificar un método de selección que elegirá aleatoriamente un subconjunto en función de un porcentaje o un "
                                                    "número absoluto. La verificación ajustará los resultados en el archivo JSON. Esto significa que puede continuar usando EcoAssist"
                                                    " con resultados verificados y realizar el posprocesamiento como de costumbre."][lang])
    text_hitl_img_selection_explanation.tag_add('explanation', '1.0', '1.end')

    # img table headers
    ttk.Label(master=hitl_img_selection_frame, text="").grid(column=0, row=1)
    ttk.Label(master=hitl_img_selection_frame, text="Class", font=f'{text_font} 13 bold').grid(column=1, row=1)
    ttk.Label(master=hitl_img_selection_frame, text="Confidence range", font=f'{text_font} 13 bold').grid(column=2, row=1)
    ttk.Label(master=hitl_img_selection_frame, text="Selection method", font=f'{text_font} 13 bold').grid(column=3, row=1)
    ttk.Label(master=hitl_img_selection_frame, text="Number of images", font=f'{text_font} 13 bold').grid(column=4, row=1)

    # ann selection frame
    hitl_ann_selection_frame = LabelFrame(hitl_settings_main_frame, text=[" Annotation selection criteria ", " Criterios de selección de anotaciones "][lang],
                                            pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue', labelanchor = 'n')
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
                                              " de los proyectos."][lang])
    text_hitl_ann_selection_explanation.tag_add('explanation', '1.0', '1.end')

    # ann same thresh
    rad_ann_var = IntVar()
    rad_ann_var.set(1)
    rad_ann_same = Radiobutton(hitl_ann_selection_frame, text=["Same annotation confidence threshold for all classes",
                                                               "Mismo umbral de confianza para todas las clases"][lang],
                                variable=rad_ann_var, value=1, command=lambda: toggle_hitl_ann_selection(rad_ann_var, hitl_ann_selection_frame))
    rad_ann_same.grid(row=1, column=1, columnspan=2, sticky='w')
    frame_ann_same = LabelFrame(hitl_ann_selection_frame, text="", pady=2, padx=5, relief=RAISED)
    frame_ann_same.grid(column=3, row=1, columnspan=2, sticky='ew')
    frame_ann_same.columnconfigure(0, weight=1, minsize=200)
    frame_ann_same.columnconfigure(1, weight=1, minsize=200)
    lbl_ann_same = ttk.Label(master=frame_ann_same, text=["All classes", "Todas las clases"][lang])
    lbl_ann_same.grid(row=0, column=0, sticky='w')
    scl_ann_var_generic = DoubleVar()
    scl_ann_var_generic.set(0.2)
    scl_ann = Scale(frame_ann_same, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, variable=scl_ann_var_generic, width=10, length=1, showvalue=0)
    scl_ann.grid(row=0, column=1, sticky='we')
    dsp_scl_ann = Label(frame_ann_same, textvariable=scl_ann_var_generic)
    dsp_scl_ann.grid(row=0, column=0, sticky='e', padx=5)

    # ann specific thresh
    rad_ann_gene = Radiobutton(hitl_ann_selection_frame, text=["Class-specific annotation confidence thresholds",
                                                               "Umbrales de confianza específicas de clase"][lang],
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
        chb_var.set(True)
        chb = tk.Checkbutton(frame, variable=chb_var, command=lambda e=row:enable_selection_widgets(e))
        lbl_class = ttk.Label(master=frame, text=v, state=NORMAL)
        min_conf = DoubleVar(value = 0.2)
        max_conf = DoubleVar(value = 0.8)
        fig = plt.figure(figsize = (2, 0.3))
        plt.hist(confs[k], bins = 10, range = (0,1))
        plt.xticks([])
        plt.yticks([])
        dist_graph = FigureCanvasTkAgg(fig, frame)
        plt.close()
        rsl = RangeSliderH(frame, [min_conf, max_conf], padX=11, digit_precision='.2f', bgColor = '#ececec', Width = 180)
        rad_var = IntVar()
        rad_var.set(1)
        rad_all = Radiobutton(frame, text=["All images in range", "Todo dentro del rango"][lang],
                                variable=rad_var, value=1, state=NORMAL, command=lambda e=row:enable_amt_per_ent(e))
        rad_per = Radiobutton(frame, text=["Subset percentage", "Subconjunto %"][lang],
                                variable=rad_var, value=2, state=NORMAL, command=lambda e=row:enable_amt_per_ent(e))
        rad_amt = Radiobutton(frame, text=["Subset number", "Subconjunto no."][lang],
                                variable=rad_var, value=3, state=NORMAL, command=lambda e=row:enable_amt_per_ent(e))
        ent_per_var = StringVar()
        ent_per = tk.Entry(frame, textvariable=ent_per_var, width=4, state=DISABLED)
        ent_amt_var = StringVar()
        ent_amt = tk.Entry(frame, textvariable=ent_amt_var, width=4, state=DISABLED)
        lbl_n_img = ttk.Label(master=frame, text="0", state=NORMAL)

        # annotation selection frame
        frame_ann = LabelFrame(hitl_ann_selection_frame, text="", pady=2, padx=5, relief=SUNKEN)
        frame_ann.grid(column=3, row=row, columnspan=2, sticky='ew')
        frame_ann.columnconfigure(0, weight=1, minsize=200)
        frame_ann.columnconfigure(1, weight=1, minsize=200)
        lbl_ann_gene = ttk.Label(master=frame_ann, text=v, state = DISABLED)
        lbl_ann_gene.grid(row=0, column=0, sticky='w')
        scl_ann_var_specific = DoubleVar()
        scl_ann_var_specific.set(0.20)
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
        rsl.grid(row = 0, rowspan= 3, column = 2, sticky = 's')
        rsl.lower()
        dist_graph.get_tk_widget().grid(row = 0, rowspan= 3, column = 2, sticky = 'n')
        rad_all.grid(row=0, column=3, sticky='w')
        rad_per.grid(row=1, column=3, sticky='w')
        ent_per.grid(row=1, column=3, sticky='e')
        rad_amt.grid(row=2, column=3, sticky='w')
        ent_amt.grid(row=2, column=3, sticky='e')
        lbl_n_img.grid(row = 1, column = 4)
        row_for_total_imgs_frame = row + 1

        # set row minsize
        set_minsize_rows(frame)

        # update window
        hitl_settings_window.update_idletasks()

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
    hitl_test_frame = LabelFrame(hitl_settings_main_frame, text=[" Actions ", " Acciones "][lang],
                                    pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue', labelanchor = 'n')
    hitl_test_frame.configure(font=(text_font, 15, "bold"))
    hitl_test_frame.grid(column=0, row=3, columnspan=2, sticky='ew')
    hitl_test_frame.columnconfigure(0, weight=1, minsize=115)
    hitl_test_frame.columnconfigure(1, weight=1, minsize=115)
    hitl_test_frame.columnconfigure(2, weight=1, minsize=115)

    # shorten texts for linux
    if sys.platform == "linux" or sys.platform == "linux2":
        btn_hitl_update_txt = ["Update counts", "La actualización cuenta"][lang]
        btn_hitl_show_txt = ["Show / hide annotation", "Mostrar / ocultar anotaciones"][lang]
        btn_hitl_start_txt = ["Start review process", "Iniciar proceso de revisión"][lang]
    else:
        btn_hitl_update_txt = ["Update counts", "La actualización cuenta"][lang]
        btn_hitl_show_txt = ["Show / hide annotation selection criteria", "Mostrar / ocultar criterios de anotaciones"][lang]
        btn_hitl_start_txt = ["Start review process with selected criteria", "Iniciar proceso de revisión"][lang]

    # buttons
    btn_hitl_update = Button(master=hitl_test_frame, text=btn_hitl_update_txt, width=1, command=lambda: select_detections(selection_dict = selection_dict, prepare_files = False))
    btn_hitl_update.grid(row=0, column=0, rowspan=1, sticky='nesw', padx=5)
    btn_hitl_show = Button(master=hitl_test_frame, text=btn_hitl_show_txt, width=1, command = toggle_hitl_ann_selection_frame)
    btn_hitl_show.grid(row=0, column=1, rowspan=1, sticky='nesw', padx=5)
    btn_hitl_start = Button(master=hitl_test_frame, text=btn_hitl_start_txt, width=1, command=lambda: select_detections(selection_dict = selection_dict, prepare_files = True))
    btn_hitl_start.grid(row=0, column=2, rowspan=1, sticky='nesw', padx=5)
    
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

############################################
############# HELPER FUNCTIONS #############
############################################

# helper function to quickly check the verification status of xml
def verification_status(xml):
    tree = ET.parse(xml)
    root = tree.getroot()
    try:
        verification_status = True if root.attrib['verified'] == 'yes' else False
    except:
        verification_status = False
    return verification_status

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
    ent_per = selection_dict[row]['ent_per']
    ent_amt = selection_dict[row]['ent_amt']
    lbl_n_img = selection_dict[row]['lbl_n_img']
    if chb_var:
        frame.config(relief = RAISED)
        lbl_class.config(state = NORMAL)
        rsl.grid(row = 0, rowspan= 3, column = 2)
        rad_all.config(state = NORMAL)
        rad_per.config(state = NORMAL)
        rad_amt.config(state = NORMAL)
        lbl_n_img.config(state = NORMAL)
    else:
        frame.config(relief = SUNKEN)
        lbl_class.config(state = DISABLED)
        rsl.grid_remove()
        rad_all.config(state = DISABLED)
        rad_per.config(state = DISABLED)
        rad_amt.config(state = DISABLED)
        lbl_n_img.config(state = DISABLED)

# update counts of the subset functions of the human-in-the-loop image selection frame
def enable_amt_per_ent(row):
    global selection_dict
    rad_var = selection_dict[row]['rad_var'].get()
    ent_per = selection_dict[row]['ent_per']
    ent_amt = selection_dict[row]['ent_amt']
    if rad_var == 1:
        ent_per.config(state = DISABLED)
        ent_amt.config(state = DISABLED)      
    if rad_var == 2:
        ent_per.config(state = NORMAL)
        ent_amt.config(state = DISABLED)
    if rad_var == 3:
        ent_per.config(state = DISABLED)
        ent_amt.config(state = NORMAL)

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

# disable annotation frame
def disable_ann_frame(row, hitl_ann_selection_frame):
    labelframe = hitl_ann_selection_frame.grid_slaves(row=row, column=3)[0]
    labelframe.config(relief=SUNKEN)
    for widget in labelframe.winfo_children():
        widget.config(state = DISABLED)

# enable annotation frame
def enable_ann_frame(row, hitl_ann_selection_frame):
    labelframe = hitl_ann_selection_frame.grid_slaves(row=row, column=3)[0]
    labelframe.config(relief=RAISED)
    for widget in labelframe.winfo_children():
        widget.config(state = NORMAL)

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

# return xml path with temp-folder squeezed in
def return_xml_path(img_path):
    head_path = var_choose_folder.get()
    tail_path = os.path.splitext(os.path.relpath(img_path, head_path))
    temp_xml_path = os.path.join(head_path, "temp-folder", tail_path[0] + ".xml")
    return os.path.normpath(temp_xml_path)

# temporary file which labelImg writes to notify EcoAssist that it should convert xml to coco
class LabelImgExchangeDir:
    def __init__(self, dir):
        self.dir = dir
        Path(self.dir).mkdir(parents=True, exist_ok=True)

    def create_file(self, content, idx):
        timestamp_miliseconds = str(str(datetime.date.today()) + str(datetime.datetime.now().strftime('%H%M%S%f'))).replace('-', '')
        temp_file = os.path.normpath(os.path.join(self.dir, f"{timestamp_miliseconds}-{idx}.txt"))
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

# simple window to show progressbar
class PatienceDialog:
    def __init__(self, total, text):
        self.root = tk.Tk()
        self.root.title("Have patience")
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
                percentage_value = round((current/self.total)*100)
                self.label.config(text = f"{self.text}\n{percentage_value}%")
            else:
                self.label.config(text = f"{self.text}\n{current} of {self.total}")
            self.root.update()

    def close(self):
        self.root.destroy()


# class for simple question with buttons
class TextButtonWindow:
    def __init__(self, title, text, buttons):
        self.root = tk.Tk()
        self.root.title(title)
        
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
        
    def close(self):
        self.root.destroy()
        
    def run(self):
        self.open()
        self.root.destroy()
        return self.selected_button

# delete temp folder
def delete_temp_folder(file_list_txt):
    temp_folder = os.path.dirname(file_list_txt)
    if os.path.isdir(temp_folder):
        shutil.rmtree(temp_folder)

# set button states to training
def set_buttons_to_training():
    btn_cancel_training.config(state=NORMAL)
    btn_start_training.config(state=DISABLED)

# set button states to idle
def set_buttons_to_idle():
    btn_cancel_training.config(state=DISABLED)
    btn_start_training.config(state=NORMAL)

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

# switch beteen versions of yolov5 git to accommodate either old or new models
def switch_yolov5_git_to(model_type):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
    
    # checkout repo
    repository = git.Repo(os.path.join(EcoAssist_files, "yolov5"))
    if model_type == "old models":
        if platform.processor() == "arm" and os.name != "nt": # M1 and M2
            repository.git.checkout("868c0e9bbb45b031e7bfd73c6d3983bcce07b9c1") 
        else:
            repository.git.checkout("c23a441c9df7ca9b1f275e8c8719c949269160d1")
    elif model_type == "new models":
        repository.git.checkout("3e55763d45f9c5f8217e4dad5ba1e6c1f42e3bf8")

# extract label map from custom model
def extract_label_map_from_model(model_file):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})")

    # import module from cameratraps dir
    from cameratraps.detection.pytorch_detector import PTDetector
            
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
        mb.showerror(title=error_txt[lang],
                     message=["An error has occurred when trying to extract classes", "Se ha producido un error al intentar extraer las clases"][lang] +
                                " (EcoAssist v" + version + "): '" + str(error) + "'" +
                                [".\n\nWill try to proceed and produce the output json file, but post-processing features of EcoAssist will not work.",
                                 ".\n\nIntentará continuar y producir el archivo json de salida, pero las características de post-procesamiento de EcoAssist no funcionarán."][lang],
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
    path = data['images'][0]['file']
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
            absolute_path = image['file']
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
    data['info']["ecoassist_metadata"]["hitl_status"] = status

    # write
    with open(path_to_json, "w") as json_file:
        json.dump(data, json_file, indent=1)

# get human-in-the-loop prgress variable
def get_hitl_var_in_json(path_to_json):
    # open
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
        ecoassist_metadata = data['info']["ecoassist_metadata"]
    
    # get status
    if "hitl_status" in ecoassist_metadata:
        status = ecoassist_metadata["hitl_status"]
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
            mb.showerror(error_txt[lang], [f"{noun.capitalize()} is not supported for videos.",
                                           f"{noun.capitalize()} no es compatible con vídeos."][lang])
            return True
        if not vid_json:
            mb.showerror(error_txt[lang], [f"No model output file present. Make sure you run step 2 before {continuous} the files. {noun.capitalize()} "
                                            "is only supported for images.",
                                           f"No hay archivos de salida del modelo. Asegúrese de ejecutar el paso 2 antes de {continuous} los archivos. "
                                           f"{noun.capitalize()} sólo es compatible con imágenes"][lang])
            return True
    if img_json:
        if vid_json:
            mb.showinfo(warning_txt[lang], [f"{noun.capitalize()} is not supported for videos. Will continue to only {infinitive} the images...",
                                            f"No se admiten {noun.capitalize()} para los vídeos. Continuará sólo {infinitive} las imágenes..."][lang])

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

# check if checkpoint file is present and assign global variable
def check_checkpnt():
    global loc_chkpnt_file
    for filename in os.listdir(var_choose_folder.get()):
            if re.search('^checkpoint_\d+\.json$', filename):
                loc_chkpnt_file = os.path.join(var_choose_folder.get(), filename)
                return True
    mb.showinfo(["No checkpoint file found", "No se ha encontrado ningún archivo de puntos de control"][lang],
                    ["There is no checkpoint file found. Cannot continue from checkpoint file...",
                    "No se ha encontrado ningún archivo de punto de control. No se puede continuar desde el archivo de punto de control..."][lang])
    return False

# order statistics from model output and return string
def create_md_progress_lbl(elapsed_time="",
                           time_left="",
                           current_im="",
                           total_im="",
                           processing_speed="",
                           percentage="",
                           GPU_param="",
                           data_type="",
                           command=""):

    # set unit
    if data_type == "img":
        unit = ["image", "imagen"][lang]
    else:
        unit = ["frame", "fotograma"][lang]
    
    # translate processing speed 
    if "it/s" in processing_speed:
        speed_prefix = [f"{unit.capitalize()} per sec:", f"{unit.capitalize()} por seg:"][lang]
        speed_suffix = processing_speed.replace("it/s", "")
    elif "s/it" in processing_speed:
        speed_prefix = [f"Sec per {unit}: ", f"seg por {unit}:"][lang]
        speed_suffix = processing_speed.replace("s/it", "")
    else:
        speed_prefix = ""
        speed_suffix = ""
        
    # loading
    if command == "load":
        return ["Algorithm is starting up...", "El algoritmo está comenzando..."][lang]
    
    # running (OS dependant)
    if command == "running":

        # windows
        if os.name == "nt":
            tab1 = "\t" if data_type == "img" else "\t\t"
            return f"{perc_done_txt[lang]}:\t\t{percentage}%\n" \
                f"{processing_txt[lang]} {unit}:{tab1}{current_im} {of_txt[lang]} {total_im}\n" \
                f"{elapsed_time_txt[lang]}:\t\t{elapsed_time}\n" \
                f"{remaining_time_txt[lang]}:\t\t{time_left}\n" \
                f"{speed_prefix}\t\t{speed_suffix}\n" \
                f"{running_on_txt[lang]}:\t\t{GPU_param}"

        # linux
        elif sys.platform == "linux" or sys.platform == "linux2":
            return f"{perc_done_txt[lang]}:\t{percentage}%\n" \
                f"{processing_txt[lang]} {unit}:\t{current_im} {of_txt[lang]} {total_im}\n" \
                f"{elapsed_time_txt[lang]}:\t\t{elapsed_time}\n" \
                f"{remaining_time_txt[lang]}:\t\t{time_left}\n" \
                f"{speed_prefix}\t\t{speed_suffix}\n" \
                f"{running_on_txt[lang]}:\t\t{GPU_param}"

        # macos
        elif sys.platform == "darwin":
            return f"{perc_done_txt[lang]}:\t{percentage}%\n" \
                f"{processing_txt[lang]} {unit}:\t{current_im} {of_txt[lang]} {total_im}\n" \
                f"{elapsed_time_txt[lang]}:\t{elapsed_time}\n" \
                f"{remaining_time_txt[lang]}:\t{time_left}\n" \
                f"{speed_prefix}\t{speed_suffix}\n" \
                f"{running_on_txt[lang]}:\t{GPU_param}"
    
    # done
    if command == "done":
        return f"{elapsed_time_txt[lang]}:\t{elapsed_time}\n" \
            f"{speed_prefix}\t{speed_suffix}"     

# get post-processing statistics and return string
def create_postprocess_lbl(elapsed_time="", time_left="", command=""):
    # waiting
    if command == "":
        return ["In queue", "Es espera"][lang]
    
    # running
    if command == "running":
        return f"{elapsed_time_txt[lang]}:\t\t{elapsed_time}\n" \
               f"{remaining_time_txt[lang]}:\t\t{time_left}"
               
    # done
    if command == "done":
        return ["Done!\n", "¡Hecho!\n"][lang]

# browse directory
def browse_dir(var, var_short, dsp, cut_off_length, n_row, n_column, str_sticky):    
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
    if len(dsp_chosen_dir) > cut_off_length:
        dsp_chosen_dir = "..." + dsp_chosen_dir[0 - cut_off_length + 3:]
    var_short.set(dsp_chosen_dir)
    dsp.grid(column=n_column, row=n_row, sticky=str_sticky)

# choose a custom classifier for animals
def model_cls_animal_options(self):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # remove or show widgets
    if self not in none_txt:
        place_cls_animal_options()
    else:
        remove_cls_animal_options()

# choose a custom classifier for vehicles
def model_cls_vehicle_options(self):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # remove or show widgets
    if self not in none_txt:
        place_cls_vehicle_options()
    else:
        remove_cls_vehicle_options()

# choose a custom classifier for persons
def model_cls_person_options(self):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # remove or show widgets
    if self not in none_txt:
        place_cls_person_options()
    else:
        remove_cls_person_options()

# load a custom yolov5 model
def model_options(self):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
   
    # if custom model is selected
    if var_model.get() == dpd_options_model[lang][2]:
        
        # choose, display and set global var
        browse_file(var_model,
                    var_model_short,
                    var_model_path,
                    dsp_model,
                    [("Yolov5 model","*.pt")],
                    30,
                    dpd_options_model[lang],
                    row_model)

    else:
        var_model_short.set("")

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
    if frame.cget('text').startswith(f' {step_txt[lang]} 2'):
        if os.path.isfile(image_recognition_file):
            open_file_or_folder(image_recognition_file)
        if os.path.isfile(video_recognition_file):
            open_file_or_folder(video_recognition_file)
    
    # open destination folder at step 4
    if frame.cget('text').startswith(f' {step_txt[lang]} 4'):
        open_file_or_folder(var_output_dir.get())

# open file or folder
def open_file_or_folder(path):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")
    
    # set language var
    error_opening_results_txt = ["Error opening results", "Error al abrir los resultados"]

    # open file
    if platform.system() == 'Darwin': # mac  
        try:
            subprocess.call(('open', path))
        except:
            mb.showerror(error_opening_results_txt[lang], [f"Could not open '{path}'. You'll have to find it yourself...",
                                                           f"No se ha podido abrir '{path}'. Tendrás que encontrarlo tú mismo..."][lang])
    elif platform.system() == 'Windows': # windows
        try:
            os.startfile(path)
        except:
            mb.showerror(error_opening_results_txt[lang], [f"Could not open '{path}'. You'll have to find it yourself...",
                                                           f"No se ha podido abrir '{path}'. Tendrás que encontrarlo tú mismo..."][lang])
    else: # linux
        try:
            subprocess.call(('xdg-open', path))
        except:
            try:
                subprocess.call(('gnome-open', path))
            except:
                mb.showerror(error_opening_results_txt[lang], [f"Could not open '{path}'. Neither the 'xdg-open' nor 'gnome-open' command worked. "
                                                               "You'll have to find it yourself...",
                                                               f"No se ha podido abrir '{path}'. Ni el comando 'xdg-open' ni el 'gnome-open' funcionaron. "
                                                               "Tendrá que encontrarlo usted mismo..."][lang])

##############################################
############# FRONTEND FUNCTIONS #############
##############################################

# refresh dropdown menu options
def update_dpd_options(dpd, master, var, options, cmd, row, lbl, from_lang):

    # recreate new option menu with updated options
    dpd.grid_forget()
    index = options[from_lang].index(var.get()) # get dpd index
    var.set(options[lang][index]) # set to previous index
    if cmd:
        dpd = OptionMenu(master, var, *options[lang], command=cmd)
    else:
        dpd = OptionMenu(master, var, *options[lang])
    dpd.configure(width=1)
    dpd.grid(row=row, column=1, sticky='nesw', padx=5)

    # only grid model_architechture if its label is displayed
    if lbl.cget("text") == lbl_model_architecture_txt[lang] and not lbl.winfo_ismapped():
        dpd.grid_forget()

    # give it same state as its label
    dpd.config(state = str(lbl['state']))

# refresh ent texts
def update_ent_text(var, string):
    if var.get() == "":
        return
    if no_user_input(var):
        original_state = str(var['state'])
        var.config(state=NORMAL, fg='grey')
        var.delete(0, tk.END)
        var.insert(0, string)
        var.config(state=original_state)

# change language
lang = 0
def set_language(to_lang):
    global lang
    from_lang = lang

    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # set language vars
    if to_lang == "gb":
        gb_widget.config(highlightbackground="black", relief="sunken")
        es_widget.config(highlightbackground="white", relief="raised")
        lang = 0
    if to_lang == "es":
        gb_widget.config(highlightbackground="white", relief="raised")
        es_widget.config(highlightbackground="black", relief="sunken")
        lang = 1

    # update addax text
    lbl_addax.config(text=lbl_addax_txt[lang])

    # update tab texts
    tabControl.tab(deploy_tab, text=deploy_tab_text[lang])
    tabControl.tab(train_tab, text=train_tab_text[lang])
    tabControl.tab(help_tab, text=help_tab_text[lang])
    tabControl.tab(about_tab, text=about_tab_text[lang])

    # update texts of deploy tab
    fst_step.config(text=" " + fst_step_txt[lang] + " ")
    lbl_choose_folder.config(text=lbl_choose_folder_txt[lang])
    btn_choose_folder.config(text=browse_txt[lang])
    snd_step.config(text=" " + snd_step_txt[lang] + " ")
    lbl_model.config(text=lbl_model_txt[lang])
    update_dpd_options(dpd_model, snd_step, var_model, dpd_options_model, model_options, row_model, lbl_model, from_lang)
    lbl_exclude_subs.config(text=lbl_exclude_subs_txt[lang])
    lbl_use_custom_img_size_for_deploy.config(text=lbl_use_custom_img_size_for_deploy_txt[lang])
    lbl_image_size_for_deploy.config(text=lbl_image_size_for_deploy_txt[lang])
    update_ent_text(ent_image_size_for_deploy, f"{eg_txt[lang]}: 640")
    lbl_abs_paths.config(text=lbl_abs_paths_txt[lang])
    lbl_process_img.config(text=lbl_process_img_txt[lang])
    cls_frame.config(text=" ↳ " + cls_frame_txt[lang] + " ")
    update_dpd_options(dpd_cls_animal_model, cls_frame, var_cls_animal_model, dpd_options_cls_animal_model, model_cls_animal_options, row_cls_animal_model, lbl_cls_animal_model, from_lang)
    lbl_cls_animal_model.config(text="     " + lbl_cls_animal_model_txt[lang])
    lbl_cls_animal_thresh.config(text="        ↳ " + lbl_cls_animal_thresh_txt[lang])
    lbl_smooth_cls_animal.config(text="        ↳ " + lbl_smooth_cls_animal_txt[lang])
    lbl_cls_vehicle_model.config(text="     " + lbl_cls_vehicle_model_txt[lang])
    lbl_cls_vehicle_thresh.config(text="        ↳ " + lbl_cls_vehicle_thresh_txt[lang])
    lbl_cls_person_model.config(text="     " + lbl_cls_person_model_txt[lang])
    lbl_cls_person_thresh.config(text="        ↳ " + lbl_cls_person_thresh_txt[lang])
    img_frame.config(text=" ↳ " + img_frame_txt[lang] + " ")
    lbl_use_checkpnts.config(text="     " + lbl_use_checkpnts_txt[lang])
    lbl_checkpoint_freq.config(text="        ↳ " + lbl_checkpoint_freq_txt[lang])
    update_ent_text(ent_checkpoint_freq, f"{eg_txt[lang]}: 500")
    lbl_cont_checkpnt.config(text="     " + lbl_cont_checkpnt_txt[lang])
    lbl_process_vid.config(text=lbl_process_vid_txt[lang])
    vid_frame.config(text=" ↳ " + vid_frame_txt[lang] + " ")
    lbl_not_all_frames.config(text="     " + lbl_not_all_frames_txt[lang])
    lbl_nth_frame.config(text="        ↳ " + lbl_nth_frame_txt[lang])
    update_ent_text(ent_nth_frame, f"{eg_txt[lang]}: 10")
    btn_start_deploy.config(text=btn_start_deploy_txt[lang])
    trd_step.config(text=" " + trd_step_txt[lang] + " ")
    lbl_hitl_main.config(text=lbl_hitl_main_txt[lang])
    btn_hitl_main.config(text=["Start", "Iniciar"][lang])
    fth_step.config(text=" " + fth_step_txt[lang] + " ")
    lbl_output_dir.config(text=lbl_output_dir_txt[lang])
    btn_output_dir.config(text=browse_txt[lang])
    lbl_separate_files.config(text=lbl_separate_files_txt[lang])
    sep_frame.config(text=" ↳ " + sep_frame_txt[lang] + " ")
    lbl_file_placement.config(text="     " + lbl_file_placement_txt[lang])
    rad_file_placement_move.config(text=["Copy", "Copiar"][lang])
    rad_file_placement_copy.config(text=["Move", "Mover"][lang])
    lbl_sep_conf.config(text="     " + lbl_sep_conf_txt[lang])
    lbl_vis_files.config(text=lbl_vis_files_txt[lang])
    lbl_crp_files.config(text=lbl_crp_files_txt[lang])
    lbl_csv.config(text=lbl_csv_txt[lang])
    lbl_thresh.config(text=lbl_thresh_txt[lang])
    btn_start_postprocess.config(text=btn_start_postprocess_txt[lang])

    
    # update texts of train tab
    req_params.config(text=" " + req_params_txt[lang] + " ")
    lbl_train_type.config(text=lbl_train_type_txt[lang])
    update_dpd_options(dpd_train_type, req_params, var_train_type, dpd_train_type_options, toggle_train_type, row_train_type, lbl_train_type, from_lang)
    lbl_annotated_data.config(text=lbl_annotated_data_txt[lang])
    btn_annotated_data.config(text=browse_txt[lang])
    lbl_learning_model.config(text=lbl_learning_model_txt[lang])
    update_dpd_options(dpd_learning_model, req_params, var_learning_model, dpd_learning_model_options, set_learning_model, row_learning_model, lbl_learning_model, from_lang)
    lbl_model_architecture.config(text=lbl_model_architecture_txt[lang])
    update_dpd_options(dpd_model_architecture, req_params, var_model_architecture, dpd_model_architecture_options, set_model_architecture, row_model_architecture, lbl_model_architecture, from_lang)
    lbl_n_epochs.config(text=lbl_n_epochs_txt[lang])
    update_ent_text(ent_n_epochs, f"{eg_txt[lang]}: 300")
    lbl_results_dir.config(text=lbl_results_dir_txt[lang])
    btn_results_dir.config(text=browse_txt[lang])
    lbl_resume_checkpoint.config(text=lbl_resume_checkpoint_txt[lang])
    btn_resume_checkpoint.config(text=browse_txt[lang])
    lbl_project_name.config(text=lbl_project_name_txt[lang])
    update_ent_text(ent_project_name, f"{eg_txt[lang]}: {['Tiger ID', 'Proyecto A'][lang]}")
    adv_params.config(text=" " + adv_params_txt[lang] + " ")
    lbl_val_prop.config(text=lbl_val_prop_txt[lang])
    lbl_test_prop.config(text=lbl_test_prop_txt[lang])
    lbl_train_gpu.config(text=lbl_train_gpu_txt[lang])
    lbl_batch_size.config(text=f"{lbl_batch_size_txt[lang]} {lbl_batch_size_txt_extra[lang]}")
    update_ent_text(ent_batch_size, f"{eg_txt[lang]}: 8")
    lbl_n_workers.config(text=f"{lbl_n_workers_txt[lang]} {lbl_n_workers_txt_extra[lang]}")
    update_ent_text(ent_n_workers, f"{eg_txt[lang]}: 2")
    lbl_image_size_for_training.config(text=f"{lbl_image_size_for_training_txt[lang]} {lbl_image_size_for_training_txt_extra[lang]}")
    update_ent_text(ent_image_size_for_training, f"{eg_txt[lang]}: 1280")
    lbl_cache_imgs.config(text=lbl_cache_imgs_txt[lang])
    lbl_hyper_file.config(text=lbl_hyper_file_txt[lang])
    update_dpd_options(dpd_hyper_file, adv_params, var_hyper_file, dpd_hyper_file_options, set_hyper_file, row_hyper_file, lbl_hyper_file, from_lang)
    lbl_evolve.config(text=lbl_evolve_txt[lang])
    lbl_n_generations.config(text=f"{lbl_n_generations_txt[lang]} {lbl_n_generations_txt_extra[lang]}")
    update_ent_text(ent_n_generations, f"{eg_txt[lang]}: 500")
    lbl_run_name.config(text=f"{lbl_run_name_txt[lang]} {lbl_run_name_txt_extra[lang]}")
    update_ent_text(ent_run_name, f"{eg_txt[lang]}: {['Initial run', 'Proceso inicial'][lang]}")
    lbl_n_freeze_layers.config(text=f"{lbl_n_freeze_layers_txt[lang]} {lbl_n_freeze_layers_txt_extra[lang]}")
    update_ent_text(ent_n_freeze_layers, f"{eg_txt[lang]}: 12")
    btn_start_training.config(text=btn_start_training_txt[lang])
    train_output.config(text=" " + train_output_txt[lang] + " ")
    btn_cancel_training.config(text=btn_cancel_training_txt[lang])

    # update texts of help tab
    help_text.config(state=NORMAL)
    help_text.delete('1.0', END)
    write_help_tab()

    # update texts of about tab
    about_text.config(state=NORMAL)
    about_text.delete('1.0', END)
    write_about_tab()

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
            btn_hitl_main.config(text = ["Start", "Iniciar"][lang])
        elif status == "in-progress":
            enable_frame(trd_step)
            btn_hitl_main.config(text = ["Continue", "Continuar"][lang])
        elif status == "done":
            complete_frame(trd_step)
    else:
        disable_frame(trd_step)

# set hyperparameter file variable based on user selection
def set_hyper_file(self):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # if "other" is selected
    if self == dpd_hyper_file_options[lang][6]:
        browse_file(var_hyper_file,
                    var_hyper_file_short,
                    var_hyper_file_path,
                    dsp_hyper_file,
                    [("YAML file","*.yaml")],
                    20,
                    dpd_hyper_file_options[lang],
                    row_hyper_file)
    
    # if one of the pre-defined files is selected
    else:
        yolo_hyps = os.path.join(EcoAssist_files, "yolov5", "data", "hyps")
        if self == dpd_hyper_file_options[lang][0]:
            var_hyper_file_path.set("")
        elif self == dpd_hyper_file_options[lang][1]:
            var_hyper_file_path.set(os.path.join(yolo_hyps, "hyp.scratch-low.yaml"))
        elif self == dpd_hyper_file_options[lang][2]:
            var_hyper_file_path.set(os.path.join(yolo_hyps, "hyp.scratch-med.yaml"))
        elif self == dpd_hyper_file_options[lang][3]:
            var_hyper_file_path.set(os.path.join(yolo_hyps, "hyp.scratch-high.yaml"))
        elif self == dpd_hyper_file_options[lang][4]:
            var_hyper_file_path.set(os.path.join(yolo_hyps, "hyp.Objects365.yaml"))
        elif self == dpd_hyper_file_options[lang][5]:
            var_hyper_file_path.set(os.path.join(yolo_hyps, "hyp.VOC.yaml"))

# set model architecture variable based on user selection
def set_model_architecture(self):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # if "other config" is selected
    if self == dpd_model_architecture_options[lang][5]:
        browse_file(var_model_architecture,
                    var_model_architecture_short,
                    var_model_architecture_path,
                    dsp_model_architecture,
                    [("YAML file","*.yaml")],
                    20,
                    dpd_model_architecture_options[lang],
                    row_model_architecture)
    
    # if one of the pre-defined archs is selected
    else:
        model_architectures = os.path.join(EcoAssist_files, "yolov5", "models")
        if self == dpd_model_architecture_options[lang][0]:
            var_model_architecture_path.set(os.path.join(model_architectures, "yolov5n.yaml"))
        elif self == dpd_model_architecture_options[lang][1]:
            var_model_architecture_path.set(os.path.join(model_architectures, "yolov5s.yaml"))
        elif self == dpd_model_architecture_options[lang][2]:
            var_model_architecture_path.set(os.path.join(model_architectures, "yolov5m.yaml"))
        elif self == dpd_model_architecture_options[lang][3]:
            var_model_architecture_path.set(os.path.join(model_architectures, "yolov5l.yaml"))
        elif self == dpd_model_architecture_options[lang][4]:
            var_model_architecture_path.set(os.path.join(model_architectures, "yolov5x.yaml"))
        elif self == dpd_model_architecture_options[lang][6]:
            var_model_architecture_path.set("")

# set learning model variable based on user selection
def set_learning_model(self):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # set variable
    choice = var_learning_model.get()

    # user selected custom model
    if choice == dpd_learning_model_options[lang][7]:
        # choose file
        browse_file(var_learning_model,
                    var_learning_model_short,
                    var_learning_model_path,
                    dsp_learning_model,
                    [("Yolov5 model","*.pt")],
                    20,
                    dpd_learning_model_options[lang],
                    row_learning_model)

        # add widget to freeze layers if custom model is selected
        if not var_learning_model_path.get().startswith(os.path.join(EcoAssist_files, "pretrained_models")):
            lbl_n_freeze_layers.grid(row=row_n_freeze_layers, sticky='nesw')
            ent_n_freeze_layers.grid(row=row_n_freeze_layers, column=1, sticky='nesw', padx=5)
    
    # user selected pre-defined model or from scratch
    else:
        var_learning_model_short.set("")
        lbl_n_freeze_layers.grid_forget()
        ent_n_freeze_layers.grid_forget()

    # show scratch learning widgets
    if choice == dpd_learning_model_options[lang][8]:
        lbl_model_architecture.grid(row=row_model_architecture, sticky='nesw', pady=2)
        dpd_model_architecture = OptionMenu(req_params, var_model_architecture, *dpd_model_architecture_options[lang], command=set_model_architecture)  # recreate dpd with translated options
        dpd_model_architecture.grid(row=row_model_architecture, column=1, sticky='nesw', padx=5)
        dpd_model_architecture.configure(width=1)
 
    # hide scratch learning widgets
    else:
        remove_widgets_based_on_location(master = req_params,
                                         rows = [row_model_architecture],
                                         cols = [0, 1])
    
    # set path to model
    pretrained_models = os.path.join(EcoAssist_files, "pretrained_models")
    if choice == dpd_learning_model_options[lang][0]:
        var_learning_model_path.set(os.path.join(pretrained_models, "md_v5a.0.0.pt"))
    elif choice == dpd_learning_model_options[lang][1]:
        var_learning_model_path.set(os.path.join(pretrained_models, "md_v5b.0.0.pt"))
    elif choice == dpd_learning_model_options[lang][2]:
        var_learning_model_path.set(os.path.join(pretrained_models, "yolov5n.pt"))
    elif choice == dpd_learning_model_options[lang][3]:
        var_learning_model_path.set(os.path.join(pretrained_models, "yolov5s.pt"))
    elif choice == dpd_learning_model_options[lang][4]:
        var_learning_model_path.set(os.path.join(pretrained_models, "yolov5m.pt"))
    elif choice == dpd_learning_model_options[lang][5]:
        var_learning_model_path.set(os.path.join(pretrained_models, "yolov5l.pt"))
    elif choice == dpd_learning_model_options[lang][6]:
        var_learning_model_path.set(os.path.join(pretrained_models, "yolov5x.pt"))
    elif choice == dpd_learning_model_options[lang][8]:
        var_learning_model_path.set("")

# set global cancel var to end training
def cancel_training():
    cancel_training_bool.set(True)

# check if user entered text in entry widget
def no_user_input(var):
    if var.get() == "" or var.get().startswith("E.g.:") or var.get().startswith("Ejem.:"):
        return True
    else:
        return False

# send text to output window and log 
def send_to_output_window(txt):
    # show user
    txt_train_output.configure(state=NORMAL)
    txt_train_output.insert(END, f"{txt}\n")
    txt_train_output.see("end")
    txt_train_output.configure(state=DISABLED)

    # log
    print(txt)

# show warning if not valid input
def invalid_value_warning(str, numeric = True):
    string = [f"You either entered an invalid value for the {str}, or none at all.", f"Ingresó un valor no válido para {str} o ninguno."][lang] 
    if numeric:
        string += [" You can only enter numberic characters.", " Solo puede ingresar caracteres numéricos."][lang]
    mb.showerror(invalid_value_txt[lang], string)

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
        widget.config(state=DISABLED)

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

# show and hide project widget depending on existing projects and user input
def grid_project_name():
    # set vars
    global var_project_name
    global ent_project_name
    global lbl_project_name
    global lbl_project_name_txt

    # remove all project name widgets
    remove_widgets_based_on_location(master = req_params,
                                     rows = [row_project_name],
                                     cols = [0, 1])

    # check dir validity
    if var_results_dir.get() in ["", "/", "\\", ".", "~", ":"] or not os.path.isdir(var_results_dir.get()):
        req_params.grid_rowconfigure(row_project_name, minsize=0)
        return
    
    # set min row size
    req_params.grid_rowconfigure(row_project_name, minsize=minsize_rows)

    # check if any existing projects
    dpd_project_name_options = sorted([o for o in os.listdir(var_results_dir.get()) if os.path.isdir(os.path.join(var_results_dir.get(), o))])

    # shared label widget
    lbl_project_name_txt = ["Project name", "Nombre del proyecto"]
    lbl_project_name = tk.Label(req_params, text=lbl_project_name_txt[lang], pady=2, width=1, anchor="w")
    lbl_project_name.grid(row=row_project_name, sticky='nesw')

    # if existing projects: dropdown menu
    if len(dpd_project_name_options) != 0:
        dpd_project_name_options.append(new_project_txt[lang])
        dpd_project_name = OptionMenu(req_params, var_project_name, *dpd_project_name_options, command=swtich_dropdown_to_entry)
        dpd_project_name.configure(width=1)
        dpd_project_name.grid(row=row_project_name, column=1, sticky='nesw', padx=5)
        var_project_name.set(dpd_project_name_options[0])

    # if no existing projects: entry box
    else:
        ent_project_name.grid(row=row_project_name, column=1, sticky='nesw', padx=5)
        var_project_name.set("")

        # first time user will see this entry box
        if ent_project_name.cget("fg") == "grey":
            ent_project_name.insert(0, f"{eg_txt[lang]}: {['Tiger ID', 'Proyecto A'][lang]}")
            ent_project_name.bind("<FocusIn>", project_name_focus_in)

# show entry box when user selected to add a new project from dropdown menu
def swtich_dropdown_to_entry(self):
    # set vars
    global var_project_name
    global ent_project_name

    # remove all project name widgets
    if self in new_project_txt: # new project
        project_name_widgets = [*req_params.grid_slaves(row_project_name, 0), *req_params.grid_slaves(row_project_name, 1)]
        for widget in project_name_widgets:
            widget.grid_forget()

        # add entry widget, label and button
        lbl_project_name_txt = ["Project name", "Nombre del proyecto"]
        lbl_project_name = tk.Label(req_params, text=lbl_project_name_txt[lang], pady=2, width=1, anchor="w")
        lbl_project_name.grid(row=row_project_name, sticky='nesw')
        var_project_name.set("")
        ent_project_name.grid(row=row_project_name, column=1, sticky='nesw', padx=5)
        ent_project_name.configure(fg="black")
        btn_project_name = Button(req_params, text="x", command=grid_project_name)
        btn_project_name.grid(row=row_project_name, column=0, sticky='e', padx=5)

# create hyperlinks (thanks marvin from GitHub) 
class HyperlinkManager:
    def __init__(self, text):
        self.text = text
        self.text.tag_config("hyper", foreground="blue", underline=1)
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
        self.text.config(cursor="hand2")

    def _leave(self, event):
        self.text.config(cursor="")

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
            child.config(state=DISABLED)

# set all children of frame to normal state
def enable_widgets(frame):
    children = frame.winfo_children()
    for child in children:
        # labelframes have no state
        if child.winfo_class() != "Labelframe":
            child.config(state=NORMAL)

# toggle options to resume from existing training
def toggle_train_type(self):
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    # change settings
    if self == dpd_train_type_options[lang][0]:
        # start new training
        for child in req_params.winfo_children():
            child.config(state=NORMAL)
        for child in adv_params.winfo_children():
            child.config(state=NORMAL)
        lbl_resume_checkpoint.grid_forget()
        btn_resume_checkpoint.grid_forget()
        dsp_resume_checkpoint.grid_forget()
    elif self == dpd_train_type_options[lang][1]:
        # resume existing training
        disable_widgets_based_on_location(req_params,
                                          rows = [1, 2, 3, 4, 5, 7],
                                          cols = [0, 1])
        disable_widgets(adv_params)
        lbl_resume_checkpoint.grid(row=row_resume_checkpoint, sticky='nesw', pady=2)
        lbl_resume_checkpoint.config(state=NORMAL)
        btn_resume_checkpoint.grid(row=row_resume_checkpoint, column=1, sticky='nesw', padx=5)
        btn_resume_checkpoint.config(state=NORMAL)
        dsp_resume_checkpoint.config(state=NORMAL)
        var_resume_checkpoint_path.set("")

# show warning for absolute paths option
shown_abs_paths_warning = True
def abs_paths_warning():
    global shown_abs_paths_warning
    if var_abs_paths.get() and shown_abs_paths_warning:
        mb.showinfo(warning_txt[lang], ["It is not recommended to use absolute paths in the output file. Third party software (such "
                    "as Timelapse, Agouti etc.) will not be able to read the json file if the paths are absolute. Only enable"
                    " this option if you know what you are doing.",
                    "No se recomienda utilizar rutas absolutas en el archivo de salida. Software de terceros (como Timelapse, "
                    "Agouti etc.) no podrán leer el archivo json si las rutas son absolutas. Sólo active esta opción si sabe lo"
                    " que está haciendo."][lang])
        shown_abs_paths_warning = False

# place animal classifier widgets
def place_cls_animal_options():    
    lbl_cls_animal_thresh.grid(row=row_cls_animal_thresh, sticky='nesw', pady=2)
    scl_cls_animal_thresh.grid(row=row_cls_animal_thresh, column=1, sticky='ew', padx=10)
    dsp_cls_animal_thresh.grid(row=row_cls_animal_thresh, column=0, sticky='e', padx=0)
    lbl_smooth_cls_animal.grid(row=row_smooth_cls_animal, sticky='nesw', pady=2)
    chb_smooth_cls_animal.grid(row=row_smooth_cls_animal, column=1, sticky='nesw', padx=5)
    cls_frame.grid_rowconfigure(row_cls_animal_thresh, minsize=minsize_rows)
    cls_frame.grid_rowconfigure(row_smooth_cls_animal, minsize=minsize_rows)

# remove animal classifier widgets
def remove_cls_animal_options():
    lbl_cls_animal_thresh.grid_remove()
    scl_cls_animal_thresh.grid_remove()
    dsp_cls_animal_thresh.grid_remove()
    cls_frame.grid_rowconfigure(row_cls_animal_thresh, minsize=0)
    lbl_smooth_cls_animal.grid_remove()
    chb_smooth_cls_animal.grid_remove()
    cls_frame.grid_rowconfigure(row_smooth_cls_animal, minsize=0)

# place vehicle classifier widgets
def place_cls_vehicle_options():    
    lbl_cls_vehicle_thresh.grid(row=row_cls_vehicle_thresh, sticky='nesw', pady=2)
    scl_cls_vehicle_thresh.grid(row=row_cls_vehicle_thresh, column=1, sticky='ew', padx=10)
    dsp_cls_vehicle_thresh.grid(row=row_cls_vehicle_thresh, column=0, sticky='e', padx=0)
    cls_frame.grid_rowconfigure(row_cls_vehicle_thresh, minsize=minsize_rows)

# remove vehicle classifier widgets
def remove_cls_vehicle_options():
    lbl_cls_vehicle_thresh.grid_remove()
    scl_cls_vehicle_thresh.grid_remove()
    dsp_cls_vehicle_thresh.grid_remove()
    cls_frame.grid_rowconfigure(row_cls_vehicle_thresh, minsize=0)

# place person classifier widgets
def place_cls_person_options():    
    lbl_cls_person_thresh.grid(row=row_cls_person_thresh, sticky='nesw', pady=2)
    scl_cls_person_thresh.grid(row=row_cls_person_thresh, column=1, sticky='ew', padx=10)
    dsp_cls_person_thresh.grid(row=row_cls_person_thresh, column=0, sticky='e', padx=0)
    cls_frame.grid_rowconfigure(row_cls_person_thresh, minsize=minsize_rows)

# remove person classifier widgets
def remove_cls_person_options():
    lbl_cls_person_thresh.grid_remove()
    scl_cls_person_thresh.grid_remove()
    dsp_cls_person_thresh.grid_remove()
    cls_frame.grid_rowconfigure(row_cls_person_thresh, minsize=0)

# toggle image size entry box
def toggle_image_size_for_deploy():
    if var_use_custom_img_size_for_deploy.get():
        lbl_image_size_for_deploy.grid(row=row_image_size_for_deploy, sticky='nesw', pady=2)
        ent_image_size_for_deploy.grid(row=row_image_size_for_deploy, column=1, sticky='nesw', padx=5)
    else:
        lbl_image_size_for_deploy.grid_remove()
        ent_image_size_for_deploy.grid_remove()

# toggle separation subframe
def toggle_sep_frame():
    if var_separate_files.get():
        sep_frame.grid(row=sep_frame_row, column=0, columnspan=2, sticky = 'ew')
        enable_widgets(sep_frame)
        sep_frame.configure(fg='black')
    else:
        disable_widgets(sep_frame)
        sep_frame.configure(fg='grey80')
        sep_frame.grid_forget()

# toggle classification subframe
def toggle_cls_frame(): 
    # log
    print(f"EXECUTED: {sys._getframe().f_code.co_name}({locals()})\n")

    if var_cls_further.get():
        cls_frame.grid(row=cls_frame_row, column=0, columnspan=2, sticky = 'ew')
        enable_widgets(cls_frame)
        toggle_checkpoint_freq()
        cls_frame.configure(fg='black')
    else:
        disable_widgets(cls_frame)
        cls_frame.configure(fg='grey80')
        cls_frame.grid_forget()

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

# convert frame to completed
def complete_frame(frame):
    global check_mark_one_row
    global check_mark_two_rows

    # check which frame 
    any_step = frame.cget('text').startswith(f' {step_txt[lang]}')
    fst_step = frame.cget('text').startswith(f' {step_txt[lang]} 1')
    snd_step = frame.cget('text').startswith(f' {step_txt[lang]} 2')
    trd_step = frame.cget('text').startswith(f' {step_txt[lang]} 3')
    fth_step = frame.cget('text').startswith(f' {step_txt[lang]} 4')

    # adjust frames
    frame.configure(relief = 'groove')
    if any_step:
        frame.configure(fg='green3')
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
            btn_hitl_main.config(text=["New session?", "¿Nueva sesión?"][lang], state = NORMAL)
            btn_hitl_main.lift()
        if fst_step:
            btn_choose_folder.config(text=f"{change_folder_txt[lang]}?", state = NORMAL)
            btn_choose_folder.lift()
            dsp_choose_folder.lift()
    
    else:
        # the rest
        if not any_step:
            # sub frames of fth_step only
            frame.configure(fg='green3')

        # add check mark
        lbl_check_mark = Label(frame, image=check_mark_two_rows)
        lbl_check_mark.image = check_mark_two_rows
        lbl_check_mark.grid(row=0, column=0, rowspan=15, columnspan=2, sticky='nesw')

        # add buttons
        btn_view_results = Button(master=frame, text=view_results_txt[lang], width=1, command=lambda: view_results(frame))
        btn_view_results.grid(row=0, column=1, sticky='nesw', padx = 5)
        btn_uncomplete = Button(master=frame, text=again_txt[lang], width=1, command=lambda: enable_frame(frame))
        btn_uncomplete.grid(row=1, column=1, sticky='nesw', padx = 5)

# enable a frame
def enable_frame(frame):
    uncomplete_frame(frame)
    enable_widgets(frame)

    # check which frame 
    any_step = frame.cget('text').startswith(f' {step_txt[lang]}')
    fst_step = frame.cget('text').startswith(f' {step_txt[lang]} 1')
    snd_step = frame.cget('text').startswith(f' {step_txt[lang]} 2')
    trd_step = frame.cget('text').startswith(f' {step_txt[lang]} 3')
    fth_step = frame.cget('text').startswith(f' {step_txt[lang]} 4')

    # all frames
    frame.configure(relief = 'solid')
    if any_step:
        frame.configure(fg='darkblue')
    if snd_step:
        toggle_cls_frame()
        cls_frame.configure(relief = 'solid')
        toggle_img_frame()
        img_frame.configure(relief = 'solid')
        toggle_vid_frame()
        vid_frame.configure(relief = 'solid')
    if fth_step:
        toggle_sep_frame()
        sep_frame.configure(relief = 'solid')

# remove checkmarks and complete buttons
def uncomplete_frame(frame):
    if not frame.cget('text').startswith(f' {step_txt[lang]}'):
        # subframes in fth_step only
        frame.configure(fg='black')
    children = frame.winfo_children()
    for child in children:
        if child.winfo_class() == "Button" or child.winfo_class() == "Label":
            if child.cget('text') == again_txt[lang] or child.cget('text') == view_results_txt[lang] or child.cget('image') != "":
                child.grid_remove()

# disable a frame
def disable_frame(frame):
    uncomplete_frame(frame)
    disable_widgets(frame)
    # all frames
    frame.configure(fg='grey80')
    frame.configure(relief = 'flat')
    if frame.cget('text').startswith(f' {step_txt[lang]} 2'):
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
    if frame.cget('text').startswith(f' {step_txt[lang]} 4'):
        # fth_step only
        disable_widgets(sep_frame)
        sep_frame.configure(fg='grey80')
        sep_frame.configure(relief = 'flat')
    
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
        lbl_checkpoint_freq.config(state=NORMAL)
        ent_checkpoint_freq.config(state=NORMAL)
    else:
        lbl_checkpoint_freq.config(state=DISABLED)
        ent_checkpoint_freq.config(state=DISABLED)

# toggle state of label placement method
def toggle_label_placement():
    if var_uniquify.get():
        lbl_label_placement.config(state=NORMAL)
        rad_label_placement_move.config(state=NORMAL)
        rad_label_placement_copy.config(state=NORMAL)
    else:
        lbl_label_placement.config(state=DISABLED)
        rad_label_placement_move.config(state=DISABLED)
        rad_label_placement_copy.config(state=DISABLED)

# toggle state of nth frame
def toggle_nth_frame():
    if var_not_all_frames.get():
        lbl_nth_frame.config(state=NORMAL)
        ent_nth_frame.config(state=NORMAL)
    else:
        lbl_nth_frame.config(state=DISABLED)
        ent_nth_frame.config(state=DISABLED)

# toggle hyperparameter evolution
def toggle_n_evolutions():
    mb.showwarning(warning_txt[lang], ["Note that evolution is generally expensive and time consuming, as the base scenario is trained hundreds of times."
                              " Be aware that it can take weeks or months to finish.",
                              "Tenga en cuenta que la evolución es generalmente costosa y requiere mucho tiempo, ya que el escenario base se entrena "
                              "cientos de veces. Tenga en cuenta que puede tardar semanas o meses en terminarse."][lang])
    if var_evolve.get():
        lbl_n_generations.grid(row=row_n_generations, sticky='nesw')
        ent_n_generations.grid(row=row_n_generations, column=1, sticky='nesw', padx=5)
        adv_params.grid_rowconfigure(row_n_generations, minsize=minsize_rows)
    else:
        lbl_n_generations.grid_forget()
        ent_n_generations.grid_forget()
        adv_params.grid_rowconfigure(row_n_generations, minsize=0)

# functions to delete the grey text in the entry boxes for the...
# ... image size fro deploy
image_size_for_deploy_init = True
def image_size_for_deploy_focus_in(_):
    global image_size_for_deploy_init
    if image_size_for_deploy_init:
        ent_image_size_for_deploy.delete(0, tk.END)
        ent_image_size_for_deploy.config(fg='black')
    image_size_for_deploy_init = False

# ... checkpoint frequency
checkpoint_freq_init = True
def checkpoint_freq_focus_in(_):
    global checkpoint_freq_init
    if checkpoint_freq_init:
        ent_checkpoint_freq.delete(0, tk.END)
        ent_checkpoint_freq.config(fg='black')
    checkpoint_freq_init = False

# ... nth frame
nth_frame_init = True
def nth_frame_focus_in(_):
    global nth_frame_init
    if nth_frame_init:
        ent_nth_frame.delete(0, tk.END)
        ent_nth_frame.config(fg='black')
    nth_frame_init = False

# ... project name
project_name_init = True
def project_name_focus_in(_):
    global project_name_init
    if project_name_init:
        ent_project_name.delete(0, tk.END)
        ent_project_name.config(fg='black')
    project_name_init = False

# ... run name
run_name_init = True
def run_name_focus_in(_):
    global run_name_init
    if run_name_init:
        ent_run_name.delete(0, tk.END)
        ent_run_name.config(fg='black')
    run_name_init = False

# ... number of epochs
n_epochs_init = True
def n_epochs_focus_in(_):
    global n_epochs_init
    if n_epochs_init:
        ent_n_epochs.delete(0, tk.END)
        ent_n_epochs.config(fg='black')
    n_epochs_init = False

# ... number of layers to freeze
n_freeze_layers_init = True
def n_freeze_layers_focus_in(_):
    global n_freeze_layers_init
    if n_freeze_layers_init:
        ent_n_freeze_layers.delete(0, tk.END)
        ent_n_freeze_layers.config(fg='black')
    n_freeze_layers_init = False

# ... batch size
batch_size_init = True
def batch_size_focus_in(_):
    global batch_size_init
    if batch_size_init:
        ent_batch_size.delete(0, tk.END)
        ent_batch_size.config(fg='black')
    batch_size_init = False

# ... number of generations
n_generations_init = True
def n_generations_focus_in(_):
    global n_generations_init
    if n_generations_init:
        ent_n_generations.delete(0, tk.END)
        ent_n_generations.config(fg='black')
    n_generations_init = False

# ... image size
image_size_for_training_init = True
def image_size_for_training_focus_in(_):
    global image_size_for_training_init
    if image_size_for_training_init:
        ent_image_size_for_training.delete(0, tk.END)
        ent_image_size_for_training.config(fg='black')
    image_size_for_training_init = False

# ... annotation classes
annot_classes_init = True
def annot_classes_focus_in(_):
    global annot_classes_init
    if annot_classes_init:
        ent_annot_classes.delete(0, tk.END)
        ent_annot_classes.config(fg='black')
    annot_classes_init = False
    
# ... n dataloader workers
n_workers_init = True
def n_workers_focus_in(_):
    global n_workers_init
    if n_workers_init:
        ent_n_workers.delete(0, tk.END)
        ent_n_workers.config(fg='black')
    n_workers_init = False

##########################################
############# TKINTER WINDOW #############
##########################################

# make it look similar on different systems
if os.name == "nt": # windows
    text_font = "TkDefaultFont"
    resize_img_factor = 0.95
    text_size_adjustment_factor = 0.83
    first_level_frame_font_size = 13
    second_level_frame_font_size = 11
    label_width = 320
    widget_width = 150
    frame_width = label_width + widget_width + 50
    minsize_rows = 28
    explanation_text_box_height_factor = 0.8
elif sys.platform == "linux" or sys.platform == "linux2": # linux
    text_font = "Times"
    resize_img_factor = 1
    text_size_adjustment_factor = 0.7
    first_level_frame_font_size = 13
    second_level_frame_font_size = 10
    label_width = 330
    widget_width = 160
    frame_width = label_width + widget_width + 50
    minsize_rows = 28
    explanation_text_box_height_factor = 1
else: # macOS
    text_font = "TkDefaultFont"
    resize_img_factor = 1
    text_size_adjustment_factor = 1
    first_level_frame_font_size = 15
    second_level_frame_font_size = 13
    label_width = 320
    widget_width = 150
    frame_width = label_width + widget_width + 50
    minsize_rows = 28
    explanation_text_box_height_factor = 1

# tkinter main window
root = Tk()
root.title(f"EcoAssist v{version}")
root.geometry()
root.configure(background="white")
tabControl = ttk.Notebook(root)

# prepare logo
logo_path = os.path.join(EcoAssist_files,'EcoAssist', 'imgs', 'logo.png')
logo = Image.open(logo_path)
logo = logo.resize((200, 90))
white_bg_logo = Image.new("RGBA", logo.size, "WHITE")
white_bg_logo.paste(logo, (0, 0), logo)
white_bg_logo.convert('RGB')
white_bg_logo = ImageTk.PhotoImage(white_bg_logo)
grey_bg_logo = ImageTk.PhotoImage(logo)

# prepare fox image
fox = Image.open(os.path.join(EcoAssist_files, 'EcoAssist', 'imgs', 'fox.jpg'))
full_width, full_height = fox.size
chosen_width = full_width
chosen_height = full_width * 0.2
top = 700
bottom = top + chosen_height
left = 0
right = chosen_width
fox = fox.crop((left, top, right, bottom))
fox = fox.resize((int(resize_img_factor * 422), 84), Image.Resampling.LANCZOS)
rad = 10
back = Image.new('RGB', (fox.size[0] + rad, fox.size[1]), (255, 255, 255))
back.paste(fox, (0, 0))
mask = Image.new('L', (fox.size[0] + rad, fox.size[1]), 255)
blck = Image.new('L', (fox.size[0] - rad, fox.size[1]), 0)
mask.paste(blck, (0, 0))
blur = back.filter(ImageFilter.GaussianBlur(rad / 2))
back.paste(blur, mask=mask)
fox = ImageTk.PhotoImage(back)

# prepare ocelot image
ocelot = Image.open(os.path.join(EcoAssist_files, 'EcoAssist', 'imgs', 'ocelot.jpg'))
full_width, full_height = ocelot.size
chosen_width = full_width
chosen_height = full_width * 0.2
top = 310
bottom = top + chosen_height
left = 0
right = chosen_width
ocelot = ocelot.crop((left, top, right, bottom))
ocelot = ocelot.resize((int(resize_img_factor * 422), 84), Image.Resampling.LANCZOS) 
back = Image.new('RGB', (ocelot.size[0], ocelot.size[1]), (255, 255, 255))
back.paste(ocelot, (rad, 0))
mask = Image.new('L', (ocelot.size[0], ocelot.size[1]), 255)
blck = Image.new('L', (ocelot.size[0], ocelot.size[1]), 0)
mask.paste(blck, (2 * rad, 0))
blur = back.filter(ImageFilter.GaussianBlur(rad / 2))
back.paste(blur, mask=mask)
ocelot = ImageTk.PhotoImage(back)

# print the images on the tkinter window
logo_widget = tk.Label(root, image=white_bg_logo, bg="white", highlightthickness=0, highlightbackground="white")
fox_widget = tk.Label(root, image=fox, bg="white", highlightthickness=0, highlightbackground="white")
ocelot_widget = tk.Label(root, image=ocelot, bg="white", highlightthickness=0, highlightbackground="white")
logo_widget.grid(column=0, row=0, sticky='ns', pady=(3, 0), padx=(0, 0))
fox_widget.grid(column=0, row=0, sticky='wns', pady=(3, 0), padx=(3, 0))
ocelot_widget.grid(column=0, row=0, sticky='ens', pady=(3, 0), padx=(0, 3))

# prepare check mark for later use
check_mark = Image.open(os.path.join(EcoAssist_files, 'EcoAssist', 'imgs', 'check_mark.png'))
check_mark_one_row = check_mark.resize((20, 20), Image.Resampling.LANCZOS)
check_mark_one_row = ImageTk.PhotoImage(check_mark_one_row)
check_mark_two_rows = check_mark.resize((45, 45), Image.Resampling.LANCZOS)
check_mark_two_rows = ImageTk.PhotoImage(check_mark_two_rows)

# english flag button
gb_flag = Image.open(os.path.join(EcoAssist_files, 'EcoAssist', 'imgs', 'flags', 'gb.png'))
gb_flag = gb_flag.resize((30, 20), Image.Resampling.LANCZOS)
gb_flag = ImageTk.PhotoImage(gb_flag)
gb_widget = tk.Button(root, image=gb_flag, bg="white", highlightthickness=1, highlightbackground="black", relief="sunken", command=lambda: set_language("gb"))
gb_widget.grid(column=0, row=1, sticky='e', pady=(0, 2), padx=(3, 5))

# spanish flag button
es_flag = Image.open(os.path.join(EcoAssist_files, 'EcoAssist', 'imgs', 'flags', 'es.png'))
es_flag = es_flag.resize((30, 20), Image.Resampling.LANCZOS)
es_flag = ImageTk.PhotoImage(es_flag)
es_widget = tk.Button(root, image=es_flag, bg="white", highlightthickness=1, highlightbackground="white", relief="raised", command=lambda: set_language("es"))
es_widget.grid(column=0, row=1, sticky='e', pady=(0, 2), padx=(3, 43))

# link to addax 
lbl_addax_txt = ['Need a model that can identify your target species? See Addax Data Science.',
                 '¿Necesita un modelo que pueda identificar sus especies objetivo? Consulte Addax Data Science.']
lbl_addax = Label(master=root, text=lbl_addax_txt[lang], anchor="w", bg="white", cursor= "hand2", fg="darkblue", font=(text_font, second_level_frame_font_size, "underline"))
lbl_addax.grid(row=1, sticky='ns', pady=2, padx=3)
lbl_addax.bind("<Button-1>", lambda e:webbrowser.open_new_tab("https://addaxdatascience.com/"))

# deploy tab
deploy_tab = ttk.Frame(tabControl)
deploy_tab.columnconfigure(0, weight=1, minsize=frame_width)
deploy_tab.columnconfigure(1, weight=1, minsize=frame_width)
deploy_tab_text = ['Deploy', 'Despliegue']
tabControl.add(deploy_tab, text=deploy_tab_text[lang])
deploy_canvas = tk.Canvas(deploy_tab)
deploy_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
deploy_y_scrollbar = ttk.Scrollbar(deploy_tab, orient=tk.VERTICAL, command=deploy_canvas.yview)
deploy_y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
deploy_canvas.configure(yscrollcommand=deploy_y_scrollbar.set)
deploy_scrollable_frame = ttk.Frame(deploy_canvas)
deploy_canvas.create_window((0, 0), window=deploy_scrollable_frame, anchor="nw")
deploy_scrollable_frame.bind("<Configure>", lambda event: deploy_canvas.configure(scrollregion=deploy_canvas.bbox("all")))

# train tab
train_tab = ttk.Frame(tabControl)
train_tab.columnconfigure(0, weight=1, minsize=frame_width)
train_tab.columnconfigure(1, weight=1, minsize=frame_width)
train_tab_text = ['Train', 'Entrenamiento']
tabControl.add(train_tab, text=train_tab_text[lang])

# help tab
help_tab = ttk.Frame(tabControl)
help_tab_text = ['Help', 'Ayuda']
tabControl.add(help_tab, text=help_tab_text[lang])

# about tab
about_tab = ttk.Frame(tabControl)
about_tab_text = ['About', 'Acerca de']
tabControl.add(about_tab, text=about_tab_text[lang])

# grid
tabControl.grid()

#### deploy tab
### first step
fst_step_txt = ['Step 1: Select folder', 'Paso 1: Seleccione carpeta']
row_fst_step = 1
fst_step = LabelFrame(deploy_scrollable_frame, text=" " + fst_step_txt[lang] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue', borderwidth=2)
fst_step.configure(font=(text_font, first_level_frame_font_size, "bold"))
fst_step.grid(column=0, row=row_fst_step, columnspan=1, sticky='ew')
fst_step.columnconfigure(0, weight=1, minsize=label_width)
fst_step.columnconfigure(1, weight=1, minsize=widget_width)

# choose folder
lbl_choose_folder_txt = ["Source folder", "Carpeta de origen"]
row_choose_folder = 0
lbl_choose_folder = Label(master=fst_step, text=lbl_choose_folder_txt[lang], width=1, anchor="w")
lbl_choose_folder.grid(row=row_choose_folder, sticky='nesw', pady=2)
var_choose_folder = StringVar()
var_choose_folder.set("")
var_choose_folder_short = StringVar()
dsp_choose_folder = Label(master=fst_step, textvariable=var_choose_folder_short, fg='grey', padx = 5)
btn_choose_folder = Button(master=fst_step, text=browse_txt[lang], width=1, command=lambda: [browse_dir(var_choose_folder, var_choose_folder_short, dsp_choose_folder, 25, row_choose_folder, 0, 'w'), update_frame_states()])
btn_choose_folder.grid(row=row_choose_folder, column=1, sticky='nesw', padx=5)

### second step
snd_step_txt = ['Step 2: Analysis', 'Paso 2: Análisis']
row_snd_step = 2
snd_step = LabelFrame(deploy_scrollable_frame, text=" " + snd_step_txt[lang] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue', borderwidth=2)
snd_step.configure(font=(text_font, first_level_frame_font_size, "bold"))
snd_step.grid(column=0, row=row_snd_step, sticky='nesw')
snd_step.columnconfigure(0, weight=1, minsize=label_width)
snd_step.columnconfigure(1, weight=1, minsize=widget_width)

# choose detector
lbl_model_txt = ['Model to detect animals, vehicles, and persons', 'Modelo para detectar animales, vehículos y personas']
row_model = 0
lbl_model = Label(master=snd_step, text=lbl_model_txt[lang], width=1, anchor="w")
lbl_model.grid(row=row_model, sticky='nesw', pady=2)
dpd_options_model = [["MegaDetector 5a", "MegaDetector 5b", "Custom model"], ["MegaDetector 5a", "MegaDetector 5b", "Otro modelo"]]
var_model = StringVar(snd_step)
var_model.set(dpd_options_model[lang][0])
var_model_short = StringVar()
var_model_path = StringVar()
dpd_model = OptionMenu(snd_step, var_model, *dpd_options_model[lang], command=model_options)
dpd_model.configure(width=1)
dpd_model.grid(row=row_model, column=1, sticky='nesw', padx=5)
dsp_model = Label(master=snd_step, textvariable=var_model_short, fg='darkred')

# check if user has classifiers installed
cls_params = []
for elem in ["cls_animals", "cls_vehicles", "cls_persons"]:
    cls_dir = os.path.join(EcoAssist_files, "classification_models", elem)
    cls_models = [f for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir, f)) and f.endswith('.pt')]
    cls_model_present = True if cls_models != [] else False
    dpd_options_cls_model = [cls_models + ["None"], cls_models + ["Ninguno"]]
    cls_params.extend([cls_models, cls_model_present, dpd_options_cls_model])
cls_animal_models, cls_animal_model_present, dpd_options_cls_animal_model, \
    cls_vehicle_models, cls_vehicle_model_present, dpd_options_cls_vehicle_model, \
    cls_person_models, cls_person_model_present, dpd_options_cls_person_model = cls_params
any_cls_model_present = True if any([cls_animal_model_present, cls_vehicle_model_present, cls_person_model_present]) else False

# use classifier
lbl_cls_further_txt = ["Further classify detections", "Seguir clasificando las detecciones"]
row_cls_further = 1
lbl_cls_further = Label(snd_step, text=lbl_cls_further_txt[lang], width=1, anchor="w")
lbl_cls_further.grid(row=row_cls_further, sticky='nesw', pady=2)
var_cls_further = BooleanVar()
# var_cls_further is set at the frame below
chb_cls_further = Checkbutton(snd_step, variable=var_cls_further, command=toggle_cls_frame, anchor="w")
chb_cls_further.grid(row=row_cls_further, column=1, sticky='nesw', padx=5)

## classification option frame (hidden by default)
cls_frame_txt = ["Classification options", "Opciones de clasificación"]
cls_frame_row = 2
cls_frame = LabelFrame(snd_step, text=" ↳ " + cls_frame_txt[lang] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")
cls_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
cls_frame.grid(row=cls_frame_row, column=0, columnspan=2, sticky = 'ew')
cls_frame.columnconfigure(0, weight=1, minsize=label_width)
cls_frame.columnconfigure(1, weight=1, minsize=widget_width)
if any_cls_model_present:
    var_cls_further.set(True)
else:
    var_cls_further.set(False)
    cls_frame.grid_forget()

# choose classifier for animals
lbl_cls_animal_model_txt = ['Model to classify animals', 'Modelo para clasificar animales']
row_cls_animal_model = 0
lbl_cls_animal_model = Label(master=cls_frame, text="     " + lbl_cls_animal_model_txt[lang], width=1, anchor="w", state=DISABLED)
lbl_cls_animal_model.grid(row=row_cls_animal_model, sticky='nesw', pady=2)
var_cls_animal_model = StringVar(cls_frame)
var_cls_animal_model.set(dpd_options_cls_animal_model[lang][0])
dpd_cls_animal_model = OptionMenu(cls_frame, var_cls_animal_model, *dpd_options_cls_animal_model[lang], command=model_cls_animal_options)
dpd_cls_animal_model.configure(width=1, state=DISABLED)
dpd_cls_animal_model.grid(row=row_cls_animal_model, column=1, sticky='nesw', padx=5)

# threshold for animal classifications
lbl_cls_animal_thresh_txt = ["Confidence threshold", "Umbral de confianza"]
row_cls_animal_thresh = 1
lbl_cls_animal_thresh = Label(cls_frame, text="        ↳ " + lbl_cls_animal_thresh_txt[lang], width=1, anchor="w", state=DISABLED)
lbl_cls_animal_thresh.grid(row=row_cls_animal_thresh, sticky='nesw', pady=2)
var_cls_animal_thresh = DoubleVar()
var_cls_animal_thresh.set(0.6)
scl_cls_animal_thresh = Scale(cls_frame, from_=0.01, to=1, resolution=0.01, orient=HORIZONTAL, variable=var_cls_animal_thresh, showvalue=0, width=10, length=1, state=DISABLED)
scl_cls_animal_thresh.grid(row=row_cls_animal_thresh, column=1, sticky='ew', padx=10)
dsp_cls_animal_thresh = Label(cls_frame, textvariable=var_cls_animal_thresh)
dsp_cls_animal_thresh.grid(row=row_cls_animal_thresh, column=0, sticky='e', padx=0)
dsp_cls_animal_thresh.config(fg="darkred")

# include subdirectories
lbl_smooth_cls_animal_txt = ["Smoothen results", "Suavizar resultados"]
row_smooth_cls_animal = 2
lbl_smooth_cls_animal = Label(cls_frame, text="        ↳ " + lbl_smooth_cls_animal_txt[lang], width=1, anchor="w")
lbl_smooth_cls_animal.grid(row=row_smooth_cls_animal, sticky='nesw', pady=2)
var_smooth_cls_animal = BooleanVar()
var_smooth_cls_animal.set(True)
chb_smooth_cls_animal = Checkbutton(cls_frame, variable=var_smooth_cls_animal, anchor="w")
chb_smooth_cls_animal.grid(row=row_smooth_cls_animal, column=1, sticky='nesw', padx=5)
model_cls_animal_options(dpd_options_cls_animal_model[lang][0])

# choose classifier for vehicles
lbl_cls_vehicle_model_txt = ['Model to classify vehicles', 'Modelo para clasificar vehículos']
row_cls_vehicle_model = 3
lbl_cls_vehicle_model = Label(master=cls_frame, text="     " + lbl_cls_vehicle_model_txt[lang], width=1, anchor="w", state=DISABLED)
lbl_cls_vehicle_model.grid(row=row_cls_vehicle_model, sticky='nesw', pady=2)
var_cls_vehicle_model = StringVar(cls_frame)
var_cls_vehicle_model.set(dpd_options_cls_vehicle_model[lang][0])
dpd_cls_vehicle_model = OptionMenu(cls_frame, var_cls_vehicle_model, *dpd_options_cls_vehicle_model[lang], command=model_cls_vehicle_options)
dpd_cls_vehicle_model.configure(width=1, state=DISABLED)
dpd_cls_vehicle_model.grid(row=row_cls_vehicle_model, column=1, sticky='nesw', padx=5)

# threshold for vehicle classifications
lbl_cls_vehicle_thresh_txt = ["Confidence threshold", "Umbral de confianza"]
row_cls_vehicle_thresh = 4
lbl_cls_vehicle_thresh = Label(cls_frame, text="        ↳ " + lbl_cls_vehicle_thresh_txt[lang], width=1, anchor="w", state=DISABLED)
lbl_cls_vehicle_thresh.grid(row=row_cls_vehicle_thresh, sticky='nesw', pady=2)
var_cls_vehicle_thresh = DoubleVar()
var_cls_vehicle_thresh.set(0.6)
scl_cls_vehicle_thresh = Scale(cls_frame, from_=0.01, to=1, resolution=0.01, orient=HORIZONTAL, variable=var_cls_vehicle_thresh, showvalue=0, width=10, length=1, state=DISABLED)
scl_cls_vehicle_thresh.grid(row=row_cls_vehicle_thresh, column=1, sticky='ew', padx=10)
dsp_cls_vehicle_thresh = Label(cls_frame, textvariable=var_cls_vehicle_thresh)
dsp_cls_vehicle_thresh.grid(row=row_cls_vehicle_thresh, column=0, sticky='e', padx=0)
dsp_cls_vehicle_thresh.config(fg="darkred")
model_cls_vehicle_options(dpd_options_cls_vehicle_model[lang][0])

# choose classifier for person
lbl_cls_person_model_txt = ['Model to classify persons', 'Modelo para clasificar personas']
row_cls_person_model = 5
lbl_cls_person_model = Label(master=cls_frame, text="     " + lbl_cls_person_model_txt[lang], width=1, anchor="w", state=DISABLED)
lbl_cls_person_model.grid(row=row_cls_person_model, sticky='nesw', pady=2)
var_cls_person_model = StringVar(cls_frame)
var_cls_person_model.set(dpd_options_cls_person_model[lang][0])
dpd_cls_person_model = OptionMenu(cls_frame, var_cls_person_model, *dpd_options_cls_person_model[lang], command=model_cls_person_options)
dpd_cls_person_model.configure(width=1, state=DISABLED)
dpd_cls_person_model.grid(row=row_cls_person_model, column=1, sticky='nesw', padx=5)

# threshold for person classifications
lbl_cls_person_thresh_txt = ["Confidence threshold", "Umbral de confianza"]
row_cls_person_thresh = 6
lbl_cls_person_thresh = Label(cls_frame, text="        ↳ " + lbl_cls_person_thresh_txt[lang], width=1, anchor="w", state=DISABLED)
lbl_cls_person_thresh.grid(row=row_cls_person_thresh, sticky='nesw', pady=2)
var_cls_person_thresh = DoubleVar()
var_cls_person_thresh.set(0.6)
scl_cls_person_thresh = Scale(cls_frame, from_=0.01, to=1, resolution=0.01, orient=HORIZONTAL, variable=var_cls_person_thresh, showvalue=0, width=10, length=1, state=DISABLED)
scl_cls_person_thresh.grid(row=row_cls_person_thresh, column=1, sticky='ew', padx=10)
dsp_cls_person_thresh = Label(cls_frame, textvariable=var_cls_person_thresh)
dsp_cls_person_thresh.grid(row=row_cls_person_thresh, column=0, sticky='e', padx=0)
dsp_cls_person_thresh.config(fg="darkred")
model_cls_person_options(dpd_options_cls_person_model[lang][0])

# include subdirectories
lbl_exclude_subs_txt = ["Don't process subdirectories", "No procesar subcarpetas"]
row_exclude_subs = 3
lbl_exclude_subs = Label(snd_step, text=lbl_exclude_subs_txt[lang], width=1, anchor="w")
lbl_exclude_subs.grid(row=row_exclude_subs, sticky='nesw', pady=2)
var_exclude_subs = BooleanVar()
var_exclude_subs.set(False)
chb_exclude_subs = Checkbutton(snd_step, variable=var_exclude_subs, anchor="w")
chb_exclude_subs.grid(row=row_exclude_subs, column=1, sticky='nesw', padx=5)

# use custom image size
lbl_use_custom_img_size_for_deploy_txt = ["Use custom image size", "Usar tamaño de imagen personalizado"]
row_use_custom_img_size_for_deploy = 4
lbl_use_custom_img_size_for_deploy = Label(snd_step, text=lbl_use_custom_img_size_for_deploy_txt[lang], width=1, anchor="w")
lbl_use_custom_img_size_for_deploy.grid(row=row_use_custom_img_size_for_deploy, sticky='nesw', pady=2)
var_use_custom_img_size_for_deploy = BooleanVar()
var_use_custom_img_size_for_deploy.set(False)
chb_use_custom_img_size_for_deploy = Checkbutton(snd_step, variable=var_use_custom_img_size_for_deploy, command=toggle_image_size_for_deploy, anchor="w")
chb_use_custom_img_size_for_deploy.grid(row=row_use_custom_img_size_for_deploy, column=1, sticky='nesw', padx=5)

# specify custom image size (not grid by default)
lbl_image_size_for_deploy_txt = ["Image size", "Tamaño imagen"]
row_image_size_for_deploy = 5
lbl_image_size_for_deploy = Label(snd_step, text=" ↳ " + lbl_image_size_for_deploy_txt[lang], width=1, anchor="w")
var_image_size_for_deploy = StringVar()
ent_image_size_for_deploy = tk.Entry(snd_step, textvariable=var_image_size_for_deploy, fg='grey', state=NORMAL, width=1)
ent_image_size_for_deploy.insert(0, f"{eg_txt[lang]}: 640")
ent_image_size_for_deploy.bind("<FocusIn>", image_size_for_deploy_focus_in)
ent_image_size_for_deploy.config(state=DISABLED)

# use absolute paths
lbl_abs_paths_txt = ["Use absolute paths in output file", "Usar rutas absolutas en archivo de salida"]
row_abs_path = 6
lbl_abs_paths = Label(snd_step, text=lbl_abs_paths_txt[lang], width=1, anchor="w")
lbl_abs_paths.grid(row=row_abs_path, sticky='nesw', pady=2)
var_abs_paths = BooleanVar()
var_abs_paths.set(False)
chb_abs_paths = Checkbutton(snd_step, variable=var_abs_paths, command=abs_paths_warning, anchor="w")
chb_abs_paths.grid(row=row_abs_path, column=1, sticky='nesw', padx=5)

# process images
lbl_process_img_txt = ["Process all images in the folder specified", "Procesar todas las imágenes en carpeta elegida"]
row_process_img = 7
lbl_process_img = Label(snd_step, text=lbl_process_img_txt[lang], width=1, anchor="w")
lbl_process_img.grid(row=row_process_img, sticky='nesw', pady=2)
var_process_img = BooleanVar()
var_process_img.set(False)
chb_process_img = Checkbutton(snd_step, variable=var_process_img, command=toggle_img_frame, anchor="w")
chb_process_img.grid(row=row_process_img, column=1, sticky='nesw', padx=5)

## image option frame (hidden by default)
img_frame_txt = ["Image options", "Opciones de imagen"]
img_frame_row = 8
img_frame = LabelFrame(snd_step, text=" ↳ " + img_frame_txt[lang] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")
img_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
img_frame.grid(row=img_frame_row, column=0, columnspan=2, sticky = 'ew')
img_frame.columnconfigure(0, weight=1, minsize=label_width)
img_frame.columnconfigure(1, weight=1, minsize=widget_width)
img_frame.grid_forget()

# use checkpoints
lbl_use_checkpnts_txt = ["Use checkpoints while running", "Usar puntos de control mientras se ejecuta"]
row_use_checkpnts = 0
lbl_use_checkpnts = Label(img_frame, text="     " + lbl_use_checkpnts_txt[lang], pady=2, state=DISABLED, width=1, anchor="w")
lbl_use_checkpnts.grid(row=row_use_checkpnts, sticky='nesw')
var_use_checkpnts = BooleanVar()
var_use_checkpnts.set(False)
chb_use_checkpnts = Checkbutton(img_frame, variable=var_use_checkpnts, command=toggle_checkpoint_freq, state=DISABLED, anchor="w")
chb_use_checkpnts.grid(row=row_use_checkpnts, column=1, sticky='nesw', padx=5)

# checkpoint frequency
lbl_checkpoint_freq_txt = ["Checkpoint frequency", "Frecuencia puntos de control"]
row_checkpoint_freq = 1
lbl_checkpoint_freq = tk.Label(img_frame, text="        ↳ " + lbl_checkpoint_freq_txt[lang], pady=2, state=DISABLED, width=1, anchor="w")
lbl_checkpoint_freq.grid(row=row_checkpoint_freq, sticky='nesw')
var_checkpoint_freq = StringVar()
ent_checkpoint_freq = tk.Entry(img_frame, textvariable=var_checkpoint_freq, fg='grey', state=NORMAL, width=1)
ent_checkpoint_freq.grid(row=row_checkpoint_freq, column=1, sticky='nesw', padx=5)
ent_checkpoint_freq.insert(0, f"{eg_txt[lang]}: 500")
ent_checkpoint_freq.bind("<FocusIn>", checkpoint_freq_focus_in)
ent_checkpoint_freq.config(state=DISABLED)

# continue from checkpoint file
lbl_cont_checkpnt_txt = ["Continue from last checkpoint file", "Continuar desde el último punto de control"]
row_cont_checkpnt = 2
lbl_cont_checkpnt = Label(img_frame, text="     " + lbl_cont_checkpnt_txt[lang], pady=2, state=DISABLED, width=1, anchor="w")
lbl_cont_checkpnt.grid(row=row_cont_checkpnt, sticky='nesw')
var_cont_checkpnt = BooleanVar()
var_cont_checkpnt.set(False)
chb_cont_checkpnt = Checkbutton(img_frame, variable=var_cont_checkpnt, state=DISABLED, command=disable_chb_cont_checkpnt, anchor="w")
chb_cont_checkpnt.grid(row=row_cont_checkpnt, column=1, sticky='nesw', padx=5)

# process videos
lbl_process_vid_txt = ["Process all videos in the folder specified", "Procesar todos los vídeos en la carpeta elegida"]
row_process_vid = 9
lbl_process_vid = Label(snd_step, text=lbl_process_vid_txt[lang], width=1, anchor="w")
lbl_process_vid.grid(row=row_process_vid, sticky='nesw', pady=2)
var_process_vid = BooleanVar()
var_process_vid.set(False)
chb_process_vid = Checkbutton(snd_step, variable=var_process_vid, command=toggle_vid_frame, anchor="w")
chb_process_vid.grid(row=row_process_vid, column=1, sticky='nesw', padx=5)

## video option frame (disabled by default)
vid_frame_txt = ["Video options", "Opciones de vídeo"]
vid_frame_row = 10
vid_frame = LabelFrame(snd_step, text=" ↳ " + vid_frame_txt[lang] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")
vid_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
vid_frame.grid(row=vid_frame_row, column=0, columnspan=2, sticky='ew')
vid_frame.columnconfigure(0, weight=1, minsize=label_width)
vid_frame.columnconfigure(1, weight=1, minsize=widget_width)
vid_frame.grid_forget()

# dont process all frames
lbl_not_all_frames_txt = ["Don't process every frame", "No procesar cada fotograma"]
row_not_all_frames = 0
lbl_not_all_frames = Label(vid_frame, text="     " + lbl_not_all_frames_txt[lang], pady=2, state=DISABLED, width=1, anchor="w")
lbl_not_all_frames.grid(row=row_not_all_frames, sticky='nesw')
var_not_all_frames = BooleanVar()
var_not_all_frames.set(False)
chb_not_all_frames = Checkbutton(vid_frame, variable=var_not_all_frames, command=toggle_nth_frame, state=DISABLED, anchor="w")
chb_not_all_frames.grid(row=row_not_all_frames, column=1, sticky='nesw', padx=5)

# process every nth frame
lbl_nth_frame_txt = ["Analyse every Nth frame", "Analizar cada Nº fotograma"]
row_nth_frame = 1
lbl_nth_frame = tk.Label(vid_frame, text="        ↳ " + lbl_nth_frame_txt[lang], pady=2, state=DISABLED, width=1, anchor="w")
lbl_nth_frame.grid(row=row_nth_frame, sticky='nesw')
var_nth_frame = StringVar()
ent_nth_frame = tk.Entry(vid_frame, textvariable=var_nth_frame, fg='grey', state=NORMAL, width=1)
ent_nth_frame.grid(row=row_nth_frame, column=1, sticky='nesw', padx=5)
ent_nth_frame.insert(0, f"{eg_txt[lang]}: 10")
ent_nth_frame.bind("<FocusIn>", nth_frame_focus_in)
ent_nth_frame.config(state=DISABLED)

# button start deploy
btn_start_deploy_txt = ["Deploy model", "Desplegar modelo"]
row_btn_start_deploy = 11
btn_start_deploy = Button(snd_step, text=btn_start_deploy_txt[lang], command=start_deploy)
btn_start_deploy.grid(row=row_btn_start_deploy, column=0, columnspan=2, sticky='ew')

### human-in-the-loop step
trd_step_txt = ["Step 3: Annotation (optional)", "Paso 3: Anotación (opcional)"]
trd_step_row = 1
trd_step = LabelFrame(deploy_scrollable_frame, text=" " + trd_step_txt[lang] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue', borderwidth=2)
trd_step.configure(font=(text_font, first_level_frame_font_size, "bold"))
trd_step.grid(column=1, row=trd_step_row, sticky='nesw')
trd_step.columnconfigure(0, weight=1, minsize=label_width)
trd_step.columnconfigure(1, weight=1, minsize=widget_width)

# human-in-the-loop 
lbl_hitl_main_txt = ["Manually verify results", "Verificar manualmente los resultados"]
row_hitl_main = 0
lbl_hitl_main = Label(master=trd_step, text=lbl_hitl_main_txt[lang], width=1, anchor="w")
lbl_hitl_main.grid(row=row_hitl_main, sticky='nesw', pady=2)
var_hitl_main = StringVar()
var_hitl_main.set("")
var_hitl_main_short = StringVar()
dsp_hitl_main = Label(master=trd_step, textvariable=var_hitl_main_short, fg='darkred')
btn_hitl_main = Button(master=trd_step, text=["Start", "Iniciar"][lang], width=1, command = start_or_continue_hitl)
btn_hitl_main.grid(row=row_hitl_main, column=1, sticky='nesw', padx=5)

### third step
fth_step_txt = ["Step 4: Post-processing (optional)", "Paso 4: Post-Procesado (opcional)"]
fth_step_row = 2
fth_step = LabelFrame(deploy_scrollable_frame, text=" " + fth_step_txt[lang] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue', borderwidth=2)
fth_step.configure(font=(text_font, first_level_frame_font_size, "bold"))
fth_step.grid(column=1, row=fth_step_row, sticky='nesw')
fth_step.columnconfigure(0, weight=1, minsize=label_width)
fth_step.columnconfigure(1, weight=1, minsize=widget_width)

# folder for results
lbl_output_dir_txt = ["Destination folder", "Carpeta de destino"]
row_output_dir = 0
lbl_output_dir = Label(master=fth_step, text=lbl_output_dir_txt[lang], width=1, anchor="w")
lbl_output_dir.grid(row=row_output_dir, sticky='nesw', pady=2)
var_output_dir = StringVar()
var_output_dir.set("")
var_output_dir_short = StringVar()
dsp_output_dir = Label(master=fth_step, textvariable=var_output_dir_short, fg='darkred')
btn_output_dir = Button(master=fth_step, text=browse_txt[lang], width=1, command=lambda: browse_dir(var_output_dir, var_output_dir_short, dsp_output_dir, 25, row_output_dir, 0, 'e'))
btn_output_dir.grid(row=row_output_dir, column=1, sticky='nesw', padx=5)

# separate files
lbl_separate_files_txt = ["Separate files into subdirectories", "Separar archivos en subcarpetas"]
row_separate_files = 1
lbl_separate_files = Label(fth_step, text=lbl_separate_files_txt[lang], width=1, anchor="w")
lbl_separate_files.grid(row=row_separate_files, sticky='nesw', pady=2)
var_separate_files = BooleanVar()
var_separate_files.set(False)
chb_separate_files = Checkbutton(fth_step, variable=var_separate_files, command=toggle_sep_frame, anchor="w")
chb_separate_files.grid(row=row_separate_files, column=1, sticky='nesw', padx=5)

## separation frame
sep_frame_txt = ["Separation options", "Opciones de separación"]
sep_frame_row = 2
sep_frame = LabelFrame(fth_step, text=" ↳ " + sep_frame_txt[lang] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, borderwidth=1, fg="grey80")
sep_frame.configure(font=(text_font, second_level_frame_font_size, "bold"))
sep_frame.grid(row=sep_frame_row, column=0, columnspan=2, sticky = 'ew')
sep_frame.columnconfigure(0, weight=1, minsize=label_width)
sep_frame.columnconfigure(1, weight=1, minsize=widget_width)
sep_frame.grid_forget()

# method of file placement
lbl_file_placement_txt = ["Method of file placement", "Método de desplazamiento de archivo"]
row_file_placement = 0
lbl_file_placement = Label(sep_frame, text="     " + lbl_file_placement_txt[lang], pady=2, width=1, anchor="w")
lbl_file_placement.grid(row=row_file_placement, sticky='nesw')
var_file_placement = IntVar()
var_file_placement.set(2)
rad_file_placement_move = Radiobutton(sep_frame, text=["Copy", "Copiar"][lang], variable=var_file_placement, value=2)
rad_file_placement_move.grid(row=row_file_placement, column=1, sticky='w', padx=5)
rad_file_placement_copy = Radiobutton(sep_frame, text=["Move", "Mover"][lang], variable=var_file_placement, value=1)
rad_file_placement_copy.grid(row=row_file_placement, column=1, sticky='e', padx=5)

# separate per confidence
lbl_sep_conf_txt = ["Sort results based on confidence", "Clasificar resultados basados en confianza"]
row_sep_conf = 1
lbl_sep_conf = Label(sep_frame, text="     " + lbl_sep_conf_txt[lang], width=1, anchor="w")
lbl_sep_conf.grid(row=row_sep_conf, sticky='nesw', pady=2)
var_sep_conf = BooleanVar()
var_sep_conf.set(False)
chb_sep_conf = Checkbutton(sep_frame, variable=var_sep_conf, anchor="w")
chb_sep_conf.grid(row=row_sep_conf, column=1, sticky='nesw', padx=5)

## visualize images
lbl_vis_files_txt = ["Draw bounding boxes and confidences", "Dibujar contornos y confianzas"]
row_vis_files = 3
lbl_vis_files = Label(fth_step, text=lbl_vis_files_txt[lang], width=1, anchor="w")
lbl_vis_files.grid(row=row_vis_files, sticky='nesw', pady=2)
var_vis_files = BooleanVar()
var_vis_files.set(False)
chb_vis_files = Checkbutton(fth_step, variable=var_vis_files, anchor="w")
chb_vis_files.grid(row=row_vis_files, column=1, sticky='nesw', padx=5)

## crop images
lbl_crp_files_txt = ["Crop detections", "Recortar detecciones"]
row_crp_files = 4
lbl_crp_files = Label(fth_step, text=lbl_crp_files_txt[lang], width=1, anchor="w")
lbl_crp_files.grid(row=row_crp_files, sticky='nesw', pady=2)
var_crp_files = BooleanVar()
var_crp_files.set(False)
chb_crp_files = Checkbutton(fth_step, variable=var_crp_files, anchor="w")
chb_crp_files.grid(row=row_crp_files, column=1, sticky='nesw', padx=5)

# create csv files
lbl_csv_txt = ["Export results to .csv and retrieve metadata", "Exportar a .csv y recuperar los metadatos"]
row_csv = 5
lbl_csv = Label(fth_step, text=lbl_csv_txt[lang], width=1, anchor="w")
lbl_csv.grid(row=row_csv, sticky='nesw', pady=2)
var_csv = BooleanVar()
var_csv.set(False)
chb_csv = Checkbutton(fth_step, variable=var_csv, anchor="w")
chb_csv.grid(row=row_csv, column=1, sticky='nesw', padx=5)

# threshold
lbl_thresh_txt = ["Confidence threshold", "Umbral de confianza"]
row_lbl_thresh = 6
lbl_thresh = Label(fth_step, text=lbl_thresh_txt[lang], width=1, anchor="w")
lbl_thresh.grid(row=row_lbl_thresh, sticky='nesw', pady=2)
var_thresh = DoubleVar()
var_thresh.set(0.2)
scl_thresh = Scale(fth_step, from_=0.01, to=1, resolution=0.01, orient=HORIZONTAL, variable=var_thresh, showvalue=0, width=10, length=1)
scl_thresh.grid(row=row_lbl_thresh, column=1, sticky='ew', padx=10)
dsp_thresh = Label(fth_step, textvariable=var_thresh)
dsp_thresh.config(fg="darkred")
dsp_thresh.grid(row=row_lbl_thresh, column=0, sticky='e', padx=0)

# postprocessing button
btn_start_postprocess_txt = ["Post-process files", "Post-procesar archivos"]
row_start_postprocess = 7
btn_start_postprocess = Button(fth_step, text=btn_start_postprocess_txt[lang], command=start_postprocess)
btn_start_postprocess.grid(row=row_start_postprocess, column=0, columnspan = 2, sticky='ew')

#### train tab
### required parameters
req_params_txt = ["Required parameters", "Parámetros requeridos"]
req_params_row = 1
req_params = LabelFrame(train_tab, text=" " + req_params_txt[lang] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue', borderwidth=2)
req_params.configure(font=(text_font, first_level_frame_font_size, "bold"))
req_params.grid(column=0, row=req_params_row, sticky='nesw')
req_params.columnconfigure(0, weight=1, minsize=label_width)
req_params.columnconfigure(1, weight=1, minsize=widget_width)

# train type
lbl_train_type_txt = ["Training type", "Tipo de entrenamiento"]
row_train_type = 0
lbl_train_type = Label(req_params, text=lbl_train_type_txt[lang], pady=2, width=1, anchor="w")
lbl_train_type.grid(row=row_train_type, sticky='nesw', pady=2)
dpd_train_type_options = [["Start new training", "Resume existing"], ["Comenzar nuevo", "Reanudar existente"]]
var_train_type = StringVar(req_params)
var_train_type.set(dpd_train_type_options[lang][0])
dpd_train_type = OptionMenu(req_params, var_train_type, *dpd_train_type_options[lang], command=toggle_train_type)
dpd_train_type.configure(width=1)
dpd_train_type.grid(row=row_train_type, column=1, sticky='nesw', padx=5)

# folder with annotated data
lbl_annotated_data_txt = ["Folder with labeled data", "Carpeta con datos etiquetados"]
row_annotated_data = 1
lbl_annotated_data = Label(master=req_params, text=lbl_annotated_data_txt[lang], width=1, anchor="w")
lbl_annotated_data.grid(row=row_annotated_data, sticky='nesw', pady=2)
var_annotated_data = StringVar()
var_annotated_data_short = StringVar()
dsp_annotated_data = Label(master=req_params, textvariable=var_annotated_data_short, fg='darkred')
btn_annotated_data = Button(master=req_params, text=browse_txt[lang], width=1, command=lambda: browse_dir(var_annotated_data, var_annotated_data_short, dsp_annotated_data, 25, row_annotated_data, 0, 'e'))
btn_annotated_data.grid(row=row_annotated_data, column=1, sticky='nesw', padx=5)

# transfer learning model
lbl_learning_model_txt = ["Retrain from", "Reentrenar desde"]
row_learning_model = 2
lbl_learning_model = Label(req_params, text=lbl_learning_model_txt[lang], pady=2, width=1, anchor="w")
lbl_learning_model.grid(row=row_learning_model, sticky='nesw', pady=2)
dpd_learning_model_options = [["MegaDetector 5a", "MegaDetector 5b", "YOLOv5 Nano", "YOLOv5 Small", "YOLOv5 Medium", "YOLOv5 Large", "YOLOv5 XLarge", "Custom model", "Scratch"], ["MegaDetector 5a", "MegaDetector 5b", "YOLOv5 Ínfimo", "YOLOv5 Pequeño", "YOLOv5 Medio", "YOLOv5 Grande", "YOLOv5 XL", "Otro modelo", "Desde cero"]]
var_learning_model = StringVar(req_params)
var_learning_model.set(dpd_learning_model_options[lang][0])
var_learning_model_short = StringVar()
var_learning_model_path = StringVar()
var_learning_model_path.set(os.path.join(EcoAssist_files, "pretrained_models", "md_v5a.0.0.pt"))
dpd_learning_model = OptionMenu(req_params, var_learning_model, *dpd_learning_model_options[lang], command=set_learning_model)
dpd_learning_model.configure(width=1)
dpd_learning_model.grid(row=row_learning_model, column=1, sticky='nesw', padx=5)
dsp_learning_model = Label(master=req_params, textvariable=var_learning_model_short, fg='darkred')

# model architecture
lbl_model_architecture_txt = ["Model architecture", "Arquitectura del modelo"]
row_model_architecture = 3
lbl_model_architecture = Label(req_params, text=lbl_model_architecture_txt[lang], pady=2, width=1, anchor="w")
dpd_model_architecture_options = [["YOLOv5 Nano", "YOLOv5 Small", "YOLOv5 Medium", "YOLOv5 Large", "YOLOv5 XLarge", "Other config", "None"], ["YOLOv5 Ínfimo", "YOLOv5 Pequeño", "YOLOv5 Medio", "YOLOv5 Grande", "YOLOv5 XL", "Otro archivo", "Ninguno"]]
var_model_architecture = StringVar(req_params)
var_model_architecture.set(dpd_model_architecture_options[lang][0])
var_model_architecture_short = StringVar()
var_model_architecture_path = StringVar()
var_model_architecture_path.set(os.path.join(EcoAssist_files, "yolov5", "models", "yolov5m.yaml"))
dpd_model_architecture = OptionMenu(req_params, var_model_architecture, *dpd_model_architecture_options[lang], command=set_model_architecture)
dpd_model_architecture.configure(width=1)
dsp_model_architecture = Label(master=req_params, textvariable=var_model_architecture_short, fg='darkred')

# number of epochs
lbl_n_epochs_txt = ["Number of epochs", "Número de épocas"]
row_n_epochs = 4
lbl_n_epochs = tk.Label(req_params, text=lbl_n_epochs_txt[lang], pady=2, width=1, anchor="w")
lbl_n_epochs.grid(row=row_n_epochs, sticky='nesw')
var_n_epochs = StringVar()
ent_n_epochs = tk.Entry(req_params, textvariable=var_n_epochs, fg='grey', width=1)
ent_n_epochs.grid(row=row_n_epochs, column=1, sticky='nesw', padx=5)
ent_n_epochs.insert(0, f"{eg_txt[lang]}: 300")
ent_n_epochs.bind("<FocusIn>", n_epochs_focus_in)

# folder for results
lbl_results_dir_txt = ["Destination folder", "Carpeta destino"]
row_results_dir = 5
lbl_results_dir = Label(master=req_params, text=lbl_results_dir_txt[lang], width=1, anchor="w")
lbl_results_dir.grid(row=row_results_dir, sticky='nesw', pady=2)
var_results_dir = StringVar()
var_results_dir_short = StringVar()
dsp_results_dir = Label(master=req_params, textvariable=var_results_dir_short, fg='darkred')
btn_results_dir = Button(master=req_params, text=browse_txt[lang], width=1, command=lambda: [browse_dir(var_results_dir, var_results_dir_short, dsp_results_dir, 25, row_results_dir, 0, 'e'), grid_project_name()])
btn_results_dir.grid(row=row_results_dir, column=1, sticky='nesw', padx=5)

# specify resume checkpoint
lbl_resume_checkpoint_txt = ["Specify resume checkpoint", "Especificar punto control reanudación"]
row_resume_checkpoint = 6
lbl_resume_checkpoint = Label(master=req_params, text=lbl_resume_checkpoint_txt[lang], width=1, anchor="w")
var_resume_checkpoint = StringVar()
var_resume_checkpoint_short = StringVar()
var_resume_checkpoint_path = StringVar()
var_resume_checkpoint_path.set("")
dsp_resume_checkpoint = Label(master=req_params, textvariable=var_resume_checkpoint_short, fg='darkred')
btn_resume_checkpoint = Button(master=req_params, text=browse_txt[lang], width=1, command=lambda: browse_file(var_resume_checkpoint, var_resume_checkpoint_short, var_resume_checkpoint_path, dsp_resume_checkpoint, [("Model file","*.pt")], 20, ["dummy"], row_resume_checkpoint))

# name of the project
row_project_name = 7
var_project_name = StringVar()
ent_project_name = tk.Entry(req_params, textvariable=var_project_name, fg='grey', width=1)
lbl_project_name_txt = ["Project name", "Nombre del proyecto"]
lbl_project_name = tk.Label(req_params, text=lbl_project_name_txt[lang], pady=2, width=1, anchor="w")
# the entry box, dropdown menu and button are created through grid_project_name() and swtich_dropdown_to_entry()

### advanced settings
adv_params_txt = ["Advanced settings (optional)", "Configuración avanzada (opcional)"]
adv_params_row = 2
adv_params = LabelFrame(train_tab, text=" " + adv_params_txt[lang] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue', borderwidth=2)
adv_params.configure(font=(text_font, first_level_frame_font_size, "bold"))
adv_params.grid(column=0, row=adv_params_row, sticky='nesw')
adv_params.columnconfigure(0, weight=1, minsize=label_width)
adv_params.columnconfigure(1, weight=1, minsize=widget_width)

# proportion of validation data
lbl_val_prop_txt = ["Proportion as validation data", "Proporción datos para validación"]
row_lbl_val_prop = 0
lbl_val_prop = Label(adv_params, text=lbl_val_prop_txt[lang], width=1, anchor="w")
lbl_val_prop.grid(row=row_lbl_val_prop, sticky='nesw', pady=2)
var_val_prop = DoubleVar()
var_val_prop.set(0.2)
scl_val_prop = Scale(adv_params, from_=0.01, to=1, resolution=0.01, orient=HORIZONTAL, variable=var_val_prop, showvalue=0, width=10, length=1)
scl_val_prop.grid(row=row_lbl_val_prop, column=1, sticky='ew', padx=10)
dsp_val_prop = Label(adv_params, textvariable=var_val_prop)
dsp_val_prop.config(fg="darkred")
dsp_val_prop.grid(row=row_lbl_val_prop, column=0, sticky='e', padx=0)

# proportion of test data
lbl_test_prop_txt = ["Proportion as test data", "Proporción datos para prueba"]
row_lbl_test_prop = 1
lbl_test_prop = Label(adv_params, text=lbl_test_prop_txt[lang], width=1, anchor="w")
lbl_test_prop.grid(row=row_lbl_test_prop, sticky='nesw', pady=2)
var_test_prop = DoubleVar()
var_test_prop.set(0)
scl_test_prop = Scale(adv_params, from_=0.000000001, to=1, resolution=0.01, orient=HORIZONTAL, variable=var_test_prop, showvalue=0, width=10, length=1) # not 0 because display needs two digits
scl_test_prop.grid(row=row_lbl_test_prop, column=1, sticky='ew', padx=10)
dsp_test_prop = Label(adv_params, textvariable=var_test_prop)
dsp_test_prop.config(fg="darkred")
dsp_test_prop.grid(row=row_lbl_test_prop, column=0, sticky='e', padx=0)

# use GPU for training
lbl_train_gpu_txt = ["Search for GPU and use if available", "Buscar GPU y utilizarla si está disponible"]
row_train_gpu = 2
lbl_train_gpu = Label(adv_params, text=lbl_train_gpu_txt[lang], width=1, anchor="w")
lbl_train_gpu.grid(row=row_train_gpu, sticky='nesw', pady=2)
var_train_gpu = BooleanVar()
if platform.system() == 'Darwin': # Apple Silicon still has some issues with running on MPS 
    var_train_gpu.set(False)
else:
    var_train_gpu.set(True)
chb_train_gpu = Checkbutton(adv_params, variable=var_train_gpu, anchor = "w")
chb_train_gpu.grid(row=row_train_gpu, column=1, sticky='nesw', padx=5)

# batch size
lbl_batch_size_txt = ["Batch size", "Tamaño de lote"]
lbl_batch_size_txt_extra = ["(leave blank for auto detect maximum)", ""]
row_batch_size = 3
lbl_batch_size = tk.Label(adv_params, text=f"{lbl_batch_size_txt[lang]} {lbl_batch_size_txt_extra[lang]}", pady=2, width=1, anchor="w")
lbl_batch_size.grid(row=row_batch_size, sticky='nesw')
var_batch_size = StringVar()
ent_batch_size = tk.Entry(adv_params, textvariable=var_batch_size, fg='grey', width=1)
ent_batch_size.grid(row=row_batch_size, column=1, sticky='nesw', padx=5)
ent_batch_size.insert(0, f"{eg_txt[lang]}: 8")
ent_batch_size.bind("<FocusIn>", batch_size_focus_in)

# number of workers
lbl_n_workers_txt = ["Number of workers", "Número de núcleos"]
lbl_n_workers_txt_extra = ["(leave blank for default 4)", ""]
row_n_workers = 4
lbl_n_workers = tk.Label(adv_params, text=f"{lbl_n_workers_txt[lang]} {lbl_n_workers_txt_extra[lang]}", pady=2, width=1, anchor="w")
lbl_n_workers.grid(row=row_n_workers, sticky='nesw')
var_n_workers = StringVar()
ent_n_workers = tk.Entry(adv_params, textvariable=var_n_workers, fg='grey', width=1)
ent_n_workers.grid(row=row_n_workers, column=1, sticky='nesw', padx=5)
ent_n_workers.insert(0, f"{eg_txt[lang]}: 2")
ent_n_workers.bind("<FocusIn>", n_workers_focus_in)

# image size
lbl_image_size_for_training_txt = ["Image size", "Tamaño imagen"]
lbl_image_size_for_training_txt_extra = ["(leave blank for auto selection)", ""]
row_image_size_for_training = 5
lbl_image_size_for_training = tk.Label(adv_params, text=f"{lbl_image_size_for_training_txt[lang]} {lbl_image_size_for_training_txt_extra[lang]}", pady=2, width=1, anchor="w")
lbl_image_size_for_training.grid(row=row_image_size_for_training, sticky='nesw')
var_image_size_for_training = StringVar()
ent_image_size_for_training = tk.Entry(adv_params, textvariable=var_image_size_for_training, fg='grey', width=1)
ent_image_size_for_training.grid(row=row_image_size_for_training, column=1, sticky='nesw', padx=5)
ent_image_size_for_training.insert(0, f"{eg_txt[lang]}: 1280")
ent_image_size_for_training.bind("<FocusIn>", image_size_for_training_focus_in)

# cache images
lbl_cache_imgs_txt = ["Cache images for faster training", "Almacenar imágenes en cache"]
row_cache_imgs = 6
lbl_cache_imgs = Label(adv_params, text=lbl_cache_imgs_txt[lang], width=1, anchor="w")
lbl_cache_imgs.grid(row=row_cache_imgs, sticky='nesw', pady=2)
var_cache_imgs = BooleanVar()
var_cache_imgs.set(False)
chb_cache_imgs = Checkbutton(adv_params, variable=var_cache_imgs, anchor = "w")
chb_cache_imgs.grid(row=row_cache_imgs, column=1, sticky='nesw', padx=5)

# hyperparameters config file
lbl_hyper_file_txt = ["Hyperparameter configuration file", "Archivo configuración hyperparámetros"]
row_hyper_file = 7
lbl_hyper_file = Label(adv_params, text=lbl_hyper_file_txt[lang], pady=2, width=1, anchor="w")
lbl_hyper_file.grid(row=row_hyper_file, sticky='nesw', pady=2)
dpd_hyper_file_options = [["None", "Low augmentation", "Med augmentation", "High augmentation", "Objects365 training", "VOC training", "Other"], ["Ninguno", "Aumento bajo", "Aumento medio", "Aumento alto", "Objects365", "VOC", "Otro"]]
var_hyper_file = StringVar(adv_params)
var_hyper_file.set(dpd_hyper_file_options[lang][0])
var_hyper_file_short = StringVar()
var_hyper_file_path = StringVar()
var_hyper_file_path.set("")
dpd_hyper_file = OptionMenu(adv_params, var_hyper_file, *dpd_hyper_file_options[lang], command=set_hyper_file)
dpd_hyper_file.configure(width=1)
dpd_hyper_file.grid(row=row_hyper_file, column=1, sticky='nesw', padx=5)
dsp_hyper_file = Label(master=adv_params, textvariable=var_hyper_file_short, fg='darkred')

# evolve hyperparameters
lbl_evolve_txt = ["Evolve hyperparameters", "Evolución de hyperparámetros"]
row_evolve = 8
lbl_evolve = Label(adv_params, text=lbl_evolve_txt[lang], width=1, anchor="w")
lbl_evolve.grid(row=row_evolve, sticky='nesw', pady=2)
var_evolve = BooleanVar()
var_evolve.set(False)
chb_evolve = Checkbutton(adv_params, variable=var_evolve, command=toggle_n_evolutions, anchor = "w")
chb_evolve.grid(row=row_evolve, column=1, sticky='nesw', padx=5)

# number of generations to evolve
lbl_n_generations_txt = ["Number of generations", "Número de generaciones"]
lbl_n_generations_txt_extra = ["(leave blank for default 300)", ""]
row_n_generations = 9
lbl_n_generations = tk.Label(adv_params, text=f"{lbl_n_generations_txt[lang]} {lbl_n_generations_txt_extra[lang]}", pady=2, width=1, anchor="w")
var_n_generations = StringVar()
ent_n_generations = tk.Entry(adv_params, textvariable=var_n_generations, fg='grey', width=1)
ent_n_generations.insert(0, f"{eg_txt[lang]}: 500")
ent_n_generations.bind("<FocusIn>", n_generations_focus_in)

# name of the run
lbl_run_name_txt = ["Run name", "Nombre de la ejecución"]
lbl_run_name_txt_extra = ["(leave blank for auto iterate)", ""]
row_run_name = 10
lbl_run_name = tk.Label(adv_params, text=f"{lbl_run_name_txt[lang]} {lbl_run_name_txt_extra[lang]}", pady=2, width=1, anchor="w")
lbl_run_name.grid(row=row_run_name, sticky='nesw')
var_run_name = StringVar()
ent_run_name = tk.Entry(adv_params, textvariable=var_run_name, fg='grey', width=1)
ent_run_name.grid(row=row_run_name, column=1, sticky='nesw', padx=5)
ent_run_name.insert(0, f"{eg_txt[lang]}: {['Initial run', 'Proceso inicial'][lang]}")
ent_run_name.bind("<FocusIn>", run_name_focus_in)

# number of frozen layers
lbl_n_freeze_layers_txt = ["Number of layers to freeze", "Número de capas a congelar"]
lbl_n_freeze_layers_txt_extra = ["(leave blank for all)", ""]
row_n_freeze_layers = 11
lbl_n_freeze_layers = tk.Label(adv_params, text=f"{lbl_n_freeze_layers_txt[lang]} {lbl_n_freeze_layers_txt_extra[lang]}", pady=2, width=1, anchor="w")
var_n_freeze_layers = StringVar()
ent_n_freeze_layers = tk.Entry(adv_params, textvariable=var_n_freeze_layers, fg='grey', width=1)
ent_n_freeze_layers.insert(0, f"{eg_txt[lang]}: 12")
ent_n_freeze_layers.bind("<FocusIn>", n_freeze_layers_focus_in)

# create command button
btn_start_training_txt = ["Start training", "Comenzar entrenamiento"]
row_start_training = 5
btn_start_training = Button(train_tab, text=btn_start_training_txt[lang], command=start_training)
btn_start_training.grid(row=row_start_training, column=0, sticky='ew')

### console output
train_output_txt = ["Console output", "Salida de consola"]
row_train_output = 0
train_output = LabelFrame(train_tab, text=" " + train_output_txt[lang] + " ", pady=2, padx=5, relief='solid', highlightthickness=5, font=100, fg='darkblue', borderwidth=2)
train_output.configure(font=(text_font, first_level_frame_font_size, "bold"))
train_output.grid(column=1, row=row_train_output, rowspan=4, sticky='nesw')
txt_train_output = Text(train_output, wrap=WORD, width=1, height=1)
txt_train_output.pack(fill="both", expand=True)
cancel_training_bool = BooleanVar()
cancel_training_bool.set(False)

# cancel button
btn_cancel_training_txt = ["Cancel training", "Cancelar entrenamiento"]
row_cancel_training = 5
btn_cancel_training = Button(train_tab, text=btn_cancel_training_txt[lang], command=cancel_training)
btn_cancel_training.grid(row=row_cancel_training, column=1, sticky='ew')
btn_cancel_training.config(state=DISABLED)

# set minsize for all rows inside labelframes...
for frame in [fst_step, snd_step, cls_frame, img_frame, vid_frame, fth_step, sep_frame, req_params, adv_params]:
    set_minsize_rows(frame)

# ... but not for the hidden rows
snd_step.grid_rowconfigure(row_cls_animal_thresh, minsize=0) # model tresh
snd_step.grid_rowconfigure(row_image_size_for_deploy, minsize=0) # image size for deploy
snd_step.grid_rowconfigure(cls_frame_row, minsize=0) # cls options
snd_step.grid_rowconfigure(img_frame_row, minsize=0) # image options
snd_step.grid_rowconfigure(vid_frame_row, minsize=0) # video options
cls_frame.grid_rowconfigure(row_cls_animal_thresh, minsize=0) # cls animal thresh
cls_frame.grid_rowconfigure(row_smooth_cls_animal, minsize=0) # cls animal smooth
cls_frame.grid_rowconfigure(row_cls_vehicle_thresh, minsize=0) # cls vehicle thresh
cls_frame.grid_rowconfigure(row_cls_person_thresh, minsize=0) # cls person thresh
fth_step.grid_rowconfigure(sep_frame_row, minsize=0) # sep options
req_params.grid_rowconfigure(row_model_architecture, minsize=0) # model architecture
adv_params.grid_rowconfigure(row_n_freeze_layers, minsize=0) # n frozen layers
adv_params.grid_rowconfigure(row_n_generations, minsize=0) # n generations

# enable scroll on mousewheel
def deploy_canvas_mousewheel(event):
    if os.name == 'nt':
        deploy_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
    else:
        deploy_canvas.yview_scroll(int(-1 * (event.delta / 2)), 'units')

# make deploy_tab scrollable
deploy_canvas.update_idletasks()
deploy_canvas.configure(scrollregion=deploy_canvas.bbox("all"))
deploy_canvas.bind_all("<MouseWheel>", deploy_canvas_mousewheel)
deploy_canvas.bind_all("<Button-4>", deploy_canvas_mousewheel)
deploy_canvas.bind_all("<Button-5>", deploy_canvas_mousewheel)

# help tab
scroll = Scrollbar(help_tab)
help_text = Text(help_tab, width=1, height=1, wrap=WORD, yscrollcommand=scroll.set) 
help_text.config(spacing1=2, spacing2=3, spacing3=2)
help_text.tag_config('intro', font=f'{text_font} {int(13 * text_size_adjustment_factor)} italic', foreground='black', lmargin1=10, lmargin2=10, underline = False) 
help_text.tag_config('tab', font=f'{text_font} {int(16 * text_size_adjustment_factor)} bold', foreground='black', lmargin1=10, lmargin2=10, underline = True) 
help_text.tag_config('frame', font=f'{text_font} {int(15 * text_size_adjustment_factor)} bold', foreground='darkblue', lmargin1=15, lmargin2=15) 
help_text.tag_config('feature', font=f'{text_font} {int(14 * text_size_adjustment_factor)} normal', foreground='black', lmargin1=20, lmargin2=20, underline = True) 
help_text.tag_config('explanation', font=f'{text_font} {int(13 * text_size_adjustment_factor)} normal', lmargin1=25, lmargin2=25)
hyperlink1 = HyperlinkManager(help_text)

# import images for help tab
yolo_models=Image.open(os.path.join(EcoAssist_files, "EcoAssist", "imgs", "yolo_models.png"))
yolo_models=yolo_models.resize((int(yolo_models.size[0] / 5), int(yolo_models.size[1] / 5)))
yolo_models=ImageTk.PhotoImage(yolo_models)
data_augs=Image.open(os.path.join(EcoAssist_files, "EcoAssist", "imgs", "data_augmentations.jpg"))
data_augs=data_augs.resize((int(data_augs.size[0] / 5), int(data_augs.size[1] / 5)))
data_augs=ImageTk.PhotoImage(data_augs)

# function to write text which can be called when user changes language settings
def write_help_tab():
    global help_text
    line_number = 1 

    # intro sentence
    help_text.insert(END, ["Below you can find detailed documentation for each setting. If you have any questions, feel free to contact me on ",
                           "A continuación encontrarás documentación detallada sobre cada ajuste. Si tienes alguna pregunta, no dudes en ponerte en contacto conmigo en "][lang])
    help_text.insert(INSERT, "petervanlunteren@hotmail.com", hyperlink1.add(partial(webbrowser.open, "mailto:petervanlunteren@hotmail.com")))
    help_text.insert(END, [" or raise an issue on the ", " o plantear una incidencia en "][lang])
    help_text.insert(INSERT, ["GitHub page", "la página de GitHub"][lang], hyperlink1.add(partial(webbrowser.open, "https://github.com/PetervanLunteren/EcoAssist/issues")))
    help_text.insert(END, ".\n\n")
    help_text.tag_add('intro', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # tutorials
    help_text.insert(END, ["TUTORIAL\n", "TUTORIAL\n"][lang])
    help_text.tag_add('tab', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1   

    # step-by-step
    help_text.insert(END, ["Step-by-step\n", "Paso-a-paso\n"][lang])
    help_text.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.insert(END, ['Click ', 'Haga clic '][lang])
    help_text.insert(INSERT, ["here", "aquí"][lang], hyperlink1.add(partial(webbrowser.open, "https://medium.com/towards-artificial-intelligence/train-and-deploy-custom-object-detection-models-without-a-single-line-of-code-a65e58b57b03")))
    help_text.insert(END, [' for a step-by-step tutorial on how to use EcoAssist.\n\n',
                           ' para ver un tutorial paso a paso sobre cómo usar EcoAssist (en inglés).\n\n'][lang])
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2 

    # deploy tab
    help_text.insert(END, ["DEPLOY TAB\n", "PESTAÑA DESPLIEGUE\n"][lang])
    help_text.tag_add('tab', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1

    # first step
    help_text.insert(END, f"{fst_step_txt[lang]}\n")
    help_text.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.insert(END, f"{browse_txt[lang]}\n")
    help_text.insert(END, ["Here you can browse for a folder which contains images and/or video\'s. The model will be deployed on this directory, as well as the post-processing analyses.\n\n",
                           "Aquí puede buscar una carpeta que contenga imágenes y/o vídeos. El modelo se desplegará en este directorio, así como los análisis de post-procesamiento.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # second step
    help_text.insert(END, f"{snd_step_txt[lang]}\n")
    help_text.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1

    # model
    help_text.insert(END, f"{lbl_model_txt[lang]}\n")
    help_text.insert(END, ["Here, you can indicate the yolov5 model that you want to deploy. If the dropdown option 'Custom model' is selected, you will be prompted to select "
                    "a .pt model file. This can be a custom model trained via EcoAssist. The preloaded 'MegaDetector' models detect animals, people, and vehicles in camera "
                    "trap imagery. It does not identify the animals; it just finds them. Version A and B differ only in their training data. Each model can outperform the "
                    "other slightly, depending on your data. Try them both and see which one works best for you. If you really don't have a clue, just stick with the default"
                    " 'MegaDetector 5a'. More info about MegaDetector models ",
                    "Aquí puede indicar el modelo yolov5 que desea desplegar. Si se selecciona la opción desplegable 'Modelo personalizado', se le pedirá que seleccione un "
                    "archivo de modelo con extensión .pt. Puede tratarse de un modelo personalizado entrenado mediante EcoAssist. Los modelos 'MegaDetector' preinstalados "
                    "detectan animales, personas y vehículos en las imágenes de las cámaras trampa. No identifica a los animales, sólo los encuentra. Las versiones A y B sólo"
                    " difieren en sus datos de entrenamiento. Cada modelo puede superar ligeramente al otro, dependiendo de sus datos. Pruebe los dos y vea cuál le funciona mejor. "
                    "Si realmente no tienes ni idea, quédate con el 'MegaDetector 5a' por defecto. Más información sobre los modelos MegaDetector "][lang])
    help_text.insert(INSERT, ["here", "aquí"][lang], hyperlink1.add(partial(webbrowser.open, "https://github.com/ecologize/CameraTraps/blob/main/megadetector.md#megadetector-v50-20220615")))
    help_text.insert(END, ".\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # exclude subs
    help_text.insert(END, f"{lbl_exclude_subs_txt[lang]}\n")
    help_text.insert(END, ["By default, EcoAssist will recurse into subdirectories. Select this option if you want to ignore the subdirectories and process only the "
                    "files directly in the chosen folder.\n\n",
                    "Por defecto, EcoAssist buscará en los subdirectorios. Seleccione esta opción si desea ignorar los subdirectorios y procesar sólo los archivos directamente en la carpeta elegida.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # exclude detections
    help_text.insert(END, f"{lbl_use_custom_img_size_for_deploy_txt[lang]} / {lbl_image_size_for_deploy_txt[lang]}\n")
    help_text.insert(END, ["EcoAssist will resize the images before they get processed. EcoAssist will by default resize the images to 1280 pixels. "
                    "Deploying a model with a lower image size will reduce the processing time, but also the detection accuracy. Best results are obtained if you use the"
                    " same image size as the model was trained on. If you trained a model in EcoAssist using the default image size, you should set this value to 640 for "
                    "the YOLOv5 models. Use the default for the MegaDetector models.\n\n",
                    "EcoAssist redimensionará las imágenes antes de procesarlas. Por defecto, EcoAssist redimensionará las imágenes a 1280 píxeles. Desplegar un modelo "
                    "con un tamaño de imagen inferior reducirá el tiempo de procesamiento, pero también la precisión de la detección. Los mejores resultados se obtienen "
                    "si se utiliza el mismo tamaño de imagen con el que se entrenó el modelo. Si ha entrenado un modelo en EcoAssist utilizando el tamaño de imagen por "
                    "defecto, debe establecer este valor en 640 para los modelos YOLOv5. Utilice el valor por defecto para los modelos MegaDetector.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # use absolute paths
    help_text.insert(END, f"{lbl_abs_paths_txt[lang]}\n")
    help_text.insert(END, ["By default, the paths in the output file are relative (i.e. 'image.jpg') instead of absolute (i.e. '/path/to/some/folder/image.jpg'). This "
                    "option will make sure the output file contains absolute paths, but it is not recommended. Third party software (such as ",
                    "Por defecto, las rutas en el archivo de salida son relativas (es decir, 'imagen.jpg') en lugar de absolutas (es decir, '/ruta/a/alguna/carpeta/"
                    "imagen.jpg'). Esta opción se asegurará de que el archivo de salida contenga rutas absolutas, pero no se recomienda. Software de terceros (como "][lang])
    help_text.insert(INSERT, "Timelapse", hyperlink1.add(partial(webbrowser.open, "https://saul.cpsc.ucalgary.ca/timelapse/")))
    help_text.insert(END, [") will not be able to read the output file if the paths are absolute. Only enable this option if you know what you are doing. More information"
                    " how to use Timelapse in conjunction with MegaDetector, see the ",
                    ") no serán capaces de leer el archivo de salida si las rutas son absolutas. Solo active esta opción si sabe lo que está haciendo. Para más información"
                    " sobre cómo utilizar Timelapse junto con MegaDetector, consulte "][lang])
    help_text.insert(INSERT, ["Timelapse Image Recognition Guide", "la Guía de Reconocimiento de Imágenes de Timelapse"][lang], hyperlink1.add(partial(webbrowser.open, "https://saul.cpsc.ucalgary.ca/timelapse/uploads/Guides/TimelapseImageRecognitionGuide.pdf")))
    help_text.insert(END, ".\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # use checkpoints
    help_text.insert(END, f"{lbl_use_checkpnts_txt[lang]}\n")
    help_text.insert(END, ["This is a functionality to save results to checkpoints intermittently, in case a technical hiccup arises. That way, you won't have to restart"
                    " the entire process again when the process is interrupted.\n\n",
                    "Se trata de una funcionalidad para guardar los resultados en puntos de control de forma intermitente, en caso de que surja un contratiempo técnico. "
                    "De esta forma, no tendrás que reiniciar todo el proceso de nuevo cuando éste se interrumpa.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # checkpoint frequency
    help_text.insert(END, f"{lbl_checkpoint_freq_txt[lang]}\n")
    help_text.insert(END, ["Fill in how often you want to save the results to checkpoints. The number indicates the number of images after which checkpoints will be saved."
                    " The entry must contain only numeric characters.\n\n",
                    "Introduzca la frecuencia con la que desea guardar los resultados en los puntos de control. El número indica el número de imágenes tras las cuales se "
                    "guardarán los puntos de control. La entrada debe contener sólo caracteres numéricos.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # continue from checkpoint
    help_text.insert(END, f"{lbl_cont_checkpnt_txt[lang]}\n")
    help_text.insert(END, ["Here you can choose to continue from the last saved checkpoint onwards so that the algorithm can continue where it left off. Checkpoints are"
                    " saved into the main folder and look like 'checkpoint_<timestamp>.json'. When choosing this option, it will search for a valid"
                    " checkpoint file and prompt you if it can't find it.\n\n",
                    "Aquí puede elegir continuar desde el último punto de control guardado para que el algoritmo pueda continuar donde lo dejó. Los puntos de control se "
                    "guardan en la carpeta principal y tienen el aspecto 'checkpoint_<fecha y hora>.json'. Al elegir esta opción, se buscará un archivo de punto de control "
                    "válido y se le preguntará si no puede encontrarlo.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # don't process every frame
    help_text.insert(END, f"{lbl_not_all_frames_txt[lang]}\n")
    help_text.insert(END,["When processing every frame of a video, it can take a long time to finish. Here, you can specify whether you want to analyse only a selection of frames."
                    f" At '{lbl_nth_frame_txt[lang]}' you can specify how many frames you want to be analysed.\n\n",
                     "Procesar todos los fotogramas de un vídeo puede llevar mucho tiempo. Aquí puede especificar si desea analizar sólo una selección de fotogramas. "
                    f"En '{lbl_nth_frame_txt[lang]}' puedes especificar cuántos fotogramas quieres que se analicen.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # analyse every nth frame
    help_text.insert(END, f"{lbl_nth_frame_txt[lang]}\n")
    help_text.insert(END, ["Specify how many frames you want to process. By entering 2, you will process every 2nd frame and thus cut process time by half. By entering 10, "
                    "you will shorten process time to 1/10, et cetera. However, keep in mind that the chance of detecting something is also cut to 1/10.\n\n",
                    "Especifique cuántos fotogramas desea procesar. Introduciendo 2, procesará cada 2 fotogramas y reducirá así el tiempo de proceso a la mitad. Introduciendo "
                    "10, reducirá el tiempo de proceso a 1/10, etcétera. Sin embargo, tenga en cuenta que la probabilidad de detectar algo también se reduce a 1/10.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # third step
    help_text.insert(END, f"{trd_step_txt[lang]}\n")
    help_text.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1

    # human verification
    help_text.insert(END, f"{lbl_hitl_main_txt[lang]}\n")
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
                           "entrenamiento y simplemente continuar con el posprocesamiento de los resultados verificados. No aplicable a vídeos.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # forth step
    help_text.insert(END, f"{fth_step_txt[lang]}\n")
    help_text.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1

    # destination folder
    help_text.insert(END, f"{lbl_output_dir_txt[lang]}\n")
    help_text.insert(END, ["Here you can browse for a folder in which the results of the post-processing features will be placed. If nothing is selected, the folder "
                    "chosen at step one will be used as the destination folder.\n\n",
                    "Aquí puede buscar una carpeta en la que se colocarán los resultados de las funciones de postprocesamiento. Si no se selecciona nada, la carpeta "
                    "elegida en el primer paso se utilizará como carpeta de destino.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # separate files
    help_text.insert(END, f"{lbl_separate_files_txt[lang]}\n")
    help_text.insert(END, ["This function divides the files into subdirectories based on their detections. Please be warned that this will be done automatically. "
                    "There will not be an option to review and adjust the detections before the images will be moved. If you want that (a human in the loop), take a look at ",
                    "Esta función divide los archivos en subdirectorios en función de sus detecciones. Tenga en cuenta que esto se hará automáticamente. No habrá opción de "
                    "revisar y ajustar las detecciones antes de mover las imágenes. Si quieres eso (una humano en el bucle), echa un vistazo a "][lang])
    help_text.insert(INSERT, "Timelapse", hyperlink1.add(partial(webbrowser.open, "https://saul.cpsc.ucalgary.ca/timelapse/")))
    help_text.insert(END, [", which offers such a feature. More information about that ",
                           ", que ofrece tal característica. Más información al respecto "][lang])
    help_text.insert(INSERT, ["here", "aquí"][lang], hyperlink1.add(partial(webbrowser.open, "https://saul.cpsc.ucalgary.ca/timelapse/uploads/Guides/TimelapseImageRecognitionGuide.pdf")))
    help_text.insert(END,[" (starting on page 9). The process of importing the output file produced by EcoAssist into Timelapse is described ",
                          " (a partir de la página 9). El proceso de importación del archivo de salida producido por EcoAssist en Timelapse se describe "][lang])
    help_text.insert(INSERT, ["here", "aquí"][lang], hyperlink1.add(partial(webbrowser.open, "https://saul.cpsc.ucalgary.ca/timelapse/pmwiki.php?n=Main.DownloadMegadetector")))
    help_text.insert(END,".\n\n")
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # method of file placement
    help_text.insert(END, f"{lbl_file_placement_txt[lang]}\n")
    help_text.insert(END, ["Here you can choose whether to move the files into subdirectories, or copy them so that the originals remain untouched.\n\n",
                           "Aquí puedes elegir si quieres mover los archivos a subdirectorios o copiarlos de forma que los originales permanezcan intactos.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # sort results based on confidence
    help_text.insert(END, f"{lbl_sep_conf_txt[lang]}\n")
    help_text.insert(END, ["This feature will further separate the files based on its confidence value (in tenth decimal intervals). That means that each class will"
                        " have subdirectories like e.g. 'conf_0.6-0.7', 'conf_0.7-0.8', 'conf_0.8-0.9', etc.\n\n",
                        "Esta función separará aún más los archivos en función de su valor de confianza (en intervalos decimales). Esto significa que cada clase tendrá"
                        " subdirectorios como, por ejemplo, 'conf_0.6-0.7', 'conf_0.7-0.8', 'conf_0.8-0.9', etc.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # visualize files
    help_text.insert(END, f"{lbl_vis_files_txt[lang]}\n")
    help_text.insert(END, ["This functionality draws boxes around the detections and prints their confidence values. This can be useful to visually check the results."
                    " Videos can't be visualized using this tool. Please be aware that this action is permanent and cannot be undone. Be wary when using this on original images.\n\n",
                    "Esta funcionalidad dibuja recuadros alrededor de las detecciones e imprime sus valores de confianza. Esto puede ser útil para comprobar visualmente los "
                    "resultados. Los vídeos no pueden visualizarse con esta herramienta.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # crop files
    help_text.insert(END, f"{lbl_crp_files_txt[lang]}\n")
    help_text.insert(END, ["This feature will crop the detections and save them as separate images. Not applicable for videos.\n\n",
                           "Esta función recortará las detecciones y las guardará como imágenes separadas. No es aplicable a los vídeos.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # export csv files
    help_text.insert(END, f"{lbl_csv_txt[lang]}\n")
    help_text.insert(END, ["This will translate the output files of step 2 into csv files and try to retrieve image metadata like date, settings, and GPS. Can be opened in "
                           "spreadsheet applications such as Excel and Numbers and imported for further processing in R, Python, etc.\n\n",
                    "Esto convertirá los archivos de salida del paso 2 en archivos csv e intentará recuperar los metadatos de las imágenes, como la fecha, configuraciones "
                    "y GPS. Pueden abrirse en aplicaciones de hojas de cálculo como Excel y Numbers e importarse para su posterior procesamiento en R, Python, etc.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # classification confidence threshold TODO: dit is nog oude text
    help_text.insert(END, f"{lbl_cls_animal_thresh_txt[lang]}\n")
    help_text.insert(END, ["Detections below this value will not be post-processed. To adjust the threshold value, you can drag the slider or press either sides next to "
                    "the slider for a 0.005 reduction or increment. Confidence values are within the [0.005, 1] interval. If you set the confidence threshold too high, "
                    "you will miss some detections. On the other hand, if you set the threshold too low, you will get false positives. When choosing a threshold for your "
                    f"project, it is important to choose a threshold based on your own data. My advice is to first visualize your data ('{lbl_vis_files_txt[lang]}') with a low "
                    "threshold to get a feeling of the confidence values in your data. This will show you how sure the model is about its detections and will give you an "
                    "insight into which threshold will work best for you. If you really don't know, 0.2 is probably a conservative threshold for most projects.\n\n",
                    "Las detecciones por debajo de este valor no se postprocesarán. Para ajustar el valor del umbral, puede arrastrar el control deslizante o pulsar "
                    "cualquiera de los lados junto al control deslizante para una reducción o incremento de 0,005. Los valores de confianza están dentro del intervalo "
                    "[0,005, 1]. Si ajusta el umbral de confianza demasiado alto, pasará por alto algunas detecciones. Por otro lado, si fija el umbral demasiado bajo, "
                    "obtendrá falsos positivos. Al elegir un umbral para su proyecto, es importante elegir un umbral basado en sus propios datos. Mi consejo es que primero"
                    f" visualice sus datos ('{lbl_vis_files_txt[lang]}') con un umbral bajo para hacerse una idea de los valores de confianza de sus datos. Esto le mostrará lo "
                    "seguro que está el modelo sobre sus detecciones y le dará una idea de qué umbral funcionará mejor para usted. Si realmente no lo sabe, 0,2 es "
                    "probablemente un umbral conservador para la mayoría de los proyectos.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # train tab
    help_text.insert(END, ["TRAIN TAB\n", "PESTAÑA ENTRENAMIENTO\n"][lang])
    help_text.tag_add('tab', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1

    # training type
    help_text.insert(END, f"{req_params_txt[lang]}\n")
    help_text.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.insert(END, f"{lbl_train_type_txt[lang]}\n")
    help_text.insert(END, ["Here, you can specify whether you want to start a new training or resume an existing one. If you want to resume, you'll need to specify the checkpoint"
                    f" file at '{lbl_resume_checkpoint_txt[lang]}'.\n\n",
                    "Aquí puede especificar si desea iniciar un nuevo entrenamiento o reanudar uno existente. Si desea reanudar, tendrá que especificar el archivo de punto de "
                    f"control en '{lbl_resume_checkpoint_txt[lang]}'.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # folder with labelled data
    help_text.insert(END, f"{lbl_annotated_data_txt[lang]}\n")
    help_text.insert(END, ["Browse the folder containing images and annotations in Pascal VOC format. All data should be in this folder, not in subfolders. EcoAssist will randomly partition"
                    " the data into a training, test and validation set (based on the proportions set by you). You can annotate your data using step 3 of the 'Deploy' tab.\n\n",
                    "Examine la carpeta que contiene las imágenes y anotaciones en formato Pascal VOC. Todos los datos deben estar en esta carpeta, no en subcarpetas. EcoAssist dividirá "
                    "aleatoriamente los datos en un conjunto de entrenamiento, prueba y validación (basado en las proporciones establecidas por usted). Puede anotar sus datos en el paso 3"
                    " de la pestaña 'Despliegue'.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # retrain from
    help_text.insert(END, f"{lbl_learning_model_txt[lang]}\n")
    help_text.insert(END, ["In machine learning, it is possible to reuse an existing model as the starting point for a new model. For example, the MegaDetector model is "
                    "excellent at detecting animals in camera trap images. We can transfer the knowledge of how an animal looks like to a new model which can, for example,"
                    " classify species. Here, you can specify which model you would like to transfer knowledge from. If your dataset is relatively small (i.e., not tens of "
                    "thousands of images), it is advised to train your own custom model using transfer learning. Besides MegaDetector 5a and b, you can choose from "
                    "five pre-trained yolov5 models (see image below). These go from small and fast to large and slow and are trained on the ",
                    "En el aprendizaje automático, es posible reutilizar un modelo existente como punto de partida para un nuevo modelo. Por ejemplo, el modelo MegaDetector"
                    " es excelente para detectar animales en imágenes de cámaras trampa. Podemos transferir los conocimientos sobre el aspecto de un animal a un nuevo modelo"
                    " que pueda, por ejemplo, clasificar especies. Aquí puede especificar de qué modelo desea transferir los conocimientos. Si su conjunto de datos es "
                    "relativamente pequeño (es decir, no decenas de miles de imágenes), se aconseja entrenar su propio modelo personalizado utilizando el aprendizaje por "
                    "transferencia. Además de MegaDetector 5a y b, puede elegir entre cinco modelos yolov5 preentrenados (véase la imagen siguiente). Estos modelos van de "
                    "pequeños y rápidos a grandes y lentos, y se han entrenado con el conjunto de "][lang])
    help_text.insert(INSERT, ["COCO dataset", "datos COCO"][lang], hyperlink1.add(partial(webbrowser.open, "https://cocodataset.org/#home")))
    help_text.insert(END, [" consisting of more than 330,000 images of 80 classes. The larger the model, the better the results. The Nano model is the smallest and fastest and is "
                    "most suitable for mobile solutions and embedded devices. The small model is perfect for a laptop which doesn't have any GPU acceleration. The medium-sized "
                    "model provides a good balance between speed and accuracy, but you'll probably want a GPU for this. The large model is ideal for detecting small objects, and "
                    "the extra-large model is the most accurate of them all (but it takes time and quite some processing power to train and deploy). The last two models are "
                    "recommended for cloud deployments. You can also specify a custom model or choose to train from scratch, but it is usually not recommended. Only train"
                    " from scratch if you know what you are doing and have a very large dataset (i.e., around 150.000 images or more).\n\n",
                    ", compuesto por más de 330.000 imágenes de 80 clases. Cuanto mayor es el modelo, mejores son los resultados. El modelo Ínfimo es el más pequeño y rápido"
                    " y es el más adecuado para soluciones móviles y dispositivos integrados. El modelo pequeño es perfecto para un portátil que no disponga de aceleración por"
                    " GPU. El modelo mediano ofrece un buen equilibrio entre velocidad y precisión, pero probablemente necesitarás una GPU para ello. El modelo grande es ideal "
                    "para detectar objetos pequeños, y el modelo extragrande es el más preciso de todos (pero requiere tiempo y bastante capacidad de procesamiento para entrenarlo"
                    " y desplegarlo). Los dos últimos modelos se recomiendan para despliegues en la nube. También puedes especificar un modelo personalizado o elegir entrenar desde"
                    " cero, pero normalmente no se recomienda. Entrena desde cero sólo si sabes lo que estás haciendo y tienes un conjunto de datos muy grande (es decir, alrededor de"
                    " 150.000 imágenes o más).\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # image of yolo models 
    help_text.image_create(tk.END, image = yolo_models)
    help_text.insert(END, "\n\n")
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # model architecture
    help_text.insert(END, f"{lbl_model_architecture_txt[lang]}\n")
    help_text.insert(END, ["When training from scratch, you can specify the model architecture here. The options link to the architectures of the models depicted above.\n\n",
                           "Cuando se entrena desde cero, puede especificar aquí la arquitectura del modelo. Las opciones están vinculadas a las arquitecturas de los modelos"
                           " descritos anteriormente.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # n epochs
    help_text.insert(END, f"{lbl_n_epochs_txt[lang]}\n")
    help_text.insert(END, ["An epoch is one cycle in which all the training data is processed once. The number of epochs required depends on the project. It is recommended to"
                    " start with 300 epochs and then check the results for overfitting. Reduce the number of epochs if the data is overfitted, and increase if not. Overfitting"
                    " is indicated by increasing validation losses. You can see this these validation losses in the 'results.png' file located in the destination folder after "
                    "completing a training. Increase the amount of data or use data augmentation to avoid overfitting.\n\n",
                    "Una época es un ciclo en el que todos los datos de entrenamiento se procesan una vez. El número de épocas necesarias depende del proyecto. Se recomienda "
                    "empezar con 300 épocas y comprobar los resultados para ver si hay sobreajuste. Reduzca el número de épocas si los datos están sobreajustados y auméntelo "
                    "si no lo están. El sobreajuste se indica mediante el aumento de las pérdidas de validación. Puede ver estas pérdidas de validación en el archivo 'results.png'"
                    " situado en la carpeta de destino después de completar un entrenamiento. Aumente la cantidad de datos o utilice el aumento de datos para evitar el "
                    "sobreajuste.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # destination folder
    help_text.insert(END, f"{lbl_output_dir_txt[lang]}\n")
    help_text.insert(END, ["Select the folder in which you want the results to be placed.\n\n", "Seleccione la carpeta en la que desea colocar los resultados.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # project name
    help_text.insert(END, ["Project name\n", "Nombre del proyecto\n"][lang])
    help_text.insert(END, ["Specify the name of the project. Results will be saved in the folder <destination folder>\<project name>\<run name>\.\n\n",
                           "Especifique el nombre del proyecto. Los resultados se guardarán en la carpeta <carpeta de destino>\<nombre del proyecto>\<nombre de la ejecución>\.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # resume checkpoint
    help_text.insert(END, f"{lbl_resume_checkpoint_txt[lang]}\n")
    help_text.insert(END, ["If your training was interrupted and you want to resume where you left off, you can specify the resume checkpoint here. It is the last.pt file"
                    " in the weights subfolder of the training you want to resume. For example: Project_name\exp\weights\last.pt.\n\n",
                    "Si tu entrenamiento fue interrumpido y quieres reanudarlo donde lo dejaste, puedes especificar aquí el punto de control de reanudación. Es el 'last.pt'"
                    " en la subcarpeta 'weights' del entrenamiento que desea reanudar. Por ejemplo: Nombre_del_proyecto\exp\weights\last.pt.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # validation prop
    help_text.insert(END, f"{adv_params_txt[lang]}\n")
    help_text.tag_add('frame', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.insert(END, f"{lbl_val_prop_txt[lang]}\n")
    help_text.insert(END, ["Here, you can select the proportion of images which will be randomly selected to become the validation subset. Validation images are required,"
                    " so choosing 0 is not allowed. To adjust the value, you can drag the slider or press either sides next to the slider for a 0.01 reduction or increment."
                    " Values are within the [0.01, 1] interval.\n\n",
                    "Aquí puede seleccionar la proporción de imágenes que se seleccionarán aleatoriamente para convertirse en el subconjunto de validación. Las imágenes de "
                    "validación son obligatorias, por lo que no se permite elegir 0. Para ajustar el valor, puede arrastrar el control deslizante o pulsar cualquiera de los "
                    "lados junto al control deslizante para una reducción o incremento de 0.01. Los valores están dentro del intervalo [0.01, 1].\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # test prop
    help_text.insert(END, f"{lbl_test_prop_txt[lang]}\n")
    help_text.insert(END, ["Here, you can select the proportion of images which will be randomly selected to become the test subset. Test images are not required, so choosing"
                    " 0 is allowed. To adjust the value, you can drag the slider or press either sides next to the slider for a 0.01 reduction or increment. Values are within"
                    " the [0, 1] interval.\n\n",
                    "Aquí puede seleccionar la proporción de imágenes que se seleccionarán aleatoriamente para convertirse en el subconjunto de prueba. Las imágenes de prueba "
                    "no son necesarias, por lo que se permite elegir 0. Para ajustar el valor, puede arrastrar el control deslizante o pulsar cualquiera de los lados junto al "
                    "control deslizante para una reducción o incremento de 0,01. Los valores están dentro del intervalo [0, 1].\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # search for GPU
    help_text.insert(END, f"{lbl_train_gpu_txt[lang]}\n")
    help_text.insert(END, ["If enabled, EcoAssist will check your device for any suitable GPU and use it to train on.\n\n",
                           "Si está activada, EcoAssist buscará en tu dispositivo una GPU adecuada y la utilizará para entrenar.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # batch size
    help_text.insert(END, f"{lbl_batch_size_txt[lang]}\n")
    help_text.insert(END, ["The batch size is the number of training examples used in one iteration. The larger the batch size, the more processing power you'll need. For "
                    "the best results, use the largest batch size that your hardware allows for. Leave the entry box empty to automatically check and use the maximum batch"
                    " size your device can handle. If your device has no GPU acceleration, the default is 16. Try lowering this if you run into out-of-memory errors.\n\n",
                    "El tamaño del lote es el número de ejemplos de entrenamiento utilizados en una iteración. Cuanto mayor sea el tamaño del lote, más potencia de procesamiento"
                    " necesitará. Para obtener los mejores resultados, utilice el tamaño de lote más grande que permita su hardware. Deje la casilla de entrada vacía para comprobar"
                    " automáticamente y utilizar el tamaño de lote máximo que su dispositivo puede manejar. Si tu dispositivo no tiene aceleración GPU, el valor por defecto es 16. "
                    "Pruebe a reducirlo si se producen errores de memoria insuficiente.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # number of dataloader workers
    help_text.insert(END, f"{lbl_n_workers_txt[lang]}\n")
    help_text.insert(END, ["Usually, not all images can be loaded into your computers RAM at once. Subsets of the images are therefore loaded at every iteration. This number "
                    "indicates the maximum amount of workers being set for this dataloading task. Normally, the default of 4 should be fine for most computers, but if you run"
                    " into out-of-memory errors, it might help to lower the number of dataloaders.\n\n",
                    "Normalmente, no se pueden cargar todas las imágenes a la vez en la memoria RAM del ordenador. Por lo tanto, subconjuntos de las imágenes se cargan en cada "
                    "iteración. Este número indica la cantidad máxima de trabajadores para esta tarea de carga de datos. Normalmente, el valor por defecto de 4 debería estar "
                    "bien para la mayoría de los ordenadores, pero si se encuentra con errores de falta de memoria, podría ayudar a reducir el número de cargadores de "
                    "datos.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # image size
    help_text.insert(END, f"{lbl_image_size_for_training_txt[lang]}\n")
    help_text.insert(END, ["Larger image sizes usually lead to better results, but take longer to process. Training on smaller images will cost less computing power. An image"
                    " that is twice as large will have 4 times as many pixels to learn from. Resizing images is therefore a crucial part of object detection. Best results are"
                    " obtained if the same image size is used as the original model you are retraining. Therefore, if you leave the image size entry box empty, EcoAssist will"
                    " take the image size of the pretrained model you selected (1280 for MegaDetector and 640 for the YOLO models). If you selected a custom model or are "
                    "training from scratch, the default is 640.\n\n",
                    "Las imágenes de mayor tamaño suelen dar mejores resultados, pero tardan más en procesarse. El entrenamiento en imágenes más pequeñas costará menos potencia"
                    " de cálculo. Una imagen el doble de grande tendrá 4 veces más píxeles de los que aprender. Cambiar el tamaño de las imágenes es, por tanto, una parte crucial"
                    " de la detección de objetos. Los mejores resultados se obtienen si se utiliza el mismo tamaño de imagen que el modelo original que se está reentrenando. Por "
                    "lo tanto, si deja la casilla de entrada de tamaño de imagen vacía, EcoAssist tomará el tamaño de imagen del modelo preentrenado que haya seleccionado (1280 "
                    "para MegaDetector y 640 para los modelos YOLO). Si ha seleccionado un modelo personalizado o está entrenando desde cero, el valor por defecto es 640.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # cache
    help_text.insert(END, f"{lbl_cache_imgs_txt[lang]}\n")
    help_text.insert(END, ["This feature caches the dataset into your RAM for faster load times.\n\n",
                           "Esta función almacena en caché el conjunto de datos en la memoria RAM para acelerar los tiempos de carga.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # hyperparam file
    help_text.insert(END, f"{lbl_hyper_file_txt[lang]}\n")
    help_text.insert(END, ["Here, you can select an existing or custom hyperparameter configuration file. This consists of parameters like learning rate and data augmentations"
                    f" options. It's recommened to train with default hyperparameters first ('{dpd_hyper_file_options[lang][0]}') before trying others. A few ready-to-use options are "
                    "preloaded into EcoAsisst:\n"
                    f"  - '{dpd_hyper_file_options[lang][0]}' to not specify any hyperparameters and use the default settings\n"
                    f"  - '{dpd_hyper_file_options[lang][1]}' for small models like {dpd_learning_model_options[lang][2]} and {dpd_learning_model_options[lang][3]}\n"
                    f"  - '{dpd_hyper_file_options[lang][2]}' for medium sized models like {dpd_learning_model_options[lang][4]}\n"
                    f"  - '{dpd_hyper_file_options[lang][3]}' for large models like {dpd_learning_model_options[lang][5]}, {dpd_learning_model_options[lang][6]} and MegaDetector\n"
                    f"  - '{dpd_hyper_file_options[lang][4]}' when training on the ",
                    "Aquí puede seleccionar un archivo de configuración de hiperparámetros existente o personalizado. Consiste en parámetros como la tasa de aprendizaje y las opciones"
                    f" de aumento de datos. Se recomienda entrenar primero con los hiperparámetros por defecto ('{dpd_hyper_file_options[lang][0]}') antes de probar otros. EcoAsisst "
                    "incluye algunas opciones listas para usar:\n"
                    f"  - '{dpd_hyper_file_options[lang][0]}' para no especificar ningún hiperparámetro y utilizar la configuración por defecto\n"
                    f"  - '{dpd_hyper_file_options[lang][1]}' para modelos pequeños como {dpd_learning_model_options[lang][2]} y {dpd_learning_model_options[lang][3]}\n"
                    f"  - '{dpd_hyper_file_options[lang][2]}' para modelos de tamaño medio como {dpd_learning_model_options[lang][4]}\n"
                    f"  - '{dpd_hyper_file_options[lang][3]}' para modelos grandes como {dpd_learning_model_options[lang][5]}, {dpd_learning_model_options[lang][6]} y MegaDetector\n"
                    f"  - '{dpd_hyper_file_options[lang][4]}' para entrenar con el conjunto de "][lang])
    help_text.insert(INSERT, ["Objects365 dataset", "datos Objects365"][lang], hyperlink1.add(partial(webbrowser.open, "https://paperswithcode.com/dataset/objects365")))
    help_text.insert(END, "\n")
    help_text.insert(END, 
                    [f"  - '{dpd_hyper_file_options[lang][5]}' for training on the ",
                     f"  - '{dpd_hyper_file_options[lang][5]}' para entrenar con el conjunto de "][lang])
    help_text.insert(INSERT, ["VOC dataset", "datos VOC"][lang], hyperlink1.add(partial(webbrowser.open, "http://host.robots.ox.ac.uk/pascal/VOC/")))
    help_text.insert(END, "\n")
    help_text.insert(END, 
                    [f"  - '{dpd_hyper_file_options[lang][6]}' to select a custom hyperparameter file\n"
                    f"Please be aware that the optimal settings are project-specific and the '{lbl_evolve_txt[lang]}' option should be used to finetune them for the best results. "
                    "Hyperparameter evolution can, however, be very time and energy consuming and therefore not always preferable. "
                    "In general, increasing augmentation hyperparameters will reduce and delay overfitting, allowing for longer trainings and higher final accuracy. YOLOv5 "
                    "applies online imagespace and colorspace augmentations to present a new and unique augmented mosaic. Images are never presented twice in the same way."
                    " You can view the effect of your augmentation policy in your train_batch*.jpg images once training starts. These images will be in your train logging "
                    "directory.\n\n",
                    f"  - '{dpd_hyper_file_options[lang][6]}' para seleccionar un archivo de hiperparámetros personalizado\n"
                    f"Tenga en cuenta que los ajustes óptimos son específicos del proyecto y que la opción '{lbl_evolve_txt[lang]}' debe utilizarse para ajustarlos y obtener los "
                    "mejores resultados. Sin embargo, la evolución de los hiperparámetros puede requerir mucho tiempo y energía, por lo que no siempre es preferible. En general, "
                    "aumentar los hiperparámetros de aumento reducirá y retrasará el sobreajuste, permitiendo entrenamientos más largos y una mayor precisión final. YOLOv5 aplica "
                    "aumentos en línea del espacio de imagen y del espacio de color para presentar un mosaico aumentado nuevo y único. Las imágenes nunca se presentan dos veces de "
                    "la misma forma. Puede ver el efecto de su política de aumento en las imágenes train_batch*.jpg una vez iniciado el entrenamiento. Estas imágenes estarán en su "
                    "directorio de registro de entrenamiento.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # data augmentation image
    help_text.image_create(tk.END, image = data_augs)
    help_text.insert(END, "\n\n")
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # evolve
    help_text.insert(END, f"{lbl_evolve_txt[lang]}\n")
    help_text.insert(END, ["Use this feature to fine-tune the hyperparameters for maximum fitness. It will run the base training multiple times and slightly adjust the "
                    "hyperparameters to find the optimum values. The resulting hyperparameter file is the 'hyp_evolve.yaml' in your train logging directory and"
                    f" can be used for subsequent trainings. To do so, select the 'hyp_evolve.yaml' file as option under '{dpd_hyper_file_options[lang][6]}' at '{lbl_hyper_file_txt[lang]}'. "
                    "Please note that evolution is generally expensive and time-consuming. It can take weeks or months to finish, depending on the number of generations "
                    "you select and your processing power.\n\n",
                    "Utilice esta función para ajustar con precisión los hiperparámetros con el fin de obtener la máxima calidad. Ejecutará el entrenamiento base varias veces y ajustará"
                    " ligeramente los hiperparámetros para encontrar los valores óptimos. El archivo de hiperparámetros resultante es el 'hyp_evolve.yaml' en su directorio de registro de"
                    " entrenamiento y puede utilizarse para entrenamientos posteriores. Para ello, seleccione el archivo 'hyp_evolve.yaml' como opción en "
                    f"'{dpd_hyper_file_options[lang][6]}' en '{lbl_hyper_file_txt[lang]}'. Tenga en cuenta que la evolución suele ser costosa y requiere mucho tiempo. Puede tardar semanas"
                    " o meses en completarse, dependiendo del número de generaciones que seleccione y de su capacidad de procesamiento.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # n generations
    help_text.insert(END, f"{lbl_n_generations_txt[lang]}\n")
    help_text.insert(END, ["Here, you can specify how often you want the base scenario to be trained during the evolution. A minimum of 300 generations of evolution is "
                    "recommended for best results. The default 300 is used when the entry box is left blank.\n\n",
                    "Aquí puede especificar la frecuencia con la que desea que se entrene el escenario base durante la evolución. Se recomienda un mínimo de 300 generaciones"
                    " de evolución para obtener los mejores resultados. Cuando la casilla de entrada se deja en blanco, se utiliza el valor por defecto 300.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # run name
    help_text.insert(END, f"{lbl_run_name_txt[lang]}\n")
    help_text.insert(END, ["Here you can specify the name of the run. Results will be saved in the folder <destination folder>\<project name>\<run name>. If you leave this "
                    "entry box blank, it will automatically iterate new names: exp, exp2, exp3, etc.\n\n",
                    "Aquí puede especificar el nombre de la ejecución. Los resultados se guardarán en la carpeta <carpeta de destino>\<nombre del proyecto>\<nombre de la "
                    "ejecución>. Si deja esta casilla en blanco, se iterarán automáticamente nuevos nombres: exp, exp2, exp3, etc.\n\n"][lang])
    help_text.tag_add('feature', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=1
    help_text.tag_add('explanation', f"{str(line_number)}.0", f"{str(line_number)}.end");line_number+=2

    # config help_text
    help_text.pack(fill="both", expand=True)
    help_text.configure(font=(text_font, 11, "bold"), state=DISABLED)
    scroll.config(command=help_text.yview)
write_help_tab()

# about tab
about_scroll = Scrollbar(about_tab)
about_text = Text(about_tab, width=1, height=1, wrap=WORD, yscrollcommand=scroll.set)
about_text.config(spacing1=2, spacing2=3, spacing3=2)
about_text.tag_config('title', font=f'{text_font} {int(15 * text_size_adjustment_factor)} bold', foreground='darkblue', lmargin1=10, lmargin2=10) 
about_text.tag_config('info', font=f'{text_font} {int(13 * text_size_adjustment_factor)} normal', lmargin1=20, lmargin2=20)
about_text.tag_config('citation', font=f'{text_font} {int(13 * text_size_adjustment_factor)} normal', lmargin1=30, lmargin2=50)
hyperlink = HyperlinkManager(about_text)

# function to write text which can be called when user changes language settings
def write_about_tab():
    global about_text
    text_line_number=1

    # contact
    about_text.insert(END, ["Contact\n", "Contacto\n"][lang])
    about_text.insert(END, ["Please also help me to keep improving EcoAssist and let me know about any improvements, bugs, or new features so that I can keep it up-to-date. You can "
                           "contact me at ",
                           "Por favor, ayúdame también a seguir mejorando EcoAssist e infórmame de cualquier mejora, error o nueva función para que pueda mantenerlo actualizado. "
                           "Puedes ponerte en contacto conmigo en "][lang])
    about_text.insert(INSERT, "petervanlunteren@hotmail.com", hyperlink.add(partial(webbrowser.open, "mailto:petervanlunteren@hotmail.com")))
    about_text.insert(END, [" or raise an issue on the ", " o plantear un problema en "][lang])
    about_text.insert(INSERT, ["GitHub page", "la página de GitHub"][lang], hyperlink.add(partial(webbrowser.open, "https://github.com/PetervanLunteren/EcoAssist/issues")))
    about_text.insert(END, ".\n\n")
    about_text.tag_add('title', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('info', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=2

    # ecoassist citation
    about_text.insert(END, ["EcoAssist citation\n", "Citar EcoAssist\n"][lang])
    about_text.insert(END, ["If you used EcoAssist in your research, please use the following citation.\n",
                            "Si ha utilizado EcoAssist en su investigación, utilice la siguiente cita.\n"][lang])
    about_text.insert(END, "- van Lunteren, P. (2023). EcoAssist: A no-code platform to train and deploy custom YOLOv5 object detection models. Journal of Open Source Software, 8(88), 5581. ")
    about_text.insert(INSERT, "https://doi.org/10.21105/joss.05581", hyperlink.add(partial(webbrowser.open, "https://doi.org/10.21105/joss.05581")))
    about_text.insert(END, ".\n\n")
    about_text.tag_add('title', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('info', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('citation', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=2

    # megadetector citation
    about_text.insert(END, ["MegaDetector citation\n", "Citar MegaDetector\n"][lang])
    about_text.insert(END, ["If you used the MegaDetector model to analyse images or retrain your model, please use the following citation.\n",
                            "Si ha utilizado el modelo MegaDetector para analizar imágenes o volver a entrenar su modelo, utilice la siguiente cita.\n"][lang])
    about_text.insert(END, "- Beery, S., Morris, D., & Yang, S. (2019). Efficient pipeline for camera trap image review. ArXiv preprint arXiv:1907.06772.\n\n")
    about_text.tag_add('title', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('info', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('citation', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=2

    # image credits
    about_text.insert(END, ["Image credits\n", "Créditos de la imagen\n"][lang])
    about_text.insert(END, ["The beautiful camera trap images of the fox and ocelot displayed at the top were taken from the ",
                            "Las bellas imágenes del zorro y el ocelote captadas por cámaras trampa que aparecen en la parte superior proceden del conjunto de "][lang])
    about_text.insert(INSERT, ["WCS Camera Traps dataset", "datos WCS Camera Traps"][lang], hyperlink.add(partial(webbrowser.open, "https://lila.science/datasets/wcscameratraps")))
    about_text.insert(END, [" provided by the ", " proporcionado por la "][lang])
    about_text.insert(INSERT, "Wildlife Conservation Society", hyperlink.add(partial(webbrowser.open, "https://www.wcs.org/")))
    about_text.insert(END, ".\n\n")
    about_text.tag_add('title', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('info', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=2

    # development credits
    about_text.insert(END, ["Development\n", "Desarrollo\n"][lang])
    about_text.insert(END, ["EcoAssist is developed by ",
                            "EcoAssist ha sido desarrollado por "][lang])
    about_text.insert(INSERT, "Addax Data Science", hyperlink.add(partial(webbrowser.open, "https://addaxdatascience.com/")))
    about_text.insert(END, [" in collaboration with ",
                            " en colaboración con "][lang])
    about_text.insert(INSERT, "Smart Parks", hyperlink.add(partial(webbrowser.open, "https://www.smartparks.org/")))
    about_text.insert(END, ".\n\n")
    about_text.tag_add('title', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=1
    about_text.tag_add('info', str(text_line_number) + '.0', str(text_line_number) + '.end');text_line_number+=2

    # config about_text
    about_text.pack(fill="both", expand=True)
    about_text.configure(font=(text_font, 11, "bold"), state=DISABLED)
    scroll.config(command=about_text.yview)
write_about_tab()

# main function
def main():

    # initialise start screen
    enable_frame(fst_step)
    disable_frame(snd_step)
    disable_frame(trd_step)
    disable_frame(fth_step)

    # run
    root.mainloop()

# executable as script
if __name__ == "__main__":
    main()
