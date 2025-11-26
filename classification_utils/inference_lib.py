# library of inference functions to be used for classifying MD crops 
# Created by Peter van Lunteren
# Latest edit by Peter van Lunteren on 19 May 2025

# import packages
import io
import os
import json
import datetime
import contextlib
import pandas as pd
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
from megadetector.data_management import read_exif
from megadetector.data_management import cct_json_utils
from megadetector.data_management.read_exif import parse_exif_datetime_string

# lets not freak out over truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# MAIN FUNCTION different workflow for videos than for images
def classify_MD_json(json_path,
                     GPU_availability,
                     cls_detec_thresh,
                     cls_class_thresh,
                     smooth_bool,
                     crop_function,
                     inference_function,
                     temp_frame_folder,
                     cls_model_fpath,
                     cls_tax_fallback,
                     cls_tax_levels_idx):
    
    # get the taxonomy mapping, if required
    if cls_tax_fallback:
        
        # load csv
        taxonomy_df = pd.read_csv(os.path.join(os.path.dirname(cls_model_fpath), "taxon-mapping.csv"))

        # check if the user wants to fix at a certain column
        if cls_tax_levels_idx == 0:
            tax_mode = "auto"
            fixed_column = None
        elif cls_tax_levels_idx > 0:
            tax_mode = "manual"
            level_cols = [col for col in taxonomy_df.columns if col.startswith('level_')]
            only_above_cols = [col for col in taxonomy_df.columns if col.startswith('only_above_')]
            fixed_cols_options = level_cols + only_above_cols
            fixed_column = fixed_cols_options[cls_tax_levels_idx - 1] # minus 1 because the first option is for auto mode and does not have a column
            print(f"fixed_column: {fixed_column}")
    else:
        taxonomy_df = None
        tax_mode = None
        fixed_column = None
    
    if json_path.endswith('video_recognition_file.json'):

        # init vars
        json_path_head = os.path.splitext(json_path)[0]
        json_path_tail = os.path.splitext(json_path)[1]
        frame_level_json = json_path_head + '.frames' + json_path_tail

        # for video's we need to classify the frames json instead of the normal json
        convert_detections_to_classification(json_path = frame_level_json,
                                            img_dir = temp_frame_folder,
                                            GPU_availability = GPU_availability,
                                            cls_detec_thresh = cls_detec_thresh,
                                            cls_class_thresh = cls_class_thresh,
                                            smooth_bool = smooth_bool,
                                            crop_function = crop_function,
                                            inference_function = inference_function,
                                            cls_model_fpath = cls_model_fpath,
                                            cls_tax_fallback = cls_tax_fallback,
                                            tax_mode = tax_mode,
                                            fixed_column = fixed_column,
                                            cls_taxonomy_df = taxonomy_df)

    # for images it's much more straight forward
    else:
        convert_detections_to_classification(json_path = json_path,
                                            img_dir = os.path.dirname(json_path),
                                            GPU_availability = GPU_availability,
                                            cls_detec_thresh = cls_detec_thresh,
                                            cls_class_thresh = cls_class_thresh,
                                            smooth_bool = smooth_bool,
                                            crop_function = crop_function,
                                            inference_function = inference_function,
                                            cls_model_fpath = cls_model_fpath,
                                            cls_tax_fallback = cls_tax_fallback,
                                            tax_mode = tax_mode,
                                            fixed_column = fixed_column,
                                            cls_taxonomy_df = taxonomy_df)

# fetch forbidden classes from the model's variables.json
def fetch_forbidden_classes(cls_model_fpath):
    var_file = os.path.join(os.path.dirname(cls_model_fpath), "variables.json")
    with open(var_file, 'r') as file:
        model_vars = json.load(file)
    all_classes = model_vars["all_classes"]
    selected_classes = model_vars["selected_classes"]
    forbidden_classes = [e for e in all_classes if e not in selected_classes]
    return forbidden_classes

# fetch label map from json
def fetch_label_map_from_json(path_to_json):
    with open(path_to_json, "r") as json_file:
        data = json.load(json_file)
    label_map = data['detection_categories']
    return label_map

# set confidence scores of forbidden classes to 0 and normalize the rest
def remove_forbidden_classes(name_classifications, forbidden_classes):
    name_classifications = [[name, 0] if name in forbidden_classes else [name, score] for name, score in name_classifications]
    total_confidence = sum(score for _, score in name_classifications if score > 0)
    name_classifications = [[name, score / total_confidence] if score > 0 else [name, 0] for name, score in name_classifications]
    return name_classifications

# this function is used to classify the crops using a hierarchical classification so it can fall back to a higher level
def hierarchical_classification(name_classifications, taxonomy_df, threshold=0.75, tax_mode = "auto", fixed_column="level_species"):

    # if the user wants to use the hierarchical classification
    if tax_mode == "auto":
        
        # check all the 'level_' columns in the taxonomy_df, as they are the ones that are used for the hierarchical classification
        # sometimes there are more than only the regular taxonomy levels, so we need to check which ones are present
        # for example, ... family, genus, species, sex-age-group
        # as long as all they are hierarchical levels, we can use them
        hierarchical_levels = [col for col in taxonomy_df.columns if col.startswith('level_')]
        hierarchical_level_map = {level: defaultdict(float) for level in hierarchical_levels}
        
        # compute the confidence scores for each level
        for name, confidence in name_classifications:
            name = name.lower()
            match = taxonomy_df[taxonomy_df["model_class"] == name]
            if not match.empty:
                for level in hierarchical_levels:
                    taxon = match.iloc[0][level]
                    if pd.notna(taxon) and taxon != "":
                        hierarchical_level_map[level][taxon] += confidence

        # # print the hierarchical level map
        # # recursively sort inner dicts by value (confidence) in descending order
        # def sort_nested_dict(d):
        #     return {outer_k: dict(sorted(inner_d.items(), key=lambda x: x[1], reverse=True)) for outer_k, inner_d in d.items()}
        # sorted_map = sort_nested_dict(hierarchical_level_map)
        # print(json.dumps(sorted_map, indent=4))

        # now check which level has a high enough confidence
        # in reverse order since that is how the code expects it
        for level in hierarchical_levels[::-1]: 
            leveled = hierarchical_level_map[level]
            if leveled:
                top_label, top_conf = max(leveled.items(), key=lambda x: x[1])
                if top_conf >= threshold:
                    return {"pred": top_label, "conf": top_conf}
        
        # if nothing is found, we can just return the highest level of the class taxon
        highest_level_class = max(hierarchical_level_map['level_class'].items(), key=lambda item: item[1])
        return {"pred": highest_level_class[0], "conf": highest_level_class[1]}

    # if the user wants to fix the classification to a certain level, namely the *fixed_column*
    elif tax_mode == "manual":        
        fixed_level_map = defaultdict(float)
        for name, confidence in name_classifications:
            name = name.lower()
            match = taxonomy_df[taxonomy_df["model_class"] == name]
            if not match.empty:
                taxon = match.iloc[0][fixed_column] # match it with the fixed column
                if pd.notna(taxon) and taxon != "":
                    fixed_level_map[taxon] += confidence
    
        # # print the predictions
        # # recursively sort dict by value (confidence) in descending order
        # sorted_data = dict(sorted(fixed_level_map.items(), key=lambda item: item[1], reverse=True))    
        # print(json.dumps(sorted_data, indent=4))

        # always return the highest confidence score
        top_label, top_conf = max(fixed_level_map.items(), key=lambda x: x[1])
        return {"pred": top_label, "conf": top_conf}

# run through json and convert detections to classficiations
def convert_detections_to_classification(json_path,
                                         img_dir,
                                         GPU_availability,
                                         cls_detec_thresh,
                                         cls_class_thresh,
                                         smooth_bool,
                                         crop_function,
                                         inference_function,
                                         cls_model_fpath,
                                         cls_tax_fallback,
                                         tax_mode,
                                         fixed_column,
                                         cls_taxonomy_df):

    # count the number of crops to classify
    n_crops_to_classify = 0
    with open(json_path) as image_recognition_file_content:
        data = json.load(image_recognition_file_content)
        label_map = fetch_label_map_from_json(json_path)
        for image in data['images']:
            if 'detections' in image:
                for detection in image['detections']:
                    conf = detection["conf"]
                    category_id = detection['category']
                    category = label_map[category_id]
                    if conf >= cls_detec_thresh and category == 'animal':
                        n_crops_to_classify += 1
    
    # send signal to catch error if there is nothing to classify
    if n_crops_to_classify == 0:
        print("n_crops_to_classify is zero. Nothing to classify.")
        return

    # crop and classify
    print(f"GPU available: {GPU_availability}")
    initial_it = True
    forbidden_classes = fetch_forbidden_classes(cls_model_fpath)
    with open(json_path) as image_recognition_file_content:
        data = json.load(image_recognition_file_content)
        label_map = fetch_label_map_from_json(json_path)
        if 'classification_categories' not in data:
            data['classification_categories'] = {}
        inverted_cls_label_map = {v: k for k, v in data['classification_categories'].items()}
        inverted_det_label_map = {v: k for k, v in data['detection_categories'].items()}
        with tqdm(total=n_crops_to_classify) as pbar:
            for image in data['images']:

                # loop
                fname = image['file']
                if 'detections' in image:
                    for detection in image['detections']:
                        conf = detection["conf"]
                        category_id = detection['category']
                        category = label_map[category_id]
                        if category == 'animal' and conf >= cls_detec_thresh:
                            img_fpath = os.path.join(img_dir, fname)
                            bbox = detection['bbox']
                            crop = crop_function(Image.open(img_fpath), bbox)
                            name_classifications = inference_function(crop)
                            name_classifications = remove_forbidden_classes(name_classifications, forbidden_classes)
                            
                            # fallback to hierarchical classification if the user specified this
                            if cls_tax_fallback:
                                hierarchical_prediction = hierarchical_classification(name_classifications, cls_taxonomy_df, threshold=cls_class_thresh, tax_mode=tax_mode, fixed_column=fixed_column)
                                name_classifications = [[hierarchical_prediction["pred"], hierarchical_prediction["conf"]]] # this overwrites the orginal predictions
                                initial_it = True # not sure why this is not always set to true, but it is needed for hierarchical prediction to always check if the name is already in the inverted_cls_label_map
                            
                            # check if name already in classification_categories
                            idx_classifications = []
                            for elem in name_classifications:
                                name = elem[0]
                                if initial_it:
                                    if name not in inverted_cls_label_map:
                                        highest_index = 0
                                        for key, value in inverted_cls_label_map.items():
                                            value = int(value)
                                            if value > highest_index:
                                                highest_index = value
                                        inverted_cls_label_map[name] = str(highest_index + 1)
                                idx_classifications.append([inverted_cls_label_map[name], round(elem[1], 5)])
                            initial_it = False

                            # sort
                            idx_classifications = sorted(idx_classifications, key=lambda x:x[1], reverse=True)
                            detection['classifications'] = idx_classifications

                            # update prgressbar
                            pbar.update(1)

    # write unaltered json for timelapse
    json_path_unaltered = os.path.splitext(json_path)[0] + "_original" + os.path.splitext(json_path)[1]
    data['classification_categories'] = {v: k for k, v in inverted_cls_label_map.items()}
    data['forbidden_classes'] = forbidden_classes
    with open(json_path_unaltered, "w") as json_file:
        json.dump(data, json_file, indent=1)

    # smooth results if user specified
    json_to_rewrite = json_path_unaltered
    if smooth_bool:
        print("<EA-status-change>smoothing<EA-status-change>")
        
        # image metadata will be read and sequences will be formed
        if json_path.endswith("image_recognition_file.json"):
            smooth_json_imgs(json_path_unaltered)
            json_to_rewrite = os.path.splitext(json_path)[0] + "_original" + os.path.splitext(json_path)[1]
    
        # for videos it is a bit easier as we already know the sequences
        # we just need average all predictions per video in the frames.json
        if json_path.endswith("video_recognition_file.frames.json"):
            smooth_json_video(json_path)

    # rewrite json to be used by AddaxAI
    with open(json_to_rewrite) as image_recognition_file_content:
        data = json.load(image_recognition_file_content)

        # fetch label maps
        cls_label_map = data['classification_categories']
        det_label_map = data['detection_categories']
        inverted_cls_label_map = {v: k for k, v in cls_label_map.items()}
        inverted_det_label_map = {v: k for k, v in det_label_map.items()}

        # add cls classes to det label map
        for k, v in inverted_cls_label_map.items():

            # only add if not already present in detection categories
            if k not in inverted_det_label_map:
                # if a model shares category names with MD, slightly modify it
                if k in ["animal", "person", "vehicle"]:
                    k += " "
                inverted_det_label_map[k] = str(len(inverted_det_label_map) + 1)

        # loop and adjust
        for image in data['images']:
            if 'detections' in image:
                for detection in image['detections']:
                    category_id = detection['category']
                    category = det_label_map[category_id]
                    if 'classifications' in detection:
                        highest_classification = detection['classifications'][0]
                        class_idx, class_conf = detection['classifications'][0]
                        if class_conf >= cls_class_thresh:
                            class_idx = highest_classification[0]
                            class_name = cls_label_map[class_idx]
                            detec_idx = inverted_det_label_map[class_name]
                            detection['prev_conf'] = detection["conf"]
                            detection['prev_category'] = detection['category']
                            detection["conf"] = highest_classification[1]
                            detection['category'] = str(detec_idx)

    # write json to be used by AddaxAI
    inverted_det_label_map['unidentified animal'] = inverted_det_label_map.pop('animal') # change all left over animals to unidentified
    data['detection_categories'] = {v: k for k, v in inverted_det_label_map.items()}
    with open(json_path, "w") as json_file:
        json.dump(data, json_file, indent=1)

# average predictions over multiple detections in images
# based on code written by Dan Morris' MegaDetector in /data_preparation/manage_local_batch.py
def smooth_json_imgs(json_input_fpath):

    # init vars
    filename_base = os.path.normpath(os.path.dirname(json_input_fpath))
    
    # read EXIF data from all images
    exif_options = read_exif.ReadExifOptions()
    exif_options.verbose = False
    exif_options.processing_library = 'pil'
    exif_options.byte_handling = 'delete'
    exif_results_file = os.path.join(filename_base,'exif_data.json')
    if os.path.isfile(exif_results_file):
        with open(exif_results_file,'r') as f:
            exif_results = json.load(f)
    else:
        exif_results = read_exif.read_exif_from_folder(filename_base,
                                                        output_file=exif_results_file,
                                                        options=exif_options)

    # prepare COCO-camera-traps-compatible image objects for EXIF results
    min_valid_timestamp_year = 1990
    now = datetime.datetime.now()
    image_info = []
    exif_datetime_tag = 'DateTimeOriginal'
    for exif_result in exif_results:

        # skip this if there is no exif data present
        if exif_result['exif_tags'] == {}:
            continue
      
        # collect info
        im = {}
        im['location'] = os.path.dirname(exif_result['file_name'])
        im['file_name'] = exif_result['file_name']
        im['id'] = im['file_name']
        
        # check if exif tags are present for this image
        if ('exif_tags' not in exif_result) or (exif_result['exif_tags'] is None) or \
            (exif_datetime_tag not in exif_result['exif_tags']): 
            exif_dt = None
        else:
            exif_dt = exif_result['exif_tags'][exif_datetime_tag]
            exif_dt = parse_exif_datetime_string(exif_dt)
        
        # if not present, do not add to image_info
        if exif_dt is None:
            print('<EA>Warning: no DateTimeOriginal in EXIF data for {}<EA>'.format(im['file_name']))

        # if present, check if valid
        else:
            dt = exif_dt
            
            # an image from the future (or within the last hour) is invalid, do not add to image_info
            if (now - dt).total_seconds() <= 1*60*60:
                print('<EA>Warning: an image from the future (or within the last hour) is invalid - datetime for {} is {}<EA>'.format(
                    im['file_name'], dt))
            
            # an image from before the dawn of time is also invalid, do not add to image_info
            elif dt.year < min_valid_timestamp_year:
                print('<EA>Warning: an image from before the dawn of time is also invalid - datetime for {} is {}<EA>'.format(
                    im['file_name'],dt))
            
            # only if valid and present, add to image_info
            else:
                im['datetime'] = dt
                image_info.append(im)

    # assemble into sequences
    dummy_stream = io.StringIO()
    with contextlib.redirect_stdout(dummy_stream):
        cct_json_utils.create_sequences(image_info)

    # make a list of images appearing at each location
    sequence_to_images = defaultdict(list)
    for im in image_info:
        sequence_to_images[im['seq_id']].append(im)
    all_sequences = list(sorted(sequence_to_images.keys()))

    # write to file
    sequence_level_smoothing_input_file = json_input_fpath
    with open(sequence_level_smoothing_input_file,'r') as f:
        d = json.load(f)

    # map each filename to classification results for that file
    filename_to_results = {}
    for im in d['images']:
        filename_to_results[im['file'].replace('\\','/')] = im
    
    # link the classifications to each image of the sequence
    def fetch_classifications_for_sequence(images_this_sequence):
        classifications_this_sequence = []
        for im in images_this_sequence:
            fn = im['file_name']
            results_this_image = filename_to_results.get(fn, {})
            detections = results_this_image.get('detections')
            if not detections:
                continue
            for det in detections:
                if det.get('category') == '1' and 'classifications' in det:
                    classifications_this_sequence.append(det['classifications'])
        return classifications_this_sequence

    # group and smooth averages for all detections in a sequence
    for _, seq_id in tqdm(enumerate(all_sequences),total=len(all_sequences)):
        images_this_sequence = sequence_to_images[seq_id]

        # link the classifications to the images of the sequence
        classifications_this_sequence = fetch_classifications_for_sequence(images_this_sequence)
        
        # group all confidences per class together in a list
        aggregated_confs = defaultdict(list)
        for conf_list in classifications_this_sequence:
            for cat_idx, conf in conf_list:
                aggregated_confs[cat_idx].append(conf)

        # calculate the average confidence per class
        smoothend_conf_list = []
        for cat_idx, conf_list in aggregated_confs.items():
            ave_conf = round(sum(conf_list) / len(conf_list), 5)
            smoothend_conf_list.append([cat_idx, ave_conf])

        # only take the highest smoothed confidence
        if smoothend_conf_list != []:
            smoothend_conf = [sorted(smoothend_conf_list, key=lambda x: x[1], reverse=True)[0]] if smoothend_conf_list != [] else [[]]

        # now we need to place these smoothed results back into the detections
        for im in images_this_sequence:
            fn = im['file_name']
            results_this_image = filename_to_results[fn]
            if "detections" in results_this_image:
                for detection in results_this_image['detections']:
                    if "classifications" in detection:
                        detection['classifications'] = smoothend_conf
    
    # remove exif.json if present
    exif_json = os.path.join(filename_base, "exif_data.json")
    if os.path.isfile(exif_json):
        os.remove(exif_json)
        
    # remove original json and replace with smoothed
    original_json = os.path.join(filename_base, "image_recognition_file_original.json")
    if os.path.isfile(original_json):
        os.remove(original_json)
    
    # write smoothed classification results as original
    with open(os.path.join(filename_base, original_json),'w') as f:
        json.dump(d, f, indent=1)

# for videos we don't need to read exif data because we will average the results per video
def smooth_json_video(json_path):
    
    # init vars
    json_path_frames = os.path.join(os.path.dirname(json_path), "video_recognition_file.frames_original.json")
    videos_dict = defaultdict(dict)

    # gather all confs per class and per video
    with open(json_path_frames, "r") as json_file:
        d = json.load(json_file)
    for im in d['images']:
        video_fn = os.path.dirname(im['file'])
        if 'detections' in im:
            for det in im['detections']:
                if 'classifications' in det:
                    for cat_idx, conf in det['classifications']:
                        if cat_idx not in videos_dict[video_fn]:
                            videos_dict[video_fn][cat_idx] = []
                        videos_dict[video_fn][cat_idx].append(conf)
    
    # smooth per video
    smoothed_confs_dict = defaultdict()
    for video, aggregated_confs in videos_dict.items():
        smoothend_conf_list = []
        
        # average each conf per class
        for cat_idx, conf_list in aggregated_confs.items():
            ave_conf = round(sum(conf_list) / len(conf_list), 5)
            smoothend_conf_list.append([cat_idx, ave_conf])

        # only take the highest smoothed confidence
        if smoothend_conf_list != []:
            smoothend_conf = [sorted(smoothend_conf_list, key=lambda x: x[1], reverse=True)[0]] if smoothend_conf_list != [] else [[]]
    
        # store in separate dict
        smoothed_confs_dict[video] = smoothend_conf

    # loop through json one last time to replace the classifications and remove unidentified animals
    for im in d['images']:
        video_fn = os.path.dirname(im['file'])
        if 'detections' in im:
            new_detections = []
            for det in im['detections']:
                if det['category'] == '1':
                    if 'classifications' in det:
                        
                        # because we select the best frame based on confidence, we want the frame with the clearest animal
                        # (i.e., the highest detection confidence) to have a slightly better confidence than the other frames
                        # therefore we take the weighted average of the two confidence scores
                        ave_conf = round(((smoothed_confs_dict[video_fn][0][1] * 29) + (det['conf'])) / 30, 5)
                        det['classifications'] = [[smoothed_confs_dict[video_fn][0][0], ave_conf]]
                        new_detections.append(det)
                        
                    # these are the unidentified animals that also have a frame that is good enough for a classification 
                    # there fore we can skip those since the other frames are better
                    elif video_fn in smoothed_confs_dict:
                        continue  
                    
                    # these are the unidentified animals without any frame that is good enough for a classification 
                    else:
                        new_detections.append(det)
                
                # these are persons and vehicles    
                else:
                    new_detections.append(det)
            
            # replace with the filtered detections
            im['detections'] = new_detections

    # write smoothed classification results
    with open(json_path_frames, 'w') as f:
        json.dump(d, f, indent = 1)     
    
