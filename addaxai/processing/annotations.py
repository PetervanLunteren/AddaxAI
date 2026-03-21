"""Annotation format conversion for AddaxAI.

Pascal VOC XML creation/parsing, COCO JSON conversion, and bbox utilities.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Recursively indent XML elements for pretty-printing.

    Modifies the element tree in place.
    """
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
        for elem in elem:
            indent_xml(elem, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def convert_bbox_pascal_to_yolo(
    size: Tuple[int, int],
    box: Tuple[float, float, float, float],
) -> Tuple[float, float, float, float]:
    """Convert a Pascal VOC bounding box to YOLO format.

    Args:
        size: (width, height) of the image.
        box: (xmin, xmax, ymin, ymax) in pixel coordinates.

    Returns:
        (x_center, y_center, width, height) normalized to [0, 1].
    """
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def return_xml_path(img_path: str, base_folder: str) -> str:
    """Return the corresponding XML annotation path with 'temp-folder' injected.

    Args:
        img_path: Absolute path to the image file.
        base_folder: Base folder for computing relative paths.

    Returns:
        Normalized path to the XML file inside temp-folder/.
    """
    tail_path = os.path.splitext(os.path.relpath(img_path, base_folder))
    temp_xml_path = os.path.join(base_folder, "temp-folder", tail_path[0] + ".xml")
    return os.path.normpath(temp_xml_path)


def convert_xml_to_coco(xml_path: str, inverted_label_map: Dict[str, str]) -> List[Any]:
    """Convert a Pascal VOC XML annotation file to COCO format.

    Args:
        xml_path: Path to the Pascal VOC XML file.
        inverted_label_map: Dict mapping class names to category IDs (strings).

    Returns:
        Tuple of (verified_image_dict, verification_status, new_class_flag,
                  updated_inverted_label_map).
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    try:
        verification_status = root.attrib['verified'] == 'yes'
    except KeyError:
        verification_status = False

    path = root.findtext('path')
    size = root.find('size')
    im_width = int(size.findtext('width'))
    im_height = int(size.findtext('height'))

    verified_detections = []
    new_class = False
    for obj in root.findall('object'):
        name = obj.findtext('name')

        if name not in inverted_label_map:
            new_class = True
            highest_index = 0
            for key, value in inverted_label_map.items():
                int_value = int(value)
                if int_value > highest_index:
                    highest_index = int_value
            inverted_label_map[name] = str(highest_index + 1)
        category = inverted_label_map[name]

        bndbox = obj.find('bndbox')
        xmin = int(float(bndbox.findtext('xmin')))
        ymin = int(float(bndbox.findtext('ymin')))
        xmax = int(float(bndbox.findtext('xmax')))
        ymax = int(float(bndbox.findtext('ymax')))

        w_box = round(abs(xmax - xmin) / im_width, 5)
        h_box = round(abs(ymax - ymin) / im_height, 5)
        xo = round(xmin / im_width, 5)
        yo = round(ymin / im_height, 5)
        bbox = [xo, yo, w_box, h_box]

        verified_detection = {
            'category': category,
            'conf': 1.0,
            'bbox': bbox,
        }
        verified_detections.append(verified_detection)

    verified_image = {
        'file': path,
        'detections': verified_detections,
    }

    return [verified_image, verification_status, new_class, inverted_label_map]


def create_pascal_voc_annotation(
    image_path: str,
    annotation_list: List[str],
    human_verified: bool,
    base_folder: str,
) -> None:
    """Create a Pascal VOC XML annotation file from a list of detections.

    Args:
        image_path: Path to the image file.
        annotation_list: List of annotation strings in format
            "xmin,ymin,xmin,ymin,xmax,ymax,xmax,ymax,conf,label".
        human_verified: Whether the annotations have been human-verified.
        base_folder: Base folder for computing the XML output path.
    """
    image_path_obj = Path(image_path)
    img = np.array(Image.open(image_path_obj).convert('RGB'))
    annotation = ET.Element('annotation')

    if human_verified:
        annotation.set('verified', 'yes')

    ET.SubElement(annotation, 'folder').text = str(image_path_obj.parent.name)
    ET.SubElement(annotation, 'filename').text = str(image_path_obj.name)
    ET.SubElement(annotation, 'path').text = str(image_path_obj)

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
        xmin, ymin, xmax, ymax = cords[0], cords[1], cords[4], cords[5]

        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = label
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'

        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    indent_xml(annotation)
    tree = ET.ElementTree(annotation)
    xml_file_name = return_xml_path(str(image_path), base_folder)
    Path(os.path.dirname(xml_file_name)).mkdir(parents=True, exist_ok=True)
    tree.write(xml_file_name)
