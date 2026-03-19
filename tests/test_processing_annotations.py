"""Tests for addaxai.processing.annotations — annotation format conversion."""

import os
import xml.etree.ElementTree as ET

import pytest
from PIL import Image

from addaxai.processing.annotations import (
    convert_bbox_pascal_to_yolo,
    convert_xml_to_coco,
    create_pascal_voc_annotation,
    indent_xml,
    return_xml_path,
)


# --- indent_xml ---

def test_indent_xml_adds_whitespace():
    root = ET.Element("root")
    child = ET.SubElement(root, "child")
    child.text = "value"
    indent_xml(root)
    xml_str = ET.tostring(root, encoding="unicode")
    assert "\n" in xml_str


# --- convert_bbox_pascal_to_yolo ---

def test_convert_bbox_pascal_to_yolo_basic():
    # Image 100x200, box from (10,20) to (50,80)
    size = (100, 200)
    box = (10, 50, 20, 80)  # xmin, xmax, ymin, ymax
    x, y, w, h = convert_bbox_pascal_to_yolo(size, box)
    assert 0 <= x <= 1
    assert 0 <= y <= 1
    assert 0 < w <= 1
    assert 0 < h <= 1


# --- return_xml_path ---

def test_return_xml_path_injects_temp_folder(tmp_path):
    base = str(tmp_path)
    img_path = os.path.join(base, "subdir", "photo.jpg")
    result = return_xml_path(img_path, base)
    assert "temp-folder" in result
    assert result.endswith(".xml")
    assert "photo" in result


# --- convert_xml_to_coco ---

def _make_voc_xml(path, width=640, height=480, objects=None, verified=False):
    """Helper: create a minimal Pascal VOC XML file."""
    annotation = ET.Element("annotation")
    if verified:
        annotation.set("verified", "yes")
    ET.SubElement(annotation, "folder").text = "test"
    ET.SubElement(annotation, "filename").text = os.path.basename(str(path))
    ET.SubElement(annotation, "path").text = str(path).replace(".xml", ".jpg")
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    if objects is None:
        objects = [("animal", 10, 20, 100, 200)]
    for name, xmin, ymin, xmax, ymax in objects:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = name
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)
    tree = ET.ElementTree(annotation)
    tree.write(str(path))


def test_convert_xml_to_coco_basic(tmp_path):
    xml_path = tmp_path / "test.xml"
    _make_voc_xml(xml_path, objects=[("deer", 10, 20, 100, 200)])
    label_map = {"deer": "1"}

    result, verified, new_class, updated_map = convert_xml_to_coco(
        str(xml_path), label_map)

    assert verified is False
    assert new_class is False
    assert len(result["detections"]) == 1
    assert result["detections"][0]["category"] == "1"
    assert len(result["detections"][0]["bbox"]) == 4


def test_convert_xml_to_coco_new_class(tmp_path):
    xml_path = tmp_path / "test.xml"
    _make_voc_xml(xml_path, objects=[("bear", 10, 20, 100, 200)])
    label_map = {"deer": "1"}

    result, verified, new_class, updated_map = convert_xml_to_coco(
        str(xml_path), label_map)

    assert new_class is True
    assert "bear" in updated_map
    assert int(updated_map["bear"]) > 1


def test_convert_xml_to_coco_verified(tmp_path):
    xml_path = tmp_path / "test.xml"
    _make_voc_xml(xml_path, verified=True,
                  objects=[("deer", 10, 20, 100, 200)])
    label_map = {"deer": "1"}

    result, verified, new_class, updated_map = convert_xml_to_coco(
        str(xml_path), label_map)

    assert verified is True


# --- create_pascal_voc_annotation ---

def test_create_pascal_voc_annotation_writes_xml(tmp_path):
    # Create a test image
    img_path = tmp_path / "subdir" / "test.jpg"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (100, 100)).save(str(img_path))

    # Annotation list format: "xmin,ymin,xmin,ymin,xmax,ymax,xmax,ymax,conf,label"
    annotations = ["10,20,10,20,80,90,80,90,0.95,deer"]

    create_pascal_voc_annotation(str(img_path), annotations,
                                human_verified=False,
                                base_folder=str(tmp_path))

    # Check that XML was created in temp-folder
    expected_xml = os.path.join(str(tmp_path), "temp-folder", "subdir", "test.xml")
    assert os.path.exists(expected_xml)

    # Parse and verify content
    tree = ET.parse(expected_xml)
    root = tree.getroot()
    assert root.find("filename").text == "test.jpg"
    assert root.find(".//object/name").text == "deer"


def test_create_pascal_voc_annotation_verified(tmp_path):
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (50, 50)).save(str(img_path))

    annotations = ["5,5,5,5,40,40,40,40,0.9,cat"]
    create_pascal_voc_annotation(str(img_path), annotations,
                                human_verified=True,
                                base_folder=str(tmp_path))

    expected_xml = os.path.join(str(tmp_path), "temp-folder", "test.xml")
    tree = ET.parse(expected_xml)
    root = tree.getroot()
    assert root.attrib.get("verified") == "yes"
