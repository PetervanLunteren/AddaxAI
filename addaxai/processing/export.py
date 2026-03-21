"""Data export utilities for AddaxAI.

CSV/XLSX/COCO format conversion and helper functions for recognition results.
"""

import datetime
import hashlib
import json
import math
from typing import Any, Dict, List


def clean_line(line: str) -> str:
    """Remove null bytes from a string."""
    return line.replace('\0', '')


def generate_unique_id(row: List[str]) -> str:
    """Generate an MD5 hash from a list of string values."""
    row_str = "".join(row).encode('utf-8')
    return hashlib.md5(row_str).hexdigest()


def format_datetime(date_str: str) -> str:
    """Convert 'DD/MM/YY HH:MM:SS' to 'YYYY-MM-DDTHH:MM:SS'.

    Returns 'NA' if the date string cannot be parsed.
    """
    try:
        dt = datetime.datetime.strptime(date_str, "%d/%m/%y %H:%M:%S")
        return dt.strftime("%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return 'NA'


def csv_to_coco(detections_df: Any, files_df: Any, output_path: str, version: str = "unknown") -> None:
    """Convert detection and file DataFrames to COCO JSON format.

    Args:
        detections_df: DataFrame with columns: relative_path, label,
            bbox_left, bbox_top, bbox_right, bbox_bottom.
        files_df: DataFrame with columns: relative_path, file_width,
            file_height, DateTimeOriginal.
        output_path: Path to write the COCO JSON output.
        version: AddaxAI version string for the info section.
    """
    coco: Dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [],
        "licenses": [{
            "id": 1,
            "name": "Unknown",
            "url": "NA",
        }],
        "info": {
            "description": f"Object detection results exported from AddaxAI (v{version}).",
            "url": "https://addaxdatascience.com/addaxai/",
            "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    category_mapping = {}
    current_category_id = 1

    for label in detections_df['label'].unique():
        if label not in category_mapping:
            category_mapping[label] = current_category_id
            coco['categories'].append({
                "id": current_category_id,
                "name": label,
            })
            current_category_id += 1

    annotation_id = 1
    for _, file_info in files_df.iterrows():
        image_id = len(coco['images']) + 1

        if isinstance(file_info['DateTimeOriginal'], float) and math.isnan(file_info['DateTimeOriginal']):
            date_captured = "NA"
        else:
            try:
                date_captured = datetime.datetime.strptime(
                    file_info['DateTimeOriginal'],
                    "%d/%m/%y %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
            except (ValueError, TypeError):
                date_captured = "NA"

        image_entry = {
            "id": image_id,
            "width": int(file_info['file_width']),
            "height": int(file_info['file_height']),
            "file_name": file_info['relative_path'],
            "license": 1,
            "date_captured": date_captured,
        }
        coco['images'].append(image_entry)

        image_detections = detections_df[
            detections_df['relative_path'] == file_info['relative_path']]
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
                "iscrowd": 0,
            }
            coco['annotations'].append(annotation_entry)
            annotation_id += 1

    with open(output_path, 'w') as f:
        json.dump(coco, f, indent=4)


# Column type mapping for pandas CSV import to control memory usage
CSV_DTYPES = {
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
