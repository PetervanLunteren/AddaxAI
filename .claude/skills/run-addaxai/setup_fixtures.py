"""Create test fixtures for the Gundi upload harness:
   - test_data/test_gps.jpg  : a JPEG with GPS + DateTimeOriginal + Make/Model EXIF
   - test_data/image_recognition_file.json : MegaDetector-format results referencing it
"""
import os, json, datetime
from PIL import Image
import piexif

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "test_data")
os.makedirs(DATA, exist_ok=True)

IMG_NAME = "test_gps.jpg"
IMG_PATH = os.path.join(DATA, IMG_NAME)

# --- location (Serengeti-ish) and capture time ---
# Both are made UNIQUE per run so EarthRanger doesn't discard the test event as a
# duplicate: recorded_at is the current time and the GPS is jittered a little,
# exactly as a genuinely different photo would differ.
now = datetime.datetime.now()
DATETIME = now.strftime("%Y:%m:%d %H:%M:%S")   # EXIF format YYYY:MM:DD HH:MM:SS
_jitter = (now.hour * 3600 + now.minute * 60 + now.second) / 1_000_000.0  # up to ~0.086 deg
LAT, LON = round(-2.3333 + _jitter, 6), round(34.8333 + _jitter, 6)
ALT = 1500.0

def deg_to_dms_rational(dd):
    dd = abs(dd)
    d = int(dd)
    m = int((dd - d) * 60)
    s = round((dd - d - m/60) * 3600, 2)
    return ((d, 1), (m, 1), (int(s * 100), 100))

gps_ifd = {
    piexif.GPSIFD.GPSVersionID: (2, 3, 0, 0),
    piexif.GPSIFD.GPSLatitudeRef: "N" if LAT >= 0 else "S",
    piexif.GPSIFD.GPSLatitude: deg_to_dms_rational(LAT),
    piexif.GPSIFD.GPSLongitudeRef: "E" if LON >= 0 else "W",
    piexif.GPSIFD.GPSLongitude: deg_to_dms_rational(LON),
    piexif.GPSIFD.GPSAltitudeRef: 0,
    piexif.GPSIFD.GPSAltitude: (int(ALT * 100), 100),
}
zeroth_ifd = {
    piexif.ImageIFD.Make: "TestCam",
    piexif.ImageIFD.Model: "AddaxAI-Gundi-Test",
    piexif.ImageIFD.DateTime: DATETIME,
}
exif_ifd = {
    piexif.ExifIFD.DateTimeOriginal: DATETIME,
    piexif.ExifIFD.DateTimeDigitized: DATETIME,
}
exif_bytes = piexif.dump({"0th": zeroth_ifd, "Exif": exif_ifd, "GPS": gps_ifd, "1st": {}, "thumbnail": None})

# a simple 640x480 image with a "detection" box drawn for realism
img = Image.new("RGB", (640, 480), (90, 120, 80))
img.save(IMG_PATH, "jpeg", exif=exif_bytes)
print(f"wrote {IMG_PATH}")

# --- MegaDetector-format recognition file ---
# bbox format is [x_min, y_min, width, height] normalized 0..1
recognition = {
    "info": {"detector": "megadetector_v5a", "detection_completion_time": "2026-07-15 07:00:00"},
    "detection_categories": {"1": "animal", "2": "person", "3": "vehicle"},
    "classification_categories": {"0": "Panthera leo (lion)", "1": "Loxodonta africana (elephant)"},
    "images": [
        {
            "file": IMG_NAME,
            "manually_checked": True,
            "detections": [
                {
                    "category": "1",
                    "conf": 0.937,
                    "bbox": [0.31, 0.22, 0.4, 0.5],
                    "classifications": [["0", 0.882], ["1", 0.06]],
                }
            ],
        }
    ],
}
JSON_PATH = os.path.join(DATA, "image_recognition_file.json")
with open(JSON_PATH, "w") as f:
    json.dump(recognition, f, indent=2)
print(f"wrote {JSON_PATH}")
