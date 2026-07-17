"""Isolated harness for AddaxAI's Gundi upload.

Replicates the payload construction + POST logic of upload_to_gundi() in
AddaxAI_GUI.py (lines ~934-1043) so we can verify Gundi accepts our event
schema and image attachment WITHOUT installing the full app.

Targets the Gundi STAGE endpoint. Reads the API key from GUNDI_API_KEY env var.

Usage:
    GUNDI_API_KEY=xxxx  .../python test_gundi_upload.py
    (optional)  GUNDI_ENV=prod  to hit production instead of stage
"""
import os, sys, json, time, datetime
import requests
import PIL.Image, PIL.ExifTags
from GPSPhoto import gpsphoto

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "test_data")

# --- config (mirrors the constant added to AddaxAI_GUI.py) ---
GUNDI_BASE_URLS = {
    "prod": "https://sensors.api.gundiservice.org/v2",
    "stage": "https://sensors.api.stage.gundiservice.org/v2",
}
GUNDI_ENV = os.environ.get("GUNDI_ENV", "stage").lower()
gundi_base_url = GUNDI_BASE_URLS.get(GUNDI_ENV, GUNDI_BASE_URLS["stage"])

DRY_RUN = os.environ.get("GUNDI_DRY_RUN", "").strip() not in ("", "0", "false")
api_key = os.environ.get("GUNDI_API_KEY", "").strip()
if not api_key and not DRY_RUN:
    sys.exit("ERROR: set GUNDI_API_KEY env var to your Gundi API key (or GUNDI_DRY_RUN=1)")

# stand-ins for GUI globals used inside the payload
var_det_model = "MEGADETECTOR_5"
var_cls_model = "test-classifier"
current_AA_version = "6.37"

thresh = 0.2  # confidence threshold

# --- load fixtures (same layout the app reads) ---
src_dir = DATA
with open(os.path.join(src_dir, "image_recognition_file.json")) as f:
    data = json.load(f)

label_map = data.get("detection_categories", {})
cls_label_map = data.get("classification_categories", {})

uploadable = []
skipped_no_gps = 0
for image in data["images"]:
    file = image["file"]
    filepath = os.path.join(src_dir, file)
    if not os.path.isfile(filepath):
        continue
    try:
        gps = gpsphoto.getGPSData(filepath)
    except Exception:
        gps = {}
    if "Latitude" not in gps or "Longitude" not in gps:
        skipped_no_gps += 1
        continue
    try:
        img_for_exif = PIL.Image.open(filepath)
        metadata = {PIL.ExifTags.TAGS[k]: v for k, v in img_for_exif._getexif().items()
                    if k in PIL.ExifTags.TAGS}
        img_for_exif.close()
    except Exception:
        metadata = {}
    manually_checked = image.get("manually_checked", False)
    for detection in image.get("detections", []):
        conf = detection["conf"]
        if conf < thresh:
            continue
        cat_id = detection["category"]
        if "classifications" in detection and len(detection["classifications"]) > 0:
            cls_id = detection["classifications"][0][0]
            label = cls_label_map.get(cls_id, label_map.get(cat_id, "unknown"))
        else:
            label = label_map.get(cat_id, "unknown")
        uploadable.append({
            "filepath": filepath, "file": file, "label": label, "conf": conf,
            "detection": detection, "gps": gps, "metadata": metadata,
            "manually_checked": manually_checked,
        })

print(f"endpoint         : {gundi_base_url}")
print(f"uploadable dets  : {len(uploadable)}  (skipped no-GPS: {skipped_no_gps})\n")

headers_json = {"apikey": api_key, "Content-Type": "application/json"}
headers_file = {"apikey": api_key}
errors = []

for i, item in enumerate(uploadable):
    # timestamp from EXIF (same fallback order as the app)
    iso_timestamp = ""
    for dt_key in ["DateTimeOriginal", "DateTime", "DateTimeDigitized"]:
        try:
            raw = str(item["metadata"][dt_key])
            dt = datetime.datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
            iso_timestamp = dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            break
        except Exception:
            continue
    if not iso_timestamp:
        iso_timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    det = item["detection"]
    bbox_raw = det.get("bbox", [0, 0, 0, 0])
    verified_str = " (verified)" if item["manually_checked"] else ""
    event_payload = {
        "title": f"{item['label']} detected ({int(item['conf'] * 100)}% confidence){verified_str}",
        "event_type": "wildlife_observation",
        "recorded_at": iso_timestamp,
        "location": {"lat": item["gps"]["Latitude"], "lon": item["gps"]["Longitude"]},
        "status": "new",
        "source": "AddaxAI",
        "event_details": {
            "species": item["label"],
            "confidence": round(item["conf"], 4),
            "camera_make": str(item["metadata"].get("Make", "")),
            "camera_model": str(item["metadata"].get("Model", "")),
            "detection_model": var_det_model,
            "classification_model": var_cls_model,
            "human_verified": item["manually_checked"],
            "altitude": item["gps"].get("Altitude", ""),
            "bbox": bbox_raw,
            "addaxai_version": current_AA_version,
            "image_filename": os.path.basename(item["file"]),
        },
    }
    print(f"[{i+1}/{len(uploadable)}] POST event -> {gundi_base_url}/events/")
    print("  payload:", json.dumps(event_payload, indent=2))

    if DRY_RUN:
        print("  (dry run — not sending)\n")
        continue

    object_id = None
    try:
        resp = requests.post(f"{gundi_base_url}/events/", json=event_payload, headers=headers_json, timeout=30)
        print(f"  -> HTTP {resp.status_code}: {resp.text[:400]}")
        if resp.status_code in (200, 201):
            object_id = resp.json().get("object_id")
        else:
            errors.append((item["file"], f"Event creation failed: HTTP {resp.status_code} - {resp.text[:200]}"))
    except Exception as e:
        errors.append((item["file"], f"Event creation failed: {str(e)[:200]}"))

    if object_id:
        print(f"  object_id: {object_id}")
        try:
            with open(item["filepath"], "rb") as photo:
                resp_att = requests.post(f"{gundi_base_url}/events/{object_id}/attachments/",
                                         files={"file1": photo}, headers=headers_file, timeout=60)
            print(f"  attachment -> HTTP {resp_att.status_code}: {resp_att.text[:300]}")
            if resp_att.status_code not in (200, 201):
                errors.append((item["file"], f"Attachment upload failed: HTTP {resp_att.status_code} - {resp_att.text[:200]}"))
        except Exception as e:
            errors.append((item["file"], f"Attachment upload failed: {str(e)[:200]}"))
    print()

print("=" * 60)
if errors:
    print(f"RESULT: {len(errors)} error(s):")
    for fpath, err in errors:
        print(f"  - {fpath}: {err}")
    sys.exit(1)
else:
    print("RESULT: all events + attachments uploaded successfully ✓")
