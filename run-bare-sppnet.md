
# ==============================
# RUN SPECIESNET OFFICIAL API
# ==============================

DIR='/Users/peter/Downloads/example-data/project_Kenya/Chui River/deployment_001'
COUNTRY='KEN'

~/AddaxAI/envs/env-addaxai-base/bin/python -m megadetector.detection.run_detector_batch ~/AddaxAI/models/det/MD5A-0-0/md_v5a.0.0.pt "$DIR" "$DIR/MD_ground_truth.json" --recursive --include_image_size 

~/AddaxAI/envs/env-addaxai-base/bin/python -m megadetector.detection.run_md_and_speciesnet "$DIR" "$DIR/SPPNET_ground_truth.json" --detections_file "$DIR/MD_ground_truth.json" --classification_model ~/AddaxAI/models/cls/SPECIESNET-v4-0-1-A-v1 --country "$COUNTRY"

# ==============================
# RUN SPECIESNET ADDAXAI
# ==============================

# run in Addax on the same DIR, same country, and smoothing OFF
# then note projectID uuid

PROJECTUUID='8edb77c3-aa60-4073-a7b8-e291bfdd4a80'

# ==============================
# COMPARE
# ==============================
# Add --verbose to see every individual difference

cd backend
source venv/bin/activate

python scripts/compare_speciesnet.py \
--gt "$DIR/SPPNET_ground_truth.json" \
--project-id $PROJECTUUID