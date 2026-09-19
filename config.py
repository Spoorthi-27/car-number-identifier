"""
config.py
---------
Central place to tweak the pipeline without hunting through the code.
"""

# --- Camera / video ---
CAMERA_INDEX = 0          # 0 = default webcam. Change to 1/2 if you have multiple cameras.
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- YOLO plate detector ---
# Pretrained license-plate detection weights, auto-downloaded from Hugging Face
# on first run (via the `yolov5` package). No training required to get started.
MODEL_NAME = "keremberke/yolov5s-license-plate"
CONF_THRESHOLD = 0.55       # minimum detection confidence to keep a box
IOU_THRESHOLD = 0.45       # non-max-suppression IoU threshold
MAX_DETECTIONS = 10        # max plates considered per frame

# --- OCR / plate-text filtering ---
MIN_PLATE_TEXT_LENGTH = 4  # OCR reads shorter than this are treated as "low confidence"

# --- Database / logging ---
DB_PATH = "database/plates.db"
DEDUPE_SECONDS = 5         # don't log the same plate text again within this many seconds
SAVE_SNAPSHOTS = True      # save a cropped plate image alongside each DB entry
SNAPSHOT_DIR = "output/snapshots"
