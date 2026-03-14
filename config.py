"""
Configuration for HelmetGuard AI
Centralised settings — edit this file to customise behaviour.
"""

# ===========================================================================
# MODEL  (hosted on Hugging Face)
# ===========================================================================
HF_REPO_ID = "KPN14/Yolov11_helmetguard"   # ← replace with your HF repo
# HF_MODEL_FILENAME = "best.pt"                  # filename inside the HF repo
LOCAL_MODEL_DIR = "weights"                    # local cache directory
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# ===========================================================================
# CLASS MAPPING (must match your YOLO training config)
# ===========================================================================
CLASS_NAMES = {
    0: "helmet",
    1: "license_plate",
    2: "cyclist",
}

# ===========================================================================
# VIOLATION ANALYSIS (rule-based heuristics)
# ===========================================================================
HELMET_CYCLIST_IOU_THRESHOLD = 0.3   # IoU between helmet & upper body
UPPER_BODY_RATIO = 0.4               # Top 40 % of cyclist bbox
MAX_CYCLIST_PLATE_DISTANCE = 200     # px – max distance to associate a plate

# ===========================================================================
# OCR
# ===========================================================================
OCR_LANGUAGE = "vi"
OCR_USE_ANGLE_CLASSIFICATION = True
PLATE_UPSCALE_FACTOR = 2
PLATE_CROP_PADDING = 5

# ===========================================================================
# OUTPUT / RECORDING
# ===========================================================================
OUTPUT_DIR = "outputs/violation_records"
SAVE_VIOLATIONS_DEFAULT = True

# ===========================================================================
# VIDEO
# ===========================================================================
VIDEO_FRAME_SKIP = 5
VIDEO_CODEC = "mp4v"

# ===========================================================================
# GRADIO
# ===========================================================================
GRADIO_SERVER_NAME = "0.0.0.0"
GRADIO_SERVER_PORT = 7860
GRADIO_SHARE = False

# ===========================================================================
# VISUALISATION (BGR)
# ===========================================================================
COLOR_CYCLIST_OK = (0, 255, 0)
COLOR_VIOLATION = (0, 0, 255)
COLOR_HELMET = (255, 255, 0)
COLOR_PLATE = (255, 0, 0)
BOX_THICKNESS = 2

# ===========================================================================
# DEBUG
# ===========================================================================
DEBUG_MODE = False
