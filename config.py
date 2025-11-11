"""
Configuration file for Helmet Violation Detection System
Modify these settings to customize the system behavior
"""

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Path to your trained YOLOv11 model
MODEL_PATH = 'best.pt'

# Confidence threshold for object detection (0.0 to 1.0)
# Lower values detect more objects but may include false positives
# Higher values are more strict but may miss some objects
CONFIDENCE_THRESHOLD = 0.5

# IOU threshold for Non-Maximum Suppression (0.0 to 1.0)
IOU_THRESHOLD = 0.45

# ============================================================================
# CLASS MAPPING
# ============================================================================

# Map model class IDs to names
# Update these if your model uses different class IDs
CLASS_NAMES = {
    0: 'helmet',
    1: 'cyclist',
    2: 'license_plate'
}

# ============================================================================
# VIOLATION DETECTION SETTINGS
# ============================================================================

# IOU threshold for determining if a helmet belongs to a cyclist
# A helmet must overlap this much with the cyclist's upper body region
HELMET_CYCLIST_IOU_THRESHOLD = 0.3

# Percentage of cyclist's body to consider as "upper body" for helmet detection
# 0.4 means top 40% of the cyclist's bounding box
UPPER_BODY_RATIO = 0.4

# Maximum distance (in pixels) between cyclist and license plate to associate them
MAX_CYCLIST_PLATE_DISTANCE = 300

# ============================================================================
# OCR SETTINGS
# ============================================================================

# PaddleOCR language setting
# Options: 'en', 'ch', 'japan', 'korean', 'fr', 'german', etc.
OCR_LANGUAGE = 'en'

# Use angle classification for text detection
OCR_USE_ANGLE_CLASSIFICATION = True

# Show OCR processing logs
OCR_SHOW_LOG = False

# License plate image upscaling factor for better OCR
# Higher values may improve accuracy but increase processing time
PLATE_UPSCALE_FACTOR = 2

# Padding around license plate crop (in pixels)
PLATE_CROP_PADDING = 5

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Directory to save violation records
OUTPUT_DIR = 'violation_records'

# Save violation records by default
SAVE_VIOLATIONS_DEFAULT = True

# Image quality for saved violation images (1-100)
SAVED_IMAGE_QUALITY = 95

# ============================================================================
# VIDEO PROCESSING SETTINGS
# ============================================================================

# Process every Nth frame in videos (higher = faster but may miss violations)
# 1 = process every frame, 5 = process every 5th frame
VIDEO_FRAME_SKIP = 5

# Output video codec
# Options: 'mp4v', 'XVID', 'H264', 'avc1'
VIDEO_CODEC = 'mp4v'

# ============================================================================
# GRADIO INTERFACE SETTINGS
# ============================================================================

# Server settings
GRADIO_SERVER_NAME = "0.0.0.0"  # 0.0.0.0 allows external access
GRADIO_SERVER_PORT = 7860

# Share link (set to True to create a public link)
GRADIO_SHARE = False

# Authentication (set username and password for access control)
# Leave as None for no authentication
GRADIO_AUTH = None  # Example: ("admin", "password123")

# Maximum file upload size (in MB)
MAX_IMAGE_SIZE_MB = 10
MAX_VIDEO_SIZE_MB = 100

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Colors for bounding boxes (BGR format)
COLOR_CYCLIST_OK = (0, 255, 0)        # Green
COLOR_VIOLATION = (0, 0, 255)          # Red
COLOR_HELMET = (255, 255, 0)           # Yellow
COLOR_PLATE = (255, 0, 0)              # Blue

# Box thickness (in pixels)
BOX_THICKNESS = 2

# Font settings
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
FONT_THICKNESS = 2

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Use GPU if available (requires CUDA-enabled PyTorch)
USE_GPU = True

# Number of threads for CPU processing
CPU_THREADS = 4

# Batch size for processing multiple images
BATCH_SIZE = 1

# ============================================================================
# DEBUG SETTINGS
# ============================================================================

# Enable debug mode (prints more information)
DEBUG_MODE = False

# Save intermediate processing images
SAVE_DEBUG_IMAGES = False

# Debug output directory
DEBUG_OUTPUT_DIR = 'debug_output'
