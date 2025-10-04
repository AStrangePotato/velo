# --- Video Configuration ---
VIDEO_INPUT_PATH = "input_video.mp4"
VIDEO_OUTPUT_PATH = "output/analysis_video.mp4"

# --- Preprocessing Configuration ---
RESIZE_WIDTH = 1280
RESIZE_HEIGHT = 720

# --- Detection and Tracking --- 
# Confidence threshold for detecting players or shuttlecock
DETECTION_CONFIDENCE = 0.75

# --- Analysis Configuration ---
# Conversion factor: pixels to meters (needs to be calibrated)
PIXELS_PER_METER = 40 # Example value

# --- Output Configuration ---
RESULTS_JSON_PATH = "output/match_analysis.json"
LOG_FILE_PATH = "output/processing.log"
