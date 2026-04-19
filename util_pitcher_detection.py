import cv2
import numpy as np
import mediapipe as mp
import logging
import os

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configuration Constants
_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "efficientdet_lite0.tflite"))
_OD_MAX_RESULTS = 3
_OD_MIN_DETECTION_CONFIDENCE = 0.7
_PITCHER_ROI_MAX_HEIGHT = 1080   # Max height (in pixels) above the mound to look for a pitcher
_PITCHER_ROI_MIN_HEIGHT = 700   # Min height (in pixels) above the mound to look for a pitcher
_PITCHER_ROI_HALF_WIDTH = 400    # Horizontal reach from the mound center

_object_detector = None

def _get_object_detector() -> mp.tasks.vision.ObjectDetector:
    """Return a lazily-initialised MediaPipe ObjectDetector."""
    global _object_detector

    if _object_detector is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"MediaPipe model file not found at: {_MODEL_PATH}")

        with open(_MODEL_PATH, "rb") as f:
            model_content = f.read()
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_content),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            max_results=_OD_MAX_RESULTS,
            score_threshold=_OD_MIN_DETECTION_CONFIDENCE,
        )
        _object_detector = mp.tasks.vision.ObjectDetector.create_from_options(options)
    
    return _object_detector

def detect_pitcher_above_mound(
    frame_bgr: np.ndarray,
    mound_cx: float,
    mound_cy: float,
    roi_height: int = _PITCHER_ROI_MAX_HEIGHT,
) -> list[tuple[int, int, int, int, float]]:
    """
    Detect a pitcher standing above the mound using MediaPipe ObjectDetector.

    Extracts an ROI extending up to *roi_height* pixels above the mound centroid 
    and runs object detection on it. Only detections with category "person" are kept.

    Returns a list of (x, y, w, h, confidence) tuples in frame coordinates.
    """
    h, w = frame_bgr.shape[:2]

    # Calculate ROI bounds
    roi_y_top = max(0, int(mound_cy) - roi_height)
    roi_y_bot = max(0, int(mound_cy) + 50) # Include a small buffer below the mound center
    roi_x_left = max(0, int(mound_cx) - _PITCHER_ROI_HALF_WIDTH)
    roi_x_right = min(w, int(mound_cx) + _PITCHER_ROI_HALF_WIDTH)

    if roi_y_bot <= roi_y_top or roi_x_right <= roi_x_left:
        return []

    roi = frame_bgr[roi_y_top:roi_y_bot, roi_x_left:roi_x_right]

    # Very tiny ROIs produce garbage results.
    if roi.shape[0] < 64 or roi.shape[1] < 64:
        return []

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)

    detector = _get_object_detector()
    detection_result = detector.detect(mp_image)

    frame_rects = []
    for detection in detection_result.detections:
        category = detection.categories[0]
        if category.category_name == "person":
            bbox = detection.bounding_box
            
            # Filtering by height to avoid detecting small background artifacts
            # adjusted for potential 4K resolution
            if bbox.height > _PITCHER_ROI_MIN_HEIGHT: 
                confidence = category.score
                # Convert ROI-local coordinates to global frame coordinates
                frame_rects.append((
                    int(bbox.origin_x + roi_x_left),
                    int(bbox.origin_y + roi_y_top),
                    int(bbox.width),
                    int(bbox.height),
                    float(confidence)
                ))

    if frame_rects:
        logger.debug("Pitcher detected above mound (%d person(s))", len(frame_rects))
    else:
        logger.debug("No pitcher detected above mound")

    return frame_rects