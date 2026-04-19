import cv2
import numpy as np
import mediapipe as mp
import os

# Constants based on the Baseball Batter Detection Implementation Plan
_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "efficientdet_lite0.tflite"))
_MIN_BATTER_HEIGHT = 500
_MAX_AREA_PCT = 0.40
_CLAY_OVERLAP_THRESHOLD = 0.01
_CENTER_PREFERENCE_PX = 300

# Lazy-loaded MediaPipe instances
_object_detector = None
_pose_landmarker = None

def _get_detector():
    """Initializes the MediaPipe ObjectDetector only when needed."""
    global _object_detector
    if _object_detector is None:
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(f"MediaPipe model file not found at: {_MODEL_PATH}")
            
        with open(_MODEL_PATH, "rb") as f:
            model_content = f.read()
        options = mp.tasks.vision.ObjectDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=model_content),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            max_results=20,
            score_threshold=0.5,
        )
        _object_detector = mp.tasks.vision.ObjectDetector.create_from_options(options)
    return _object_detector

def get_clay_mask(frame):
    """Stage 1: Detect clay area near home plate using HSV and morphology."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1.1 HSV threshold: Hue < 25 OR >= 170, Sat >= 30, Val >= 50
    mask1 = cv2.inRange(hsv, np.array([0, 30, 50]), np.array([25, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, 30, 50]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    # 1.2 Morphological cleanup
    # Open 15x15 ellipse (removes noise); Close 30x30 ellipse (fills internal gaps)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 1.3 Largest connected component in bottom region
    def extract_largest_in_roi(m, y_start):
        roi = m[y_start:, :]
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(roi, 8)
        if num_labels <= 1:
            return None
        
        largest_label = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        res = np.zeros_like(m)
        res[y_start:][labels == largest_label] = 255
        return res

    # Try bottom 1/3; fallback to bottom 1/2 if nothing found
    result = extract_largest_in_roi(mask, int(h * 2/3))
    if result is None:
        result = extract_largest_in_roi(mask, int(h * 1/2))
    
    return result if result is not None else np.zeros((h, w), dtype=np.uint8)

def detect_batter(frame, clay_mask):
    """Stages 2 & 3: Detect and filter person candidates to isolate the batter."""
    h, w = frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    detector = _get_detector()
    detection_result = detector.detect(mp_image)
    
    candidates = []
    for detection in detection_result.detections:
        category = detection.categories[0]
        if category.category_name != "person":
            continue
            
        bbox = detection.bounding_box
        bx, by, bw, bh = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
        
        # 3.1 Filtering Criteria
        if bh < _MIN_BATTER_HEIGHT:
            continue
        if (bw * bh) / (w * h) >= _MAX_AREA_PCT:
            continue
            
        # Bottom 25% of Bbox (feet) must overlap with the clay mask
        q_h = bh // 4
        bottom_roi = clay_mask[max(0, by + 3*q_h) : min(h, by + bh), max(0, bx) : min(w, bx + bw)]
        if bottom_roi.size == 0 or (np.count_nonzero(bottom_roi) / bottom_roi.size) < _CLAY_OVERLAP_THRESHOLD:
            continue
            
        candidates.append({
            'bbox': (bx, by, bw, bh),
            'cx': bx + bw // 2,
            'score': category.score
        })
        
    if not candidates:
        return None, None

    # 3.2 Tie-breaking: Center Preference
    img_cx = w // 2
    candidates.sort(key=lambda c: abs(c['cx'] - img_cx))
    
    return candidates[0]['bbox'], "BATTER"

def get_swing_box(frame_shape, batter_bbox):
    """Stage 4.2: Calculate the expanded swing bounding box."""
    img_h, img_w = frame_shape[:2]
    bx, by, bw, bh = batter_bbox
    
    pad_x1 = 50
    pad_x2 = 650
    pad_y = int(bh * 0.1)
    
    # Expansion: Left 10%, Top 5%, Bottom 5%, Right 33% (to capture the follow-through)
    sx = max(0, bx - pad_x1)
    sy = max(0, by - pad_y)
    sx2 = min(img_w, bx + pad_x2)
    sy2 = min(img_h, by + bh + pad_y)
    
    return (sx, sy, sx2 - sx, sy2 - sy)

if __name__ == "__main__":
    # Unit test logic would go here to verify detection on a sample frame
    pass