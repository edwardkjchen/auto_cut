import cv2
import numpy as np
import argparse
import os
import glob
from util_pitcher_detection import detect_pitcher_above_mound

_SAVE_MASKS = True
TEST_VIDEO = "20260411_pitching/processed/260411121252.MOV"

def detect_mound(video_path):
    """
    Detects the pitcher's mound in 4K video using HSV thresholding and gap detection.
    Returns (mound_cx, mound_cy) if a pitcher is detected, else None. """
    # 1. Open a MOV 4K video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # 2. Take 5 frames evenly distributed throughout the video sequence and analyze each frame separately.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(0, total_frames - 1), 5, dtype=int)

    sampled_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            sampled_frames.append(frame)

    if not sampled_frames:
        print("Error: Could not read frames from video.")
        cap.release()
        return

    # Save the first sampled frame for the final display
    first_frame = sampled_frames[0].copy()
    h, w = first_frame.shape[:2]
    w_small, h_small = w // 2, h // 2

    roi_start_y = int(h_small * 0.60)
    roi_end_y = int(h_small * 0.90)

    lower_brown1 = np.array([0, 0, 0])
    upper_brown1 = np.array([25, 255, 255])
    lower_brown2 = np.array([170, 0, 0])
    upper_brown2 = np.array([180, 255, 255])

    kernel_open = np.ones((8, 8), np.uint8)
    kernel_close = np.ones((15, 15), np.uint8)

    combined_mask = np.zeros((h_small, w_small), dtype=np.uint8)

    for frame in sampled_frames:
        resized = cv2.resize(frame, (w_small, h_small))
        roi = resized[roi_start_y:roi_end_y, :]

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv_roi, lower_brown1, upper_brown1)
        mask2 = cv2.inRange(hsv_roi, lower_brown2, upper_brown2)
        mask = cv2.bitwise_or(mask1, mask2)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            largest_label = np.argmax(areas) + 1
            selected_blob = np.zeros_like(mask)
            selected_blob[labels == largest_label] = 255
            combined_mask[roi_start_y:roi_end_y, :] = cv2.bitwise_or(combined_mask[roi_start_y:roi_end_y, :], selected_blob)

    # 3. Use the combined blob mask across sampled frames to find the final mound region.
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined_mask, connectivity=8)

    mound_cx, mound_cy = None, None
    mask = np.zeros_like(combined_mask)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        cx_roi, cy_roi = centroids[largest_label]
        mound_cx = int(cx_roi * (w / w_small))
        mound_cy = int(cy_roi * (h / h_small))
        mask[labels == largest_label] = 255

    # 8. Once it is blob is detected, overlay the full mask over the first frame.
    full_mask_small = mask
    full_mask = cv2.resize(full_mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create an overlay (Green mask)
    overlay = first_frame.copy()
    overlay[full_mask > 0] = [255, 0, 0]
    result = cv2.addWeighted(first_frame, 0.7, overlay, 0.3, 0)

    detected_coords = None
    if mound_cx is not None:
        print(f"\nAnalysis for: {video_path}")
        print(f"Mound Centroid: x={mound_cx}, y={mound_cy}")

        # Draw a red asterisk (*) at the mound centroid
        cv2.putText(result, "*", (mound_cx - 20, mound_cy + 25), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)

        # Use pitcher detector on the first frame using the calculated mound centroid
        pitcher_rects = detect_pitcher_above_mound(first_frame, float(mound_cx), float(mound_cy))

        for (px, py, pw, ph, conf) in pitcher_rects:
            print(f"Pitcher Bbox: min_x={px}, max_x={px + pw}")
            # Draw bounding box and confidence score on the original frame scale
            cv2.rectangle(result, (px, py), (px + pw, py + ph), (0, 255, 0), 5)
            label = f"Pitcher: {conf:.2f}"
            cv2.putText(result, label, (px, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        # Adjust centroid_x so that it is between min_x and max_x of the first detected pitcher
        if pitcher_rects:
            p_min_x = pitcher_rects[0][0]
            p_max_x = p_min_x + pitcher_rects[0][2]
            mound_cx = max(p_min_x, min(mound_cx, p_max_x))
            print(f"Adjusted Centroid X: {mound_cx}")

            # Store for returning to caller
            detected_coords = (mound_cx, mound_cy)

            # Draw the target 1920x1080 crop bounding box
            cv2.rectangle(result, (mound_cx - 485, mound_cy - 1025), (mound_cx - 485 + 1920, mound_cy - 1025 + 1080), (255, 255, 0), 5)

    # Save the result to a PNG file instead of showing it
    if _SAVE_MASKS:
        output_filename = os.path.splitext(os.path.basename(video_path))[0] + "_mound_detect.png"
        cv2.imwrite(output_filename, result)
        print(f"Mound detection complete. Result saved to {output_filename}")
    cap.release()
    return detected_coords

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect pitcher mound in 4K video.")
    parser.add_argument("--video", type=str, default=None, help="Path to the input 4K MOV file")
    args = parser.parse_args()
    
    if args.video:
        detect_mound(args.video)
    else:
        processed_dir = "20260417_pitching"
        mov_files = sorted(glob.glob(os.path.join(processed_dir, "*.MOV")))
        
        if not mov_files:
            print(f"No MOV files found in {processed_dir}")
        else:
            for i in range(0, len(mov_files), 5):
                detect_mound(mov_files[i])