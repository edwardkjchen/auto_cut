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

    # 2. Take 5 frames evenly distributed throughout the video sequence and average them
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Generate 5 indices ranging from the first frame to the last
    indices = np.linspace(0, max(0, total_frames - 1), 5, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame.astype(np.float32))

    if not frames:
        print("Error: Could not read frames from video.")
        cap.release()
        return

    # Save the first frame for the final display
    first_frame = frames[0].astype(np.uint8)
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)

    # 3. Reduce its resolution by 2x
    h, w = avg_frame.shape[:2]
    resized = cv2.resize(avg_frame, (w // 2, h // 2))
    h_small, w_small = resized.shape[:2]

    # 4. Only examines the region from 60% to 90% of the frame height
    roi_start_y = int(h_small * 0.60)
    roi_end_y = int(h_small * 0.90)
    roi = resized[roi_start_y:roi_end_y, :]

    # 5. Identifies brown/clay pixels by HSV:
    # Hue <= 25 or h >= 170. Saturation >= 30, V >= 50
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    #lower_brown1 = np.array([0, 30, 50])
    #pper_brown1 = np.array([25, 255, 255])
    #lower_brown2 = np.array([170, 30, 50])
    #upper_brown2 = np.array([180, 255, 255])

    lower_brown1 = np.array([0, 0, 0])
    upper_brown1 = np.array([25, 255, 255])
    lower_brown2 = np.array([170, 0, 0])
    upper_brown2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv_roi, lower_brown1, upper_brown1)
    mask2 = cv2.inRange(hsv_roi, lower_brown2, upper_brown2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 6. Morphological Cleanup:
    # MORPH_OPEN (8x8 kernel): removes small noise specks
    # MORPH_CLOSE (15x15 kernel): fills small internal gaps
    kernel_open = np.ones((8, 8), np.uint8)
    kernel_close = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    # 7. Detection Strategy: Largest Connected Blob
    # Heuristic Gap detection often fails if the grass is not clean or if shadows connect dirt areas.
    # Selecting the largest blob is more robust for isolating the mound in 4K sequences.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    mound_cx, mound_cy = None, None
    if num_labels > 1:
        # stats: [left, top, width, height, area]
        # Filter out the background (label 0) and find the largest remaining blob
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(areas) + 1
        cx_roi, cy_roi = centroids[largest_label]
        mound_cx = int(cx_roi * (w / w_small))
        mound_cy = int((cy_roi + roi_start_y) * (h / h_small))
        mask = np.zeros_like(mask)
        mask[labels == largest_label] = 255

    # 8. Once it is blob is detect, please show the blob mask over the first frame.
    # Reconstruct the full mask for the original resolution
    full_mask_small = np.zeros((h_small, w_small), dtype=np.uint8)
    full_mask_small[roi_start_y:roi_end_y, :] = mask
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