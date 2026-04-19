"""
This module provides functionality to calculate the scale ratio between two videos
of a pitcher. It works by identifying a static phase before the main motion and
measuring the distance between the right knee and right ankle during that phase.
"""
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
import os

debug_mode = False

def analyze_pitching_motion(video_path: str):
    """
    Analyzes a video to extract raw pose landmark coordinates and motion features.

    This function processes a video frame by frame to:
    1. Track the 2D position of all relevant landmarks.
    2. Calculate the speed of each landmark between frames for motion detection.

    Args:
        video_path (str): The path to the input video file.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary where keys are landmark names (e.g., "RIGHT_KNEE") 
                    and values are lists of their (x, y) coordinates for each frame.
            - dict: A dictionary of speeds for each landmark.
            - float: The frames per second (fps) of the video.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # Define the landmarks we want to track
    tracked_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
        mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_WRIST,
        mp_pose.PoseLandmark.LEFT_HEEL, mp_pose.PoseLandmark.RIGHT_HEEL
    ]

    all_landmark_coords = {lm.name: [] for lm in tracked_landmarks}
    all_landmark_speeds = {lm.name: [] for lm in tracked_landmarks}
    prev_landmark_pos = {lm.name: None for lm in tracked_landmarks}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            for lm_enum in tracked_landmarks:
                lm_data = landmarks[lm_enum.value]
                current_pos = np.array([lm_data.x * w, lm_data.y * h])
                
                # --- Speed Calculation ---
                prev_pos = prev_landmark_pos[lm_enum.name]
                if prev_pos is not None:
                    speed = np.linalg.norm(current_pos - prev_pos)
                    all_landmark_speeds[lm_enum.name].append(speed)
                else:
                    all_landmark_speeds[lm_enum.name].append(0)
                prev_landmark_pos[lm_enum.name] = current_pos

                # --- Store Raw 2D Coordinates ---
                all_landmark_coords[lm_enum.name].append(current_pos)

        else:
            # Append placeholders if no pose is detected
            for lm_name in all_landmark_coords:
                all_landmark_coords[lm_name].append(None)
                all_landmark_speeds[lm_name].append(None)
                prev_landmark_pos[lm_name] = None

    cap.release()
    pose.close()
    
    return all_landmark_coords, all_landmark_speeds, fps

def find_motion_start_frame(all_landmark_speeds: dict, fps: float) -> int:
    """
    Finds the frame where the first major motion starts by analyzing each landmark's speed.

    This method calculates a motion start time for each tracked landmark and returns the
    earliest (minimum) frame number among them.
    """
    if not all_landmark_speeds:
        return 0
    
    all_start_frames = []
    
    # 1. Determine the frame with the absolute maximum motion across all landmarks
    # This helps to anchor the backward search globally.
    max_speed = -1
    global_peak_motion_frame = 0
    for lm_name, speeds in all_landmark_speeds.items():
        if not speeds:
            continue
        
        speeds_series = pd.Series(speeds).fillna(0)
        smoothing_window = int(fps / 5) # 200ms smoothing window
        smoothed_speeds = speeds_series.rolling(window=smoothing_window, min_periods=1, center=True).mean()

        # print("    [Debug] ", smoothed_speeds)

        if smoothed_speeds.empty:
            continue
        
        peak_frame_for_lm = smoothed_speeds.idxmax()
        peak_speed_for_lm = smoothed_speeds.max()
        
        if peak_speed_for_lm > max_speed:
            max_speed = peak_speed_for_lm
            global_peak_motion_frame = peak_frame_for_lm

    print(f"    [Debug] Global peak motion frame found at: {global_peak_motion_frame}")

    # 2. Find the start frame where ALL landmarks are below quiet threshold simultaneously
    # for at least 3 consecutive frames.
    
    # First, calculate the quiet threshold for each landmark
    landmark_thresholds = {}
    smoothed_speeds_dict = {}
    
    for lm_name, speeds in all_landmark_speeds.items():
        if not speeds:
            continue
            
        speeds_series = pd.Series(speeds).fillna(0)
        smoothing_window = int(fps / 5)  # 200ms smoothing window
        smoothed_speeds = speeds_series.rolling(window=smoothing_window, min_periods=1, center=True).mean()

        if smoothed_speeds.empty:
            continue
        
        quiet_threshold = smoothed_speeds.quantile(0.25)
        landmark_thresholds[lm_name] = quiet_threshold
        smoothed_speeds_dict[lm_name] = smoothed_speeds

    if not smoothed_speeds_dict:
        print("    [Debug] No valid speed data found.")
        return 0

    # Search backwards from global peak to find where ALL landmarks are quiet simultaneously
    min_consecutive_frames = 2
    overall_motion_start_frame = 0
    
    for i in range(global_peak_motion_frame, 0, -1):
        # Check if all landmarks are below their thresholds at frame i
        all_quiet = all(
            smoothed_speeds_dict[lm_name].iloc[i] < landmark_thresholds[lm_name]
            for lm_name in smoothed_speeds_dict
        )

        if debug_mode:
            print (i, all_quiet)
            if (i<10):
                for lm_name in smoothed_speeds_dict:
                    print(lm_name, smoothed_speeds_dict[lm_name].iloc[i], landmark_thresholds[lm_name])
        
        if all_quiet:
            # Verify at least min_consecutive_frames are all quiet
            consecutive_count = 1
            for j in range(i - 1, max(0, i - min_consecutive_frames), -1):
                all_quiet_at_j = all(
                    smoothed_speeds_dict[lm_name].iloc[j] < landmark_thresholds[lm_name]
                    for lm_name in smoothed_speeds_dict
                )
                if all_quiet_at_j:
                    consecutive_count += 1
                else:
                    break
            
            if consecutive_count >= min_consecutive_frames:
                overall_motion_start_frame = i - consecutive_count + 1
                break

    if overall_motion_start_frame == 0:
        print("    [Debug] Could not find a frame where all landmarks are quiet simultaneously.")

        for i in range(global_peak_motion_frame, 0, -1):
            # Check if all landmarks are below their thresholds at frame i
            quiet_count = sum(
                smoothed_speeds_dict[lm_name].iloc[i] < landmark_thresholds[lm_name]
                for lm_name in smoothed_speeds_dict
            )
            all_quiet = quiet_count >= len(smoothed_speeds_dict) - 1

            if debug_mode:
                print (i, all_quiet)
                if (i<10):
                    for lm_name in smoothed_speeds_dict:
                        print(lm_name, smoothed_speeds_dict[lm_name].iloc[i], landmark_thresholds[lm_name])
            
            if all_quiet:
                # Verify at least min_consecutive_frames are all quiet
                consecutive_count = 1
                for j in range(i - 1, max(0, i - min_consecutive_frames), -1):
                    all_quiet_at_j = sum(
                        smoothed_speeds_dict[lm_name].iloc[j] < landmark_thresholds[lm_name]
                        for lm_name in smoothed_speeds_dict
                    ) >= len(smoothed_speeds_dict) - 1
                    if all_quiet_at_j:
                        consecutive_count += 1
                    else:
                        break
                
                if consecutive_count >= min_consecutive_frames:
                    overall_motion_start_frame = i - consecutive_count + 1
                    break

    print(f"    [Debug] Motion start frame (all landmarks quiet): {overall_motion_start_frame}")
    
    return overall_motion_start_frame

def draw_landmarks_on_frame(video_path: str, frame_number: int, landmarks: dict, segments: dict, output_image_path: str):
    """
    Draws smoothed landmarks and segments on a specific frame of a video and saves it.

    Args:
        video_path (str): Path to the input video.
        frame_number (int): The frame to extract and draw on.
        landmarks (dict): A dictionary of smoothed landmark coordinates.
        segments (dict): A dictionary defining connections between landmarks.
        output_image_path (str): Path to save the output image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  -> Error: Could not open video {video_path} to draw landmarks.")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  -> Error: Could not read frame {frame_number} from {video_path}.")
        return

    # Draw segments first
    if False:
        print (f"  -> Drawing segments on frame {frame_number}...")
        for seg_name, (lm1_name, lm2_name) in segments.items():
            p1 = landmarks.get(lm1_name)
            p2 = landmarks.get(lm2_name)
            if p1 is not None and p2 is not None:
                pt1 = (int(p1[0]), int(p1[1]))
                pt2 = (int(p2[0]), int(p2[1]))
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2) # Green lines for segments

    # Draw landmarks on top of segments
    print (f"  -> Drawing landmarks on frame {frame_number}...")
    for lm_name, coords in landmarks.items():
        if coords is not None:
            pt = (int(coords[0]), int(coords[1]))
            cv2.circle(frame, pt, 5, (0, 0, 255), -1)      # Red-filled circle
            cv2.circle(frame, pt, 7, (255, 255, 255), 1)  # White outline

    cv2.imwrite(output_image_path, frame)
    print(f"\n  -> Saved visualization of representative joints to: {output_image_path}")

def get_representative_segment_lengths(video_path: str) -> dict[str, float]:
    """
    Calculates representative lengths for body segments by first smoothing landmark
    coordinates and then calculating distances using only 2D (x, y) coordinates.

    Args:
        video_path (str): The path to the video file.

    Returns:
        dict: A dictionary of segment names and their calculated median lengths.
    """
    # Define segments by the landmark names they connect
    segments = {
        "right_lower_leg": ("RIGHT_KNEE", "RIGHT_ANKLE"),
        # "right_thigh": ("RIGHT_HIP", "RIGHT_KNEE"),
        # "shoulder_width": ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
        # "right_torso": ("RIGHT_SHOULDER", "RIGHT_HIP"),
        # "left_lower_leg": ("LEFT_KNEE", "LEFT_ANKLE"),
        # "left_thigh": ("LEFT_HIP", "LEFT_KNEE"),
    }

    print(f"Analyzing video for scale: {video_path}")
    all_landmark_coords, all_landmark_speeds, fps = analyze_pitching_motion(video_path)
    
    if not any(val is not None for val_list in all_landmark_coords.values() for val in val_list):
        print("  -> Could not detect pose in the video.")
        return {}

    motion_start_frame = find_motion_start_frame(all_landmark_speeds, fps)
    print(f"  -> Detected major motion starting around frame {motion_start_frame}.")

    if debug_mode and motion_start_frame == 0:
        for frame_num in range(0, 10):
            output_image_name = f"{os.path.splitext(os.path.basename(video_path))[0]}_frame_{frame_num:04d}.png"
            instant_landmarks = {}
            for lm_name, coords_list in all_landmark_coords.items():
                static_coords = [c for c in coords_list[frame_num:frame_num+1] if c is not None]
                print(f"    [Debug] Found {len(static_coords)} valid coordinates for {lm_name} in the static window.")

                if lm_name == "RIGHT_KNEE":
                    # Limit printing to avoid excessive output
                    print(f"    [Debug] Raw RIGHT_KNEE coords (first 5): {[f'({c[0]:.1f}, {c[1]:.1f})' for c in static_coords[:5]]}")

                if not static_coords:
                    print(f"  -> No valid coordinates for {lm_name} in the static period.")
                    instant_landmarks[lm_name] = None
                    continue

                # Apply median filter element-wise (x, y)
                x_coords = [c[0] for c in static_coords]
                y_coords = [c[1] for c in static_coords]
                
                median_coord = np.array([np.median(x_coords), np.median(y_coords)]) # Only x and y
                instant_landmarks[lm_name] = median_coord

            print (output_image_name)
            draw_landmarks_on_frame(video_path, frame_num, instant_landmarks, segments, output_image_name)

    window_size = int(fps * 0.5)
    static_period_start = max(0, motion_start_frame - window_size)
    static_period_end = motion_start_frame

    print(f"  -> Using static analysis window from frame {static_period_start} to {static_period_end}.")
    
    # Smooth the coordinates first
    smoothed_landmarks = {}
    for lm_name, coords_list in all_landmark_coords.items():
        static_coords = [c for c in coords_list[static_period_start:static_period_end] if c is not None]
        print(f"    [Debug] Found {len(static_coords)} valid coordinates for {lm_name} in the static window.")

        if lm_name == "RIGHT_KNEE":
             # Limit printing to avoid excessive output
            print(f"    [Debug] Raw RIGHT_KNEE coords (first 5): {[f'({c[0]:.1f}, {c[1]:.1f})' for c in static_coords[:5]]}")

        if not static_coords:
            print(f"  -> No valid coordinates for {lm_name} in the static period.")
            smoothed_landmarks[lm_name] = None
            continue

        # Apply median filter element-wise (x, y)
        x_coords = [c[0] for c in static_coords]
        y_coords = [c[1] for c in static_coords]
        
        median_coord = np.array([np.median(x_coords), np.median(y_coords)]) # Only x and y
        smoothed_landmarks[lm_name] = median_coord
    
    # --- Print representative landmark locations ---
    print("\n  -> Representative Landmark Locations (x, y):") # Updated print statement
    for lm_name, coords in smoothed_landmarks.items():
        if coords is not None:
            print(f"    {lm_name.replace('_', ' ').title()}: ({coords[0]:.2f}, {coords[1]:.2f})") # Updated print statement
        else:
            print(f"    {lm_name.replace('_', ' ').title()}: Not available")
    # --- End of print representative landmark locations ---

    # --- Visualize the representative joints on a frame ---
    vis_frame_number = (static_period_start + static_period_end) // 2
    output_image_name = f"representative_joints_{os.path.splitext(os.path.basename(video_path))[0]}.png"
    draw_landmarks_on_frame(video_path, vis_frame_number, smoothed_landmarks, segments, output_image_name)
    # --- End of visualization ---

    # Calculate lengths from smoothed coordinates
    representative_lengths = {}
    for seg_name, (lm1_name, lm2_name) in segments.items():
        p1 = smoothed_landmarks.get(lm1_name)
        p2 = smoothed_landmarks.get(lm2_name)

        if p1 is not None and p2 is not None:
            length = np.linalg.norm(p1 - p2)
            representative_lengths[seg_name] = length
            print(f"  -> Representative length for {seg_name}: {length:.2f}")
        else:
            print(f"  -> Could not calculate length for {seg_name} due to missing landmarks. Setting to 0.")
            representative_lengths[seg_name] = 0.0
            
    return representative_lengths

def calculate_scale_ratios(video_path1: str, video_path2: str) -> dict[str, float]:
    """
    Calculates the scale ratios for various body segments between two videos.

    Args:
        video_path1 (str): Path to the first video.
        video_path2 (str): Path to the second video.
        
    Returns:
        dict: A dictionary of segment names and their calculated scale ratios.
    """
    print("\nCalculating representative lengths for Video 1...")
    lengths1 = get_representative_segment_lengths(video_path1)
    print("\nCalculating representative lengths for Video 2...")
    lengths2 = get_representative_segment_lengths(video_path2)

    print("\n--- Scale Ratios (Video 1 / Video 2) ---")
    ratios = {}
    all_zero_ratios = True
    for segment_name in lengths1.keys():
        len1 = lengths1.get(segment_name, 0.0)
        len2 = lengths2.get(segment_name, 0.0)

        if len2 > 0:
            ratio = len1 / len2
            ratios[segment_name] = ratio
            print(f"  {segment_name.replace('_', ' ').title()}: {ratio:.4f}")
            all_zero_ratios = False
        else:
            ratios[segment_name] = 0.0
            print(f"  {segment_name.replace('_', ' ').title()}: Cannot calculate (Video 2 length is zero)")
    
    if all_zero_ratios:
        print("No valid scale ratios could be calculated for any segment.")
    else:
        print("\nNote: A ratio > 1 means the subject in Video 1 is larger than in Video 2 for that segment.")
        print("To make the subject in Video 2 match Video 1, scale Video 2 by these factors.")
        
    return ratios


if __name__ == '__main__':
    # Example of how to use the function.
    # Provide actual video file paths to run this.
    video1_path = "Input_Video/cutsIMG_2726.mp4"
    video2_path = "Input_Video/cutsIMG_2727.mp4"
    
    print("Starting scale alignment process...")
    try:
        scale_ratios = calculate_scale_ratios(video1_path, video2_path)
        # The function now also prints the ratios, but we can use the returned dict for other purposes.
    except (IOError, FileNotFoundError) as e:
        print(f"Error: {e}. Please ensure video paths are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

