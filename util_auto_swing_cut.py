import cv2

import tensorflow as tf
tf.config.optimizer.set_jit(True)  # Enable XLA
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Iterate through available GPUs to find the NVIDIA card
    selected_gpu = gpus[0]  # Fallback to the first detected GPU
    for gpu in gpus:
        details = tf.config.experimental.get_device_details(gpu)
        print (f"GPU found: {details.get('device_name', 'Unknown')}")
        if 'NVIDIA' in details.get('device_name', ''):
            selected_gpu = gpu
            break

    tf.config.set_visible_devices(selected_gpu, 'GPU')
    print(f"TensorFlow found {len(gpus)} GPU(s). Selected: {tf.config.experimental.get_device_details(selected_gpu).get('device_name')}")
else:
    print("No GPU found by TensorFlow. Processing will default to CPU.")

import mediapipe as mp
import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
from collections import deque
import csv
import argparse
from func_scale_alignment import find_motion_start_frame
import util_batter_detection as bd

INPUT_DIR = "20260417_hitting"
OUTPUT_DIR = "20260417_hitting_cuts"

IPHONE_16_PRO_MAX = False

CROP_WIDTH = 1920
CROP_HEIGHT = 1080

SHOW_VIDEO = False
OUTPUT_LANDMARKS = True
DEBUG = False

SKIP_THRESHOLD = 0.5        # seconds to skip at the start of the video to avoid false positives from initial movement
PROCESS_THREADHOLD = 5      # seconds to process in the video
EXTRA_END_TIME = 0.5        # seconds to include after the detected motion end
EXTRA_START_TIME = 1.5      # seconds to include before the detected motion start

import mediapipe as mp
mp_pose = mp.solutions.pose  # For connections and landmarks

# Selected joints for tracking
SELECTED_JOINTS = {
    'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
    'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
    'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
    'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
    'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE,
    'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE,
    'left_heel': mp_pose.PoseLandmark.LEFT_HEEL,
    'right_heel': mp_pose.PoseLandmark.RIGHT_HEEL
}

# Selected joints for tracking swing speed
SELECTED_JOINTS_1 = {
    'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST,
    'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST,
}

# Selected joints for tracking stable base of support
SELECTED_JOINTS_2 = {
    'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
    'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
    'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE,
    'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE,
    'left_heel': mp_pose.PoseLandmark.LEFT_HEEL,
    'right_heel': mp_pose.PoseLandmark.RIGHT_HEEL
}

# Check CUDA and cuDNN versions (if available)
try:
    cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
    cudnn_version = tf.sysconfig.get_build_info()["cudnn_version"]
    print("CUDA version:", cuda_version)
    print("cuDNN version:", cudnn_version)
except Exception as e:
    print("Could not retrieve CUDA/cuDNN version from TensorFlow build info:", e)

# Initialize MediaPipe Pose.
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    #if True:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
    
    # Use PoseLandmarker for multi-pose detection
    # Download the model from https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
    # Using the heavy model for maximum accuracy in sports motion
    base_options = BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    options = PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=3,  # Focus exclusively on the batter within the swing_box
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )
    pose_landmarker = PoseLandmarker.create_from_options(options)
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose  # For connections and landmarks

def select_largest_pose(pose_landmarks_list):
    """Select the pose with the largest bounding box area."""
    if not pose_landmarks_list:
        return None
    largest_pose = None
    max_area = 0
    for pose in pose_landmarks_list:
        xs = [lm.x for lm in pose]
        ys = [lm.y for lm in pose]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        area = (max_x - min_x) * (max_y - min_y)
        if area > max_area:
            max_area = area
            largest_pose = pose
    return largest_pose

def convert_to_pose_landmark_list(normalized_landmarks):
    """Convert list of NormalizedLandmark to a mock PoseLandmarkList for drawing."""
    if normalized_landmarks is None:
        return None
    
    class MockLandmark:
        def __init__(self, lm):
            self.x = lm.x
            self.y = lm.y
            self.z = lm.z
            self.visibility = getattr(lm, 'visibility', 1.0)
            self.presence = getattr(lm, 'presence', 1.0)
        
        def HasField(self, field):
            return True  # Assume all fields are set
    
    class MockPoseLandmarkList:
        def __init__(self, landmark):
            self.landmark = [MockLandmark(lm) for lm in landmark]
    
    return MockPoseLandmarkList(normalized_landmarks)


def track_video(input_file, output_video_file1, output_video_file2, plot_file, speed_csv_file):
    # Create a fresh PoseLandmarker instance for each video to reset timestamp tracking
    local_base_options = BaseOptions(model_asset_path='pose_landmarker_heavy.task')
    local_options = PoseLandmarkerOptions(
        base_options=local_base_options,
        num_poses=3,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
        running_mode=mp.tasks.vision.RunningMode.VIDEO
    )
    local_pose_landmarker = PoseLandmarker.create_from_options(local_options)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cap = cv2.VideoCapture(input_file)
    if input_file.endswith(".MOV") and not IPHONE_16_PRO_MAX:
        ROTATE_VIDEO = 1
        print(f"Rotating video {input_file} by 180 degrees.")
    else:
        ROTATE_VIDEO = -1
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    skip_thread = int(SKIP_THRESHOLD * fps)
    process_thread = int(PROCESS_THREADHOLD * fps)
    extra_end_frames = int(EXTRA_END_TIME * fps)
    extra_start_frames = int(EXTRA_START_TIME * fps)

    swing_box = None

    # Prepare to collect all landmark coordinates for cropping
    all_landmark_coords = []

    # Dictionaries to store horizontal and vertical speeds for each selected joint over time
    joint_horizontal_speeds = {joint: [] for joint in SELECTED_JOINTS.keys()}
    joint_vertical_speeds = {joint: [] for joint in SELECTED_JOINTS.keys()}

    # Optional: also track knee-to-heel distance (if needed)
    knee_to_heel_lengths = []

    # Use a sliding window for smoothing landmark positions
    landmark_window = deque(maxlen=5)
    prev_landmarks = None

    frames = []
    smoothed_landmarks_list = []
    raw_landmarks_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if ROTATE_VIDEO == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        if not ret:
            break

        frames.append(frame.copy())  # store all frames in memory

        # Detect batter and determine swing box. 
        # We try to find the batter for up to 1 second before defaulting to the full frame.
        if swing_box is None:
            clay_mask = bd.get_clay_mask(frame)
            batter_bbox, _ = bd.detect_batter(frame, clay_mask)
            if batter_bbox:
                swing_box = bd.get_swing_box(frame.shape, batter_bbox)
                print(f"Batter detected for {input_file} at frame {len(frames)}. Swing box: {swing_box}")
            elif len(frames) > fps:
                print(f"Warning: Batter not detected in {input_file} after {fps} frames. Using full frame.")
                swing_box = (0, 0, frame_width, frame_height)
            else:
                # Skip pose processing until batter is found or timeout reached
                raw_landmarks_list.append(None)
                smoothed_landmarks_list.append(None)
                all_landmark_coords.append(None)
                for joint_name in SELECTED_JOINTS.keys():
                    joint_horizontal_speeds[joint_name].append(0)
                    joint_vertical_speeds[joint_name].append(0)
                knee_to_heel_lengths.append(0)
                continue

        # Mask all pixels outside the swing_box as grey (128, 128, 128)
        masked_frame = np.full(frame.shape, 128, dtype=np.uint8)
        sx, sy, sw, sh = swing_box
        masked_frame[sy:sy+sh, sx:sx+sw] = frame[sy:sy+sh, sx:sx+sw]

        rgb_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        # Calculate timestamp in milliseconds for VIDEO mode
        timestamp_ms = int((len(frames) - 1) * 1000 / fps)
        results = local_pose_landmarker.detect_for_video(mp_image, timestamp_ms)
        selected_pose = select_largest_pose(results.pose_landmarks)
        if DEBUG:
            print(f"Processed frame {len(frames)} for {input_file}. Pose landmarks detected: {selected_pose is not None}")

        if selected_pose:
            pose_landmark_list = convert_to_pose_landmark_list(selected_pose)
            raw_landmarks_list.append(pose_landmark_list)
            # Extract landmark positions and smooth via median of sliding window.
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in selected_pose])
            landmark_window.append(landmarks)
            smoothed_landmarks = np.median(landmark_window, axis=0)
            smoothed_landmarks_list.append(smoothed_landmarks)

            # Collect all landmark coordinates for cropping
            all_landmark_coords.append(smoothed_landmarks[:, :2])

            # Draw all smoothed landmark points.
            for lm in smoothed_landmarks:
                x, y = int(lm[0] * frame.shape[1]), int(lm[1] * frame.shape[0])
                #cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Calculate speeds for each selected joint.
            if prev_landmarks is not None:
                for joint_name, landmark_enum in SELECTED_JOINTS.items():
                    idx = landmark_enum.value
                    # Horizontal speed: difference in x coordinates scaled by frame width.
                    horizontal_speed = (smoothed_landmarks[idx, 0] - prev_landmarks[idx, 0]) * frame_width
                    # Vertical speed: difference in y coordinates scaled by frame height.
                    vertical_speed = (smoothed_landmarks[idx, 1] - prev_landmarks[idx, 1]) * frame_height
                    joint_horizontal_speeds[joint_name].append(horizontal_speed)
                    joint_vertical_speeds[joint_name].append(vertical_speed)
            else:
                # For the first frame, append 0 to initialize for each joint.
                for joint_name in SELECTED_JOINTS.keys():
                    joint_horizontal_speeds[joint_name].append(0)
                    joint_vertical_speeds[joint_name].append(0)

            # Optional: Compute knee-to-heel distance for both legs and take the average.
            try:
                left_distance = np.linalg.norm(smoothed_landmarks[25] - smoothed_landmarks[27])
                right_distance = np.linalg.norm(smoothed_landmarks[26] - smoothed_landmarks[28])
                avg_leg_length = (left_distance + right_distance) / 2 * frame_width
                knee_to_heel_lengths.append(avg_leg_length)
            except Exception as e:
                # In case required landmarks are missing.
                knee_to_heel_lengths.append(0)

            prev_landmarks = smoothed_landmarks

            # Optional: Draw MediaPipe connections over the landmarks.
            mp.solutions.drawing_utils.draw_landmarks(
                frame, pose_landmark_list, mp_pose.POSE_CONNECTIONS)
        else:
            raw_landmarks_list.append(None)
            smoothed_landmarks_list.append(None)
            all_landmark_coords.append(None)
            prev_landmarks = None
            for joint_name in SELECTED_JOINTS.keys():
                joint_horizontal_speeds[joint_name].append(0)
                joint_vertical_speeds[joint_name].append(0)
            knee_to_heel_lengths.append(0)

        if SHOW_VIDEO:
            cv2.imshow('Pose Tracking', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # Stop processing after the threshold duration (5 seconds)
        if len(frames) >= process_thread:
            break

    cap.release()

    # Check for the specific condition for the right wrist & left wrist horizontal speed to determine the cut point.
    right_wrist_horizontal = joint_horizontal_speeds['right_wrist']
    left_wrist_horizontal = joint_horizontal_speeds['left_wrist']
    handspeed = [(h + v) for h, v in zip(right_wrist_horizontal, left_wrist_horizontal)]
    max_horizontal_speed_frame = None

    # Find the first frame where the right wrist's horizontal speed > 100 pixels/frame, and then look for the maximum horizontal speed in the next 10 frames.
    for i in range(skip_thread, len(handspeed) - 10):
        if DEBUG:
            print (handspeed[i])
        if handspeed[i] > 100:
            horizontal_max = max(handspeed[i:i + 10])
            max_horizontal_speed_frame = i + handspeed[i:i + 10].index(horizontal_max)
            # Find the frame with the maximum horizontal speed in this range
            print(f"Found frame with handspeed speed > 100 at frame {i}. Marking frame {max_horizontal_speed_frame} as the maximum.")
            break

    # If no frame meets the condition, find the frame with the maximum horizontal speed in the entire video starting from frame 30.
    if max_horizontal_speed_frame is None:
        print(f"Warning: No max_horizontal_speed_frame marker found in {input_file}.")
        horizontal_max = max(handspeed[skip_thread:])
        max_horizontal_speed_frame = skip_thread + handspeed[skip_thread:].index(horizontal_max)

    print(f"Max horizontal speed frame: {max_horizontal_speed_frame}, Speed: {handspeed[max_horizontal_speed_frame]}")

    # Prepare speed data for find_motion_start_frame
    all_landmark_speeds = {}
    joint_name_to_enum_name = {k: v.name for k, v in SELECTED_JOINTS_2.items()}
    
    for joint_name, h_speeds in joint_horizontal_speeds.items():
        if joint_name in SELECTED_JOINTS_2:
            enum_name = joint_name_to_enum_name[joint_name]
            v_speeds = joint_vertical_speeds[joint_name]
            # Calculate the magnitude of the speed vector
            all_landmark_speeds[enum_name] = [np.sqrt(h**2 + v**2) for h, v in zip(h_speeds, v_speeds)]

    motion_start_frame = find_motion_start_frame(all_landmark_speeds, fps)
    
    # Trim the video to include at least 10 frames before motion starts
    motion_start_with_buffer = motion_start_frame - 10
    original_start_frame_proposal = max_horizontal_speed_frame - extra_start_frames
    start_frame = max(0, min(original_start_frame_proposal, motion_start_with_buffer))
    
    end_frame = min(len(frames), max(max_horizontal_speed_frame + extra_end_frames, start_frame + extra_start_frames + extra_end_frames))

    out_motion = cv2.VideoWriter(output_video_file2, fourcc, fps, (frame_width, frame_height))
    # Save motion-only video
    for i in range(start_frame, end_frame):
        out_motion.write(frames[i])
    out_motion.release()
    print(f"Pitching video saved to {output_video_file2}")    

    # Save the speeds to CSV 
    if all(len(speeds) > 0 for speeds in joint_horizontal_speeds.values()):
        with open(speed_csv_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            header = [f"{joint}_horizontal" for joint in SELECTED_JOINTS.keys()] + \
                    [f"{joint}_vertical" for joint in SELECTED_JOINTS.keys()] + ['knee_to_heel']
            writer.writerow(header)
            for i in range(len(frames)-1):
                row = [joint_horizontal_speeds[joint][i] for joint in SELECTED_JOINTS.keys()] + \
                    [joint_vertical_speeds[joint][i] for joint in SELECTED_JOINTS.keys()] + \
                    [knee_to_heel_lengths[i]]
                writer.writerow(row)
        print(f"Trimmed speed data saved to {speed_csv_file}")
    
    # Plot the speeds over time.
    plt.figure(figsize=(10, 6))

    # Plot horizontal and vertical speeds for each joint in SELECTED_JOINTS_1
    for joint_name in SELECTED_JOINTS_1.keys():
        speed_list_h = joint_horizontal_speeds[joint_name]
        speed_list_v = joint_vertical_speeds[joint_name]
        plt.plot(speed_list_h, label=f'{joint_name} horizontal speed')
        plt.plot(speed_list_v, label=f'{joint_name} vertical speed', linestyle='--')

    # Mark the frame with the maximum horizontal speed
    if max_horizontal_speed_frame is not None:
        plt.axvline(max_horizontal_speed_frame, color='red', linestyle='--', label='Max Horizontal Speed')
        plt.axvline(start_frame, color='green', linestyle='--', label='Start Frame')
        plt.axvline(end_frame, color='blue', linestyle='--', label='End Frame')
        print(f"Marked frame: {max_horizontal_speed_frame}, Horizontal Speed: {right_wrist_horizontal[max_horizontal_speed_frame]}")

    plt.title("Selected Joint Speeds Over Time")
    plt.xlabel("Frame Index")
    plt.ylabel("Speed (pixels/frame)")
    plt.legend()
    plt.savefig(plot_file)
    print(f"Speed plot saved to {plot_file}")

    if OUTPUT_LANDMARKS:
        # Debugging lengths to ensure lists are synchronized
        print(f"DEBUG: Starting annotated video generation for {input_file}")
        print(f"DEBUG: Frame count = {len(frames)}, Landmark count = {len(raw_landmarks_list)}")
        print(f"DEBUG: Smoothed landmark count = {len(smoothed_landmarks_list)}")

        # Set up video writer for annotated output
        out = cv2.VideoWriter(output_video_file1, fourcc, fps, (frame_width, frame_height))

        for i, frame in enumerate(frames):
            current_frame = frame.copy() # Use a copy to draw on
            # Draw original landmarks and skeleton connections
            if raw_landmarks_list[i] is not None:
                mp_drawing.draw_landmarks(
                    current_frame, 
                    raw_landmarks_list[i], 
                    mp_pose.POSE_CONNECTIONS)
            out.write(current_frame)

        out.release()
        print(f"Landmarked video saved to {output_video_file1}")
        if SHOW_VIDEO:
            cv2.destroyAllWindows()
    
    # Clean up the local PoseLandmarker instance
    local_pose_landmarker.close()


#############################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Track pose landmarks in videos and cut based on motion.")
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR,
                        help="Directory containing input video files. Default is current directory.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Directory to save output videos, plots, and CSVs. Default is 'cuts'.")
    args = parser.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    unprocessed_files = [f for f in glob.glob(os.path.join(input_dir, "IMG*.MOV")) if "tracked" not in f]
    #unprocessed_files = [f for f in glob.glob(os.path.join(input_dir, "cutsIMG*.mp4")) if "tracked" not in f]

    for filename in unprocessed_files:
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        if base_filename.endswith("landmarks"):
            print(f"WARNING: {base_filename}_landmarks already exists. Skipping.")
            continue

        output_video1 = os.path.join(output_dir, base_filename + "_landmarks.mp4")
        output_video2 = os.path.join(output_dir, "cuts" + base_filename + ".mp4")
        output_plot = os.path.join(output_dir, base_filename + "_speed.png")
        output_csv = os.path.join(output_dir, base_filename + "_speed.csv")

        print(f"Processing {filename} → {output_video2}, {output_plot}, {output_csv}")
        track_video(filename, output_video1, output_video2, output_plot, output_csv)
