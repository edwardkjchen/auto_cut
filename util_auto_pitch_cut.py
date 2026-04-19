import cv2

"""
import tensorflow as tf
tf.config.optimizer.set_jit(True)  # Enable XLA
tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_visible_devices(gpus[0], 'GPU')
"""

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

INPUT_DIR = "20260417_pitching_mp4"
OUTPUT_DIR = "20260417_pitching_cuts"

IPHONE_16_PRO_MAX = True

CROP_WIDTH = 1920
CROP_HEIGHT = 1080

FOCUS_LEFT = 0
FOCUS_RIGHT = 1
SHOW_VIDEO = False
OUTPUT_LANDMARKS = False
SKIP_THRESHOLD = 30

EXTRA_END_FRAMES = 45 
EXTRA_START_FRAMES = 135

"""
# Check CUDA and cuDNN versions (if available)
try:
    cuda_version = tf.sysconfig.get_build_info()["cuda_version"]
    cudnn_version = tf.sysconfig.get_build_info()["cudnn_version"]
    print("CUDA version:", cuda_version)
    print("cuDNN version:", cudnn_version)
except Exception as e:
    print("Could not retrieve CUDA/cuDNN version from TensorFlow build info:", e)

# Initialize MediaPipe Pose.
with tf.device('/GPU:0'):
"""
if True:
    mp_pose = mp.solutions.pose
    # Use model_complexity=2 for the heavy model
    pose = mp_pose.Pose(static_image_mode=False, 
                        model_complexity=2,
                        min_detection_confidence=0.5, 
                        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

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


def track_video(input_file, output_video_file1, output_video_file2, plot_file, speed_csv_file):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    cap = cv2.VideoCapture(input_file)
    if input_file.endswith(".MOV") and not IPHONE_16_PRO_MAX:
        ROTATE_VIDEO = 1
        print(f"Rotating video {input_file} by 180 degrees for correct orientation.")
    else:
        ROTATE_VIDEO = -1
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    left_crop = int(frame_width * FOCUS_LEFT)
    right_crop = int(frame_width * FOCUS_RIGHT)

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

    while cap.isOpened():
        ret, frame = cap.read()
        if ROTATE_VIDEO == 1:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        if not ret:
            break

        frames.append(frame.copy())  # store all frames in memory
        # Make copies to avoid modifying the original frame
        masked_frame = frame.copy()

        # Gray out left and right margins
        masked_frame[:, :left_crop] = (128, 128, 128)
        masked_frame[:, right_crop:] = (128, 128, 128)

        rgb_frame = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        if results.pose_landmarks:
            # Extract landmark positions and smooth via median of sliding window.
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in results.pose_landmarks.landmark])
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
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        else:
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

    cap.release()

    # Check for the specific condition for the right wrist & left wrist horizontal speed to determine the cut point.
    right_wrist_horizontal = joint_horizontal_speeds['right_wrist']
    left_wrist_horizontal = joint_horizontal_speeds['left_wrist']
    handspeed = [(h + v) for h, v in zip(right_wrist_horizontal, joint_vertical_speeds['right_wrist'])]
    max_horizontal_speed_frame = None

    # Find the first frame where the right wrist's horizontal speed > 100 pixels/frame, and then look for the maximum horizontal speed in the next 10 frames.
    for i in range(SKIP_THRESHOLD, len(handspeed) - 10):
        if handspeed[i] > 100:
            horizontal_max = max(handspeed[i:i + 10])
            max_horizontal_speed_frame = i + handspeed[i:i + 10].index(horizontal_max)
            # Find the frame with the maximum horizontal speed in this range
            break

    # If no frame meets the condition, find the frame with the maximum horizontal speed in the entire video starting from frame 30.
    if max_horizontal_speed_frame is None:
        print(f"Warning: No backup marker found in {input_file}.")
        horizontal_max = max(handspeed[SKIP_THRESHOLD:])
        max_horizontal_speed_frame = handspeed[SKIP_THRESHOLD:].index(horizontal_max)

    if max_horizontal_speed_frame is None:
        # Save the speeds to CSV; headers include each selected joint's horizontal and vertical speeds, and the knee_to_heel distance.
        if all(len(speeds) > 0 for speeds in joint_horizontal_speeds.values()):
            with open(speed_csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                header = [f"{joint}_horizontal" for joint in SELECTED_JOINTS.keys()] + \
                        [f"{joint}_vertical" for joint in SELECTED_JOINTS.keys()] + ['knee_to_heel']
                writer.writerow(header)
                num_frames = len(next(iter(joint_horizontal_speeds.values())))
                for i in range(num_frames):
                    row = [joint_horizontal_speeds[joint][i] for joint in SELECTED_JOINTS.keys()] + \
                        [joint_vertical_speeds[joint][i] for joint in SELECTED_JOINTS.keys()] + \
                        [knee_to_heel_lengths[i] if i < len(knee_to_heel_lengths) else 0]
                    writer.writerow(row)
            print(f"Speed data saved to {speed_csv_file}")
    else:
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
        original_start_frame_proposal = max_horizontal_speed_frame - EXTRA_START_FRAMES
        start_frame = max(0, min(original_start_frame_proposal, motion_start_with_buffer))
        
        end_frame = min(len(frames), max((max_horizontal_speed_frame + EXTRA_END_FRAMES), motion_start_frame + EXTRA_START_FRAMES+EXTRA_END_FRAMES))

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
            for i in range(0, len(frames)):
                row = [joint_horizontal_speeds[joint][i] for joint in SELECTED_JOINTS.keys()] + \
                    [joint_vertical_speeds[joint][i] for joint in SELECTED_JOINTS.keys()] + \
                    [knee_to_heel_lengths[i]]
                writer.writerow(row)
        print(f"Trimmed speed data saved to {speed_csv_file}")
    
    # Plot the speeds over time.
    plt.figure(figsize=(10, 6))

    # Plot horizontal and vertical speeds for each joint
    for joint_name, speed_list in joint_horizontal_speeds.items():
        plt.plot(speed_list, label=f'{joint_name} horizontal speed')
    for joint_name, speed_list in joint_vertical_speeds.items():
        plt.plot(speed_list, label=f'{joint_name} vertical speed', linestyle='--')

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
        # Set up video writer for annotated output
        out = cv2.VideoWriter(output_video_file1, fourcc, fps, (frame_width, frame_height))

        for i, frame in enumerate(frames):
            current_frame = frame.copy() # Use a copy to draw on
            # Optionally, draw landmarks on current_frame
            if smoothed_landmarks_list[i] is not None:
                for lm in smoothed_landmarks_list[i]:
                    x, y = int(lm[0] * frame_width), int(lm[1] * frame_height)
                    cv2.circle(current_frame, (x, y), 5, (0, 255, 0), -1)
            if ROTATE_VIDEO == 1:
                rotated_frame = cv2.rotate(current_frame, cv2.ROTATE_180)
                out.write(rotated_frame)
            else:
                out.write(current_frame)

        out.release()
        print(f"Landmarked video saved to {output_video_file1}")
        if SHOW_VIDEO:
            cv2.destroyAllWindows()


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

    unprocessed_files = [f for f in glob.glob(os.path.join(input_dir, "*.mp4")) if "tracked" not in f]
    print (f"Found {len(unprocessed_files)} unprocessed video files in {input_dir}.")
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
