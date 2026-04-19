import cv2
import os
import glob
import argparse

def extract_first_frames(input_dir, output_dir):
    """
    Scans the input directory for video files and saves the first frame of each 
    as a PNG file in the output directory.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Define supported video extensions
    extensions = ['*.mp4', '*.MOV', '*.avi', '*.mkv', '*.mov', '*.MP4']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    # Remove duplicates and sort
    video_files = sorted(list(set(video_files)))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return

    print(f"Found {len(video_files)} videos. Extracting first frames...")

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            continue
        
        # Read the first frame
        ret, frame = cap.read()
        if ret:
            # Construct output filename using the video's base name
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            cv2.imwrite(output_path, frame)
            print(f"Saved: {output_path}")
        else:
            print(f"Error: Could not read first frame from {video_path}")
        
        cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract the first frame of each video in a directory as a PNG.")
    parser.add_argument("--input_dir", type=str, help="Directory containing video files.")
    parser.add_argument("--output_dir", type=str, help="Directory to save PNG files. Defaults to input_dir.")
    args = parser.parse_args()
    
    if args.input_dir:
        out_dir = args.output_dir if args.output_dir else args.input_dir
        extract_first_frames(args.input_dir, out_dir)
    else:
        input_dir = "20260411_hitting\processed"
        extract_first_frames(input_dir, input_dir)
