import os
import glob
import argparse
import subprocess
from util_mound_detection import detect_mound

# Configuration
INPUT_DIR = "20260417_pitching"
OUTPUT_DIR = "20260417_pitching_mp4"

def convert_to_mp4(input_path, output_path, crop_x, crop_y):
    """
    Uses FFmpeg to convert a MOV video to a web-compatible H.264 MP4,
    cropping to the center portion (25% to 75% in both directions).
    """
    try:
        # Filter details:
        # crop=width:height:x:y
        # width = 50% of input (iw*0.5)
        # height = 50% of input (ih*0.5)
        # x = 25% offset (iw*0.25)
        # y = 25% offset (ih*0.25)
        crop_filter = f"crop=1920:1080:{crop_x}:{crop_y}"
        
        command = [
            'ffmpeg', '-y', '-i', input_path,
            '-vf', crop_filter,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            '-preset', 'medium', '-crf', '23',
            '-c:a', 'aac', '-b:a', '128k',
            output_path
        ]
        
        print(f"Converting & Cropping: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
        subprocess.run(command, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path}: {e.stderr.decode()}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MOV to MP4 with center crop (25%-75%).")
    parser.add_argument("--input_dir", default=INPUT_DIR, help="Directory containing MOV files")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Directory to save MP4 files")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    files = glob.glob(os.path.join(args.input_dir, "*.[mM][oO][vV]"))
    
    limit = -1
    count = 0
    if not files:
        print(f"No MOV files found in {args.input_dir}")
    else:
        print(f"Found {len(files)} files. Starting conversion with center crop...")
        for f in files:
            # Step 1: Detect mound and pitcher
            coords = detect_mound(f)
            if coords:
                mound_cx, mound_cy = coords
                crop_x = mound_cx - 485
                crop_y = mound_cy - 1025
                
                base_name = os.path.splitext(os.path.basename(f))[0]
                output_file = os.path.join(args.output_dir, f"{base_name}.mp4")
                convert_to_mp4(f, output_file, crop_x, crop_y)
            count += 1
            if count == limit:
                print(f"Processed {limit} files, stopping for now.")
                break
        print("Done.")
