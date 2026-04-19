import os
import shutil
import glob
import argparse

# Configuration
INPUT_DIR = "20260417_hitting"
OUTPUT_DIR = "20260417_hitting_cuts"
PROCESSED_DIR = os.path.join(INPUT_DIR, "processed")

def move_processed_videos(input_dir, output_dir, processed_dir):
    """
    For any file in the output_dir, derives the original name by removing the 'cuts' prefix.
    If the corresponding original video exists in input_dir, it is moved to processed_dir.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    # Get all files in the output directory
    output_files = os.listdir(output_dir)
    
    moved_count = 0
    
    for out_file in output_files:
        # Get the basename (stem) without extension
        base_name, _ = os.path.splitext(out_file)
        
        # Derive original name by removing "cuts" prefix if it exists
        original_stem = base_name
        if base_name.startswith("cuts"):
            original_stem = base_name[4:]
            
        # Search for matching original video in input_dir (case-insensitive MOV or mp4)
        originals = []
        for ext in ["[mM][oO][vV]", "[mM][pP]4"]:
            originals.extend(glob.glob(os.path.join(input_dir, f"{original_stem}.{ext}")))
        
        for original_path in originals:
            target_path = os.path.join(processed_dir, os.path.basename(original_path))
            
            print(f"Moving original: {os.path.basename(original_path)} -> {processed_dir}")
            try:
                shutil.move(original_path, target_path)
                moved_count += 1
            except Exception as e:
                print(f"Error moving {original_path}: {e}")

    print(f"\nTask complete. Total original files moved: {moved_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move original videos to processed folder based on output results.")
    parser.add_argument("--input_dir", default=INPUT_DIR, help="Directory containing original videos")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Directory containing output videos")
    parser.add_argument("--processed_dir", default=PROCESSED_DIR, help="Directory to move originals to")
    args = parser.parse_args()

    move_processed_videos(args.input_dir, args.output_dir, args.processed_dir)
