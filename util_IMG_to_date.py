import os
import glob
import time
from datetime import datetime

def rename_mov_files(input_dir):
    """
    Reads all .MOV files in the input directory and renames them 
    to a compact date+time format (YYMMDDHHMM) based on their 
    last modification time.
    """
    # Search for .MOV files (case-insensitive)
    search_pattern = os.path.join(input_dir, "*.[mM][oO][vV]")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No MOV files found in {input_dir}")
        return

    print(f"Found {len(files)} files. Renaming to YYMMDDHHMM format...")
    
    for file_path in files:
        # Get file extension
        ext = os.path.splitext(file_path)[1]
        
        # Get modification time
        mtime = os.path.getmtime(file_path)
        dt_obj = datetime.fromtimestamp(mtime)
        
        # Format to YYMMDDHHMMSS
        new_name_base = dt_obj.strftime("%y%m%d%H%M%S")
        
        # Handle potential name collisions by adding a suffix if needed
        new_name = f"{new_name_base}{ext}"
        target_path = os.path.join(input_dir, new_name)
        
        counter = 1
        while os.path.exists(target_path):
            new_name = f"{new_name_base}_{counter}{ext}"
            target_path = os.path.join(input_dir, new_name)
            counter += 1
            
        try:
            os.rename(file_path, target_path)
            print(f"Renamed: {os.path.basename(file_path)} -> {new_name}")
        except Exception as e:
            print(f"Error renaming {file_path}: {e}")

if __name__ == "__main__":
    # Set your input folder here
    INPUT_FOLDER = "20260417_pitching" 
    rename_mov_files(INPUT_FOLDER)
