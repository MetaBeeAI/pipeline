#!/usr/bin/env python3
"""
Script to copy folders from 'all' dataset to 'dataset3' excluding those already in 'processed'.
"""

import os
import shutil
from pathlib import Path

def main():
    # Define paths
    processed_path = Path("/Users/user/Documents/MetaBeeAI_dataset2/papers")
    all_path = Path("/Users/user/Documents/MetaBeeAI_data/papers")
    destination_path = Path("/Users/user/Documents/MetaBeeAI_dataset3/papers")
    
    # Create destination directory if it doesn't exist
    destination_path.mkdir(parents=True, exist_ok=True)
    
    # Check which folders exist in processed
    processed_folders = set()
    if processed_path.exists():
        processed_folders = {folder.name for folder in processed_path.iterdir() if folder.is_dir()}
        print(f"Found {len(processed_folders)} folders in processed dataset:")
        for folder in sorted(processed_folders):
            print(f"  - {folder}")
    else:
        print(f"Processed path does not exist: {processed_path}")
    
    # Check which folders exist in all
    all_folders = set()
    if all_path.exists():
        all_folders = {folder.name for folder in all_path.iterdir() if folder.is_dir()}
        print(f"\nFound {len(all_folders)} folders in all dataset:")
        for folder in sorted(all_folders):
            print(f"  - {folder}")
    else:
        print(f"All dataset path does not exist: {all_path}")
        return
    
    # Find folders to copy (in all but not in processed)
    folders_to_copy = all_folders - processed_folders
    print(f"\nFolders to copy ({len(folders_to_copy)}):")
    for folder in sorted(folders_to_copy):
        print(f"  - {folder}")
    
    # Copy folders
    copied_count = 0
    for folder_name in folders_to_copy:
        source_folder = all_path / folder_name
        dest_folder = destination_path / folder_name
        
        try:
            if dest_folder.exists():
                print(f"  Skipping {folder_name} (already exists in destination)")
                continue
                
            print(f"  Copying {folder_name}...")
            shutil.copytree(source_folder, dest_folder)
            copied_count += 1
            print(f"    ✓ Copied successfully")
            
        except Exception as e:
            print(f"    ✗ Error copying {folder_name}: {e}")
    
    print(f"\nSummary:")
    print(f"  - Processed folders: {len(processed_folders)}")
    print(f"  - All folders: {len(all_folders)}")
    print(f"  - Folders to copy: {len(folders_to_copy)}")
    print(f"  - Successfully copied: {copied_count}")

if __name__ == "__main__":
    main()
