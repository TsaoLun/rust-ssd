#!/usr/bin/env python3
"""
Create an even smaller test dataset with just a few images to debug the issue.
"""

import json
import shutil
from pathlib import Path

def create_test_dataset():
    """Create a tiny dataset for testing"""
    
    # Source and destination paths
    source_root = Path("./assets/small_datasets")
    dest_root = Path("./assets/test_datasets")
    
    # Create destination directories
    for subdir in ["annotations", "train2017", "val2017"]:
        (dest_root / subdir).mkdir(parents=True, exist_ok=True)
    
    # Process training set (only 5 images)
    print("Creating test training set...")
    process_split(
        source_root / "annotations/instances_train2017.json",
        dest_root / "annotations/instances_train2017.json",
        source_root / "train2017",
        dest_root / "train2017",
        max_images=5
    )
    
    # Process validation set (only 2 images)
    print("Creating test validation set...")
    process_split(
        source_root / "annotations/instances_val2017.json",
        dest_root / "annotations/instances_val2017.json",
        source_root / "val2017",
        dest_root / "val2017",
        max_images=2
    )
    
    print("Test dataset creation completed!")

def process_split(source_json, dest_json, source_images, dest_images, max_images):
    """Process a single split (train/val) with only a few images"""
    
    # Load annotations
    with open(source_json, 'r') as f:
        data = json.load(f)
    
    # Take only first few images that have annotations
    selected_images = []
    selected_image_ids = set()
    
    for img in data['images']:
        if len(selected_images) >= max_images:
            break
            
        # Check if this image has annotations
        has_annotations = any(ann['image_id'] == img['id'] for ann in data['annotations'])
        if has_annotations:
            selected_images.append(img)
            selected_image_ids.add(img['id'])
    
    # Filter annotations for selected images
    filtered_annotations = [
        ann for ann in data['annotations'] 
        if ann['image_id'] in selected_image_ids
    ]
    
    # Create new annotation file
    new_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
        'images': selected_images,
        'annotations': filtered_annotations
    }
    
    # Save new annotations
    with open(dest_json, 'w') as f:
        json.dump(new_data, f)
    
    # Copy selected images
    copied_count = 0
    for img in selected_images:
        source_path = source_images / img['file_name']
        dest_path = dest_images / img['file_name']
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            copied_count += 1
    
    print(f"  - Selected {len(selected_images)} images from {len(data['images'])}")
    print(f"  - Filtered to {len(filtered_annotations)} annotations")
    print(f"  - Copied {copied_count} image files")

if __name__ == "__main__":
    create_test_dataset()
