#!/usr/bin/env python3
"""
Create a small subset of the filtered dataset for fast training experiments.
This will take only 10% of the training data and 20% of validation data.
"""

import json
import shutil
import random
from pathlib import Path

def create_small_dataset():
    """Create a smaller dataset for faster training"""
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Source and destination paths
    source_root = Path("./assets/filtered_datasets")
    dest_root = Path("./assets/small_datasets")
    
    # Create destination directories
    for subdir in ["annotations", "train2017", "val2017"]:
        (dest_root / subdir).mkdir(parents=True, exist_ok=True)
    
    # Process training set (10% of data)
    print("Processing training set...")
    process_split(
        source_root / "annotations/instances_train2017.json",
        dest_root / "annotations/instances_train2017.json",
        source_root / "train2017",
        dest_root / "train2017",
        sample_ratio=0.1
    )
    
    # Process validation set (20% of data)
    print("Processing validation set...")
    process_split(
        source_root / "annotations/instances_val2017.json",
        dest_root / "annotations/instances_val2017.json",
        source_root / "val2017",
        dest_root / "val2017",
        sample_ratio=0.2
    )
    
    print("Small dataset creation completed!")

def process_split(source_json, dest_json, source_images, dest_images, sample_ratio):
    """Process a single split (train/val)"""
    
    # Load annotations
    with open(source_json, 'r') as f:
        data = json.load(f)
    
    # Sample images
    original_images = data['images']
    sampled_images = random.sample(original_images, int(len(original_images) * sample_ratio))
    sampled_image_ids = {img['id'] for img in sampled_images}
    
    # Filter annotations for sampled images
    filtered_annotations = [
        ann for ann in data['annotations'] 
        if ann['image_id'] in sampled_image_ids
    ]
    
    # Create new annotation file
    new_data = {
        'info': data['info'],
        'licenses': data['licenses'],
        'categories': data['categories'],
        'images': sampled_images,
        'annotations': filtered_annotations
    }
    
    # Save new annotations
    with open(dest_json, 'w') as f:
        json.dump(new_data, f)
    
    # Copy sampled images
    copied_count = 0
    for img in sampled_images:
        source_path = source_images / img['file_name']
        dest_path = dest_images / img['file_name']
        if source_path.exists():
            shutil.copy2(source_path, dest_path)
            copied_count += 1
    
    print(f"  - Sampled {len(sampled_images)} images from {len(original_images)}")
    print(f"  - Filtered to {len(filtered_annotations)} annotations")
    print(f"  - Copied {copied_count} image files")

if __name__ == "__main__":
    create_small_dataset()
