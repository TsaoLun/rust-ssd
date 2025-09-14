#!/usr/bin/env python3
"""
Filter COCO dataset to only include images with specific categories.
This script filters the COCO dataset to create a subset containing only images
that have annotations for the specified categories (bicycle and car).
"""

import json
import os
import shutil
import argparse
from pathlib import Path


def filter_coco_annotations(annotation_file, output_file, target_categories, image_dir, output_image_dir):
    """
    Filter COCO annotations to only include specified categories.
    
    Args:
        annotation_file: Path to input COCO annotation JSON file
        output_file: Path to output filtered annotation JSON file
        target_categories: List of category names to keep
        image_dir: Directory containing original images
        output_image_dir: Directory to copy filtered images to
    """
    
    print(f"Loading annotations from {annotation_file}...")
    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)
    
    # Get category IDs for target categories
    target_category_ids = []
    category_name_to_id = {}
    
    for category in coco_data['categories']:
        category_name_to_id[category['name']] = category['id']
        if category['name'] in target_categories:
            target_category_ids.append(category['id'])
            print(f"Found target category: {category['name']} (ID: {category['id']})")
    
    if not target_category_ids:
        print("Warning: No target categories found in dataset!")
        return
    
    print(f"Target category IDs: {target_category_ids}")
    
    # Filter annotations to only include target categories
    filtered_annotations = []
    valid_image_ids = set()
    
    for annotation in coco_data['annotations']:
        if annotation['category_id'] in target_category_ids:
            filtered_annotations.append(annotation)
            valid_image_ids.add(annotation['image_id'])
    
    print(f"Found {len(filtered_annotations)} annotations for target categories")
    print(f"Found {len(valid_image_ids)} images with target categories")
    
    # Filter images to only include those with valid annotations
    filtered_images = []
    copied_images = 0
    
    # Create output image directory
    os.makedirs(output_image_dir, exist_ok=True)
    
    for image in coco_data['images']:
        if image['id'] in valid_image_ids:
            filtered_images.append(image)
            
            # Copy image file
            src_path = os.path.join(image_dir, image['file_name'])
            dst_path = os.path.join(output_image_dir, image['file_name'])
            
            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_images += 1
            else:
                print(f"Warning: Image file not found: {src_path}")
    
    print(f"Copied {copied_images} images to {output_image_dir}")
    
    # Filter categories to only include target categories
    filtered_categories = []
    for category in coco_data['categories']:
        if category['id'] in target_category_ids:
            filtered_categories.append(category)
    
    # Create filtered dataset
    filtered_data = {
        'info': coco_data['info'],
        'licenses': coco_data['licenses'],
        'images': filtered_images,
        'annotations': filtered_annotations,
        'categories': filtered_categories
    }
    
    # Save filtered annotations
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Saving filtered annotations to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f)
    
    print("Filtering complete!")
    print(f"  Original: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    print(f"  Filtered: {len(filtered_images)} images, {len(filtered_annotations)} annotations")
    print(f"  Categories: {[cat['name'] for cat in filtered_categories]}")


def main():
    parser = argparse.ArgumentParser(description='Filter COCO dataset for specific categories')
    parser.add_argument('--dataset-root', required=True, help='Root directory of COCO dataset')
    parser.add_argument('--output-root', required=True, help='Output directory for filtered dataset')
    parser.add_argument('--categories', nargs='+', default=['bicycle', 'car'], 
                       help='Categories to include (default: bicycle car)')
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    output_root = Path(args.output_root)
    
    # Filter training set
    print("=== Filtering Training Set ===")
    filter_coco_annotations(
        annotation_file=dataset_root / 'annotations' / 'instances_train2017.json',
        output_file=output_root / 'annotations' / 'instances_train2017.json',
        target_categories=args.categories,
        image_dir=dataset_root / 'train2017',
        output_image_dir=output_root / 'train2017'
    )
    
    print("\n=== Filtering Validation Set ===")
    filter_coco_annotations(
        annotation_file=dataset_root / 'annotations' / 'instances_val2017.json',
        output_file=output_root / 'annotations' / 'instances_val2017.json',
        target_categories=args.categories,
        image_dir=dataset_root / 'val2017',
        output_image_dir=output_root / 'val2017'
    )
    
    print(f"\nFiltered dataset saved to: {output_root}")
    print(f"You can now train with: --r {output_root}")


if __name__ == '__main__':
    main()
