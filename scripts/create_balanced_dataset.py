#!/usr/bin/env python3
"""
为不平衡数据集创建平衡的训练策略
"""
import json
import random
from pathlib import Path
from collections import defaultdict

def create_balanced_training_strategy(annotations_path, output_path, target_ratio=3.0):
    """
    创建平衡的训练策略，通过重采样少数类别
    
    Args:
        annotations_path: COCO标注文件路径
        output_path: 输出的平衡标注文件路径
        target_ratio: 目标的多数类/少数类比例
    """
    with open(annotations_path) as f:
        data = json.load(f)
    
    # 按类别分组图片
    category_images = defaultdict(set)
    image_annotations = defaultdict(list)
    
    # 收集每个类别的图片ID
    for ann in data['annotations']:
        cat_id = ann['category_id']
        img_id = ann['image_id']
        category_images[cat_id].add(img_id)
        image_annotations[img_id].append(ann)
    
    # 找到bicycle和car的图片
    bicycle_cat_id = None
    car_cat_id = None
    for cat in data['categories']:
        if cat['name'] == 'bicycle':
            bicycle_cat_id = cat['id']
        elif cat['name'] == 'car':
            car_cat_id = cat['id']
    
    if not bicycle_cat_id or not car_cat_id:
        raise ValueError("未找到bicycle或car类别")
    
    bicycle_images = list(category_images[bicycle_cat_id])
    car_images = list(category_images[car_cat_id])
    
    print(f"原始数据: bicycle图片 {len(bicycle_images)}, car图片 {len(car_images)}")
    
    # 计算需要的bicycle图片数量
    target_bicycle_count = int(len(car_images) / target_ratio)
    
    # 如果bicycle图片不够，进行重采样
    if len(bicycle_images) < target_bicycle_count:
        # 重复采样bicycle图片
        multiplier = target_bicycle_count // len(bicycle_images)
        remainder = target_bicycle_count % len(bicycle_images)
        
        balanced_bicycle_images = bicycle_images * multiplier
        balanced_bicycle_images.extend(random.sample(bicycle_images, remainder))
        
        print(f"平衡后: bicycle图片 {len(balanced_bicycle_images)} (重采样), car图片 {len(car_images)}")
    else:
        # 如果够用，随机采样
        balanced_bicycle_images = random.sample(bicycle_images, target_bicycle_count)
        print(f"平衡后: bicycle图片 {len(balanced_bicycle_images)} (下采样), car图片 {len(car_images)}")
    
    # 合并所有需要的图片ID
    all_image_ids = set(balanced_bicycle_images + car_images)
    
    # 创建新的数据集
    new_data = {
        'images': [img for img in data['images'] if img['id'] in all_image_ids],
        'annotations': [ann for ann in data['annotations'] if ann['image_id'] in all_image_ids],
        'categories': data['categories'],
        'info': data.get('info', {}),
        'licenses': data.get('licenses', [])
    }
    
    # 保存平衡的数据集
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    print(f"平衡数据集已保存到: {output_path}")
    print(f"最终图片数: {len(new_data['images'])}")
    print(f"最终标注数: {len(new_data['annotations'])}")

def main():
    """创建平衡的filtered_datasets"""
    
    # 输入和输出路径
    input_train = "/Users/ele/Projects/rust-ssd/assets/filtered_datasets/annotations/instances_train2017.json"
    input_val = "/Users/ele/Projects/rust-ssd/assets/filtered_datasets/annotations/instances_val2017.json"
    
    output_dir = Path("/Users/ele/Projects/rust-ssd/assets/balanced_datasets/annotations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_train = output_dir / "instances_train2017.json"
    output_val = output_dir / "instances_val2017.json"
    
    print("创建平衡的训练数据集...")
    create_balanced_training_strategy(input_train, output_train, target_ratio=3.0)
    
    print("\n创建平衡的验证数据集...")
    create_balanced_training_strategy(input_val, output_val, target_ratio=3.0)
    
    print("\n完成！现在可以使用balanced_datasets进行训练了。")
    print("建议的训练命令:")
    print("cargo run --release -- --o bicycle,car train --r assets/balanced_datasets/ --c 0")

if __name__ == "__main__":
    main()
