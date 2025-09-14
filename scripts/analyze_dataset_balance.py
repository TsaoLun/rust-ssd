#!/usr/bin/env python3
"""
脚本用于分析数据集不平衡情况并建议解决方案
"""
import json
from pathlib import Path

def analyze_dataset_balance(annotations_path, categories=['bicycle', 'car']):
    """分析数据集的类别平衡情况"""
    with open(annotations_path) as f:
        data = json.load(f)
    
    # 统计各类别的标注数量
    cat_counts = {}
    for ann in data['annotations']:
        cat_id = ann['category_id']
        cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1
    
    # 获取目标类别的统计
    target_stats = {}
    for cat in data['categories']:
        if cat['name'] in categories:
            count = cat_counts.get(cat['id'], 0)
            target_stats[cat['name']] = {
                'id': cat['id'],
                'count': count
            }
    
    return target_stats, len(data['images']), len(data['annotations'])

def calculate_class_weights(stats):
    """计算类别权重（逆频率）"""
    total_samples = sum(stat['count'] for stat in stats.values())
    num_classes = len(stats) + 1  # +1 for background
    
    weights = {'background': 1.0}  # Background weight
    for name, stat in stats.items():
        if stat['count'] > 0:
            weight = total_samples / (num_classes * stat['count'])
            weights[name] = weight
        else:
            weights[name] = 1.0
    
    return weights

def recommend_solutions(stats):
    """基于数据分析推荐解决方案"""
    counts = [stat['count'] for stat in stats.values()]
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    recommendations = []
    
    if imbalance_ratio > 5:
        recommendations.append("🚨 严重的类别不平衡问题（比例 > 5:1）")
        recommendations.append("建议使用 Focal Loss (alpha=0.25, gamma=2.0)")
        recommendations.append("建议使用类别权重")
        recommendations.append("考虑数据增强策略")
    elif imbalance_ratio > 3:
        recommendations.append("⚠️  中等程度的类别不平衡（比例 > 3:1）")
        recommendations.append("建议使用类别权重")
        recommendations.append("可以考虑 Focal Loss")
    else:
        recommendations.append("✅ 相对平衡的数据集")
        recommendations.append("可以使用标准的交叉熵损失")
    
    return recommendations, imbalance_ratio

def main():
    datasets = {
        'small_datasets': '/Users/ele/Projects/rust-ssd/assets/small_datasets/annotations/instances_train2017.json',
        'filtered_datasets': '/Users/ele/Projects/rust-ssd/assets/filtered_datasets/annotations/instances_train2017.json'
    }
    
    print("=" * 60)
    print("数据集类别平衡分析报告")
    print("=" * 60)
    
    for dataset_name, ann_path in datasets.items():
        if not Path(ann_path).exists():
            print(f"⚠️  {dataset_name}: 文件不存在 - {ann_path}")
            continue
            
        print(f"\n📊 {dataset_name.upper()}")
        print("-" * 40)
        
        stats, num_images, num_annotations = analyze_dataset_balance(ann_path)
        weights = calculate_class_weights(stats)
        recommendations, ratio = recommend_solutions(stats)
        
        print(f"图片数量: {num_images:,}")
        print(f"总标注数: {num_annotations:,}")
        print("\n类别统计:")
        for name, stat in stats.items():
            print(f"  {name}: {stat['count']:,} 个标注")
        
        print(f"\n不平衡比例: {ratio:.2f}:1")
        
        print("\n建议的类别权重:")
        for name, weight in weights.items():
            print(f"  {name}: {weight:.3f}")
        
        print("\n推荐方案:")
        for rec in recommendations:
            print(f"  {rec}")
    
    print("\n" + "=" * 60)
    print("配置建议:")
    print("=" * 60)
    print("1. 对于 small_datasets (比例 5.9:1):")
    print("   - 使用类别权重")
    print("   - batch_size: 16-24")
    print("   - learning_rate: 0.0001")
    
    print("\n2. 对于 filtered_datasets (比例 6.17:1):")
    print("   - 使用 Focal Loss + 类别权重")
    print("   - batch_size: 8-16 (由于数据量大)")
    print("   - learning_rate: 0.00005 (较低学习率)")
    print("   - 考虑使用梯度累积")

if __name__ == "__main__":
    main()
