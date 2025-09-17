#!/usr/bin/env python3
"""
Check Training Annotations for Symmetry Issues

Analyzes your training annotations to see if there's an imbalance 
between left and right side annotations.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict


def analyze_annotations():
    """Analyze dual board training annotations for symmetry"""
    
    print("🔍 TRAINING ANNOTATION ANALYSIS")
    print("=" * 50)
    
    # Check for annotation directories
    annotation_dirs = [
        "dual_board_annotations",
        "annotations", 
        "data/training",
        "dual_board_augmented_dataset/labels/train"
    ]
    
    found_dir = None
    for dir_path in annotation_dirs:
        if Path(dir_path).exists():
            found_dir = Path(dir_path)
            print(f"📂 Found annotations in: {dir_path}")
            break
    
    if not found_dir:
        print("❌ No annotation directory found!")
        print("   Looked for:")
        for dir_path in annotation_dirs:
            print(f"   • {dir_path}")
        return
    
    # Find all annotation files
    txt_files = list(found_dir.glob("*.txt"))
    print(f"📄 Found {len(txt_files)} annotation files")
    
    if len(txt_files) == 0:
        print("❌ No .txt annotation files found!")
        return
    
    # Load class names
    class_names = [
        'wire', 'switch', 'button', 'battery_holder', 'led', 'speaker', 
        'music_circuit', 'motor', 'resistor', 'connection_node', 'lamp', 
        'fan', 'buzzer', 'photoresistor', 'microphone', 'alarm'
    ]
    
    # Analyze annotations
    left_components = defaultdict(int)
    right_components = defaultdict(int)
    total_components = defaultdict(int)
    
    image_count = 0
    total_annotations = 0
    
    for txt_file in txt_files:
        try:
            with open(txt_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 0:
                image_count += 1
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    
                    if class_id < len(class_names):
                        class_name = class_names[class_id]
                        total_components[class_name] += 1
                        total_annotations += 1
                        
                        # Categorize by side (x_center < 0.5 = left, >= 0.5 = right)
                        if x_center < 0.5:
                            left_components[class_name] += 1
                        else:
                            right_components[class_name] += 1
                
        except Exception as e:
            print(f"⚠️  Error reading {txt_file.name}: {e}")
    
    print(f"\n📊 TRAINING DATA SUMMARY:")
    print(f"   Images annotated: {image_count}")
    print(f"   Total annotations: {total_annotations}")
    print(f"   Average annotations per image: {total_annotations/max(image_count,1):.1f}")
    
    print(f"\n🔍 COMPONENT DISTRIBUTION ANALYSIS:")
    print(f"{'Component':<15} {'Total':<8} {'Left':<8} {'Right':<8} {'Balance':<10}")
    print("-" * 55)
    
    symmetric_issues = []
    
    for component in sorted(total_components.keys()):
        total = total_components[component]
        left = left_components[component]
        right = right_components[component]
        
        if total > 0:
            left_ratio = left / total
            balance_status = "BALANCED"
            
            if left_ratio < 0.3:
                balance_status = "RIGHT-HEAVY"
                symmetric_issues.append(f"{component}: {left} left vs {right} right")
            elif left_ratio > 0.7:
                balance_status = "LEFT-HEAVY"
                symmetric_issues.append(f"{component}: {left} left vs {right} right")
            
            print(f"{component:<15} {total:<8} {left:<8} {right:<8} {balance_status:<10}")
    
    print(f"\n🎯 SYMMETRY ANALYSIS:")
    if len(symmetric_issues) == 0:
        print("   ✅ All components are reasonably balanced between left/right sides")
    else:
        print("   ❌ ASYMMETRY DETECTED in training data:")
        for issue in symmetric_issues:
            print(f"      • {issue}")
    
    # Focus on battery holders specifically
    bh_left = left_components['battery_holder']
    bh_right = right_components['battery_holder']
    bh_total = total_components['battery_holder']
    
    print(f"\n🔋 BATTERY HOLDER SPECIFIC ANALYSIS:")
    print(f"   Total battery_holder annotations: {bh_total}")
    print(f"   Left side: {bh_left} ({bh_left/max(bh_total,1)*100:.1f}%)")
    print(f"   Right side: {bh_right} ({bh_right/max(bh_total,1)*100:.1f}%)")
    
    if bh_left == 0:
        print(f"   🚨 CRITICAL: NO battery_holder annotations on left side!")
        print(f"   🔧 This explains why your model can't detect left battery holders!")
    elif bh_right == 0:
        print(f"   🚨 CRITICAL: NO battery_holder annotations on right side!")
    elif abs(bh_left - bh_right) > max(bh_left, bh_right) * 0.5:
        print(f"   ⚠️  IMBALANCED: Significant difference between left/right sides")
    else:
        print(f"   ✅ Battery holder annotations are reasonably balanced")
    
    # Recommendations
    print(f"\n💡 RECOMMENDATIONS:")
    if len(symmetric_issues) > 0:
        print("   🔄 Consider retraining with more balanced annotations")
        print("   📸 Add more images with components on under-represented sides")
        
        if bh_left == 0:
            print("   🔋 URGENT: Add battery_holder annotations for LEFT side")
        if any("battery_holder" in issue for issue in symmetric_issues):
            print("   🔋 Focus on balancing battery_holder annotations specifically")
    else:
        print("   ✅ Training data looks balanced - issue might be physical setup")
        print("   📹 Check current camera setup matches training data positioning")


if __name__ == "__main__":
    analyze_annotations()
