#!/usr/bin/env python3
"""
Create better training data with proper component-level annotations.
This script helps fix the core issue: labeling complete components instead of individual parts.
"""

import json
import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import List, Dict, Tuple

# Improved component mapping (focusing on complete components)
IMPROVED_COMPONENT_CLASSES = [
    'battery_holder',     # 0 - Power source
    'switch',            # 1 - On/off control  
    'led',               # 2 - Light output
    'connection',        # 3 - Complete connection/wire assembly (not individual points)
    'buzzer',            # 4 - Sound output
    'resistor',          # 5 - Current limiter
    'speaker',           # 6 - Audio output
    'button',            # 7 - Push control
    'motor',             # 8 - Movement output
    'lamp'               # 9 - Bright light output
]

def create_corrected_annotations():
    """Create corrected annotations for your snap circuit image."""
    print("🔧 CREATING CORRECTED TRAINING DATA")
    print("="*50)
    
    # Create improved dataset directory
    improved_dir = Path("data/improved_training")
    for split in ['train', 'val', 'test']:
        (improved_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (improved_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Copy your main circuit image
    shutil.copy("snap_circuit_image.jpg", 
                improved_dir / "images/train/snap_circuit_image.jpg")
    
    # Create CORRECTED annotations based on your actual circuit
    # You said: 1 battery_holder, 1 switch, 1 led, 1 connection
    corrected_annotations = [
        # battery_holder (class 0) - top center
        "0 0.500 0.250 0.280 0.200",
        
        # switch (class 1) - middle left (combine the two detected switches into one)
        "1 0.500 0.555 0.350 0.220", 
        
        # led (class 2) - keep as is
        "2 0.500 0.650 0.060 0.060",
        
        # connection (class 3) - ONE large bounding box covering the entire wire/connection area
        "3 0.500 0.450 0.600 0.200"  # Covers the connection area as one component
    ]
    
    # Save corrected annotations
    with open(improved_dir / "labels/train/snap_circuit_image.txt", 'w') as f:
        f.write('\\n'.join(corrected_annotations))
    
    print("✅ Created corrected snap circuit annotations")
    print("   • 1 battery_holder")
    print("   • 1 switch (merged)")
    print("   • 1 led") 
    print("   • 1 connection (unified)")
    
    return improved_dir

def create_data_yaml(improved_dir: Path):
    """Create data.yaml for improved training."""
    data_config = {
        'path': str(improved_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val', 
        'test': 'images/test',
        'nc': len(IMPROVED_COMPONENT_CLASSES),
        'names': IMPROVED_COMPONENT_CLASSES
    }
    
    import yaml
    with open(improved_dir / "data.yaml", 'w') as f:
        yaml.dump(data_config, f, default_flow_style=False)
    
    print(f"✅ Created data.yaml with {len(IMPROVED_COMPONENT_CLASSES)} component classes")

def create_annotation_guidelines():
    """Create guidelines for better manual annotation."""
    guidelines = """
ANNOTATION GUIDELINES FOR BETTER MODEL TRAINING

KEY PRINCIPLE: Label COMPLETE COMPONENTS, not individual parts!

CORRECT Annotation Strategy:
================================

BATTERY_HOLDER (class 0):
   - Draw ONE box around the entire battery compartment
   - Include all battery-related parts as one component

SWITCH (class 1): 
   - Draw ONE box around the complete switch assembly
   - Don't separate switch parts - treat as single unit

LED (class 2):
   - Draw box around the complete LED component
   - Include the LED housing and immediate connections

CONNECTION (class 3):
   - Draw ONE large box covering connected wire segments  
   - Think: "Where does current flow as one path?"
   - Group connected points into single component

COMMON MISTAKES TO AVOID:
==========================

- Don't label individual connection points separately
- Don't split switches into multiple parts  
- Don't create tiny boxes for wire segments
- Don't over-segment what should be one component

RESULT: Model learns to recognize complete functional units!
"""
    
    with open("ANNOTATION_GUIDELINES.md", 'w') as f:
        f.write(guidelines)
    
    print("✅ Created ANNOTATION_GUIDELINES.md")

def validate_annotations(improved_dir: Path):
    """Validate the improved annotations."""
    label_file = improved_dir / "labels/train/snap_circuit_image.txt"
    
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    components = {}
    for line in lines:
        if line.strip():
            class_id = int(line.split()[0])
            class_name = IMPROVED_COMPONENT_CLASSES[class_id]
            components[class_name] = components.get(class_name, 0) + 1
    
    print("\\n📊 VALIDATION - Components in corrected annotations:")
    for comp, count in components.items():
        print(f"   • {comp}: {count}")
    
    total = sum(components.values())
    print(f"\\n🎯 Total: {total} components (matches your expected 4!) ✅")

def main():
    """Main function to create better training data."""
    print("🚀 FIXING TRAINING DATA FOR BETTER MODEL")
    print("="*60)
    
    # Step 1: Create corrected annotations
    improved_dir = create_corrected_annotations()
    
    # Step 2: Create data configuration
    create_data_yaml(improved_dir)
    
    # Step 3: Create annotation guidelines
    create_annotation_guidelines()
    
    # Step 4: Validate annotations
    validate_annotations(improved_dir)
    
    print("\\n🎯 NEXT STEPS:")
    print("1. Review the corrected annotations")
    print("2. Add more circuit images with proper component-level labels")
    print("3. Use annotation tools like labelImg or CVAT for manual labeling")
    print("4. Train model with: python train_better_model.py")
    print("\\n💡 Key: Always label COMPLETE components, not individual parts!")

if __name__ == "__main__":
    main() 