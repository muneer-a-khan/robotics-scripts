#!/usr/bin/env python3
"""
Create Scaled Versions of Red Plus Templates
Creates plus_small and plus_tiny versions like the white plus templates
"""

import cv2
import numpy as np
from pathlib import Path

def create_scaled_templates():
    """Create small and tiny versions of red plus templates"""
    print("="*60)
    print("Creating Scaled Red Plus Templates")
    print("="*60)
    
    template_folder = Path("red_plus_folder")
    
    # Find the FullSizeRender images
    template_files = list(template_folder.glob("FullSizeRender*.png"))
    
    if not template_files:
        print(f"❌ No FullSizeRender files found in {template_folder}")
        return
    
    # Use the first one as the base
    base_template_file = template_files[0]
    print(f"\nUsing base template: {base_template_file.name}")
    
    template = cv2.imread(str(base_template_file))
    if template is None:
        print(f"❌ Could not load {base_template_file}")
        return
    
    print(f"Original size: {template.shape[1]}x{template.shape[0]}")
    
    # Create plus_small (similar scale to white_plus_folder/plus_small.png which is 102x135)
    # Scale down to around 100-150 pixels width
    target_small_width = 120
    scale_small = target_small_width / template.shape[1]
    small_height = int(template.shape[0] * scale_small)
    plus_small = cv2.resize(template, (target_small_width, small_height), interpolation=cv2.INTER_AREA)
    
    small_path = template_folder / "plus_small.png"
    cv2.imwrite(str(small_path), plus_small)
    print(f"✓ Created plus_small.png: {plus_small.shape[1]}x{plus_small.shape[0]}")
    
    # Create plus_tiny (similar scale to white_plus_folder/plus_tiny.png which is 51x68)
    # Scale down to around 50-60 pixels width
    target_tiny_width = 55
    scale_tiny = target_tiny_width / template.shape[1]
    tiny_height = int(template.shape[0] * scale_tiny)
    plus_tiny = cv2.resize(template, (target_tiny_width, tiny_height), interpolation=cv2.INTER_AREA)
    
    tiny_path = template_folder / "plus_tiny.png"
    cv2.imwrite(str(tiny_path), plus_tiny)
    print(f"✓ Created plus_tiny.png: {plus_tiny.shape[1]}x{plus_tiny.shape[0]}")
    
    print("\n" + "="*60)
    print("Scaled Templates Created Successfully!")
    print("="*60)
    print(f"\nNow you have {len(list(template_folder.glob('*.png')))} red plus templates:")
    for f in sorted(template_folder.glob("*.png")):
        img = cv2.imread(str(f))
        if img is not None:
            print(f"  - {f.name}: {img.shape[1]}x{img.shape[0]}")
    
    print("\nThis will help detect red plus signs at different distances/scales!")
    print()

if __name__ == "__main__":
    create_scaled_templates()

