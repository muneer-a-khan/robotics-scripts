#!/usr/bin/env python3
"""
Debug Red Extraction
Visualize how red plus extraction is working on the templates
"""

import cv2
import numpy as np
from pathlib import Path

def debug_red_extraction():
    """Show what the red extraction does to the templates"""
    print("="*60)
    print("Debug Red Plus Extraction")
    print("="*60)
    
    template_folder = Path("red_plus_folder")
    template_files = list(template_folder.glob("*.png")) + list(template_folder.glob("*.jpg"))
    
    if not template_files:
        print(f"❌ No templates found in {template_folder}")
        return
    
    print(f"\n✓ Found {len(template_files)} template(s)")
    
    for i, template_file in enumerate(template_files):
        print(f"\nProcessing: {template_file.name}")
        template = cv2.imread(str(template_file))
        
        if template is None:
            print(f"  ❌ Could not load {template_file.name}")
            continue
        
        # Show original
        print(f"  Original size: {template.shape[1]}x{template.shape[0]}")
        
        # Extract red using the new method
        b, g, r = cv2.split(template)
        red_enhanced = cv2.subtract(r, cv2.addWeighted(g, 0.5, b, 0.5, 0))
        
        # Apply threshold (use same value as actual detector)
        _, red_binary = cv2.threshold(red_enhanced, 80, 255, cv2.THRESH_BINARY)
        
        # Blur slightly
        red_processed = cv2.GaussianBlur(red_binary, (3, 3), 0)
        
        # Save processed versions
        output_name = f"debug_red_extraction_{i}_{template_file.stem}.jpg"
        
        # Create side-by-side comparison
        # Resize for display if too large
        max_height = 400
        if template.shape[0] > max_height:
            scale = max_height / template.shape[0]
            new_width = int(template.shape[1] * scale)
            template_resized = cv2.resize(template, (new_width, max_height))
            red_processed_resized = cv2.resize(red_processed, (new_width, max_height))
        else:
            template_resized = template
            red_processed_resized = red_processed
        
        # Convert grayscale to BGR for concatenation
        red_processed_bgr = cv2.cvtColor(red_processed_resized, cv2.COLOR_GRAY2BGR)
        
        # Concatenate side by side
        comparison = np.hstack([template_resized, red_processed_bgr])
        
        cv2.imwrite(output_name, comparison)
        print(f"  ✓ Saved comparison to: {output_name}")
        
        # Count non-zero pixels (red regions found)
        red_pixels = np.count_nonzero(red_processed)
        total_pixels = red_processed.shape[0] * red_processed.shape[1]
        red_percentage = (red_pixels / total_pixels) * 100
        print(f"  Red regions: {red_pixels}/{total_pixels} pixels ({red_percentage:.1f}%)")
    
    print("\n" + "="*60)
    print("Red Extraction Debug Complete!")
    print("="*60)
    print("\nCheck the debug_red_extraction_*.jpg files to see:")
    print("  LEFT: Original template image")
    print("  RIGHT: Red-extracted binary version")
    print("\nThe RIGHT side shows what the detector is actually matching against.")
    print("If the plus sign isn't clearly visible on the right, try:")
    print("  1. Taking new photos with better lighting")
    print("  2. Ensuring the red plus is clearly visible")
    print("  3. Adjusting the threshold value (currently 30)")
    print()

if __name__ == "__main__":
    debug_red_extraction()

