#!/usr/bin/env python3
"""
Test script for your own circuit images with the precision-focused model.
"""

import cv2
import os
import sys
from pathlib import Path
from main import SnapCircuitVisionSystem

def test_user_images(image_folder: str):
    """Test all images in a folder with the precision-focused model."""
    
    # Initialize vision system (uses precision-focused model from config)
    system = SnapCircuitVisionSystem(
        save_outputs=True,
        display_results=True
    )
    
    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_folder = Path(image_folder)
    
    if not image_folder.exists():
        print(f"❌ Folder not found: {image_folder}")
        return
    
    image_files = [f for f in image_folder.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ No images found in {image_folder}")
        return
    
    print(f"🔍 Found {len(image_files)} images to test")
    print("Press any key to continue to next image, 'q' to quit")
    
    results = []
    
    for i, image_file in enumerate(image_files):
        print(f"\n📸 [{i+1}/{len(image_files)}] Testing: {image_file.name}")
        
        try:
            # Process image
            result = system.process_image(str(image_file))
            
            # Print results
            components = result.connection_graph.components
            circuit_closed = result.connection_graph.state.is_circuit_closed
            
            print(f"   ✅ Found {len(components)} components")
            print(f"   🔌 Circuit closed: {circuit_closed}")
            
            if components:
                component_types = [comp.component_type.value for comp in components]
                print(f"   📋 Components: {', '.join(component_types)}")
            
            print(f"   ⏱️  Processing time: {result.processing_time:.3f}s")
            
            results.append({
                'file': image_file.name,
                'components_found': len(components),
                'circuit_closed': circuit_closed,
                'processing_time': result.processing_time
            })
            
            # Wait for user input (image is displayed)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                print("🛑 Testing stopped by user")
                break
                
        except Exception as e:
            print(f"   ❌ Error processing {image_file.name}: {e}")
    
    cv2.destroyAllWindows()
    
    # Summary
    print(f"\n📊 TESTING SUMMARY")
    print("=" * 50)
    total_components = sum(r['components_found'] for r in results)
    avg_time = sum(r['processing_time'] for r in results) / len(results) if results else 0
    
    print(f"Images processed: {len(results)}")
    print(f"Total components found: {total_components}")
    print(f"Average processing time: {avg_time:.3f}s")
    print(f"Results saved to: output/frames/")
    
    return results

def main():
    """Main function."""
    if len(sys.argv) != 2:
        print("Usage: python test_your_images.py <image_folder>")
        print("\nExample:")
        print("  python test_your_images.py my_circuit_photos/")
        print("  python test_your_images.py C:/Users/me/Pictures/circuits/")
        return
    
    image_folder = sys.argv[1]
    test_user_images(image_folder)

if __name__ == "__main__":
    main() 