#!/usr/bin/env python3
"""
Test Horn Orientation Detection
Quick test to verify horn orientation detection is working properly
"""

import cv2
import numpy as np
from horn_orientation_detector import HornOrientationDetector

def test_horn_detector():
    """Test the horn orientation detector with basic functionality"""
    print("="*60)
    print("Testing Horn Orientation Detector")
    print("="*60)
    
    # Initialize detector
    print("\n1. Initializing Horn Orientation Detector...")
    detector = HornOrientationDetector()
    print(f"   ✓ Loaded {len(detector.templates)} template(s)")
    
    if len(detector.templates) == 0:
        print("   ✗ ERROR: No templates loaded!")
        print("   Make sure the red_plus_folder exists with plus sign images")
        return False
    
    # Show template information
    print("\n2. Template Information:")
    for template_info in detector.templates:
        print(f"   - {template_info['name']}: {template_info['width']}x{template_info['height']} pixels")
    
    print("\n3. Testing with mock image...")
    # Create a simple test image (white background with a dark rectangle representing a horn)
    test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Draw a horizontal "horn" component (dark rectangle)
    cv2.rectangle(test_image, (150, 150), (450, 250), (100, 100, 100), -1)
    
    # Test bounding box for a horizontal horn
    test_bbox = [150, 150, 450, 250]
    
    print(f"   Testing with bbox: {test_bbox}")
    result = detector.detect_orientation(test_image, test_bbox, debug=False)
    
    print(f"\n4. Detection Result:")
    print(f"   Orientation: {result['orientation']}")
    print(f"   Confidence: {result['confidence']:.2f}")
    if 'horn_orientation' in result:
        print(f"   Horn Orientation: {result['horn_orientation']}")
    if 'plus_position' in result and result['plus_position']:
        print(f"   '+' Position: {result['plus_position']}")
    if 'reason' in result:
        print(f"   Reason: {result['reason']}")
    
    print("\n" + "="*60)
    print("Horn Orientation Detector Test Complete!")
    print("="*60)
    
    return True

if __name__ == "__main__":
    test_horn_detector()

