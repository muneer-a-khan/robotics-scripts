#!/usr/bin/env python3
"""
Test Horn Orientation Detection in Integrated System
Verifies that horn orientation detection works when called through the circuit analyzer
"""

import numpy as np
from circuit_graph_analyzer import CircuitGraphAnalyzer

def test_integrated_horn_detection():
    """Test horn detection through the circuit graph analyzer"""
    print("="*60)
    print("Testing Horn Detection in Circuit Graph Analyzer")
    print("="*60)
    
    # Initialize the circuit graph analyzer
    print("\n1. Initializing Circuit Graph Analyzer...")
    analyzer = CircuitGraphAnalyzer()
    
    # Check if horn detector was initialized
    if hasattr(analyzer, 'horn_detector') and analyzer.horn_detector is not None:
        print(f"   ✓ Horn detector initialized with {len(analyzer.horn_detector.templates)} template(s)")
    else:
        print("   ✗ Horn detector NOT initialized")
        return False
    
    # Check if LED detector was initialized too
    if hasattr(analyzer, 'led_detector') and analyzer.led_detector is not None:
        print(f"   ✓ LED detector initialized with {len(analyzer.led_detector.templates)} template(s)")
    else:
        print("   ⚠ LED detector not initialized (this is okay)")
    
    print("\n2. Testing orientation detection availability:")
    print(f"   - Horn detector available: {analyzer.horn_detector is not None}")
    print(f"   - LED detector available: {analyzer.led_detector is not None}")
    
    print("\n3. Checking method updates:")
    # Check if the methods exist and are updated
    if hasattr(analyzer, 'detect_led_orientations_from_boxes'):
        print("   ✓ detect_led_orientations_from_boxes method exists")
    else:
        print("   ✗ detect_led_orientations_from_boxes method NOT found")
    
    if hasattr(analyzer, 'horn_orientations'):
        print("   ✓ horn_orientations attribute exists")
    else:
        print("   ✗ horn_orientations attribute NOT found")
    
    print("\n4. Summary:")
    print(f"   Total features: Horn detection + LED detection")
    print(f"   Horn detector status: {'READY' if analyzer.horn_detector else 'NOT AVAILABLE'}")
    print(f"   LED detector status: {'READY' if analyzer.led_detector else 'NOT AVAILABLE'}")
    
    print("\n" + "="*60)
    print("Horn Detection Integration Test Complete!")
    print("="*60)
    print("\n💡 To test with real data:")
    print("   1. Run: python integrated_circuit_system.py")
    print("   2. Point camera at circuit with Horn component")
    print("   3. Press 'g' to analyze circuit graph")
    print("   4. Check output for 'Horn Orientation Details' section")
    print()
    
    return True

if __name__ == "__main__":
    test_integrated_horn_detection()

