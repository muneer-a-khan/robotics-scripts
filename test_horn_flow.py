#!/usr/bin/env python3
"""
Test Horn Detection Flow
Debug script to understand the complete flow of horn detection
"""

from circuit_graph_analyzer import CircuitGraphAnalyzer
from pathlib import Path
from ultralytics import YOLO
from model_class_renamer import rename_model_classes
import cv2

def test_horn_flow():
    """Test the complete horn detection flow"""
    print("="*60)
    print("Testing Horn Detection Flow")
    print("="*60)
    
    # Load model
    model_path = Path("dual_board_training/photos_model_fixed/weights/best.pt")
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"\n1. Loading model from {model_path}...")
    model = YOLO(str(model_path))
    
    # Rename classes
    print("\n2. Renaming model classes...")
    renamed = rename_model_classes(model)
    if renamed:
        print(f"   Renamed classes: {renamed}")
    else:
        print("   No classes renamed")
    
    print(f"\n3. Model class names: {model.names}")
    
    # Check which classes might be horns
    horn_related = []
    for class_id, class_name in model.names.items():
        if class_name in ['Horn', 'Lamp', 'Photoresistor']:
            horn_related.append((class_id, class_name))
    
    print(f"\n4. Horn-related classes in model:")
    if horn_related:
        for class_id, class_name in horn_related:
            print(f"   Class {class_id}: {class_name}")
    else:
        print("   None found")
    
    # Initialize analyzer
    print("\n5. Initializing Circuit Graph Analyzer...")
    analyzer = CircuitGraphAnalyzer()
    
    print(f"\n6. Analyzer status:")
    print(f"   - LED detector: {'✓' if analyzer.led_detector else '✗'}")
    print(f"   - Horn detector: {'✓' if analyzer.horn_detector else '✗'}")
    print(f"   - Horn reclassifications dict: {'✓' if hasattr(analyzer, 'horn_reclassifications') else '✗'}")
    print(f"   - Horn orientations dict: {'✓' if hasattr(analyzer, 'horn_orientations') else '✗'}")
    
    # Check detection method
    print(f"\n7. Detection methods:")
    print(f"   - detect_and_reclassify_horns: {'✓' if hasattr(analyzer, 'detect_and_reclassify_horns') else '✗'}")
    print(f"   - detect_led_orientations_from_boxes: {'✓' if hasattr(analyzer, 'detect_led_orientations_from_boxes') else '✗'}")
    
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print("When you press 'g' in integrated_circuit_system.py:")
    print("1. Model detects components (Photoresistor→Horn already renamed)")
    print("2. analyze_circuit() is called with frame")
    print("3. detect_and_reclassify_horns() runs first")
    print("   - Checks Lamp, Photoresistor, Horn for red plus")
    print("   - If found, stores in horn_reclassifications")
    print("4. detect_led_orientations_from_boxes() runs")
    print("   - Detects LED orientations")
    print("   - Detects Horn orientations (if not already in horn_orientations)")
    print("5. add_detections() creates nodes with orientations")
    print("6. Graph summary should show Horn Orientation Details")
    print()
    print("💡 Possible issues:")
    print("   - Red plus template not matching (confidence < 0.5)")
    print("   - Horn not being detected by model at all")
    print("   - Frame not being passed to analyze_circuit()")
    print()

if __name__ == "__main__":
    test_horn_flow()

