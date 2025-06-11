"""
Test script to process the real Snap Circuit image.
"""

import cv2
import numpy as np
from pathlib import Path
import json

from main import SnapCircuitVisionSystem


def test_real_snap_circuit_image():
    """Test our system on the real Snap Circuit image."""
    
    # Initialize the vision system
    print("Initializing Snap Circuit Vision System...")
    system = SnapCircuitVisionSystem(
        save_outputs=True,
        display_results=True
    )
    
    # For this test, we'll create a test image path
    # In practice, you would save the uploaded image to a file first
    test_image_path = "test_snap_circuit.jpg"
    
    # Check if image exists
    if not Path(test_image_path).exists():
        print(f"Please save your Snap Circuit image as '{test_image_path}' and run again.")
        print("You can do this by right-clicking the image and saving it.")
        return
    
    try:
        # Process the image
        print(f"Processing image: {test_image_path}")
        result = system.process_image(test_image_path)
        
        # Print detailed analysis
        print("\n" + "="*50)
        print("SNAP CIRCUIT ANALYSIS RESULTS")
        print("="*50)
        
        components = result.connection_graph.components
        connections = result.connection_graph.edges
        state = result.connection_graph.state
        
        print(f"\n📊 DETECTION SUMMARY:")
        print(f"   • Components detected: {len(components)}")
        print(f"   • Connections found: {len(connections)}")
        print(f"   • Processing time: {result.processing_time:.3f}s")
        
        print(f"\n🔧 DETECTED COMPONENTS:")
        for i, comp in enumerate(components, 1):
            bbox = comp.bbox
            print(f"   {i}. {comp.label.upper()}")
            print(f"      - ID: {comp.id}")
            print(f"      - Confidence: {comp.confidence:.2f}")
            print(f"      - Position: ({bbox.center[0]:.0f}, {bbox.center[1]:.0f})")
            print(f"      - Size: {bbox.width:.0f}x{bbox.height:.0f}")
            print(f"      - Orientation: {comp.orientation:.0f}°")
        
        print(f"\n🔗 DETECTED CONNECTIONS:")
        if connections:
            for i, conn in enumerate(connections, 1):
                print(f"   {i}. {conn.component_id_1} ↔ {conn.component_id_2}")
                print(f"      - Type: {conn.connection_type}")
                print(f"      - Confidence: {conn.confidence:.2f}")
        else:
            print("   No connections detected")
        
        print(f"\n⚡ CIRCUIT STATE ANALYSIS:")
        print(f"   • Circuit closed: {'✅ YES' if state.is_circuit_closed else '❌ NO'}")
        print(f"   • Power on: {'✅ YES' if state.power_on else '❌ NO'}")
        print(f"   • Active components: {len(state.active_components)}")
        
        if state.active_components:
            print(f"   • Active: {', '.join(state.active_components)}")
        
        if state.power_flow_path:
            print(f"   • Power flow: {' → '.join(state.power_flow_path)}")
        
        if state.estimated_voltage:
            print(f"   • Estimated voltage: {state.estimated_voltage:.1f}V")
        
        if state.estimated_current:
            print(f"   • Estimated current: {state.estimated_current*1000:.0f}mA")
        
        # Save detailed results
        results_file = "snap_circuit_analysis.json"
        with open(results_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\n💾 Detailed results saved to: {results_file}")
        
        # Recommendations based on what we can see in the image
        print(f"\n🤖 ANALYSIS NOTES:")
        print("   Based on the image, I can see:")
        print("   • Battery holder at top (power source)")
        print("   • Green multi-connection component (left)")  
        print("   • Blue component with connections (right)")
        print("   • Small red component (center) - likely LED")
        print("   • Clear hexagonal connection board")
        
        print(f"\n🎯 RECOMMENDATIONS:")
        if len(components) == 0:
            print("   • No components detected - model needs training!")
            print("   • This image would be perfect for training data")
            print("   • Consider adjusting detection thresholds")
        elif len(components) < 4:
            print("   • Some components may not be detected")
            print("   • Model could benefit from training on this type of setup")
        else:
            print("   • Good detection performance!")
            print("   • System working well on real Snap Circuits")
        
        return result
        
    except Exception as e:
        print(f"Error processing image: {e}")
        print("This is expected if the model hasn't been trained yet.")
        print("Let's use this image for training data!")
        return None


def create_training_annotation():
    """Create a manual annotation for this image to use as training data."""
    
    print("\n" + "="*50)
    print("CREATING TRAINING ANNOTATION")
    print("="*50)
    
    # Based on visual analysis of the image, create manual annotations
    # These are approximate bounding boxes for the components we can see
    
    annotations = []
    
    # Battery holder (top center)
    annotations.append({
        "class": "battery_holder",
        "bbox": [0.35, 0.15, 0.65, 0.35],  # normalized [x_center, y_center, width, height]
        "confidence": 1.0
    })
    
    # Green component (left side)
    annotations.append({
        "class": "switch",  # or could be a control module
        "bbox": [0.25, 0.55, 0.15, 0.25],
        "confidence": 1.0
    })
    
    # Blue component (right side) 
    annotations.append({
        "class": "switch",
        "bbox": [0.75, 0.55, 0.15, 0.25],
        "confidence": 1.0
    })
    
    # Red component (center)
    annotations.append({
        "class": "led",
        "bbox": [0.50, 0.65, 0.08, 0.08],
        "confidence": 1.0
    })
    
    # Convert to YOLO format
    yolo_annotations = []
    class_mapping = {
        "wire": 0,
        "switch": 1,
        "button": 2,
        "battery_holder": 3,
        "led": 4,
        "speaker": 5,
        "music_circuit": 6,
        "motor": 7,
        "resistor": 8,
        "connection_node": 9,
        "lamp": 10,
        "fan": 11,
        "buzzer": 12,
        "photoresistor": 13,
        "microphone": 14,
        "alarm": 15
    }
    
    for ann in annotations:
        class_id = class_mapping.get(ann["class"], 0)
        bbox = ann["bbox"]
        yolo_line = f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"
        yolo_annotations.append(yolo_line)
    
    # Save annotation file
    annotation_file = "test_snap_circuit.txt"
    with open(annotation_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))
    
    print(f"Created training annotation: {annotation_file}")
    print("Annotation contains:")
    for ann in annotations:
        print(f"   • {ann['class']}: {ann['bbox']}")
    
    print(f"\n📚 TO USE FOR TRAINING:")
    print(f"   1. Save your image as 'test_snap_circuit.jpg'")
    print(f"   2. Use the annotation file: {annotation_file}")
    print(f"   3. Run: python training/data_preparation.py --create-samples")
    print(f"   4. Add your image and annotation to the training dataset")
    print(f"   5. Train with: python -c \"from models.component_detector import ComponentDetector; d=ComponentDetector(); d.train('data/training/data.yaml')\"")


if __name__ == "__main__":
    print("🔧 SNAP CIRCUIT COMPUTER VISION TEST")
    print("Testing on real Snap Circuit image...")
    
    # Test the current system
    result = test_real_snap_circuit_image()
    
    # Create training annotation regardless
    create_training_annotation()
    
    print(f"\n✅ Test complete!")
    if result is None:
        print("Ready to train the model with this excellent example!")
    else:
        print("System analysis complete. Check the annotated output image!") 