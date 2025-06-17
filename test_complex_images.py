#!/usr/bin/env python3
"""
Test the precision-focused model on complex circuit images with multiple components.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from models.component_detector import ComponentDetector


def test_complex_circuit():
    """Test on a complex circuit image with multiple components."""
    model_path = "models/weights/precision_focused.pt"
    conf_threshold = 0.2
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Initialize detector
    detector = ComponentDetector(model_path, conf_threshold=conf_threshold)
    
    # Find a circuit image with multiple components
    test_image_paths = [
        "data/augmented_training/images/val",
        "data/augmented_training/images/train",
        "data/augmented_training/images/test"
    ]
    
    # Look for images that might have multiple components
    potential_circuit_images = []
    for test_path in test_image_paths:
        image_dir = Path(test_path)
        if image_dir.exists():
            # Look for images that might be circuit boards (not individual components)
            for img_file in image_dir.glob("*.jpg"):
                if any(keyword in img_file.name.lower() for keyword in ['circuit', 'snap', 'board']):
                    potential_circuit_images.append(img_file)
            
            # If no circuit images found, just use the first few images
            if not potential_circuit_images:
                potential_circuit_images.extend(list(image_dir.glob("*.jpg"))[:3])
    
    if not potential_circuit_images:
        print("❌ No test images found")
        return
    
    print(f"🔍 Testing on {len(potential_circuit_images)} images...")
    
    os.makedirs("output/complex_test", exist_ok=True)
    
    for i, image_path in enumerate(potential_circuit_images[:3]):  # Test on first 3
        print(f"\n📊 Testing: {image_path.name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"   ❌ Could not load {image_path}")
            continue
        
        img_height, img_width = image.shape[:2]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get predictions
        detections = detector.detect(image)
        
        print(f"   🎯 Found {len(detections)} components")
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))
        ax.imshow(image_rgb)
        ax.set_title(f"Precision Model Predictions\n"
                    f"Image: {image_path.name} | Found: {len(detections)} components | "
                    f"Conf ≥ {conf_threshold}", fontsize=16, fontweight='bold')
        
        # Colors for different component types
        component_colors = {
            'wire': 'red',
            'switch': 'blue', 
            'button': 'green',
            'battery_holder': 'orange',
            'led': 'purple',
            'speaker': 'brown',
            'music_circuit': 'pink',
            'motor': 'gray',
            'resistor': 'cyan',
            'connection_node': 'yellow',
            'lamp': 'lime',
            'fan': 'navy',
            'buzzer': 'maroon',
            'photoresistor': 'olive',
            'microphone': 'coral',
            'alarm': 'gold'
        }
        
        # Group detections by type for summary
        component_counts = {}
        
        # Draw predicted boxes
        for j, detection in enumerate(detections):
            bbox = detection.bbox
            label = detection.label
            confidence = detection.confidence
            
            # Count components
            component_counts[label] = component_counts.get(label, 0) + 1
            
            # Get color for this component type
            color = component_colors.get(label, 'red')
            
            # Draw bounding box
            rect = patches.Rectangle(
                (bbox.x1, bbox.y1), 
                bbox.x2 - bbox.x1, 
                bbox.y2 - bbox.y1,
                linewidth=3, 
                edgecolor=color, 
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label with confidence
            ax.text(
                bbox.x1, bbox.y1 - 5,
                f"{label} ({confidence:.2f})",
                fontsize=11, color=color, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
        
        # Add component summary
        summary_text = "🔧 Detected Components:\n"
        for comp_type, count in sorted(component_counts.items()):
            summary_text += f"• {comp_type}: {count}\n"
        
        # Add confidence and detection info
        if detections:
            avg_conf = np.mean([d.confidence for d in detections])
            min_conf = min([d.confidence for d in detections])
            max_conf = max([d.confidence for d in detections])
            
            summary_text += f"\n📊 Confidence Stats:\n"
            summary_text += f"• Average: {avg_conf:.3f}\n"
            summary_text += f"• Range: {min_conf:.3f} - {max_conf:.3f}"
        
        # Add text box with summary
        ax.text(0.02, 0.98, summary_text, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor='lightblue', alpha=0.9))
        
        ax.axis('off')
        
        # Save visualization
        output_file = f"output/complex_test/{image_path.stem}_complex_prediction.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📁 Saved: {output_file}")
        
        # Print detailed results
        print(f"   📋 Component Summary:")
        for comp_type, count in sorted(component_counts.items()):
            print(f"      {comp_type}: {count}")
        
        if detections:
            print(f"   📊 Avg Confidence: {avg_conf:.3f}")
    
    print(f"\n✅ Complex testing complete!")
    print(f"📁 Results saved in: output/complex_test/")


if __name__ == "__main__":
    test_complex_circuit() 