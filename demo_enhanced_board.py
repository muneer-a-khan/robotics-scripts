#!/usr/bin/env python3
"""
Demo Enhanced Board Visualization
Shows how to use the new enhanced visualization system with your existing detection
"""

import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
from enhanced_board_visualizer import EnhancedBoardVisualizer
from circuit_flow_analyzer import CircuitFlowAnalyzer, analyze_circuit_from_detection
import threading
from collections import defaultdict

def demo_with_sample_data():
    """Demo with sample circuit data"""
    print("🎨 ENHANCED BOARD VISUALIZATION DEMO")
    print("=" * 50)
    
    # Sample detection data (like what you get from your detection system)
    sample_left_detections = {
        'U_2 red alarm circuit': [
            {'x1': 2, 'y1': 4, 'x2': 3, 'y2': 5, 'confidence': 0.94, 'side': 'left'}
        ],
        'Slide switch': [
            {'x1': 4, 'y1': 6, 'x2': 4, 'y2': 7, 'confidence': 0.95, 'side': 'left'}
        ],
        'Wire': [
            {'x1': 5, 'y1': 4, 'x2': 8, 'y2': 4, 'confidence': 0.85, 'side': 'left'}
        ]
    }
    
    sample_right_detections = {
        'Battery Holder': [
            {'x1': 9, 'y1': 13, 'x2': 10, 'y2': 14, 'confidence': 0.92, 'side': 'right'}
        ],
        'Lamp': [
            {'x1': 8, 'y1': 6, 'x2': 9, 'y2': 7, 'confidence': 0.94, 'side': 'right'}
        ],
        'Wire': [
            {'x1': 8, 'y1': 4, 'x2': 8, 'y2': 12, 'confidence': 0.87, 'side': 'right'}
        ]
    }
    
    # Create visualizer
    visualizer = EnhancedBoardVisualizer()
    
    # Combine detections for visualization
    all_detections = {}
    all_detections.update(sample_left_detections)
    all_detections.update(sample_right_detections)
    
    print("📊 Creating enhanced board visualization...")
    
    # Create the enhanced board visualization
    fig = visualizer.create_enhanced_board(all_detections)
    
    # Add circuit flow analysis
    print("🔄 Analyzing circuit flow...")
    analyzer = analyze_circuit_from_detection(sample_left_detections, sample_right_detections)
    
    # Create circuit flow diagram
    flow_fig = analyzer.create_circuit_flow_diagram(all_detections)
    
    print("✅ Visualizations created!")
    print("📋 Features demonstrated:")
    print("   • Component placement on virtual board")
    print("   • Component statistics and counts")
    print("   • Connection analysis")
    print("   • Circuit completion suggestions")
    
    # Show the visualizations
    plt.show()

def integrate_with_live_detection():
    """Show how to integrate with your existing live detection system"""
    print("\n🔧 INTEGRATION WITH LIVE DETECTION")
    print("=" * 50)
    
    print("To integrate with your existing simple_live_detection.py:")
    print()
    print("1. Import the enhanced modules:")
    print("   from enhanced_board_visualizer import EnhancedBoardVisualizer")
    print("   from circuit_flow_analyzer import analyze_circuit_from_detection")
    print()
    print("2. Initialize the visualizer in your main function:")
    print("   visualizer = EnhancedBoardVisualizer()")
    print("   plt.ion()  # Enable interactive mode")
    print()
    print("3. After processing detections, update visualization:")
    print("   # Convert your left_classes and right_classes to detection format")
    print("   left_detections = convert_to_detection_format(left_classes, 'left')")
    print("   right_detections = convert_to_detection_format(right_classes, 'right')")
    print("   ")
    print("   # Update visualization")
    print("   all_detections = {**left_detections, **right_detections}")
    print("   fig = visualizer.create_enhanced_board(all_detections)")
    print("   plt.pause(0.01)  # Update display")
    print()
    print("4. Add keyboard controls for toggling visualization")
    
def convert_detection_format_example():
    """Show how to convert your detection format to enhanced visualization format"""
    print("\n🔄 DETECTION FORMAT CONVERSION")
    print("=" * 40)
    
    # Your current format (from simple_live_detection.py)
    left_classes = {
        'U_2 red alarm circuit': 1,
        'Slide switch': 1, 
        'Wire': 4
    }
    
    right_classes = {
        'Battery Holder': 1,
        'Lamp': 1,
        'Wire': 3
    }
    
    print("Your current detection format:")
    print(f"left_classes = {left_classes}")
    print(f"right_classes = {right_classes}")
    
    def convert_to_detection_format(classes_dict, side, boxes_info=None):
        """Convert from your current format to enhanced visualization format"""
        detections = {}
        
        for comp_type, count in classes_dict.items():
            detections[comp_type] = []
            
            for i in range(count):
                # Create sample position data (in real use, you'd use actual bounding boxes)
                if side == 'left':
                    x = np.random.randint(0, 6)  # Left side of board
                    y = np.random.randint(0, 14)
                else:
                    x = np.random.randint(7, 12)  # Right side of board  
                    y = np.random.randint(0, 14)
                
                detection = {
                    'x1': x, 'y1': y,
                    'x2': x + 1, 'y2': y + 1,
                    'confidence': 0.85 + np.random.random() * 0.1,  # Random confidence
                    'side': side
                }
                
                detections[comp_type].append(detection)
        
        return detections
    
    # Convert to new format
    left_detections = convert_to_detection_format(left_classes, 'left')
    right_detections = convert_to_detection_format(right_classes, 'right')
    
    print("\nConverted to enhanced visualization format:")
    print("left_detections structure:")
    for comp_type, detections in left_detections.items():
        print(f"  {comp_type}: {len(detections)} detections")
        if detections:
            print(f"    Example: {detections[0]}")
    
    return left_detections, right_detections

def create_integration_example():
    """Create a complete integration example"""
    print("\n💡 COMPLETE INTEGRATION EXAMPLE")
    print("=" * 45)
    
    integration_code = '''
# Add these imports to your simple_live_detection.py
from enhanced_board_visualizer import EnhancedBoardVisualizer
from circuit_flow_analyzer import analyze_circuit_from_detection
import matplotlib.pyplot as plt

def main():
    # ... your existing code ...
    
    # Initialize enhanced visualization
    visualizer = EnhancedBoardVisualizer()
    plt.ion()  # Enable interactive plotting
    show_enhanced = False
    
    # In your detection loop, after getting left_classes and right_classes:
    if show_enhanced and (left_classes or right_classes):
        # Convert to enhanced format
        left_detections = convert_classes_to_detections(left_classes, 'left', result.boxes)
        right_detections = convert_classes_to_detections(right_classes, 'right', result.boxes)
        
        # Create visualization
        all_detections = {**left_detections, **right_detections}
        fig = visualizer.create_enhanced_board(all_detections)
        plt.pause(0.01)
    
    # Add to your key handling:
    if key == ord('v'):
        show_enhanced = not show_enhanced
        print(f"Enhanced visualization: {'ON' if show_enhanced else 'OFF'}")

def convert_classes_to_detections(classes_dict, side, boxes):
    """Convert your detection results to enhanced visualization format"""
    detections = {}
    box_idx = 0
    
    for comp_type, count in classes_dict.items():
        detections[comp_type] = []
        
        for i in range(count):
            if box_idx < len(boxes):
                # Use actual bounding box data
                box = boxes[box_idx]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                
                # Convert to grid coordinates (simplified)
                grid_x = int((x1 + x2) / 2 / frame_width * 13)
                grid_y = int((y1 + y2) / 2 / frame_height * 15)
                
                detection = {
                    'x1': grid_x, 'y1': grid_y,
                    'x2': grid_x + 1, 'y2': grid_y + 1,
                    'confidence': confidence,
                    'side': side,
                    'bbox': [x1, y1, x2, y2]
                }
                
                detections[comp_type].append(detection)
                box_idx += 1
    
    return detections
'''
    
    print("Here's the complete integration code:")
    print(integration_code)

def main():
    """Main demo function"""
    print("🚀 ENHANCED CIRCUIT BOARD VISUALIZATION SYSTEM")
    print("=" * 55)
    print()
    print("This demo shows the new enhanced visualization capabilities")
    print("for your circuit detection system.")
    print()
    
    # Run demo with sample data
    demo_with_sample_data()
    
    # Show integration instructions  
    integrate_with_live_detection()
    
    # Show format conversion
    left_det, right_det = convert_detection_format_example()
    
    # Create integration example
    create_integration_example()
    
    print("\n🎯 QUICK START:")
    print("1. Run: python demo_enhanced_board.py")
    print("2. Check out enhanced_live_detection.py for full integration")
    print("3. Or modify your simple_live_detection.py with the examples above")
    print()
    print("🔧 NEW FEATURES:")
    print("• Virtual board grid showing exact component positions")
    print("• Component connection analysis") 
    print("• Circuit completion suggestions")
    print("• Real-time visualization updates")
    print("• Save high-quality board diagrams")
    print()
    print("✨ Your circuit detection just got a major upgrade!")

if __name__ == "__main__":
    main()
