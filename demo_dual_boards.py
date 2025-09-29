#!/usr/bin/env python3
"""
Demo Dual Board Visualization
Shows what the side-by-side board visualization looks like
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
from dual_board_visualizer import DualBoardVisualizer, convert_simple_detections_to_grid
import time

def demo_matplotlib_version():
    """Demo the matplotlib version with sample data"""
    print("🎨 DUAL BOARD VISUALIZATION DEMO")
    print("=" * 40)
    
    # Sample detection data (like what you get from your live detection)
    sample_left_classes = {
        'U_2 red alarm circuit': 1,
        'Slide switch': 1,
        'Wire': 2,
        'Battery Holder': 1
    }
    
    sample_right_classes = {
        'Lamp': 1,
        'Resistor': 1,
        'Wire': 3,
        'LED_2 (Red)': 2,
        'Green tape': 1
    }
    
    print("Sample LEFT board detections:")
    for comp, count in sample_left_classes.items():
        print(f"  • {comp}: {count}")
    
    print("\nSample RIGHT board detections:")
    for comp, count in sample_right_classes.items():
        print(f"  • {comp}: {count}")
    
    # Create visualizer
    visualizer = DualBoardVisualizer(cell_size=30)
    
    # Convert to grid format
    left_detections, right_detections = convert_simple_detections_to_grid(
        sample_left_classes, sample_right_classes)
    
    # Create visualization
    print("\n🎨 Creating dual board visualization...")
    fig = visualizer.create_dual_board_visualization(left_detections, right_detections)
    
    print("✅ Visualization created!")
    print("📋 Features shown:")
    print("   • Two side-by-side circuit boards")
    print("   • Components fill grid squares where detected")
    print("   • Different colors for each component type")
    print("   • Component labels (W=Wire, BAT=Battery, etc.)")
    print("   • Legend showing component counts for each board")
    
    plt.show()

def demo_opencv_version():
    """Demo the OpenCV version for real-time display"""
    print("\n📺 OPENCV REAL-TIME VERSION DEMO")
    print("=" * 40)
    
    # Sample data
    sample_left_classes = {
        'U_2 red alarm circuit': 1,
        'Slide switch': 1,
        'Wire': 4,
        'Battery Holder': 2
    }
    
    sample_right_classes = {
        'Lamp': 2,
        'Resistor': 1,
        'Wire': 2,
        'LED_1 (Yellow)': 1,
        'Photoresistor': 1
    }
    
    visualizer = DualBoardVisualizer(cell_size=25)  # Smaller cells for OpenCV
    
    # Convert to grid format
    left_detections, right_detections = convert_simple_detections_to_grid(
        sample_left_classes, sample_right_classes)
    
    # Create OpenCV visualization
    print("🎨 Creating OpenCV board visualization...")
    board_img = visualizer.create_opencv_visualization(left_detections, right_detections)
    
    if board_img is not None:
        print("✅ OpenCV visualization created!")
        print("📋 This version can be shown in real-time alongside your camera feed")
        print("Press any key to close...")
        
        cv2.imshow('Dual Board Grid Visualization', board_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("❌ Failed to create OpenCV visualization")

def demo_animated_sequence():
    """Demo showing how boards update as components are added"""
    print("\n🎬 ANIMATED SEQUENCE DEMO")
    print("=" * 30)
    print("Shows how boards update as components are detected...")
    
    visualizer = DualBoardVisualizer(cell_size=30)
    
    # Sequence of detection states
    detection_sequence = [
        # Step 1: Just a battery
        ({'Battery Holder': 1}, {}),
        
        # Step 2: Add switch on left, lamp on right
        ({'Battery Holder': 1, 'Slide switch': 1}, {'Lamp': 1}),
        
        # Step 3: Add wires to connect
        ({'Battery Holder': 1, 'Slide switch': 1, 'Wire': 2}, {'Lamp': 1, 'Wire': 1}),
        
        # Step 4: Add more components
        ({'Battery Holder': 1, 'Slide switch': 1, 'Wire': 3, 'U_2 red alarm circuit': 1}, 
         {'Lamp': 1, 'Wire': 2, 'LED_2 (Red)': 1, 'Resistor': 1}),
        
        # Step 5: Final circuit
        ({'Battery Holder': 2, 'Slide switch': 1, 'Wire': 4, 'U_2 red alarm circuit': 1}, 
         {'Lamp': 1, 'Wire': 3, 'LED_2 (Red)': 2, 'Resistor': 1, 'Speaker': 1})
    ]
    
    for i, (left_classes, right_classes) in enumerate(detection_sequence):
        print(f"\n📸 Step {i+1}: Updating boards...")
        
        # Convert to grid format
        left_detections, right_detections = convert_simple_detections_to_grid(
            left_classes, right_classes)
        
        # Create visualization
        fig = visualizer.create_dual_board_visualization(left_detections, right_detections)
        fig.suptitle(f'Circuit Assembly - Step {i+1}', fontsize=16, fontweight='bold')
        
        plt.pause(2)  # Show for 2 seconds
        
        if i < len(detection_sequence) - 1:
            plt.close(fig)  # Close previous figure
    
    print("\n✅ Animation complete!")
    print("Final circuit shown - press close to continue")
    plt.show()

def integration_instructions():
    """Show how to integrate with existing detection system"""
    print("\n🔧 INTEGRATION INSTRUCTIONS")
    print("=" * 35)
    
    integration_code = '''
# Add to your existing simple_live_detection.py:

from dual_board_visualizer import DualBoardVisualizer, convert_simple_detections_to_grid

# Initialize at the start
visualizer = DualBoardVisualizer(cell_size=25)
show_boards = False

# After getting left_classes and right_classes from detection:
if show_boards:
    # Convert your detection counts to grid format
    left_detections, right_detections = convert_simple_detections_to_grid(
        left_classes, right_classes)
    
    # Create board visualization
    fig = visualizer.create_dual_board_visualization(left_detections, right_detections)
    plt.pause(0.01)  # Update display

# Add keyboard control:
if key == ord('b'):
    show_boards = not show_boards
    if show_boards:
        plt.ion()  # Enable interactive plotting
        print("🎨 Board visualization enabled")
'''
    
    print("Here's how to integrate with your existing detection:")
    print(integration_code)
    
    print("\n🚀 QUICK START GUIDE:")
    print("1. Run: python demo_dual_boards.py")
    print("2. Then try: python simple_dual_board_detection.py")
    print("3. In the live detection, press 'b' to toggle board view")
    print("4. Press 'o' to show OpenCV overlay alongside camera")

def main():
    """Main demo function"""
    print("🎨 DUAL BOARD VISUALIZATION SYSTEM")
    print("=" * 45)
    print()
    print("This demonstrates the simple side-by-side board visualization")
    print("that shows detected components as filled grid squares.")
    print()
    
    # Run demos
    demo_matplotlib_version()
    
    print("\nContinuing to OpenCV demo...")
    demo_opencv_version()
    
    print("\nContinuing to animated demo...")
    demo_animated_sequence()
    
    # Show integration instructions
    integration_instructions()
    
    print("\n🎯 This is exactly what you requested:")
    print("✅ Two boards side by side")
    print("✅ Components fill in grid squares where detected")
    print("✅ Simple, clean visualization")
    print("✅ Real-time updates as components are detected")
    print("✅ No complex graphs - just boards!")

if __name__ == "__main__":
    main()
