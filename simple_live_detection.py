#!/usr/bin/env python3
"""
Simple Live Detection - Enhanced with board visualization
"""

import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import threading
import numpy as np
from enhanced_board_visualizer import EnhancedBoardVisualizer
from circuit_flow_analyzer import CircuitFlowAnalyzer, analyze_circuit_from_detection

class EnhancedSimpleLiveDetection:
    def __init__(self):
        self.visualizer = EnhancedBoardVisualizer()
        self.circuit_analyzer = CircuitFlowAnalyzer()
        self.current_detections = {'left': {}, 'right': {}}
        self.show_enhanced_viz = False
        
        # Setup matplotlib for enhanced visualization
        plt.ion()
        self.fig = None
        
    def toggle_enhanced_visualization(self):
        """Toggle enhanced visualization on/off"""
        self.show_enhanced_viz = not self.show_enhanced_viz
        if self.show_enhanced_viz:
            self.setup_matplotlib()
            print("🎨 Enhanced visualization enabled")
        else:
            if self.fig:
                plt.close(self.fig)
            print("🎨 Enhanced visualization disabled")
    
    def setup_matplotlib(self):
        """Setup matplotlib for enhanced visualization"""
        if self.fig is None:
            self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            self.fig.suptitle('Enhanced Circuit Board Analysis', fontsize=16, fontweight='bold')
    
    def update_enhanced_visualization(self, left_detections, right_detections):
        """Update enhanced visualization with current detections"""
        if not self.show_enhanced_viz or self.fig is None:
            return
            
        # Clear previous plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        # Combine detections
        all_detections = {}
        all_detections.update(left_detections)
        all_detections.update(right_detections)
        
        # Panel 1: Component Board
        self.visualizer._draw_component_board(self.ax1, all_detections)
        self.ax1.set_title("Circuit Board Layout", fontweight='bold')
        
        # Panel 2: Component Analysis
        self.visualizer._draw_connection_analysis(self.ax2, all_detections)
        self.ax2.set_title("Component Statistics", fontweight='bold')
        
        # Panel 3: Circuit Analysis
        analyzer = analyze_circuit_from_detection(left_detections, right_detections)
        analyzer._draw_connection_diagram(self.ax3, all_detections)
        self.ax3.set_title("Connection Analysis", fontweight='bold')
        
        # Panel 4: Suggestions
        analyzer._draw_flow_analysis(self.ax4, all_detections)
        self.ax4.set_title("Circuit Completion Suggestions", fontweight='bold')
        
        plt.pause(0.01)

def main():
    """Enhanced live detection with visualization options"""
    print("🎥 ENHANCED DUAL BOARD LIVE DETECTION")
    print("=" * 50)
    print("📋 Using SAME approach as test_photos_model.py")
    print("🔄 Split view: LEFT board | RIGHT board")
    print("🎨 Press 'v' to toggle enhanced visualization")
    print("💾 Press 's' to save current visualization")
    
    # Initialize enhanced detection system
    enhanced_detector = EnhancedSimpleLiveDetection()
    
    # Use EXACT same model path as test_photos_model.py
    model_path = Path("dual_board_training/photos_model_fixed/weights/best.pt")
    
    if not model_path.exists():
        # Try alternative paths (same as test_photos_model.py)
        alt_paths = [
            Path("dual_board_training/photos_model_fixed/weights/last.pt"),
            Path("dual_board_training/photos_model/weights/best.pt"),
            Path("dual_board_training/photos_model/weights/last.pt")
        ]
        
        model_path = None
        for alt_path in alt_paths:
            if alt_path.exists():
                model_path = alt_path
                break
        
        if not model_path:
            print("❌ No trained model found!")
            return
    
    print(f"🤖 Model: {model_path}")
    
    # Load model EXACTLY like test_photos_model.py
    try:
        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully")
        print(f"   Classes: {model.names}")
        print(f"   Number of classes: {len(model.names)}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("🎥 Camera opened successfully!")
    print("📋 Controls:")
    print("   • 'q': Quit")
    print("   • 's': Save current frame")
    print("   • SPACE: Force detection on current frame")
    
    frame_count = 0
    last_detection_time = 0
    detection_interval = 2.0  # Process every 2 seconds
    last_annotated_frame = None
    split_ratio = 0.5  # Split frame in half
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            current_time = time.time()
            should_detect = (current_time - last_detection_time) >= detection_interval
            
            display_frame = frame.copy()
            
            # Run detection at intervals (or when forced)
            key = cv2.waitKey(1) & 0xFF
            force_detection = (key == ord(' '))
            
            if should_detect or force_detection:
                print(f"\n🔄 Processing dual frame {frame_count}...")
                
                # Process FULL frame (same as before) - don't change detection quality!
                results = model(frame, conf=0.6, iou=0.5)
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if result.boxes is not None:
                        # Get frame dimensions for splitting
                        height, width = frame.shape[:2]
                        split_x = int(width * split_ratio)
                        
                        # Split detections by position (left vs right of split line)
                        left_boxes = []
                        right_boxes = []
                        
                        for i, box in enumerate(result.boxes):
                            # Get box center x coordinate
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = (x1 + x2) / 2
                            
                            if center_x < split_x:
                                left_boxes.append((i, box))
                            else:
                                right_boxes.append((i, box))
                        
                        # Count and show LEFT side results
                        print(f"   🎯 LEFT Detections: {len(left_boxes)}")
                        if len(left_boxes) > 0:
                            left_classes = {}
                            for i, box in left_boxes:
                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]
                                confidence = float(box.conf[0])
                                
                                if class_name not in left_classes:
                                    left_classes[class_name] = []
                                left_classes[class_name].append(confidence)
                            
                            print(f"   📋 LEFT board classes:")
                            for class_name, confidences in left_classes.items():
                                avg_conf = sum(confidences) / len(confidences)
                                print(f"     • {class_name}: {len(confidences)} (avg: {avg_conf:.2f})")
                        
                        # Count and show RIGHT side results
                        print(f"   🎯 RIGHT Detections: {len(right_boxes)}")
                        if len(right_boxes) > 0:
                            right_classes = {}
                            for i, box in right_boxes:
                                class_id = int(box.cls[0])
                                class_name = model.names[class_id]
                                confidence = float(box.conf[0])
                                
                                if class_name not in right_classes:
                                    right_classes[class_name] = []
                                right_classes[class_name].append(confidence)
                            
                            print(f"   📋 RIGHT board classes:")
                            for class_name, confidences in right_classes.items():
                                avg_conf = sum(confidences) / len(confidences)
                                print(f"     • {class_name}: {len(confidences)} (avg: {avg_conf:.2f})")
                        
                        # Use the original full-frame annotated result
                        display_frame = result.plot()
                        
                        # Add split line and labels
                        cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                        cv2.putText(display_frame, "LEFT BOARD", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(display_frame, "RIGHT BOARD", (split_x + 10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        last_annotated_frame = display_frame.copy()
                
                last_detection_time = current_time
                frame_count += 1
            
            # Use last annotated frame if available
            elif last_annotated_frame is not None:
                display_frame = last_annotated_frame
            else:
                # Show split line on raw frame when no detections yet
                height, width = frame.shape[:2]
                split_x = int(width * split_ratio)
                display_frame = frame.copy()
                cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                cv2.putText(display_frame, "LEFT BOARD", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, "RIGHT BOARD", (split_x + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame with dual board split
            cv2.imshow('Dual Board Live Detection (Split View)', display_frame)
            
            # Handle keys
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame EXACTLY like test_photos_model.py
                save_path = f"live_detection_save_{int(time.time())}.jpg"
                cv2.imwrite(save_path, display_frame)
                print(f"💾 Saved: {save_path}")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
