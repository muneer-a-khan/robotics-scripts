#!/usr/bin/env python3
"""
Simple Dual Board Detection with Visual Board Representation
Shows two side-by-side boards with components filling grid squares
"""

import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np

# Import the dual board visualizer
try:
    from dual_board_visualizer import DualBoardVisualizer, convert_simple_detections_to_grid, convert_detections_with_positions
    VISUALIZER_AVAILABLE = True
    print("✅ Dual board visualizer available")
except ImportError as e:
    VISUALIZER_AVAILABLE = False
    print("⚠️ Dual board visualizer not available:", str(e))

class SimpleDualBoardSystem:
    def __init__(self):
        self.show_board_viz = False
        self.visualizer = None
        self.fig = None
        
        if VISUALIZER_AVAILABLE:
            self.visualizer = DualBoardVisualizer(cell_size=25)  # Smaller cells for better display
            
    def toggle_board_visualization(self):
        """Toggle board visualization on/off"""
        if not VISUALIZER_AVAILABLE:
            print("⚠️ Board visualizer not available")
            return
            
        self.show_board_viz = not self.show_board_viz
        
        if self.show_board_viz:
            plt.ion()  # Enable interactive plotting
            print("🎨 Board visualization enabled")
        else:
            if self.fig:
                plt.close(self.fig)
                self.fig = None
            print("🎨 Board visualization disabled")
    
    def update_board_visualization(self, left_boxes, right_boxes, model_names, frame_width, frame_height):
        """Update the dual board visualization using actual detection positions"""
        if not self.show_board_viz or not VISUALIZER_AVAILABLE:
            return
            
        try:
            # Convert actual detection positions to grid format
            left_detections, right_detections = convert_detections_with_positions(
                left_boxes, right_boxes, model_names, frame_width, frame_height)
            
            # Close previous figure if exists
            if self.fig:
                plt.close(self.fig)
            
            # Create new visualization
            self.fig = self.visualizer.create_dual_board_visualization(left_detections, right_detections)
            plt.pause(0.01)  # Update display
            
        except Exception as e:
            print(f"⚠️ Board visualization error: {e}")
    
    def create_opencv_board_overlay(self, left_boxes, right_boxes, model_names, frame_width, frame_height):
        """Create OpenCV overlay for real-time display using actual positions"""
        if not VISUALIZER_AVAILABLE:
            return None
            
        try:
            # Convert actual detection positions to grid format
            left_detections, right_detections = convert_detections_with_positions(
                left_boxes, right_boxes, model_names, frame_width, frame_height)
            
            # Create OpenCV visualization
            board_img = self.visualizer.create_opencv_visualization(left_detections, right_detections)
            
            return board_img
            
        except Exception as e:
            print(f"⚠️ OpenCV board visualization error: {e}")
            return None

def main():
    """Simple dual board detection with visual board representation"""
    print("🎥 SIMPLE DUAL BOARD DETECTION WITH VISUAL BOARDS")
    print("=" * 55)
    print("📋 Shows two side-by-side boards with component placement")
    print("🔄 Components fill in grid squares where detected")
    
    # Initialize system
    dual_board_system = SimpleDualBoardSystem()
    
    # Use EXACT same model path as test_photos_model.py
    model_path = Path("dual_board_training/photos_model_fixed/weights/best.pt")
    
    if not model_path.exists():
        # Try alternative paths
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
    
    # Load model
    try:
        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully")
        print(f"   Classes: {len(model.names)} total")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    print("🎥 Camera opened successfully!")
    print("📋 Controls:")
    print("   • 'q': Quit")
    print("   • 's': Save current frame")
    if VISUALIZER_AVAILABLE:
        print("   • 'b': Toggle board visualization")
        print("   • 'o': Show OpenCV board overlay")
        print("   • 'p': Save board visualization")
    print("   • SPACE: Force detection on current frame")
    
    frame_count = 0
    last_detection_time = 0
    detection_interval = 2.0
    last_annotated_frame = None
    split_ratio = 0.5
    show_opencv_overlay = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            current_time = time.time()
            should_detect = (current_time - last_detection_time) >= detection_interval
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            force_detection = (key == ord(' '))
            
            if should_detect or force_detection:
                print(f"\n🔄 Processing dual frame {frame_count}...")
                
                # Process FULL frame
                results = model(frame, conf=0.6, iou=0.5)
                
                if results and len(results) > 0:
                    result = results[0]
                    
                    if result.boxes is not None:
                        height, width = frame.shape[:2]
                        split_x = int(width * split_ratio)
                        
                        # Split detections by position
                        left_boxes = []
                        right_boxes = []
                        
                        for i, box in enumerate(result.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = (x1 + x2) / 2
                            
                            if center_x < split_x:
                                left_boxes.append((i, box))
                            else:
                                right_boxes.append((i, box))
                        
                        # Process LEFT side
                        print(f"   🎯 LEFT Detections: {len(left_boxes)}")
                        left_classes = {}
                        for i, box in left_boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            if class_name not in left_classes:
                                left_classes[class_name] = []
                            left_classes[class_name].append(confidence)
                        
                        if left_classes:
                            print(f"   📋 LEFT board components:")
                            for class_name, confidences in left_classes.items():
                                avg_conf = sum(confidences) / len(confidences)
                                print(f"     • {class_name}: {len(confidences)} (avg: {avg_conf:.2f})")
                        
                        # Process RIGHT side
                        print(f"   🎯 RIGHT Detections: {len(right_boxes)}")
                        right_classes = {}
                        for i, box in right_boxes:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            confidence = float(box.conf[0])
                            
                            if class_name not in right_classes:
                                right_classes[class_name] = []
                            right_classes[class_name].append(confidence)
                        
                        if right_classes:
                            print(f"   📋 RIGHT board components:")
                            for class_name, confidences in right_classes.items():
                                avg_conf = sum(confidences) / len(confidences)
                                print(f"     • {class_name}: {len(confidences)} (avg: {avg_conf:.2f})")
                        
                        # Update board visualization with actual positions
                        dual_board_system.update_board_visualization(
                            left_boxes, right_boxes, model.names, width, height)
                        
                        # Create camera overlay
                        display_frame = result.plot()
                        cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                        cv2.putText(display_frame, "LEFT BOARD", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.putText(display_frame, "RIGHT BOARD", (split_x + 10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
                        last_annotated_frame = display_frame.copy()
                        
                        # Show OpenCV board overlay if requested
                        if show_opencv_overlay:
                            board_overlay = dual_board_system.create_opencv_board_overlay(
                                left_boxes, right_boxes, model.names, width, height)
                            if board_overlay is not None:
                                cv2.imshow('Board Grid Visualization', board_overlay)
                
                last_detection_time = current_time
                frame_count += 1
            
            # Show main camera feed
            if last_annotated_frame is not None:
                display_frame = last_annotated_frame
            else:
                height, width = frame.shape[:2]
                split_x = int(width * split_ratio)
                display_frame = frame.copy()
                cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                cv2.putText(display_frame, "LEFT BOARD", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, "RIGHT BOARD", (split_x + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Dual Board Detection', display_frame)
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_path = f"dual_board_detection_{int(time.time())}.jpg"
                cv2.imwrite(save_path, display_frame)
                print(f"💾 Saved camera view: {save_path}")
            elif key == ord('b') and VISUALIZER_AVAILABLE:
                dual_board_system.toggle_board_visualization()
            elif key == ord('o') and VISUALIZER_AVAILABLE:
                show_opencv_overlay = not show_opencv_overlay
                if show_opencv_overlay:
                    print("🎨 OpenCV board overlay enabled")
                else:
                    print("🎨 OpenCV board overlay disabled")
                    cv2.destroyWindow('Board Grid Visualization')
            elif key == ord('p') and VISUALIZER_AVAILABLE:
                if dual_board_system.fig is not None:
                    timestamp = int(time.time())
                    filename = f"dual_board_grid_{timestamp}.png"
                    dual_board_system.fig.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"💾 Saved board visualization: {filename}")
                else:
                    print("⚠️ Board visualization not active - press 'b' to enable")
    
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if dual_board_system.fig:
            plt.close(dual_board_system.fig)
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
