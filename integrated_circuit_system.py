#!/usr/bin/env python3
"""
Integrated Circuit Building System - Simplified

Combines live detection and board visualization for circuit building activities.
"""

import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt

# Import our components
try:
    from dual_board_visualizer import DualBoardVisualizer, convert_detections_with_positions
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False

class IntegratedCircuitSystem:
    def __init__(self):
        self.model = None
        self.visualizer = None
        
        # System state
        self.show_board_viz = False
        self.processing_paused = False
        
        # Visualization
        self.fig = None
        self.ax1 = None
        self.ax2 = None 
        
        if VISUALIZER_AVAILABLE:
            self.visualizer = DualBoardVisualizer(cell_size=25)
    
    def load_model(self):
        """Load YOLO model"""
        model_path = Path("dual_board_training/photos_model_fixed/weights/best.pt")
        
        if not model_path.exists():
            alt_paths = [
                Path("dual_board_training/photos_model_fixed/weights/last.pt"),
                Path("dual_board_training/photos_model/weights/best.pt")
            ]
            
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    break
        
        if not model_path or not model_path.exists():
            print("❌ No trained model found!")
            return False
            
        try:
            self.model = YOLO(str(model_path))
            print(f"✅ Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

    def update_board_visualization(self, left_detections, right_detections):
        """Update board visualization"""
        if not self.show_board_viz or not VISUALIZER_AVAILABLE:
            return
            
        try:
            # Create figure only once
            if self.fig is None:
                self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
                plt.ion()  # Enable interactive mode
            
            # Clear and redraw
            self.ax1.clear()
            self.ax2.clear()
            
            self.visualizer._draw_single_board(self.ax1, left_detections, "LEFT BOARD")
            self.visualizer._draw_single_board(self.ax2, right_detections, "RIGHT BOARD")
            self.fig.suptitle("Dual Board Visualization", fontsize=16, fontweight='bold')
            
            # Update display
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            
        except Exception as e:
            print(f"⚠️ Visualization error: {e}")
    
    def run_system(self):
        """Run the integrated circuit system"""
        if not self.load_model():
            return
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        print("\n🎯 SIMPLIFIED DUAL BOARD DETECTION SYSTEM")
        print("=" * 45)
        print("\n💡 Instructions:")
        print("   • Live detection with dual board splitting")
        print("   • 🟢 Green tape visible = Processing active (board visualization updates)")
        print("   • 🔴 Green tape covered = Processing paused (no updates)")
        
        print("\n📋 Controls:")
        print("   • 'b': Toggle board visualization")
        print("   • 's': Save current frame") 
        print("   • 'q': Quit")
        print()
        
        split_ratio = 0.5
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                key = cv2.waitKey(1) & 0xFF
                
                # Handle controls
                if key == ord('b') and VISUALIZER_AVAILABLE:
                    self.show_board_viz = not self.show_board_viz
                    if self.show_board_viz:
                        plt.ion()
                        print("🎨 Board visualization enabled")
                    else:
                        if self.fig:
                            plt.close(self.fig)
                            self.fig = None
                            self.ax1 = None
                            self.ax2 = None
                        print("🎨 Board visualization disabled")
                
                elif key == ord('s'):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"circuit_frame_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Frame saved: {filename}")
                
                elif key == ord('q'):
                    break
                
                # Run detection continuously
                results = self.model(frame, conf=0.6, iou=0.5)
                
                if results and len(results) > 0:
                    result = results[0]
                    display_frame = result.plot()
                    
                    if result.boxes is not None:
                        height, width = frame.shape[:2]
                        split_x = int(width * split_ratio)
                        
                        # Check for green tape to control processing
                        green_tape_detected = False
                        left_boxes = []
                        right_boxes = []
                        
                        for i, box in enumerate(result.boxes):
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            center_x = (x1 + x2) / 2
                            class_id = int(box.cls[0])
                            class_name = self.model.names[class_id]
                            
                            # Check for green tape
                            if class_name == "Green tape":
                                green_tape_detected = True
                            
                            if center_x < split_x:
                                left_boxes.append((i, box))
                            else:
                                right_boxes.append((i, box))
                        
                        # Update processing status based on green tape
                        if green_tape_detected:
                            if self.processing_paused:
                                print("🟢 Green tape detected - RESUMING processing")
                                self.processing_paused = False
                        else:
                            if not self.processing_paused:
                                print("🔴 Green tape covered - PAUSING processing")
                                self.processing_paused = True
                        
                        # Only process if not paused (green tape visible)
                        if not self.processing_paused:
                            # Convert detections
                            left_detections, right_detections = convert_detections_with_positions(
                                left_boxes, right_boxes, self.model.names, width, height)
                            
                            # Update visualization
                            self.update_board_visualization(left_detections, right_detections)
                else:
                    display_frame = frame.copy()
                
                frame_count += 1
                
                # Add overlay information
                height, width = frame.shape[:2]
                split_x = int(width * split_ratio)
                cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                cv2.putText(display_frame, "LEFT", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, "RIGHT", (split_x + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add processing status
                y_offset = 60
                if self.processing_paused:
                    status_text = "PROCESSING PAUSED - Uncover green tape"
                    status_color = (0, 0, 255)  # Red
                else:
                    status_text = "PROCESSING ACTIVE - Green tape visible"
                    status_color = (0, 255, 0)  # Green
                
                cv2.putText(display_frame, status_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                
                cv2.imshow('Dual Board Detection System', display_frame)
        
        except KeyboardInterrupt:
            print("\n🛑 System interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.fig:
                plt.close(self.fig)
                plt.close('all')  # Close any remaining matplotlib windows
            print("✅ System shutdown complete")

def main():
    """Main function"""
    print("🎯 SIMPLIFIED DUAL BOARD DETECTION SYSTEM")
    print("=" * 40)
    print()
    print("Features:")
    print("• Live component detection with YOLO")
    print("• Dual board splitting (left/right)")
    print("• Real-time board visualization")
    print("• Green tape pause/resume control")
    print()
    
    system = IntegratedCircuitSystem()
    system.run_system()

if __name__ == "__main__":
    main()
