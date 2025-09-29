#!/usr/bin/env python3
"""
Enhanced Live Detection with Advanced Board Visualization
Combines your existing detection system with enhanced visual board representation
"""

import cv2
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from enhanced_board_visualizer import EnhancedBoardVisualizer, process_detection_for_visualization
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from collections import defaultdict
import queue

class EnhancedLiveDetectionSystem:
    def __init__(self, model_path, confidence_threshold=0.6, split_ratio=0.5):
        """
        Initialize enhanced live detection system
        
        Args:
            model_path: Path to YOLO model
            confidence_threshold: Detection confidence threshold
            split_ratio: Split ratio for dual board setup
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.split_ratio = split_ratio
        
        # Load model
        print(f"🤖 Loading model: {model_path}")
        self.model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully")
        print(f"   Classes: {self.model.names}")
        print(f"   Number of classes: {len(self.model.names)}")
        
        # Initialize visualizer
        self.visualizer = EnhancedBoardVisualizer()
        
        # Initialize detection storage
        self.detection_history = []
        self.current_detections = {}
        self.detection_queue = queue.Queue(maxsize=10)
        
        # Matplotlib setup for real-time plotting
        self.fig = None
        self.setup_matplotlib()
        
    def setup_matplotlib(self):
        """Setup matplotlib for real-time visualization"""
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.suptitle('Enhanced Circuit Board Detection', fontsize=16, fontweight='bold')
        
    def process_frame(self, frame):
        """Process a single frame and return detection results"""
        # Split frame into left and right halves
        height, width = frame.shape[:2]
        split_x = int(width * self.split_ratio)
        
        # Process full frame with YOLO
        results = self.model(frame, conf=self.confidence_threshold, iou=0.5, verbose=False)
        
        if not results or len(results) == 0:
            return {}, {}, 0, 0
        
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return {}, {}, 0, 0
        
        # Get detections with positions
        boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()
        
        # Split detections into left and right
        left_detections = defaultdict(list)
        right_detections = defaultdict(list)
        left_count = 0
        right_count = 0
        
        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confidences)):
            x1, y1, x2, y2 = box
            class_name = self.model.names[int(cls)]
            
            # Determine which side the detection is on
            center_x = (x1 + x2) / 2
            
            detection_info = {
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'center': [(x1 + x2) / 2, (y1 + y2) / 2]
            }
            
            if center_x < split_x:
                # Left side
                left_detections[class_name].append(detection_info)
                left_count += 1
            else:
                # Right side  
                right_detections[class_name].append(detection_info)
                right_count += 1
        
        return left_detections, right_detections, left_count, right_count
    
    def update_visualization(self, left_detections, right_detections, left_count, right_count):
        """Update the enhanced visualization"""
        # Clear previous plots
        self.ax1.clear()
        self.ax2.clear()
        
        # Combine detections for visualization
        all_detections = {}
        
        # Add left side detections
        for comp_type, detections in left_detections.items():
            if comp_type not in all_detections:
                all_detections[comp_type] = []
            for detection in detections:
                # Convert to grid coordinates (simplified)
                grid_info = self.convert_to_grid_coords(detection, 'left')
                all_detections[comp_type].append(grid_info)
        
        # Add right side detections  
        for comp_type, detections in right_detections.items():
            if comp_type not in all_detections:
                all_detections[comp_type] = []
            for detection in detections:
                # Convert to grid coordinates (simplified)
                grid_info = self.convert_to_grid_coords(detection, 'right') 
                all_detections[comp_type].append(grid_info)
        
        # Draw component board
        self.visualizer._draw_component_board(self.ax1, all_detections)
        
        # Draw analysis
        self.visualizer._draw_connection_analysis(self.ax2, all_detections)
        
        # Add real-time info
        info_text = f"LEFT: {left_count} components | RIGHT: {right_count} components\n"
        info_text += f"Total: {left_count + right_count} components detected"
        self.fig.suptitle(f'Enhanced Circuit Board Detection - {info_text}', 
                         fontsize=14, fontweight='bold')
        
        plt.pause(0.01)  # Small pause to update display
        
    def convert_to_grid_coords(self, detection, side):
        """Convert camera coordinates to grid coordinates (simplified)"""
        # This is a simplified conversion - you can enhance based on your coordinate mapping
        x_center, y_center = detection['center']
        confidence = detection['confidence']
        
        if side == 'left':
            # Map to left side of grid (columns 0-7)
            grid_x = int((x_center / 320) * 6)  # Assuming 640px width, left half
            grid_y = int((y_center / 480) * 7)  # Assuming 480px height
        else:
            # Map to right side of grid (columns 8-14)
            grid_x = int(((x_center - 320) / 320) * 6) + 7
            grid_y = int((y_center / 480) * 7)
        
        # Ensure within bounds
        grid_x = max(0, min(grid_x, 12))
        grid_y = max(0, min(grid_y, 14))
        
        return {
            'x1': grid_x, 'y1': grid_y,
            'x2': grid_x + 1, 'y2': grid_y + 1,
            'confidence': confidence,
            'side': side
        }
    
    def create_detection_overlay(self, frame, left_detections, right_detections):
        """Create an overlay on the camera frame showing detections"""
        overlay = frame.copy()
        height, width = frame.shape[:2]
        split_x = int(width * self.split_ratio)
        
        # Draw split line
        cv2.line(overlay, (split_x, 0), (split_x, height), (255, 255, 255), 3)
        
        # Labels for sides
        cv2.putText(overlay, "LEFT BOARD", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, "RIGHT BOARD", (split_x + 10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Draw detections on left side
        for comp_type, detections in left_detections.items():
            color = self.get_component_color(comp_type)
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                conf = detection['confidence']
                
                # Draw bounding box
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label
                label = f"{comp_type}: {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (int(x1), int(y1-text_h-5)), 
                             (int(x1+text_w), int(y1)), color, -1)
                cv2.putText(overlay, label, (int(x1), int(y1-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw detections on right side
        for comp_type, detections in right_detections.items():
            color = self.get_component_color(comp_type)
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                conf = detection['confidence']
                
                # Draw bounding box
                cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Draw label
                label = f"{comp_type}: {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (int(x1), int(y1-text_h-5)), 
                             (int(x1+text_w), int(y1)), color, -1)
                cv2.putText(overlay, label, (int(x1), int(y1-5)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return overlay
    
    def get_component_color(self, comp_type):
        """Get BGR color for component type"""
        color_hex = self.visualizer.component_colors.get(comp_type, '#FFFFFF')
        # Convert hex to BGR
        return tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))
    
    def run_live_detection(self):
        """Main loop for live detection"""
        print("🎥 ENHANCED DUAL BOARD LIVE DETECTION")
        print("=" * 50)
        print("📋 Enhanced visualization with component analysis")
        print("🔄 Split view: LEFT board | RIGHT board")
        print("Press 'q' to quit, 's' to save current visualization")
        
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
        
        frame_count = 0
        last_detection_time = 0
        detection_interval = 2.0  # Process every 2 seconds
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Could not read frame")
                    break
                
                current_time = time.time()
                should_detect = (current_time - last_detection_time) >= detection_interval
                
                if should_detect:
                    print(f"\n🔄 Processing dual frame {frame_count}...")
                    last_detection_time = current_time
                    
                    # Process frame
                    left_det, right_det, left_count, right_count = self.process_frame(frame)
                    
                    if left_count > 0 or right_count > 0:
                        print(f"   🎯 LEFT Detections: {left_count}")
                        if left_det:
                            print(f"   📋 LEFT board classes:")
                            for comp_type, detections in left_det.items():
                                avg_conf = sum(d['confidence'] for d in detections) / len(detections)
                                print(f"     • {comp_type}: {len(detections)} (avg: {avg_conf:.2f})")
                        
                        print(f"   🎯 RIGHT Detections: {right_count}")
                        if right_det:
                            print(f"   📋 RIGHT board classes:")
                            for comp_type, detections in right_det.items():
                                avg_conf = sum(d['confidence'] for d in detections) / len(detections)
                                print(f"     • {comp_type}: {len(detections)} (avg: {avg_conf:.2f})")
                        
                        # Update enhanced visualization
                        self.update_visualization(left_det, right_det, left_count, right_count)
                        
                        # Store current detections
                        self.current_detections = {
                            'left': left_det,
                            'right': right_det,
                            'left_count': left_count,
                            'right_count': right_count,
                            'timestamp': current_time
                        }
                    
                    frame_count += 1
                
                # Create and show overlay
                if hasattr(self, 'current_detections') and self.current_detections:
                    overlay = self.create_detection_overlay(
                        frame, 
                        self.current_detections.get('left', {}),
                        self.current_detections.get('right', {})
                    )
                else:
                    overlay = frame.copy()
                    height, width = frame.shape[:2]
                    split_x = int(width * self.split_ratio)
                    cv2.line(overlay, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                    cv2.putText(overlay, "LEFT BOARD", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(overlay, "RIGHT BOARD", (split_x + 10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('Enhanced Dual Board Detection', overlay)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n✅ Stopping detection...")
                    break
                elif key == ord('s'):
                    if hasattr(self, 'fig'):
                        timestamp = int(time.time())
                        filename = f"enhanced_board_detection_{timestamp}.png"
                        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                        print(f"💾 Saved visualization as {filename}")
                
        except KeyboardInterrupt:
            print("\n✅ Interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            plt.close('all')
            print("✅ Cleanup complete")

def main():
    """Main function"""
    # Model path - update this to your model
    model_paths = [
        Path("dual_board_training/photos_model_fixed/weights/best.pt"),
        Path("dual_board_training/photos_model_fixed/weights/last.pt"),
        Path("dual_board_training/photos_model/weights/best.pt"),
        Path("dual_board_training/photos_model/weights/last.pt")
    ]
    
    model_path = None
    for path in model_paths:
        if path.exists():
            model_path = path
            break
    
    if model_path is None:
        print("❌ No model found! Please check model paths.")
        return
    
    print(f"🤖 Using model: {model_path}")
    
    # Initialize and run enhanced detection system
    detector = EnhancedLiveDetectionSystem(
        model_path=model_path,
        confidence_threshold=0.6,  # Higher confidence for cleaner results
        split_ratio=0.5
    )
    
    detector.run_live_detection()

if __name__ == "__main__":
    main()
