#!/usr/bin/env python3
"""
Simple Live Detection - Enhanced Version
Adds optional enhanced board visualization to your working detection system
Press 'v' to toggle enhanced visualization on/off
"""

import cv2
import time
from pathlib import Path
from ultralytics import YOLO
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Import enhanced visualization (optional - system works without it)
try:
    from enhanced_board_visualizer import EnhancedBoardVisualizer
    from circuit_flow_analyzer import CircuitFlowAnalyzer, analyze_circuit_from_detection
    ENHANCED_AVAILABLE = True
    print("✅ Enhanced visualization available")
except ImportError as e:
    ENHANCED_AVAILABLE = False
    print("⚠️ Enhanced visualization not available:", str(e))
    print("   Basic detection will still work perfectly!")

class EnhancedDetectionSystem:
    def __init__(self):
        self.enhanced_viz_enabled = False
        self.visualizer = None
        self.fig = None
        
        if ENHANCED_AVAILABLE:
            self.visualizer = EnhancedBoardVisualizer()
            
    def toggle_enhanced_visualization(self):
        """Toggle enhanced visualization on/off"""
        if not ENHANCED_AVAILABLE:
            print("⚠️ Enhanced visualization not available")
            return
            
        self.enhanced_viz_enabled = not self.enhanced_viz_enabled
        
        if self.enhanced_viz_enabled:
            self.setup_matplotlib()
            print("🎨 Enhanced visualization enabled")
        else:
            if self.fig:
                plt.close(self.fig)
                self.fig = None
            print("🎨 Enhanced visualization disabled")
    
    def setup_matplotlib(self):
        """Setup matplotlib for enhanced visualization"""
        if self.fig is None:
            plt.ion()
            self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(16, 10))
            self.fig.suptitle('Enhanced Circuit Board Analysis', fontsize=16, fontweight='bold')
    
    def convert_classes_to_detections(self, classes_dict, side, boxes, frame_width, frame_height):
        """Convert detection classes to enhanced visualization format"""
        detections = {}
        
        for comp_type, confidences in classes_dict.items():
            detections[comp_type] = []
            
            for conf in confidences:
                # Create simplified position data
                # In real implementation, you'd map actual bounding boxes to grid positions
                if side == 'left':
                    grid_x = np.random.randint(0, 6)  # Left side
                    grid_y = np.random.randint(0, 14)
                else:
                    grid_x = np.random.randint(7, 12)  # Right side
                    grid_y = np.random.randint(0, 14)
                
                detection = {
                    'x1': grid_x, 'y1': grid_y,
                    'x2': grid_x + 1, 'y2': grid_y + 1,
                    'confidence': conf,
                    'side': side
                }
                
                detections[comp_type].append(detection)
        
        return detections
    
    def update_enhanced_visualization(self, left_classes, right_classes, frame_width, frame_height):
        """Update enhanced visualization with detection results"""
        if not self.enhanced_viz_enabled or not ENHANCED_AVAILABLE or self.fig is None:
            return
        
        try:
            # Convert to enhanced format
            left_detections = self.convert_classes_to_detections(left_classes, 'left', None, frame_width, frame_height)
            right_detections = self.convert_classes_to_detections(right_classes, 'right', None, frame_width, frame_height)
            
            # Clear previous plots
            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
                ax.clear()
            
            # Combine detections
            all_detections = {}
            all_detections.update(left_detections)
            all_detections.update(right_detections)
            
            # Panel 1: Component Board Layout
            self.visualizer._draw_component_board(self.ax1, all_detections)
            self.ax1.set_title("Circuit Board Layout", fontweight='bold')
            
            # Panel 2: Component Statistics
            self.visualizer._draw_connection_analysis(self.ax2, all_detections)
            self.ax2.set_title("Component Statistics", fontweight='bold')
            
            # Panel 3: Circuit Analysis
            if left_detections or right_detections:
                analyzer = analyze_circuit_from_detection(left_detections, right_detections)
                analyzer._draw_connection_diagram(self.ax3, all_detections)
            self.ax3.set_title("Connection Analysis", fontweight='bold')
            
            # Panel 4: Suggestions
            if left_detections or right_detections:
                analyzer._draw_flow_analysis(self.ax4, all_detections)
            self.ax4.set_title("Circuit Completion Suggestions", fontweight='bold')
            
            plt.pause(0.01)  # Update display
            
        except Exception as e:
            print(f"⚠️ Enhanced visualization error: {e}")
    
    def save_visualization(self):
        """Save current enhanced visualization"""
        if not self.enhanced_viz_enabled or self.fig is None:
            print("⚠️ Enhanced visualization not active")
            return
            
        try:
            timestamp = int(time.time())
            filename = f"enhanced_board_analysis_{timestamp}.png"
            self.fig.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"💾 Saved enhanced visualization as {filename}")
        except Exception as e:
            print(f"❌ Error saving visualization: {e}")

def main():
    """Enhanced live detection with optional advanced visualization"""
    print("🎥 ENHANCED DUAL BOARD LIVE DETECTION")
    print("=" * 50)
    print("📋 Using SAME approach as test_photos_model.py")
    print("🔄 Split view: LEFT board | RIGHT board")
    if ENHANCED_AVAILABLE:
        print("🎨 Enhanced visualization available - press 'v' to toggle")
    
    # Initialize enhanced detection system
    enhanced_system = EnhancedDetectionSystem()
    
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
    
    print("🎥 Camera opened successfully!")
    print("📋 Controls:")
    print("   • 'q': Quit")
    print("   • 's': Save current frame")
    if ENHANCED_AVAILABLE:
        print("   • 'v': Toggle enhanced visualization")
        print("   • 'e': Save enhanced visualization")
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
            height, width = frame.shape[:2]
            
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
                        left_classes = {}
                        if len(left_boxes) > 0:
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
                        right_classes = {}
                        if len(right_boxes) > 0:
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
                        
                        # Update enhanced visualization if enabled
                        enhanced_system.update_enhanced_visualization(left_classes, right_classes, width, height)
                        
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
                split_x = int(width * split_ratio)
                display_frame = frame.copy()
                cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                cv2.putText(display_frame, "LEFT BOARD", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, "RIGHT BOARD", (split_x + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Show frame with dual board split
            cv2.imshow('Enhanced Dual Board Live Detection', display_frame)
            
            # Handle keys
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_path = f"live_detection_save_{int(time.time())}.jpg"
                cv2.imwrite(save_path, display_frame)
                print(f"💾 Saved: {save_path}")
            elif key == ord('v') and ENHANCED_AVAILABLE:
                enhanced_system.toggle_enhanced_visualization()
            elif key == ord('e') and ENHANCED_AVAILABLE:
                enhanced_system.save_visualization()
    
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if enhanced_system.fig:
            plt.close(enhanced_system.fig)
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
