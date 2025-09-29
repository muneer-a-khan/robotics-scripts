#!/usr/bin/env python3
"""
Integrated Circuit Building System

Combines live detection, board visualization, and circuit verification
into one comprehensive system for circuit building activities.
"""

import cv2
import json
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

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
        self.ground_truths = {}
        
        # System state
        self.current_target_circuit = None
        self.completion_status = False
        self.show_board_viz = False
        self.show_verification_overlay = True
        
        # Timing
        self.last_verification_time = 0
        self.verification_interval = 5.0
        self.processing_paused = False
        
        # Visualization
        self.fig = None
        self.ax1 = None
        self.ax2 = None 
        self.ax3 = None
        self.ax4 = None
        self.last_suggestions = []
        self.last_comparison = None
        
        if VISUALIZER_AVAILABLE:
            self.visualizer = DualBoardVisualizer(cell_size=25)
        
        self.load_ground_truths()
    
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
    
    def load_ground_truths(self):
        """Load ground truth data"""
        gt_file = Path("ground_truths.json")
        if gt_file.exists():
            try:
                with open(gt_file, 'r') as f:
                    self.ground_truths = json.load(f)
                print(f"📥 Loaded {len(self.ground_truths)} ground truth circuits")
            except Exception as e:
                print(f"❌ Error loading ground truths: {e}")
                self.ground_truths = {}
        else:
            print("⚠️ No ground truths found - verification features disabled")
            print("   Run ground_truth_capture.py to create reference circuits")
    
    def select_target_circuit(self, circuit_id):
        """Select target circuit"""
        if str(circuit_id) not in self.ground_truths:
            print(f"❌ Circuit {circuit_id} not found in ground truths")
            return False
        
        self.current_target_circuit = circuit_id
        self.completion_status = False
        
        gt = self.ground_truths[str(circuit_id)]
        print(f"\n🎯 Building Circuit {circuit_id}: {gt['name']}")
        print(f"   Difficulty: {gt['difficulty'].upper()}")
        print(f"   Description: {gt['description']}")
        print(f"   Target Components: {gt['total_components']}")
        print("   🔄 Verification will begin in 5 seconds...")
        
        return True
    
    def compare_circuits(self, current_left, current_right):
        """Compare current circuit with target"""
        if not self.current_target_circuit:
            return None, []
        
        gt = self.ground_truths[str(self.current_target_circuit)]
        target_left = gt['left_detections']
        target_right = gt['right_detections']
        
        # Count components in current circuit
        current_components = defaultdict(int)
        for comp_type, detections in current_left.items():
            current_components[comp_type] += len(detections)
        for comp_type, detections in current_right.items():
            current_components[comp_type] += len(detections)
        
        # Count target components (ground truth captured on LEFT board only)
        # Allow participants to build on either board or both boards
        target_components = defaultdict(int)
        for comp_type, detections in target_left.items():
            target_components[comp_type] += len(detections)
        # target_right should be empty since we only capture on left, but check anyway
        for comp_type, detections in target_right.items():
            target_components[comp_type] += len(detections)
        
        # Calculate matches
        total_matches = 0
        total_target = sum(target_components.values())
        missing = {}
        extra = {}
        
        for comp_type, target_count in target_components.items():
            current_count = current_components.get(comp_type, 0)
            matches = min(current_count, target_count)
            total_matches += matches
            
            if current_count < target_count:
                missing[comp_type] = target_count - current_count
            elif current_count > target_count:
                extra[comp_type] = current_count - target_count
        
        for comp_type, current_count in current_components.items():
            if comp_type not in target_components and current_count > 0:
                extra[comp_type] = current_count
        
        match_percentage = (total_matches / total_target * 100) if total_target > 0 else 0
        
        # Generate suggestions
        suggestions = []
        if missing:
            suggestions.append("🔧 NEED TO ADD:")
            for comp_type, count in missing.items():
                suggestions.append(f"   • {count} {comp_type}{'s' if count > 1 else ''}")
        
        if extra:
            suggestions.append("⚠️ NEED TO REMOVE:")
            for comp_type, count in extra.items():
                suggestions.append(f"   • {count} {comp_type}{'s' if count > 1 else ''}")
        
        if not missing and not extra:
            suggestions.append("🎉 CIRCUIT COMPLETE!")
        
        is_complete = match_percentage >= 95 and len(missing) == 0
        
        return {
            'match_percentage': match_percentage,
            'total_matches': total_matches,
            'total_target': total_target,
            'is_complete': is_complete
        }, suggestions
    
    def update_board_visualization(self, left_detections, right_detections):
        """Update board visualization (update in place, don't recreate window)"""
        if not self.show_board_viz or not VISUALIZER_AVAILABLE:
            return
            
        try:
            # Create figure only once
            if self.fig is None:
                if self.current_target_circuit:
                    # Show comparison view
                    gt = self.ground_truths[str(self.current_target_circuit)]
                    self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                    plt.ion()  # Enable interactive mode
                else:
                    # Show regular dual board view
                    self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))
                    plt.ion()  # Enable interactive mode
            
            # Update existing figure
            if self.current_target_circuit:
                gt = self.ground_truths[str(self.current_target_circuit)]
                target_left = gt['left_detections']  
                target_for_display = target_left  # Use the captured left circuit as the target pattern
                
                # Clear and redraw all axes
                self.ax1.clear()
                self.ax2.clear() 
                self.ax3.clear()
                self.ax4.clear()
                
                self.visualizer._draw_single_board(self.ax1, left_detections, "CURRENT - LEFT")
                self.visualizer._draw_single_board(self.ax2, right_detections, "CURRENT - RIGHT")
                self.visualizer._draw_single_board(self.ax3, target_for_display, "TARGET PATTERN")
                self.visualizer._draw_single_board(self.ax4, target_for_display, "TARGET PATTERN")
                
                # Add note about flexible placement
                self.ax3.text(0.5, -0.1, "(Can build on either board)", transform=self.ax3.transAxes, 
                            ha='center', va='top', fontsize=10, style='italic')
                self.ax4.text(0.5, -0.1, "(Can build on either board)", transform=self.ax4.transAxes, 
                            ha='center', va='top', fontsize=10, style='italic')
                
                status = "COMPLETED ✅" if self.completion_status else "IN PROGRESS ⏳"
                self.fig.suptitle(f"Circuit {self.current_target_circuit}: {gt['name']} - {status}", 
                               fontsize=16, fontweight='bold')
            else:
                # Update regular dual board view
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
        
        print("\n🎯 INTEGRATED CIRCUIT BUILDING SYSTEM")
        print("=" * 45)
        
        if self.ground_truths:
            print("📋 Available Target Circuits:")
            easy_circuits = []
            hard_circuits = []
            for circuit_id, gt in self.ground_truths.items():
                if gt['difficulty'] == 'easy':
                    easy_circuits.append(f"   {circuit_id}: {gt['name']} ({gt['total_components']} components)")
                else:
                    hard_circuits.append(f"   {circuit_id}: {gt['name']} ({gt['total_components']} components)")
            
            if easy_circuits:
                print("   EASY:")
                for circuit in easy_circuits:
                    print(circuit)
            
            if hard_circuits:
                print("   HARD:")
                for circuit in hard_circuits:
                    print(circuit)
        
        print("\n💡 Building Instructions:")
        print("   • You can build circuits on either the LEFT or RIGHT board")
        print("   • Or build the same circuit on both boards if you prefer")
        print("   • The system will verify your circuit components regardless of placement")
        print("   • 🟢 Green tape visible = Processing active (board visualization updates)")
        print("   • 🔴 Green tape covered = Processing paused (no updates)")
        
        print("\n📋 Controls:")
        print("   • 1-6: Select target circuit to build")
        print("   • 'b': Toggle board visualization")
        print("   • 'v': Toggle verification overlay")
        print("   • 'n': Select new circuit")
        print("   • 'h': Show current suggestions")
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
                
                current_time = time.time()
                key = cv2.waitKey(1) & 0xFF
                
                # Handle controls
                if key >= ord('1') and key <= ord('6'):
                    circuit_id = key - ord('0')
                    self.select_target_circuit(circuit_id)
                
                elif key == ord('b') and VISUALIZER_AVAILABLE:
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
                            self.ax3 = None
                            self.ax4 = None
                        print("🎨 Board visualization disabled")
                
                elif key == ord('v'):
                    self.show_verification_overlay = not self.show_verification_overlay
                    status = "enabled" if self.show_verification_overlay else "disabled"
                    print(f"🔍 Verification overlay {status}")
                
                elif key == ord('n'):
                    print("🔄 Select new target circuit (1-6)")
                    self.current_target_circuit = None
                    self.completion_status = False
                
                elif key == ord('h'):
                    if self.last_suggestions:
                        print("\n💡 Current Suggestions:")
                        for suggestion in self.last_suggestions:
                            print(f"   {suggestion}")
                    else:
                        if self.current_target_circuit:
                            print("\n💡 No suggestions available yet")
                            print("   Circuit verification happens every 5 seconds")
                        else:
                            print("\n💡 Select a target circuit first (press 1-6)")
                            print("   Then suggestions will appear after verification runs")
                
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
                            
                            # Verification every 5 seconds
                            if (self.current_target_circuit and 
                                current_time - self.last_verification_time >= self.verification_interval):
                                
                                comparison, suggestions = self.compare_circuits(left_detections, right_detections)
                                
                                if comparison:
                                    self.last_comparison = comparison
                                    self.last_suggestions = suggestions
                                    
                                    print(f"\n🔍 Verification - {datetime.now().strftime('%H:%M:%S')}")
                                    print(f"   Progress: {comparison['match_percentage']:.1f}% ({comparison['total_matches']}/{comparison['total_target']} components)")
                                    
                                    if suggestions:
                                        for suggestion in suggestions:
                                            print(f"   {suggestion}")
                                    
                                    # Check completion
                                    if comparison['is_complete'] and not self.completion_status:
                                        self.completion_status = True
                                        print("\n🎉🎉🎉 CIRCUIT COMPLETED! 🎉🎉🎉")
                                        print(f"Excellent work! You successfully built Circuit {self.current_target_circuit}!")
                                    
                                    self.last_verification_time = current_time
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
                
                # Add verification overlay
                if self.show_verification_overlay:
                    y_offset = 60
                    
                    if self.current_target_circuit:
                        gt = self.ground_truths[str(self.current_target_circuit)]
                        
                        # Circuit info
                        status = "COMPLETED ✅" if self.completion_status else "IN PROGRESS"
                        circuit_text = f"Target: {gt['name']} - {status}"
                        cv2.putText(display_frame, circuit_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.completion_status else (0, 255, 255), 2)
                        y_offset += 20
                        
                        # Add note about flexible placement
                        placement_text = "(Build on either LEFT or RIGHT board)"
                        cv2.putText(display_frame, placement_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                        y_offset += 20
                        
                        # Add processing status
                        if self.processing_paused:
                            status_text = "PROCESSING PAUSED - Uncover green tape"
                            status_color = (0, 0, 255)  # Red
                        else:
                            status_text = "PROCESSING ACTIVE - Green tape visible"
                            status_color = (0, 255, 0)  # Green
                        
                        cv2.putText(display_frame, status_text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
                        y_offset += 25
                        
                        # Progress
                        if self.last_comparison:
                            progress_text = f"Progress: {self.last_comparison['match_percentage']:.0f}%"
                            cv2.putText(display_frame, progress_text, (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                            y_offset += 25
                        
                        # Next suggestion (first one only)
                        if self.last_suggestions and not self.completion_status:
                            suggestion = self.last_suggestions[0][:50]  # Truncate long suggestions
                            cv2.putText(display_frame, suggestion, (10, y_offset), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    else:
                        cv2.putText(display_frame, "Press 1-6 to select target circuit", (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow('Integrated Circuit Building System', display_frame)
        
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
    print("🎯 INTEGRATED CIRCUIT BUILDING SYSTEM")
    print("=" * 40)
    print()
    print("Complete system for guided circuit building with:")
    print("• Live component detection")
    print("• Real-time board visualization") 
    print("• Circuit verification against reference solutions")
    print("• Step-by-step guidance")
    print()
    
    system = IntegratedCircuitSystem()
    system.run_system()

if __name__ == "__main__":
    main()
