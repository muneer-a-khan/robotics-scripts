#!/usr/bin/env python3
"""
Ground Truth Circuit Capture System

This script helps you capture reference circuits that will be used as ground truth
for circuit verification. Build your circuits physically and use this to record them.
"""

import cv2
import json
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
import matplotlib.pyplot as plt

# Import our visualizer
try:
    from dual_board_visualizer import DualBoardVisualizer, convert_detections_with_positions
    VISUALIZER_AVAILABLE = True
except ImportError:
    VISUALIZER_AVAILABLE = False
    print("⚠️ Dual board visualizer not available")

class GroundTruthCapture:
    def __init__(self):
        self.model = None
        self.visualizer = None
        self.ground_truths = {}
        self.current_circuit = None
        self.capture_active = False
        
        # Circuit definitions
        self.circuits = {
            1: {"name": "Light Switch Circuit", "difficulty": "easy", "description": "Build a circuit that lights up with the switch"},
            2: {"name": "Easy Circuit 2", "difficulty": "easy", "description": "TBD - Build on LEFT board only"}, 
            3: {"name": "Easy Circuit 3", "difficulty": "easy", "description": "TBD - Build on LEFT board only"},
            4: {"name": "Red Light Resistor Circuit", "difficulty": "hard", "description": "Build a circuit that emits red light with the resistor"},
            5: {"name": "Music Speaker Circuit", "difficulty": "hard", "description": "Make a circuit that plays music on a speaker"},
            6: {"name": "Hard Circuit 6", "difficulty": "hard", "description": "TBD - Build on LEFT board only"}
        }
        
        if VISUALIZER_AVAILABLE:
            self.visualizer = DualBoardVisualizer(cell_size=25)
        
        self.load_existing_ground_truths()
    
    def load_model(self):
        """Load YOLO model"""
        model_path = Path("dual_board_training/photos_model_fixed/weights/best.pt")
        
        if not model_path.exists():
            alt_paths = [
                Path("dual_board_training/photos_model_fixed/weights/last.pt"),
                Path("dual_board_training/photos_model/weights/best.pt")
            ]
            
            model_path = None
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    break
        
        if not model_path:
            print("❌ No trained model found!")
            return False
            
        try:
            self.model = YOLO(str(model_path))
            print(f"✅ Model loaded: {model_path}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_existing_ground_truths(self):
        """Load any existing ground truth data"""
        gt_file = Path("ground_truths.json")
        if gt_file.exists():
            try:
                with open(gt_file, 'r') as f:
                    self.ground_truths = json.load(f)
                print(f"📥 Loaded {len(self.ground_truths)} existing ground truths")
            except Exception as e:
                print(f"⚠️ Error loading ground truths: {e}")
                self.ground_truths = {}
        else:
            print("📝 No existing ground truths found - starting fresh")
    
    def save_ground_truths(self):
        """Save ground truth data to file"""
        try:
            # Convert numpy types to Python native types for JSON serialization
            def convert_numpy_types(obj):
                """Recursively convert numpy types to native Python types"""
                if isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif hasattr(obj, 'tolist'):  # numpy array
                    return obj.tolist()
                else:
                    return obj
            
            json_safe_data = convert_numpy_types(self.ground_truths)
            
            with open("ground_truths.json", 'w') as f:
                json.dump(json_safe_data, f, indent=2)
            print("💾 Ground truths saved successfully")
        except Exception as e:
            print(f"❌ Error saving ground truths: {e}")
            import traceback
            traceback.print_exc()
    
    def capture_ground_truth(self, circuit_id, frame, left_boxes, right_boxes):
        """Capture a ground truth circuit"""
        if not self.model:
            print("❌ Model not loaded")
            return False
            
        # Convert detections to our format
        height, width = frame.shape[:2]
        left_detections, right_detections = convert_detections_with_positions(
            left_boxes, right_boxes, self.model.names, width, height)
        
        # Store ground truth
        circuit_info = self.circuits[circuit_id]
        timestamp = datetime.now().isoformat()
        
        self.ground_truths[str(circuit_id)] = {
            "circuit_id": circuit_id,
            "name": circuit_info["name"],
            "difficulty": circuit_info["difficulty"],
            "description": circuit_info["description"],
            "timestamp": timestamp,
            "left_detections": left_detections,
            "right_detections": right_detections,
            "total_components": len(left_boxes) + len(right_boxes),
            "frame_size": {"width": width, "height": height}
        }
        
        print(f"✅ Captured ground truth for Circuit {circuit_id}")
        print(f"   📊 Total components: {len(left_boxes) + len(right_boxes)}")
        print(f"   📍 Left side: {len(left_boxes)} components")
        print(f"   📍 Right side: {len(right_boxes)} components")
        
        return True
    
    def preview_ground_truth(self, circuit_id):
        """Preview a captured ground truth"""
        if str(circuit_id) not in self.ground_truths:
            print(f"❌ No ground truth found for Circuit {circuit_id}")
            return
            
        gt = self.ground_truths[str(circuit_id)]
        print(f"\n📋 Ground Truth Preview - Circuit {circuit_id}")
        print(f"   Name: {gt['name']}")
        print(f"   Difficulty: {gt['difficulty']}")
        print(f"   Description: {gt['description']}")
        print(f"   Components: {gt['total_components']}")
        print(f"   Captured: {gt['timestamp']}")
        
        if VISUALIZER_AVAILABLE:
            # Show visualization
            left_det = gt["left_detections"]
            right_det = gt["right_detections"]
            
            fig = self.visualizer.create_dual_board_visualization(left_det, right_det)
            plt.suptitle(f"Ground Truth: {gt['name']}", fontsize=16, fontweight='bold')
            plt.show()
        
        # Show component details
        print("   📍 Left Board Components:")
        for comp_type, detections in gt["left_detections"].items():
            print(f"      • {comp_type}: {len(detections)} instances")
        
        print("   📍 Right Board Components:")
        for comp_type, detections in gt["right_detections"].items():
            print(f"      • {comp_type}: {len(detections)} instances")
    
    def run_capture_session(self):
        """Run the ground truth capture session"""
        if not self.load_model():
            return
            
        # Open camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open camera")
            return
            
        print("\n🎥 GROUND TRUTH CAPTURE SESSION")
        print("=" * 50)
        print("📋 Instructions:")
        print("   1. Build your circuit physically on the LEFT board ONLY")
        print("   2. Press number keys (1-6) to select which circuit you're capturing")
        print("   3. Press SPACE to capture the current frame as ground truth")
        print("   4. Press 'p' to preview existing ground truths")
        print("   5. Press 's' to save all ground truths to file")
        print("   6. Press 'q' to quit")
        print()
        print("⚠️  IMPORTANT: Build circuits on LEFT board only during capture!")
        print("   Participants will later be able to build on either board.")
        print()
        print("🔢 Circuit Numbers:")
        for cid, info in self.circuits.items():
            status = "✅ Captured" if str(cid) in self.ground_truths else "❌ Not captured"
            print(f"   {cid}: {info['name']} ({info['difficulty']}) - {status}")
        print()
        
        split_ratio = 0.5
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Failed to read frame")
                    break
                
                key = cv2.waitKey(1) & 0xFF
                
                # Handle circuit selection
                if key >= ord('1') and key <= ord('6'):
                    circuit_id = key - ord('0')
                    self.current_circuit = circuit_id
                    circuit_info = self.circuits[circuit_id]
                    print(f"\n🎯 Selected Circuit {circuit_id}: {circuit_info['name']}")
                    print(f"   Description: {circuit_info['description']}")
                    print("   Build this circuit and press SPACE to capture!")
                
                # Handle ground truth capture
                elif key == ord(' ') and self.current_circuit:
                    print(f"📸 Capturing ground truth for Circuit {self.current_circuit}...")
                    print("🔍 Running detection on LEFT board...")
                    print("   (Note: Only build circuit on LEFT side - system will duplicate for both boards)")
                    
                    # Run detection
                    results = self.model(frame, conf=0.6, iou=0.5)
                    
                    if results and len(results) > 0:
                        result = results[0]
                        if result.boxes is not None:
                            height, width = frame.shape[:2]
                            split_x = int(width * split_ratio)
                            
                            print(f"   Found {len(result.boxes)} total detections")
                            
                            # Only capture components on LEFT side for ground truth
                            left_boxes = []
                            right_boxes = []  # Will be empty for ground truth capture
                            left_components_found = 0
                            
                            for i, box in enumerate(result.boxes):
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                center_x = (x1 + x2) / 2
                                class_id = int(box.cls[0])
                                class_name = self.model.names[class_id]
                                confidence = float(box.conf[0])
                                
                                side = 'LEFT' if center_x < split_x else 'RIGHT'
                                print(f"   • {class_name} ({confidence:.2f}) - {side}")
                                
                                if center_x < split_x:
                                    left_boxes.append((i, box))
                                    left_components_found += 1
                                else:
                                    # Warn about components on right side during ground truth capture
                                    print(f"   ⚠️ Ignoring {class_name} on RIGHT side (ground truth captures LEFT side only)")
                            
                            if left_components_found == 0:
                                print("❌ No components found on LEFT board!")
                                print("   Please build your circuit on the LEFT side only")
                            else:
                                print(f"✅ Found {left_components_found} components on LEFT board - capturing...")
                                
                                # Capture ground truth (empty right_boxes for ground truth)
                                if self.capture_ground_truth(self.current_circuit, frame, left_boxes, []):
                                    # Save immediately after capture
                                    self.save_ground_truths()
                                    
                                    # Show preview
                                    if VISUALIZER_AVAILABLE:
                                        self.preview_ground_truth(self.current_circuit)
                        else:
                            print("⚠️ No bounding boxes found in detection results")
                    else:
                        print("⚠️ No detections found - make sure circuit is visible and well-lit")
                
                # Handle preview
                elif key == ord('p'):
                    print("\n📋 Ground Truth Status:")
                    for cid, info in self.circuits.items():
                        if str(cid) in self.ground_truths:
                            gt = self.ground_truths[str(cid)]
                            print(f"   ✅ Circuit {cid}: {gt['total_components']} components")
                        else:
                            print(f"   ❌ Circuit {cid}: Not captured")
                
                # Handle save
                elif key == ord('s'):
                    self.save_ground_truths()
                
                # Handle quit
                elif key == ord('q'):
                    break
                
                # Just show the raw camera feed - no automatic processing
                display_frame = frame.copy()
                height, width = frame.shape[:2]
                split_x = int(width * split_ratio)
                
                # Add split line and labels
                cv2.line(display_frame, (split_x, 0), (split_x, height), (255, 255, 255), 3)
                cv2.putText(display_frame, "LEFT", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, "RIGHT", (split_x + 10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Add current circuit info
                if self.current_circuit:
                    circuit_info = self.circuits[self.current_circuit]
                    info_text = f"Circuit {self.current_circuit}: {circuit_info['name']}"
                    cv2.putText(display_frame, info_text, (10, height - 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display_frame, "BUILD ON LEFT BOARD ONLY", (10, height - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(display_frame, "Press SPACE to capture this circuit!", (10, height - 20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "Press 1-6 to select circuit to capture", (10, height - 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(display_frame, "Build on LEFT board only!", (10, height - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                cv2.imshow('Ground Truth Capture', display_frame)
                
        except KeyboardInterrupt:
            print("\n🛑 Capture session interrupted")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Capture session complete")

def main():
    """Main function"""
    print("🏗️ GROUND TRUTH CIRCUIT CAPTURE SYSTEM")
    print("=" * 45)
    print()
    print("This tool helps you capture reference circuits for verification.")
    print("You'll build each circuit physically and capture it as ground truth.")
    print()
    
    capturer = GroundTruthCapture()
    capturer.run_capture_session()

if __name__ == "__main__":
    main()
