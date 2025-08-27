#!/usr/bin/env python3
"""
Debug version of main.py to troubleshoot live circuit visualization issues.
"""

import cv2
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from config import VIDEO_CONFIG, OUTPUT_CONFIG, YOLO_CONFIG
from models.component_detector import ComponentDetector
from vision.connection_detector import ConnectionDetector
from circuit.graph_builder import CircuitGraphBuilder
from data_structures import DetectionResult
from graph_output_converter import DetectionToGraphConverter
from live_circuit_visualizer import create_live_visualization
from circuit_validator import CircuitValidator
from enhanced_orientation_detector import EnhancedOrientationDetector


def debug_camera_mode():
    """Debug version of camera mode with detailed logging."""
    
    print("🔍 DEBUG MODE: Starting camera with detailed logging")
    print("=" * 60)
    
    # Initialize components
    print("1. Initializing component detector...")
    component_detector = ComponentDetector()
    
    print("2. Initializing connection detector...")
    connection_detector = ConnectionDetector()
    
    print("3. Initializing graph builder...")
    graph_builder = CircuitGraphBuilder()
    
    print("4. Initializing graph converter...")
    graph_converter = DetectionToGraphConverter()
    
    # Initialize camera
    print("5. Starting camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera started successfully!")
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    frame_count = 0
    last_process_time = 0
    processing_interval = 3.0  # 3 seconds
    
    print(f"🔄 Processing every {processing_interval} seconds")
    print("Press 'q' to quit, 's' to save current frame")
    print("=" * 60)
    
    try:
        while True:
            current_time = time.time()
            
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Process frame at intervals
            should_process = (current_time - last_process_time) >= processing_interval
            
            if should_process:
                print(f"\n📸 Processing frame {frame_count + 1}...")
                
                # Step 1: Detect components
                print("  🔍 Detecting components...")
                components = component_detector.detect(frame)
                print(f"    Found {len(components)} components")
                
                # Show component details
                for i, comp in enumerate(components):
                    print(f"    Component {i}: {comp.component_type} (confidence: {comp.confidence:.3f})")
                
                # Step 2: Detect connections
                print("  🔗 Detecting connections...")
                connections = connection_detector.detect_connections(frame, components)
                print(f"    Found {len(connections)} connections")
                
                # Step 3: Build graph
                print("  🕸️ Building circuit graph...")
                connection_graph = graph_builder.build_graph(
                    components, connections, current_time, frame_count
                )
                print(f"    Graph has {len(connection_graph.components)} components and {len(connection_graph.edges)} edges")
                
                # Step 4: Convert to graph format
                print("  📊 Converting to graph format...")
                circuit_graph = graph_converter.convert_detection_result(
                    DetectionResult(
                        connection_graph=connection_graph,
                        raw_detections=[comp.to_dict() for comp in components],
                        processing_time=0,
                        validation_result=None
                    )
                )
                
                # Step 5: Generate visualization
                print("  🎨 Generating live circuit visualization...")
                try:
                    graph_data = json.loads(circuit_graph.to_json())
                    
                    # Check if we have high-confidence components
                    high_conf_components = [
                        comp for comp in graph_data.get("connection_graph", {}).get("components", [])
                        if comp.get("confidence", 0) > 0.75
                    ]
                    
                    print(f"    High-confidence components (>75%): {len(high_conf_components)}")
                    for comp in high_conf_components:
                        print(f"      {comp['component_type']}: {comp['confidence']:.3f}")
                    
                    if not high_conf_components:
                        print("    ⚠️ No high-confidence components found - visualization will be empty")
                    
                    # Create visualization
                    timestamp = int(current_time * 1000)
                    visualization_path = create_live_visualization(
                        graph_data, 
                        timestamp=timestamp, 
                        output_dir=str(output_dir),
                        validation_data=None
                    )
                    
                    if visualization_path:
                        print(f"    ✅ Visualization created: {visualization_path}")
                        
                        # Also save as latest
                        latest_path = output_dir / "latest_circuit_visual.png"
                        import shutil
                        shutil.copy2(visualization_path, latest_path)
                        print(f"    📁 Latest visualization: {latest_path}")
                    else:
                        print("    ❌ Visualization creation failed - no path returned")
                        
                except Exception as e:
                    print(f"    ❌ ERROR creating visualization: {e}")
                    import traceback
                    traceback.print_exc()
                
                # Update counters
                frame_count += 1
                last_process_time = current_time
                
                print(f"✅ Frame {frame_count} processed successfully!")
                print("-" * 40)
            
            # Display frame
            cv2.imshow("Debug Camera", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                save_path = output_dir / f"debug_save_{int(time.time())}.jpg"
                cv2.imwrite(str(save_path), frame)
                print(f"📸 Frame saved to {save_path}")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("🧹 Cleanup complete")


if __name__ == "__main__":
    debug_camera_mode() 