#!/usr/bin/env python3
"""
Debug the confidence value pipeline to see where they get lost
"""

import cv2
import numpy as np
import time
from models.component_detector import ComponentDetector
from circuit.graph_builder import CircuitGraphBuilder
from graph_output_converter import DetectionToGraphConverter
import json

def debug_confidence_pipeline():
    print("🔍 CONFIDENCE PIPELINE DEBUG")
    print("=" * 50)
    
    # Initialize components
    detector = ComponentDetector("models/weights/dual_board_dual_board_1758049257.pt")
    detector.confidence_threshold = 0.1
    detector.model.conf = 0.1
    
    graph_builder = CircuitGraphBuilder()
    graph_converter = DetectionToGraphConverter()
    
    # Get a frame from camera
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Could not capture frame")
        return
    
    print("📹 Captured frame")
    
    # Step 1: Component detection
    print("\n1️⃣ COMPONENT DETECTION")
    components = detector.detect(frame)
    print(f"   Detected {len(components)} components:")
    for i, comp in enumerate(components):
        print(f"      {i+1}. {comp.component_type.value}: confidence={comp.confidence:.3f}")
    
    if len(components) == 0:
        print("   ❌ No components detected - check camera view")
        return
    
    # Step 2: Graph building
    print("\n2️⃣ GRAPH BUILDING")
    connection_graph = graph_builder.build_graph(components, [], time.time(), 0)
    print(f"   Graph has {len(connection_graph.components)} components:")
    for i, comp in enumerate(connection_graph.components):
        print(f"      {i+1}. {comp.component_type.value}: confidence={comp.confidence:.3f}")
    
    # Step 3: Graph conversion
    print("\n3️⃣ GRAPH CONVERSION")
    from data_structures import DetectionResult
    detection_result = DetectionResult(
        connection_graph=connection_graph,
        raw_detections=[comp.to_dict() for comp in components],
        processing_time=0.1
    )
    
    try:
        print(f"   Converting {len(connection_graph.components)} components...")
        circuit_graph = graph_converter.convert_detection_result(detection_result)
        print(f"   ✅ Conversion completed")
        
        # Try to convert to JSON
        json_str = circuit_graph.to_json()
        graph_data = json.loads(json_str)
        print(f"   ✅ JSON conversion completed")
        
        print(f"   Converted graph has data:")
        
        # Print the actual JSON structure to debug
        print(f"   📋 JSON structure keys: {list(graph_data.keys())}")
        
        # Check for components in multiple possible locations
        components_data = graph_data.get("connection_graph", {}).get("components", [])
        print(f"   📍 Looking in connection_graph.components: {len(components_data)} components")
        
        # Also check root level
        root_components = graph_data.get("components", [])
        print(f"   📍 Looking in root.components: {len(root_components)} components")
        
        # Check if connection_graph exists at all
        conn_graph = graph_data.get("connection_graph", {})
        print(f"   📍 connection_graph keys: {list(conn_graph.keys()) if isinstance(conn_graph, dict) else 'Not a dict'}")
        
        # Use whichever has components
        if len(components_data) > 0:
            actual_components = components_data
            print(f"   ✅ Using connection_graph.components")
        elif len(root_components) > 0:
            actual_components = root_components  
            print(f"   ✅ Using root.components")
        else:
            actual_components = []
            print(f"   ❌ No components found anywhere!")
            print(f"   📋 Full JSON structure:")
            print(json.dumps(graph_data, indent=2)[:1000])  # First 1000 chars
        
        print(f"   Found {len(actual_components)} components total:")
        for i, comp_data in enumerate(actual_components):
            conf = comp_data.get("confidence", "MISSING")
            comp_type = comp_data.get("component_type", "UNKNOWN")
            print(f"      {i+1}. {comp_type}: confidence={conf}")
        
        # Store for later use
        components_data = actual_components
            
    except Exception as e:
        print(f"   ❌ ERROR in graph conversion: {e}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        components_data = []
    
    # Step 4: Visualizer filtering
    print("\n4️⃣ VISUALIZER FILTERING")
    visualizer_threshold = 0.15
    high_conf_components = [
        comp for comp in components_data 
        if comp.get("confidence", 0) > visualizer_threshold
    ]
    
    print(f"   Components above {visualizer_threshold*100}% threshold: {len(high_conf_components)}")
    for i, comp in enumerate(high_conf_components):
        conf = comp.get("confidence", 0)
        comp_type = comp.get("component_type", "UNKNOWN")
        print(f"      {i+1}. {comp_type}: confidence={conf:.3f}")
    
    print(f"\n🎯 SUMMARY:")
    print(f"   Step 1 (Detection): {len(components)} components")
    print(f"   Step 2 (Graph): {len(connection_graph.components)} components")  
    print(f"   Step 3 (Conversion): {len(components_data)} components")
    print(f"   Step 4 (Visualizer): {len(high_conf_components)} components")
    
    if len(components) > 0 and len(high_conf_components) == 0:
        print(f"\n❌ CONFIDENCE LOST between detection and visualizer!")
        print(f"   Detection confidence: {components[0].confidence:.3f}")
        if len(components_data) > 0:
            vis_conf = components_data[0].get("confidence", "MISSING")
            print(f"   Visualizer confidence: {vis_conf}")

if __name__ == "__main__":
    debug_confidence_pipeline()
