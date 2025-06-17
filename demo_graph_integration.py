#!/usr/bin/env python3
"""
Demonstration of graph format integration with existing detection pipeline.
This shows how the system now outputs both traditional JSON and graph format.
"""

import json
import time
from pathlib import Path
from data_structures import *
from graph_output_converter import DetectionToGraphConverter

def create_sample_detection_result():
    """Create a sample detection result for demonstration."""
    
    # Create a sample component detection
    component = ComponentDetection(
        id="wire-0",
        label="wire",
        bbox=BoundingBox(x1=100, y1=200, x2=300, y2=400),
        orientation=0.0,
        confidence=0.85,
        component_type=ComponentType.WIRE,
        switch_state=None,
        connection_points=[(100, 200), (300, 400)],
        metadata={"yolo_class_id": 0, "detection_index": 0}
    )
    
    # Create connection graph
    connection_graph = ConnectionGraph(
        components=[component],
        edges=[],
        state=CircuitState(
            is_circuit_closed=False,
            power_on=True,
            active_components=[],
            power_flow_path=[],
            estimated_voltage=None,
            estimated_current=None
        ),
        timestamp=time.time(),
        frame_id=1
    )
    
    # Create detection result
    detection_result = DetectionResult(
        connection_graph=connection_graph,
        raw_detections=[{
            "id": "wire-0",
            "label": "wire",
            "bbox": [100, 200, 300, 400],
            "orientation": 0.0,
            "confidence": 0.85,
            "component_type": "wire",
            "switch_state": None,
            "connection_points": [[100, 200], [300, 400]],
            "metadata": {"yolo_class_id": 0, "detection_index": 0}
        }],
        processing_time=0.025,
        error_message=None
    )
    
    return detection_result

def demo_integration():
    """Demonstrate the graph format integration."""
    print("🔧 Graph Format Integration Demo")
    print("=" * 40)
    
    # Create sample detection result
    detection_result = create_sample_detection_result()
    
    # Show traditional JSON output
    print("📄 TRADITIONAL JSON OUTPUT:")
    traditional_json = json.dumps(detection_result.to_dict(), indent=2)
    print(f"   Size: {len(traditional_json)} characters")
    print(f"   Components: {len(detection_result.connection_graph.components)}")
    
    # Convert to graph format
    print("\n🔄 CONVERTING TO GRAPH FORMAT...")
    converter = DetectionToGraphConverter()
    circuit_graph = converter.convert_detection_result(detection_result)
    
    # Show graph format output
    print("📊 GRAPH FORMAT OUTPUT:")
    graph_json = circuit_graph.to_json()
    graph_data = json.loads(graph_json)
    
    print(f"   Size: {len(graph_json)} characters")
    print(f"   Nodes: {len(graph_data['graph']['nodes'])}")
    print(f"   Edges: {len(graph_data['graph']['edges'])}")
    print(f"   Has circuit analysis: {'circuit_analysis' in graph_data['graph']}")
    
    # Save both formats
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time() * 1000)
    
    # Save traditional format
    traditional_path = output_dir / f"demo_detection_{timestamp}.json"
    with open(traditional_path, 'w', encoding='utf-8') as f:
        f.write(traditional_json)
    
    # Save graph format
    graph_path = output_dir / f"demo_graph_{timestamp}.json"
    with open(graph_path, 'w', encoding='utf-8') as f:
        f.write(graph_json)
    
    print(f"\n💾 FILES SAVED:")
    print(f"   Traditional: {traditional_path}")
    print(f"   Graph format: {graph_path}")
    
    # Show sample of graph structure
    print(f"\n📋 GRAPH STRUCTURE SAMPLE:")
    if graph_data['graph']['nodes']:
        node = graph_data['graph']['nodes'][0]
        print(f"   Node: {node['component_type']} @ ({node['position']['x']:.0f}, {node['position']['y']:.0f})")
        print(f"   Confidence: {node['detection_confidence']:.2f}")
        print(f"   Accessibility: {node['accessibility']:.2f}")
    
    analysis = graph_data['graph']['circuit_analysis']
    print(f"   Circuit complete: {analysis['circuit_complete']}")
    print(f"   Connected components: {len(analysis['connectivity']['connected_components'])}")
    
    print(f"\n✅ INTEGRATION SUCCESSFUL!")
    print(f"   Your detection pipeline now outputs both formats.")
    print(f"   Graph files are ready for TD-BKT algorithm consumption.")
    print(f"   No recommendations included - external system handles TD-BKT.")

if __name__ == "__main__":
    demo_integration() 