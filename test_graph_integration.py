#!/usr/bin/env python3
"""
Test script to verify graph format integration with existing detection pipeline.
"""

import json
from pathlib import Path
from main import SnapCircuitVisionSystem

def test_graph_integration():
    """Test that graph format is properly integrated."""
    print("🧪 Testing Graph Format Integration")
    print("=" * 40)
    
    # Initialize the vision system
    vision_system = SnapCircuitVisionSystem(
        model_path="models/weights/snap_circuit_yolov8.pt",
        save_outputs=True,
        display_results=False
    )
    
    # Test with an image
    test_image = "snap_circuit_image.jpg"
    if not Path(test_image).exists():
        print(f"❌ Test image {test_image} not found")
        return
    
    print(f"📸 Processing test image: {test_image}")
    
    # Process the image
    result = vision_system.process_image(test_image)
    
    # Check if detection worked
    if result.error_message:
        print(f"❌ Detection failed: {result.error_message}")
        return
    
    print(f"✅ Detection completed successfully")
    print(f"   • Processing time: {result.processing_time:.3f}s")
    print(f"   • Components found: {len(result.connection_graph.components)}")
    print(f"   • Connections found: {len(result.connection_graph.edges)}")
    
    # Test graph conversion
    try:
        circuit_graph = vision_system.graph_converter.convert_detection_result(result)
        print(f"✅ Graph conversion successful")
        print(f"   • Graph nodes: {len(circuit_graph.graph.nodes)}")
        print(f"   • Graph edges: {len(circuit_graph.graph.edges)}")
        
        # Test JSON export
        graph_json = circuit_graph.to_json()
        graph_data = json.loads(graph_json)
        
        print(f"✅ JSON export successful")
        print(f"   • JSON size: {len(graph_json)} characters")
        print(f"   • Contains nodes: {'nodes' in graph_data['graph']}")
        print(f"   • Contains edges: {'edges' in graph_data['graph']}")
        print(f"   • Contains analysis: {'circuit_analysis' in graph_data['graph']}")
        
        # Save test output
        output_path = Path("output/test_graph_output.json")
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(graph_json)
        
        print(f"✅ Test graph saved to: {output_path}")
        
        # Show sample of the graph structure
        print("\n📊 Sample Graph Structure:")
        if graph_data['graph']['nodes']:
            node = graph_data['graph']['nodes'][0]
            print(f"   Node example: {node['component_type']} @ ({node['position']['x']:.0f}, {node['position']['y']:.0f})")
        
        analysis = graph_data['graph']['circuit_analysis']
        print(f"   Circuit complete: {analysis['circuit_complete']}")
        print(f"   Power sources: {len(analysis['power_sources'])}")
        
    except Exception as e:
        print(f"❌ Graph conversion failed: {e}")
        return
    
    print(f"\n🎉 Integration test completed successfully!")
    print(f"   The system now outputs both traditional detection JSON and graph format.")
    print(f"   Graph files are saved with 'graph_' prefix for TD-BKT pipeline consumption.")

if __name__ == "__main__":
    test_graph_integration() 