#!/usr/bin/env python3
"""
Batch Network Graph Converter
Converts all JSON graph outputs to professional network diagrams.
"""

import os
from pathlib import Path
from network_graph_visualizer import NetworkGraphVisualizer

def convert_all_to_network_graphs():
    """Convert all graph JSON files to network diagram style."""
    
    # Setup paths
    data_dir = Path("output/data")
    network_dir = Path("output/network_graphs")
    network_dir.mkdir(exist_ok=True)
    
    # Find all graph JSON files
    graph_files = list(data_dir.glob("graph_*.json"))
    
    if not graph_files:
        print("❌ No graph files found in output/data/")
        return
    
    print(f"🔗 Found {len(graph_files)} graph files to convert to network diagrams...")
    
    # Initialize visualizer
    visualizer = NetworkGraphVisualizer()
    
    converted = 0
    failed = 0
    
    for graph_file in graph_files:
        try:
            # Create output filename
            output_file = network_dir / f"{graph_file.stem}_network.png"
            
            print(f"📊 Converting: {graph_file.name}")
            
            # Convert to network graph
            visualizer.create_network_visualization(str(graph_file), str(output_file))
            converted += 1
            
        except Exception as e:
            print(f"❌ Failed to convert {graph_file.name}: {e}")
            failed += 1
    
    print(f"\n✅ Network graph conversion complete!")
    print(f"📊 Converted: {converted}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Network diagrams saved in: {network_dir}")

if __name__ == "__main__":
    convert_all_to_network_graphs() 