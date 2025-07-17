#!/usr/bin/env python3
"""
Batch Graph Converter
Converts all JSON graph outputs to visual diagrams with one command.
"""

import os
from pathlib import Path
from visualize_graph_output import CircuitVisualizer

def convert_all_graphs():
    """Convert all graph JSON files to visual diagrams."""
    
    # Setup paths
    data_dir = Path("output/data")
    visual_dir = Path("output/visuals")
    visual_dir.mkdir(exist_ok=True)
    
    # Find all graph JSON files
    graph_files = list(data_dir.glob("graph_*.json"))
    
    if not graph_files:
        print("❌ No graph files found in output/data/")
        return
    
    print(f"🔍 Found {len(graph_files)} graph files to convert...")
    
    # Initialize visualizer
    visualizer = CircuitVisualizer()
    
    converted = 0
    failed = 0
    
    for graph_file in graph_files:
        try:
            # Create output filename
            output_file = visual_dir / f"{graph_file.stem}_visual.png"
            
            print(f"📸 Converting: {graph_file.name}")
            
            # Convert to visual
            visualizer.create_visual_diagram(str(graph_file), str(output_file))
            converted += 1
            
        except Exception as e:
            print(f"❌ Failed to convert {graph_file.name}: {e}")
            failed += 1
    
    print(f"\n✅ Conversion complete!")
    print(f"📊 Converted: {converted}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Visual diagrams saved in: {visual_dir}")

if __name__ == "__main__":
    convert_all_graphs() 