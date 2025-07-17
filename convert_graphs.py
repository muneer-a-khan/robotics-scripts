#!/usr/bin/env python3
"""
Comprehensive Graph Converter
Choose between circuit diagram style or professional network graph style.
"""

import sys
import os
from pathlib import Path
from visualize_graph_output import CircuitVisualizer
from network_graph_visualizer import NetworkGraphVisualizer

def show_menu():
    """Display conversion options menu."""
    print("\n🎨 GRAPH VISUALIZATION CONVERTER")
    print("=" * 40)
    print("Choose your visualization style:")
    print("")
    print("1. 📊 Circuit Diagram Style")
    print("   - Component details table")
    print("   - Circuit analysis panel") 
    print("   - Session metadata")
    print("   - Traditional circuit layout")
    print("")
    print("2. 🔗 Network Graph Style") 
    print("   - Professional network diagram")
    print("   - Oval nodes with clean labels")
    print("   - Directed arrows")
    print("   - Similar to Cytoscape/Gephi")
    print("")
    print("3. 🎯 Both Styles")
    print("   - Generate both visualizations")
    print("")
    print("4. 📁 List Available Files")
    print("")
    print("5. ❌ Exit")
    print("")

def list_available_files():
    """List all available graph files."""
    data_dir = Path("output/data")
    graph_files = list(data_dir.glob("graph_*.json"))
    
    if not graph_files:
        print("❌ No graph files found in output/data/")
        return []
    
    print(f"\n📋 Available Graph Files ({len(graph_files)}):")
    print("-" * 50)
    
    for i, graph_file in enumerate(graph_files, 1):
        file_size = graph_file.stat().st_size / 1024  # KB
        print(f"{i:2d}. {graph_file.name} ({file_size:.1f} KB)")
    
    return graph_files

def convert_single_file(graph_file: Path, style: str):
    """Convert a single file to specified style."""
    try:
        if style == "circuit" or style == "both":
            # Circuit diagram style
            circuit_visualizer = CircuitVisualizer()
            circuit_output = Path("output/visuals") / f"{graph_file.stem}_circuit.png"
            circuit_output.parent.mkdir(exist_ok=True)
            
            print(f"📊 Creating circuit diagram: {graph_file.name}")
            circuit_visualizer.create_visual_diagram(str(graph_file), str(circuit_output))
        
        if style == "network" or style == "both":
            # Network graph style  
            network_visualizer = NetworkGraphVisualizer()
            network_output = Path("output/network_graphs") / f"{graph_file.stem}_network.png"
            network_output.parent.mkdir(exist_ok=True)
            
            print(f"🔗 Creating network graph: {graph_file.name}")
            network_visualizer.create_network_visualization(str(graph_file), str(network_output))
            
        return True
        
    except Exception as e:
        print(f"❌ Error converting {graph_file.name}: {e}")
        return False

def convert_all_files(style: str):
    """Convert all graph files to specified style."""
    data_dir = Path("output/data")
    graph_files = list(data_dir.glob("graph_*.json"))
    
    if not graph_files:
        print("❌ No graph files found in output/data/")
        return
    
    print(f"\n🔄 Converting {len(graph_files)} files...")
    
    converted = 0
    failed = 0
    
    for graph_file in graph_files:
        if convert_single_file(graph_file, style):
            converted += 1
        else:
            failed += 1
    
    print(f"\n✅ Conversion complete!")
    print(f"📊 Converted: {converted}")
    print(f"❌ Failed: {failed}")
    
    if style == "circuit" or style == "both":
        print(f"📁 Circuit diagrams in: output/visuals/")
    if style == "network" or style == "both":
        print(f"📁 Network graphs in: output/network_graphs/")

def main():
    """Main interactive conversion program."""
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                # Circuit diagram style
                convert_all_files("circuit")
                
            elif choice == "2":
                # Network graph style
                convert_all_files("network")
                
            elif choice == "3":
                # Both styles
                convert_all_files("both")
                
            elif choice == "4":
                # List files
                files = list_available_files()
                
                if files:
                    print("\nWould you like to convert a specific file?")
                    file_choice = input("Enter file number (or press Enter to return to menu): ").strip()
                    
                    if file_choice.isdigit():
                        file_idx = int(file_choice) - 1
                        if 0 <= file_idx < len(files):
                            selected_file = files[file_idx]
                            
                            print("\nChoose style for this file:")
                            print("1. Circuit Diagram")
                            print("2. Network Graph") 
                            print("3. Both")
                            
                            style_choice = input("Enter choice (1-3): ").strip()
                            
                            if style_choice == "1":
                                convert_single_file(selected_file, "circuit")
                            elif style_choice == "2":
                                convert_single_file(selected_file, "network")
                            elif style_choice == "3":
                                convert_single_file(selected_file, "both")
                            else:
                                print("❌ Invalid choice")
                        else:
                            print("❌ Invalid file number")
                
            elif choice == "5":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1-5.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 