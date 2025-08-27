#!/usr/bin/env python3
"""
Live Circuit Visualization Monitor

This script monitors the output/data/ folder for new graph JSON files and automatically
generates live circuit visualizations with validation, similar to the confidence_filtered_validation images.

The visualizations will look exactly like the attached image with:
- High confidence components (>75%)
- Validation scores and status
- Component connections
- Real-time updates every 3 seconds
"""

import json
import time
import os
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from live_circuit_visualizer import create_live_visualization
from circuit_validator import CircuitValidator
from enhanced_orientation_detector import EnhancedOrientationDetector


class GraphFileHandler(FileSystemEventHandler):
    """Handler for monitoring graph JSON files and creating visualizations."""
    
    def __init__(self, output_dir="output", validation_enabled=True):
        self.output_dir = Path(output_dir)
        self.data_dir = self.output_dir / "data"
        self.validation_enabled = validation_enabled
        
        # Initialize validation components if enabled
        if self.validation_enabled:
            print("🔧 Initializing circuit validation system...")
            self.circuit_validator = CircuitValidator()
            self.orientation_detector = EnhancedOrientationDetector()
        else:
            self.circuit_validator = None
            self.orientation_detector = None
        
        # Track processed files to avoid duplicates
        self.processed_files = set()
        
        print(f"📁 Monitoring: {self.data_dir}")
        print(f"📊 Validation: {'ENABLED' if validation_enabled else 'DISABLED'}")
        print("🔄 Auto-generating live circuit visualizations...")
    
    def on_created(self, event):
        """Called when a new file is created."""
        if not event.is_directory and event.src_path.endswith('.json'):
            self.process_graph_file(event.src_path)
    
    def on_modified(self, event):
        """Called when a file is modified."""
        if not event.is_directory and event.src_path.endswith('.json'):
            self.process_graph_file(event.src_path)
    
    def process_graph_file(self, file_path):
        """Process a graph JSON file and create visualization."""
        file_path = Path(file_path)
        
        # Skip if already processed
        if file_path in self.processed_files:
            return
        
        # Only process graph_*.json files
        if not file_path.name.startswith('graph_'):
            return
        
        print(f"\n📄 Processing: {file_path.name}")
        
        try:
            # Load graph data
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Extract timestamp from filename
            timestamp = None
            if 'graph_' in file_path.name:
                try:
                    timestamp_str = file_path.name.split('_')[1]
                    timestamp = int(timestamp_str)
                except:
                    timestamp = int(time.time() * 1000)
            
            # Convert the graph data to the format expected by the visualizer
            converted_graph_data = self._convert_graph_format(graph_data)
            
            # Generate validation data if enabled
            validation_data = None
            if self.validation_enabled and self.circuit_validator:
                validation_data = self._generate_validation_data(converted_graph_data)
            
            # Create visualization
            visualization_path = create_live_visualization(
                graph_data=converted_graph_data,
                timestamp=timestamp,
                output_dir=str(self.output_dir),
                validation_data=validation_data
            )
            
            if visualization_path:
                print(f"✅ Visualization created: {visualization_path}")
                
                # Also save as "latest" for easy access
                latest_path = self.output_dir / "latest_circuit_visual.png"
                import shutil
                shutil.copy2(visualization_path, latest_path)
                print(f"📁 Latest visualization: {latest_path}")
                
                # Mark as processed
                self.processed_files.add(file_path)
            else:
                print(f"❌ Failed to create visualization for {file_path.name}")
                
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
    
    def _convert_graph_format(self, graph_data):
        """Convert the graph data format to match what the visualizer expects."""
        # The visualizer expects: graph_data.get("connection_graph", {}).get("components", [])
        # But we have: graph_data.get("graph", {}).get("nodes", [])
        
        converted_data = {
            "connection_graph": {
                "components": [],
                "edges": [],
                "state": {
                    "is_circuit_closed": False,
                    "power_on": False
                }
            }
        }
        
        # Convert nodes to components
        nodes = graph_data.get("graph", {}).get("nodes", [])
        for node in nodes:
            # Extract position from the node
            position = node.get("position", {})
            x = position.get("x", 0)
            y = position.get("y", 0)
            width = position.get("width", 100)
            height = position.get("height", 100)
            
            # Create bbox from position
            bbox = [x, y, x + width, y + height]
            
            # Use placement_confidence as confidence
            confidence = node.get("placement_confidence", 0.0)
            
            # Create component in the expected format
            component = {
                "id": node.get("id", "unknown"),
                "component_type": node.get("component_type", "unknown"),
                "confidence": confidence,
                "bbox": bbox,
                "connection_points": []  # We'll need to extract these if available
            }
            
            converted_data["connection_graph"]["components"].append(component)
        
        # Convert edges if available
        edges = graph_data.get("graph", {}).get("edges", [])
        for edge in edges:
            converted_edge = {
                "component_1": edge.get("source", "unknown"),
                "component_2": edge.get("target", "unknown"),
                "connection_type": "wire"
            }
            converted_data["connection_graph"]["edges"].append(converted_edge)
        
        return converted_data
    
    def _generate_validation_data(self, graph_data):
        """Generate validation data for the circuit."""
        try:
            # Extract components from graph data
            components = graph_data.get("connection_graph", {}).get("components", [])
            
            if not components:
                return {
                    "overall_result": "unknown",
                    "summary": {"score": 0, "errors": 0, "warnings": 0},
                    "issues": []
                }
            
            # Convert to component objects for validation
            from data_structures import Component
            component_objects = []
            
            for comp_data in components:
                comp = Component(
                    component_type=comp_data.get("component_type", "unknown"),
                    bbox=comp_data.get("bbox", [0, 0, 100, 100]),
                    confidence=comp_data.get("confidence", 0.0),
                    connection_points=comp_data.get("connection_points", [])
                )
                component_objects.append(comp)
            
            # Find matching reference design
            from circuit.graph_builder import CircuitGraphBuilder
            graph_builder = CircuitGraphBuilder()
            
            # Create a simple connection graph for validation
            connection_graph = graph_builder.build_graph(
                component_objects, [], time.time(), 0
            )
            
            # Find matching reference design
            reference_design = self.circuit_validator.find_matching_reference_design(
                connection_graph
            )
            
            # Validate circuit
            validation_result = self.circuit_validator.validate_circuit(
                connection_graph, reference_design, confidence_threshold=0.75
            )
            
            return validation_result
            
        except Exception as e:
            print(f"⚠️ Warning: Could not generate validation data: {e}")
            return {
                "overall_result": "unknown",
                "summary": {"score": 0, "errors": 0, "warnings": 0},
                "issues": []
            }
    
    def process_existing_files(self):
        """Process all existing graph files in the data directory."""
        print("\n🔍 Processing existing graph files...")
        
        if not self.data_dir.exists():
            print(f"❌ Data directory not found: {self.data_dir}")
            return
        
        graph_files = list(self.data_dir.glob("graph_*.json"))
        print(f"📄 Found {len(graph_files)} existing graph files")
        
        # Process files in chronological order
        graph_files.sort(key=lambda x: x.stat().st_mtime)
        
        for graph_file in graph_files:
            self.process_graph_file(graph_file)
    
    def start_monitoring(self):
        """Start monitoring the data directory."""
        observer = Observer()
        observer.schedule(self, str(self.data_dir), recursive=False)
        observer.start()
        
        print(f"👀 Started monitoring {self.data_dir}")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\n🛑 Stopped monitoring")
        
        observer.join()


def main():
    """Main function to start the visualization monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Circuit Visualization Monitor")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--no-validation", action="store_true", help="Disable validation")
    parser.add_argument("--process-existing", action="store_true", help="Process existing files first")
    
    args = parser.parse_args()
    
    # Create handler
    handler = GraphFileHandler(
        output_dir=args.output_dir,
        validation_enabled=not args.no_validation
    )
    
    # Process existing files if requested
    if args.process_existing:
        handler.process_existing_files()
    
    # Start monitoring
    handler.start_monitoring()


if __name__ == "__main__":
    main() 