#!/usr/bin/env python3
"""
Dual Circuit Board Visualization Monitor

This script monitors the output/data/left/ and output/data/right/ folders for new graph JSON files
and automatically generates live circuit visualizations with validation for both circuit boards.

The visualizations will look exactly like the attached image with:
- High confidence components (>75%)
- Validation scores and status
- Component connections
- Real-time updates for both left and right boards
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


class DualGraphFileHandler(FileSystemEventHandler):
    """Handler for monitoring dual graph JSON files and creating visualizations."""
    
    def __init__(self, output_dir="output", validation_enabled=True):
        self.output_dir = Path(output_dir)
        self.left_data_dir = self.output_dir / "data" / "left"
        self.right_data_dir = self.output_dir / "data" / "right"
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
        
        print(f"📁 Monitoring LEFT: {self.left_data_dir}")
        print(f"📁 Monitoring RIGHT: {self.right_data_dir}")
        print(f"📊 Validation: {'ENABLED' if validation_enabled else 'DISABLED'}")
        print("🔄 Auto-generating dual live circuit visualizations...")
    
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
        
        # Determine which side this file belongs to
        side = self._determine_side(file_path)
        if not side:
            return
        
        print(f"\n📄 Processing {side.upper()} side: {file_path.name}")
        
        try:
            # Load graph data
            with open(file_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            
            # Extract timestamp from filename
            timestamp = None
            if 'graph_' in file_path.name:
                try:
                    # Handle dual board naming: graph_left_timestamp_frame.json
                    parts = file_path.name.split('_')
                    if len(parts) >= 3:
                        timestamp_str = parts[2]  # Skip 'graph' and 'left'/'right'
                        timestamp = int(timestamp_str)
                except:
                    timestamp = int(time.time() * 1000)
            
            # Convert the graph data to the format expected by the visualizer
            converted_graph_data = self._convert_graph_format(graph_data)
            
            # Generate validation data if enabled
            validation_data = None
            if self.validation_enabled and self.circuit_validator:
                validation_data = self._generate_validation_data(converted_graph_data)
            
            # Create visualization with side-specific naming
            visualization_path = self._create_side_visualization(
                converted_graph_data, timestamp, side, validation_data
            )
            
            if visualization_path:
                print(f"✅ {side.upper()} visualization created: {visualization_path}")
                
                # Also save as "latest" for this side
                latest_path = self.output_dir / f"latest_circuit_visual_{side}.png"
                import shutil
                shutil.copy2(visualization_path, latest_path)
                print(f"📁 Latest {side} visualization: {latest_path}")
                
                # Mark as processed
                self.processed_files.add(file_path)
            else:
                print(f"❌ Failed to create visualization for {side} side: {file_path.name}")
                
        except Exception as e:
            print(f"❌ Error processing {side} side {file_path.name}: {e}")
    
    def _determine_side(self, file_path: Path) -> str:
        """Determine if file belongs to left or right side."""
        if "/left/" in str(file_path) or "_left_" in file_path.name:
            return "left"
        elif "/right/" in str(file_path) or "_right_" in file_path.name:
            return "right"
        else:
            return None
    
    def _create_side_visualization(self, graph_data, timestamp, side, validation_data):
        """Create visualization with side-specific naming."""
        # Create visualization
        visualization_path = create_live_visualization(
            graph_data=graph_data,
            timestamp=timestamp,
            output_dir=str(self.output_dir),
            validation_data=validation_data
        )
        
        if visualization_path:
            # Rename to include side information
            original_path = Path(visualization_path)
            side_viz_path = self.output_dir / f"live_circuit_visual_{side}_{timestamp}.png"
            
            import shutil
            shutil.move(original_path, side_viz_path)
            return side_viz_path
        
        return None
    
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
        """Process all existing graph files in both left and right data directories."""
        print("\n🔍 Processing existing graph files...")
        
        total_processed = 0
        
        # Process left side files
        if self.left_data_dir.exists():
            left_files = list(self.left_data_dir.glob("graph_*.json"))
            print(f"📄 Found {len(left_files)} LEFT side graph files")
            
            # Sort by modification time
            left_files.sort(key=lambda x: x.stat().st_mtime)
            
            for graph_file in left_files:
                self.process_graph_file(graph_file)
                total_processed += 1
        
        # Process right side files
        if self.right_data_dir.exists():
            right_files = list(self.right_data_dir.glob("graph_*.json"))
            print(f"📄 Found {len(right_files)} RIGHT side graph files")
            
            # Sort by modification time
            right_files.sort(key=lambda x: x.stat().st_mtime)
            
            for graph_file in right_files:
                self.process_graph_file(graph_file)
                total_processed += 1
        
        if total_processed == 0:
            print("⚠️ No existing graph files found in left or right directories")
            print("Make sure to run the dual camera system first to generate graph files")
        else:
            print(f"✅ Processed {total_processed} existing graph files")
    
    def start_monitoring(self):
        """Start monitoring both left and right data directories."""
        observer = Observer()
        
        # Monitor left directory
        if self.left_data_dir.exists():
            observer.schedule(self, str(self.left_data_dir), recursive=False)
        else:
            print(f"⚠️ Left data directory doesn't exist: {self.left_data_dir}")
            self.left_data_dir.mkdir(parents=True, exist_ok=True)
            observer.schedule(self, str(self.left_data_dir), recursive=False)
        
        # Monitor right directory
        if self.right_data_dir.exists():
            observer.schedule(self, str(self.right_data_dir), recursive=False)
        else:
            print(f"⚠️ Right data directory doesn't exist: {self.right_data_dir}")
            self.right_data_dir.mkdir(parents=True, exist_ok=True)
            observer.schedule(self, str(self.right_data_dir), recursive=False)
        
        observer.start()
        
        print(f"👀 Started monitoring dual circuit boards")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            print("\n🛑 Stopped dual monitoring")
        
        observer.join()


def main():
    """Main function to start the dual visualization monitor."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dual Circuit Board Visualization Monitor")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--no-validation", action="store_true", help="Disable validation")
    parser.add_argument("--process-existing", action="store_true", help="Process existing files first")
    
    args = parser.parse_args()
    
    print("🚀 Starting Dual Circuit Board Visualization Monitor")
    print("=" * 60)
    print("This will monitor BOTH left and right circuit boards:")
    print("✅ Process existing graph files from both sides")
    print("👀 Watch for new graph files in left/ and right/")
    print("🎨 Create separate live circuit visualizations")
    print("📊 Include validation data for each side")
    print("=" * 60)
    
    # Create handler
    handler = DualGraphFileHandler(
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
