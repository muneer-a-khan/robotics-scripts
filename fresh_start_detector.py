"""
Fresh Start Component and Connection Detector
Detects snap circuit components and their connections
"""

import cv2
import numpy as np
import torch
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import math
from dataclasses import dataclass
from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import ConnectionPatch
from datetime import datetime

from fresh_start_config import FreshStartConfig

@dataclass
class ComponentDetection:
    """Represents a detected component"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]
    area: float

@dataclass
class Connection:
    """Represents a connection between two components"""
    component1: ComponentDetection
    component2: ComponentDetection
    distance: float
    angle: float
    connection_type: str  # "wire", "direct", "proximity"

class FreshStartDetector:
    def __init__(self, model_path: Optional[str] = None):
        self.config = FreshStartConfig()
        
        # Load model
        if model_path is None:
            # Try to find the most recent model
            models_dir = self.config.MODELS_DIR
            if models_dir.exists():
                model_files = list(models_dir.glob("*.pt"))
                if model_files:
                    # Sort by modification time and get the most recent
                    model_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                    model_path = model_files[0]
                else:
                    model_path = self.config.get_model_save_path()
            else:
                model_path = self.config.get_model_save_path()
            
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
            
        self.model = YOLO(model_path)
        self.logger = self._setup_logging()
        
    def _setup_logging(self):
        """Setup logging for the detector"""
        import logging
        logging.basicConfig(level=logging.INFO)
        return logging.getLogger(__name__)
        
    def detect_components(self, image_path: str) -> List[ComponentDetection]:
        """Detect components in an image"""
        self.logger.info(f"Detecting components in: {image_path}")
        
        # Run inference
        results = self.model(
            image_path,
            conf=self.config.CONFIDENCE_THRESHOLD,
            iou=self.config.NMS_THRESHOLD,
            max_det=self.config.MAX_DETECTIONS,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            if result.boxes is not None:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    # Get detection info
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = boxes.conf[i].cpu().numpy()
                    cls = int(boxes.cls[i].cpu().numpy())
                    
                    # Calculate center and area
                    x1, y1, x2, y2 = box
                    center = ((x1 + x2) / 2, (y1 + y2) / 2)
                    area = (x2 - x1) * (y2 - y1)
                    
                    # Create detection object
                    detection = ComponentDetection(
                        class_id=cls,
                        class_name=self.config.COMPONENT_CLASSES[cls],
                        confidence=float(conf),
                        bbox=tuple(box),
                        center=center,
                        area=area
                    )
                    
                    detections.append(detection)
        
        self.logger.info(f"Detected {len(detections)} components")
        return detections
    
    def detect_connections(self, detections: List[ComponentDetection]) -> List[Connection]:
        """Detect connections between components"""
        if not self.config.CONNECTION_DETECTION_ENABLED:
            return []
            
        connections = []
        
        for i, comp1 in enumerate(detections):
            for j, comp2 in enumerate(detections[i+1:], i+1):
                # Calculate distance between component centers
                dx = comp2.center[0] - comp1.center[0]
                dy = comp2.center[1] - comp1.center[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Check if components are close enough
                if distance <= self.config.CONNECTION_DISTANCE_THRESHOLD:
                    # Calculate angle
                    angle = math.degrees(math.atan2(dy, dx))
                    
                    # Determine connection type
                    connection_type = self._determine_connection_type(comp1, comp2, distance, angle)
                    
                    # Create connection object
                    connection = Connection(
                        component1=comp1,
                        component2=comp2,
                        distance=distance,
                        angle=angle,
                        connection_type=connection_type
                    )
                    
                    connections.append(connection)
        
        self.logger.info(f"Detected {len(connections)} connections")
        return connections
    
    def _determine_connection_type(self, comp1: ComponentDetection, comp2: ComponentDetection, 
                                 distance: float, angle: float) -> str:
        """Determine the type of connection between two components"""
        
        # Check if either component is a wire
        if comp1.class_name == "wire" or comp2.class_name == "wire":
            return "wire"
            
        # Check if components are very close (direct connection)
        if distance <= 20:
            return "direct"
            
        # Check if components are close and aligned
        if distance <= self.config.CONNECTION_DISTANCE_THRESHOLD:
            # Check if angle is within threshold (horizontal or vertical alignment)
            angle_normalized = abs(angle) % 90
            if angle_normalized <= self.config.CONNECTION_ANGLE_THRESHOLD or \
               angle_normalized >= (90 - self.config.CONNECTION_ANGLE_THRESHOLD):
                return "aligned"
                
        return "proximity"
    
    def visualize_detections(self, image_path: str, detections: List[ComponentDetection], 
                           connections: List[Connection], save_path: Optional[str] = None) -> str:
        """Visualize component detections and connections"""
        
        # Read image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        ax.imshow(image_rgb)
        
        # Color map for different component types
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.config.COMPONENT_CLASSES)))
        
        # Draw component detections
        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            color = colors[detection.class_id]
            
            # Draw bounding box
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add label
            label = f"{detection.class_name} ({detection.confidence:.2f})"
            ax.text(x1, y1-5, label, fontsize=8, color=color, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
            
            # Draw center point
            ax.plot(detection.center[0], detection.center[1], 'o', 
                   color=color, markersize=6)
        
        # Draw connections
        for connection in connections:
            comp1_center = connection.component1.center
            comp2_center = connection.component2.center
            
            # Choose color based on connection type
            if connection.connection_type == "wire":
                color = "red"
                linestyle = "-"
                linewidth = 3
            elif connection.connection_type == "direct":
                color = "green"
                linestyle = "-"
                linewidth = 2
            elif connection.connection_type == "aligned":
                color = "blue"
                linestyle = "--"
                linewidth = 2
            else:
                color = "gray"
                linestyle = ":"
                linewidth = 1
            
            # Draw connection line
            ax.plot([comp1_center[0], comp2_center[0]], 
                   [comp1_center[1], comp2_center[1]], 
                   color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.7)
        
        # Set title and remove axes
        ax.set_title(f"Snap Circuit Detection - {len(detections)} components, {len(connections)} connections")
        ax.axis('off')
        
        # Save or show
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.config.VISUALIZATIONS_DIR / f"detection_{timestamp}.png"
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualization saved to: {save_path}")
        return str(save_path)
    
    def analyze_circuit(self, image_path: str) -> Dict:
        """Complete circuit analysis - components and connections"""
        self.logger.info(f"Analyzing circuit: {image_path}")
        
        # Detect components
        detections = self.detect_components(image_path)
        
        # Detect connections
        connections = self.detect_connections(detections)
        
        # Create analysis result
        analysis = {
            "image_path": image_path,
            "total_components": len(detections),
            "total_connections": len(connections),
            "components": [
                {
                    "class_id": int(d.class_id),
                    "class_name": d.class_name,
                    "confidence": float(d.confidence),
                    "bbox": [float(x) for x in d.bbox],
                    "center": [float(x) for x in d.center],
                    "area": float(d.area)
                }
                for d in detections
            ],
            "connections": [
                {
                    "component1": c.component1.class_name,
                    "component2": c.component2.class_name,
                    "distance": float(c.distance),
                    "angle": float(c.angle),
                    "type": c.connection_type
                }
                for c in connections
            ],
            "component_summary": self._get_component_summary(detections),
            "connection_summary": self._get_connection_summary(connections)
        }
        
        return analysis
    
    def _get_component_summary(self, detections: List[ComponentDetection]) -> Dict:
        """Get summary statistics of detected components"""
        summary = {}
        for detection in detections:
            class_name = detection.class_name
            if class_name not in summary:
                summary[class_name] = {"count": 0, "avg_confidence": 0.0}
            summary[class_name]["count"] += 1
            summary[class_name]["avg_confidence"] += detection.confidence
        
        # Calculate averages
        for class_name in summary:
            count = summary[class_name]["count"]
            summary[class_name]["avg_confidence"] /= count
            
        return summary
    
    def _get_connection_summary(self, connections: List[Connection]) -> Dict:
        """Get summary statistics of detected connections"""
        summary = {
            "wire": 0,
            "direct": 0,
            "aligned": 0,
            "proximity": 0,
            "avg_distance": 0.0
        }
        
        total_distance = 0.0
        
        for connection in connections:
            summary[connection.connection_type] += 1
            total_distance += connection.distance
        
        if connections:
            summary["avg_distance"] = total_distance / len(connections)
            
        return summary

def main():
    """Test the detector on sample images"""
    detector = FreshStartDetector()
    
    # Test on a sample image
    test_image = "data/augmented_training/images/val/battery_holder_002.jpg"
    
    if Path(test_image).exists():
        # Analyze circuit
        analysis = detector.analyze_circuit(test_image)
        
        # Visualize results
        detections = [ComponentDetection(**comp) for comp in analysis["components"]]
        connections = detector.detect_connections(detections)
        
        visualization_path = detector.visualize_detections(
            test_image, detections, connections
        )
        
        # Print results
        print(f"Analysis complete!")
        print(f"Components detected: {analysis['total_components']}")
        print(f"Connections detected: {analysis['total_connections']}")
        print(f"Visualization saved to: {visualization_path}")
        
        # Save analysis to JSON
        analysis_path = detector.config.OUTPUT_DIR / "circuit_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"Analysis saved to: {analysis_path}")
        
    else:
        print(f"Test image not found: {test_image}")

if __name__ == "__main__":
    main() 