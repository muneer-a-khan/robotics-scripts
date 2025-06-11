"""
Data structures for the Snap Circuit computer vision system.
Implements the interfaces specified in the requirements.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum
import json
import numpy as np


class ComponentType(Enum):
    """Enumeration of all detectable component types."""
    WIRE = "wire"
    SWITCH = "switch"
    BUTTON = "button"
    BATTERY_HOLDER = "battery_holder"
    LED = "led"
    SPEAKER = "speaker"
    MUSIC_CIRCUIT = "music_circuit"
    MOTOR = "motor"
    RESISTOR = "resistor"
    CONNECTION_NODE = "connection_node"
    LAMP = "lamp"
    FAN = "fan"
    BUZZER = "buzzer"
    PHOTORESISTOR = "photoresistor"
    MICROPHONE = "microphone"
    ALARM = "alarm"


class SwitchState(Enum):
    """Switch states for components that have on/off states."""
    ON = "on"
    OFF = "off"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Represents a bounding box with coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get the center point of the bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def width(self) -> float:
        """Get the width of the bounding box."""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """Get the height of the bounding box."""
        return self.y2 - self.y1
    
    @property
    def area(self) -> float:
        """Get the area of the bounding box."""
        return self.width * self.height
    
    def to_list(self) -> List[float]:
        """Convert to [x1, y1, x2, y2] format."""
        return [
            float(self.x1) if hasattr(self.x1, 'item') else float(self.x1),
            float(self.y1) if hasattr(self.y1, 'item') else float(self.y1),
            float(self.x2) if hasattr(self.x2, 'item') else float(self.x2),
            float(self.y2) if hasattr(self.y2, 'item') else float(self.y2)
        ]


@dataclass
class ComponentDetection:
    """
    Represents a detected component in the circuit.
    Matches the ComponentDetection interface from requirements.
    """
    id: str
    label: str
    bbox: BoundingBox
    orientation: float  # degrees (0-360)
    confidence: float  # 0.0 to 1.0
    component_type: ComponentType
    switch_state: Optional[SwitchState] = None
    connection_points: List[Tuple[float, float]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate data after initialization."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be between 0 and 1, got {self.confidence}")
        if not 0 <= self.orientation <= 360:
            self.orientation = self.orientation % 360
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": str(self.id),
            "label": str(self.label),
            "bbox": self.bbox.to_list(),
            "orientation": float(self.orientation) if hasattr(self.orientation, 'item') else float(self.orientation),
            "confidence": float(self.confidence) if hasattr(self.confidence, 'item') else float(self.confidence),
            "component_type": self.component_type.value,
            "switch_state": self.switch_state.value if self.switch_state else None,
            "connection_points": [[float(x) if hasattr(x, 'item') else float(x), 
                                   float(y) if hasattr(y, 'item') else float(y)] 
                                  for x, y in self.connection_points],
            "metadata": self.metadata
        }


@dataclass
class Connection:
    """Represents a connection between two components."""
    component_id_1: str
    component_id_2: str
    connection_type: str = "wire"  # wire, direct, etc.
    confidence: float = 1.0
    path_points: List[Tuple[float, float]] = field(default_factory=list)
    
    def to_tuple(self) -> Tuple[str, str]:
        """Convert to tuple format for NetworkX."""
        return (self.component_id_1, self.component_id_2)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "component_1": str(self.component_id_1),
            "component_2": str(self.component_id_2),
            "connection_type": str(self.connection_type),
            "confidence": float(self.confidence) if hasattr(self.confidence, 'item') else float(self.confidence),
            "path_points": [[float(x) if hasattr(x, 'item') else float(x), 
                             float(y) if hasattr(y, 'item') else float(y)] 
                            for x, y in self.path_points]
        }


@dataclass
class CircuitState:
    """Represents the overall state of the circuit."""
    is_circuit_closed: bool
    power_on: bool
    active_components: List[str] = field(default_factory=list)
    power_flow_path: List[str] = field(default_factory=list)
    estimated_voltage: Optional[float] = None
    estimated_current: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "is_circuit_closed": self.is_circuit_closed,
            "power_on": self.power_on,
            "active_components": self.active_components,
            "power_flow_path": self.power_flow_path,
            "estimated_voltage": self.estimated_voltage,
            "estimated_current": self.estimated_current
        }


@dataclass
class ConnectionGraph:
    """
    Main data structure representing the complete circuit analysis.
    Matches the ConnectionGraph interface from requirements.
    """
    components: List[ComponentDetection]
    edges: List[Connection]
    state: CircuitState
    timestamp: float
    frame_id: Optional[int] = None
    
    def get_component_by_id(self, component_id: str) -> Optional[ComponentDetection]:
        """Get a component by its ID."""
        for component in self.components:
            if component.id == component_id:
                return component
        return None
    
    def get_connections_for_component(self, component_id: str) -> List[Connection]:
        """Get all connections involving a specific component."""
        connections = []
        for edge in self.edges:
            if edge.component_id_1 == component_id or edge.component_id_2 == component_id:
                connections.append(edge)
        return connections
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "components": [comp.to_dict() for comp in self.components],
            "edges": [edge.to_dict() for edge in self.edges],
            "state": self.state.to_dict(),
            "timestamp": float(self.timestamp) if hasattr(self.timestamp, 'item') else float(self.timestamp),
            "frame_id": int(self.frame_id) if self.frame_id is not None else None
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class DetectionResult:
    """Result from the complete detection pipeline for a single frame."""
    connection_graph: ConnectionGraph
    raw_detections: List[Dict[str, Any]]
    processing_time: float
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "connection_graph": self.connection_graph.to_dict(),
            "raw_detections": self.raw_detections,
            "processing_time": float(self.processing_time) if hasattr(self.processing_time, 'item') else float(self.processing_time),
            "error_message": str(self.error_message) if self.error_message else None
        }


# Utility functions for working with the data structures

def create_component_id(component_type: ComponentType, index: int) -> str:
    """Create a standardized component ID."""
    return f"{component_type.value}-{index}"


def bbox_from_yolo(yolo_box: np.ndarray, img_width: int, img_height: int) -> BoundingBox:
    """Convert YOLO detection box to BoundingBox object."""
    x1, y1, x2, y2 = yolo_box
    return BoundingBox(
        x1=float(x1),
        y1=float(y1), 
        x2=float(x2),
        y2=float(y2)
    )


def calculate_orientation(bbox: BoundingBox) -> float:
    """
    Calculate orientation based on bounding box aspect ratio.
    This is a simple heuristic - real orientation detection would use 
    more sophisticated methods.
    """
    if bbox.width > bbox.height:
        return 0.0  # Horizontal
    else:
        return 90.0  # Vertical 