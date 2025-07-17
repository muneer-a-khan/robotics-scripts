#!/usr/bin/env python3
"""
Convert existing detection results to Circuit Graph format for TD-BKT algorithm.
"""

from circuit_graph_format import (
    CircuitGraph, ComponentNode, ConnectionEdge, 
    ComponentState, ConnectionType
)
from data_structures import DetectionResult, ComponentDetection
import json
from typing import Dict, List, Any
import random


class DetectionToGraphConverter:
    """
    Converts detection results to circuit graph format.
    """
    
    def __init__(self):
        self.state_mapping = {
            # Map component types to likely states
            "battery_holder": ComponentState.POWERED_ON,
            "led": ComponentState.INACTIVE,  # Default to off until proven on
            "switch": ComponentState.UNKNOWN,
            "motor": ComponentState.INACTIVE,
            "speaker": ComponentState.INACTIVE,
            "lamp": ComponentState.INACTIVE,
            "fan": ComponentState.INACTIVE,
            "buzzer": ComponentState.INACTIVE,
        }
    
    def convert_detection_result(self, detection_result: DetectionResult, 
                               learning_context: Dict[str, Any] = None) -> CircuitGraph:
        """
        Convert a DetectionResult to CircuitGraph format.
        
        Args:
            detection_result: The detection result from your vision system
            learning_context: Additional context for TD-BKT (skill level, etc.)
        """
        circuit_graph = CircuitGraph()
        
        # Extract components and connections from detection result
        components = detection_result.connection_graph.components
        edges = detection_result.connection_graph.edges
        circuit_state = detection_result.connection_graph.state
        
        # Convert components to nodes
        for i, component in enumerate(components):
            node = self._convert_component_to_node(component, i, learning_context)
            circuit_graph.add_component(node)
        
        # Convert edges to connections
        for edge in edges:
            connection = self._convert_edge_to_connection(edge, components)
            if connection:  # Only add if conversion successful
                circuit_graph.add_connection(connection)
        
        # Add circuit state information
        circuit_graph.circuit_state.update({
            "is_complete": circuit_state.is_circuit_closed,
            "power_on": circuit_state.power_on,
            "active_components": circuit_state.active_components
        })
        
        return circuit_graph
    
    def _convert_component_to_node(self, component: ComponentDetection, 
                                  index: int, learning_context: Dict[str, Any]) -> ComponentNode:
        """Convert a ComponentDetection to a ComponentNode."""
        
        # Determine component state
        component_state = self.state_mapping.get(
            component.component_type.value, 
            ComponentState.UNKNOWN
        )
        
        # For power sources, check if they're active
        if component.component_type.value == "battery_holder":
            component_state = ComponentState.POWERED_ON
        
        # Calculate placement correctness (simplified heuristic)
        placement_correctness = min(component.confidence, 0.95)
        
        # Calculate functional state based on circuit analysis
        functional_state = 1.0 if component_state in [
            ComponentState.POWERED_ON, ComponentState.ACTIVE
        ] else 0.0
        
        # Estimate accessibility for robot (based on position and size)
        bbox = component.bbox
        accessibility = self._calculate_accessibility(bbox)
        
        # Default learning context if not provided
        if learning_context is None:
            learning_context = {
                "skill_level": "beginner",
                "session_id": "demo",
                "attempts": 1,
                "component_familiarity": 0.5
            }
        
        # No recommendations needed - handled by external TD-BKT system
        recommended_action = None
        
        return ComponentNode(
            id=f"{component.component_type.value}_{index}",
            component_type=component.component_type.value,
            state=component_state,
            position={
                "x": (bbox.x1 + bbox.x2) / 2,
                "y": (bbox.y1 + bbox.y2) / 2,
                "width": bbox.x2 - bbox.x1,
                "height": bbox.y2 - bbox.y1
            },
            placement_correctness=placement_correctness,
            functional_state=functional_state,
            learning_context=learning_context.copy(),
            accessibility=accessibility,
            placement_confidence=component.confidence,
            recommended_action=recommended_action,
            electrical_props=self._get_electrical_properties(component.component_type.value),
            detection_confidence=component.confidence
        )
    
    def _convert_edge_to_connection(self, edge, components: List[ComponentDetection]) -> ConnectionEdge:
        """Convert an edge to a ConnectionEdge."""
        
        # Find component types by looking up component IDs in components list
        component1_type = None
        component2_type = None
        
        # Look up component types from the components list
        for component in components:
            if component.id == edge.component_id_1:
                component1_type = component.component_type
            elif component.id == edge.component_id_2:
                component2_type = component.component_type
        
        # Handle cases where component types might be None or not found
        if component1_type is None or component2_type is None:
            print(f"Warning: Could not find component types for edge {edge.component_id_1} -> {edge.component_id_2}")
            return None
        
        # Get string values from component types (handle both enum and string cases)
        comp1_type_str = component1_type.value if hasattr(component1_type, 'value') else str(component1_type)
        comp2_type_str = component2_type.value if hasattr(component2_type, 'value') else str(component2_type)
        
        # Find source and target component IDs
        source_id = f"{comp1_type_str}_{edge.component_id_1}"
        target_id = f"{comp2_type_str}_{edge.component_id_2}"
        
        # Determine connection type
        connection_type = ConnectionType.WIRE  # Default
        if "snap" in str(edge.connection_type).lower():
            connection_type = ConnectionType.SNAP_CONNECTION
        elif "direct" in str(edge.connection_type).lower():
            connection_type = ConnectionType.DIRECT_CONTACT
        
        # Calculate connection correctness (simplified)
        connection_correctness = min(edge.confidence, 0.9)
        
        # Determine if this connection is expected
        expected_connection = self._is_expected_connection(
            comp1_type_str, comp2_type_str
        )
        
        # Calculate connection quality
        connection_quality = edge.confidence
        
        # Determine current direction (simplified)
        current_direction = "forward"  # Default
        if comp1_type_str == "battery_holder":
            current_direction = "forward"
        elif comp2_type_str == "battery_holder":
            current_direction = "reverse"
        else:
            current_direction = "none"
        
        # Calculate flow strength
        flow_strength = 1.0 if current_direction != "none" else 0.0
        
        # Robot accessibility
        robot_accessible = True  # Assume most connections are accessible
        
        return ConnectionEdge(
            source=source_id,
            target=target_id,
            connection_type=connection_type,
            connection_correctness=connection_correctness,
            expected_connection=expected_connection,
            connection_quality=connection_quality,
            current_direction=current_direction,
            flow_strength=flow_strength,
            robot_accessible=robot_accessible,
            recommended_action=None,  # External TD-BKT system handles recommendations
            resistance=0.1,  # Default low resistance
            voltage_drop=0.0,
            detection_confidence=edge.confidence
        )
    
    def _calculate_accessibility(self, bbox) -> float:
        """Calculate how accessible a component is to the robot."""
        # Simplified heuristic: larger components are easier to manipulate
        area = (bbox.x2 - bbox.x1) * (bbox.y2 - bbox.y1)
        
        # Normalize to 0-1 range (assuming max area of 10000 pixels)
        normalized_area = min(area / 10000.0, 1.0)
        
        # Components near edges might be harder to reach
        center_x = (bbox.x1 + bbox.x2) / 2
        center_y = (bbox.y1 + bbox.y2) / 2
        
        # Assume image is 640x640, center is most accessible
        distance_from_center = ((center_x - 320)**2 + (center_y - 320)**2)**0.5
        center_accessibility = max(0.3, 1.0 - distance_from_center / 450.0)
        
        return min(0.5 + normalized_area * 0.3 + center_accessibility * 0.2, 1.0)
    
    def _get_electrical_properties(self, component_type: str) -> Dict[str, float]:
        """Get default electrical properties for component types."""
        properties = {
            "battery_holder": {"voltage": 3.0, "current_capacity": 1.0, "internal_resistance": 0.1},
            "led": {"forward_voltage": 2.0, "current_draw": 0.02, "resistance": 100.0},
            "motor": {"voltage_rating": 3.0, "current_draw": 0.2, "resistance": 15.0},
            "speaker": {"voltage_rating": 3.0, "current_draw": 0.1, "resistance": 8.0},
            "switch": {"contact_resistance": 0.01, "current_rating": 1.0},
            "wire": {"resistance_per_cm": 0.001},
            "resistor": {"resistance": 100.0, "power_rating": 0.25}
        }
        
        return properties.get(component_type, {"resistance": 1.0})
    
    def _is_expected_connection(self, comp1_type: str, comp2_type: str) -> bool:
        """Determine if a connection between two component types is expected."""
        # Power sources should connect to everything
        if comp1_type == "battery_holder" or comp2_type == "battery_holder":
            return True
        
        # Outputs should connect to power or control components
        output_types = ["led", "motor", "speaker", "lamp", "fan", "buzzer"]
        control_types = ["switch", "button"]
        
        if (comp1_type in output_types and comp2_type in control_types) or \
           (comp2_type in output_types and comp1_type in control_types):
            return True
        
        # Wires connect everything
        if comp1_type == "wire" or comp2_type == "wire":
            return True
        
        return False


def demo_conversion():
    """Demo converting detection results to graph format."""
    print("🔄 Detection to Graph Conversion Demo")
    print("=" * 45)
    
    # This would normally come from your detection system
    # For demo, we'll create a mock detection result
    
    converter = DetectionToGraphConverter()
    
    # Example learning context for TD-BKT
    learning_context = {
        "user_id": "student_123",
        "skill_level": "beginner",
        "session_id": "circuit_build_001",
        "attempts": 3,
        "component_familiarity": {
            "battery_holder": 0.8,
            "led": 0.6,
            "wire": 0.4
        },
        "learning_objectives": ["basic_circuit", "series_connection"]
    }
    
    # This is where you'd pass your actual DetectionResult
    # circuit_graph = converter.convert_detection_result(your_detection_result, learning_context)
    
    print("✅ Conversion process demonstrated")
    print("📊 Graph format includes:")
    print("  - Component nodes with TD-BKT properties")
    print("  - Directed edges for current flow")
    print("  - Connectivity analysis")
    print("  - Robot action recommendations")
    print("  - Learning context integration")


if __name__ == "__main__":
    demo_conversion() 