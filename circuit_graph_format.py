#!/usr/bin/env python3
"""
Circuit Graph Format for TD-BKT Algorithm and Robot Assistance.
Converts detection results into a directed graph suitable for component recommendation.
"""

import networkx as nx
import json
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import time


class ComponentState(Enum):
    """Component operational states for TD-BKT tracking."""
    UNKNOWN = "unknown"
    POWERED_ON = "powered_on"
    POWERED_OFF = "powered_off"
    ACTIVE = "active"
    INACTIVE = "inactive"
    BROKEN = "broken"
    MISSING = "missing"


class ConnectionType(Enum):
    """Types of connections between components."""
    WIRE = "wire"
    DIRECT_CONTACT = "direct_contact"
    SNAP_CONNECTION = "snap_connection"
    VIRTUAL = "virtual"  # For potential connections


@dataclass
class ComponentNode:
    """Node representing a circuit component."""
    id: str
    component_type: str
    state: ComponentState
    position: Dict[str, float]  # {"x": 100, "y": 200}
    
    # For TD-BKT algorithm
    placement_correctness: float  # 0.0-1.0 how correctly placed
    functional_state: float       # 0.0-1.0 how well it's working
    learning_context: Dict[str, Any]  # Context for knowledge tracing
    
    # For robot assistance
    accessibility: float          # 0.0-1.0 how easy for robot to reach
    placement_confidence: float   # 0.0-1.0 detection confidence
    recommended_action: Optional[str]  # "place", "adjust", "remove"
    
    # Electrical properties (secondary importance)
    electrical_props: Dict[str, float] = None
    
    # Metadata
    timestamp: float = None
    detection_confidence: float = 1.0


@dataclass
class ConnectionEdge:
    """Edge representing a connection between components."""
    source: str
    target: str
    connection_type: ConnectionType
    
    # For TD-BKT algorithm
    connection_correctness: float  # 0.0-1.0 how correct this connection is
    expected_connection: bool      # Is this connection supposed to exist?
    connection_quality: float     # 0.0-1.0 physical connection quality
    
    # For current flow (directed)
    current_direction: str         # "forward", "reverse", "none"
    flow_strength: float          # 0.0-1.0 relative current flow
    
    # For robot assistance
    robot_accessible: bool        # Can robot manipulate this connection?
    recommended_action: Optional[str]  # "connect", "disconnect", "adjust"
    
    # Electrical properties
    resistance: float = 0.0
    voltage_drop: float = 0.0
    
    # Metadata
    detection_confidence: float = 1.0


class CircuitGraph:
    """
    Directed graph representation of the circuit for TD-BKT and robot assistance.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed for current flow
        self.timestamp = time.time()
        self.circuit_state = {
            "is_complete": False,
            "power_sources": [],
            "active_paths": [],
            "recommended_components": [],
            "robot_actions": []
        }
    
    def add_component(self, component: ComponentNode):
        """Add a component node to the graph."""
        if component.timestamp is None:
            component.timestamp = self.timestamp
            
        # Add node with all properties
        self.graph.add_node(
            component.id,
            **asdict(component)
        )
    
    def add_connection(self, connection: ConnectionEdge):
        """Add a connection edge to the graph."""
        self.graph.add_edge(
            connection.source,
            connection.target,
            **asdict(connection)
        )
    
    def analyze_connectivity(self) -> Dict[str, Any]:
        """Analyze circuit connectivity for TD-BKT algorithm."""
        analysis = {
            "connected_components": list(nx.weakly_connected_components(self.graph)),
            "strongly_connected": list(nx.strongly_connected_components(self.graph)),
            "power_paths": self._find_power_paths(),
            "isolated_components": self._find_isolated_components(),
            "potential_connections": self._find_potential_connections()
        }
        
        # Update circuit state
        self.circuit_state.update({
            "is_complete": len(analysis["power_paths"]) > 0,
            "power_sources": self._get_power_sources(),
            "active_paths": analysis["power_paths"]
        })
        
        return analysis
    
    def _find_power_paths(self) -> List[List[str]]:
        """Find all paths from power sources to outputs."""
        power_sources = self._get_power_sources()
        output_components = self._get_output_components()
        
        paths = []
        for source in power_sources:
            for output in output_components:
                try:
                    path = nx.shortest_path(self.graph, source, output)
                    paths.append(path)
                except nx.NetworkXNoPath:
                    continue
        
        return paths
    
    def _get_power_sources(self) -> List[str]:
        """Get all power source components."""
        power_sources = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('component_type') in ['battery_holder'] and \
               data.get('state') in [ComponentState.POWERED_ON, ComponentState.ACTIVE]:
                power_sources.append(node_id)
        return power_sources
    
    def _get_output_components(self) -> List[str]:
        """Get all output components (LEDs, motors, etc.)."""
        output_types = ['led', 'motor', 'speaker', 'lamp', 'fan', 'buzzer']
        outputs = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('component_type') in output_types:
                outputs.append(node_id)
        return outputs
    
    def _find_isolated_components(self) -> List[str]:
        """Find components with no connections."""
        isolated = []
        for node_id in self.graph.nodes():
            if self.graph.degree(node_id) == 0:
                isolated.append(node_id)
        return isolated
    
    def _find_potential_connections(self) -> List[Dict[str, Any]]:
        """Find potential connections for robot recommendations."""
        # This would analyze spatial proximity and circuit requirements
        # For now, simplified version
        potential = []
        isolated = self._find_isolated_components()
        
        for component in isolated:
            component_data = self.graph.nodes[component]
            # Find nearby components that could connect
            nearby = self._find_nearby_components(component_data['position'])
            for nearby_id in nearby:
                if nearby_id != component:
                    potential.append({
                        "from": component,
                        "to": nearby_id,
                        "connection_type": "wire",
                        "confidence": 0.8,
                        "robot_feasible": True
                    })
        
        return potential
    
    def _find_nearby_components(self, position: Dict[str, float], threshold: float = 100.0) -> List[str]:
        """Find components within spatial threshold."""
        nearby = []
        for node_id, data in self.graph.nodes(data=True):
            other_pos = data.get('position', {})
            if other_pos:
                distance = ((position['x'] - other_pos['x'])**2 + 
                           (position['y'] - other_pos['y'])**2)**0.5
                if distance <= threshold:
                    nearby.append(node_id)
        return nearby
    
    def get_circuit_analysis(self) -> Dict[str, Any]:
        """Get basic circuit analysis without recommendations."""
        connectivity = self.analyze_connectivity()
        
        return {
            "connectivity": connectivity,
            "circuit_complete": self.circuit_state["is_complete"],
            "power_sources": self.circuit_state["power_sources"],
            "active_paths": self.circuit_state["active_paths"]
        }
    
    def to_json(self) -> str:
        """Export graph to JSON format for downstream algorithms."""
        # Convert NetworkX graph to JSON-serializable format
        json_data = {
            "graph": {
                "directed": True,
                "timestamp": self.timestamp,
                "nodes": [],
                "edges": [],
                "circuit_analysis": self.get_circuit_analysis()
            }
        }
        
        # Add nodes
        for node_id, data in self.graph.nodes(data=True):
            node_data = dict(data)
            node_data["id"] = node_id
            # Convert enum to string
            if isinstance(node_data.get("state"), ComponentState):
                node_data["state"] = node_data["state"].value
            json_data["graph"]["nodes"].append(node_data)
        
        # Add edges
        for source, target, data in self.graph.edges(data=True):
            edge_data = dict(data)
            edge_data["source"] = source
            edge_data["target"] = target
            # Convert enum to string
            if isinstance(edge_data.get("connection_type"), ConnectionType):
                edge_data["connection_type"] = edge_data["connection_type"].value
            json_data["graph"]["edges"].append(edge_data)
        
        return json.dumps(json_data, indent=2, default=str)
    
    def to_networkx(self) -> nx.DiGraph:
        """Return the underlying NetworkX graph for advanced algorithms."""
        return self.graph
    
    def export_for_robot(self) -> Dict[str, Any]:
        """Export specific data needed for robot control."""
        robot_data = {
            "timestamp": self.timestamp,
            "actionable_components": [],
            "spatial_layout": {}
        }
        
        # Get components robot can act on
        for node_id, data in self.graph.nodes(data=True):
            if data.get('accessibility', 0) > 0.5:  # Robot can reach
                robot_data["actionable_components"].append({
                    "id": node_id,
                    "type": data.get('component_type'),
                    "position": data.get('position'),
                    "accessibility": data.get('accessibility')
                })
                
                # Add to spatial layout
                robot_data["spatial_layout"][node_id] = data.get('position')
        
        return robot_data


def demo_circuit_graph():
    """Demonstrate the circuit graph format."""
    print("🔧 Circuit Graph Format Demo")
    print("=" * 40)
    
    # Create graph
    circuit = CircuitGraph()
    
    # Add components
    battery = ComponentNode(
        id="battery_1",
        component_type="battery_holder",
        state=ComponentState.POWERED_ON,
        position={"x": 100, "y": 200},
        placement_correctness=0.95,
        functional_state=1.0,
        learning_context={"skill_level": "beginner", "attempts": 1},
        accessibility=0.8,
        placement_confidence=0.95,
        recommended_action=None,
        electrical_props={"voltage": 3.0, "current_capacity": 1.0}
    )
    
    led = ComponentNode(
        id="led_1",
        component_type="led",
        state=ComponentState.INACTIVE,
        position={"x": 300, "y": 200},
        placement_correctness=0.9,
        functional_state=0.0,  # Not lit up
        learning_context={"skill_level": "beginner", "attempts": 2},
        accessibility=0.9,
        placement_confidence=0.87,
        recommended_action="connect",
        electrical_props={"forward_voltage": 2.0, "current_draw": 0.02}
    )
    
    circuit.add_component(battery)
    circuit.add_component(led)
    
    # Add connection (or lack thereof)
    # For demo, let's say they're not connected yet
    
    # Generate analysis and recommendations
    print("📊 Graph Analysis:")
    connectivity = circuit.analyze_connectivity()
    print(f"Connected components: {len(connectivity['connected_components'])}")
    print(f"Isolated components: {connectivity['isolated_components']}")
    
    print("\n🤖 TD-BKT Recommendations:")
    recommendations = circuit.generate_td_bkt_recommendations()
    for rec in recommendations:
        print(f"- {rec['type']}: {rec['action']} (confidence: {rec['confidence']})")
    
    print("\n🔧 Robot Export:")
    robot_data = circuit.export_for_robot()
    print(f"Actionable components: {len(robot_data['actionable_components'])}")
    print(f"Priority actions: {len(robot_data['priority_actions'])}")
    
    return circuit


if __name__ == "__main__":
    demo_circuit_graph() 