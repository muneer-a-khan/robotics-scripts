"""
Circuit graph builder using NetworkX to represent component connections.
"""

import networkx as nx
from typing import List, Dict, Set, Optional, Tuple, Any
import numpy as np
from collections import defaultdict

from config import CIRCUIT_CONFIG
from data_structures import (
    ComponentDetection, Connection, CircuitState, 
    ConnectionGraph, ComponentType
)


class CircuitGraphBuilder:
    """
    Builds and analyzes circuit graphs using NetworkX.
    """
    
    def __init__(self):
        """Initialize the graph builder."""
        self.config = CIRCUIT_CONFIG
        self.power_components = set(self.config["power_components"])
        self.output_components = set(self.config["output_components"])
        self.input_components = set(self.config["input_components"])
        self.passive_components = set(self.config["passive_components"])
        
    def build_graph(self, 
                   components: List[ComponentDetection],
                   connections: List[Connection],
                   timestamp: float,
                   frame_id: Optional[int] = None) -> ConnectionGraph:
        """
        Build a complete circuit graph from components and connections.
        
        Args:
            components: List of detected components
            connections: List of detected connections
            timestamp: Timestamp of the frame
            frame_id: Optional frame identifier
            
        Returns:
            Complete ConnectionGraph object
        """
        # Create NetworkX graph
        graph = self._create_networkx_graph(components, connections)
        
        # Analyze circuit state
        circuit_state = self._analyze_circuit_state(graph, components)
        
        # Create and return ConnectionGraph
        connection_graph = ConnectionGraph(
            components=components,
            edges=connections,
            state=circuit_state,
            timestamp=timestamp,
            frame_id=frame_id
        )
        
        return connection_graph
    
    def _create_networkx_graph(self, 
                              components: List[ComponentDetection],
                              connections: List[Connection]) -> nx.Graph:
        """
        Create a NetworkX graph from components and connections.
        
        Args:
            components: List of detected components
            connections: List of detected connections
            
        Returns:
            NetworkX graph representing the circuit
        """
        graph = nx.Graph()
        
        # Add component nodes
        for component in components:
            graph.add_node(
                component.id,
                label=component.label,
                component_type=component.component_type.value,
                bbox=component.bbox.to_list(),
                orientation=component.orientation,
                confidence=component.confidence,
                switch_state=component.switch_state.value if component.switch_state else None,
                connection_points=component.connection_points,
                metadata=component.metadata
            )
        
        # Add connection edges
        for connection in connections:
            graph.add_edge(
                connection.component_id_1,
                connection.component_id_2,
                connection_type=connection.connection_type,
                confidence=connection.confidence,
                path_points=connection.path_points
            )
        
        return graph
    
    def _analyze_circuit_state(self, 
                              graph: nx.Graph,
                              components: List[ComponentDetection]) -> CircuitState:
        """
        Analyze the circuit state from the graph.
        
        Args:
            graph: NetworkX graph of the circuit
            components: List of components
            
        Returns:
            CircuitState object with analysis results
        """
        # Find power sources and outputs
        power_sources = self._find_power_sources(graph, components)
        output_devices = self._find_output_devices(graph, components)
        
        # Check if circuit is closed (has complete paths)
        is_closed = self._check_circuit_closed(graph, power_sources, output_devices)
        
        # Determine power state
        power_on = self._check_power_on(graph, components)
        
        # Find active components
        active_components = self._find_active_components(graph, components, is_closed, power_on)
        
        # Find power flow path
        power_flow_path = self._find_power_flow_path(graph, power_sources, output_devices)
        
        # Estimate electrical properties (basic heuristics)
        voltage, current = self._estimate_electrical_properties(graph, components)
        
        return CircuitState(
            is_circuit_closed=is_closed,
            power_on=power_on,
            active_components=active_components,
            power_flow_path=power_flow_path,
            estimated_voltage=voltage,
            estimated_current=current
        )
    
    def _find_power_sources(self, 
                           graph: nx.Graph,
                           components: List[ComponentDetection]) -> List[str]:
        """
        Find power source components in the circuit.
        
        Args:
            graph: Circuit graph
            components: List of components
            
        Returns:
            List of power source component IDs
        """
        power_sources = []
        
        for component in components:
            if component.component_type.value in self.power_components:
                power_sources.append(component.id)
        
        return power_sources
    
    def _find_output_devices(self, 
                            graph: nx.Graph,
                            components: List[ComponentDetection]) -> List[str]:
        """
        Find output device components in the circuit.
        
        Args:
            graph: Circuit graph
            components: List of components
            
        Returns:
            List of output device component IDs
        """
        output_devices = []
        
        for component in components:
            if component.component_type.value in self.output_components:
                output_devices.append(component.id)
        
        return output_devices
    
    def _check_circuit_closed(self, 
                             graph: nx.Graph,
                             power_sources: List[str],
                             output_devices: List[str]) -> bool:
        """
        Check if the circuit has closed loops between power sources and outputs.
        
        Args:
            graph: Circuit graph
            power_sources: List of power source IDs
            output_devices: List of output device IDs
            
        Returns:
            True if circuit is closed, False otherwise
        """
        if not power_sources or not output_devices:
            return False
        
        # Check if there's a path from any power source to any output device
        for power_source in power_sources:
            for output_device in output_devices:
                if nx.has_path(graph, power_source, output_device):
                    # Check if there's a return path (complete circuit)
                    try:
                        # Look for cycles that include both power source and output
                        cycles = list(nx.simple_cycles(graph.to_directed()))
                        for cycle in cycles:
                            if power_source in cycle and output_device in cycle:
                                return True
                    except:
                        # Fallback: check if path exists (for simple circuits)
                        return True
        
        return False
    
    def _check_power_on(self, 
                       graph: nx.Graph,
                       components: List[ComponentDetection]) -> bool:
        """
        Check if the circuit is powered on based on switch states.
        
        Args:
            graph: Circuit graph
            components: List of components
            
        Returns:
            True if power is on, False otherwise
        """
        # Find all switches and buttons in the circuit
        switches = [comp for comp in components 
                   if comp.component_type.value in self.input_components]
        
        # If no switches, assume always on
        if not switches:
            return True
        
        # Check if any switches are in the "on" state
        for switch in switches:
            if (switch.switch_state and 
                switch.switch_state.value == "on"):
                return True
        
        # If we can't determine switch states, assume on for now
        return True
    
    def _find_active_components(self, 
                               graph: nx.Graph,
                               components: List[ComponentDetection],
                               is_closed: bool,
                               power_on: bool) -> List[str]:
        """
        Find components that should be active given the circuit state.
        
        Args:
            graph: Circuit graph
            components: List of components
            is_closed: Whether circuit is closed
            power_on: Whether power is on
            
        Returns:
            List of active component IDs
        """
        active_components = []
        
        if not (is_closed and power_on):
            return active_components
        
        # Find all output components in complete circuits
        power_sources = self._find_power_sources(graph, components)
        
        for component in components:
            if component.component_type.value in self.output_components:
                # Check if this output is connected to a power source
                for power_source in power_sources:
                    if nx.has_path(graph, power_source, component.id):
                        active_components.append(component.id)
                        break
        
        return active_components
    
    def _find_power_flow_path(self, 
                             graph: nx.Graph,
                             power_sources: List[str],
                             output_devices: List[str]) -> List[str]:
        """
        Find the path of power flow through the circuit.
        
        Args:
            graph: Circuit graph
            power_sources: List of power source IDs
            output_devices: List of output device IDs
            
        Returns:
            List of component IDs in the power flow path
        """
        if not power_sources or not output_devices:
            return []
        
        # Find shortest path from first power source to first output device
        try:
            path = nx.shortest_path(graph, power_sources[0], output_devices[0])
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []
    
    def _estimate_electrical_properties(self, 
                                      graph: nx.Graph,
                                      components: List[ComponentDetection]) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate voltage and current in the circuit using basic heuristics.
        
        Args:
            graph: Circuit graph
            components: List of components
            
        Returns:
            Tuple of (estimated_voltage, estimated_current)
        """
        # Basic heuristics for Snap Circuits
        voltage = None
        current = None
        
        # Count battery holders (typical Snap Circuit batteries are 1.5V AA)
        battery_count = sum(1 for comp in components 
                           if comp.component_type == ComponentType.BATTERY_HOLDER)
        
        if battery_count > 0:
            voltage = battery_count * 1.5  # Assuming AA batteries in series
            
            # Estimate current based on load (very basic)
            output_count = sum(1 for comp in components
                             if comp.component_type.value in self.output_components)
            if output_count > 0:
                # Rough estimate: 20mA per output device
                current = output_count * 0.02
        
        return voltage, current
    
    def analyze_graph_topology(self, graph: nx.Graph) -> Dict[str, Any]:
        """
        Analyze the topology of the circuit graph.
        
        Args:
            graph: Circuit graph to analyze
            
        Returns:
            Dictionary of topology metrics
        """
        if len(graph.nodes) == 0:
            return {}
        
        try:
            analysis = {
                "num_components": len(graph.nodes),
                "num_connections": len(graph.edges),
                "num_connected_components": nx.number_connected_components(graph),
                "is_connected": nx.is_connected(graph),
                "graph_density": nx.density(graph),
                "average_clustering": nx.average_clustering(graph),
                "diameter": nx.diameter(graph) if nx.is_connected(graph) else None,
                "average_shortest_path_length": (
                    nx.average_shortest_path_length(graph) 
                    if nx.is_connected(graph) else None
                )
            }
            
            # Node degree statistics
            degrees = [graph.degree(node) for node in graph.nodes]
            if degrees:
                analysis.update({
                    "average_degree": np.mean(degrees),
                    "max_degree": max(degrees),
                    "min_degree": min(degrees)
                })
            
            return analysis
            
        except Exception as e:
            print(f"Error analyzing graph topology: {e}")
            return {"error": str(e)}
    
    def find_potential_errors(self, 
                             graph: nx.Graph,
                             components: List[ComponentDetection]) -> List[Dict[str, Any]]:
        """
        Find potential errors or issues in the circuit.
        
        Args:
            graph: Circuit graph
            components: List of components
            
        Returns:
            List of potential errors/issues
        """
        errors = []
        
        # Check for isolated components
        isolated_nodes = list(nx.isolates(graph))
        if isolated_nodes:
            errors.append({
                "type": "isolated_components",
                "message": f"Found {len(isolated_nodes)} isolated components",
                "components": isolated_nodes
            })
        
        # Check for missing power sources
        power_sources = self._find_power_sources(graph, components)
        if not power_sources:
            errors.append({
                "type": "no_power_source",
                "message": "No power source detected in circuit",
                "components": []
            })
        
        # Check for missing outputs
        outputs = self._find_output_devices(graph, components)
        if not outputs:
            errors.append({
                "type": "no_output_device",
                "message": "No output device detected in circuit", 
                "components": []
            })
        
        # Check for short circuits (direct connection between power terminals)
        if len(power_sources) >= 2:
            for i, ps1 in enumerate(power_sources):
                for ps2 in power_sources[i+1:]:
                    if nx.has_path(graph, ps1, ps2):
                        try:
                            path = nx.shortest_path(graph, ps1, ps2)
                            if len(path) <= 3:  # Direct or very short connection
                                errors.append({
                                    "type": "potential_short_circuit",
                                    "message": f"Potential short circuit between {ps1} and {ps2}",
                                    "components": path
                                })
                        except:
                            pass
        
        return errors
    
    def suggest_improvements(self, 
                           graph: nx.Graph,
                           components: List[ComponentDetection]) -> List[str]:
        """
        Suggest improvements to the circuit.
        
        Args:
            graph: Circuit graph
            components: List of components
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Check circuit completeness
        power_sources = self._find_power_sources(graph, components)
        outputs = self._find_output_devices(graph, components)
        
        if power_sources and outputs:
            is_complete = any(
                nx.has_path(graph, ps, out) 
                for ps in power_sources 
                for out in outputs
            )
            
            if not is_complete:
                suggestions.append(
                    "Circuit is not complete. Try connecting the power source to an output device."
                )
        
        # Check for switches
        switches = [comp for comp in components 
                   if comp.component_type.value in self.input_components]
        if not switches and power_sources and outputs:
            suggestions.append(
                "Consider adding a switch to control when the circuit is active."
            )
        
        # Check component count
        if len(components) < self.config["min_circuit_length"]:
            suggestions.append(
                f"Circuit seems simple. Try adding more components for complexity."
            )
        
        return suggestions


def test_graph_builder():
    """Test function for the graph builder."""
    builder = CircuitGraphBuilder()
    
    # Create mock components
    from data_structures import ComponentType, BoundingBox
    
    components = [
        ComponentDetection(
            id="battery-1",
            label="battery_holder",
            bbox=BoundingBox(0, 0, 50, 50),
            orientation=0,
            confidence=0.9,
            component_type=ComponentType.BATTERY_HOLDER
        ),
        ComponentDetection(
            id="led-1",
            label="led", 
            bbox=BoundingBox(100, 100, 150, 150),
            orientation=0,
            confidence=0.8,
            component_type=ComponentType.LED
        ),
        ComponentDetection(
            id="switch-1",
            label="switch",
            bbox=BoundingBox(50, 50, 100, 100),
            orientation=0,
            confidence=0.85,
            component_type=ComponentType.SWITCH
        )
    ]
    
    # Create mock connections
    connections = [
        Connection("battery-1", "switch-1", "wire"),
        Connection("switch-1", "led-1", "wire")
    ]
    
    # Build graph
    import time
    graph = builder.build_graph(components, connections, time.time())
    
    print(f"Built graph with {len(graph.components)} components")
    print(f"Circuit closed: {graph.state.is_circuit_closed}")
    print(f"Power on: {graph.state.power_on}")
    print(f"Active components: {graph.state.active_components}")
    
    # Test topology analysis
    nx_graph = builder._create_networkx_graph(components, connections)
    topology = builder.analyze_graph_topology(nx_graph)
    print(f"Topology: {topology}")
    
    # Test error detection
    errors = builder.find_potential_errors(nx_graph, components)
    print(f"Errors: {errors}")


if __name__ == "__main__":
    test_graph_builder() 