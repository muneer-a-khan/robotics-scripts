#!/usr/bin/env python3
"""
Test the graph output format for TD-BKT algorithm.
"""

from circuit_graph_format import (
    CircuitGraph, ComponentNode, ConnectionEdge, 
    ComponentState, ConnectionType
)
import json

def create_sample_circuit():
    """Create a sample circuit with battery, LED, and connections."""
    circuit = CircuitGraph()
    
    # Add battery (power source)
    battery = ComponentNode(
        id="battery_1",
        component_type="battery_holder",
        state=ComponentState.POWERED_ON,
        position={"x": 100, "y": 200},
        placement_correctness=0.95,
        functional_state=1.0,
        learning_context={
            "user_id": "student_123",
            "skill_level": "beginner", 
            "attempts": 1,
            "component_familiarity": 0.8
        },
        accessibility=0.8,
        placement_confidence=0.95,
        recommended_action=None,
        electrical_props={"voltage": 3.0, "current_capacity": 1.0}
    )
    
    # Add LED (output component)
    led = ComponentNode(
        id="led_1", 
        component_type="led",
        state=ComponentState.INACTIVE,
        position={"x": 300, "y": 200},
        placement_correctness=0.9,
        functional_state=0.0,
        learning_context={
            "user_id": "student_123",
            "skill_level": "beginner",
            "attempts": 2,
            "component_familiarity": 0.6
        },
        accessibility=0.9,
        placement_confidence=0.87,
        recommended_action="connect",
        electrical_props={"forward_voltage": 2.0, "current_draw": 0.02}
    )
    
    # Add switch (control component)
    switch = ComponentNode(
        id="switch_1",
        component_type="switch", 
        state=ComponentState.UNKNOWN,
        position={"x": 200, "y": 150},
        placement_correctness=0.85,
        functional_state=0.5,
        learning_context={
            "user_id": "student_123",
            "skill_level": "beginner",
            "attempts": 3,
            "component_familiarity": 0.4
        },
        accessibility=0.7,
        placement_confidence=0.82,
        recommended_action="test",
        electrical_props={"contact_resistance": 0.01}
    )
    
    circuit.add_component(battery)
    circuit.add_component(led)
    circuit.add_component(switch)
    
    # Add connections
    # Battery to switch
    connection1 = ConnectionEdge(
        source="battery_1",
        target="switch_1",
        connection_type=ConnectionType.WIRE,
        connection_correctness=0.9,
        expected_connection=True,
        connection_quality=0.85,
        current_direction="forward",
        flow_strength=1.0,
        robot_accessible=True,
        recommended_action=None,
        resistance=0.1,
        detection_confidence=0.88
    )
    
    # Switch to LED (missing connection for demonstration)
    # This will show up as a recommendation
    
    circuit.add_connection(connection1)
    
    return circuit

def main():
    """Test the graph output format."""
    print("🧪 Testing Graph Output for TD-BKT Algorithm")
    print("=" * 50)
    
    # Create sample circuit
    circuit = create_sample_circuit()
    
    # Generate JSON output
    json_output = circuit.to_json(include_recommendations=True)
    
    print("📊 JSON Output for TD-BKT Algorithm:")
    print("=" * 40)
    print(json_output)
    
    print("\n" + "=" * 50)
    print("🤖 Robot-Specific Export:")
    print("=" * 25)
    
    robot_data = circuit.export_for_robot()
    print(json.dumps(robot_data, indent=2, default=str))
    
    print("\n" + "=" * 50)
    print("📈 Key Features for TD-BKT:")
    print("=" * 30)
    print("✅ Component states for knowledge tracking")
    print("✅ Placement correctness metrics")
    print("✅ Learning context integration")
    print("✅ Connectivity analysis")
    print("✅ Automated recommendations")
    print("✅ Robot action planning")
    print("✅ Directed edges for current flow")
    print("✅ Spatial positioning for robot guidance")

if __name__ == "__main__":
    main() 