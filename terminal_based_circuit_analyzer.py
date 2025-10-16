#!/usr/bin/env python3
"""
Terminal-Based Circuit Analyzer
Analyzes circuits by tracking connections between component terminals (left/right or top/bottom)
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import time

@dataclass
class ComponentTerminal:
    """Represents a terminal (connection point) on a component"""
    component_id: str
    component_type: str
    terminal_side: str  # 'left', 'right', 'top', 'bottom', 'positive', 'negative'
    bbox: List[float]  # Terminal bounding box [x1, y1, x2, y2]
    center: Tuple[float, float]
    board_side: str  # 'left' or 'right'

@dataclass
class TerminalConnection:
    """Represents a connection between two component terminals"""
    from_component_id: str
    from_terminal: str
    to_component_id: str
    to_terminal: str
    overlap_percentage: float
    distance: float

class TerminalBasedCircuitAnalyzer:
    def __init__(self, min_overlap_percentage: float = 15.0):
        """
        Initialize terminal-based circuit analyzer
        
        Args:
            min_overlap_percentage: Minimum overlap percentage for valid connection (default 15%)
        """
        self.min_overlap_percentage = min_overlap_percentage
        self.components = {}  # component_id -> component data
        self.terminals = {}  # component_id -> {'left': Terminal, 'right': Terminal} or {'positive': Terminal, 'negative': Terminal}
        self.connections = []  # List of TerminalConnections
        self.circuit_trace = []  # Trace from Battery+ to Battery-
        self.led_orientations = {}  # LED orientation results
        
    def add_component(self, component_id: str, component_type: str, bbox: List[float], 
                     confidence: float, board_side: str, led_orientation: Optional[Dict] = None):
        """
        Add a component and create its terminals
        
        Args:
            component_id: Unique component identifier
            component_type: Type of component (e.g., 'Battery Holder', 'Wire', 'LED_2 (Red)')
            bbox: Bounding box [x1, y1, x2, y2]
            confidence: Detection confidence
            board_side: 'left' or 'right'
            led_orientation: Optional LED orientation data (for LEDs only)
        """
        # Apply component name mapping (e.g., Photoresistor -> Horn)
        from component_name_mapper import map_component_name
        component_type = map_component_name(component_type)
        
        self.components[component_id] = {
            'type': component_type,
            'bbox': bbox,
            'confidence': confidence,
            'board_side': board_side,
            'led_orientation': led_orientation
        }
        
        # Create terminals based on component type
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Determine if component is horizontal or vertical
        is_horizontal = width > height
        
        if 'Battery Holder' in component_type:
            # Battery: split horizontally (top = negative, bottom = positive)
            # Top half (negative)
            negative_bbox = [x1, y1, x2, center_y]
            negative_center = (center_x, (y1 + center_y) / 2)
            
            # Bottom half (positive)
            positive_bbox = [x1, center_y, x2, y2]
            positive_center = (center_x, (center_y + y2) / 2)
            
            self.terminals[component_id] = {
                'negative': ComponentTerminal(
                    component_id=component_id,
                    component_type=component_type,
                    terminal_side='negative',
                    bbox=negative_bbox,
                    center=negative_center,
                    board_side=board_side
                ),
                'positive': ComponentTerminal(
                    component_id=component_id,
                    component_type=component_type,
                    terminal_side='positive',
                    bbox=positive_bbox,
                    center=positive_center,
                    board_side=board_side
                )
            }
        else:
            # All other components: split into left/right
            # If horizontal: left = left side, right = right side
            # If vertical: left = top side, right = bottom side
            
            if is_horizontal:
                # Horizontal component
                left_bbox = [x1, y1, center_x, y2]
                left_center = ((x1 + center_x) / 2, center_y)
                
                right_bbox = [center_x, y1, x2, y2]
                right_center = ((center_x + x2) / 2, center_y)
            else:
                # Vertical component (top = left terminal, bottom = right terminal)
                left_bbox = [x1, y1, x2, center_y]
                left_center = (center_x, (y1 + center_y) / 2)
                
                right_bbox = [x1, center_y, x2, y2]
                right_center = (center_x, (center_y + y2) / 2)
            
            self.terminals[component_id] = {
                'left': ComponentTerminal(
                    component_id=component_id,
                    component_type=component_type,
                    terminal_side='left',
                    bbox=left_bbox,
                    center=left_center,
                    board_side=board_side
                ),
                'right': ComponentTerminal(
                    component_id=component_id,
                    component_type=component_type,
                    terminal_side='right',
                    bbox=right_bbox,
                    center=right_center,
                    board_side=board_side
                )
            }
            
            # Store LED polarity information for later validation
            if 'LED_2 (Red)' in component_type and led_orientation:
                self.led_orientations[component_id] = led_orientation
    
    def calculate_overlap_percentage(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate overlap percentage between two bounding boxes
        Returns the overlap area as a percentage of the smaller bbox
        """
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        # Calculate intersection
        x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        overlap_area = x_overlap * y_overlap
        
        if overlap_area == 0:
            return 0.0
        
        # Calculate areas
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        smaller_area = min(area1, area2)
        
        if smaller_area == 0:
            return 0.0
        
        # Return overlap as percentage of smaller component
        return (overlap_area / smaller_area) * 100.0
    
    def calculate_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    def analyze_terminal_connections(self):
        """
        Analyze connections between component terminals
        Process batteries FIRST to ensure they connect to different wires
        """
        self.connections.clear()
        
        # Track which terminals are already reserved
        reserved_terminals = set()  # Set of (comp_id, terminal_name) tuples
        
        # PHASE 1: Process battery terminals FIRST with forced differentiation
        battery_components = [(cid, cdata) for cid, cdata in self.components.items() 
                            if 'Battery Holder' in cdata['type']]
        
        for battery_id, battery_data in battery_components:
            board_side = battery_data['board_side']
            battery_terminals = self.terminals[battery_id]
            
            # Find connections for positive and negative, ensuring they're different
            positive_candidates = []
            negative_candidates = []
            
            # Scan all potential connections for both terminals
            for comp_id, comp_terminals in self.terminals.items():
                if comp_id == battery_id:
                    continue
                if self.components[comp_id]['board_side'] != board_side:
                    continue
                
                for terminal_name, terminal in comp_terminals.items():
                    # Check positive terminal
                    if 'positive' in battery_terminals:
                        overlap_pct = self.calculate_overlap_percentage(
                            battery_terminals['positive'].bbox, terminal.bbox)
                        if overlap_pct >= self.min_overlap_percentage:
                            distance = self.calculate_distance(
                                battery_terminals['positive'].center, terminal.center)
                            score = overlap_pct - (distance / 1000.0)
                            positive_candidates.append((score, comp_id, terminal_name, overlap_pct, distance))
                    
                    # Check negative terminal
                    if 'negative' in battery_terminals:
                        overlap_pct = self.calculate_overlap_percentage(
                            battery_terminals['negative'].bbox, terminal.bbox)
                        if overlap_pct >= self.min_overlap_percentage:
                            distance = self.calculate_distance(
                                battery_terminals['negative'].center, terminal.center)
                            score = overlap_pct - (distance / 1000.0)
                            negative_candidates.append((score, comp_id, terminal_name, overlap_pct, distance))
            
            # Sort by score
            positive_candidates.sort(reverse=True, key=lambda x: x[0])
            negative_candidates.sort(reverse=True, key=lambda x: x[0])
            
            # Assign connections, ensuring they're DIFFERENT
            pos_conn = None
            neg_conn = None
            
            # Try to find non-conflicting pairs
            for pos_score, pos_comp, pos_term, pos_overlap, pos_dist in positive_candidates:
                for neg_score, neg_comp, neg_term, neg_overlap, neg_dist in negative_candidates:
                    # Check if they connect to different terminals
                    if (pos_comp, pos_term) != (neg_comp, neg_term):
                        pos_conn = TerminalConnection(battery_id, 'positive', pos_comp, pos_term, pos_overlap, pos_dist)
                        neg_conn = TerminalConnection(battery_id, 'negative', neg_comp, neg_term, neg_overlap, neg_dist)
                        reserved_terminals.add((pos_comp, pos_term))
                        reserved_terminals.add((neg_comp, neg_term))
                        break
                if pos_conn and neg_conn:
                    break
            
            # Add the connections
            if pos_conn:
                self.connections.append(pos_conn)
            if neg_conn:
                self.connections.append(neg_conn)
        
        # PHASE 2: Process all other components
        for comp1_id, comp1_terminals in self.terminals.items():
            # Skip batteries (already processed)
            if 'Battery Holder' in self.components[comp1_id]['type']:
                continue
            
            comp1_board_side = self.components[comp1_id]['board_side']
            
            for terminal1_name, terminal1 in comp1_terminals.items():
                # Skip if this terminal is already reserved by a battery
                if (comp1_id, terminal1_name) in reserved_terminals:
                    continue
                
                # Find best connection
                best_connection = None
                best_score = 0.0
                
                for comp2_id, comp2_terminals in self.terminals.items():
                    if comp1_id == comp2_id:
                        continue
                    if self.components[comp2_id]['board_side'] != comp1_board_side:
                        continue
                    
                    for terminal2_name, terminal2 in comp2_terminals.items():
                        # Skip if reserved
                        if (comp2_id, terminal2_name) in reserved_terminals:
                            continue
                        
                        overlap_pct = self.calculate_overlap_percentage(terminal1.bbox, terminal2.bbox)
                        if overlap_pct >= self.min_overlap_percentage:
                            distance = self.calculate_distance(terminal1.center, terminal2.center)
                            score = overlap_pct - (distance / 1000.0)
                            
                            if score > best_score:
                                best_score = score
                                best_connection = TerminalConnection(
                                    comp1_id, terminal1_name, comp2_id, terminal2_name,
                                    overlap_pct, distance)
                
                if best_connection:
                    self.connections.append(best_connection)
    
    def trace_circuit_from_battery(self, battery_id: str, debug: bool = True) -> Dict[str, Any]:
        """
        Trace circuit path from Battery+ to Battery-
        Uses BFS to find if there's a path from positive to negative terminal
        """
        self.circuit_trace.clear()
        
        if battery_id not in self.terminals:
            return {
                'is_complete': False,
                'reason': 'Battery holder not found in components',
                'trace': []
            }
        
        # Use BFS to find path from Battery+ to Battery-
        from collections import deque
        
        start = (battery_id, 'positive')
        goal = (battery_id, 'negative')
        
        queue = deque([(start, [start])])
        visited = {start}
        
        if debug:
            print(f"\n🔍 CIRCUIT TRACE DEBUG for {battery_id} ({self.components[battery_id]['board_side']} board)")
            print(f"   Start: {start}, Goal: {goal}")
        
        step_count = 0
        while queue:
            (current_comp, current_terminal), path = queue.popleft()
            step_count += 1
            
            if debug and step_count <= 20:  # Limit debug output
                print(f"   Step {step_count}: At {current_comp}[{current_terminal}], path length: {len(path)}")
            
            # Check if we've reached Battery-
            if (current_comp, current_terminal) == goal:
                # Found a complete circuit!
                trace_path = [(comp_id, term, self.components[comp_id]['type']) for comp_id, term in path]
                if debug:
                    print(f"   ✅ FOUND COMPLETE CIRCUIT! Path length: {len(path)}")
                return {
                    'is_complete': True,
                    'reason': f'Complete circuit: Battery+ → {len(path)-2} components → Battery-',
                    'trace': trace_path,
                    'path_length': len(path)
                }
            
            # Find all connections from current terminal
            connections_found = 0
            for conn in self.connections:
                next_comp = None
                next_entry_terminal = None
                
                # Check both directions of the connection
                if conn.from_component_id == current_comp and conn.from_terminal == current_terminal:
                    next_comp = conn.to_component_id
                    next_entry_terminal = conn.to_terminal
                elif conn.to_component_id == current_comp and conn.to_terminal == current_terminal:
                    next_comp = conn.from_component_id
                    next_entry_terminal = conn.from_terminal
                
                if next_comp and next_entry_terminal:
                    # Current enters the next component at next_entry_terminal
                    # It exits from the other terminal
                    comp_terminals = list(self.terminals[next_comp].keys())
                    
                    # Find the exit terminal (the one that's NOT the entry terminal)
                    exit_terminal = None
                    for term in comp_terminals:
                        if term != next_entry_terminal:
                            exit_terminal = term
                            break
                    
                    # If component has only one terminal (shouldn't happen), use the same one
                    if exit_terminal is None:
                        exit_terminal = next_entry_terminal
                    
                    next_state = (next_comp, exit_terminal)
                    
                    if next_state not in visited:
                        visited.add(next_state)
                        new_path = path + [next_state]
                        queue.append((next_state, new_path))
                        connections_found += 1
                        if debug and step_count <= 20:
                            print(f"      → Found connection to {next_comp}[{next_entry_terminal}] → exits at [{exit_terminal}]")
            
            if debug and step_count <= 20 and connections_found == 0:
                print(f"      ⚠️ No connections found from {current_comp}[{current_terminal}]")
        
        # No path found from Battery+ to Battery-
        if debug:
            print(f"   ❌ NO COMPLETE CIRCUIT (checked {len(visited)} states)")
        return {
            'is_complete': False,
            'reason': f'No complete path from Battery+ to Battery- (checked {len(visited)} component terminals)',
            'trace': [],
            'path_length': 0
        }
    
    def validate_led_polarity(self, led_id: str, trace_path: List[Tuple]) -> Dict[str, Any]:
        """
        Validate LED polarity based on its position in the circuit trace
        The '+' symbol on the LED should be on the side closer to Battery+
        """
        if led_id not in self.led_orientations:
            return {
                'valid': None,
                'reason': 'No LED orientation data available'
            }
        
        led_orientation = self.led_orientations[led_id]
        
        # Find LED in trace path
        led_position_in_trace = None
        led_entry_terminal = None
        
        for idx, (comp_id, terminal, comp_type) in enumerate(trace_path):
            if comp_id == led_id:
                led_position_in_trace = idx
                led_entry_terminal = terminal
                break
        
        if led_position_in_trace is None:
            return {
                'valid': False,
                'reason': 'LED not found in circuit trace'
            }
        
        # Determine which side the '+' is on
        plus_position = led_orientation.get('plus_position')  # 'LEFT', 'RIGHT', 'TOP', 'BOTTOM'
        led_type = led_orientation.get('led_orientation')  # 'horizontal' or 'vertical'
        
        # Map physical position to terminal side
        if led_type == 'horizontal':
            plus_on_left = (plus_position == 'LEFT')
        else:  # vertical
            plus_on_left = (plus_position == 'TOP')  # top = left terminal for vertical components
        
        # The '+' should be on the entry side (closer to Battery+)
        # Entry terminal is where current enters the LED from Battery+
        if plus_on_left and led_entry_terminal == 'left':
            return {
                'valid': True,
                'reason': 'LED+ on left terminal (entry side from Battery+)',
                'plus_side': 'left',
                'entry_side': led_entry_terminal
            }
        elif not plus_on_left and led_entry_terminal == 'right':
            return {
                'valid': True,
                'reason': 'LED+ on right terminal (entry side from Battery+)',
                'plus_side': 'right',
                'entry_side': led_entry_terminal
            }
        else:
            return {
                'valid': False,
                'reason': f'LED+ on {"left" if plus_on_left else "right"} terminal but current enters from {led_entry_terminal} - LED is reversed',
                'plus_side': 'left' if plus_on_left else 'right',
                'entry_side': led_entry_terminal
            }
    
    def _label_wires_by_position(self, battery_id: str) -> Dict[str, int]:
        """
        Label wires by their distance from Battery+
        Wire 1 is closest to Battery+, Wire 2 is next, etc.
        Returns a mapping of component_id -> wire_number
        """
        wire_labels = {}
        board_side = self.components[battery_id]['board_side']
        
        # Get all wires on this board
        wires_on_board = [(cid, cdata) for cid, cdata in self.components.items() 
                         if 'Wire' in cdata['type'] and cdata['board_side'] == board_side]
        
        # BFS from Battery+ to label wires by distance
        from collections import deque
        queue = deque([(battery_id, 'positive', 0)])
        visited = set()
        wire_distances = {}  # wire_id -> min_distance_from_battery_plus
        
        while queue:
            current_comp, current_terminal, distance = queue.popleft()
            
            state = (current_comp, current_terminal)
            if state in visited:
                continue
            visited.add(state)
            
            # If this is a wire, record its distance
            if 'Wire' in self.components[current_comp]['type']:
                if current_comp not in wire_distances:
                    wire_distances[current_comp] = distance
            
            # Find all connections from this terminal
            for conn in self.connections:
                if conn.from_component_id == current_comp and conn.from_terminal == current_terminal:
                    # Current flows to the other component's terminal
                    next_comp = conn.to_component_id
                    next_terminal_entry = conn.to_terminal
                    
                    # Find the exit terminal (opposite of entry)
                    next_terminals = self.terminals[next_comp]
                    if next_terminal_entry in next_terminals:
                        # Get the opposite terminal
                        if next_terminal_entry in ['left', 'positive']:
                            next_terminal_exit = 'right' if 'right' in next_terminals else 'negative'
                        else:
                            next_terminal_exit = 'left' if 'left' in next_terminals else 'positive'
                        
                        queue.append((next_comp, next_terminal_exit, distance + 1))
        
        # Sort wires by distance and assign numbers
        sorted_wires = sorted(wire_distances.items(), key=lambda x: x[1])
        for idx, (wire_id, _) in enumerate(sorted_wires, 1):
            wire_labels[wire_id] = idx
        
        return wire_labels
    
    def get_summary(self) -> str:
        """Generate human-readable summary of terminal-based analysis"""
        lines = []
        lines.append("=== TERMINAL-BASED CIRCUIT ANALYSIS ===")
        lines.append(f"Total components: {len(self.components)}")
        lines.append(f"Total terminal connections: {len(self.connections)}")
        lines.append("")
        
        # Separate connections by board side
        left_connections = []
        right_connections = []
        
        for conn in self.connections:
            from_board_side = self.components[conn.from_component_id]['board_side']
            if from_board_side == 'left':
                left_connections.append(conn)
            else:
                right_connections.append(conn)
        
        lines.append(f"LEFT BOARD: {len(left_connections)} terminal connections")
        lines.append(f"RIGHT BOARD: {len(right_connections)} terminal connections")
        lines.append("")
        
        # Label wires for easier identification
        wire_labels = {}
        for comp_id, comp_data in self.components.items():
            if 'Battery Holder' in comp_data['type']:
                labels = self._label_wires_by_position(comp_id)
                wire_labels.update(labels)
        
        def get_component_display_name(comp_id):
            """Get display name with wire number if applicable"""
            comp_type = self.components[comp_id]['type']
            if 'Wire' in comp_type and comp_id in wire_labels:
                return f"Wire #{wire_labels[comp_id]}"
            return comp_type
        
        lines.append("")
        
        # Analyze circuit completeness for each board
        lines.append("Circuit Completeness Analysis:")
        
        # Find batteries on each side
        left_battery_id = None
        right_battery_id = None
        
        for comp_id, comp_data in self.components.items():
            if 'Battery Holder' in comp_data['type']:
                if comp_data['board_side'] == 'left':
                    left_battery_id = comp_id
                elif comp_data['board_side'] == 'right':
                    right_battery_id = comp_id
        
        # Check LEFT board
        if left_battery_id:
            result = self.trace_circuit_from_battery(left_battery_id)
            status = "CLOSED ✓" if result['is_complete'] else "OPEN ✗"
            lines.append(f"  LEFT BOARD: {status}")
            lines.append(f"    {result['reason']}")
            if 'path_length' in result:
                lines.append(f"    Path length: {result['path_length']} steps")
        else:
            lines.append(f"  LEFT BOARD: UNKNOWN")
            lines.append(f"    No battery holder detected")
        
        # Check RIGHT board
        if right_battery_id:
            result = self.trace_circuit_from_battery(right_battery_id)
            status = "CLOSED ✓" if result['is_complete'] else "OPEN ✗"
            lines.append(f"  RIGHT BOARD: {status}")
            lines.append(f"    {result['reason']}")
            if 'path_length' in result:
                lines.append(f"    Path length: {result['path_length']} steps")
        else:
            lines.append(f"  RIGHT BOARD: UNKNOWN")
            lines.append(f"    No battery holder detected")
        
        lines.append("")
        
        # Component terminals grouped by board side
        lines.append("Component Terminals:")
        lines.append("  LEFT BOARD:")
        for comp_id, terminals in sorted(self.terminals.items()):
            if self.components[comp_id]['board_side'] == 'left':
                comp_type = self.components[comp_id]['type']
                lines.append(f"    {comp_id} ({comp_type}):")
                for terminal_name, terminal in terminals.items():
                    lines.append(f"      {terminal_name}: bbox {[f'{x:.0f}' for x in terminal.bbox]}")
        
        lines.append("  RIGHT BOARD:")
        for comp_id, terminals in sorted(self.terminals.items()):
            if self.components[comp_id]['board_side'] == 'right':
                comp_type = self.components[comp_id]['type']
                lines.append(f"    {comp_id} ({comp_type}):")
                for terminal_name, terminal in terminals.items():
                    lines.append(f"      {terminal_name}: bbox {[f'{x:.0f}' for x in terminal.bbox]}")
        lines.append("")
        
        # Terminal connections grouped by board side and sorted logically
        lines.append("Terminal Connections:")
        
        # Helper function to sort connections (keep all bidirectional connections for circuit tracing)
        def sort_connections(connections):
            if not connections:
                return []
            
            # Build a map of component types for sorting priority
            priority_order = {
                'Battery Holder': 0,
                'Wire': 1,
                'LED_2 (Red)': 2,
                'Resistor': 2,
                'Press switch': 3,
                'Slide switch': 3,
                'U_1 blue music circuit': 4,
                'U_2 red alarm circuit': 4,
                'U_3 green space war circuit': 4,
                'Speaker': 4,
                'Whistle chip': 4,
                'Lamp': 4,
                'Horn': 4,
            }
            
            def get_priority(conn):
                from_type = self.components[conn.from_component_id]['type']
                to_type = self.components[conn.to_component_id]['type']
                from_priority = priority_order.get(from_type, 99)
                to_priority = priority_order.get(to_type, 99)
                
                # Sort by: from_priority, then from_terminal (positive before negative, left before right)
                terminal_order = {'positive': 0, 'negative': 1, 'left': 0, 'right': 1}
                from_terminal_priority = terminal_order.get(conn.from_terminal, 2)
                
                return (from_priority, from_terminal_priority, to_priority, conn.from_component_id, conn.to_component_id)
            
            return sorted(connections, key=get_priority)
        
        lines.append("  LEFT BOARD:")
        if left_connections:
            sorted_left = sort_connections(left_connections)
            for conn in sorted_left:
                from_comp = get_component_display_name(conn.from_component_id)
                to_comp = get_component_display_name(conn.to_component_id)
                lines.append(
                    f"    {from_comp}[{conn.from_terminal}] → {to_comp}[{conn.to_terminal}] "
                    f"(overlap: {conn.overlap_percentage:.1f}%, dist: {conn.distance:.1f}px)"
                )
        else:
            lines.append("    No connections on left board")
        
        lines.append("  RIGHT BOARD:")
        if right_connections:
            sorted_right = sort_connections(right_connections)
            for conn in sorted_right:
                from_comp = get_component_display_name(conn.from_component_id)
                to_comp = get_component_display_name(conn.to_component_id)
                lines.append(
                    f"    {from_comp}[{conn.from_terminal}] → {to_comp}[{conn.to_terminal}] "
                    f"(overlap: {conn.overlap_percentage:.1f}%, dist: {conn.distance:.1f}px)"
                )
        else:
            lines.append("    No connections on right board")
        
        return "\n".join(lines)

if __name__ == "__main__":
    print("Terminal-Based Circuit Analyzer initialized")

