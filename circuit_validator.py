"""
Circuit Validator System
Validates detected circuits against reference designs and electrical principles.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import networkx as nx
from pathlib import Path

from data_structures import ComponentDetection, ConnectionGraph, ComponentType
from config import CIRCUIT_CONFIG


class ValidationResult(Enum):
    """Validation result types."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class ValidationIssue:
    """Represents a validation issue found in the circuit."""
    type: str  # "orientation", "placement", "connection", "electrical"
    component_id: str
    severity: str  # "error", "warning", "info"
    message: str
    suggestion: str
    position: Optional[Tuple[float, float]] = None


@dataclass
class ReferenceDesign:
    """Represents a reference circuit design."""
    name: str
    description: str
    components: List[Dict[str, Any]]  # Expected components with positions/orientations
    connections: List[Dict[str, Any]]  # Expected connections
    electrical_rules: Dict[str, Any]  # Electrical validation rules
    learning_objectives: List[str]  # What this circuit teaches
    difficulty_level: str  # "beginner", "intermediate", "advanced"


class CircuitValidator:
    """
    Main circuit validation system.
    Validates detected circuits against reference designs and electrical principles.
    """
    
    def __init__(self, reference_designs_path: str = "reference_designs/"):
        self.reference_designs_path = Path(reference_designs_path)
        self.reference_designs: Dict[str, ReferenceDesign] = {}
        self.electrical_rules = self._load_electrical_rules()
        self._load_reference_designs()
    
    def _load_reference_designs(self) -> None:
        """Load reference designs from JSON files."""
        if not self.reference_designs_path.exists():
            self.reference_designs_path.mkdir(parents=True, exist_ok=True)
            self._create_default_reference_designs()
        
        for design_file in self.reference_designs_path.glob("*.json"):
            with open(design_file, 'r') as f:
                design_data = json.load(f)
                design = ReferenceDesign(**design_data)
                self.reference_designs[design.name] = design
    
    def _create_default_reference_designs(self) -> None:
        """Create default reference designs for common Snap Circuit patterns."""
        # Simple LED circuit
        led_circuit = {
            "name": "simple_led_circuit",
            "description": "Basic LED circuit with battery and switch",
            "components": [
                {
                    "type": "battery_holder",
                    "position": {"x": 100, "y": 100},
                    "orientation": 0,
                    "required": True
                },
                {
                    "type": "switch",
                    "position": {"x": 200, "y": 100},
                    "orientation": 0,
                    "required": True
                },
                {
                    "type": "led",
                    "position": {"x": 300, "y": 100},
                    "orientation": 0,
                    "polarity": "correct",
                    "required": True
                }
            ],
            "connections": [
                {"from": "battery_holder", "to": "switch", "type": "positive"},
                {"from": "switch", "to": "led", "type": "positive"},
                {"from": "led", "to": "battery_holder", "type": "negative"}
            ],
            "electrical_rules": {
                "voltage_range": {"min": 1.5, "max": 9.0},
                "current_limit": 0.1,
                "polarity_sensitive": ["led", "battery_holder"]
            },
            "learning_objectives": [
                "Basic circuit closure",
                "Switch operation",
                "LED polarity"
            ],
            "difficulty_level": "beginner"
        }
        
        # Save to file
        with open(self.reference_designs_path / "simple_led_circuit.json", 'w') as f:
            json.dump(led_circuit, f, indent=2)
    
    def _load_electrical_rules(self) -> Dict[str, Any]:
        """Load electrical validation rules."""
        return {
            "voltage_limits": {
                "led": {"min": 1.5, "max": 9.0},
                "motor": {"min": 3.0, "max": 9.0},
                "speaker": {"min": 1.5, "max": 9.0}
            },
            "current_limits": {
                "led": 0.1,
                "motor": 0.5,
                "speaker": 0.2
            },
            "polarity_sensitive": ["led", "battery_holder", "motor"],
            "required_components": {
                "power_source": ["battery_holder"],
                "control": ["switch", "button"],
                "output": ["led", "speaker", "motor", "lamp", "buzzer"]
            }
        }
    
    def validate_circuit(self, circuit_graph: ConnectionGraph, 
                        reference_design: Optional[str] = None,
                        confidence_threshold: float = 0.75) -> Dict[str, Any]:
        """
        Validate a circuit against reference design and electrical rules.
        Only validates high-confidence components (same as visualizer).
        
        Args:
            circuit_graph: The detected circuit graph
            reference_design: Name of reference design to validate against
            confidence_threshold: Minimum confidence threshold for validation (default: 0.75)
            
        Returns:
            Validation results with issues and suggestions
        """
        issues: List[ValidationIssue] = []
        
        # Filter components by confidence threshold (same as visualizer)
        high_confidence_components = [
            comp for comp in circuit_graph.components 
            if comp.confidence > confidence_threshold
        ]
        
        print(f"Validating {len(high_confidence_components)} high-confidence components (>{confidence_threshold*100}%)")
        for comp in high_confidence_components:
            print(f"  - {comp.component_type.value}: {comp.confidence:.3f}")
        
        # Create a filtered graph with only high-confidence components
        filtered_graph = ConnectionGraph(
            components=high_confidence_components,
            edges=circuit_graph.edges,  # Keep original edges for now
            state=circuit_graph.state,
            timestamp=circuit_graph.timestamp,
            frame_id=circuit_graph.frame_id
        )
        
        # 1. Validate against reference design if provided
        if reference_design and reference_design in self.reference_designs:
            design_issues = self._validate_against_reference(
                filtered_graph, self.reference_designs[reference_design]
            )
            issues.extend(design_issues)
        
        # 2. Validate electrical rules (only high-confidence components)
        electrical_issues = self._validate_electrical_rules(filtered_graph)
        issues.extend(electrical_issues)
        
        # 3. Validate component orientations (only high-confidence components)
        orientation_issues = self._validate_component_orientations(filtered_graph)
        issues.extend(orientation_issues)
        
        # 4. Validate component placements (only high-confidence components)
        placement_issues = self._validate_component_placements(filtered_graph)
        issues.extend(placement_issues)
        
        # 5. Validate connections (only high-confidence components)
        connection_issues = self._validate_connections(filtered_graph)
        issues.extend(connection_issues)
        
        # Calculate overall validation result
        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")
        
        if error_count == 0 and warning_count == 0:
            overall_result = ValidationResult.CORRECT
        elif error_count == 0:
            overall_result = ValidationResult.PARTIAL
        else:
            overall_result = ValidationResult.INCORRECT
        
        return {
            "overall_result": overall_result.value,
            "issues": [self._issue_to_dict(issue) for issue in issues],
            "summary": {
                "total_issues": len(issues),
                "errors": error_count,
                "warnings": warning_count,
                "score": max(0, 100 - (error_count * 20) - (warning_count * 5))
            },
            "suggestions": self._generate_suggestions(issues)
        }
    
    def _validate_against_reference(self, circuit_graph: ConnectionGraph, 
                                  reference: ReferenceDesign) -> List[ValidationIssue]:
        """Validate circuit against a reference design."""
        issues = []
        
        # Check if required components are present
        detected_types = [comp.component_type.value for comp in circuit_graph.components]
        required_types = [comp["type"] for comp in reference.components if comp.get("required", False)]
        
        for required_type in required_types:
            if required_type not in detected_types:
                issues.append(ValidationIssue(
                    type="component",
                    component_id="missing",
                    severity="error",
                    message=f"Missing required component: {required_type}",
                    suggestion=f"Add a {required_type} component to complete the circuit"
                ))
        
        # Check component positions (if specified in reference)
        for ref_comp in reference.components:
            matching_components = [
                comp for comp in circuit_graph.components 
                if comp.component_type.value == ref_comp["type"]
            ]
            
            if matching_components:
                comp = matching_components[0]  # Take first match
                ref_pos = ref_comp.get("position", {})
                
                if ref_pos:
                    # Calculate position deviation
                    comp_center_x = (comp.bbox.x1 + comp.bbox.x2) / 2
                    comp_center_y = (comp.bbox.y1 + comp.bbox.y2) / 2
                    
                    distance = np.sqrt(
                        (comp_center_x - ref_pos["x"])**2 + 
                        (comp_center_y - ref_pos["y"])**2
                    )
                    
                    if distance > 100:  # Threshold for position deviation
                        issues.append(ValidationIssue(
                            type="placement",
                            component_id=comp.id,
                            severity="warning",
                            message=f"Component {comp.component_type.value} is not in expected position",
                            suggestion=f"Move {comp.component_type.value} closer to expected position",
                            position=(comp_center_x, comp_center_y)
                        ))
        
        return issues
    
    def _validate_electrical_rules(self, circuit_graph: ConnectionGraph) -> List[ValidationIssue]:
        """Validate electrical rules and principles."""
        issues = []
        
        # Check for power source
        power_sources = [
            comp for comp in circuit_graph.components 
            if comp.component_type.value in self.electrical_rules["required_components"]["power_source"]
        ]
        
        if not power_sources:
            issues.append(ValidationIssue(
                type="electrical",
                component_id="circuit",
                severity="error",
                message="No power source found in circuit",
                suggestion="Add a battery holder to provide power"
            ))
        
        # Check for output components
        output_components = [
            comp for comp in circuit_graph.components 
            if comp.component_type.value in self.electrical_rules["required_components"]["output"]
        ]
        
        if not output_components:
            issues.append(ValidationIssue(
                type="electrical",
                component_id="circuit",
                severity="warning",
                message="No output components found",
                suggestion="Add an LED, speaker, or motor to see the circuit in action"
            ))
        
        # Check voltage compatibility
        for comp in circuit_graph.components:
            comp_type = comp.component_type.value
            if comp_type in self.electrical_rules["voltage_limits"]:
                limits = self.electrical_rules["voltage_limits"][comp_type]
                
                # Estimate circuit voltage (simplified)
                circuit_voltage = circuit_graph.state.estimated_voltage or 3.0
                
                if circuit_voltage < limits["min"]:
                    issues.append(ValidationIssue(
                        type="electrical",
                        component_id=comp.id,
                        severity="warning",
                        message=f"{comp_type} may not work properly with {circuit_voltage}V",
                        suggestion=f"Use higher voltage battery (min {limits['min']}V for {comp_type})"
                    ))
                elif circuit_voltage > limits["max"]:
                    issues.append(ValidationIssue(
                        type="electrical",
                        component_id=comp.id,
                        severity="error",
                        message=f"{comp_type} may be damaged by {circuit_voltage}V",
                        suggestion=f"Use lower voltage battery (max {limits['max']}V for {comp_type})"
                    ))
        
        return issues
    
    def _validate_component_orientations(self, circuit_graph: ConnectionGraph) -> List[ValidationIssue]:
        """Validate component orientations and polarity."""
        issues = []
        
        for comp in circuit_graph.components:
            comp_type = comp.component_type.value
            
            # Check polarity-sensitive components
            if comp_type in self.electrical_rules["polarity_sensitive"]:
                # For now, just check if orientation is reasonable
                # In a full implementation, we need more sophisticated orientation detection
                
                if comp_type == "led":
                    # LEDs should typically be oriented with anode towards positive
                    # This is a placeholder - we need actual polarity detection
                    issues.append(ValidationIssue(
                        type="orientation",
                        component_id=comp.id,
                        severity="info",
                        message=f"LED polarity should be verified",
                        suggestion="Ensure LED anode (longer leg) connects to positive terminal"
                    ))
                
                elif comp_type == "battery_holder":
                    # Battery holders should have clear positive/negative terminals
                    issues.append(ValidationIssue(
                        type="orientation",
                        component_id=comp.id,
                        severity="info",
                        message="Verify battery holder polarity",
                        suggestion="Ensure + terminal connects to positive side of circuit"
                    ))
        
        return issues
    
    def _validate_component_placements(self, circuit_graph: ConnectionGraph) -> List[ValidationIssue]:
        """Validate component placements and spacing."""
        issues = []
        
        components = circuit_graph.components
        
        # Check for component overlaps
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components[i+1:], i+1):
                if self._components_overlap(comp1, comp2):
                    issues.append(ValidationIssue(
                        type="placement",
                        component_id=comp1.id,
                        severity="error",
                        message=f"Component {comp1.component_type.value} overlaps with {comp2.component_type.value}",
                        suggestion="Separate overlapping components"
                    ))
        
        # Check for reasonable spacing
        for comp in components:
            nearby_components = self._find_nearby_components(comp, components, threshold=50)
            if len(nearby_components) > 3:
                issues.append(ValidationIssue(
                    type="placement",
                    component_id=comp.id,
                    severity="warning",
                    message=f"Component {comp.component_type.value} is crowded",
                    suggestion="Provide more space around components for better connections"
                ))
        
        return issues
    
    def _validate_connections(self, circuit_graph: ConnectionGraph) -> List[ValidationIssue]:
        """Validate circuit connections using tolerance-based detection (same as visualizer)."""
        issues = []
        
        # Use tolerance-based connection detection like the visualizer
        tolerance_connections = self._find_tolerance_connections(circuit_graph.components)
        
        # Check for circuit completeness using tolerance connections
        if len(tolerance_connections) < 2:
            issues.append(ValidationIssue(
                type="connection",
                component_id="circuit",
                severity="error",
                message="Circuit appears incomplete - very few connections detected",
                suggestion="Connect components to form a complete circuit loop"
            ))
        else:
            # Check if we have a reasonable circuit (power source + output component)
            component_types = [comp.component_type.value for comp in circuit_graph.components]
            has_power = any(comp_type in ["battery_holder"] for comp_type in component_types)
            has_output = any(comp_type in ["led", "motor", "speaker", "lamp", "buzzer"] for comp_type in component_types)
            
            if has_power and has_output and len(tolerance_connections) >= 3:
                # This looks like a complete circuit - don't flag any connection errors
                pass
            elif has_power and has_output and len(tolerance_connections) >= 2:
                # Minimal circuit with power and output - likely working
                pass
            elif len(tolerance_connections) < 3:
                issues.append(ValidationIssue(
                    type="connection",
                    component_id="circuit",
                    severity="warning",
                    message="Circuit may be incomplete - few connections detected",
                    suggestion="Verify all connections are properly made"
                ))
        
        # Check for isolated components using tolerance connections
        connected_components = set()
        for comp1_id, comp2_id, _ in tolerance_connections:
            connected_components.add(comp1_id)
            connected_components.add(comp2_id)
        
        for comp in circuit_graph.components:
            if comp.id not in connected_components:
                issues.append(ValidationIssue(
                    type="connection",
                    component_id=comp.id,
                    severity="warning",
                    message=f"Component {comp.component_type.value} is not connected",
                    suggestion="Connect this component to the circuit"
                ))
        
        return issues
    
    def _find_tolerance_connections(self, components: List[ComponentDetection], tolerance: float = 35) -> List[tuple]:
        """
        Find connections between components using tolerance-based clustering (same as visualizer).
        
        Args:
            components: List of components with connection points
            tolerance: Distance tolerance for considering points connected (pixels)
        
        Returns:
            List of (comp1_id, comp2_id, distance) tuples for connected components
        """
        from scipy.spatial.distance import cdist
        
        connections = []
        component_points = {}
        
        # Collect all connection points for each component
        for component in components:
            comp_id = component.id
            points = component.connection_points
            component_points[comp_id] = np.array(points) if points else np.array([]).reshape(0, 2)
        
        # Compare each pair of components
        component_ids = list(component_points.keys())
        for i in range(len(component_ids)):
            for j in range(i + 1, len(component_ids)):
                comp1_id = component_ids[i]
                comp2_id = component_ids[j]
                
                points1 = component_points[comp1_id]
                points2 = component_points[comp2_id]
                
                if len(points1) == 0 or len(points2) == 0:
                    continue
                
                # Calculate pairwise distances between all connection points
                distances = cdist(points1, points2)
                min_distance = np.min(distances)
                
                # If any points are within tolerance, consider components connected
                if min_distance <= tolerance:
                    connections.append((comp1_id, comp2_id, min_distance))
        
        return connections
    
    def _components_overlap(self, comp1: ComponentDetection, comp2: ComponentDetection) -> bool:
        """Check if two components overlap significantly (allow minor overlaps in snap circuits)."""
        # Calculate overlap area
        overlap_width = max(0, min(comp1.bbox.x2, comp2.bbox.x2) - max(comp1.bbox.x1, comp2.bbox.x1))
        overlap_height = max(0, min(comp1.bbox.y2, comp2.bbox.y2) - max(comp1.bbox.y1, comp2.bbox.y1))
        overlap_area = overlap_width * overlap_height
        
        # Calculate areas of both components
        area1 = (comp1.bbox.x2 - comp1.bbox.x1) * (comp1.bbox.y2 - comp1.bbox.y1)
        area2 = (comp2.bbox.x2 - comp2.bbox.x1) * (comp2.bbox.y2 - comp2.bbox.y1)
        
        # Only consider it an overlap if more than 50% of either component overlaps
        # (snap circuits naturally have components that touch/connect)
        overlap_ratio1 = overlap_area / area1 if area1 > 0 else 0
        overlap_ratio2 = overlap_area / area2 if area2 > 0 else 0
        
        return overlap_ratio1 > 0.5 or overlap_ratio2 > 0.5
    
    def _find_nearby_components(self, target: ComponentDetection, 
                               components: List[ComponentDetection], 
                               threshold: float) -> List[ComponentDetection]:
        """Find components within threshold distance."""
        nearby = []
        target_center = ((target.bbox.x1 + target.bbox.x2) / 2, 
                        (target.bbox.y1 + target.bbox.y2) / 2)
        
        for comp in components:
            if comp.id == target.id:
                continue
                
            comp_center = ((comp.bbox.x1 + comp.bbox.x2) / 2, 
                          (comp.bbox.y1 + comp.bbox.y2) / 2)
            
            distance = np.sqrt(
                (target_center[0] - comp_center[0])**2 + 
                (target_center[1] - comp_center[1])**2
            )
            
            if distance < threshold:
                nearby.append(comp)
        
        return nearby
    
    def _issue_to_dict(self, issue: ValidationIssue) -> Dict[str, Any]:
        """Convert ValidationIssue to dictionary."""
        return {
            "type": issue.type,
            "component_id": issue.component_id,
            "severity": issue.severity,
            "message": issue.message,
            "suggestion": issue.suggestion,
            "position": issue.position
        }
    
    def _generate_suggestions(self, issues: List[ValidationIssue]) -> List[str]:
        """Generate prioritized suggestions based on issues."""
        suggestions = []
        
        # Prioritize by severity
        errors = [issue for issue in issues if issue.severity == "error"]
        warnings = [issue for issue in issues if issue.severity == "warning"]
        
        if errors:
            suggestions.append("🔴 Critical Issues Found:")
            for error in errors[:3]:  # Show top 3 errors
                suggestions.append(f"  • {error.suggestion}")
        
        if warnings:
            suggestions.append("🟡 Improvements:")
            for warning in warnings[:2]:  # Show top 2 warnings
                suggestions.append(f"  • {warning.suggestion}")
        
        return suggestions
    
    def find_matching_reference_design(self, circuit_graph: ConnectionGraph) -> Optional[str]:
        """Find the best matching reference design for a circuit."""
        best_match = None
        best_score = 0
        
        for name, design in self.reference_designs.items():
            score = self._calculate_design_similarity(circuit_graph, design)
            if score > best_score:
                best_score = score
                best_match = name
        
        return best_match if best_score > 0.5 else None
    
    def _calculate_design_similarity(self, circuit_graph: ConnectionGraph, 
                                   design: ReferenceDesign) -> float:
        """Calculate similarity score between circuit and reference design."""
        # Count matching component types
        detected_types = [comp.component_type.value for comp in circuit_graph.components]
        expected_types = [comp["type"] for comp in design.components]
        
        matching_types = set(detected_types) & set(expected_types)
        total_types = set(detected_types) | set(expected_types)
        
        if not total_types:
            return 0.0
        
        type_similarity = len(matching_types) / len(total_types)
        
        # Could add more sophisticated similarity metrics here
        # (position similarity, connection similarity, etc.)
        
        return type_similarity 