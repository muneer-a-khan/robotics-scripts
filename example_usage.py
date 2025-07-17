#!/usr/bin/env python3
"""
Example usage of the Circuit Validator System
Shows how to validate circuits against reference designs and detect orientation issues.
"""

import cv2
from main import SnapCircuitVisionSystem


def example_validation_usage():
    """Example of using the circuit validation system."""
    
    print("=== Circuit Validation Example ===\n")
    
    # 1. Initialize system with validation enabled
    print("1. Initializing system with validation enabled...")
    system = SnapCircuitVisionSystem(
        save_outputs=True,
        display_results=True,
        enable_validation=True  # Enable circuit validation
    )
    
    # 2. Process a test image
    print("\n2. Processing test image...")
    test_image_path = "snap_circuit_image.jpg"
    
    try:
        result = system.process_image(test_image_path)
        
        # 3. Display validation results
        print("\n3. Validation Results:")
        print("=" * 50)
        
        if result.validation_result:
            validation = result.validation_result
            
            # Overall result
            overall_result = validation.get("overall_result", "unknown")
            print(f"Overall Result: {overall_result.upper()}")
            
            # Score and summary
            summary = validation.get("summary", {})
            score = summary.get("score", 0)
            errors = summary.get("errors", 0)
            warnings = summary.get("warnings", 0)
            
            print(f"Score: {score}%")
            print(f"Issues: {errors} errors, {warnings} warnings")
            
            # List specific issues
            issues = validation.get("issues", [])
            if issues:
                print("\nSpecific Issues:")
                for issue in issues:
                    severity = issue.get("severity", "unknown")
                    message = issue.get("message", "No message")
                    suggestion = issue.get("suggestion", "No suggestion")
                    
                    severity_symbol = {
                        "error": "❌",
                        "warning": "⚠️",
                        "info": "ℹ️"
                    }.get(severity, "•")
                    
                    print(f"  {severity_symbol} {message}")
                    print(f"    → {suggestion}")
            
            # Suggestions
            suggestions = validation.get("suggestions", [])
            if suggestions:
                print("\nSuggestions:")
                for suggestion in suggestions:
                    print(f"  {suggestion}")
        else:
            print("No validation results available")
        
        # 4. Circuit analysis
        print("\n4. Circuit Analysis:")
        print("=" * 50)
        
        circuit = result.connection_graph
        print(f"Components found: {len(circuit.components)}")
        print(f"Connections: {len(circuit.edges)}")
        print(f"Circuit closed: {circuit.state.is_circuit_closed}")
        print(f"Power on: {circuit.state.power_on}")
        
        # List detected components
        if circuit.components:
            print("\nDetected Components:")
            for comp in circuit.components:
                print(f"  • {comp.component_type.value} (confidence: {comp.confidence:.2f})")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        print("Make sure 'snap_circuit_image.jpg' exists in the current directory")


def example_live_validation():
    """Example of using live camera validation."""
    
    print("\n=== Live Camera Validation Example ===\n")
    
    print("To use live camera validation, run:")
    print("python main.py --mode camera --validate")
    print("")
    print("This will:")
    print("• Enable real-time circuit validation")
    print("• Show validation results in the camera feed")
    print("• Detect component orientations")
    print("• Check against reference designs")
    print("• Provide correction suggestions")


def example_reference_design_creation():
    """Example of creating custom reference designs."""
    
    print("\n=== Creating Custom Reference Designs ===\n")
    
    reference_design_example = {
        "name": "motor_circuit",
        "description": "Basic motor circuit with battery, switch, and motor",
        "components": [
            {
                "type": "battery_holder",
                "position": {"x": 100, "y": 200},
                "orientation": 0,
                "required": True
            },
            {
                "type": "switch",
                "position": {"x": 300, "y": 200},
                "orientation": 0,
                "required": True
            },
            {
                "type": "motor",
                "position": {"x": 500, "y": 200},
                "orientation": 0,
                "polarity": "correct",
                "required": True
            }
        ],
        "connections": [
            {"from": "battery_holder", "to": "switch", "type": "positive"},
            {"from": "switch", "to": "motor", "type": "positive"},
            {"from": "motor", "to": "battery_holder", "type": "negative"}
        ],
        "electrical_rules": {
            "voltage_range": {"min": 3.0, "max": 9.0},
            "current_limit": 0.5,
            "polarity_sensitive": ["motor", "battery_holder"]
        },
        "learning_objectives": [
            "Motor control",
            "Switch operation",
            "Power distribution"
        ],
        "difficulty_level": "beginner"
    }
    
    print("Example reference design structure:")
    print("Save this as 'reference_designs/motor_circuit.json':")
    print("")
    
    import json
    print(json.dumps(reference_design_example, indent=2))


if __name__ == "__main__":
    # Run examples
    example_validation_usage()
    example_live_validation()
    example_reference_design_creation()
    
    print("\n" + "=" * 70)
    print("CIRCUIT VALIDATOR FEATURES SUMMARY")
    print("=" * 70)
    print("✅ Reference design validation")
    print("✅ Component orientation detection")
    print("✅ Electrical rules checking")
    print("✅ Component placement validation")
    print("✅ Connection analysis")
    print("✅ Real-time validation feedback")
    print("✅ Custom reference design support")
    print("✅ Detailed error reporting")
    print("✅ Correction suggestions")
    print("=" * 70) 