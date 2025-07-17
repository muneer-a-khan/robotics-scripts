#!/usr/bin/env python3
"""
Test script to demonstrate the validation visualization feature.
Creates sample validation data and generates visualizations.
"""

import json
import time
from pathlib import Path
from live_circuit_visualizer import create_live_visualization


def create_sample_graph_data():
    """Create sample graph data for testing."""
    return {
        "connection_graph": {
            "components": [
                {
                    "id": "battery_holder-0",
                    "component_type": "battery_holder",
                    "bbox": [100, 100, 200, 150],
                    "confidence": 0.95,
                    "connection_points": []
                },
                {
                    "id": "switch-0", 
                    "component_type": "switch",
                    "bbox": [250, 100, 300, 150],
                    "confidence": 0.88,
                    "connection_points": []
                },
                {
                    "id": "led-0",
                    "component_type": "led", 
                    "bbox": [350, 100, 400, 150],
                    "confidence": 0.92,
                    "connection_points": []
                }
            ],
            "edges": [
                {
                    "component_1": "battery_holder-0",
                    "component_2": "switch-0",
                    "connection_type": "wire"
                },
                {
                    "component_1": "switch-0", 
                    "component_2": "led-0",
                    "connection_type": "wire"
                }
            ],
            "state": {
                "is_circuit_closed": True,
                "power_on": True
            }
        }
    }


def create_sample_validation_data(result_type="correct"):
    """Create sample validation data for testing."""
    
    if result_type == "correct":
        return {
            "overall_result": "correct",
            "summary": {
                "total_issues": 0,
                "errors": 0,
                "warnings": 0,
                "score": 100
            },
            "issues": [],
            "suggestions": []
        }
    
    elif result_type == "partial":
        return {
            "overall_result": "partial",
            "summary": {
                "total_issues": 2,
                "errors": 0,
                "warnings": 2,
                "score": 80
            },
            "issues": [
                {
                    "type": "orientation",
                    "severity": "warning",
                    "message": "LED polarity should be verified",
                    "suggestion": "Check LED anode/cathode orientation"
                },
                {
                    "type": "placement",
                    "severity": "warning",
                    "message": "Components could be spaced better",
                    "suggestion": "Provide more space between components"
                }
            ],
            "suggestions": [
                "🟡 Improvements:",
                "  • Check LED anode/cathode orientation",
                "  • Provide more space between components"
            ]
        }
    
    elif result_type == "incorrect":
        return {
            "overall_result": "incorrect",
            "summary": {
                "total_issues": 3,
                "errors": 2,
                "warnings": 1,
                "score": 40
            },
            "issues": [
                {
                    "type": "electrical",
                    "severity": "error",
                    "message": "No power source found in circuit",
                    "suggestion": "Add a battery holder to provide power"
                },
                {
                    "type": "orientation",
                    "severity": "error",
                    "message": "LED polarity is reversed",
                    "suggestion": "Flip LED to correct polarity"
                },
                {
                    "type": "connection",
                    "severity": "warning",
                    "message": "Circuit is not closed",
                    "suggestion": "Connect components to form complete loop"
                }
            ],
            "suggestions": [
                "🔴 Critical Issues Found:",
                "  • Add a battery holder to provide power",
                "  • Flip LED to correct polarity",
                "🟡 Improvements:",
                "  • Connect components to form complete loop"
            ]
        }
    
    else:  # unknown
        return {
            "overall_result": "unknown",
            "summary": {
                "total_issues": 0,
                "errors": 0,
                "warnings": 0,
                "score": 0
            },
            "issues": [],
            "suggestions": []
        }


def test_validation_visualization():
    """Test the validation visualization with different scenarios."""
    
    print("🧪 Testing Validation Visualization Feature")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("test_validation_output")
    output_dir.mkdir(exist_ok=True)
    
    # Sample graph data
    graph_data = create_sample_graph_data()
    
    # Test scenarios
    scenarios = [
        ("correct", "✅ Perfect Circuit"),
        ("partial", "⚠️ Circuit with Warnings"),
        ("incorrect", "❌ Circuit with Errors"),
        ("unknown", "❓ Unknown Status")
    ]
    
    for scenario_type, description in scenarios:
        print(f"\n📊 Testing: {description}")
        
        # Create validation data
        validation_data = create_sample_validation_data(scenario_type)
        
        # Generate timestamp
        timestamp = int(time.time() * 1000)
        
        # Create visualization
        try:
            visualization_path = create_live_visualization(
                graph_data=graph_data,
                timestamp=timestamp,
                output_dir=str(output_dir),
                validation_data=validation_data
            )
            
            if visualization_path:
                print(f"   ✅ Generated: {visualization_path}")
                
                # Print validation summary
                summary = validation_data.get("summary", {})
                print(f"   📈 Score: {summary.get('score', 0)}%")
                print(f"   📊 Issues: {summary.get('total_issues', 0)} "
                      f"(Errors: {summary.get('errors', 0)}, "
                      f"Warnings: {summary.get('warnings', 0)})")
            else:
                print(f"   ❌ Failed to generate visualization")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print(f"\n📁 All test visualizations saved to: {output_dir}")
    print("\n🎉 Validation visualization test completed!")
    
    # Show usage instructions
    print("\n" + "=" * 50)
    print("🚀 HOW TO USE IN REAL SYSTEM:")
    print("=" * 50)
    print("1. Enable validation: python main.py --mode camera --validate")
    print("2. Watch live feed with validation scores in top right")
    print("3. See validation results in saved PNG files")
    print("4. Check validation data in JSON output files")
    print("=" * 50)


if __name__ == "__main__":
    test_validation_visualization() 