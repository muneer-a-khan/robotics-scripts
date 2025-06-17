#!/usr/bin/env python3
"""
Quick validation of the trained model to see metrics.
"""

from models.component_detector import ComponentDetector

def validate_trained_model():
    """Validate the existing trained model."""
    print("📊 Validating trained model...")
    print("="*40)
    
    # Load the trained model
    detector = ComponentDetector()
    
    # Check dataset path
    augmented_data_yaml = "data/augmented_training/data.yaml"
    
    print(f"Using dataset: {augmented_data_yaml}")
    print("Running validation...")
    
    # Run validation
    val_results = detector.validate(augmented_data_yaml)
    
    print("✅ Validation complete!")
    
    # Handle validation results
    if val_results and isinstance(val_results, dict):
        # Try to extract mAP metrics from the results dict
        map50 = val_results.get('metrics/mAP50(B)', val_results.get('mAP50', 'N/A'))
        map50_95 = val_results.get('metrics/mAP50-95(B)', val_results.get('mAP50-95', 'N/A'))
        
        print(f"📈 mAP50: {map50}")
        print(f"📈 mAP50-95: {map50_95}")
        
        # Print all available metrics
        print("\n📊 All validation metrics:")
        for key, value in val_results.items():
            print(f"   {key}: {value}")
    else:
        print("⚠️  No validation metrics available")

if __name__ == "__main__":
    validate_trained_model() 