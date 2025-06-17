#!/usr/bin/env python3
"""
Improved training script for Snap Circuit component detection.
Uses augmented dataset with individual component images.
"""

import os
from pathlib import Path
from models.component_detector import ComponentDetector


def train_improved_model():
    """Train model with improved parameters and augmented dataset."""
    print("🚀 Starting Improved Snap Circuit Model Training...")
    print("="*50)
    
    # Check for augmented dataset
    augmented_data_yaml = "data/augmented_training/data.yaml"
    
    if not os.path.exists(augmented_data_yaml):
        print("❌ Augmented dataset not found!")
        print("   Run: python create_individual_component_dataset.py first")
        return
    
    # Initialize detector
    detector = ComponentDetector()
    
    # Improved training parameters (EXCLUDE data path - it goes as first positional arg)
    training_params = {
        "epochs": 200,          # More epochs for better learning
        "imgsz": 640,           # Standard YOLO image size
        "batch": 16,            # Increased for RTX 4070 Ti (12GB VRAM)
        "lr0": 0.001,           # Lower learning rate for fine-tuning
        "patience": 50,         # More patience for convergence
        "device": "cuda",       # GPU training enabled!
        "optimizer": "AdamW",   # Better optimizer
        "cos_lr": True,         # Cosine learning rate schedule
        "augment": True,        # Enable data augmentation
        "mixup": 0.1,           # Mixup augmentation
        "mosaic": 0.5,          # Mosaic augmentation
        "copy_paste": 0.1,      # Copy-paste augmentation
        "save_period": 10,      # Save every 10 epochs
        "val": True,            # Enable validation
        "plots": True,          # Generate training plots
        "verbose": True         # Verbose output
    }
    
    print("📊 Training Parameters:")
    print(f"   data: {augmented_data_yaml}")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    # Train the model - CORRECT: pass data_yaml_path as FIRST positional argument
    print("\n🏋️ Starting training...")
    results = detector.train(augmented_data_yaml, **training_params)
    
    print("✅ Training complete!")
    print(f"📈 Results saved to: {results.save_dir}")
    
    # Validate the model
    print("\n📊 Running validation...")
    val_results = detector.validate(augmented_data_yaml)
    
    print("✅ Validation complete!")
    
    # Handle validation results (could be dict or empty)
    if val_results and isinstance(val_results, dict):
        # Try to extract mAP metrics from the results dict
        map50 = val_results.get('metrics/mAP50(B)', val_results.get('mAP50', 'N/A'))
        map50_95 = val_results.get('metrics/mAP50-95(B)', val_results.get('mAP50-95', 'N/A'))
        
        print(f"📈 mAP50: {map50}")
        print(f"📈 mAP50-95: {map50_95}")
        
        # Print all available metrics for debugging
        print("\n📊 Available validation metrics:")
        for key, value in val_results.items():
            if 'mAP' in str(key) or 'precision' in str(key) or 'recall' in str(key):
                print(f"   {key}: {value}")
    else:
        print("⚠️  No validation metrics available")
    
    return results


if __name__ == "__main__":
    train_improved_model()
