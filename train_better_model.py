#!/usr/bin/env python3
"""
Train a better model using improved annotations that label complete components.
"""

import os
from pathlib import Path
from models.component_detector import ComponentDetector


def train_better_model():
    """Train model with corrected component-level annotations."""
    print("🚀 TRAINING BETTER MODEL WITH CORRECTED ANNOTATIONS")
    print("="*60)
    
    # Check for improved dataset
    improved_data_yaml = "data/improved_training/data.yaml"
    
    if not os.path.exists(improved_data_yaml):
        print("❌ Improved dataset not found!")
        print("   Run: python create_better_training_data.py first")
        return
    
    # Initialize detector
    detector = ComponentDetector()
    
    # Optimized training parameters for better model
    training_params = {
        "epochs": 300,          # More epochs for learning complete components
        "imgsz": 640,           # Standard YOLO image size
        "batch": 8,             # Smaller batch for precision training
        "lr0": 0.0005,          # Lower learning rate for careful learning
        "patience": 100,        # More patience for convergence
        "device": "cuda",       # GPU training
        "optimizer": "AdamW",   # Best optimizer for this task
        "cos_lr": True,         # Cosine learning rate schedule
        
        # Data augmentation (reduced to prevent confusion)
        "augment": True,
        "mixup": 0.0,           # Disable mixup (can confuse component boundaries)
        "mosaic": 0.2,          # Light mosaic
        "copy_paste": 0.0,      # Disable copy-paste
        "translate": 0.1,       # Light translation
        "scale": 0.2,           # Light scaling
        "flipud": 0.0,          # No vertical flips (circuits have orientation)
        "fliplr": 0.5,          # Allow horizontal flips
        
        # Loss weights (emphasize precision)
        "cls": 1.5,             # Higher classification weight
        "box": 1.0,             # Standard box regression weight
        "dfl": 1.0,             # Distribution focal loss weight
        
        # Confidence and IoU
        "conf": 0.25,           # Detection confidence threshold
        "iou": 0.5,             # NMS IoU threshold
        
        # Validation
        "save_period": 25,      # Save every 25 epochs
        "val": True,            # Enable validation
        "plots": True,          # Generate training plots
        "verbose": True,        # Verbose output
        
        # Early stopping based on precision (not mAP)
        "patience": 100,        # Wait longer for improvements
    }
    
    print("📊 Training Parameters (Optimized for Component Detection):")
    print(f"   Dataset: {improved_data_yaml}")
    print("   Strategy: Component-level detection (not individual parts)")
    print("   Focus: High precision to avoid false positives")
    
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    # Train the model
    print("\n🏋️ Starting better model training...")
    print("   Expected improvements:")
    print("   • No more individual connection point detections")
    print("   • Unified component recognition")
    print("   • Fewer false positives")
    print("   • Better component boundaries")
    
    results = detector.train(improved_data_yaml, **training_params)
    
    print("✅ Training complete!")
    print(f"📈 Results saved to: {results.save_dir}")
    
    # Save the best model with a meaningful name
    best_model_path = Path(results.save_dir) / "weights" / "best.pt"
    if best_model_path.exists():
        better_model_path = Path("models/weights/better_component_detector.pt")
        better_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        import shutil
        shutil.copy2(best_model_path, better_model_path)
        print(f"✅ Saved best model to: {better_model_path}")
    
    # Validate the model
    print("\n📊 Running validation...")
    val_results = detector.validate(improved_data_yaml)
    
    if val_results and isinstance(val_results, dict):
        # Extract relevant metrics
        precision = val_results.get('metrics/precision(B)', val_results.get('precision', 'N/A'))
        recall = val_results.get('metrics/recall(B)', val_results.get('recall', 'N/A'))
        map50 = val_results.get('metrics/mAP50(B)', val_results.get('mAP50', 'N/A'))
        
        print(f"📈 Precision: {precision}")
        print(f"📈 Recall: {recall}")
        print(f"📈 mAP50: {map50}")
        
        print("\n🎯 EXPECTED IMPROVEMENTS:")
        if isinstance(precision, (int, float)) and precision > 0.85:
            print("✅ High precision - fewer false positives!")
        if isinstance(recall, (int, float)) and recall > 0.80:
            print("✅ Good recall - catches real components!")
        
    else:
        print("⚠️  No validation metrics available")
    
    print("\n🎯 NEXT STEPS:")
    print("1. Test with: python test_better_model.py")
    print("2. Update config.py to use: better_component_detector.pt")
    print("3. Compare results to previous models")
    
    return results


if __name__ == "__main__":
    train_better_model() 