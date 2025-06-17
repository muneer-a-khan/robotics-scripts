#!/usr/bin/env python3
"""
DSC Loss Training with YOLOv8 Built-in Early Stopping.
Focuses on precision and uses built-in patience mechanism.
"""

import os
import time
import json
from pathlib import Path
from ultralytics import YOLO


def train_dsc_enhanced_builtin():
    """Train with enhanced parameters and built-in early stopping."""
    print("🚀 DSC Loss Training with Enhanced Built-in Early Stopping")
    print("="*60)
    
    # Check for dataset
    data_yaml = "data/augmented_training/data.yaml"
    if not os.path.exists(data_yaml):
        print("❌ Dataset not found!")
        return
    
    # Initialize model
    print("🔧 Initializing YOLOv8x for precision-focused training...")
    model = YOLO('yolov8x.pt')
    
    # Enhanced training parameters focused on precision and early stopping
    training_params = {
        "epochs": 150,              # Max epochs
        "imgsz": 640,
        "batch": 12,               # Smaller batch for stability
        "lr0": 0.0003,             # Lower learning rate for precision
        "lrf": 0.005,              # Lower final LR
        "momentum": 0.95,          # Higher momentum for stability
        "weight_decay": 0.001,     # Increased regularization
        "warmup_epochs": 5,        # More warmup
        "device": "cuda",
        "optimizer": "AdamW",      # AdamW for better convergence
        "cos_lr": True,            # Cosine LR schedule
        
        # Enhanced loss weighting for bounding box accuracy
        "box": 10.0,               # Very high box loss weight
        "cls": 0.3,                # Lower classification weight
        "dfl": 1.0,                # Standard DFL weight
        
        # Conservative augmentation for precision
        "augment": True,
        "mixup": 0.0,              # Disabled mixup
        "mosaic": 0.2,             # Reduced mosaic
        "copy_paste": 0.0,         # Disabled
        "hsv_h": 0.005,            # Very conservative HSV
        "hsv_s": 0.3,
        "hsv_v": 0.2,
        "degrees": 3.0,            # Minimal rotation
        "translate": 0.03,         # Minimal translation
        "scale": 0.2,              # Minimal scale variation
        "shear": 0.5,              # Minimal shear
        "flipud": 0.0,             # No vertical flip
        "fliplr": 0.5,             # Horizontal flip only
        
        # Built-in early stopping and validation
        "patience": 20,            # Built-in early stopping patience
        "val": True,
        "save_period": 3,          # Save every 3 epochs
        "plots": True,
        "verbose": True,
        "exist_ok": True,
        "project": "snap_circuit_training",
        "name": "dsc_enhanced_builtin",
        
        # Enhanced validation monitoring
        "single_cls": False,
        "rect": False,             # No rectangular training
        "resume": False,
        "amp": True,               # Mixed precision
        "fraction": 1.0,           # Use full dataset
        "profile": False,
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.0,
        # "val_freq": 1,           # Not valid in this YOLOv8 version
    }
    
    print("📊 Enhanced Training Configuration:")
    print(f"   🎯 Early Stopping Patience: {training_params['patience']} epochs")
    print(f"   📈 Validation: Every epoch")
    print(f"   🎓 Learning Rate: {training_params['lr0']} → {training_params['lrf']}")
    print(f"   📦 Batch Size: {training_params['batch']}")
    print(f"   ⚖️  Loss Weights - Box: {training_params['box']}, Cls: {training_params['cls']}")
    print(f"   🔄 Focus: Maximum precision + early stopping")
    
    try:
        print(f"\n🏋️ Starting Enhanced Training...")
        start_time = time.time()
        
        results = model.train(
            data=data_yaml,
            **training_params
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        print("✅ Training Complete!")
        print(f"⏱️  Duration: {duration/3600:.2f} hours")
        print(f"📁 Results: {results.save_dir}")
        
        # Final validation
        print("\n📊 Final Validation...")
        val_results = model.val(data=data_yaml, imgsz=640)
        
        print(f"📈 Final Validation Metrics:")
        if hasattr(val_results, 'box'):
            print(f"   mAP50: {getattr(val_results.box, 'map50', 'N/A'):.4f}")
            print(f"   mAP50-95: {getattr(val_results.box, 'map', 'N/A'):.4f}")
            print(f"   Precision: {getattr(val_results.box, 'mp', 'N/A'):.4f}")
            print(f"   Recall: {getattr(val_results.box, 'mr', 'N/A'):.4f}")
        
        # Copy best model
        best_model_path = results.save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            import shutil
            models_dir = Path("models/weights")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = models_dir / "dsc_enhanced_builtin.pt"
            shutil.copy(best_model_path, target_path)
            print(f"📁 Best model copied: {target_path}")
        
        # Enhanced evaluation
        print("\n🔍 Running Enhanced Evaluation...")
        if best_model_path.exists():
            # Import evaluation tools
            from evaluate_model import ModelEvaluator
            
            # Evaluate the model
            evaluator = ModelEvaluator(str(best_model_path), data_yaml)
            metrics = evaluator.evaluate_dataset('val')
            evaluator.print_results(metrics)
            
            # Save results
            enhanced_results_path = "output/dsc_enhanced_builtin_results.json"
            evaluator.save_results(metrics, enhanced_results_path)
            
            # Compare with baseline
            print(f"\n📊 Comparison with Baseline:")
            compare_with_baseline(enhanced_results_path)
            
        return results
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_with_baseline(new_results_path):
    """Compare new model with baseline results."""
    baseline_path = "output/evaluation_results.json"
    
    if not os.path.exists(baseline_path):
        print("⚠️  Baseline results not available")
        return
    
    try:
        import json
        
        with open(new_results_path, 'r') as f:
            new_results = json.load(f)
        
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        print("📊 Performance Comparison:")
        print(f"   {'Metric':<20} {'Baseline':<12} {'New Model':<12} {'Change':<12}")
        print("-" * 60)
        
        # Compare key metrics
        metrics_to_compare = [
            ('rmse_pixels', 'lower_better'),
            ('mae_pixels', 'lower_better'), 
            ('precision', 'higher_better'),
            ('recall', 'higher_better'),
            ('f1_score', 'higher_better')
        ]
        
        for metric, direction in metrics_to_compare:
            baseline_val = baseline['overall_metrics'][metric]
            new_val = new_results['overall_metrics'][metric]
            
            if direction == 'lower_better':
                improvement = ((baseline_val - new_val) / baseline_val * 100) if baseline_val > 0 else 0
                symbol = "📉" if improvement > 0 else "📈"
            else:
                improvement = ((new_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                symbol = "📈" if improvement > 0 else "📉"
            
            print(f"   {metric.replace('_', ' ').title():<20} {baseline_val:<12.3f} {new_val:<12.3f} {symbol}{abs(improvement):<11.1f}%")
        
        # Detection counts
        baseline_matched = baseline['detection_counts']['matched_predictions']
        new_matched = new_results['detection_counts']['matched_predictions']
        baseline_total = baseline['detection_counts']['total_predictions']
        new_total = new_results['detection_counts']['total_predictions']
        
        print(f"\n📊 Detection Counts:")
        print(f"   Matched detections: {baseline_matched} → {new_matched}")
        print(f"   Total predictions: {baseline_total} → {new_total}")
        
    except Exception as e:
        print(f"❌ Comparison error: {e}")


if __name__ == "__main__":
    results = train_dsc_enhanced_builtin()
    
    if results:
        print("\n🎉 Enhanced DSC Training Complete!")
        print("🏆 Check results above for performance improvements!")
    else:
        print("❌ Training failed - check error messages") 