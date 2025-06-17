#!/usr/bin/env python3
"""
DSC Loss Training with Proper Early Stopping and Validation-Based Model Selection.
Focuses on both detection accuracy AND bounding box precision.
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch


class ValidationBasedEarlyStopping:
    """Enhanced early stopping that considers both mAP and bounding box accuracy."""
    
    def __init__(self, patience=20, min_delta=0.001, validation_freq=5):
        self.patience = patience
        self.min_delta = min_delta
        self.validation_freq = validation_freq
        self.best_score = -float('inf')
        self.best_epoch = 0
        self.wait = 0
        self.best_metrics = {}
        self.validation_history = []
        
    def calculate_validation_score(self, metrics):
        """Calculate combined validation score considering multiple factors."""
        # Extract key metrics
        map50 = getattr(metrics.box, 'map50', 0)
        precision = getattr(metrics.box, 'mp', 0)  
        recall = getattr(metrics.box, 'mr', 0)
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Combined score prioritizing precision and F1 for bounding box accuracy
        # Weight: 40% mAP50, 35% precision, 25% F1
        combined_score = 0.4 * map50 + 0.35 * precision + 0.25 * f1
        
        return {
            'combined_score': combined_score,
            'map50': map50,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def should_stop(self, trainer):
        """Determine if training should stop based on validation metrics."""
        if not hasattr(trainer, 'metrics') or trainer.metrics is None:
            return False
            
        current_epoch = getattr(trainer, 'epoch', 0)
        
        # Only validate every N epochs to save time
        if current_epoch % self.validation_freq != 0 and current_epoch > 10:
            return False
            
        # Calculate validation score
        val_metrics = self.calculate_validation_score(trainer.metrics)
        self.validation_history.append({
            'epoch': current_epoch,
            **val_metrics
        })
        
        current_score = val_metrics['combined_score']
        
        print(f"\n📊 Validation Metrics (Epoch {current_epoch}):")
        print(f"   🎯 Combined Score: {current_score:.4f}")
        print(f"   📈 mAP50: {val_metrics['map50']:.4f}")
        print(f"   🎯 Precision: {val_metrics['precision']:.4f}")
        print(f"   🔍 Recall: {val_metrics['recall']:.4f}")
        print(f"   ⚖️  F1-Score: {val_metrics['f1']:.4f}")
        
        # Check for improvement
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.best_epoch = current_epoch
            self.best_metrics = val_metrics.copy()
            self.wait = 0
            print(f"   ✅ New best combined score: {current_score:.4f}")
            return False
        else:
            self.wait += 1
            epochs_since_best = current_epoch - self.best_epoch
            print(f"   ⏳ No improvement for {epochs_since_best} epochs ({self.wait} validation checks)")
            
            if self.wait >= self.patience:
                print(f"\n🛑 Early stopping triggered!")
                print(f"   Best epoch: {self.best_epoch}")
                print(f"   Best score: {self.best_score:.4f}")
                print(f"   Best mAP50: {self.best_metrics.get('map50', 0):.4f}")
                print(f"   Best precision: {self.best_metrics.get('precision', 0):.4f}")
                return True
                
        return False


def train_dsc_with_proper_early_stopping():
    """Train DSC loss model with proper early stopping based on validation metrics."""
    print("🚀 DSC Loss Training with Validation-Based Early Stopping")
    print("="*65)
    
    # Check for dataset
    data_yaml = "data/augmented_training/data.yaml"
    if not os.path.exists(data_yaml):
        print("❌ Dataset not found!")
        return
    
    # Initialize model
    print("🔧 Initializing YOLOv8 with Enhanced Early Stopping...")
    model = YOLO('yolov8x.pt')
    
    # Initialize early stopping
    early_stopping = ValidationBasedEarlyStopping(
        patience=15,        # Stop after 15 validation checks without improvement
        min_delta=0.002,    # Minimum improvement threshold
        validation_freq=3   # Validate every 3 epochs
    )
    
    # Custom callback for early stopping
    def on_val_end(validator):
        # Get the trainer from the validator
        trainer = validator.trainer if hasattr(validator, 'trainer') else None
        if trainer and early_stopping.should_stop(trainer):
            trainer.stop = True
    
    # Add callback
    model.add_callback("on_val_end", on_val_end)
    
    # Enhanced training parameters focused on precision
    training_params = {
        "epochs": 100,              # Reduced max epochs
        "imgsz": 640,
        "batch": 16,               # Increased batch size for stability
        "lr0": 0.0005,             # Lower learning rate for precision
        "lrf": 0.01,               # Final LR factor
        "momentum": 0.937,
        "weight_decay": 0.0008,    # Increased regularization
        "warmup_epochs": 3,
        "device": "cuda",
        "optimizer": "AdamW",       # AdamW for better convergence
        "cos_lr": True,
        
        # Loss weighting for better bounding box accuracy
        "box": 8.0,                # Increased box loss weight
        "cls": 0.5,                # Standard classification weight  
        "dfl": 1.2,                # Slightly reduced DFL weight
        
        # Conservative augmentation for better precision
        "augment": True,
        "mixup": 0.05,             # Reduced mixup
        "mosaic": 0.3,             # Reduced mosaic
        "copy_paste": 0.0,         # Disabled copy-paste
        "hsv_h": 0.01,             # Reduced HSV augmentation
        "hsv_s": 0.5,
        "hsv_v": 0.3,
        "degrees": 5.0,            # Reduced rotation
        "translate": 0.05,         # Reduced translation
        "scale": 0.3,              # Reduced scale variation
        "shear": 1.0,              # Reduced shear
        "flipud": 0.0,             # No vertical flip
        "fliplr": 0.5,             # Horizontal flip only
        
        # Validation and saving
        "val": True,
        "save_period": 5,          # Save every 5 epochs
        "plots": True,
        "verbose": True,
        "exist_ok": True,
        "project": "snap_circuit_training",
        "name": "dsc_proper_early_stopping"
    }
    
    print("📊 Enhanced Training Configuration:")
    print(f"   🎯 Early Stopping Patience: {early_stopping.patience} validation checks")
    print(f"   📈 Validation Frequency: Every {early_stopping.validation_freq} epochs")
    print(f"   🎓 Learning Rate: {training_params['lr0']}")
    print(f"   📦 Batch Size: {training_params['batch']}")
    print(f"   ⚖️  Loss Weights - Box: {training_params['box']}, Cls: {training_params['cls']}")
    print(f"   🔄 Focus: Precision + mAP balance")
    
    try:
        print(f"\n🏋️ Starting Enhanced DSC Training...")
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
        
        if early_stopping.best_epoch > 0:
            print(f"🏆 Best model at epoch {early_stopping.best_epoch}")
            print(f"📊 Best metrics:")
            for key, value in early_stopping.best_metrics.items():
                print(f"   {key}: {value:.4f}")
        
        # Final validation
        print("\n📊 Final Validation...")
        val_results = model.val(data=data_yaml, imgsz=640)
        
        # Save validation history
        history_file = results.save_dir / "validation_history.json"
        with open(history_file, 'w') as f:
            json.dump(early_stopping.validation_history, f, indent=2)
        print(f"📁 Validation history saved: {history_file}")
        
        # Copy best model
        best_model_path = results.save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            import shutil
            models_dir = Path("models/weights")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = models_dir / "dsc_proper_early_stopping.pt"
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
            enhanced_results_path = "output/dsc_proper_early_stopping_results.json"
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
    results = train_dsc_with_proper_early_stopping()
    
    if results:
        print("\n🎉 Enhanced DSC Training Complete!")
        print("🏆 Check results above for performance improvements!")
    else:
        print("❌ Training failed - check error messages") 