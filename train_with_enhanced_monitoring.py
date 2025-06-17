#!/usr/bin/env python3
"""
Enhanced training script with comprehensive monitoring and early stopping.
Focuses on validation accuracy improvements with DSC-inspired loss modifications.
"""

import os
import time
import json
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.callbacks import default_callbacks
import torch
import numpy as np


class TrainingMonitor:
    """Enhanced training monitor with early stopping and comprehensive logging."""
    
    def __init__(self, patience=20, min_delta=0.001, monitor_metric='mAP50'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_metric = monitor_metric
        self.best_score = -float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        self.history = {
            'epochs': [],
            'train_loss': [],
            'val_map50': [],
            'val_map50_95': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'box_loss': [],
            'cls_loss': [],
            'dfl_loss': []
        }
    
    def on_train_epoch_end(self, trainer):
        """Callback after each training epoch."""
        current_epoch = trainer.epoch
        
        # Get training metrics
        if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
            box_loss = trainer.loss_items[0] if len(trainer.loss_items) > 0 else 0
            cls_loss = trainer.loss_items[1] if len(trainer.loss_items) > 1 else 0  
            dfl_loss = trainer.loss_items[2] if len(trainer.loss_items) > 2 else 0
        else:
            box_loss = cls_loss = dfl_loss = 0
        
        # Store training metrics
        self.history['epochs'].append(current_epoch)
        self.history['box_loss'].append(float(box_loss))
        self.history['cls_loss'].append(float(cls_loss))
        self.history['dfl_loss'].append(float(dfl_loss))
        
        print(f"📊 Epoch {current_epoch} Training Loss Components:")
        print(f"   🎯 Box Loss: {box_loss:.4f}")
        print(f"   🔍 Classification Loss: {cls_loss:.4f}")  
        print(f"   📐 DFL Loss: {dfl_loss:.4f}")
    
    def on_val_end(self, trainer):
        """Callback after validation."""
        if not hasattr(trainer, 'metrics') or trainer.metrics is None:
            return
            
        current_epoch = trainer.epoch
        
        # Extract validation metrics
        metrics = trainer.metrics
        val_map50 = float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0.0
        val_map50_95 = float(metrics.box.map) if hasattr(metrics.box, 'map') else 0.0
        val_precision = float(metrics.box.mp) if hasattr(metrics.box, 'mp') else 0.0
        val_recall = float(metrics.box.mr) if hasattr(metrics.box, 'mr') else 0.0
        
        # Calculate F1 score
        val_f1 = 2 * val_precision * val_recall / (val_precision + val_recall) if (val_precision + val_recall) > 0 else 0.0
        
        # Store validation metrics
        self.history['val_map50'].append(val_map50)
        self.history['val_map50_95'].append(val_map50_95)
        self.history['val_precision'].append(val_precision)
        self.history['val_recall'].append(val_recall)
        self.history['val_f1'].append(val_f1)
        
        print(f"\n📈 Epoch {current_epoch} Validation Metrics:")
        print(f"   🎯 mAP50: {val_map50:.4f}")
        print(f"   📊 mAP50-95: {val_map50_95:.4f}")
        print(f"   🔍 Precision: {val_precision:.4f}")
        print(f"   📋 Recall: {val_recall:.4f}")
        print(f"   ⚖️  F1-Score: {val_f1:.4f}")
        
        # Early stopping logic
        current_score = val_map50 if self.monitor_metric == 'mAP50' else val_f1
        
        if current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.wait = 0
            print(f"   ✅ New best {self.monitor_metric}: {current_score:.4f}")
        else:
            self.wait += 1
            print(f"   ⏳ No improvement for {self.wait}/{self.patience} epochs")
            
        if self.wait >= self.patience:
            self.stopped_epoch = current_epoch
            print(f"\n🛑 Early stopping triggered at epoch {current_epoch}")
            print(f"   Best {self.monitor_metric}: {self.best_score:.4f}")
            trainer.stop = True
    
    def save_history(self, save_path):
        """Save training history to JSON."""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"📁 Training history saved to: {save_path}")


def train_with_enhanced_monitoring():
    """Train model with comprehensive monitoring and early stopping."""
    print("🚀 Enhanced Snap Circuit Training with Monitoring & Early Stopping")
    print("="*70)
    
    # Check for augmented dataset
    augmented_data_yaml = "data/augmented_training/data.yaml"
    
    if not os.path.exists(augmented_data_yaml):
        print("❌ Augmented dataset not found!")
        print("   Please ensure the dataset exists at:", augmented_data_yaml)
        return
    
    # Initialize model
    print("🔧 Initializing YOLOv8 with Enhanced Monitoring...")
    model = YOLO('yolov8x.pt')
    
    # Initialize training monitor
    monitor = TrainingMonitor(
        patience=25,           # Wait 25 epochs for improvement
        min_delta=0.005,       # Minimum improvement threshold
        monitor_metric='mAP50' # Monitor mAP50 for early stopping
    )
    
    # Add custom callbacks
    model.add_callback("on_train_epoch_end", monitor.on_train_epoch_end)
    model.add_callback("on_val_end", monitor.on_val_end)
    
    # Enhanced training parameters optimized for bounding box accuracy
    training_params = {
        "epochs": 150,              # Max epochs
        "imgsz": 640,              # Image size
        "batch": 12,               # Batch size for RTX 4070 Ti
        "lr0": 0.0008,             # Lower learning rate for stability
        "lrf": 0.005,              # Final learning rate
        "momentum": 0.937,         # SGD momentum
        "weight_decay": 0.0005,    # L2 regularization
        "warmup_epochs": 3,        # Warmup epochs
        "warmup_momentum": 0.8,    # Warmup momentum
        "warmup_bias_lr": 0.1,     # Warmup bias learning rate
        "device": "cuda",          # GPU training
        "optimizer": "SGD",        # SGD often better for object detection
        "cos_lr": True,           # Cosine learning rate scheduler
        
        # Box loss weighting (focus on bounding box accuracy)
        "box": 7.5,               # Increased box loss weight (default: 7.5)
        "cls": 0.5,               # Classification loss weight  
        "dfl": 1.5,               # Distribution focal loss weight
        
        # Data augmentation (moderate for stable training)
        "augment": True,
        "mixup": 0.1,             # Mixup probability
        "mosaic": 0.5,            # Mosaic probability  
        "copy_paste": 0.1,        # Copy-paste probability
        "hsv_h": 0.015,           # HSV hue augmentation
        "hsv_s": 0.7,             # HSV saturation
        "hsv_v": 0.4,             # HSV value
        "degrees": 8.0,           # Rotation degrees
        "translate": 0.1,         # Translation fraction
        "scale": 0.5,             # Image scale variation
        "shear": 2.0,             # Shear degrees
        "perspective": 0.0,       # Perspective transform
        "flipud": 0.0,            # Vertical flip probability
        "fliplr": 0.5,            # Horizontal flip probability
        
        # Training settings
        "close_mosaic": 10,       # Disable mosaic in last N epochs
        "save_period": 10,        # Save every 10 epochs
        "val": True,              # Enable validation
        "plots": True,            # Generate training plots
        "verbose": True,          # Verbose output
        "exist_ok": True,         # Allow overwriting
        "project": "snap_circuit_training",
        "name": "enhanced_monitoring_model"
    }
    
    print("📊 Enhanced Training Configuration:")
    print(f"   📁 Dataset: {augmented_data_yaml}")
    print(f"   🎯 Early Stopping Patience: {monitor.patience} epochs")
    print(f"   📈 Monitor Metric: {monitor.monitor_metric}")
    print(f"   📦 Batch Size: {training_params['batch']}")
    print(f"   🎓 Learning Rate: {training_params['lr0']}")
    print(f"   🔄 Optimizer: {training_params['optimizer']}")
    print(f"   ⚖️  Loss Weights - Box: {training_params['box']}, Cls: {training_params['cls']}, DFL: {training_params['dfl']}")
    
    try:
        # Start training
        print(f"\n🏋️ Starting Enhanced Training...")
        print(f"   Focus: Bounding box accuracy with early stopping")
        print(f"   Expected Duration: ~2-4 hours (depending on early stopping)")
        
        start_time = time.time()
        
        results = model.train(
            data=augmented_data_yaml,
            **training_params
        )
        
        end_time = time.time()
        training_duration = end_time - start_time
        
        print("✅ Enhanced Training Complete!")
        print(f"⏱️  Total Training Time: {training_duration/3600:.2f} hours")
        print(f"📁 Results saved to: {results.save_dir}")
        
        if monitor.stopped_epoch > 0:
            print(f"🛑 Early stopping at epoch {monitor.stopped_epoch}")
            print(f"🏆 Best {monitor.monitor_metric}: {monitor.best_score:.4f}")
        
        # Save training history
        history_path = results.save_dir / "training_history.json"
        monitor.save_history(history_path)
        
        # Validate the final model
        print("\n📊 Final Model Validation...")
        val_results = model.val(data=augmented_data_yaml, imgsz=640)
        
        if hasattr(val_results, 'box'):
            metrics = val_results.box
            print(f"\n🏆 Final Validation Results:")
            print(f"   📊 mAP50: {metrics.map50:.4f}")
            print(f"   📊 mAP50-95: {metrics.map:.4f}")
            print(f"   🎯 Precision: {metrics.mp:.4f}")
            print(f"   🔍 Recall: {metrics.mr:.4f}")
        
        # Copy best model
        best_model_path = results.save_dir / "weights" / "best.pt"
        if best_model_path.exists():
            import shutil
            models_dir = Path("models/weights")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = models_dir / "snap_circuit_enhanced.pt"
            shutil.copy(best_model_path, target_path)
            print(f"📁 Best model copied to: {target_path}")
        
        # Run comprehensive evaluation
        print("\n🔍 Running Comprehensive Evaluation...")
        if best_model_path.exists():
            from evaluate_model import ModelEvaluator
            
            evaluator = ModelEvaluator(str(best_model_path), augmented_data_yaml)
            metrics = evaluator.evaluate_dataset('val')
            evaluator.print_results(metrics)
            
            # Save enhanced results
            enhanced_results_path = "output/enhanced_model_evaluation.json"
            evaluator.save_results(metrics, enhanced_results_path)
            
            print(f"\n📊 Enhanced Model Performance Summary:")
            print(f"   📈 RMSE: {metrics.rmse_pixels:.2f} pixels")
            print(f"   📈 MAE: {metrics.mae_pixels:.2f} pixels")  
            print(f"   🎯 Precision: {metrics.precision:.3f}")
            print(f"   🔍 Recall: {metrics.recall:.3f}")
            print(f"   ⚖️  F1-Score: {metrics.f1_score:.3f}")
            print(f"   📁 Results: {enhanced_results_path}")
            
            # Compare with original if available
            compare_with_baseline(enhanced_results_path)
        
        return results
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        
        # Save partial history if available
        if hasattr(monitor, 'history') and monitor.history['epochs']:
            partial_history_path = "output/partial_training_history.json"
            monitor.save_history(partial_history_path)
            print(f"📁 Partial training history saved: {partial_history_path}")
        
        return None


def compare_with_baseline(enhanced_results_path):
    """Compare enhanced model with baseline results."""
    print(f"\n🔍 Comparing Enhanced Model with Baseline...")
    print("="*50)
    
    baseline_path = "output/evaluation_results.json"
    
    if os.path.exists(baseline_path):
        import json
        
        with open(enhanced_results_path, 'r') as f:
            enhanced = json.load(f)
        
        with open(baseline_path, 'r') as f:
            baseline = json.load(f)
        
        print("📊 Performance Comparison:")
        print(f"   {'Metric':<20} {'Baseline':<12} {'Enhanced':<12} {'Improvement':<15}")
        print("-" * 65)
        
        # Compare bounding box accuracy
        for metric in ['rmse_pixels', 'mae_pixels']:
            baseline_val = baseline['overall_metrics'][metric]
            enhanced_val = enhanced['overall_metrics'][metric] 
            improvement = ((baseline_val - enhanced_val) / baseline_val * 100) if baseline_val > 0 else 0
            
            print(f"   {metric.replace('_', ' ').title():<20} {baseline_val:<12.2f} {enhanced_val:<12.2f} {improvement:<14.1f}%")
        
        # Compare detection performance  
        for metric in ['precision', 'recall', 'f1_score']:
            baseline_val = baseline['overall_metrics'][metric]
            enhanced_val = enhanced['overall_metrics'][metric]
            improvement = ((enhanced_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            
            print(f"   {metric.replace('_', ' ').title():<20} {baseline_val:<12.3f} {enhanced_val:<12.3f} {improvement:<14.1f}%")
        
        # Detection counts
        baseline_matched = baseline['detection_counts']['matched_predictions']
        enhanced_matched = enhanced['detection_counts']['matched_predictions']
        detection_improvement = enhanced_matched - baseline_matched
        
        print(f"   {'Matched Detections':<20} {baseline_matched:<12} {enhanced_matched:<12} +{detection_improvement:<14}")
        
    else:
        print("⚠️  Baseline results not found - run baseline evaluation first")


def monitor_current_training():
    """Monitor the currently running training process."""
    print("👁️  Monitoring Current Training Process...")
    
    # Look for latest training run
    training_dirs = list(Path("snap_circuit_training").glob("*/"))
    if training_dirs:
        latest_dir = max(training_dirs, key=lambda p: p.stat().st_mtime)
        results_file = latest_dir / "results.csv"
        
        if results_file.exists():
            print(f"📊 Latest training: {latest_dir.name}")
            
            # Read and display recent metrics
            try:
                import pandas as pd
                df = pd.read_csv(results_file)
                
                if len(df) > 0:
                    latest = df.iloc[-1]
                    print(f"   Epoch: {int(latest.get('epoch', 0))}")
                    print(f"   mAP50: {latest.get('metrics/mAP50(B)', 0):.4f}")
                    print(f"   Precision: {latest.get('metrics/precision(B)', 0):.4f}")
                    print(f"   Recall: {latest.get('metrics/recall(B)', 0):.4f}")
                    
            except Exception as e:
                print(f"   Could not read results: {e}")
    else:
        print("   No training directories found")


if __name__ == "__main__":
    # Check if training is already running
    import psutil
    python_processes = [p for p in psutil.process_iter(['pid', 'name', 'cmdline']) 
                       if p.info['name'] == 'python.exe' and 
                       any('train_with_dsc_loss.py' in ' '.join(p.info['cmdline']) 
                           for _ in [None] if p.info['cmdline'])]
    
    if python_processes:
        print("🔄 Training already in progress!")
        monitor_current_training()
        
        response = input("\n❓ Start new enhanced training? (y/N): ").lower()
        if response != 'y':
            print("👋 Monitoring existing training...")
            exit(0)
    
    # Start enhanced training
    results = train_with_enhanced_monitoring()
    
    if results:
        print("\n🎉 Enhanced Training Pipeline Complete!")
        print("🏆 Check the comparison results above for improvements!")
    else:
        print("❌ Training encountered issues - check logs above") 