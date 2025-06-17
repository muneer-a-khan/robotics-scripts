#!/usr/bin/env python3
"""
Precision-Focused Training: High Recall + Better Precision
Focus on reducing false positives while maintaining excellent recall.
"""

import os
import time
import json
from pathlib import Path
from ultralytics import YOLO


def train_precision_focused():
    """Train with focus on precision while maintaining high recall."""
    print("🎯 Precision-Focused Training (High Recall + Better Precision)")
    print("="*65)
    
    # Check for dataset
    data_yaml = "data/augmented_training/data.yaml"
    if not os.path.exists(data_yaml):
        print("❌ Dataset not found!")
        return
    
    # Initialize model
    print("🔧 Initializing YOLOv8x for precision-focused training...")
    model = YOLO('yolov8x.pt')
    
    # Precision-focused parameters
    training_params = {
        "epochs": 120,              # Sufficient epochs
        "imgsz": 640,
        "batch": 8,                # Smaller batch for more stable gradients
        "lr0": 0.0001,             # Very low learning rate for fine-tuning
        "lrf": 0.001,              # Low final LR
        "momentum": 0.95,          # High momentum for stability
        "weight_decay": 0.002,     # Higher regularization to reduce overfitting
        "warmup_epochs": 8,        # More warmup for stability
        "device": "cuda",
        "optimizer": "AdamW",      # AdamW for better convergence
        "cos_lr": True,            # Cosine LR schedule
        
        # CRITICAL: Loss weighting for precision
        "box": 15.0,               # Very high box loss weight for tight boxes
        "cls": 2.0,                # Higher classification weight for precision
        "dfl": 0.8,                # Lower DFL weight
        
        # Conservative augmentation to reduce false positives
        "augment": True,
        "mixup": 0.0,              # No mixup - can create false positives
        "mosaic": 0.1,             # Very low mosaic
        "copy_paste": 0.0,         # Disabled
        "hsv_h": 0.002,            # Minimal HSV changes
        "hsv_s": 0.2,
        "hsv_v": 0.1,
        "degrees": 1.0,            # Minimal rotation
        "translate": 0.01,         # Minimal translation
        "scale": 0.1,              # Minimal scale variation
        "shear": 0.2,              # Minimal shear
        "flipud": 0.0,             # No vertical flip
        "fliplr": 0.3,             # Reduced horizontal flip
        
        # Precision-focused validation settings
        "patience": 25,            # More patience for fine-tuning
        "val": True,
        "save_period": 5,
        "plots": True,
        "verbose": True,
        "exist_ok": True,
        "project": "snap_circuit_training",
        "name": "precision_focused",
        
        # Additional precision settings
        "single_cls": False,
        "rect": False,
        "resume": False,
        "amp": True,
        "fraction": 1.0,
        "profile": False,
        "overlap_mask": True,
        "mask_ratio": 4,
        "dropout": 0.1,            # Add dropout for regularization
        
        # NMS settings for precision (will be applied during inference)
        "iou": 0.5,                # Lower IoU threshold for NMS
        "conf": 0.25,              # Higher confidence threshold
    }
    
    print("📊 Precision-Focused Configuration:")
    print(f"   🎯 Strategy: Reduce false positives, maintain high recall")
    print(f"   📈 Early Stopping Patience: {training_params['patience']} epochs")
    print(f"   🎓 Learning Rate: {training_params['lr0']} → {training_params['lrf']}")
    print(f"   📦 Batch Size: {training_params['batch']}")
    print(f"   ⚖️  Loss Weights - Box: {training_params['box']}, Cls: {training_params['cls']}")
    print(f"   🔧 Confidence Threshold: {training_params['conf']}")
    print(f"   🔄 Focus: Tight bounding boxes + fewer false positives")
    
    try:
        print(f"\n🏋️ Starting Precision-Focused Training...")
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
        
        # Final validation with different confidence thresholds
        print("\n📊 Multi-Threshold Validation...")
        best_model_path = results.save_dir / "weights" / "best.pt"
        
        if best_model_path.exists():
            # Test different confidence thresholds
            thresholds = [0.1, 0.25, 0.4, 0.5, 0.6]
            print(f"Testing confidence thresholds: {thresholds}")
            
            for conf_thresh in thresholds:
                print(f"\n🔍 Confidence Threshold: {conf_thresh}")
                val_results = model.val(data=data_yaml, imgsz=640, conf=conf_thresh)
                
                if hasattr(val_results, 'box'):
                    precision = getattr(val_results.box, 'mp', 0)
                    recall = getattr(val_results.box, 'mr', 0)
                    map50 = getattr(val_results.box, 'map50', 0)
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    print(f"   Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, mAP50: {map50:.3f}")
        
        # Copy best model
        if best_model_path.exists():
            import shutil
            models_dir = Path("models/weights")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            target_path = models_dir / "precision_focused.pt"
            shutil.copy(best_model_path, target_path)
            print(f"📁 Best model copied: {target_path}")
        
        # Enhanced evaluation with optimal confidence
        print("\n🔍 Running Precision-Focused Evaluation...")
        if best_model_path.exists():
            # Find optimal confidence threshold first
            optimal_conf = find_optimal_confidence(str(best_model_path), data_yaml)
            print(f"🎯 Optimal confidence threshold: {optimal_conf}")
            
            # Evaluate with optimal confidence
            from evaluate_model import ModelEvaluator
            
            # Create evaluator with custom confidence
            evaluator = ModelEvaluator(str(best_model_path), data_yaml, conf_threshold=optimal_conf)
            metrics = evaluator.evaluate_dataset('val')
            evaluator.print_results(metrics)
            
            # Save results
            results_path = "output/precision_focused_results.json"
            evaluator.save_results(metrics, results_path)
            
            # Compare with baseline
            print(f"\n📊 Comparison with Baseline:")
            compare_with_baseline(results_path)
            
        return results
        
    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_optimal_confidence(model_path, data_yaml):
    """Find optimal confidence threshold that balances precision and recall."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    best_f1 = 0
    best_conf = 0.25
    
    # Test confidence thresholds
    for conf in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        try:
            results = model.val(data=data_yaml, conf=conf, verbose=False)
            if hasattr(results, 'box'):
                precision = getattr(results.box, 'mp', 0)
                recall = getattr(results.box, 'mr', 0)
                
                if precision > 0 and recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    
                    # Prefer high recall (weight recall more)
                    # But also want reasonable precision (>40%)
                    if recall > 0.85 and precision > 0.4:
                        weighted_score = 0.3 * precision + 0.7 * recall
                        if weighted_score > best_f1:
                            best_f1 = weighted_score
                            best_conf = conf
        except:
            continue
    
    return best_conf


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
        print(f"   Precision ratio: {new_matched/new_total:.3f} vs {baseline_matched/baseline_total:.3f}")
        
    except Exception as e:
        print(f"❌ Comparison error: {e}")


if __name__ == "__main__":
    results = train_precision_focused()
    
    if results:
        print("\n🎉 Precision-Focused Training Complete!")
        print("🏆 Model optimized for high recall + better precision!")
    else:
        print("❌ Training failed - check error messages") 