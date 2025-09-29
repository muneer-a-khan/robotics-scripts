#!/usr/bin/env python3
"""
Monitor Photos Model Training Progress

This script monitors the training progress and provides updates
on the dual board model training with your annotated photos.
"""

import time
from pathlib import Path
import json


def monitor_training():
    """Monitor training progress"""
    print("🔍 MONITORING PHOTOS MODEL TRAINING")
    print("=" * 60)
    
    # Training directory
    training_dir = Path("dual_board_training/photos_model_fixed")
    
    if not training_dir.exists():
        print(f"❌ Training directory not found: {training_dir}")
        print("   Training may not have started yet...")
        return
    
    print(f"📂 Training directory: {training_dir}")
    
    # Check for training files
    weights_dir = training_dir / "weights"
    if weights_dir.exists():
        weight_files = list(weights_dir.glob("*.pt"))
        print(f"🏋️  Weight files: {len(weight_files)}")
        
        if weight_files:
            latest_weight = max(weight_files, key=lambda p: p.stat().st_mtime)
            print(f"   Latest: {latest_weight.name}")
    
    # Check for results
    results_file = training_dir / "results.csv"
    if results_file.exists():
        with open(results_file, 'r') as f:
            lines = f.readlines()
        
        if len(lines) > 1:  # Header + data
            print(f"📊 Training epochs completed: {len(lines) - 1}")
            
            # Show last few epochs
            if len(lines) > 5:
                print("   Recent progress:")
                for line in lines[-4:-1]:  # Last 3 epochs
                    parts = line.strip().split(',')
                    if len(parts) >= 4:
                        epoch = parts[0].strip()
                        box_loss = parts[1].strip()
                        cls_loss = parts[2].strip()
                        print(f"     Epoch {epoch}: box_loss={box_loss}, cls_loss={cls_loss}")
    
    # Check logs
    log_file = Path("logs/dual_board_training_photos_model_fixed.log")
    if log_file.exists():
        print(f"📝 Log file: {log_file}")
        print(f"   Size: {log_file.stat().st_size / 1024:.1f} KB")
    
    print(f"\n⏳ Training in progress...")
    print(f"💡 Expected completion: 1-3 hours depending on hardware")
    print(f"🎯 Target: 100 epochs with 15 classes and 1,397 annotations")


def check_completion():
    """Check if training is complete"""
    training_dir = Path("dual_board_training/photos_model_fixed")
    
    if not training_dir.exists():
        return False, None
    
    # Check for best.pt (final model)
    best_model = training_dir / "weights" / "best.pt"
    last_model = training_dir / "weights" / "last.pt"
    
    if best_model.exists() and last_model.exists():
        return True, str(best_model)
    
    return False, None


def main():
    """Main monitoring function"""
    monitor_training()
    
    print(f"\n🚀 NEXT STEPS WHEN TRAINING COMPLETES:")
    print(f"   1. Check model location: dual_board_training/photos_model_fixed/weights/best.pt")
    print(f"   2. Test the model: python test_dual_model.py")
    print(f"   3. Run live detection: python dual_board_live_system.py")
    print(f"\n📈 Monitor with: python monitor_photos_training.py")


if __name__ == "__main__":
    main()
