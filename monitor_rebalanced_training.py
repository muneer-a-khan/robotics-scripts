#!/usr/bin/env python3
"""
Monitor Rebalanced Training Progress

Quick script to check training progress for the rebalanced dual board model.
"""

import os
import time
from pathlib import Path
import subprocess


def monitor_training():
    """Monitor the training progress"""
    
    print("🎯 REBALANCED MODEL TRAINING MONITOR")
    print("=" * 50)
    
    # Check if training is running
    try:
        result = subprocess.run(['pgrep', '-f', 'train_dual_board_model'], 
                               capture_output=True, text=True)
        if result.stdout.strip():
            print("✅ Training process is running!")
            pids = result.stdout.strip().split('\n')
            print(f"📊 Process IDs: {', '.join(pids)}")
        else:
            print("❌ Training process not found")
    except:
        print("⚠️  Could not check process status")
    
    # Check training outputs
    runs_dir = Path("dual_board_training")
    if runs_dir.exists():
        experiments = list(runs_dir.glob("rebalanced_dual_board_model*"))
        if experiments:
            latest_exp = max(experiments, key=lambda x: x.stat().st_mtime)
            print(f"\n📂 Latest experiment: {latest_exp.name}")
            
            # Check for results
            results_file = latest_exp / "results.csv"
            if results_file.exists():
                print(f"📈 Results file exists: {results_file}")
                # Show last few lines
                try:
                    with open(results_file, 'r') as f:
                        lines = f.readlines()
                    if len(lines) > 1:
                        print(f"📊 Current epoch: {len(lines) - 1}")
                        if len(lines) > 2:
                            last_line = lines[-1].strip().split(',')
                            if len(last_line) > 5:
                                epoch = last_line[0]
                                train_loss = last_line[1] 
                                val_loss = last_line[4] if len(last_line) > 4 else "N/A"
                                print(f"   Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}")
                except Exception as e:
                    print(f"⚠️  Error reading results: {e}")
            
            # Check for weights
            weights_dir = latest_exp / "weights"
            if weights_dir.exists():
                weights = list(weights_dir.glob("*.pt"))
                print(f"💾 Weights saved: {len(weights)} files")
                if weights:
                    best_weight = weights_dir / "best.pt"
                    last_weight = weights_dir / "last.pt"
                    if best_weight.exists():
                        print(f"   ✅ best.pt available")
                    if last_weight.exists():
                        print(f"   ✅ last.pt available")
        else:
            print(f"\n📂 No experiments found in {runs_dir}")
    
    # Check GPU usage
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if lines and lines[0]:
                gpu_util, mem_used, mem_total = lines[0].split(', ')
                print(f"\n🎮 GPU Status:")
                print(f"   Utilization: {gpu_util}%")
                print(f"   Memory: {mem_used}MB / {mem_total}MB")
    except:
        print(f"\n🎮 GPU status: nvidia-smi not available")
    
    print(f"\n💡 Training Info:")
    print(f"   🎯 Model: Rebalanced dual board detection")
    print(f"   📊 Dataset: 613 images (55.8% left battery holders)")
    print(f"   🎪 Epochs: 100 (with early stopping)")
    print(f"   📝 Expected time: 30-60 minutes")
    
    print(f"\n🎮 Commands:")
    print(f"   • python monitor_rebalanced_training.py  # Check progress") 
    print(f"   • tail -f dual_board_training/rebalanced_dual_board_model*/train.log  # Live logs")
    print(f"   • pkill -f train_dual_board_model  # Stop training (if needed)")


if __name__ == "__main__":
    monitor_training()
