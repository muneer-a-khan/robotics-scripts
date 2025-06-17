#!/usr/bin/env python3
"""
Real-time training monitoring script for DSC loss training.
Tracks progress, metrics, and provides early stopping recommendations.
"""

import os
import time
import json
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import psutil


def find_current_training():
    """Find the currently running training process and directory."""
    training_dirs = []
    
    # Check for training directories
    if Path("snap_circuit_training").exists():
        training_dirs = list(Path("snap_circuit_training").glob("*/"))
        training_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    # Check for running Python processes
    python_processes = []
    try:
        for p in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info']):
            if (p.info['name'] == 'python.exe' and 
                p.info['cmdline'] and 
                any('train' in cmd.lower() for cmd in p.info['cmdline'])):
                python_processes.append(p.info)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    
    return training_dirs, python_processes


def read_training_metrics(training_dir):
    """Read current training metrics from results files."""
    results_file = training_dir / "results.csv"
    
    if not results_file.exists():
        return None
    
    try:
        df = pd.read_csv(results_file)
        return df
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        return None


def analyze_progress(df):
    """Analyze training progress and provide insights."""
    if df is None or len(df) == 0:
        return {}
    
    latest = df.iloc[-1]
    current_epoch = int(latest.get('epoch', 0))
    
    # Get key metrics
    metrics = {
        'current_epoch': current_epoch,
        'total_epochs': len(df),
        'map50': latest.get('metrics/mAP50(B)', 0),
        'map50_95': latest.get('metrics/mAP50-95(B)', 0),
        'precision': latest.get('metrics/precision(B)', 0),
        'recall': latest.get('metrics/recall(B)', 0),
        'box_loss': latest.get('train/box_loss', 0),
        'cls_loss': latest.get('train/cls_loss', 0),
        'dfl_loss': latest.get('train/dfl_loss', 0)
    }
    
    # Calculate trends (last 5 epochs)
    if len(df) >= 5:
        recent_df = df.tail(5)
        
        # mAP50 trend
        map50_values = recent_df['metrics/mAP50(B)'].values
        map50_trend = 'improving' if map50_values[-1] > map50_values[0] else 'declining'
        
        # Loss trend
        box_loss_values = recent_df['train/box_loss'].values
        loss_trend = 'improving' if box_loss_values[-1] < box_loss_values[0] else 'declining'
        
        metrics['map50_trend'] = map50_trend
        metrics['loss_trend'] = loss_trend
        metrics['map50_change'] = map50_values[-1] - map50_values[0]
        metrics['loss_change'] = box_loss_values[-1] - box_loss_values[0]
    
    # Find best epoch
    if 'metrics/mAP50(B)' in df.columns:
        best_epoch_idx = df['metrics/mAP50(B)'].idxmax()
        best_epoch = df.loc[best_epoch_idx]
        metrics['best_epoch'] = int(best_epoch['epoch'])
        metrics['best_map50'] = best_epoch['metrics/mAP50(B)']
        metrics['epochs_since_best'] = current_epoch - metrics['best_epoch']
    
    return metrics


def display_progress(training_dir, metrics, processes):
    """Display current training progress."""
    print("="*70)
    print("🔍 SNAP CIRCUIT TRAINING MONITOR")
    print("="*70)
    
    print(f"\n📁 Training Directory: {training_dir.name}")
    print(f"⏰ Last Modified: {time.ctime(training_dir.stat().st_mtime)}")
    
    # Process information
    if processes:
        for proc in processes:
            memory_info = proc.get('memory_info', {})
            if hasattr(memory_info, 'rss'):
                memory_mb = memory_info.rss / (1024*1024)
            else:
                memory_mb = 0
            print(f"🖥️  Process: PID {proc['pid']}, Memory: {memory_mb:.0f}MB")
    
    if not metrics:
        print("❌ No training metrics available yet")
        return
    
    # Current status
    print(f"\n📊 CURRENT STATUS:")
    print(f"   🎯 Epoch: {metrics['current_epoch']}")
    print(f"   📈 mAP50: {metrics['map50']:.4f}")
    print(f"   📊 mAP50-95: {metrics['map50_95']:.4f}")
    print(f"   🎯 Precision: {metrics['precision']:.4f}")
    print(f"   🔍 Recall: {metrics['recall']:.4f}")
    
    # Loss information
    print(f"\n🔥 CURRENT LOSSES:")
    print(f"   📦 Box Loss: {metrics['box_loss']:.4f}")
    print(f"   🏷️  Cls Loss: {metrics['cls_loss']:.4f}")
    print(f"   📐 DFL Loss: {metrics['dfl_loss']:.4f}")
    
    # Best performance
    if 'best_epoch' in metrics:
        print(f"\n🏆 BEST PERFORMANCE:")
        print(f"   🎖️  Best mAP50: {metrics['best_map50']:.4f} (Epoch {metrics['best_epoch']})")
        print(f"   ⏳ Epochs since best: {metrics['epochs_since_best']}")
        
        # Early stopping recommendation
        if metrics['epochs_since_best'] > 20:
            print(f"   ⚠️  Consider early stopping (no improvement for {metrics['epochs_since_best']} epochs)")
        elif metrics['epochs_since_best'] > 10:
            print(f"   ⚡ Watch closely (no improvement for {metrics['epochs_since_best']} epochs)")
    
    # Trend analysis
    if 'map50_trend' in metrics:
        print(f"\n📈 RECENT TRENDS (last 5 epochs):")
        trend_emoji = "📈" if metrics['map50_trend'] == 'improving' else "📉"
        print(f"   {trend_emoji} mAP50: {metrics['map50_trend']} ({metrics['map50_change']:+.4f})")
        
        loss_emoji = "📉" if metrics['loss_trend'] == 'improving' else "📈"
        print(f"   {loss_emoji} Box Loss: {metrics['loss_trend']} ({metrics['loss_change']:+.4f})")


def create_progress_plot(training_dir, df):
    """Create a real-time progress plot."""
    if df is None or len(df) < 2:
        return
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = df['epoch']
        
        # mAP metrics
        ax1.plot(epochs, df['metrics/mAP50(B)'], 'b-', label='mAP50', linewidth=2)
        ax1.plot(epochs, df['metrics/mAP50-95(B)'], 'r-', label='mAP50-95', linewidth=2)
        ax1.set_title('Validation mAP', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('mAP')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Precision & Recall
        ax2.plot(epochs, df['metrics/precision(B)'], 'g-', label='Precision', linewidth=2)
        ax2.plot(epochs, df['metrics/recall(B)'], 'orange', label='Recall', linewidth=2)
        ax2.set_title('Precision & Recall', fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Training Losses
        ax3.plot(epochs, df['train/box_loss'], 'b-', label='Box Loss', linewidth=2)
        ax3.plot(epochs, df['train/cls_loss'], 'r-', label='Cls Loss', linewidth=2)
        ax3.set_title('Training Losses', fontweight='bold')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning Rate
        if 'lr/pg0' in df.columns:
            ax4.plot(epochs, df['lr/pg0'], 'purple', linewidth=2)
            ax4.set_title('Learning Rate', fontweight='bold')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Learning Rate\nData Not Available', 
                    ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = training_dir / "progress_monitor.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Progress plot saved: {plot_path}")
        
        # Show plot briefly then close
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        
    except Exception as e:
        print(f"Error creating plot: {e}")


def monitor_loop():
    """Main monitoring loop."""
    print("🚀 Starting Training Monitor...")
    
    try:
        while True:
            # Clear screen (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')
            
            # Find current training
            training_dirs, processes = find_current_training()
            
            if not training_dirs:
                print("❌ No training directories found")
                print("   Start training first with: python train_with_dsc_loss.py")
                break
            
            # Use most recent training directory
            current_dir = training_dirs[0]
            
            # Read metrics
            df = read_training_metrics(current_dir)
            metrics = analyze_progress(df)
            
            # Display progress
            display_progress(current_dir, metrics, processes)
            
            # Create progress plot every 10 seconds
            if int(time.time()) % 30 == 0:  # Every 30 seconds
                create_progress_plot(current_dir, df)
            
            # Check if training is still running
            if not processes:
                print("\n⚠️  No training processes detected")
                response = input("Continue monitoring? (y/N): ").lower()
                if response != 'y':
                    break
            
            # Wait before next update
            print(f"\n⏱️  Next update in 10 seconds... (Ctrl+C to exit)")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")


def quick_status():
    """Show quick status without continuous monitoring."""
    training_dirs, processes = find_current_training()
    
    if not training_dirs:
        print("❌ No training directories found")
        return
    
    current_dir = training_dirs[0]
    df = read_training_metrics(current_dir)
    metrics = analyze_progress(df)
    
    display_progress(current_dir, metrics, processes)
    
    if df is not None:
        create_progress_plot(current_dir, df)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor Snap Circuit training progress')
    parser.add_argument('--loop', action='store_true', 
                       help='Continuous monitoring (default: single status check)')
    parser.add_argument('--plot', action='store_true',
                       help='Generate progress plot')
    
    args = parser.parse_args()
    
    if args.loop:
        monitor_loop()
    else:
        quick_status()
        
        if args.plot:
            training_dirs, _ = find_current_training()
            if training_dirs:
                df = read_training_metrics(training_dirs[0])
                create_progress_plot(training_dirs[0], df) 