#!/usr/bin/env python3
"""
Smart Early Stopping Controller for running training processes.
Monitors current training and provides early stopping recommendations.
"""

import os
import time
import json
import pandas as pd
from pathlib import Path
import psutil
import signal


class EarlyStoppingController:
    """Monitor and control early stopping for running training."""
    
    def __init__(self, patience=25, min_delta=0.005, monitor_metric='mAP50'):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor_metric = monitor_metric
        self.monitoring = False
        
    def analyze_training_progress(self, training_dir):
        """Analyze current training progress and determine if early stopping should trigger."""
        results_file = training_dir / "results.csv"
        
        if not results_file.exists():
            return None, "No results file found"
        
        try:
            df = pd.read_csv(results_file)
            if len(df) < 5:
                return None, "Insufficient data for analysis"
            
            # Get current metrics
            latest = df.iloc[-1]
            current_epoch = int(latest.get('epoch', 0))
            current_map50 = latest.get('metrics/mAP50(B)', 0)
            
            # Find best epoch
            best_epoch_idx = df['metrics/mAP50(B)'].idxmax()
            best_epoch_data = df.loc[best_epoch_idx]
            best_epoch = int(best_epoch_data['epoch'])
            best_map50 = best_epoch_data['metrics/mAP50(B)']
            
            epochs_since_best = current_epoch - best_epoch
            
            # Analyze recent trend (last 10 epochs)
            recent_epochs = min(10, len(df))
            recent_df = df.tail(recent_epochs)
            recent_map50_values = recent_df['metrics/mAP50(B)'].values
            
            # Calculate trend
            if len(recent_map50_values) >= 5:
                trend_slope = (recent_map50_values[-1] - recent_map50_values[-5]) / 5
                is_improving = trend_slope > self.min_delta / 5
            else:
                is_improving = current_map50 > (best_map50 - self.min_delta)
            
            # Early stopping decision
            should_stop = epochs_since_best >= self.patience and not is_improving
            
            analysis = {
                'current_epoch': current_epoch,
                'current_map50': current_map50,
                'best_epoch': best_epoch,
                'best_map50': best_map50,
                'epochs_since_best': epochs_since_best,
                'is_improving': is_improving,
                'should_stop': should_stop,
                'patience': self.patience,
                'trend_slope': trend_slope if 'trend_slope' in locals() else 0
            }
            
            return analysis, None
            
        except Exception as e:
            return None, f"Error analyzing progress: {e}"
    
    def get_training_process(self):
        """Find the current training process."""
        try:
            for p in psutil.process_iter(['pid', 'name', 'cmdline']):
                if (p.info['name'] == 'python.exe' and 
                    p.info['cmdline'] and 
                    any('train' in cmd.lower() for cmd in p.info['cmdline'])):
                    return p
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return None
    
    def send_stop_signal(self, process):
        """Send graceful stop signal to training process."""
        try:
            # Try graceful termination first
            process.terminate()
            
            # Wait up to 30 seconds for graceful shutdown
            try:
                process.wait(timeout=30)
                return True, "Process terminated gracefully"
            except psutil.TimeoutExpired:
                # Force kill if necessary
                process.kill()
                return True, "Process force killed"
                
        except Exception as e:
            return False, f"Error stopping process: {e}"
    
    def monitor_and_control(self, auto_stop=False):
        """Monitor training and optionally apply early stopping."""
        print("🎯 Early Stopping Controller Started")
        print("="*50)
        
        while True:
            try:
                # Find training directories and process
                training_dirs = list(Path("snap_circuit_training").glob("*/"))
                if not training_dirs:
                    print("❌ No training directories found")
                    break
                
                current_dir = max(training_dirs, key=lambda p: p.stat().st_mtime)
                process = self.get_training_process()
                
                if not process:
                    print("⚠️  No training process found")
                    break
                
                # Analyze progress
                analysis, error = self.analyze_training_progress(current_dir)
                
                if error:
                    print(f"❌ Analysis error: {error}")
                    time.sleep(30)
                    continue
                
                # Display current status
                print(f"\n📊 Training Analysis (Epoch {analysis['current_epoch']}):")
                print(f"   📈 Current mAP50: {analysis['current_map50']:.4f}")
                print(f"   🏆 Best mAP50: {analysis['best_map50']:.4f} (Epoch {analysis['best_epoch']})")
                print(f"   ⏳ Epochs since best: {analysis['epochs_since_best']}/{self.patience}")
                print(f"   📈 Trend: {'Improving' if analysis['is_improving'] else 'Plateauing'}")
                
                # Early stopping decision
                if analysis['should_stop']:
                    print(f"\n🛑 EARLY STOPPING TRIGGERED!")
                    print(f"   Reason: No improvement for {analysis['epochs_since_best']} epochs")
                    print(f"   Best model: Epoch {analysis['best_epoch']} (mAP50: {analysis['best_map50']:.4f})")
                    
                    if auto_stop:
                        print("   Stopping training automatically...")
                        success, message = self.send_stop_signal(process)
                        print(f"   Result: {message}")
                        break
                    else:
                        response = input("   Stop training now? (y/N): ").lower()
                        if response == 'y':
                            success, message = self.send_stop_signal(process)
                            print(f"   Result: {message}")
                            break
                
                elif analysis['epochs_since_best'] > self.patience // 2:
                    print(f"   ⚡ Watch zone: {analysis['epochs_since_best']} epochs without improvement")
                else:
                    print(f"   ✅ Training healthy - continuing")
                
                # Save monitoring data
                monitor_data = {
                    'timestamp': time.time(),
                    'analysis': analysis
                }
                
                monitor_file = current_dir / "early_stopping_monitor.json"
                with open(monitor_file, 'w') as f:
                    json.dump(monitor_data, f, indent=2)
                
                # Wait before next check
                print(f"\n⏱️  Next check in 60 seconds...")
                time.sleep(60)
                
            except KeyboardInterrupt:
                print("\n👋 Monitoring stopped by user")
                break
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                time.sleep(30)
    
    def quick_recommendation(self):
        """Provide quick early stopping recommendation."""
        training_dirs = list(Path("snap_circuit_training").glob("*/"))
        if not training_dirs:
            print("❌ No training directories found")
            return
        
        current_dir = max(training_dirs, key=lambda p: p.stat().st_mtime)
        analysis, error = self.analyze_training_progress(current_dir)
        
        if error:
            print(f"❌ {error}")
            return
        
        print("🎯 Early Stopping Analysis")
        print("="*30)
        print(f"📊 Current Status:")
        print(f"   Epoch: {analysis['current_epoch']}")
        print(f"   mAP50: {analysis['current_map50']:.4f}")
        print(f"   Best: {analysis['best_map50']:.4f} (Epoch {analysis['best_epoch']})")
        print(f"   Since best: {analysis['epochs_since_best']} epochs")
        
        print(f"\n🎯 Recommendation:")
        if analysis['should_stop']:
            print("   🛑 STOP TRAINING - Early stopping criteria met")
            print(f"   📈 Best model at epoch {analysis['best_epoch']}")
        elif analysis['epochs_since_best'] > self.patience // 2:
            print("   ⚡ WATCH CLOSELY - Approaching patience limit")
            print(f"   🔄 Continue for {self.patience - analysis['epochs_since_best']} more epochs")
        else:
            print("   ✅ CONTINUE TRAINING - Model still improving")
            print("   📈 Training is healthy")


def main():
    """Main function for early stopping controller."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Early Stopping Controller for Snap Circuit Training')
    parser.add_argument('--patience', type=int, default=25,
                       help='Patience for early stopping (default: 25)')
    parser.add_argument('--min-delta', type=float, default=0.005,
                       help='Minimum improvement threshold (default: 0.005)')
    parser.add_argument('--monitor', action='store_true',
                       help='Start continuous monitoring')
    parser.add_argument('--auto-stop', action='store_true',
                       help='Automatically stop training when criteria met')
    
    args = parser.parse_args()
    
    controller = EarlyStoppingController(
        patience=args.patience,
        min_delta=args.min_delta
    )
    
    if args.monitor:
        print(f"🎯 Starting Early Stopping Monitor")
        print(f"   Patience: {args.patience} epochs")
        print(f"   Min Delta: {args.min_delta}")
        print(f"   Auto Stop: {'Yes' if args.auto_stop else 'No'}")
        controller.monitor_and_control(auto_stop=args.auto_stop)
    else:
        controller.quick_recommendation()


if __name__ == "__main__":
    main() 