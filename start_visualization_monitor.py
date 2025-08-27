#!/usr/bin/env python3
"""
Start the Live Circuit Visualization Monitor

This script starts the monitoring system that automatically creates
live circuit visualizations from graph JSON files in the output/data/ folder.

Usage:
    python start_visualization_monitor.py

The monitor will:
1. Process existing graph files
2. Watch for new graph files
3. Create live circuit visualizations with validation
4. Save them as live_circuit_visual_*.png files
"""

import subprocess
import sys
import time
from pathlib import Path

def main():
    """Start the visualization monitor."""
    print("🚀 Starting Live Circuit Visualization Monitor")
    print("=" * 50)
    print("This will:")
    print("✅ Process existing graph files")
    print("👀 Watch for new graph files")
    print("🎨 Create live circuit visualizations")
    print("📊 Include validation data")
    print("=" * 50)
    
    # Check if the monitor script exists
    monitor_script = Path("live_visualization_monitor.py")
    if not monitor_script.exists():
        print("❌ Error: live_visualization_monitor.py not found")
        return
    
    try:
        # Start the monitor with process-existing flag
        print("🔄 Starting monitor...")
        process = subprocess.Popen([
            sys.executable, "live_visualization_monitor.py", "--process-existing"
        ])
        
        print("✅ Monitor started successfully!")
        print("📁 Watching: output/data/")
        print("🎨 Visualizations will be saved to: output/")
        print("🔄 Press Ctrl+C to stop the monitor")
        
        # Keep the script running
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitor...")
            process.terminate()
            process.wait()
            print("✅ Monitor stopped")
    
    except Exception as e:
        print(f"❌ Error starting monitor: {e}")

if __name__ == "__main__":
    main() 