# Snap Circuit Computer Vision System 🔧🤖

A comprehensive computer vision pipeline for analyzing Snap Circuit boards using YOLOv8 object detection, OpenCV connection detection, and NetworkX graph analysis with TD-BKT integration.

## 🧠 System Overview

This system provides real-time analysis of Snap Circuit boards through:

- **Component Detection**: YOLOv8-based identification of circuit pieces (LEDs, batteries, wires, etc.)
- **Connection Analysis**: OpenCV-based wire tracing and component connection detection  
- **Circuit State Inference**: NetworkX graph analysis to determine if circuits are closed, powered, and functional
- **Live Circuit Visualization**: Automatic generation of beautiful circuit board visualizations in real-time
- **Graph Format Output**: TD-BKT compatible graph format for downstream algorithms
- **Real-time Processing**: Live camera feed analysis with annotated output

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenCV-compatible camera (for real-time mode)
- CUDA-compatible GPU (recommended for faster inference)

### Installation

1. **Clone and setup environment:**
```bash
git clone <repository-url>
cd robotics-research
pip install -r requirements.txt
```

2. **Test basic functionality:**
```bash
python main.py --mode camera --no-save
```

## 📁 Project Structure (Cleaned & Organized)

```
robotics-research/
├── main.py                    # Main application entry point
├── config.py                  # Configuration settings
├── data_structures.py         # Core data classes
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── snap_circuit_image.jpg     # Test image
├── train_improved_model.py    # Model training script
│
├── models/                    # AI Models
│   ├── component_detector.py  # YOLOv8 component detection
│   └── weights/               # Trained model files
│       └── snap_circuit_yolov8.pt
│
├── vision/                    # Computer Vision
│   └── connection_detector.py # OpenCV connection detection  
│
├── circuit/                   # Circuit Analysis
│   └── graph_builder.py      # NetworkX circuit analysis
│
├── live_circuit_visualizer.py # Live visualization generator
│
├── utils/                     # Utilities (Less frequently used)
│   ├── setup_improved_training.py
│   ├── create_individual_component_dataset.py
│   ├── expand_training_dataset.py
│   ├── data_preparation.py
│   └── COMPONENT_PHOTO_GUIDE.md
│
├── data/                      # Training & Test Data
│   ├── individual_components/ # Component photos
│   ├── augmented_training/    # Augmented dataset
│   └── expanded_training/     # Expanded dataset
│
├── output/                    # System Outputs
│   ├── frames/               # Annotated video frames
│   ├── data/                 # JSON outputs
│   │   ├── detection_*.json  # Traditional format
│   │   └── graph_*.json      # TD-BKT graph format
│   ├── live_circuit_visual_*.png  # Timestamped visualizations
│   └── latest_circuit_visual.png  # Most recent visualization
│
├── snap_circuit_training/     # Training Results
│   └── expanded_snap_circuit_model/
│       └── weights/
│           └── best.pt
│
└── Graph Format Integration   # NEW: TD-BKT Integration
    ├── circuit_graph_format.py    # Graph data structures
    └── graph_output_converter.py  # Detection → Graph converter
```

## 🎯 Usage

### Real-time Camera Analysis

```bash
# Basic real-time detection (outputs JSON + live visualizations)
python main.py --mode camera

# Use specific camera and custom model
python main.py --mode camera --camera 1 --model models/weights/snap_circuit_yolov8.pt

# Disable display for headless operation
python main.py --mode camera --no-display

# The system will automatically:
# ✅ Process frames every 3 seconds
# ✅ Generate live circuit visualizations (PNG files)
# ✅ Save detection data (JSON files)
# ✅ Display real-time annotated camera feed
```

**Real-time Controls:**
- `q`: Quit
- `s`: Save current frame
- `p`: Pause/resume detection
- `+/-`: Adjust processing interval

### 🎨 Live Circuit Visualization

The system automatically generates beautiful circuit board visualizations in real-time:

**🔄 Automatic Generation**: 
- Processes camera feed every 3 seconds
- Creates PNG visualizations automatically
- No manual intervention required

**📸 Visual Features**:
- **Hexagonal grid background** mimicking actual snap circuit boards
- **Color-coded components** with distinct colors for each type:
  - Battery holders (blue), LEDs (pink), switches (teal)
  - Buttons (green), wires (gold), motors (purple)
  - Resistors (red), speakers (magenta), and 10+ other types
- **Connection lines** showing component relationships
- **Component labels** with confidence scores
- **Circuit status** indicating if circuit is complete and powered
- **Timestamp information** for each visualization

**📁 Output Files**:
```bash
output/live_circuit_visual_[timestamp].png  # Timestamped versions
output/latest_circuit_visual.png           # Always the most recent
```

**🎯 Features**:
- Real-time digital twin of your snap circuit board
- Intelligent connection detection between nearby components
- High-confidence component filtering (>75% confidence)
- Actual layout based on camera-detected positions

### Video File Processing

```bash
# Process video file
python main.py --mode video --input video.mp4 --output annotated_video.mp4
```

### Single Image Analysis

```bash
# Analyze single image
python main.py --mode image --input circuit_image.jpg
```

## 📊 Dual Output Format

The system now outputs **both** traditional JSON and graph format:

### Traditional Format (`detection_*.json`)
```json
{
  "connection_graph": {
    "components": [...],
    "edges": [...],
    "state": {...}
  },
  "processing_time": 0.045
}
```

### Graph Format (`graph_*.json`) - TD-BKT Compatible
```json
{
  "graph": {
    "directed": true,
    "nodes": [
      {
        "id": "battery_holder_0",
        "component_type": "battery_holder",
        "position": {"x": 200, "y": 300},
        "accessibility": 0.95,
        "placement_correctness": 0.9
      }
    ],
    "edges": [...],
    "circuit_analysis": {
      "connectivity": {...},
      "circuit_complete": false,
      "power_sources": [...]
    }
  }
}
```

## 🏋️ Training Your Own Model

### 1. Individual Component Dataset
```bash
# Create individual component dataset
python utils/create_individual_component_dataset.py

# Follow photo guide
cat utils/COMPONENT_PHOTO_GUIDE.md
```

### 2. Train Improved Model
```bash
python train_improved_model.py
```

### 3. Use Trained Model
```bash
python main.py --mode camera --model snap_circuit_training/expanded_snap_circuit_model/weights/best.pt
```

## 🔧 Component Classes

The system detects these Snap Circuit components:

- **Power**: `battery_holder`
- **Outputs**: `led`, `speaker`, `motor`, `lamp`, `fan`, `buzzer`, `alarm`
- **Inputs**: `button`, `switch`, `photoresistor`, `microphone`
- **Passive**: `resistor`, `wire`, `connection_node`
- **Complex**: `music_circuit`

## ⚙️ Configuration

Key settings in `config.py`:

```python
# Performance optimized for real-time
VIDEO_CONFIG = {
    "processing_interval": 3.0,  # Process every 3 seconds (optimized for visualization)
    "resolution": [1920, 1080],
    "fps": 30
}

# YOLO Detection
YOLO_CONFIG = {
    "confidence_threshold": 0.5,
    "device": "cuda"  # Uses GPU if available
}

# Live Visualization
VISUALIZATION_CONFIG = {
    "confidence_threshold": 0.75,  # Only show high-confidence components
    "connection_tolerance": 35,     # Pixel distance for connection detection
    "save_latest": True            # Always save latest_circuit_visual.png
}
```

## 🎯 Integration with TD-BKT Systems

The graph format outputs are designed for TD-BKT (Temporal Difference - Bayesian Knowledge Tracing) algorithms:

- **Spatial positioning** for robot guidance
- **Component accessibility** scores
- **Circuit connectivity** analysis
- **No built-in recommendations** (handled by external TD-BKT system)

## 🧹 Recent Cleanup

**Removed obsolete files:**
- Old test scripts (`test_*.py`)
- Demo scripts (`demo_*.py`) 
- Summary scripts (`final_results_summary.py`)
- Duplicate training scripts
- Unused model files (`yolo11n.pt`, `yolov8n.pt`, `yolov8x.pt`)
- Old detection data files

**Organized structure:**
- Core system files in root
- Utilities moved to `utils/` directory
- Clear separation of concerns

## 📈 Performance

- **Real-time detection**: ~1-2 seconds per frame
- **GPU accelerated**: 10-20x faster training
- **Optimized connection detection**: Limited path processing for performance
- **Dual output**: Traditional + graph format without performance impact

## 🤝 Contributing

1. The system is production-ready with clean codebase
2. Graph format integration complete for TD-BKT consumption
3. Training pipeline optimized for individual component photos
4. Real-time performance optimized for practical use

---

**System Status**: ✅ Production Ready | Graph Integration Complete | Live Visualization Active | Performance Optimized
