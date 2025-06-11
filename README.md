# Snap Circuit Computer Vision System 🔧🤖

A comprehensive computer vision pipeline for analyzing Snap Circuit boards using YOLOv8 object detection, OpenCV connection detection, and NetworkX graph analysis.

## 🧠 System Overview

This system provides real-time analysis of Snap Circuit boards through:

- **Component Detection**: YOLOv8-based identification of circuit pieces (LEDs, batteries, wires, etc.)
- **Connection Analysis**: OpenCV-based wire tracing and component connection detection  
- **Circuit State Inference**: NetworkX graph analysis to determine if circuits are closed, powered, and functional
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
cd robotics-scripts
pip install -r requirements.txt
```

2. **Test basic functionality:**
```bash
python main.py --mode camera --no-save
```

## 📁 Project Structure

```
snap-circuit-vision/
├── config.py                 # Configuration settings
├── data_structures.py        # Core data classes
├── main.py                   # Main application
├── requirements.txt          # Dependencies
├── models/
│   └── component_detector.py # YOLOv8 component detection
├── vision/
│   └── connection_detector.py # OpenCV connection detection  
├── circuit/
│   └── graph_builder.py     # NetworkX circuit analysis
├── training/
│   └── data_preparation.py  # Dataset preparation tools
├── data/                    # Training data
├── models/weights/          # Trained model files
└── output/                  # Detection outputs
    ├── frames/             # Annotated video frames
    └── data/               # JSON detection results
```

## 🎯 Usage

### Real-time Camera Analysis

```bash
# Basic real-time detection
python main.py --mode camera

# Use specific camera and custom model
python main.py --mode camera --camera 1 --model path/to/model.pt

# Disable display for headless operation
python main.py --mode camera --no-display
```

**Controls:**
- `q`: Quit
- `s`: Save current frame
- `p`: Pause/resume detection

### Video File Processing

```bash
# Process video file
python main.py --mode video --input video.mp4 --output annotated_video.mp4

# Process without saving output video
python main.py --mode video --input video.mp4 --no-save
```

### Single Image Analysis

```bash
# Analyze single image
python main.py --mode image --input circuit_image.jpg
```

## 🏋️ Training Your Own Model

### 1. Prepare Training Data

```bash
# Create sample dataset structure
python training/data_preparation.py --create-samples

# Organize your images and annotations
python training/data_preparation.py \
    --images /path/to/images \
    --annotations /path/to/annotations \
    --format yolo \
    --train-split 0.7 \
    --val-split 0.2 \
    --test-split 0.1

# Validate prepared dataset
python training/data_preparation.py --validate
```

### 2. Train YOLOv8 Model

```python
from models.component_detector import ComponentDetector

detector = ComponentDetector()
detector.train('data/training/data.yaml', epochs=100)
```

Or use command line:
```bash
yolo train data=data/training/data.yaml model=yolov8x.pt epochs=100 imgsz=640
```

### 3. Use Trained Model

```bash
python main.py --mode camera --model models/weights/snap_circuit_yolov8.pt
```

## 🔧 Component Classes

The system can detect these Snap Circuit components:

- **Power**: `battery_holder`
- **Outputs**: `led`, `speaker`, `motor`, `lamp`, `fan`, `buzzer`, `alarm`
- **Inputs**: `button`, `switch`, `photoresistor`, `microphone`
- **Passive**: `resistor`, `wire`, `connection_node`
- **Complex**: `music_circuit`

## 📊 Output Format

### JSON Detection Results
```json
{
  "connection_graph": {
    "components": [
      {
        "id": "led-1",
        "label": "led",
        "bbox": [100, 100, 150, 150],
        "orientation": 0,
        "confidence": 0.95,
        "component_type": "led"
      }
    ],
    "edges": [
      {
        "component_1": "battery-1",
        "component_2": "led-1",
        "connection_type": "wire",
        "confidence": 0.8
      }
    ],
    "state": {
      "is_circuit_closed": true,
      "power_on": true,
      "active_components": ["led-1"]
    }
  },
  "processing_time": 0.045
}
```

## ⚙️ Configuration

Edit `config.py` to customize:

- **Component classes** and detection thresholds
- **Video processing** settings (resolution, FPS)
- **Connection detection** parameters (wire color ranges, proximity thresholds)
- **Circuit analysis** rules (power components, output devices)

### Key Configuration Options

```python
# YOLOv8 Settings
YOLO_CONFIG = {
    "confidence_threshold": 0.5,
    "iou_threshold": 0.45,
    "image_size": 640
}

# Connection Detection
CONNECTION_CONFIG = {
    "wire_color_range": {
        "lower": [0, 0, 100],
        "upper": [180, 50, 255]
    },
    "connection_proximity_threshold": 30
}
```

## 🔬 Advanced Features

### Circuit Analysis

The system provides advanced circuit analysis:

```python
from circuit.graph_builder import CircuitGraphBuilder

builder = CircuitGraphBuilder()
graph = builder.build_graph(components, connections, timestamp)

# Check circuit state
print(f"Circuit closed: {graph.state.is_circuit_closed}")
print(f"Active components: {graph.state.active_components}")

# Analyze topology
nx_graph = builder._create_networkx_graph(components, connections)
topology = builder.analyze_graph_topology(nx_graph)

# Find potential errors
errors = builder.find_potential_errors(nx_graph, components)
```

### Custom Component Detection

Extend the system for new components:

```python
# Add to config.py
COMPONENT_CLASSES.append("custom_component")

# Train model with new data
detector = ComponentDetector()
detector.train('data/custom_data.yaml')
```

### Integration with Robotics

The system outputs structured data suitable for robot control:

```python
# Get component coordinates for robot manipulation
for component in graph.components:
    x, y = component.bbox.center
    # Convert to robot coordinates using ROBOT_CONFIG
    robot_x = x * ROBOT_CONFIG["pixel_to_mm_ratio"]
    robot_y = y * ROBOT_CONFIG["pixel_to_mm_ratio"]
```

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**: Download or train a model first
2. **Camera not detected**: Check camera ID and permissions
3. **Poor detection accuracy**: Adjust lighting and camera angle
4. **Slow performance**: Reduce image resolution or use GPU acceleration

### Performance Optimization

```python
# Benchmark performance
detector = ComponentDetector()
metrics = detector.benchmark(test_image)
print(f"FPS: {metrics['fps']:.1f}")

# Use smaller model for speed
detector = ComponentDetector("yolov8n.pt")  # Nano model
```

## 📖 API Reference

### Core Classes

- **`SnapCircuitVisionSystem`**: Main orchestration class
- **`ComponentDetector`**: YOLOv8 object detection wrapper
- **`ConnectionDetector`**: OpenCV connection analysis
- **`CircuitGraphBuilder`**: NetworkX graph construction and analysis

### Data Structures

- **`ComponentDetection`**: Individual component detection
- **`Connection`**: Connection between components  
- **`ConnectionGraph`**: Complete circuit representation
- **`CircuitState`**: Circuit analysis results

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Add tests for new functionality
4. Commit changes (`git commit -m 'Add new feature'`)
5. Push to branch (`git push origin feature/new-feature`)
6. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics YOLOv8** for object detection
- **OpenCV** for computer vision processing
- **NetworkX** for graph analysis
- **Snap Circuits** for inspiring educational electronics

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review configuration options

---

**Ready to analyze some circuits? Start with `python main.py --mode camera` and point your camera at a Snap Circuit board!** 🎯 
