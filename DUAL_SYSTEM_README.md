# Dual Circuit Board Vision System

This system extends the original single-board detection to handle **two circuit boards simultaneously** using a single high-mounted camera. The camera feed is split into left and right halves, with each half processed independently.

## 🎯 **New Requirements Addressed**

### ✅ **Requirement #1: Single Camera → Dual Boards**
- **Problem**: Need to monitor two circuit boards with one camera
- **Solution**: Camera feed splitting with independent processing pipelines
- **Implementation**: `dual_camera_system.py` splits frame and processes each side separately

### ✅ **Requirement #2: High-Up Camera Optimization**
- **Problem**: Camera positioned high up requires model trained on lower-resolution/zoomed-out images
- **Solution**: Complete training data preparation and model retraining pipeline
- **Implementation**: `high_camera_training_prep.py` with multiple data augmentation strategies

## 🚀 **Quick Start**

### **Option 1: Start Everything (Recommended)**
```bash
python start_dual_system.py --mode all --validate
```

### **Option 2: Start Components Separately**
```bash
# Terminal 1: Start dual camera system
python start_dual_system.py --mode dual-camera --validate

# Terminal 2: Start dual visualization monitor
python start_dual_system.py --mode dual-monitor --process-existing
```

### **Option 3: Prepare Training Data for High Camera**
```bash
# Collect live training data
python start_dual_system.py --mode train-prep --collect-data --duration 15

# Process existing data for high camera
python high_camera_training_prep.py --mode all --source data/existing_training
```

## 📁 **File Structure**

### **New Core Files**
- `dual_camera_system.py` - Main dual board detection system
- `dual_visualization_monitor.py` - Monitor for dual board visualizations
- `high_camera_training_prep.py` - Training data preparation for high camera
- `start_dual_system.py` - Easy launcher for all systems

### **Output Structure**
```
output/
├── data/
│   ├── left/                          # Left board graph files
│   │   ├── graph_left_*.json
│   │   └── detection_left_*.json
│   └── right/                         # Right board graph files
│       ├── graph_right_*.json
│       └── detection_right_*.json
├── frames/
│   ├── left/                          # Left board annotated frames
│   ├── right/                         # Right board annotated frames
│   └── dual_frame_*.jpg               # Full dual frame
├── live_circuit_visual_left_*.png     # Left board visualizations
├── live_circuit_visual_right_*.png    # Right board visualizations
├── latest_circuit_visual_left.png     # Latest left board
└── latest_circuit_visual_right.png    # Latest right board
```

## 🎥 **Dual Camera System Features**

### **Camera Feed Splitting**
- **Split Ratio**: Configurable (default 0.5 = equal halves)
- **Processing**: Parallel processing of both sides using ThreadPoolExecutor
- **Display**: Live view with split line and dual overlays

### **Independent Processing**
- **Detection**: Separate component detection for each side
- **Validation**: Independent circuit validation with scores
- **Graphs**: Separate graph generation and JSON output
- **Visualizations**: Individual live circuit visualizations

### **Usage Examples**
```bash
# Basic dual camera with validation
python dual_camera_system.py --validate

# Custom split ratio (60% left, 40% right)
python dual_camera_system.py --split-ratio 0.6 --validate

# No display mode for headless operation
python dual_camera_system.py --no-display --validate
```

## 👀 **Dual Visualization Monitor**

### **Features**
- **Dual Monitoring**: Watches both `output/data/left/` and `output/data/right/`
- **Side-Specific**: Creates separate visualizations for each board
- **Validation**: Includes validation scores and status
- **Real-Time**: Processes files as they're created

### **Output Files**
- `live_circuit_visual_left_*.png` - Left board visualizations
- `live_circuit_visual_right_*.png` - Right board visualizations
- `latest_circuit_visual_left.png` - Always updated with latest left
- `latest_circuit_visual_right.png` - Always updated with latest right

### **Usage Examples**
```bash
# Start monitoring with existing file processing
python dual_visualization_monitor.py --process-existing

# Monitor only (no validation)
python dual_visualization_monitor.py --no-validation
```

## 🎓 **High Camera Training Preparation**

### **Problem Addressed**
The original model was trained on close-up, high-resolution images. For a high-mounted camera:
- Components appear smaller
- Lower effective resolution
- Different perspective/angle
- Need for dual-board layout training

### **Training Data Strategies**

#### **1. Live Data Collection**
```bash
python high_camera_training_prep.py --mode collect --duration 15
```
- Captures images from high-mounted camera
- Shows dual board layout guides
- Creates base dataset for annotation

#### **2. Resolution Variants**
```bash
python high_camera_training_prep.py --mode variants --source data/existing_training
```
- Creates lower resolution versions: 640x640, 416x416, 320x320
- Maintains YOLO label compatibility
- Simulates distance effect

#### **3. Zoomed-Out Variants**
```bash
python high_camera_training_prep.py --mode zoom --source data/existing_training
```
- Adds padding/background to simulate distance
- Zoom factors: 70%, 50%, 30% of original size
- Adjusts bounding box coordinates accordingly

#### **4. Dual Board Layouts**
```bash
python high_camera_training_prep.py --mode dual --source data/existing_training
```
- Combines two circuit images side-by-side
- Creates training data for dual-board scenarios
- Adjusts labels for left/right positioning

#### **5. Complete Pipeline**
```bash
python high_camera_training_prep.py --mode all --source data/existing_training --epochs 150
```
- Runs all data preparation steps
- Trains optimized model for high camera
- Uses YOLOv8n for better performance at distance

### **Training Configuration**
```python
# Optimized for high-distance detection
model.train(
    conf=0.15,      # Lower confidence threshold
    iou=0.4,        # NMS IoU threshold
    augment=True,   # Enable augmentation
    mosaic=1.0,     # Mosaic augmentation
    mixup=0.1,      # Mixup augmentation
)
```

## 🔧 **Configuration**

### **Split Ratio Adjustment**
```python
# In dual_camera_system.py or via command line
--split-ratio 0.6  # 60% left, 40% right
--split-ratio 0.4  # 40% left, 60% right
```

### **Processing Interval**
```python
# In config.py
VIDEO_CONFIG = {
    "processing_interval": 3.0,  # Process every 3 seconds
}
```

### **Confidence Thresholds**
```python
# For high camera model
YOLO_CONFIG = {
    "confidence_threshold": 0.15,  # Lower for distant detection
}

# For visualization filtering
confidence_threshold = 0.75  # Only show high-confidence components
```

## 📊 **Validation Features**

### **Dual Validation**
- **Independent**: Each board validated separately
- **Scores**: Separate validation scores for left/right
- **Status**: Individual status indicators (✅ ⚠️ ❌)
- **Issues**: Side-specific error reporting

### **Validation Display**
- **Left Board**: Shows in left half of display
- **Right Board**: Shows in right half of display
- **Visualizations**: Validation boxes in top-right of each PNG

## 🚨 **Troubleshooting**

### **No Visualizations Generated**
1. **Check camera detection**: Ensure components are being detected
2. **Check confidence**: Lower confidence threshold if needed
3. **Check file structure**: Ensure `output/data/left/` and `output/data/right/` exist
4. **Check monitor**: Ensure dual visualization monitor is running

### **Poor Detection from High Camera**
1. **Retrain model**: Use `high_camera_training_prep.py` to create training data
2. **Lower confidence**: Adjust `confidence_threshold` in config
3. **Better lighting**: Ensure good lighting conditions
4. **Camera positioning**: Adjust camera angle/height

### **Split Not Aligned**
1. **Adjust split ratio**: Use `--split-ratio` parameter
2. **Camera positioning**: Center camera between boards
3. **Board positioning**: Ensure boards are properly positioned

## 🎯 **Performance Optimization**

### **Parallel Processing**
- **ThreadPoolExecutor**: Processes left/right sides simultaneously
- **Independent pipelines**: No blocking between sides
- **Memory efficient**: Shared model instances

### **High Camera Model**
- **YOLOv8n**: Smaller, faster model for distant detection
- **Lower resolution**: 640x640 input for speed
- **Optimized thresholds**: Tuned for high-distance scenarios

## 📈 **Usage Workflow**

### **1. Initial Setup**
```bash
# Collect training data for your specific setup
python start_dual_system.py --mode train-prep --collect-data --duration 20
```

### **2. Model Training** (if needed)
```bash
# Train model optimized for high camera
python high_camera_training_prep.py --mode train --epochs 150
```

### **3. Run Dual System**
```bash
# Start complete dual system
python start_dual_system.py --mode all --validate
```

### **4. Monitor Results**
- **Live Display**: See dual board detection in real-time
- **Visualizations**: Check `output/live_circuit_visual_left_*.png` and `right_*.png`
- **Validation**: Monitor validation scores for both boards

## 🔄 **Integration with Original System**

### **Backward Compatibility**
- **Original system**: Still works with `python main.py --mode camera`
- **Single board**: Use original system for single board detection
- **Dual board**: Use new dual system for two boards

### **File Compatibility**
- **JSON format**: Same format as original system
- **Visualization**: Same style as original live circuit visualizations
- **Validation**: Same validation system, applied per board

---

## 🎉 **Summary**

The dual circuit board system successfully addresses both requirements:

1. ✅ **Single camera → Dual boards**: Complete splitting and independent processing
2. ✅ **High camera optimization**: Comprehensive training data preparation and model retraining

The system maintains all original functionality while adding powerful dual-board capabilities and high-camera optimization.
