
<img width="1680" height="260" alt="image" src="https://github.com/user-attachments/assets/778f5a31-4184-40e3-9046-135cb2acc508" />

# Nasa Crater Detection System




A complete automated pipeline for detecting and characterizing lunar craters from satellite imagery using YOLOv8 object detection and ellipse fitting algorithms.

---
https://github.com/user-attachments/assets/06d8fc98-7df0-4144-ac38-88cbd0f1c754






---
<img width="2592" height="2048" alt="image" src="https://github.com/user-attachments/assets/4ed27e68-f87f-4dc8-9d51-3f5e4fcbb48e" />



## 📋 Project Overview

This is a comprehensive **Lunar Crater Detection System** that uses YOLOv8 deep learning model for detecting craters in satellite imagery. The project includes:

- **YOLO-based crater detection**: Detects crater locations and bounding boxes
- **Ellipse fitting**: Accurately fits ellipses to crater shapes using image processing
- **Web interface**: Flask-based web application for easy crater detection
- **Model training**: Complete training pipeline with evaluation metrics
- **Batch processing**: Process multiple images for crater analysis

### Key Features
✅ Automated crater detection in satellite images  
✅ Ellipse parameter estimation for crater shape analysis  
✅ Web UI for real-time image processing  
✅ Batch processing capabilities  
✅ Detailed performance metrics and scoring  

---

## 🏗️ Project Structure

```
Nase_Crater_Detection/
├── app/                          # Flask web application
│   ├── app.py                    # Main Flask server
│   ├── model_utils.py            # Detection and processing functions
│   ├── requirements.txt          # Python dependencies
│   ├── run.bat                   # Windows batch launcher
│   ├── static/
│   │   ├── css/style.css         # Web interface styling
│   │   ├── js/script.js          # Frontend functionality
│   │   └── uploads/              # Uploaded images storage
│   └── templates/index.html      # Web interface HTML
│
├── submission/                   # Final submission package
│   └── code/
│       ├── solution.py           # Main prediction script
│       ├── best.pt               # Trained YOLO model weights
│       ├── train.sh              # Training script (training not required)
│       ├── test.sh               # Testing/inference script
│       └── Dockerfile            # Docker containerization
│
├── ModelTraining/                # Model training configuration
│   └── Model/
│       ├── args.yaml             # YOLO training hyperparameters
│       ├── weights/              # Model checkpoints
│       ├── results.csv           # Training results
│       ├── predict/              # Prediction output directory
│       └── val/                  # Validation output directory
│
├── Notebooks/                    # Jupyter notebooks for analysis
│   ├── Datadownloader.ipynb      # Data download utilities
│   ├── Ellipsprediction.ipynb    # Ellipse fitting experimentation
│   ├── FinalSolution.ipynb       # Final solution development
│   ├── Testdata.ipynb            # Test data preparation
│   ├── Testsolution.ipynb        # Solution testing
│   └── Yoloprediction.ipynb      # YOLO prediction experiments
│
├── train/ & test/                # Training and test datasets
│   └── altitude01-10/            # Images organized by altitude
│
├── yolo/                         # YOLO-specific directories
│   ├── dataset/                  # YOLO format dataset
│   │   ├── images/               # Image files
│   │   └── labels/               # Annotation labels
│   ├── predictions/              # Model predictions
│   └── runs/                     # Training run outputs
│
├── provided files/               # External utilities
│   ├── scorer.py                 # Offline scoring script
│   ├── data_combiner.py          # Data combining utilities
│   ├── detections-04-16.csv      # Detection results
│   └── train-gt.csv              # Ground truth labels
│
└── crater_detection_output/      # Processing outputs
    └── orientation01_light02/
        ├── crops/                # Extracted crater crops
        ├── results/              # Detection results JSON
        └── visualizations/       # Visual outputs
```

---

## 🚀 Quick Start

### 1. **Environment Setup**

#### Windows Users:
```bash
# Navigate to project directory
cd d:\Nase_Crater_Detection

# Create virtual environment (if not already created)
python -m venv env

# Activate virtual environment
env\Scripts\activate

# Install dependencies
pip install -r app/requirements.txt
```

#### Linux/Mac Users:
```bash
cd /path/to/Nase_Crater_Detection
python3 -m venv env
source env/bin/activate
pip install -r app/requirements.txt
```

### 2. **Run Web Application**

```bash
# Windows
cd app
python app.py
# OR use the batch file
run.bat

# Linux/Mac
cd app
python3 app.py
```

The web interface will be available at: **http://localhost:5000**

### 3. **Run Batch Processing**

```bash
cd submission/code
python solution.py --input <image_path> --output <output_csv>
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| flask | Latest | Web framework |
| ultralytics | Latest | YOLO model framework |
| opencv-python-headless | Latest | Image processing |
| numpy | Latest | Numerical computations |
| pandas | Latest | Data manipulation |

Install all dependencies:
```bash
pip install flask ultralytics opencv-python-headless numpy pandas
```

---

## 🧠 Core Components

### 1. **app/app.py** - Flask Web Server
Main web application providing REST API for crater detection.

**Key Routes:**
- `GET /` - Returns web interface
- `POST /detect` - Accepts image upload and returns detection results

**Output Format:**
```json
{
  "original_url": "/static/uploads/image.jpg",
  "processed_url": "/static/uploads/processed_image.jpg",
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "class": 0,
      "ellipse": {
        "cx": center_x,
        "cy": center_y,
        "major": semi_major_axis,
        "minor": semi_minor_axis,
        "angle": rotation_angle
      }
    }
  ],
  "count": 5
}
```

### 2. **app/model_utils.py** - Detection Pipeline
Implements crater detection and ellipse fitting.

**Key Functions:**
- `detect_craters(image_path)` - Main detection function
- `process_crop(crop_img)` - Fits ellipse to crater crop

**Detection Pipeline:**
1. Load image with OpenCV
2. Run YOLO model for crater localization
3. Extract bounding boxes
4. For each detection, fit ellipse using image processing
5. Return annotated image with detections

### 3. **submission/code/solution.py** - Batch Processing
Standalone solution for batch processing multiple images.

**Features:**
- Processes single or multiple images
- Outputs results in CSV format
- Compatible with scoring system
- Standalone execution without Flask

### 4. **provided files/scorer.py** - Evaluation Metrics
Scoring system for evaluating detection accuracy.

**Metrics:**
- Crater matching using Gaussian Area (dGA)
- Precision, Recall, F1-Score
- Localization accuracy
- Ellipse parameter accuracy

---

## 🔄 Crater Detection Process

### Detection Pipeline Flow:

```
Input Image
    ↓
YOLO Object Detection (YOLOv8n)
    ↓
Extract Bounding Boxes
    ↓
Process Each Crop:
    - Convert to Grayscale
    - Gaussian Blur
    - CLAHE Enhancement
    - Canny Edge Detection
    - Distance Transform
    - Rim Point Extraction
    ↓
Fit Ellipse to Rim Points
    ↓
Generate Global Coordinates
    ↓
Draw Ellipse on Original Image
    ↓
Output Detections + Annotated Image
```

### Image Processing Steps:

1. **Preprocessing**
   - Gaussian blur (7×7 kernel)
   - CLAHE enhancement (clip_limit=3.0, grid=8×8)

2. **Edge Detection**
   - Canny edge detector (50, 140 thresholds)
   - Remove 10px borders

3. **Distance Transform**
   - L2 distance metric
   - Normalize to [0, 1]

4. **Rim Point Extraction**
   - Select points at distance < 0.12 from edges
   - Minimum 30 points required

5. **Ellipse Fitting**
   - OpenCV fitEllipse() function
   - Returns: center, axes, rotation angle

---

## 💻 Usage Examples

### Example 1: Web Interface Detection
1. Open http://localhost:5000
2. Click "Upload Image"
3. Select a crater image
4. View detection results with drawn ellipses
5. Download annotated image

### Example 2: Batch Processing
```bash
# Single image
python solution.py --image test.jpg --output results.csv

# Multiple images
python solution.py --input_dir ./images --output results.csv
```

### Example 3: Using model_utils in Python
```python
from model_utils import detect_craters

image_path = "satellite_image.jpg"
annotated_img, detections = detect_craters(image_path)

for detection in detections:
    print(f"Crater at {detection['bbox']}")
    if detection['ellipse']:
        ellipse = detection['ellipse']
        print(f"  Center: ({ellipse['cx']}, {ellipse['cy']})")
        print(f"  Semi-major: {ellipse['major']}, Semi-minor: {ellipse['minor']}")
```

---

## 🎯 Model Details

### YOLO Model Configuration
- **Architecture**: YOLOv8 Nano (yolov8n.pt)
- **Input Size**: 640×640 pixels
- **Confidence Threshold**: 0.25
- **IOU Threshold**: 0.5
- **Device**: CPU
- **Model Location**: `submission/code/best.pt`

### Training Configuration (args.yaml)
```yaml
task: detect
mode: train
epochs: 30
batch_size: 8
image_size: 640
workers: 8
patience: 10 (early stopping)
optimizer: auto
device: GPU (0) / CPU fallback
```

---

## 📊 Output Files

### 1. Detection JSON Output
```json
{
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "ellipse_params": {
        "center_x": 256.5,
        "center_y": 128.3,
        "semi_major": 45.2,
        "semi_minor": 38.1,
        "angle": 23.5
      }
    }
  ]
}
```

### 2. CSV Format Output
Columns: image_id, crater_id, center_x, center_y, semi_major, semi_minor, angle

---

## 🐛 Troubleshooting

### Issue: Model fails to load
```
Error loading model from best.pt
```
**Solution:**
```bash
# Ensure model file exists
ls submission/code/best.pt

# Verify ultralytics is installed
pip install --upgrade ultralytics
```

### Issue: Port 5000 already in use
```python
# In app.py, change port:
app.run(debug=True, port=5001)  # Use different port
```

### Issue: No craters detected
- Verify image format (JPG, PNG supported)
- Check image size (should be reasonable satellite image)
- Lower confidence threshold in model_utils.py

### Issue: CUDA/GPU not available
- Model automatically falls back to CPU
- Installation: `pip install opencv-python-headless` (headless version)

---

## 🔧 Configuration

### Model Confidence Threshold
Edit `model_utils.py`:
```python
results = model.predict(
    conf=0.25,  # ← Change this value (0.0-1.0)
    iou=0.5
)
```

### Image Upload Folder
Edit `app.py`:
```python
UPLOAD_FOLDER = os.path.join(..., 'static', 'uploads')
```

### Ellipse Fitting Parameters
Edit `model_utils.py` in `process_crop()`:
```python
rim_mask = (dist < 0.12).astype(np.uint8) * 255  # ← Adjust threshold
```

---

## 📈 Performance Metrics

The scoring system evaluates:
- **Precision**: Detected craters / Total detections
- **Recall**: Detected craters / Ground truth craters
- **F1-Score**: Harmonic mean of precision & recall
- **Localization Error**: Distance between detected & actual centers
- **Shape Error**: Difference in ellipse parameters

Run scorer:
```bash
python provided\ files/scorer.py \
    --pred output.csv \
    --truth train-gt.csv \
    --out_dir results/
```

---

## 📝 Training & Validation Data

### Data Structure
```
train/
├── altitude01/image_001.jpg
├── altitude02/image_045.jpg
...
└── altitude10/image_999.jpg

test/
├── altitude01-10 (same structure)
```

### Labels Format (YOLO)
Located in `yolo/dataset/labels/`:
```
<class_id> <center_x> <center_y> <width> <height>
```

---

## 🚢 Docker Deployment

Build Docker image:
```bash
cd submission/code
docker build -t crater-detection .
```

Run container:
```bash
docker run -p 5000:5000 crater-detection
```

---

## 📚 Jupyter Notebooks

| Notebook | Purpose |
|----------|---------|
| FinalSolution.ipynb | Complete solution development & testing |
| Yoloprediction.ipynb | YOLO model experimentation |
| Ellipsprediction.ipynb | Ellipse fitting algorithm development |
| Testdata.ipynb | Test data preparation & analysis |
| Datadownloader.ipynb | Data download & processing |
| Testsolution.ipynb | Solution validation |

---

## 🤝 Contributing

To extend the project:

1. **Improve Detection**: Modify `model_utils.py` detection pipeline
2. **Add Features**: Extend Flask routes in `app.py`
3. **Retrain Model**: Use `ModelTraining/Model/args.yaml` configuration
4. **Optimize Performance**: Adjust hyperparameters in configuration files

---

## 📄 License & Credits

- **YOLO Framework**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- **Image Processing**: OpenCV
- **Scoring**: Gaussian Area metric (dGA)

---

## 🎓 Learning Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Tutorials](https://docs.opencv.org/)
- [Ellipse Fitting Theory](https://en.wikipedia.org/wiki/Ellipse)
- [Crater Science](https://en.wikipedia.org/wiki/Impact_crater)

---

**Last Updated**: January 2026  
**Status**: Production Ready ✅
