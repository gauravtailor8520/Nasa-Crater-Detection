# 🏗️ System Architecture

Comprehensive technical architecture documentation for the Crater Detection System.

---

## 🏛️ High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                             │
├─────────────────────────────────────────────────────────────┤
│  Web Browser        │  Python Client      │  CLI Tools      │
│  (HTML/JS)          │  (requests library) │  (batch)        │
└────────────┬─────────────────────────────┬──────────────────┘
             │                             │
             ▼                             ▼
┌─────────────────────────────────────────────────────────────┐
│              API & Application Layer                         │
├─────────────────────────────────────────────────────────────┤
│  Flask Web Server (app.py)                                  │
│  ├─ Route: GET /        (Web Interface)                     │
│  ├─ Route: POST /detect (Detection API)                     │
│  └─ Static File Server  (Images, CSS, JS)                   │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│          Core Processing Layer                              │
├─────────────────────────────────────────────────────────────┤
│  Model Utils (model_utils.py)                               │
│  ├─ YOLO Detection Engine                                   │
│  └─ Ellipse Fitting Algorithm                               │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│         Deep Learning & Image Processing                    │
├─────────────────────────────────────────────────────────────┤
│  YOLOv8 Model (best.pt)    │  OpenCV (cv2)                 │
│  ├─ Object Detection       │  ├─ Edge Detection            │
│  ├─ Bounding Boxes         │  ├─ Distance Transform        │
│  └─ Confidence Scores      │  └─ Ellipse Fitting           │
└────────────┬─────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│           Data & Storage Layer                              │
├─────────────────────────────────────────────────────────────┤
│  Uploaded Images  │  Processed Images  │  Detection JSON    │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Module Architecture

### 1. **Flask Application Layer** (`app/app.py`)

```
Flask App
├── Route: GET /
│   └── Returns index.html
│
├── Route: POST /detect
│   ├── Receive file upload
│   ├── Validate file
│   ├── Save original image
│   ├── Call detect_craters()
│   ├── Save processed image
│   └── Return JSON response
│
└── Static Files
    ├── CSS serving
    ├── JavaScript serving
    └── Image serving
```

**Key Components:**
- Flask instance initialization
- Route definitions
- CORS handling
- Error handling middleware

---

### 2. **Model Utils Layer** (`app/model_utils.py`)

```
model_utils.py
├── Global: Load YOLO Model
│   └── MODEL_PATH = best.pt
│
├── Function: detect_craters(image_path)
│   ├── Load image with OpenCV
│   ├── Run YOLO prediction
│   ├── Extract boxes and classes
│   ├── For each detection:
│   │   ├── Extract crop region
│   │   └── Call process_crop()
│   ├── Generate detection info
│   ├── Draw annotations
│   └── Return (annotated_img, detections)
│
└── Function: process_crop(crop_img)
    ├── Convert to grayscale
    ├── Gaussian blur
    ├── CLAHE enhancement
    ├── Canny edge detection
    ├── Distance transform
    ├── Extract rim points
    ├── Fit ellipse
    └── Return ellipse params
```

---

### 3. **Frontend Layer** (`app/templates/index.html` & `app/static/js/script.js`)

```
Frontend
├── HTML Structure
│   ├── Upload form
│   ├── File input
│   ├── Submit button
│   ├── Image display areas
│   └── Results panel
│
├── CSS Styling
│   ├── Layout
│   ├── Colors
│   ├── Responsive design
│   └── Animations
│
└── JavaScript Logic
    ├── File selection
    ├── Fetch API calls
    ├── Response handling
    ├── Image display
    ├── Results rendering
    └── Error handling
```

---

## 🔄 Data Flow Diagram

### Request Flow
```
User Upload
    ↓
Flask /detect Route
    ↓
File Validation
    ├─ Check file exists
    ├─ Check MIME type
    └─ Check file size
    ↓
Save Original Image
    └─ timestamp_filename.jpg
    ↓
detect_craters()
    ├─ cv2.imread(image_path)
    ├─ model.predict()
    ├─ For each detection box:
    │   ├─ Extract crop
    │   ├─ process_crop()
    │   └─ Fit ellipse
    └─ Return (annotated_img, detections)
    ↓
Save Processed Image
    └─ processed_timestamp_filename.jpg
    ↓
Generate JSON Response
    ├─ original_url
    ├─ processed_url
    ├─ detections[]
    └─ count
    ↓
Return to Client
    ↓
Display Results
```

---

## 🧠 Detection Algorithm Pipeline

### Step 1: Image Input
```python
original_img = cv2.imread(image_path)  # Shape: (H, W, 3)
```

### Step 2: YOLO Detection
```python
results = model.predict(
    source=image_path,
    imgsz=640,           # Input size
    conf=0.25,           # Confidence threshold
    iou=0.5              # NMS IOU threshold
)
# Output: boxes (x1, y1, x2, y2), classes
```

### Step 3: Crop Extraction
```python
for box in boxes:
    x1, y1, x2, y2 = box
    crop = original_img[y1:y2, x1:x2]
```

### Step 4: Image Preprocessing
```
Crop → Grayscale → Blur → CLAHE → Enhanced Image
```

**Parameters:**
- Gaussian Blur: kernel (7, 7), sigma 0
- CLAHE: clip_limit 3.0, grid (8, 8)

### Step 5: Edge Detection
```
Enhanced → Canny (50, 140) → Edge Map
```

### Step 6: Distance Transform
```
Edge Map → Bitwise NOT → Distance Transform → Normalized
```

### Step 7: Rim Point Extraction
```
Normalized Distance < 0.12 → Rim Mask
```

### Step 8: Ellipse Fitting
```python
ellipse = cv2.fitEllipse(rim_points)
# Returns: ((cx, cy), (major, minor), angle)
```

### Step 9: Coordinate Transformation
```python
global_cx = crop_x1 + local_cx
global_cy = crop_y1 + local_cy
```

---

## 🗄️ Data Structures

### Detection Object
```python
{
    "bbox": [x1, y1, x2, y2],        # Bounding box pixels
    "class": 0,                       # Crater class ID
    "ellipse": {
        "cx": float,                  # Center X (global coords)
        "cy": float,                  # Center Y (global coords)
        "major": float,               # Semi-major axis pixels
        "minor": float,               # Semi-minor axis pixels
        "angle": float                # Rotation degrees
    } or None
}
```

### Response Object
```python
{
    "original_url": str,              # URL to original image
    "processed_url": str,             # URL to annotated image
    "detections": [Detection],        # List of detections
    "count": int                      # Number of detections
}
```

---

## 🔗 Dependencies Graph

```
app.py
├─ Flask (web framework)
├─ model_utils.py
│  ├─ cv2 (OpenCV)
│  ├─ numpy
│  └─ ultralytics.YOLO
│     ├─ torch
│     └─ torchvision
└─ render_template() (templates/index.html)

model_utils.py
├─ cv2 (image processing)
├─ numpy (numerical)
├─ ultralytics.YOLO (deep learning)
│  ├─ torch
│  ├─ torchvision
│  └─ numpy
└─ logging (debugging)

templates/index.html
└─ static/js/script.js
   └─ Fetch API (built-in)

static/js/script.js
└─ Fetch API (browser built-in)
```

---

## 📊 Class Diagram

```
┌─────────────────────────────┐
│      FlaskApplication       │
├─────────────────────────────┤
│ - app: Flask                │
│ - UPLOAD_FOLDER: str        │
├─────────────────────────────┤
│ + index()                   │
│ + detect()                  │
└─────────────────────────────┘
           ▲
           │ uses
           │
┌─────────────────────────────┐
│      YOLODetector           │
├─────────────────────────────┤
│ - model: YOLO               │
│ - MODEL_PATH: str           │
├─────────────────────────────┤
│ + detect_craters()          │
│ + process_crop()            │
└─────────────────────────────┘
           ▲
           │ uses
           │
┌─────────────────────────────┐
│    ImageProcessor           │
├─────────────────────────────┤
│ - cv2                       │
│ - numpy                     │
├─────────────────────────────┤
│ + blur()                    │
│ + enhance()                 │
│ + detect_edges()            │
│ + fit_ellipse()             │
└─────────────────────────────┘
```

---

## 🔢 Algorithm Complexity

### Time Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| YOLO Detection | O(n) | n = image pixels |
| Gaussian Blur | O(n × k²) | k = kernel size (7×7) |
| Canny Edge | O(n) | Linear scan |
| Distance Transform | O(n) | Using cv2.distanceTransform |
| Ellipse Fitting | O(m²) | m = rim points (~100-500) |

**Overall Detection Time:** O(n + m²) per image

### Space Complexity

| Component | Space | Notes |
|-----------|-------|-------|
| Original Image | O(H × W × 3) | Full resolution |
| YOLO Model | ~50MB | Loaded once globally |
| Intermediate Buffers | O(H × W) | Grayscale, edges, distance |
| Rim Points Array | O(m) | m = detected points |

---

## ⚡ Performance Optimizations

### 1. Model Loading
```python
# Load once at startup, not per request
MODEL_PATH = os.path.abspath(...) 
model = YOLO(MODEL_PATH)  # Global

# Use cached model in detect()
if model is None:
    return None, []
```

### 2. CPU/GPU Selection
```python
results = model.predict(
    device="cpu",  # Or 0 for GPU
    save=False,
    verbose=False
)
```

### 3. Batch Processing
```python
# Process multiple images at once
results = model.predict(source=image_list, batch=8)
```

### 4. Image Resizing
```python
# Adaptive input size
imgsz=640  # Or adjust based on content
```

---

## 🔐 Security Architecture

### File Upload Security Layer
```
Input File
    ↓
Type Validation
    ├─ Magic number check
    └─ Extension check
    ↓
Size Validation
    └─ MAX_FILE_SIZE check
    ↓
Quarantine/Temp Storage
    └─ app/static/uploads/
    ↓
Sanitize Filename
    └─ timestamp_hash_filename
```

### Model Execution Isolation
```
User Request
    ↓
Input Validation
    ↓
Resource Limit (Memory/Time)
    ↓
Model Inference
    ↓
Output Sanitization
    ↓
Resource Cleanup
```

---

## 🔄 State Management

### Model State
```python
# Global state - loaded once
model = YOLO(MODEL_PATH)  # Initialized at module import

# Thread-safe for inference
# YOLO is thread-safe for prediction
```

### Session State (Optional)
```python
# Could add session tracking
@app.route('/detect', methods=['POST'])
def detect():
    session_id = request.headers.get('X-Session-ID')
    # Track uploads per session
```

---

## 📈 Scalability Architecture

### Horizontal Scaling
```
Client Requests
    ↓
Load Balancer
    ├─ Instance 1: Flask + Model
    ├─ Instance 2: Flask + Model
    └─ Instance N: Flask + Model
    ↓
Shared Upload Storage (S3/NFS)
```

### Vertical Scaling
```
Single Server Optimization
├─ Larger batch sizes
├─ GPU acceleration
├─ Parallel processing
└─ Caching layer (Redis)
```

### Queue-Based Processing
```
Flask API (lightweight)
    ↓
Task Queue (Celery/RQ)
    ├─ Worker 1
    ├─ Worker 2
    └─ Worker N
    ↓
Results Database
```

---

## 🧪 Testing Architecture

### Unit Testing
```
test/
├─ test_model_utils.py
│  ├─ test_detect_craters()
│  └─ test_process_crop()
└─ test_app.py
   ├─ test_detect_endpoint()
   └─ test_file_upload()
```

### Integration Testing
```
Mock Image → detect_craters() → JSON Output
         ↓
    Assert results
```

### Load Testing
```
Multiple concurrent requests
    ↓
Monitor response times
    ↓
Check resource usage
```

---

## 🚀 Deployment Architecture

### Development
```
Single Machine
├─ Flask debug mode
├─ Console logging
└─ Hot reload
```

### Production (Docker)
```
┌──────────────────┐
│   Docker Image   │
├──────────────────┤
│ - Python runtime │
│ - Dependencies   │
│ - Model weights  │
│ - App code       │
└──────────────────┘
    ↓
┌──────────────────┐
│  Docker Container│
├──────────────────┤
│ gunicorn/uwsgi   │
│ Flask app        │
│ Port: 5000       │
└──────────────────┘
```

### Production (Cloud)
```
Kubernetes Cluster
├─ Deployment (replicas=3)
├─ Service (LoadBalancer)
├─ ConfigMap (settings)
├─ Secret (credentials)
└─ PersistentVolume (uploads)
```

---

## 📊 Monitoring Architecture

### Metrics to Track
```
Performance
├─ Request latency (p50, p95, p99)
├─ Detection accuracy
└─ Throughput (requests/sec)

Resource
├─ CPU usage
├─ Memory usage
└─ Disk I/O

Errors
├─ 4xx errors
├─ 5xx errors
└─ Model failures
```

### Logging Architecture
```
Application Logs
    ├─ Flask requests
    ├─ YOLO predictions
    └─ Errors/exceptions
    ↓
Log Aggregation
    ├─ File logs
    ├─ CloudWatch
    └─ ELK Stack
```

---

## 🔍 Debugging Architecture

### Debug Mode
```python
app.run(debug=True)  # Enables:
# - Hot reload
# - Detailed errors
# - Interactive debugger
# - Request logging
```

### Logging Levels
```python
logging.DEBUG     # Detailed information
logging.INFO      # General info
logging.WARNING   # Warning messages
logging.ERROR     # Error messages
logging.CRITICAL  # Critical errors
```

---

**Architecture Version**: 1.0  
**Last Updated**: January 2026
