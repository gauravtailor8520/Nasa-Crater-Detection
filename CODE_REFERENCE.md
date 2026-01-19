# 📚 Code Module Reference

Complete reference documentation for all Python modules and functions in the Crater Detection project.

---

## 📁 File Structure Reference

```
app/
├── app.py ..................... Flask web application
├── model_utils.py ............. Detection and image processing
├── requirements.txt ........... Python dependencies
└── run.bat ..................... Windows launcher

submission/code/
├── solution.py ................ Batch processing solution
├── best.pt .................... Trained YOLO model
├── train.sh ................... Training script
├── test.sh .................... Testing script
└── Dockerfile ................. Docker configuration

provided files/
├── scorer.py .................. Scoring/evaluation
├── data_combiner.py ........... Data utilities
└── *.csv ...................... Data files
```

---

## 🔌 app/app.py

### Purpose
Main Flask web application providing HTTP API for crater detection.

### Imports
```python
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import time
from model_utils import detect_craters
```

### Global Variables

#### `UPLOAD_FOLDER`
```python
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
```
- **Type**: `str`
- **Purpose**: Directory for storing uploaded and processed images
- **Default**: `app/static/uploads/`
- **Permissions**: Must be writable

### Routes

#### **Route: GET /**

```python
@app.route('/')
def index():
    return render_template('index.html')
```

**Parameters**: None

**Response**: 
- HTTP 200 OK
- Content-Type: `text/html`
- Body: Rendered HTML interface

**Example**:
```
GET http://localhost:5000/
```

#### **Route: POST /detect**

```python
@app.route('/detect', methods=['POST'])
def detect():
```

**Parameters**:
- `image` (multipart/form-data, required): Image file

**Process**:
1. Validate file exists and has filename
2. Generate timestamp-based filename
3. Save uploaded image
4. Call `detect_craters()` from model_utils
5. Save annotated image
6. Return JSON response

**Response Format**:
```json
{
  "original_url": "/static/uploads/upload_1705695234_image.jpg",
  "processed_url": "/static/uploads/processed_1705695234_image.jpg",
  "detections": [...],
  "count": 3
}
```

**Error Responses**:
- 400: No image uploaded or no filename
- 500: Model processing failed

**Example**:
```bash
curl -X POST http://localhost:5000/detect \
  -F "image=@satellite.jpg"
```

### Configuration

```python
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
```

**Configurable Settings**:
- Debug mode: `app.run(debug=True/False)`
- Port: `app.run(port=5000)`
- Host: `app.run(host='0.0.0.0')`

### Main Entry Point

```python
if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Parameters**:
- `debug=True`: Enable auto-reload and detailed errors
- `port=5000`: HTTP port

---

## 🔍 app/model_utils.py

### Purpose
Core crater detection and image processing functions.

### Imports
```python
import os
import cv2
import numpy as np
from ultralytics import YOLO
import logging
```

### Global Variables

#### `MODEL_PATH`
```python
MODEL_PATH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 
    '../submission/code/best.pt'
))
```
- **Type**: `str`
- **Purpose**: Path to pre-trained YOLO model
- **Format**: `.pt` (PyTorch)
- **Size**: ~50MB

#### `model`
```python
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    model = None
```
- **Type**: `YOLO` or `None`
- **Scope**: Global (loaded once)
- **Purpose**: YOLO detection model instance

### Functions

#### **Function: process_crop(crop_img)**

Fits an ellipse to a crater crop region using image processing.

**Signature**:
```python
def process_crop(crop_img):
    """
    Fits an ellipse to a crater crop using the logic from ff.ipynb.
    Returns: ((cx, cy), (semi_major, semi_minor), angle) or None
    """
```

**Parameters**:
- `crop_img` (numpy.ndarray): Cropped image region (BGR or grayscale)
  - Shape: (H, W, 3) or (H, W)
  - Dtype: uint8

**Returns**:
- `tuple`: `((cx, cy), (W, H), angle)` on success
  - `cx, cy`: Center coordinates (pixels)
  - `W, H`: Full ellipse dimensions (pixels)
  - `angle`: Rotation angle (degrees, 0-180)
- `None`: If processing fails or insufficient points

**Algorithm**:

1. **Convert to Grayscale**
   ```python
   if len(crop_img.shape) == 3:
       gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
   else:
       gray_crop = crop_img
   ```

2. **Preprocessing**
   ```python
   blur = cv2.GaussianBlur(gray_crop, (7, 7), 0)
   clahe = cv2.createCLAHE(3.0, (8, 8))
   enhanced = clahe.apply(blur)
   ```
   - Gaussian Blur: kernel (7, 7), sigma 0
   - CLAHE: clip_limit 3.0, grid (8, 8)

3. **Edge Detection**
   ```python
   edges = cv2.Canny(enhanced, 50, 140)
   ```
   - Thresholds: lower=50, upper=140

4. **Border Removal** (10 pixels)
   ```python
   edges[:border, :] = 0
   edges[-border:, :] = 0
   edges[:, :border] = 0
   edges[:, -border:] = 0
   ```

5. **Distance Transform**
   ```python
   inv = cv2.bitwise_not(edges)
   dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
   dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
   ```
   - Metric: L2 (Euclidean)
   - Mask: 5 (3×3 neighborhood)

6. **Rim Point Extraction**
   ```python
   rim_mask = (dist < 0.12).astype(np.uint8) * 255
   ys, xs = np.where(rim_mask > 0)
   points = np.column_stack((xs, ys))
   ```
   - Distance threshold: 0.12 (normalized)
   - Minimum points: 30

7. **Ellipse Fitting**
   ```python
   ellipse = cv2.fitEllipse(points)
   return ellipse
   ```

**Error Handling**:
- Returns `None` if crop is empty
- Returns `None` if fewer than 30 rim points
- Returns `None` on any exception

**Example**:
```python
crop = image[y1:y2, x1:x2]
ellipse_data = process_crop(crop)

if ellipse_data:
    (cx, cy), (W, H), angle = ellipse_data
    print(f"Ellipse center: ({cx}, {cy})")
    print(f"Dimensions: {W} x {H}")
    print(f"Angle: {angle}°")
```

#### **Function: detect_craters(image_path)**

Main detection function for identifying craters in satellite images.

**Signature**:
```python
def detect_craters(image_path):
    """
    Runs detection on a single image.
    Returns:
        - annotated_image: numpy array (BGR) with drawings
        - detections: list of dicts with crater info
    """
```

**Parameters**:
- `image_path` (str): Path to image file
  - Supported formats: JPG, PNG, BMP
  - Expected: Satellite/lunar imagery

**Returns**:
- `tuple[numpy.ndarray, list[dict]]`
  - Annotated image (BGR array)
  - Detection list

**Return Format**:

Annotated Image:
- Type: `numpy.ndarray`
- Shape: (H, W, 3)
- Dtype: uint8
- Content: Original image with drawn detections

Detection List:
```python
[
    {
        "bbox": [x1, y1, x2, y2],
        "class": 0,
        "ellipse": {
            "cx": center_x,
            "cy": center_y,
            "major": semi_major,
            "minor": semi_minor,
            "angle": rotation_angle
        }
    },
    ...
]
```

**Algorithm**:

1. **Load Image**
   ```python
   original_img = cv2.imread(image_path)
   ```

2. **YOLO Prediction**
   ```python
   results = model.predict(
       source=image_path,
       imgsz=640,
       conf=0.25,
       iou=0.5,
       device="cpu",
       save=False,
       verbose=False
   )
   ```
   - Input size: 640×640
   - Confidence: 0.25
   - NMS IOU: 0.5

3. **Process Detections**
   - For each detected box:
     - Extract crop region
     - Fit ellipse (if possible)
     - Generate detection dict
     - Draw on image

4. **Annotation**
   - Ellipse detected: Draw green ellipse
   - No ellipse: Draw red rectangle

**Example**:
```python
annotated_img, detections = detect_craters('satellite.jpg')

print(f"Found {len(detections)} craters")

for i, det in enumerate(detections):
    print(f"Crater {i}:")
    print(f"  BBox: {det['bbox']}")
    if det['ellipse']:
        print(f"  Ellipse: center=({det['ellipse']['cx']}, {det['ellipse']['cy']})")

# Save annotated image
cv2.imwrite('output.jpg', annotated_img)
```

### Constants

| Name | Value | Purpose |
|------|-------|---------|
| Gaussian Blur Kernel | (7, 7) | Image smoothing |
| CLAHE Clip Limit | 3.0 | Contrast enhancement |
| CLAHE Grid | (8, 8) | Tile size |
| Canny Low | 50 | Edge detection lower |
| Canny High | 140 | Edge detection upper |
| Border Remove | 10 px | Crop border handling |
| Distance Threshold | 0.12 | Rim point extraction |
| Min Rim Points | 30 | Ellipse fitting |
| YOLO Input Size | 640 | Model input resolution |
| YOLO Confidence | 0.25 | Detection threshold |
| YOLO IOU | 0.5 | NMS threshold |

---

## 💾 submission/code/solution.py

### Purpose
Batch processing solution for crater detection on datasets.

### Key Differences from app.py
- Standalone, no Flask dependency
- Batch processing support
- CSV output format
- Command-line interface

### Main Function

```python
def main():
    # Parse arguments
    # Load images
    # Run predictions
    # Save results CSV
```

### Output Format

CSV with columns:
```
image_id, crater_id, center_x, center_y, semi_major, semi_minor, angle
```

---

## 📊 provided files/scorer.py

### Purpose
Evaluation and scoring of crater detections against ground truth.

### Key Functions

#### **dGA (Gaussian Area Distance)**

```python
def dGA(crater_A, crater_B):
    """Computes Gaussian Area distance between two ellipses"""
```

**Purpose**: Measure similarity between predicted and ground truth craters

**Algorithm**:
1. Compute Y-matrices for both ellipses
2. Calculate overlap measure
3. Return angular distance (0-1)

### Scoring Metrics

- **True Positives**: dGA < threshold
- **False Positives**: Detected but no match
- **False Negatives**: Ground truth with no detection
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1-Score**: Harmonic mean

---

## 🎨 Frontend Files

### app/templates/index.html

**Structure**:
- File input for image upload
- Upload button
- Progress indicator
- Results display area
- Image preview

**Key Elements**:
- `<form>` for file upload
- `<div>` for displaying original image
- `<div>` for displaying processed image
- `<div>` for results (JSON)

### app/static/js/script.js

**Key Functions**:

#### **handleFileUpload()**
- Validates file selection
- Shows preview
- Calls detect API

#### **sendDetectionRequest()**
- Uses Fetch API
- Sends POST to `/detect`
- Handles response
- Updates UI

#### **displayResults()**
- Shows detection count
- Displays images
- Lists crater details

---

## 🔧 Requirements

### app/requirements.txt

```
flask
ultralytics
opencv-python-headless
numpy
```

**Version Specifications** (optional):
```
flask>=2.0
ultralytics>=8.0
opencv-python-headless>=4.5
numpy>=1.20
```

---

## 📈 Data Structures

### Detection Object

```python
{
    "bbox": [x1, y1, x2, y2],  # list[int]
    "class": 0,                 # int
    "ellipse": {                # dict or None
        "cx": 200.5,            # float
        "cy": 150.3,            # float
        "major": 85.2,          # float
        "minor": 72.1,          # float
        "angle": 23.5           # float (0-180)
    }
}
```

### API Response

```python
{
    "original_url": "/static/uploads/...",      # str
    "processed_url": "/static/uploads/...",     # str
    "detections": [...],                         # list[dict]
    "count": 5                                   # int
}
```

---

## 🔄 Function Call Flow

```
User Upload
    ↓
Flask receives POST
    ↓
app.py: detect()
    ├─ Validate file
    ├─ Save upload
    ├─ Call model_utils.detect_craters()
    │   ├─ Load image
    │   ├─ Run YOLO
    │   ├─ For each detection:
    │   │   ├─ Extract crop
    │   │   ├─ Call process_crop()
    │   │   │   ├─ Preprocess
    │   │   │   ├─ Edge detect
    │   │   │   ├─ Distance transform
    │   │   │   └─ Fit ellipse
    │   │   └─ Generate detection
    │   └─ Return results
    ├─ Save output image
    └─ Return JSON
    ↓
Client receives JSON + URLs
```

---

## 🧪 Module Testing

### Test detect_craters()

```python
from app.model_utils import detect_craters
import cv2

# Test image
test_image = 'test.jpg'

# Run detection
annotated_img, detections = detect_craters(test_image)

# Verify
assert annotated_img is not None, "No image returned"
assert isinstance(detections, list), "Detections not a list"
assert len(detections) > 0, "No detections found"

print("✓ Detection test passed")
```

### Test process_crop()

```python
from app.model_utils import process_crop
import cv2
import numpy as np

# Create test crop
test_crop = cv2.imread('crater_sample.jpg')

# Process
ellipse = process_crop(test_crop)

# Verify
if ellipse:
    (cx, cy), (W, H), angle = ellipse
    print(f"✓ Ellipse fitted: center=({cx}, {cy})")
else:
    print("✗ Ellipse fitting failed")
```

---

## 📝 Documentation Standards

### Docstring Format

```python
def function_name(param1, param2):
    """
    Short description.
    
    Long description if needed.
    
    Args:
        param1 (type): Description
        param2 (type): Description
    
    Returns:
        type: Description
    
    Raises:
        ExceptionType: Description
    
    Example:
        >>> result = function_name(arg1, arg2)
        >>> print(result)
    """
```

---

**Module Reference Version**: 1.0  
**Last Updated**: January 2026
