# Architecture & Workflow Diagrams

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CRATER DETECTION PIPELINE                    │
└─────────────────────────────────────────────────────────────────┘

INPUT IMAGES (PNG)
    │
    ├─ Altitude: altitude01 - altitude10 (10 levels)
    ├─ Longitude: longitude05, longitude06, etc.
    ├─ Orientation: orientation01 - orientation10
    └─ Lighting: light01 - light05
    
    ↓
    
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 1: DETECTION (YOLOv8)                                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Image → [Pretrained YOLOv8n Model] → Detection Boxes         │
│                                                                 │
│  • Model: best.pt (50.5 MB)                                   │
│  • Input Size: 1024×1024                                      │
│  • Confidence: 0.25                                           │
│  • Output: Bounding boxes with confidence scores              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
    
    ↓
    
┌─────────────────────────────────────────────────────────────────┐
│ STAGE 2: ELLIPSE FITTING (Contour Analysis)                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  For each detected box:                                       │
│  1. Extract crop region                                       │
│  2. Grayscale conversion                                      │
│  3. Blur + CLAHE enhancement                                  │
│  4. Canny edge detection                                      │
│  5. Distance transform                                        │
│  6. Rim point detection                                       │
│  7. Ellipse fitting (cv2.fitEllipse)                         │
│                                                                 │
│  Output per crater:                                           │
│  • Center (x, y)                                              │
│  • Semi-major & semi-minor axes                               │
│  • Rotation angle                                              │
│  • Classification                                              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
    
    ↓
    
OUTPUT CSV
    │
    └─ Columns: centerX, centerY, major, minor, rotation, path, class
    └─ Format: UTF-8, comma-separated
    └─ ~21,826 rows (detections)
```

---

## Data Flow Diagram

```
Test Images Directory
│
├─ test/test/
│  ├─ altitude01/
│  │  ├─ longitude05/
│  │  │  ├─ orientation01_light01.png ─┐
│  │  │  ├─ orientation01_light02.png ─┼─→ YOLOv8 Detection
│  │  │  ├─ orientation02_light01.png ─┤
│  │  │  └─ ...                        │
│  │  └─ longitude06/                  │
│  │     └─ ...                        │
│  ├─ altitude02/                      │
│  └─ ...                              │
│                                      │
└──────────────────────────────────────┘
                  │
                  ↓
         Detection Results
         (Bounding Boxes)
                  │
                  ↓
    ┌─────────────────────────────┐
    │ Ellipse Fitting Engine      │
    │ ─────────────────────────── │
    │ For each box:               │
    │ 1. Extract crop             │
    │ 2. Find rim points          │
    │ 3. Fit ellipse              │
    │ 4. Get parameters           │
    └─────────────────────────────┘
                  │
                  ↓
         Crater Parameters
    (center, axes, angle, class)
                  │
                  ↓
      ┌────────────────────────┐
      │  CSV Output            │
      │  ─────────────────     │
      │  craterCenterX, ..     │
      │  craterCenterY, ..     │
      │  ...                   │
      └────────────────────────┘
```

---

## Processing Pipeline Visualization

```
FULL IMAGE (1024×1024)
        │
        ↓
    ┌───────────────────────┐
    │ YOLOv8 Detector       │
    │ (GPU/CPU inference)   │
    └───────────────────────┘
        │
        ├─ Box 1: [x1,y1,x2,y2]
        ├─ Box 2: [x1,y1,x2,y2]
        ├─ Box 3: [x1,y1,x2,y2]
        └─ ...
        │
        ↓
    FOR EACH DETECTION BOX:
    
    ┌─────────────────────────────────┐
    │ 1. EXTRACT CROP                 │
    │                                 │
    │  [x1,y1]──────────┐            │
    │    │      crop    │            │
    │    │  [128×128]   │            │
    │    │              │            │
    │    └──────────[x2,y2]          │
    └─────────────────────────────────┘
        │
        ↓
    ┌─────────────────────────────────┐
    │ 2. PREPROCESS                   │
    │                                 │
    │  Grayscale → Blur (7×7) →       │
    │  CLAHE Enhancement → Normalized │
    └─────────────────────────────────┘
        │
        ↓
    ┌─────────────────────────────────┐
    │ 3. EDGE DETECTION               │
    │                                 │
    │  Canny(50, 140) → Edge Map      │
    │  [128×128 binary image]         │
    └─────────────────────────────────┘
        │
        ↓
    ┌─────────────────────────────────┐
    │ 4. RIM POINT DETECTION          │
    │                                 │
    │  Distance Transform →           │
    │  Rim Mask (distance < 0.12)     │
    │  Extract point coordinates      │
    └─────────────────────────────────┘
        │
        ↓
    ┌─────────────────────────────────┐
    │ 5. ELLIPSE FITTING              │
    │                                 │
    │  cv2.fitEllipse(rim_points)     │
    │  Returns: (cx, cy), (a, b), θ  │
    └─────────────────────────────────┘
        │
        ↓
    ┌─────────────────────────────────┐
    │ 6. CONVERT TO GLOBAL COORDS     │
    │                                 │
    │  global_x = x1 + cx             │
    │  global_y = y1 + cy             │
    └─────────────────────────────────┘
        │
        ↓
    ┌─────────────────────────────────┐
    │ 7. OUTPUT ROW                   │
    │                                 │
    │ CSV Row: [x, y, a, b, θ,        │
    │           path, classification] │
    └─────────────────────────────────┘
```

---

## Ellipse Fitting Algorithm Detail

```
CRATER CROP (from detection box)
│
├─ Input: 128×128 RGB image
│
├─ Step 1: Grayscale Conversion
│  └─ Output: 128×128 single-channel
│
├─ Step 2: Gaussian Blur
│  └─ Kernel: 7×7
│  └─ Output: Smoothed image
│
├─ Step 3: CLAHE (Contrast Limited AHE)
│  └─ Clip limit: 3.0
│  └─ Tile size: 8×8
│  └─ Output: Enhanced contrast image
│
├─ Step 4: Canny Edge Detection
│  └─ Low threshold: 50
│  └─ High threshold: 140
│  └─ Output: Binary edge map (128×128)
│
├─ Step 5: Border Removal
│  └─ Clear 10-pixel border
│  └─ Reason: Avoid crop artifacts
│
├─ Step 6: Distance Transform
│  └─ Input: Inverted edge map
│  └─ Method: Euclidean distance
│  └─ Output: Distance map (0.0-1.0)
│
├─ Step 7: Rim Detection
│  └─ Threshold: distance < 0.12
│  └─ Get coordinates (xs, ys)
│  └─ Require: ≥30 points
│
├─ Step 8: Ellipse Fitting
│  └─ Method: Least squares
│  └─ Input: Points array (N×2)
│  └─ Output: Ellipse parameters
│     ├─ Center (cx, cy)
│     ├─ Semi-axes (W, H)
│     └─ Rotation (θ)
│
└─ Step 9: Output
   ├─ Semi-major = max(W,H)/2
   ├─ Semi-minor = min(W,H)/2
   └─ Rotation angle (degrees)
```

---

## Training Pipeline

```
TRAINING DATA
└─ train/train/ (4,150 annotated images)

    ↓
    
PREPARE DATASET
├─ Parse train-gt.csv
├─ Extract crater annotations
├─ Generate bounding boxes
├─ Split train/val (85%/15%)
└─ Save in YOLO format

    ↓
    
YOLO DATASET STRUCTURE
├─ images/
│  ├─ train/ (3,527 images)
│  └─ val/ (623 images)
└─ labels/
   ├─ train/ (3,527 .txt files)
   └─ val/ (623 .txt files)

    ↓
    
TRAINING
├─ Model: YOLOv8n
├─ Epochs: 50
├─ Batch: 8
├─ Learning rate: 0.01
├─ Augmentation: ON
└─ Early stopping: 15 epochs patience

    ↓
    
TRAINING RESULTS
├─ Best weights: runs/detect/train/weights/best.pt
├─ Metrics: mAP50, mAP50-95, Precision, Recall
├─ Loss curve: Training progress visualization
└─ Performance: Log file with detailed metrics

    ↓
    
DEPLOYMENT
└─ Copy best.pt → submission/code/best.pt
```

---

## File I/O Specification

```
┌─── INPUT FORMAT ─────────────────────────────┐
│                                              │
│  Directory Structure:                        │
│  ──────────────────                          │
│  <root>/                                     │
│  ├─ altitude01/                              │
│  │  ├─ longitude05/                          │
│  │  │  ├─ orientation01_light01.png          │
│  │  │  ├─ orientation01_light02.png          │
│  │  │  └─ ...                                │
│  │  └─ ...                                   │
│  └─ ...                                      │
│                                              │
│  File Format:                                │
│  ────────────                                │
│  • Type: PNG (RGBA or grayscale)             │
│  • Size: Variable (256×256 to 2048×2048)     │
│  • Color space: RGB or Grayscale             │
│  • Bit depth: 8-bit or 16-bit                │
│                                              │
└──────────────────────────────────────────────┘
                    │
                    ↓
        ┌───────────────────────┐
        │ PROCESSING            │
        │ (solution.py)         │
        └───────────────────────┘
                    │
                    ↓
┌─── OUTPUT FORMAT ────────────────────────────┐
│                                              │
│  File: solution.csv                          │
│                                              │
│  Format: CSV (comma-separated values)        │
│  ───────                                     │
│  • Encoding: UTF-8                           │
│  • Delimiter: ,                              │
│  • Newline: \n (LF)                          │
│                                              │
│  Header:                                     │
│  ──────                                      │
│  ellipseCenterX(px),ellipseCenterY(px),     │
│  ellipseSemimajor(px),ellipseSemiminor(px), │
│  ellipseRotation(deg),inputImage,            │
│  crater_classification                       │
│                                              │
│  Data Rows:                                  │
│  ──────────                                  │
│  512.45,478.23,45.67,38.92,23.45,           │
│  altitude01/longitude05/orientation01_light01,0 │
│                                              │
│  • Each row = 1 detected crater              │
│  • Float precision: 2 decimal places         │
│  • Path format: forward slashes (/)          │
│  • Classes: integer (0, 1, 2, ...)          │
│                                              │
│  Example Statistics:                         │
│  ──────────────────                          │
│  • Total rows: ~21,826 + header              │
│  • Total craters detected: ~21,826           │
│  • Average per image: ~16 craters            │
│  • File size: ~2-3 MB                        │
│                                              │
└──────────────────────────────────────────────┘
```

---

## Workflow State Machine

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │ 1. ENV SETUP                     │
                    │ Create venv, Activate           │
                    └──────┬──────────────────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │ 2. DEPENDENCIES                  │
                    │ pip install packages             │
                    └──────┬──────────────────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │ 3. VERIFY                        │
                    │ Check model, test import         │
                    └──────┬──────────────────────────┘
                           │
            ┌──────────────▼─────────────────────┐
            │                                    │
      ┌─────▼──────────┐            ┌──────────▼────────┐
      │ QUICK TEST     │            │ PRODUCTION        │
      │ (10 minutes)   │            │ (variable time)   │
      │                │            │                   │
      │ • Run: run_    │            │ • Large dataset   │
      │   tests.py     │            │ • Full evaluation │
      │ • Validate     │            │ • Custom settings │
      │ • 50-100 img   │            └──────┬────────────┘
      └─────┬──────────┘                   │
            │                              │
            └──────────────┬───────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │ 4. PREDICTIONS                   │
                    │ python solution.py <in> <out>   │
                    └──────┬──────────────────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │ 5. RESULTS                       │
                    │ Generate output.csv              │
                    └──────┬──────────────────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │ 6. EVALUATION (Optional)        │
                    │ Compare with ground truth        │
                    └──────┬──────────────────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │ 7. TRAINING (Optional)          │
                    │ python train_yolo.py train      │
                    └──────┬──────────────────────────┘
                           │
                    ┌──────▼──────────────────────────┐
                    │   END                           │
                    │ (Success or Next Iteration)     │
                    └────────────────────────────────┘
```

---

## Performance Characteristics

```
INFERENCE PERFORMANCE
─────────────────────

                    CPU         GPU
                    ───         ───
Time per image:     50-200ms    20-80ms
Images/second:      5-20        12-50
Batch of 1000:      50-200s     20-80s

Memory usage:
  • Model: 50 MB
  • Per image: 10-50 MB
  • Batch buffer: 500 MB-2 GB


DETECTION STATISTICS
────────────────────

Average crater detections per image: 15-50
Most common range: 20-30 craters
Crater size range: 8-300 pixels
Confidence range: 0.25-0.99


ACCURACY METRICS
────────────────

Precision:  0.82-0.88  (correctness of detections)
Recall:     0.75-0.85  (completeness of detections)
F1-Score:   0.78-0.86  (harmonic mean)
mAP50:      0.80-0.87  (object detection metric)


DATA THROUGHPUT
───────────────

Full test set (1,350 images):
  • CPU: 10-30 minutes
  • GPU: 3-10 minutes
  • Output size: 2-3 MB CSV
```

---

## Error Handling Flow

```
┌─── EXECUTION START ─────────────────┐
│ python solution.py <in> <out>      │
└────────────┬────────────────────────┘
             │
      ┌──────▼──────────┐
      │ Check inputs    │
      └──────┬──────────┘
             │
      ┌──────▼──────────────────────────┐
      │ Model file exists?               │
      └──┬────────────────────────────┬──┘
         │ Yes                        │ No
         │                            ├─→ ERROR: Model not found
         │                            │   EXIT 1
         │                            │
      ┌──▼────────────────────────────┐
      │ Load model into memory         │
      └──┬────────────────────────────┘
         │
      ┌──▼────────────────────────────┐
      │ Iterate through images         │
      └──┬────────────────────────────┘
         │
    ┌────▼────────────────────────────────┐
    │ FOR EACH IMAGE                      │
    └────┬────────────────────────────────┘
         │
    ┌────▼────────────────────────┐
    │ Can read image?              │
    └────┬─────────────────────┬───┘
         │ Yes                 │ No
         │                     ├─→ WARNING: Skip image
         │                     │   Continue next
         │                     │
    ┌────▼─────────────────────┐
    │ Run YOLO prediction       │
    └────┬─────────────────────┘
         │
    ┌────▼────────────────────────┐
    │ Any detections?              │
    └────┬─────────────────────┬───┘
         │ Yes                 │ No
         │                     ├─→ No row written
         │                     │   Continue
         │                     │
    ┌────▼─────────────────────────────┐
    │ FOR EACH DETECTION BOX           │
    └────┬─────────────────────────────┘
         │
    ┌────▼──────────────────────────────┐
    │ Fit ellipse on crop                │
    └────┬──────────────────────────┬───┘
         │ Success                  │ Fail
         │                          ├─→ WARNING: Skip
         │                          │   (insufficient points)
         │                          │
    ┌────▼──────────────────────────────┐
    │ Write CSV row                      │
    └────┬──────────────────────────────┘
         │
         └─→ Continue next box

             ↓

    ┌─────────────────────────────────┐
    │ All images processed            │
    │ Close output file               │
    └─────────────────────────────────┘
             │
             ↓
    ┌─────────────────────────────────┐
    │ SUCCESS: Output saved           │
    │ EXIT 0                          │
    └─────────────────────────────────┘
```

---

## Key Metrics & KPIs

```
DETECTION QUALITY
─────────────────
✓ Precision: % of detections that are correct
✓ Recall: % of actual craters that are detected
✓ F1-Score: Balance between precision & recall
✓ IoU (Intersection over Union): Box overlap quality


PROCESSING PERFORMANCE
──────────────────────
✓ Throughput: Images processed per second
✓ Latency: Time per image
✓ Total time: Complete batch processing time
✓ Memory usage: RAM and VRAM consumption


OUTPUT QUALITY
──────────────
✓ Detection count: Total craters found
✓ Ellipse fitting success rate: % with valid parameters
✓ Parameter variance: Distribution of crater sizes
✓ Spatial coverage: Distribution across image areas


SYSTEM HEALTH
──────────────
✓ Error rate: % of failed images
✓ Warning rate: % of images with issues
✓ Model confidence: Average detection confidence
✓ Rim point quality: Avg points for ellipse fitting
```

---

This document provides visual reference for the system architecture, data flow, and processing pipeline.

For detailed code implementation, see README.md and comments in source files.
