# 🤖 Model Training Guide

Comprehensive guide for training and fine-tuning the YOLO crater detection model.

---

## 📋 Overview

This project uses **YOLOv8 (Nano)** for crater detection. The current model (`best.pt`) has been pre-trained. This guide covers:

- Model architecture details
- Training configuration
- Fine-tuning on custom data
- Evaluation and validation
- Model optimization

---

## 🏗️ Model Architecture

### YOLOv8 Nano Specifications

| Aspect | Details |
|--------|---------|
| Architecture | YOLOv8 (You Only Look Once v8) |
| Size | Nano (Lightweight) |
| Parameters | ~3.2M |
| Model Size | ~12MB (full) / ~6MB (quantized) |
| Input Resolution | 640×640 pixels (configurable) |
| Output | Bounding boxes + class predictions |
| Speed (CPU) | ~50-100 ms per image |
| Speed (GPU) | ~5-10 ms per image |

### Model Components

```
Input Image (640×640)
    ↓
Backbone (Feature Extraction)
    ├─ Conv blocks
    ├─ Residual connections
    └─ Multi-scale features
    ↓
Neck (Feature Aggregation)
    ├─ FPN (Feature Pyramid Network)
    └─ PANet
    ↓
Head (Detection)
    ├─ Bounding box regression
    ├─ Object classification
    └─ Confidence scoring
    ↓
Output (Detections)
    └─ Boxes (x, y, w, h)
    └─ Classes
    └─ Confidence
```

---

## 📊 Current Model Performance

### Training Results

From `ModelTraining/Model/results.csv`:

| Metric | Value |
|--------|-------|
| Epochs Trained | 30 |
| Final mAP50 | ~0.85 |
| Final mAP50-95 | ~0.65 |
| Precision | ~0.88 |
| Recall | ~0.82 |
| Training Time | ~2-3 hours (GPU) |

---

## 🚀 Training Setup

### Step 1: Install Training Dependencies

```bash
# Activate virtual environment
env\Scripts\activate  # Windows
source env/bin/activate  # Linux/Mac

# Install training packages
pip install ultralytics torch torchvision torchaudio
pip install opencv-python pillow numpy pandas tensorboard
```

### Step 2: Prepare Dataset

**Dataset Structure (YOLO Format):**

```
crater_dataset/
├── images/
│   ├── train/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   ├── val/
│   │   ├── image101.jpg
│   │   └── ...
│   └── test/
│       └── ...
├── labels/
│   ├── train/
│   │   ├── image001.txt
│   │   ├── image002.txt
│   │   └── ...
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
└── data.yaml
```

**Label Format** (each .txt file):
```
<class_id> <center_x> <center_y> <width> <height>
```

Example:
```
0 0.5 0.5 0.3 0.4
0 0.2 0.8 0.25 0.35
```

### Step 3: Create data.yaml

```yaml
path: /path/to/crater_dataset
train: images/train
val: images/val
test: images/test

nc: 1  # Number of classes
names: ['crater']  # Class names
```

---

## 🎯 Training

### Basic Training

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')  # Nano model

# Train
results = model.train(
    data='crater_dataset/data.yaml',
    epochs=30,
    imgsz=640,
    batch=8,
    device=0,  # GPU device (0 = first GPU, 'cpu' = CPU)
    patience=10,  # Early stopping patience
    save=True,
    project='runs/train',
    name='exp1'
)
```

### Advanced Training Configuration

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(
    data='data.yaml',
    
    # Basic settings
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,
    workers=8,
    
    # Optimization
    optimizer='SGD',  # SGD, Adam, AdamW, RMSProp
    lr0=0.01,        # Initial learning rate
    lrf=0.01,        # Final learning rate
    momentum=0.937,  # Optimizer momentum
    weight_decay=0.0005,  # Weight decay
    
    # Data augmentation
    hsv_h=0.015,     # HSV hue augmentation
    hsv_s=0.7,       # HSV saturation
    hsv_v=0.4,       # HSV value
    degrees=0.0,     # Rotation
    translate=0.1,   # Translation
    scale=0.5,       # Scale
    flipud=0.0,      # Vertical flip
    fliplr=0.5,      # Horizontal flip
    mosaic=1.0,      # Mosaic augmentation
    
    # Training settings
    patience=10,     # Early stopping patience
    save=True,       # Save checkpoints
    save_period=10,  # Save period
    exist_ok=False,  # Overwrite existing runs
    
    # Validation
    val=True,
    patience=10,
    
    # Output
    project='runs/train',
    name='crater_detector_v1'
)
```

### Training with Custom Config File

```bash
# Create training.yaml
epochs: 100
batch: 16
imgsz: 640
device: 0
data: data.yaml
```

```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(cfg='training.yaml')
```

### Using Command Line

```bash
# Basic training
yolo detect train data=data.yaml model=yolov8n.pt epochs=30 imgsz=640

# Advanced training
yolo detect train \
    data=data.yaml \
    model=yolov8n.pt \
    epochs=100 \
    batch=16 \
    imgsz=640 \
    device=0 \
    patience=10 \
    save=True \
    project=runs/train \
    name=exp1
```

---

## 📈 Model Selection & Comparison

### Available YOLOv8 Models

| Model | Size | Parameters | Speed (CPU) | Speed (GPU) | mAP50 |
|-------|------|-----------|------------|-----------|-------|
| Nano | 6MB | 3.2M | 40ms | 5ms | 0.50 |
| Small | 22MB | 11.2M | 80ms | 10ms | 0.60 |
| Medium | 50MB | 25.9M | 200ms | 15ms | 0.65 |
| Large | 94MB | 43.7M | 400ms | 20ms | 0.68 |
| XLarge | 168MB | 68.2M | 800ms | 30ms | 0.69 |

**Recommendation:** Nano for real-time, Medium/Large for accuracy

### Switch Model Size

```python
# Switch from Nano to Small
model = YOLO('yolov8s.pt')  # Small
results = model.train(data='data.yaml', epochs=30)

# Switch to Medium
model = YOLO('yolov8m.pt')  # Medium
results = model.train(data='data.yaml', epochs=30)
```

---

## ✅ Validation & Evaluation

### Validate Model

```python
from ultralytics import YOLO

model = YOLO('runs/train/exp1/weights/best.pt')

# Validate on validation set
metrics = model.val(
    data='data.yaml',
    imgsz=640,
    device=0
)

print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
print(f"Precision: {metrics.box.mp}")
print(f"Recall: {metrics.box.mr}")
```

### Test on Custom Images

```python
from ultralytics import YOLO

model = YOLO('runs/train/exp1/weights/best.pt')

# Predict on single image
results = model.predict('test_image.jpg', conf=0.25)

for result in results:
    print(f"Boxes: {result.boxes}")
    print(f"Detections: {len(result.boxes)}")
    
    # Save annotated image
    result.save(filename='output.jpg')
```

### Batch Evaluation

```python
from ultralytics import YOLO
import os

model = YOLO('runs/train/exp1/weights/best.pt')

# Predict on directory
test_dir = 'test_images/'
results = model.predict(source=test_dir, conf=0.25, save=True)

# Collect statistics
total_detections = sum(len(r.boxes) for r in results)
print(f"Total craters detected: {total_detections}")
```

---

## 🔍 Performance Analysis

### Confusion Matrix

```python
from ultralytics import YOLO

model = YOLO('runs/train/exp1/weights/best.pt')

# Get confusion matrix
results = model.val(data='data.yaml')

# Access confusion matrix
if hasattr(results, 'confusion_matrix'):
    cm = results.confusion_matrix
    print(cm)
```

### ROC Curve

```python
from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO('runs/train/exp1/weights/best.pt')
results = model.val(data='data.yaml', plots=True)

# Plots saved in runs/detect/val/
```

### mAP Analysis

```python
# mAP at different confidence thresholds
model = YOLO('best.pt')

for conf in [0.25, 0.5, 0.75]:
    results = model.val(
        data='data.yaml',
        conf=conf,
        imgsz=640
    )
    print(f"Conf={conf}: mAP50={results.box.map50:.3f}")
```

---

## 🔧 Fine-Tuning

### Transfer Learning

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO('yolov8n.pt')

# Freeze backbone (transfer learning)
for param in model.model.backbone.parameters():
    param.requires_grad = False

# Train only head
results = model.train(
    data='data.yaml',
    epochs=50,
    batch=16,
    freeze=10  # Freeze first 10 layers
)
```

### Progressive Resizing

```python
# Train at lower resolution first
model = YOLO('yolov8n.pt')
model.train(data='data.yaml', epochs=20, imgsz=416)

# Continue at higher resolution
model.train(data='data.yaml', epochs=30, imgsz=640, resume=True)
```

---

## 📊 Monitoring Training

### Real-time Monitoring

```bash
# Start TensorBoard
tensorboard --logdir runs/train/exp1/

# View at http://localhost:6006
```

### Training Metrics

Training automatically generates:
- `results.csv` - Metrics per epoch
- `confusion_matrix.png` - Classification confusion
- `F1_curve.png` - F1-confidence curve
- `PR_curve.png` - Precision-Recall curve
- `P_curve.png` - Precision-confidence curve
- `R_curve.png` - Recall-confidence curve

---

## 🎯 Hyperparameter Optimization

### Grid Search

```python
from ultralytics import YOLO
from itertools import product

model = YOLO('yolov8n.pt')

# Parameter grid
lr_options = [0.001, 0.01, 0.1]
batch_options = [8, 16, 32]
momentum_options = [0.9, 0.937]

best_mAP = 0
best_params = {}

for lr, batch, momentum in product(lr_options, batch_options, momentum_options):
    results = model.train(
        data='data.yaml',
        epochs=10,
        batch=batch,
        lr0=lr,
        momentum=momentum,
        device=0,
        exist_ok=True,
        project='runs/hpo'
    )
    
    if results.box.map > best_mAP:
        best_mAP = results.box.map
        best_params = {'lr': lr, 'batch': batch, 'momentum': momentum}

print(f"Best mAP: {best_mAP}")
print(f"Best params: {best_params}")
```

---

## 💾 Model Export & Conversion

### Export Trained Model

```python
from ultralytics import YOLO

model = YOLO('runs/train/exp1/weights/best.pt')

# Export to different formats
model.export(format='onnx')      # ONNX
model.export(format='torchscript')  # TorchScript
model.export(format='tflite')    # TensorFlow Lite
model.export(format='pb')        # TensorFlow
model.export(format='coreml')    # Core ML
model.export(format='paddle')    # PaddlePaddle
```

### Quantization

```python
# INT8 Quantization (for edge devices)
model = YOLO('best.pt')
model.export(format='tflite', imgsz=640, int8=True)
```

---

## 📦 Model Versioning

### Save Checkpoints

```python
# Automatic checkpoint saving
model.train(
    data='data.yaml',
    epochs=100,
    save=True,
    save_period=10,  # Save every 10 epochs
    project='runs/train',
    name='crater_v1'
)

# Checkpoints in: runs/train/crater_v1/weights/
# last.pt (last epoch)
# best.pt (best mAP)
```

### Resume Training

```python
model = YOLO('runs/train/crater_v1/weights/last.pt')

# Continue from checkpoint
results = model.train(
    data='data.yaml',
    epochs=150,
    resume=True
)
```

---

## 🚀 Production Deployment

### Optimize for Inference

```python
from ultralytics import YOLO

model = YOLO('best.pt')

# FP16 precision (faster, less memory)
model.train(data='data.yaml', device=0, half=True)

# Dynamic batch size
results = model.predict('image.jpg', batch=32)

# Use optimized model
model_opt = YOLO('best_optimized.pt')
```

### Benchmark Performance

```python
from ultralytics import YOLO
import time

model = YOLO('best.pt')

# Benchmark on test set
results = model.predict('test_images/', save=False)

# Time per image
n_images = len(results)
total_time = sum(r.speed['inference'] for r in results)
avg_time = total_time / n_images

print(f"Average inference time: {avg_time:.2f}ms")
print(f"FPS: {1000/avg_time:.1f}")
```

---

## 🐛 Troubleshooting Training

### Issue: Out of Memory (OOM)
```python
# Solution: Reduce batch size
model.train(data='data.yaml', batch=8)  # Smaller batch

# Or use gradient accumulation
model.train(data='data.yaml', batch=16, accumulate=2)
```

### Issue: Poor Validation Performance
```python
# Check for data issues
# 1. Verify labels are correct
# 2. Check class imbalance
# 3. Increase epochs/learning rate
# 4. Use data augmentation

model.train(
    data='data.yaml',
    epochs=100,
    hsv_h=0.015,
    translate=0.1,
    scale=0.5
)
```

### Issue: Overfitting
```python
# Increase regularization
model.train(
    data='data.yaml',
    weight_decay=0.001,  # Increase
    dropout=0.5,  # Add dropout
    augment=True  # Enable augmentation
)
```

### Issue: Underfitting
```python
# Increase model capacity
model = YOLO('yolov8m.pt')  # Larger model

# Train longer
model.train(
    data='data.yaml',
    epochs=200,
    patience=50
)
```

---

## 📚 Advanced Topics

### Custom Loss Function

```python
# Implement custom loss (advanced)
from ultralytics.models.yolo.detect import DetectionTrainer

class CustomTrainer(DetectionTrainer):
    def build_targets(self, model):
        # Custom target building
        pass
    
    def criterion(self, preds, targets):
        # Custom loss computation
        pass
```

### Data Stratification

```python
# Ensure train/val split is stratified
from sklearn.model_selection import train_test_split

# Split data maintaining class distribution
train_idx, val_idx = train_test_split(
    range(len(data)),
    test_size=0.2,
    stratify=class_labels
)
```

---

## 📊 Training Results Interpretation

### Key Metrics Explained

| Metric | Meaning | Good Range |
|--------|---------|-----------|
| mAP50 | Accuracy at IOU=0.5 | >0.75 |
| mAP50-95 | Accuracy at IOU=0.5-0.95 | >0.60 |
| Precision | TP/(TP+FP) | >0.80 |
| Recall | TP/(TP+FN) | >0.75 |
| F1 | Harmonic mean P & R | >0.75 |

---

## 🔄 Continuous Training

### Automated Pipeline

```bash
#!/bin/bash
# train_pipeline.sh

# Download new data
python download_data.py

# Prepare dataset
python prepare_data.py

# Train model
yolo detect train \
    data=data.yaml \
    model=yolov8n.pt \
    epochs=50 \
    batch=16

# Evaluate
python evaluate.py

# Deploy if performance improved
python deploy.py
```

---

**Training Guide Version**: 1.0  
**Last Updated**: January 2026
