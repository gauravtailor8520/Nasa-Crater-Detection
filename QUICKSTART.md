# Quick Reference Guide

## ⚡ One-Minute Setup

```bash
# 1. Create environment
python -m venv env

# 2. Activate (Windows)
.\env\Scripts\activate.bat

# 3. Install packages
pip install numpy opencv-python ultralytics pandas scikit-image matplotlib

# 4. Run predictions
cd submission\code
python solution.py ..\..\test\test ..\..\output.csv
```

## 🚀 Command Cheat Sheet

### Running Tests
```bash
# Full automated test (Windows)
run_tests.bat

# Full automated test (Linux/Mac/Windows)
python run_tests.py

# Quick test
python run_tests.py --quick
```

### Predictions
```bash
# Predict on test set
cd submission/code
python solution.py ..\..\test\test predictions.csv

# Predict on custom data
python solution.py C:\path\to\images output.csv

# Predict with custom settings
python solution.py <input> <output> --conf 0.1 --imgsz 512 --device cpu
```

### Training
```bash
# Train model (50 epochs)
cd yolo
python train_yolo.py train --epochs 50 --batch 8

# Train with GPU
python train_yolo.py train --device 0 --epochs 100

# Use larger model
python train_yolo.py train --model yolov8l --epochs 50
```

### Evaluation
```bash
# Evaluate predictions
cd train-sb
python scorer.py --predictions ../output.csv --ground-truth train-gt.csv

# Combine multiple CSVs
python data_combiner.py --input-files pred1.csv pred2.csv --output combined.csv
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Run specific notebook
jupyter notebook FinalSolution.ipynb
jupyter notebook Yoloprediction.ipynb
jupyter notebook Ellipsprediction.ipynb
```

---

## 📊 File Locations

```
README.md                          # Main documentation (START HERE!)
run_tests.bat / run_tests.py       # Automated test scripts
run_tests_quick.txt                # This quick reference

submission/code/
  ├── solution.py                  # Main prediction script
  ├── best.pt                      # Trained model
  ├── Dockerfile                   # Docker configuration
  ├── train.sh                     # Training entry point
  └── test.sh                      # Testing entry point

yolo/
  ├── train_yolo.py                # Training script
  ├── example_workflow.py           # Complete workflow
  ├── train_crater_yolo.ipynb      # Training notebook
  └── crater.yaml                  # Dataset config

train-sb/
  ├── train-gt.csv                 # Ground truth labels
  ├── scorer.py                    # Evaluation script
  └── data_combiner.py             # CSV utility

test/test/                         # Test images (1,350 images)
train/train/                       # Training images (4,150 images)
```

---

## 🔍 Typical Workflow

### 1. Setup (5 minutes)
```bash
# Create virtual environment
python -m venv env
.\env\Scripts\activate

# Install dependencies
pip install numpy opencv-python ultralytics pandas scikit-image matplotlib
```

### 2. Quick Test (15 minutes)
```bash
# Run automated tests
python run_tests.py
```

### 3. Make Predictions (varies)
```bash
cd submission\code
python solution.py ..\..\test\test ..\..\output.csv
```

### 4. Evaluate Results (5 minutes)
```bash
cd train-sb
python scorer.py --predictions ../output.csv --ground-truth train-gt.csv
```

### 5. Train New Model (1-2 hours)
```bash
cd yolo
python train_yolo.py train --epochs 50
```

---

## 📥 Input Format

**Directory Structure:**
```
data/
├── altitude01/
│   ├── longitude05/
│   │   ├── orientation01_light01.png
│   │   ├── orientation01_light02.png
│   │   └── ...
│   └── longitude06/
└── altitude02/
```

**Image Files:**
- Format: PNG
- Size: Variable (256×256 to 2048×2048)
- Color: RGB or Grayscale

---

## 📤 Output Format

**CSV File Structure:**
```csv
ellipseCenterX(px),ellipseCenterY(px),ellipseSemimajor(px),ellipseSemiminor(px),ellipseRotation(deg),inputImage,crater_classification
512.45,478.23,45.67,38.92,23.45,altitude01/longitude05/orientation01_light01,0
623.12,389.56,52.34,47.89,15.67,altitude01/longitude05/orientation01_light02,1
```

**Columns:**
1. **ellipseCenterX(px)**: X-coordinate (float)
2. **ellipseCenterY(px)**: Y-coordinate (float)
3. **ellipseSemimajor(px)**: Semi-major axis (float)
4. **ellipseSemiminor(px)**: Semi-minor axis (float)
5. **ellipseRotation(deg)**: Rotation angle (float, 0-180°)
6. **inputImage**: Image path (string, with forward slashes)
7. **crater_classification**: Class/type (integer)

---

## ❌ Troubleshooting

### "Model not found"
```bash
# Solution: Download or train model
cd yolo
python train_yolo.py train --epochs 10
cp runs/detect/train/weights/best.pt ../submission/code/
```

### "Out of memory"
```bash
# Use CPU instead of GPU
python solution.py --device cpu

# Reduce batch size
python solution.py --batch 4

# Reduce image size
python solution.py --imgsz 512
```

### "No detections"
```bash
# Lower confidence threshold (default 0.25)
python solution.py --conf 0.1

# Check image format
python -c "from PIL import Image; print(Image.open('path/image.png').size)"
```

### "Virtual environment not working"
```bash
# Recreate environment
rmdir /s env
python -m venv env
.\env\Scripts\activate
pip install numpy opencv-python ultralytics pandas scikit-image
```

---

## 🔗 Key Scripts Summary

| Script | Purpose | Command |
|--------|---------|---------|
| `solution.py` | Make predictions | `python submission/code/solution.py <input> <output>` |
| `train_yolo.py` | Train model | `python yolo/train_yolo.py train --epochs 50` |
| `scorer.py` | Evaluate | `python train-sb/scorer.py --predictions <csv>` |
| `data_combiner.py` | Merge CSVs | `python train-sb/data_combiner.py --input-files <files>` |

---

## 📊 Expected Performance

**Typical Results:**
- Precision: 0.82-0.88
- Recall: 0.75-0.85
- F1-Score: 0.78-0.86
- Inference: 50-200ms per image (CPU), 20-80ms (GPU)

**Output Size:**
- ~21,826 detections for test set (1,350 images)
- ~25-30 craters per image on average

---

## 🎓 Learning Resources

- **YOLO Docs**: https://docs.ultralytics.com/
- **OpenCV Docs**: https://docs.opencv.org/
- **Python Docs**: https://docs.python.org/3/
- **NumPy Docs**: https://numpy.org/doc/
- **Pandas Docs**: https://pandas.pydata.org/docs/

---

## 📞 Common Questions

**Q: How long does training take?**
A: ~1-2 hours for 50 epochs on GPU, ~4-6 hours on CPU

**Q: How long does prediction take?**
A: ~30-60 seconds for 100 images on GPU, ~3-5 minutes on CPU

**Q: Can I use my own model?**
A: Yes, replace `best.pt` in `submission/code/` with your model

**Q: How many detections is normal?**
A: 15-50 craters per image (average ~25)

**Q: What if confidence is too low?**
A: Lower `--conf` parameter (e.g., `--conf 0.1`)

---

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed (YOLO, OpenCV, NumPy)
- [ ] Model file exists (`submission/code/best.pt`)
- [ ] Test data exists (`test/test/`)
- [ ] `solution.py` runs without errors
- [ ] Output CSV generated successfully
- [ ] Output format matches specification

---

**Last Updated**: January 2026
**Version**: 1.0

For complete documentation, see **README.md**
