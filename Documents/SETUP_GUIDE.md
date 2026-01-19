# SETUP & EXECUTION GUIDE

## 🎯 Complete Step-by-Step Instructions

This guide walks you through everything needed to run the crater detection system from start to finish.

---

## ✅ Pre-Requisites (5 minutes)

### System Requirements
- **OS**: Windows 10+, Linux, or macOS
- **RAM**: 4GB minimum (8GB+ recommended)
- **Disk Space**: 5GB free (for models and data)
- **GPU** (optional): NVIDIA GPU with CUDA support (3-4x faster)

### Software Requirements
- **Python**: 3.8 or higher
- **Git** (optional): For version control

### Verification

Check Python installation:
```powershell
python --version
```

Expected output:
```
Python 3.10.x or higher
```

---

## 📦 STEP 1: Environment Setup (10 minutes)

### 1.1 Open Terminal

**Windows (PowerShell):**
- Press `Win + X` → PowerShell
- Or search for "PowerShell" in Start menu

**Windows (Command Prompt):**
- Press `Win + R` → type `cmd` → Enter

**Linux/Mac:**
- Open Terminal from applications

### 1.2 Navigate to Project

```powershell
cd d:\datashare
```

Or wherever you extracted the project.

### 1.3 Create Virtual Environment

```powershell
# Create environment
python -m venv env

# Activate (Windows PowerShell)
.\env\Scripts\Activate.ps1

# Activate (Windows CMD)
.\env\Scripts\activate.bat

# Activate (Linux/Mac)
source env/bin/activate
```

**Expected output after activation:**
```
(env) d:\datashare>
```

The `(env)` prefix indicates the environment is active.

### 1.4 Verify Virtual Environment

```powershell
python --version
python -m pip --version
```

---

## 📚 STEP 2: Install Dependencies (10-15 minutes)

### 2.1 Upgrade pip

```powershell
python -m pip install --upgrade pip
```

### 2.2 Install Packages

```powershell
pip install numpy opencv-python ultralytics pandas scikit-image matplotlib jupyter ipykernel
```

**This installs:**
- **numpy**: Numerical computing
- **opencv-python**: Image processing
- **ultralytics**: YOLO v8 framework
- **pandas**: Data manipulation
- **scikit-image**: Image algorithms
- **matplotlib**: Plotting/visualization
- **jupyter**: Interactive notebooks
- **ipykernel**: Jupyter kernel

### 2.3 Verify Installation

```powershell
# Test each package
python -c "import numpy; print('NumPy: OK')"
python -c "import cv2; print('OpenCV: OK')"
python -c "from ultralytics import YOLO; print('YOLO: OK')"
python -c "import pandas; print('Pandas: OK')"
```

Expected output:
```
NumPy: OK
OpenCV: OK
YOLO: OK
Pandas: OK
```

---

## 🚀 STEP 3: Quick Test (5-10 minutes)

### 3.1 Automated Test Script

**Option A: Using Python (Recommended)**

```powershell
python run_tests.py
```

This will:
1. ✓ Verify Python version
2. ✓ Check dependencies
3. ✓ Locate model file
4. ✓ Run sample predictions
5. ✓ Generate output CSV
6. ✓ Display results

**Option B: Using Batch Script (Windows Only)**

```powershell
.\run_tests.bat
```

### 3.2 Expected Output

```
============================================
  LUNAR CRATER DETECTION SYSTEM
============================================

[STEP 1/7] Checking Python version...
[✓ OK] Python 3.10.12

[STEP 2/7] Setting up virtual environment...
[✓ OK] Virtual environment exists

[STEP 3/7] Verifying dependencies...
  ✓ YOLO
  ✓ OpenCV
  ✓ NumPy
  ✓ Pandas
[✓ OK] All dependencies verified

[STEP 4/7] Checking model file...
[✓ OK] Model file found (50.5 MB)

[STEP 5/7] Preparing test environment...
[✓ OK] Test output directory ready

[STEP 6/7] Running predictions...
[✓ OK] Predictions complete (21827 detections)

============================================
  SAMPLE OUTPUT
============================================
...output preview...

============================================
  TEST SUMMARY
============================================
Overall Status: ✓ PASSED
Total Detections: 21827
Output File: test_output/predictions.csv
```

---

## 📊 STEP 4: Run Full Predictions (Variable)

### 4.1 Basic Prediction

```powershell
cd submission\code
python solution.py ..\..\test\test ..\..\output.csv
```

**Parameters:**
- **Argument 1**: Input directory path
- **Argument 2**: Output CSV file path

### 4.2 Prediction with Custom Settings

```powershell
# Use CPU instead of GPU
python solution.py ..\..\test\test output.csv --device cpu

# Lower confidence threshold (more detections)
python solution.py ..\..\test\test output.csv --conf 0.1

# Smaller image size (faster, less accurate)
python solution.py ..\..\test\test output.csv --imgsz 512

# Reduce batch size (less memory)
python solution.py ..\..\test\test output.csv --batch 4
```

### 4.3 Monitor Progress

The script will display progress:
```
Processing: 0 altitude01/longitude05/orientation01_light01
Processing: 1 altitude01/longitude05/orientation01_light02
Processing: 2 altitude01/longitude05/orientation01_light03
...
```

**Typical speed:**
- CPU: 100-150 images/minute
- GPU: 300-500 images/minute

### 4.4 View Results

**Option A: View in Excel/Spreadsheet**
```powershell
# Windows
start output.csv

# Or open manually: right-click → Open with → Excel
```

**Option B: View in Terminal**
```powershell
# First 20 lines
Get-Content output.csv -Head 20

# Last 10 lines
Get-Content output.csv -Tail 10

# Count detections
(Get-Content output.csv | Measure-Object -Line).Lines - 1
```

**Option C: View with Python**
```powershell
python -c "
import pandas as pd
df = pd.read_csv('output.csv')
print(f'Detections: {len(df)}')
print(f'Average crater size: {df['ellipseSemimajor(px)'].mean():.1f} px')
print(f'Images processed: {df['inputImage'].nunique()}')
print(df.head())
"
```

---

## ✅ STEP 5: Evaluate Results (Optional, 5 minutes)

### 5.1 Against Ground Truth

```powershell
cd train-sb
python scorer.py --predictions ../output.csv --ground-truth train-gt.csv
```

**Output:**
```
Evaluation Results:
===================
Precision: 0.85
Recall: 0.78
F1-Score: 0.81
Mean IoU: 0.79
```

---

## 🎓 STEP 6: Train Custom Model (Optional, 1-2 hours)

### 6.1 Quick Training (10 epochs)

```powershell
cd yolo
python train_yolo.py train --epochs 10
```

### 6.2 Full Training (50 epochs)

```powershell
python train_yolo.py train --epochs 50 --batch 8
```

### 6.3 Monitor Training

Real-time progress:
```
Epoch 1/50: 100%|██████| 100/100 [00:45<00:00, 2.21 it/s]
Epoch 2/50: 100%|██████| 100/100 [00:44<00:00, 2.25 it/s]
...
```

### 6.4 Use Trained Model

```powershell
# After training, model is at:
# yolo/runs/detect/train/weights/best.pt

# Copy to submission directory
Copy-Item "yolo\runs\detect\train\weights\best.pt" -Destination "submission\code\best.pt"

# Test with new model
cd submission\code
python solution.py ..\..\test\test output_new.csv
```

---

## 📓 STEP 7: Interactive Exploration (Optional)

### 7.1 Start Jupyter

```powershell
jupyter notebook
```

Browser opens automatically (http://localhost:8888)

### 7.2 Open Notebooks

Available notebooks:

1. **FinalSolution.ipynb**
   - Complete end-to-end pipeline
   - Best for understanding entire workflow

2. **Yoloprediction.ipynb**
   - YOLO detection details
   - Visualization of detections

3. **Ellipsprediction.ipynb**
   - Ellipse fitting algorithm
   - Parameter extraction

4. **Testsolution.ipynb**
   - Solution testing and validation
   - Performance metrics

5. **Testdata.ipynb**
   - Data exploration and analysis

### 7.3 Run Notebook Cells

- Click cell to select
- Press `Shift + Enter` to run
- Or click ▶ (Run) button

---

## 🔄 COMPLETE WORKFLOW SUMMARY

```
Start
  ↓
[Step 1] Environment Setup (10 min)
  ├─ Create venv
  └─ Activate venv
  ↓
[Step 2] Install Dependencies (15 min)
  ├─ pip install packages
  └─ Verify installation
  ↓
[Step 3] Quick Test (10 min)
  ├─ Run: python run_tests.py
  └─ Check output
  ↓
[Step 4] Make Predictions (varies)
  ├─ Run: python solution.py <input> <output>
  └─ Generate CSV
  ↓
[Step 5] Evaluate (Optional, 5 min)
  ├─ Compare with ground truth
  └─ View metrics
  ↓
[Step 6] Train Model (Optional, 1-2 hr)
  ├─ Run: python train_yolo.py train
  └─ Evaluate performance
  ↓
[Step 7] Explore (Optional)
  ├─ Open Jupyter
  └─ Run notebooks
  ↓
End
```

---

## 📋 TROUBLESHOOTING

### Problem: "Python not found"
```
Solution:
1. Restart terminal
2. Verify Python is in PATH: echo %PATH%
3. Reinstall Python if needed
```

### Problem: "Virtual environment won't activate"
```
Solution:
1. Delete env folder: rmdir /s env
2. Recreate: python -m venv env
3. Activate again: .\env\Scripts\Activate.ps1
```

### Problem: "Module not found (numpy, cv2, etc.)"
```
Solution:
1. Verify venv is activated: (env) should show
2. Reinstall: pip install numpy opencv-python ultralytics
3. Check installation: python -c "import numpy; print(numpy.__version__)"
```

### Problem: "Model file not found"
```
Solution 1: Download model (if available)
Solution 2: Quick train:
  cd yolo
  python train_yolo.py train --epochs 10
  cp runs/detect/train/weights/best.pt ../submission/code/
```

### Problem: "Out of memory / CUDA out of memory"
```
Solution:
1. Use CPU: python solution.py --device cpu
2. Reduce batch: python solution.py --batch 4
3. Smaller images: python solution.py --imgsz 512
```

### Problem: "Predictions too slow"
```
Solution:
1. Use GPU (if available)
2. Reduce image size: --imgsz 512
3. Increase batch: --batch 32
```

---

## 🆘 Getting Help

1. **Check README.md**: Comprehensive documentation
2. **Check QUICKSTART.md**: Command reference
3. **Review notebooks**: Step-by-step examples
4. **Check DATA_DESCRIPTION.md**: Data format details

---

## ✨ Success Indicators

After completing all steps, you should have:

- ✓ Virtual environment created and activated
- ✓ All dependencies installed
- ✓ Test script runs successfully
- ✓ Predictions generated in `output.csv`
- ✓ CSV contains 15,000+ crater detections
- ✓ Output format matches specification
- ✓ (Optional) Evaluated results with scorer
- ✓ (Optional) Trained custom model

---

## 📞 Quick Reference

| Task | Command |
|------|---------|
| Create venv | `python -m venv env` |
| Activate venv | `.\env\Scripts\Activate.ps1` |
| Install packages | `pip install numpy opencv-python ultralytics pandas scikit-image` |
| Run test | `python run_tests.py` |
| Make predictions | `python submission\code\solution.py test\test output.csv` |
| Evaluate | `python train-sb\scorer.py --predictions output.csv` |
| Train model | `python yolo\train_yolo.py train --epochs 50` |
| Start Jupyter | `jupyter notebook` |

---

**Next Steps:**
1. Follow Step 1 through Step 5 above
2. Check `test_output/predictions.csv` for results
3. Review README.md for advanced usage
4. Explore notebooks for detailed explanations

**Good luck!** 🚀
