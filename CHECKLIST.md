# Quick Checklist - Get Started in 30 Minutes

## ✅ Pre-Flight Checklist (5 minutes)

- [ ] Python 3.8+ installed
  ```powershell
  python --version
  ```
  
- [ ] Internet connection available (for downloading packages)

- [ ] 5GB free disk space

- [ ] Enough RAM (4GB minimum)

---

## ⚙️ Setup Checklist (10 minutes)

### Step 1: Create Virtual Environment
```powershell
cd d:\datashare
python -m venv env
```
- [ ] `env` folder created

### Step 2: Activate Environment
```powershell
.\env\Scripts\Activate.ps1
```
- [ ] Terminal shows `(env)` prefix

### Step 3: Install Packages
```powershell
pip install numpy opencv-python ultralytics pandas scikit-image matplotlib
```
- [ ] Installation completes without errors

### Step 4: Verify Installation
```powershell
python -c "from ultralytics import YOLO; print('OK')"
```
- [ ] Output shows "OK"

---

## 🧪 Testing Checklist (10 minutes)

### Option 1: Quick Test
```powershell
python run_tests.py
```
- [ ] Script runs successfully
- [ ] Output shows "✓ PASSED"
- [ ] Detections found (>1000)
- [ ] `test_output/predictions.csv` created

### Option 2: Manual Test
```powershell
cd submission\code
python solution.py ..\..\test\test ..\..\output.csv
```
- [ ] Script completes without errors
- [ ] Output file created
- [ ] Output file has content (>100 lines)

---

## 📊 Verification Checklist

After testing:

- [ ] Output CSV exists
- [ ] CSV has header row
- [ ] CSV has data rows (>1000)
- [ ] Format is correct (7 columns)
- [ ] File can be opened in Excel/CSV viewer

### Check CSV Structure
```powershell
# View first few lines
Get-Content output.csv -Head 5

# Count rows
(Get-Content output.csv | Measure-Object -Line).Lines
```

Expected columns:
1. ellipseCenterX(px)
2. ellipseCenterY(px)
3. ellipseSemimajor(px)
4. ellipseSemiminor(px)
5. ellipseRotation(deg)
6. inputImage
7. crater_classification

---

## 🎯 Optional: Evaluation Checklist (5 minutes)

If you want to evaluate against ground truth:

```powershell
cd train-sb
python scorer.py --predictions ../output.csv --ground-truth train-gt.csv
```

- [ ] Scorer runs successfully
- [ ] Precision/Recall displayed
- [ ] F1-Score displayed

---

## 📚 Documentation Checklist

- [ ] Read SETUP_GUIDE.md
- [ ] Bookmarked QUICKSTART.md
- [ ] Understand basic workflow
- [ ] Know how to make predictions
- [ ] Know where to find help

---

## 🚀 Ready to Go!

If all items above are checked:

✓ **Your system is set up correctly**
✓ **You can make predictions**
✓ **You know where to find help**

---

## 📋 Troubleshooting Quick Check

### Issue: Tests fail or no output

**Quick fixes to try:**

1. Reactivate environment
   ```powershell
   .\env\Scripts\Activate.ps1
   ```

2. Reinstall packages
   ```powershell
   pip install --upgrade numpy opencv-python ultralytics
   ```

3. Check Python version
   ```powershell
   python --version
   ```

4. Check model file exists
   ```powershell
   ls -la submission\code\best.pt
   ```

### Issue: "ModuleNotFoundError"

```powershell
# Verify you're in virtual environment (should show (env) prefix)
# If not, run: .\env\Scripts\Activate.ps1

# Reinstall:
pip install numpy opencv-python ultralytics pandas scikit-image
```

### Issue: Memory error

```powershell
# Use CPU instead:
python submission\code\solution.py --device cpu test\test output.csv

# Or reduce batch size:
python submission\code\solution.py --batch 4 test\test output.csv
```

---

## 📞 Next Steps

1. **Run predictions**: 
   ```powershell
   python submission\code\solution.py test\test output.csv
   ```

2. **Check results**:
   ```powershell
   # View in Excel or:
   Get-Content output.csv | head -20
   ```

3. **For more help**:
   - See QUICKSTART.md for commands
   - See README.md for detailed info
   - See SETUP_GUIDE.md for step-by-step help

---

## ✨ Success Indicators

You're all set when:

✓ run_tests.py completes successfully
✓ Predictions generate without errors
✓ Output CSV has >1000 rows
✓ Output format is correct
✓ You can open CSV in Excel/viewer

---

## 🎓 Quick Commands Reference

```powershell
# SETUP
python -m venv env
.\env\Scripts\Activate.ps1
pip install numpy opencv-python ultralytics pandas scikit-image matplotlib

# TEST
python run_tests.py

# PREDICT
cd submission\code
python solution.py ..\..\test\test ..\..\output.csv

# EVALUATE
python train-sb\scorer.py --predictions output.csv --ground-truth train-sb\train-gt.csv

# TRAIN
cd yolo
python train_yolo.py train --epochs 50

# JUPYTER
jupyter notebook
```

---

## ⏱️ Time Estimates

| Task | Time |
|------|------|
| Pre-flight check | 5 min |
| Setup | 10 min |
| Testing | 10 min |
| Verification | 5 min |
| **Total** | **30 min** |

---

## 🎯 You Can Now:

- [ ] Run crater detection predictions
- [ ] Generate output CSV files
- [ ] Understand the data format
- [ ] Troubleshoot basic issues
- [ ] Train models (optional)
- [ ] Evaluate results (optional)

---

**Ready? Start with**: `python run_tests.py`

**Need help?** See [QUICKSTART.md](QUICKSTART.md) or [SETUP_GUIDE.md](SETUP_GUIDE.md)

---

**Date**: January 2026
**Version**: 1.0
