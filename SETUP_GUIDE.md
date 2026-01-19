# 🔧 Setup & Installation Guide

This guide provides step-by-step instructions to set up and run the Crater Detection project on different operating systems.

---

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **pip**: Package installer for Python
- **Virtual Environment**: Recommended for dependency isolation
- **RAM**: Minimum 4GB (8GB recommended for large batches)
- **Disk Space**: 2GB for dependencies and model

---

## 🪟 Windows Setup

### Step 1: Clone/Extract Project
```bash
# Navigate to where you want to store the project
cd d:\
# Project should be in d:\Nase_Crater_Detection\
```

### Step 2: Create Virtual Environment
```bash
# Open PowerShell or Command Prompt
cd d:\Nase_Crater_Detection

# Create virtual environment
python -m venv env

# Activate virtual environment
env\Scripts\activate
```

**Expected output**: Your command prompt should show `(env)` prefix

### Step 3: Upgrade pip (Important!)
```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies
```bash
# Navigate to app directory
cd app

# Install requirements
pip install -r requirements.txt

# If installation fails, try individually:
pip install flask
pip install ultralytics
pip install opencv-python-headless
pip install numpy
pip install pandas
```

### Step 5: Verify Installation
```bash
# Check if YOLO model can be imported
python -c "from ultralytics import YOLO; print('✓ YOLO installed')"

# Check OpenCV
python -c "import cv2; print('✓ OpenCV installed')"

# Check Flask
python -c "import flask; print('✓ Flask installed')"
```

### Step 6: Run Web Application
```bash
# Option 1: Run Python script directly
python app.py

# Option 2: Use batch file (from project root)
cd ..
run.bat
```

**Expected output**:
```
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

### Step 7: Access Web Interface
Open browser and navigate to: **http://localhost:5000**

---

## 🐧 Linux Setup

### Step 1: Install Python & Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3 python3-pip python3-venv

# CentOS/RHEL
sudo yum install python3 python3-pip
```

### Step 2: Navigate to Project
```bash
cd /path/to/Nase_Crater_Detection
```

### Step 3: Create Virtual Environment
```bash
python3 -m venv env
source env/bin/activate
```

**Expected output**: Prompt should show `(env)` prefix

### Step 4: Install Dependencies
```bash
cd app
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run Application
```bash
python3 app.py
```

---

## 🍎 macOS Setup

### Step 1: Install Homebrew (if not installed)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### Step 2: Install Python
```bash
brew install python3
```

### Step 3: Create Virtual Environment
```bash
cd /path/to/Nase_Crater_Detection
python3 -m venv env
source env/bin/activate
```

### Step 4: Install Dependencies
```bash
cd app
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 5: Run Application
```bash
python3 app.py
```

---

## 🐳 Docker Setup (All Platforms)

### Step 1: Install Docker
- [Docker Installation Guide](https://docs.docker.com/get-docker/)

### Step 2: Build Docker Image
```bash
cd submission/code
docker build -t crater-detection:latest .
```

### Step 3: Run Docker Container
```bash
docker run -p 5000:5000 crater-detection:latest
```

Access at: **http://localhost:5000**

### Step 4: Stop Container
```bash
# Find container ID
docker ps

# Stop container
docker stop <container_id>
```

---

## ⚙️ Advanced Configuration

### 1. Change Flask Port
**File**: `app/app.py`
```python
# Line 47
if __name__ == '__main__':
    app.run(debug=True, port=8080)  # ← Change port here
```

### 2. Disable Debug Mode (Production)
```python
if __name__ == '__main__':
    app.run(debug=False, port=5000)
```

### 3. Adjust Model Confidence
**File**: `app/model_utils.py`
```python
# Line 96 in detect_craters()
results = model.predict(
    conf=0.30,  # ← Increase for stricter detections (0.0-1.0)
    iou=0.5
)
```

### 4. Increase Upload Size Limit
**File**: `app/app.py`
```python
# Add after Flask initialization
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit
```

### 5. Change Upload Directory
**File**: `app/app.py`
```python
# Line 8
UPLOAD_FOLDER = r"C:\path\to\custom\uploads"  # Windows
UPLOAD_FOLDER = "/path/to/custom/uploads"      # Linux/Mac
```

---

## 🔍 Verification Checklist

After setup, verify everything works:

- [ ] Virtual environment activated (see `(env)` in prompt)
- [ ] All packages installed (`pip list` shows flask, ultralytics, etc.)
- [ ] Model file exists: `submission/code/best.pt`
- [ ] Static folders exist: `app/static/css`, `app/static/js`, `app/templates`
- [ ] Flask starts without errors
- [ ] Web interface loads at http://localhost:5000
- [ ] Image upload works
- [ ] Detection produces results

---

## 🚨 Common Installation Issues

### Issue 1: `ModuleNotFoundError: No module named 'torch'`
```bash
# Solution: Install PyTorch (ultralytics dependency)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue 2: `pip install` hangs
```bash
# Solution: Upgrade pip and try again
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

### Issue 3: OpenCV error on Linux
```bash
# Solution: Install dependencies
sudo apt-get install python3-opencv
pip install opencv-python-headless
```

### Issue 4: Port 5000 already in use
```bash
# Find process using port 5000
netstat -ano | findstr :5000  # Windows
lsof -i :5000                  # Linux/Mac

# Kill process or change port in app.py
```

### Issue 5: `best.pt` model not found
```bash
# Verify model location
ls submission/code/best.pt

# If missing, check if file was properly extracted from archive
# The model should be ~50MB in size
```

---

## 📦 Dependency Details

### Core Dependencies

| Package | Version | Notes |
|---------|---------|-------|
| Flask | ≥2.0 | Web framework |
| ultralytics | ≥8.0 | YOLO framework |
| opencv-python-headless | ≥4.5 | Image processing (no GUI) |
| numpy | ≥1.20 | Numerical operations |
| torch | ≥1.9 | Deep learning backend |

### Optional Dependencies

```bash
# For Jupyter notebook support
pip install jupyter ipython

# For enhanced performance
pip install pillow  # Image format support

# For GPU support (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 🔐 Security Best Practices

### 1. Set Strong Uploads Limit
```python
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB
```

### 2. Validate File Types
```python
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
if not allowed_file(file.filename):
    return error
```

### 3. Production Deployment
```bash
# Use production WSGI server instead of Flask development server
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### 4. Use HTTPS
```bash
pip install pyopenssl
# Configure SSL certificates
```

---

## 🚀 Performance Optimization

### 1. Batch Processing CPU Optimization
```python
# In model_utils.py
results = model.predict(
    device="cpu",
    batch=8,  # Adjust batch size
    verbose=False
)
```

### 2. Use GPU (if available)
```python
results = model.predict(
    device=0,  # GPU device 0
    batch=16   # Larger batch for GPU
)
```

### 3. Model Inference Speed
```bash
# Use smaller YOLO model for faster inference
# yolov8n.pt (nano) - fastest
# yolov8s.pt (small)
# yolov8m.pt (medium)
# yolov8l.pt (large)
```

---

## 📊 Memory Usage

### Typical Memory Footprint
- **Base Installation**: ~500MB
- **YOLO Model**: ~50MB
- **Python Runtime**: ~100MB
- **Per Image Processing**: ~50-200MB depending on size

### Reduce Memory Usage
```python
# Process images in smaller batches
# Reduce image resolution
# Use CPU instead of GPU for smaller memory footprint
```

---

## 🔄 Updating & Maintenance

### Update Dependencies
```bash
pip install --upgrade -r requirements.txt
```

### Check for Outdated Packages
```bash
pip list --outdated
```

### Update YOLO Model
```bash
# Download latest model
from ultralytics import YOLO
model = YOLO('yolov8n.pt')  # Downloads if not present
```

---

## 📞 Getting Help

### Check Logs
```bash
# Flask debug output shows in console
# Check browser console (F12) for frontend errors
```

### Enable Debug Mode
```python
# In app.py
app.run(debug=True)  # Shows detailed error messages
```

### Test Installation
```bash
python -c "from model_utils import detect_craters; print('✓ Ready')"
```

---

## ✅ First Run Checklist

After completing setup:

1. **Activate virtual environment**
   ```bash
   env\Scripts\activate  # Windows
   source env/bin/activate  # Linux/Mac
   ```

2. **Start Flask server**
   ```bash
   cd app
   python app.py
   ```

3. **Open web browser**
   - Navigate to `http://localhost:5000`
   - Should see upload interface

4. **Test with sample image**
   - Upload a test image
   - Check for crater detections
   - Verify ellipse fitting

5. **Check outputs**
   - Review generated images in `app/static/uploads/`
   - Check console for any errors

---

**You're all set! 🎉 Start detecting craters!**
