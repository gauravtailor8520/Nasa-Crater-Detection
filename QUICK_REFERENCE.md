# ⚡ Quick Reference Card

Essential commands and information for Crater Detection project - print or bookmark this!

---

## 🚀 Quick Start Commands

### Windows
```bash
# Setup
cd d:\Nase_Crater_Detection
python -m venv env
env\Scripts\activate
cd app
pip install -r requirements.txt

# Run
python app.py
# Open: http://localhost:5000
```

### Linux/Mac
```bash
# Setup
cd /path/to/Nase_Crater_Detection
python3 -m venv env
source env/bin/activate
cd app
pip install -r requirements.txt

# Run
python3 app.py
# Open: http://localhost:5000
```

---

## 🔌 API Quick Reference

### Upload Image (POST /detect)
```bash
curl -X POST http://localhost:5000/detect \
  -F "image=@image.jpg"
```

### Python
```python
import requests

files = {'image': open('image.jpg', 'rb')}
response = requests.post('http://localhost:5000/detect', files=files)
print(response.json())
```

### Response Format
```json
{
  "count": 3,
  "detections": [
    {
      "bbox": [x1, y1, x2, y2],
      "ellipse": {
        "cx": center_x,
        "cy": center_y,
        "major": semi_major,
        "minor": semi_minor,
        "angle": rotation
      }
    }
  ]
}
```

---

## 📊 Key Files & Locations

```
Project Root: d:\Nase_Crater_Detection\

Web App
├── app.py                 Main Flask server
├── model_utils.py         Detection functions
├── requirements.txt       Dependencies
└── static/uploads/        Image storage

Model
└── submission/code/best.pt    Trained model (50MB)

Data
├── train/                 Training images
└── test/                  Test images

Notebooks
├── FinalSolution.ipynb    Main solution
├── Yoloprediction.ipynb   YOLO experiments
└── Ellipsprediction.ipynb Ellipse fitting
```

---

## 🔧 Common Configuration Changes

### Change Port
**File: app/app.py (line ~47)**
```python
app.run(debug=True, port=8080)  # Change 5000 to 8080
```

### Change Detection Confidence
**File: app/model_utils.py (line ~96)**
```python
conf=0.30,  # Lower = more detections, Higher = fewer false positives
```

### Change Upload Folder
**File: app/app.py (line ~8)**
```python
UPLOAD_FOLDER = r"C:\custom\path\uploads"
```

---

## 📦 Docker Quick Commands

```bash
# Build
docker build -t crater-detection .

# Run
docker run -p 5000:5000 crater-detection

# Run with volume mount
docker run -p 5000:5000 -v $(pwd)/uploads:/app/static/uploads crater-detection

# Docker Compose
docker-compose up -d
docker-compose logs -f
docker-compose down
```

---

## 🤖 Model Training Quick Start

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='data.yaml',
    epochs=50,
    batch=16,
    imgsz=640,
    device=0
)

# Validate
metrics = model.val()

# Predict
results = model.predict('image.jpg')
```

---

## 🐛 Troubleshooting Checklist

### App won't start
- [ ] Is virtual environment activated? (`(env)` should be in prompt)
- [ ] Are all packages installed? (`pip list`)
- [ ] Does model file exist? (`ls submission/code/best.pt`)
- [ ] Is port 5000 free? (Check with `netstat -ano | findstr :5000`)

### Detection not working
- [ ] Check browser console (F12)
- [ ] Check Flask logs in terminal
- [ ] Verify image format (JPG, PNG supported)
- [ ] Check image size (reasonable satellite images)

### Installation failed
- [ ] Upgrade pip: `python -m pip install --upgrade pip`
- [ ] Try individually: `pip install flask` then `pip install ultralytics`
- [ ] Check internet connection
- [ ] Use headless OpenCV: `pip install opencv-python-headless`

---

## 📈 Performance Metrics

| Operation | CPU | GPU |
|-----------|-----|-----|
| Single image | 5-10s | 1-2s |
| Batch (10 images) | 50-100s | 10-20s |
| Model loading | ~2s | ~3s |

---

## 🔗 Important URLs

| Purpose | URL |
|---------|-----|
| Web App | http://localhost:5000 |
| API Endpoint | http://localhost:5000/detect |
| TensorBoard | http://localhost:6006 |
| YOLOv8 Docs | https://docs.ultralytics.com/ |

---

## 📚 Documentation Files (In Order)

1. **DOCUMENTATION_INDEX.md** ← You are here
2. **README.md** - Start here for overview
3. **SETUP_GUIDE.md** - Installation instructions
4. **API_DOCUMENTATION.md** - HTTP API reference
5. **ARCHITECTURE.md** - System design
6. **CODE_REFERENCE.md** - Code details
7. **DEPLOYMENT_GUIDE.md** - Production deployment
8. **MODEL_TRAINING_GUIDE.md** - Model development

---

## 🎯 30-Second Summary

**What**: Lunar crater detection using AI (YOLOv8)  
**How**: Upload satellite image → YOLO detects craters → Ellipse fitting for shape  
**Output**: Crater locations + ellipse parameters  
**Tech**: Python, Flask, YOLO, OpenCV  
**Access**: http://localhost:5000 (web) or REST API

---

## 💻 Python One-Liners

```python
# Test detection
from app.model_utils import detect_craters; img, dets = detect_craters('test.jpg'); print(f'Found {len(dets)} craters')

# Check YOLO
from ultralytics import YOLO; m = YOLO('submission/code/best.pt'); print('✓ Model loaded')

# Flask test
from app.app import app; c = app.test_client(); print(c.get('/').status_code)

# Verify install
import flask, cv2, numpy, ultralytics; print('✓ All packages installed')
```

---

## 📞 Key Contacts/Links

- **Project**: Nase Crater Detection
- **Framework**: YOLOv8 (https://github.com/ultralytics/ultralytics)
- **Python**: 3.8+ required
- **Main Files**: app.py, model_utils.py, best.pt

---

## ✅ Verification Commands

```bash
# Check Python
python --version

# Check packages
pip list | grep -E "flask|ultralytics|opencv"

# Check model
ls -lh submission/code/best.pt

# Test Flask
python -c "import flask; print('✓ Flask OK')"

# Test YOLO
python -c "from ultralytics import YOLO; print('✓ YOLO OK')"

# Test OpenCV
python -c "import cv2; print('✓ OpenCV OK')"
```

---

## 🔐 Security Checklist

- [ ] Disable debug mode in production (`debug=False`)
- [ ] Use WSGI server (gunicorn, not Flask dev)
- [ ] Set `MAX_CONTENT_LENGTH` for upload limit
- [ ] Enable HTTPS/SSL in production
- [ ] Validate file types on upload
- [ ] Set appropriate file permissions
- [ ] Use environment variables for secrets
- [ ] Keep dependencies updated

---

## 🚀 Deployment Checklist

- [ ] Run tests locally
- [ ] Build Docker image
- [ ] Test in Docker container
- [ ] Push to registry (Docker Hub, ECR, etc.)
- [ ] Deploy to cluster (K8s, ECS, Cloud Run, etc.)
- [ ] Setup monitoring & logging
- [ ] Configure auto-scaling
- [ ] Test load and failover
- [ ] Document deployment process

---

## 📝 Log File Locations

| Type | Location |
|------|----------|
| Flask logs | Console output |
| Model training | `runs/train/exp1/` |
| Results | `crater_detection_output/` |
| Uploads | `app/static/uploads/` |

---

## 🎓 Learning Path (Choose One)

**30 min**: Setup + Quick test  
→ [SETUP_GUIDE.md](SETUP_GUIDE.md) + [Quick Start](README.md#-quick-start)

**2 hours**: Understand codebase  
→ [README.md](README.md) + [CODE_REFERENCE.md](CODE_REFERENCE.md) + [ARCHITECTURE.md](ARCHITECTURE.md)

**3 hours**: Deploy to production  
→ [SETUP_GUIDE.md](SETUP_GUIDE.md) + [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

**2 hours**: Integrate API  
→ [API_DOCUMENTATION.md](API_DOCUMENTATION.md) + Examples

**4 hours**: Train custom model  
→ [MODEL_TRAINING_GUIDE.md](MODEL_TRAINING_GUIDE.md) + Hands-on

---

## 🎯 Success Indicators

✅ **Installation Complete**
- Virtual environment created
- All packages installed
- Flask starts without errors

✅ **Running**
- Web interface loads
- Can upload image
- Detections are returned

✅ **API Working**
- POST /detect returns JSON
- Annotated image saved
- Ellipse parameters correct

✅ **Production Ready**
- Running in Docker
- Using gunicorn/WSGI
- Monitoring active
- Logging configured

---

## 🔄 Common Workflows

### Workflow 1: Quick Test
```
Setup env → Install deps → Run Flask → Upload image → Verify output
```
**Time**: 15 minutes

### Workflow 2: Develop Feature
```
Setup env → Read CODE_REFERENCE → Edit code → Test locally → Verify
```
**Time**: 1-2 hours

### Workflow 3: Deploy to Cloud
```
Local test → Build Docker → Push to registry → Deploy manifests → Verify
```
**Time**: 1-2 hours

### Workflow 4: Train Model
```
Prepare data → Create data.yaml → Run training → Evaluate → Deploy
```
**Time**: 2-4 hours

---

## 📱 Mobile/API Integration

```python
# Quick API wrapper
import requests

class CraterDetector:
    def __init__(self, api_url="http://localhost:5000"):
        self.api_url = api_url
    
    def detect(self, image_path):
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(
                f"{self.api_url}/detect",
                files=files
            )
        return response.json()

# Usage
detector = CraterDetector()
result = detector.detect('satellite.jpg')
print(f"Found {result['count']} craters")
```

---

## 🆘 Emergency Commands

```bash
# Restart everything
docker-compose down && docker-compose up -d

# Clear all uploads
rm -rf app/static/uploads/*

# Reset to known-good state
git checkout app/ submission/

# Kill Flask
pkill -f "python app.py"

# Check system resources
docker stats
ps aux | grep python
```

---

## 💡 Pro Tips

1. Use `export FLASK_ENV=production` for production
2. Enable HTTPS with reverse proxy (nginx)
3. Add authentication if exposing publicly
4. Implement request queuing for high load
5. Monitor GPU memory if using GPU
6. Use Redis for caching
7. Enable CDN for static files
8. Set up alerts for errors

---

## 📊 Monitoring Dashboard

Essential metrics to track:
- Request latency (p50, p95, p99)
- Requests per second
- Error rate
- Model inference time
- CPU/Memory usage
- Disk space (uploads)
- Model accuracy

---

**Quick Reference Version**: 1.0  
**Print friendly**: YES  
**Last Updated**: January 2026
