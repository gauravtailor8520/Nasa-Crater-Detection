# 🔌 API Documentation

Complete API reference for the Crater Detection Web Service.

---

## Base URL

```
http://localhost:5000
```

---

## Endpoints

### 1. **GET /** - Web Interface

Returns the HTML interface for crater detection.

**Request:**
```bash
GET http://localhost:5000/
```

**Response:**
- HTTP Status: `200 OK`
- Content-Type: `text/html`
- Body: HTML page with upload form

**Example (Browser):**
```
Simply navigate to http://localhost:5000
```

---

### 2. **POST /detect** - Detect Craters

Main endpoint for crater detection. Accepts an image file and returns detection results.

#### Request

```bash
curl -X POST http://localhost:5000/detect \
  -F "image=@path/to/image.jpg"
```

**Parameters:**

| Name | Type | Required | Description |
|------|------|----------|-------------|
| image | File (multipart) | Yes | Image file (JPG, PNG, BMP supported) |

**Supported Formats:**
- JPG/JPEG (recommended)
- PNG
- BMP
- WebP

**Maximum File Size:**
- Default: 16MB (configurable in app.py)

#### Response

**Success Response (HTTP 200):**
```json
{
  "original_url": "/static/uploads/upload_1705695234_satellite.jpg",
  "processed_url": "/static/uploads/processed_1705695234_satellite.jpg",
  "detections": [
    {
      "bbox": [120, 95, 280, 210],
      "class": 0,
      "ellipse": {
        "cx": 200,
        "cy": 152,
        "major": 85.5,
        "minor": 72.3,
        "angle": 23.5
      }
    },
    {
      "bbox": [350, 180, 480, 290],
      "class": 0,
      "ellipse": {
        "cx": 415,
        "cy": 235,
        "major": 72.0,
        "minor": 68.5,
        "angle": 45.2
      }
    }
  ],
  "count": 2
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| original_url | string | URL to uploaded image |
| processed_url | string | URL to annotated image with detections |
| detections | array | Array of detected craters |
| count | integer | Number of craters detected |

**Detection Object:**

| Field | Type | Description |
|-------|------|-------------|
| bbox | array[4] | Bounding box [x1, y1, x2, y2] in pixels |
| class | integer | Object class (0 = crater) |
| ellipse | object | Fitted ellipse parameters |

**Ellipse Object:**

| Field | Type | Description |
|-------|------|-------------|
| cx | float | Center X coordinate (pixels) |
| cy | float | Center Y coordinate (pixels) |
| major | float | Semi-major axis length (pixels) |
| minor | float | Semi-minor axis length (pixels) |
| angle | float | Rotation angle (degrees, 0-180) |

#### Error Responses

**No Image Uploaded (HTTP 400):**
```json
{
  "error": "No image uploaded"
}
```

**No File Selected (HTTP 400):**
```json
{
  "error": "No selected file"
}
```

**Model Load Failure (HTTP 500):**
```json
{
  "error": "Model failed to load or process image"
}
```

**Unsupported File Type (HTTP 400):**
```json
{
  "error": "File type not supported"
}
```

---

## 📝 Request Examples

### Python (requests)
```python
import requests
import json

# Prepare image
files = {'image': open('satellite.jpg', 'rb')}

# Send request
response = requests.post('http://localhost:5000/detect', files=files)

# Parse response
if response.status_code == 200:
    data = response.json()
    print(f"Detected {data['count']} craters")
    for detection in data['detections']:
        ellipse = detection['ellipse']
        print(f"  Crater at ({ellipse['cx']}, {ellipse['cy']})")
else:
    print(f"Error: {response.json()['error']}")
```

### JavaScript (Fetch API)
```javascript
async function detectCraters(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);

  try {
    const response = await fetch('http://localhost:5000/detect', {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const error = await response.json();
      console.error('Error:', error.error);
      return;
    }

    const data = await response.json();
    console.log(`Found ${data.count} craters`);
    
    data.detections.forEach(detection => {
      console.log(`Crater: ${detection.ellipse.cx}, ${detection.ellipse.cy}`);
    });
  } catch (error) {
    console.error('Request failed:', error);
  }
}
```

### cURL
```bash
# Basic request
curl -X POST http://localhost:5000/detect \
  -F "image=@satellite.jpg"

# Save response to file
curl -X POST http://localhost:5000/detect \
  -F "image=@satellite.jpg" \
  -o response.json

# Verbose output
curl -v -X POST http://localhost:5000/detect \
  -F "image=@satellite.jpg"
```

### Batch Processing with Python
```python
import requests
import os
from pathlib import Path

def batch_detect(image_dir):
    """Process all images in directory"""
    results = {}
    
    for image_path in Path(image_dir).glob('*.jpg'):
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post('http://localhost:5000/detect', files=files)
            
            if response.status_code == 200:
                results[image_path.name] = response.json()
            else:
                results[image_path.name] = {'error': response.json()['error']}
    
    return results

# Process all images
results = batch_detect('./test_images')
for filename, data in results.items():
    if 'error' not in data:
        print(f"{filename}: {data['count']} craters detected")
    else:
        print(f"{filename}: {data['error']}")
```

---

## 🔌 Static File Access

### Get Uploaded Image
```
GET /static/uploads/upload_1705695234_satellite.jpg
```

### Get Processed Image
```
GET /static/uploads/processed_1705695234_satellite.jpg
```

---

## ⚙️ Response Content-Types

| Endpoint | Content-Type |
|----------|--------------|
| GET / | text/html |
| POST /detect | application/json |
| GET /static/* | image/jpeg, image/png (varies) |

---

## 🔒 Error Handling

### Common HTTP Status Codes

| Code | Meaning | Reason |
|------|---------|--------|
| 200 | OK | Detection successful |
| 400 | Bad Request | Missing file, wrong format |
| 413 | Payload Too Large | File exceeds size limit |
| 500 | Server Error | Model failure, system error |
| 503 | Service Unavailable | Server temporarily down |

### Error Response Structure
```json
{
  "error": "Human-readable error message"
}
```

---

## 📊 Data Flow Diagram

```
Client
   │
   ├─→ Upload Image (multipart/form-data)
   │
   ▼
Flask Server (app.py)
   │
   ├─→ Save uploaded image
   ├─→ Load YOLO model (model_utils.py)
   ├─→ Run detection
   ├─→ Process crops (ellipse fitting)
   ├─→ Annotate image
   ├─→ Save processed image
   │
   ▼
Response (JSON + Image URLs)
   │
   └─→ Client displays results
```

---

## 🧪 Testing the API

### Test 1: Simple Upload
```bash
# Upload and check response
curl -X POST http://localhost:5000/detect \
  -F "image=@test.jpg" | python -m json.tool
```

### Test 2: Check Response Time
```bash
# Measure response time
curl -w "Response Time: %{time_total}s\n" \
  -X POST http://localhost:5000/detect \
  -F "image=@test.jpg" > /dev/null
```

### Test 3: Load Testing
```python
import requests
import time
from concurrent.futures import ThreadPoolExecutor

def send_request():
    with open('test.jpg', 'rb') as f:
        files = {'image': f}
        return requests.post('http://localhost:5000/detect', files=files)

# Send 10 concurrent requests
with ThreadPoolExecutor(max_workers=10) as executor:
    start = time.time()
    futures = [executor.submit(send_request) for _ in range(10)]
    results = [f.result() for f in futures]
    elapsed = time.time() - start
    
print(f"10 requests completed in {elapsed:.2f}s")
print(f"Success: {sum(1 for r in results if r.status_code == 200)}/10")
```

---

## 🔄 Rate Limiting (Optional)

To add rate limiting:

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@app.route('/detect', methods=['POST'])
@limiter.limit("10 per minute")
def detect():
    # ... detection code
```

---

## 📈 Performance Metrics

### Expected Response Times (CPU)
- **Small image (640×480)**: 2-5 seconds
- **Medium image (1280×960)**: 5-10 seconds
- **Large image (2560×1920)**: 15-30 seconds

### Expected Response Times (GPU)
- **Small image**: 0.5-1 second
- **Medium image**: 1-2 seconds
- **Large image**: 2-5 seconds

### Memory Usage per Request
- **Peak**: ~500MB
- **Average**: ~300MB
- **Cleanup**: Automatic after response

---

## 🔐 Security Considerations

### File Upload Security
```python
# Validate file type
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Check file size
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB
if len(file.read()) > MAX_FILE_SIZE:
    return error
```

### CORS Headers (for cross-origin requests)
```python
from flask_cors import CORS
CORS(app)

# Or specific origins
@app.after_request
def set_cors_headers(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    return response
```

---

## 📦 Response Caching

Add caching for repeated requests:

```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

@app.route('/detect', methods=['POST'])
@cache.cached(timeout=3600, key_prefix='detect')
def detect():
    # ... detection code
```

---

## 🎯 Use Cases

### 1. Satellite Image Analysis
```python
# Analyze satellite imagery for crater detection
for image_file in satellite_images:
    response = requests.post('http://localhost:5000/detect', 
                            files={'image': image_file})
    craters = response.json()['detections']
```

### 2. Real-time Monitoring
```python
# Continuous monitoring with periodic uploads
while True:
    image = capture_from_camera()
    response = requests.post('http://localhost:5000/detect',
                            files={'image': image})
    if response.json()['count'] > threshold:
        trigger_alert()
```

### 3. Batch Analysis Pipeline
```python
# Process dataset and store results
for image_path in dataset:
    response = requests.post('http://localhost:5000/detect',
                            files={'image': open(image_path, 'rb')})
    store_results(image_path, response.json())
```

---

## 🚀 Integration Examples

### Integration with Data Processing Pipeline
```python
import requests
import pandas as pd

# Send images and collect results
results_list = []
for img in image_list:
    resp = requests.post('http://localhost:5000/detect',
                        files={'image': open(img, 'rb')})
    data = resp.json()
    
    for detection in data['detections']:
        results_list.append({
            'image': img,
            'center_x': detection['ellipse']['cx'],
            'center_y': detection['ellipse']['cy'],
            'major_axis': detection['ellipse']['major'],
            'minor_axis': detection['ellipse']['minor']
        })

# Convert to DataFrame
df = pd.DataFrame(results_list)
df.to_csv('crater_detections.csv', index=False)
```

---

**API Version**: 1.0  
**Last Updated**: January 2026
