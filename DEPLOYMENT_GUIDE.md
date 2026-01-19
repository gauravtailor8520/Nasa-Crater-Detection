# 🚀 Deployment Guide

Complete guide for deploying the Crater Detection system to production environments.

---

## 📋 Pre-Deployment Checklist

- [ ] All tests passing
- [ ] Model file verified (best.pt exists and is valid)
- [ ] Dependencies listed in requirements.txt
- [ ] No debug mode enabled in production
- [ ] Environment variables configured
- [ ] SSL certificates ready (if using HTTPS)
- [ ] Monitoring setup completed
- [ ] Backup strategy in place

---

## 🐳 Docker Deployment

### Step 1: Create Dockerfile

Location: `Dockerfile` in project root

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY app/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app/ /app/
COPY submission/code/best.pt /app/best.pt

# Create upload directory
RUN mkdir -p /app/static/uploads

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000')"

# Run application with gunicorn (production server)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]
```

### Step 2: Create .dockerignore

```
env/
__pycache__/
*.pyc
.git
.gitignore
*.egg-info
dist/
build/
.env
.DS_Store
test/
train/
*.ipynb
.jupyter/
```

### Step 3: Build Docker Image

```bash
docker build -t crater-detection:1.0 .

# Tag for registry (if using Docker Hub)
docker tag crater-detection:1.0 yourusername/crater-detection:1.0
```

### Step 4: Run Docker Container

**Development:**
```bash
docker run -p 5000:5000 \
  -v $(pwd)/app/static/uploads:/app/static/uploads \
  crater-detection:1.0
```

**Production:**
```bash
docker run -d \
  --name crater-detection \
  --restart unless-stopped \
  -p 5000:5000 \
  -e FLASK_ENV=production \
  -v crater-uploads:/app/static/uploads \
  crater-detection:1.0
```

### Step 5: Docker Compose Setup

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  crater-detection:
    build: .
    container_name: crater-detection-app
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - WORKERS=4
    volumes:
      - ./app/static/uploads:/app/static/uploads
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

  # Optional: Redis for caching
  redis:
    image: redis:7-alpine
    container_name: crater-cache
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
```

**Start with Docker Compose:**
```bash
docker-compose up -d
docker-compose logs -f crater-detection
```

---

## ☸️ Kubernetes Deployment

### Step 1: Create Namespace

```bash
kubectl create namespace crater-detection
kubectl config set-context --current --namespace=crater-detection
```

### Step 2: Push Image to Registry

```bash
# Tag image
docker tag crater-detection:1.0 gcr.io/PROJECT_ID/crater-detection:1.0

# Push to Google Container Registry
docker push gcr.io/PROJECT_ID/crater-detection:1.0

# Or Docker Hub
docker tag crater-detection:1.0 yourusername/crater-detection:1.0
docker push yourusername/crater-detection:1.0
```

### Step 3: Create Kubernetes Manifests

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crater-detection
  namespace: crater-detection
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crater-detection
  template:
    metadata:
      labels:
        app: crater-detection
    spec:
      containers:
      - name: crater-detection
        image: yourusername/crater-detection:1.0
        ports:
        - containerPort: 5000
        env:
        - name: FLASK_ENV
          value: "production"
        - name: WORKERS
          value: "4"
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /
            port: 5000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /
            port: 5000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: uploads
          mountPath: /app/static/uploads
      volumes:
      - name: uploads
        emptyDir: {}
```

**service.yaml:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: crater-detection-service
  namespace: crater-detection
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 5000
    protocol: TCP
  selector:
    app: crater-detection
```

**ingress.yaml:**
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: crater-detection-ingress
  namespace: crater-detection
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - crater-detection.example.com
    secretName: crater-detection-tls
  rules:
  - host: crater-detection.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: crater-detection-service
            port:
              number: 80
```

### Step 4: Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# View logs
kubectl logs -f deployment/crater-detection
```

### Step 5: Scale Deployment

```bash
# Scale to 5 replicas
kubectl scale deployment crater-detection --replicas=5

# Auto-scaling based on CPU
kubectl autoscale deployment crater-detection --min=2 --max=10 --cpu-percent=80
```

---

## 🔗 AWS Deployment

### Step 1: Push to ECR

```bash
# Create ECR repository
aws ecr create-repository --repository-name crater-detection

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account_id>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag crater-detection:1.0 <account_id>.dkr.ecr.us-east-1.amazonaws.com/crater-detection:1.0
docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/crater-detection:1.0
```

### Step 2: Deploy with ECS

**task-definition.json:**
```json
{
  "family": "crater-detection",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "containerDefinitions": [
    {
      "name": "crater-detection",
      "image": "<account_id>.dkr.ecr.us-east-1.amazonaws.com/crater-detection:1.0",
      "portMappings": [
        {
          "containerPort": 5000,
          "hostPort": 5000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "FLASK_ENV",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/crater-detection",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**Deploy:**
```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster crater-detection \
  --service-name crater-detection-service \
  --task-definition crater-detection:1 \
  --desired-count 3 \
  --launch-type FARGATE
```

### Step 3: Setup ALB (Application Load Balancer)

```bash
# Create load balancer
aws elbv2 create-load-balancer \
  --name crater-detection-alb \
  --subnets subnet-xxxxx subnet-yyyyy

# Create target group
aws elbv2 create-target-group \
  --name crater-detection-tg \
  --protocol HTTP \
  --port 5000 \
  --vpc-id vpc-xxxxx
```

---

## 🌐 Google Cloud Deployment

### Step 1: Push to GCR

```bash
# Configure Docker
gcloud auth configure-docker

# Tag image
docker tag crater-detection:1.0 gcr.io/PROJECT_ID/crater-detection:1.0

# Push
docker push gcr.io/PROJECT_ID/crater-detection:1.0
```

### Step 2: Deploy to Cloud Run

```bash
gcloud run deploy crater-detection \
  --image gcr.io/PROJECT_ID/crater-detection:1.0 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --max-instances 10
```

### Step 3: Deploy to GKE

```bash
# Create cluster
gcloud container clusters create crater-detection-cluster \
  --zone us-central1-a \
  --num-nodes 3

# Deploy
kubectl apply -f deployment.yaml

# Expose service
kubectl expose deployment crater-detection \
  --type=LoadBalancer \
  --port=80 \
  --target-port=5000
```

---

## 🔐 Production Configuration

### Step 1: Environment Variables

Create `.env` file (do NOT commit):
```env
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your-secret-key-here
UPLOAD_FOLDER=/var/uploads
MODEL_PATH=/models/best.pt
LOG_LEVEL=INFO
```

### Step 2: WSGI Server Setup

Use Gunicorn instead of Flask development server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Production config:
```bash
gunicorn \
  --workers 4 \
  --worker-class sync \
  --bind 0.0.0.0:5000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  app:app
```

### Step 3: Reverse Proxy (Nginx)

```nginx
upstream crater_detection {
    server 127.0.0.1:5000;
    server 127.0.0.1:5001;
    server 127.0.0.1:5002;
}

server {
    listen 80;
    server_name crater-detection.example.com;

    client_max_body_size 50M;

    location / {
        proxy_pass http://crater_detection;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
    }

    location /static/ {
        alias /var/www/crater-detection/static/;
        expires 30d;
    }
}
```

### Step 4: SSL/HTTPS

Using Let's Encrypt:

```bash
# Install certbot
apt-get install certbot python3-certbot-nginx

# Get certificate
certbot certonly --standalone -d crater-detection.example.com

# Update Nginx config
server {
    listen 443 ssl http2;
    ssl_certificate /etc/letsencrypt/live/crater-detection.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/crater-detection.example.com/privkey.pem;
}
```

---

## 📊 Monitoring & Logging

### Step 1: Setup Prometheus

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'crater-detection'
    static_configs:
      - targets: ['localhost:8000']
```

### Step 2: Add Logging

In `app.py`:
```python
import logging
from logging.handlers import RotatingFileHandler

# Setup logging
if not app.debug:
    file_handler = RotatingFileHandler('crater_detection.log', maxBytes=10240000, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Crater Detection Application Started')
```

### Step 3: CloudWatch Logging (AWS)

```python
import watchtower
import logging

# Setup CloudWatch
handler = watchtower.CloudWatchLogHandler()
app.logger.addHandler(handler)
```

---

## 🧪 Pre-Deployment Testing

### Load Testing

```bash
# Install Apache Bench
apt-get install apache2-utils

# Run test
ab -n 1000 -c 100 -p image.jpg http://localhost:5000/detect
```

### Stress Testing

```python
import requests
import concurrent.futures
import time

def load_test(num_requests=100, concurrent=10):
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent) as executor:
        futures = []
        start = time.time()
        
        for _ in range(num_requests):
            with open('test.jpg', 'rb') as f:
                files = {'image': f}
                futures.append(executor.submit(
                    requests.post,
                    'http://localhost:5000/detect',
                    files=files
                ))
        
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        elapsed = time.time() - start
        
        success = sum(1 for r in results if r.status_code == 200)
        print(f"Completed {num_requests} requests in {elapsed:.2f}s")
        print(f"Success rate: {success/num_requests*100:.1f}%")
        print(f"Avg response: {elapsed/num_requests*1000:.1f}ms")

load_test(num_requests=1000, concurrent=50)
```

---

## 🔄 Continuous Deployment

### GitHub Actions Workflow

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -t crater-detection:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          docker login -u ${{ secrets.DOCKER_USER }} -p ${{ secrets.DOCKER_PASS }}
          docker tag crater-detection:${{ github.sha }} yourusername/crater-detection:latest
          docker push yourusername/crater-detection:latest
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/crater-detection \
            crater-detection=yourusername/crater-detection:latest
```

---

## 🔧 Maintenance & Monitoring

### Health Checks

```bash
# Check service status
curl http://localhost:5000/

# Monitor metrics
docker stats crater-detection

# View logs
docker logs -f crater-detection
```

### Backup Strategy

```bash
# Backup uploaded images
tar -czf crater-uploads-$(date +%Y%m%d).tar.gz app/static/uploads/

# Backup database (if using)
mysqldump crater_db > crater-db-$(date +%Y%m%d).sql
```

### Recovery Procedures

```bash
# Rollback to previous version
docker pull yourusername/crater-detection:1.0
docker run -d crater-detection:1.0

# Restore from backup
tar -xzf crater-uploads-20250119.tar.gz
```

---

## 📈 Auto-Scaling Configuration

### Horizontal Pod Autoscaling (Kubernetes)

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: crater-detection-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: crater-detection
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 🎓 Troubleshooting Deployment

### Issue: Container exits with error
```bash
docker logs crater-detection
# Check for missing model file or dependencies
```

### Issue: High memory usage
```bash
# Reduce batch size
# Add memory limits in manifest
# Use smaller YOLO model
```

### Issue: Slow response times
```bash
# Check CPU usage
# Increase worker count
# Use GPU if available
# Add caching layer
```

---

**Deployment Version**: 1.0  
**Last Updated**: January 2026
