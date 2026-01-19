@echo off
REM YOLO Crater Detection - Quick Command Reference Card
REM Copy-paste ready commands for common tasks

echo.
echo ============================================================
echo YOLO CRATER DETECTION - COMMAND REFERENCE
echo ============================================================
echo.
echo SETUP:
echo   .\env\Scripts\activate
echo   pip install ultralytics torch torchvision
echo.
echo ============================================================
echo DATASET PREPARATION:
echo.
echo Standard (80/20 split):
echo   python yolo/prepare_yolo.py
echo.
echo Custom (85/15 split):
echo   python yolo/prepare_yolo.py --train-ratio 0.85
echo.
echo ============================================================
echo QUICK TRAINING:
echo.
echo Fast (nano, good for testing):
echo   python yolo/train_yolo.py train --model yolov8n --epochs 50
echo.
echo Standard (medium, recommended):
echo   python yolo/train_yolo.py train --model yolov8m --epochs 100
echo.
echo Accurate (large, best accuracy):
echo   python yolo/train_yolo.py train --model yolov8l --epochs 150
echo.
echo One-liner batch script:
echo   .\yolo\quick_train.bat m 100
echo.
echo ============================================================
echo VALIDATION:
echo.
echo Set WEIGHTS variable first:
echo   set WEIGHTS=yolo\runs\detect\crater-yolov8m-YYYYMMDD_HHMMSS\weights\best.pt
echo.
echo Then validate:
echo   python yolo/train_yolo.py validate --weights !WEIGHTS!
echo.
echo ============================================================
echo INFERENCE / PREDICTIONS:
echo.
echo Single image:
echo   python yolo/train_yolo.py predict --weights !WEIGHTS! --images image.png
echo.
echo Directory of images:
echo   python yolo/train_yolo.py predict --weights !WEIGHTS! --images "train/train/altitude08/longitude15"
echo.
echo Multiple images with options:
echo   python yolo/train_yolo.py predict --weights !WEIGHTS! --images img1.png img2.png --conf 0.3
echo.
echo ============================================================
echo MODEL EXPORT:
echo.
echo Export to ONNX (recommended):
echo   python yolo/train_yolo.py export --weights !WEIGHTS! --format onnx
echo.
echo Export to TorchScript:
echo   python yolo/train_yolo.py export --weights !WEIGHTS! --format torchscript
echo.
echo Export to TFLite (mobile):
echo   python yolo/train_yolo.py export --weights !WEIGHTS! --format tflite
echo.
echo ============================================================
echo ADVANCED TRAINING OPTIONS:
echo.
echo Custom hyperparameters:
echo   python yolo/train_yolo.py train ^
echo     --model yolov8m ^
echo     --epochs 150 ^
echo     --batch 24 ^
echo     --imgsz 1280 ^
echo     --lr 0.0005 ^
echo     --patience 30
echo.
echo Resume training:
echo   python yolo/train_yolo.py train --model yolov8m --resume
echo.
echo CPU-only training:
echo   python yolo/train_yolo.py train --model yolov8n --device cpu
echo.
echo ============================================================
echo MONITORING:
echo.
echo List latest training run:
echo   dir yolo\runs\detect | sort /+9 | tail -1
echo.
echo View training results:
echo   type yolo\runs\detect\crater-yolov8m-*\results.csv
echo.
echo ============================================================
echo COMMON WORKFLOWS:
echo.
echo 1. Full training pipeline:
echo   python yolo/prepare_yolo.py
echo   python yolo/train_yolo.py train --model yolov8m --epochs 100
echo   set WEIGHTS=yolo\runs\detect\crater-yolov8m-*\weights\best.pt
echo   python yolo/train_yolo.py validate --weights !WEIGHTS!
echo   python yolo/train_yolo.py export --weights !WEIGHTS! --format onnx
echo.
echo 2. Quick test:
echo   python yolo/prepare_yolo.py
echo   .\yolo\quick_train.bat n 30
echo.
echo 3. Fine-tune existing model:
echo   python yolo/train_yolo.py train --model yolov8m --epochs 50 --lr 0.0001 --resume
echo.
echo ============================================================
echo TROUBLESHOOTING:
echo.
echo GPU out of memory:
echo   Reduce --batch 4 or --imgsz 640 or use smaller model (yolov8n)
echo.
echo Training too slow:
echo   Check GPU: nvidia-smi
echo   Or reduce --workers or use smaller --imgsz
echo.
echo No detections:
echo   Reduce --conf threshold (e.g., 0.1 instead of 0.25)
echo.
echo ============================================================
echo DOCUMENTATION:
echo.
echo Detailed guide:      yolo/TRAINING_GUIDE.md
echo Configuration tips:  yolo/CONFIGURATION_EXAMPLES.md
echo Quick reference:     yolo/QUICK_REFERENCE.md
echo README:              yolo/README.md
echo.
echo ============================================================
echo.

endlocal
