"""
YOLO Model Training Script for Crater Detection
Trains YOLOv8 models with configurable parameters and comprehensive logging
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

from ultralytics import YOLO
import torch


def get_device():
    """Detect available device (GPU or CPU)"""
    if torch.cuda.is_available():
        device = 0  # Use GPU 0
        print(f"✓ GPU detected: {torch.cuda.get_device_name(device)}")
        print(f"  CUDA version: {torch.version.cuda}")
    else:
        device = "cpu"
        print("⚠ No GPU detected. Using CPU (slower training)")
    return device


def train_crater_yolo(
    data_yaml: Path = Path("yolo/crater.yaml"),
    model_name: str = "yolov8n",
    epochs: int = 100,
    imgsz: int = 1024,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    patience: int = 20,
    project_dir: Path = Path("yolo/runs"),
    run_name: str = None,
    device: int | str = 0,
    workers: int = 4,
    augment: bool = True,
    resume: bool = False,
    weights_path: str = None,
) -> dict:
    """
    Train YOLO model for crater detection
    
    Args:
        data_yaml: Path to crater.yaml dataset config
        model_name: YOLO model variant (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x)
        epochs: Number of training epochs
        imgsz: Input image size (square)
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        patience: Early stopping patience
        project_dir: Directory to save runs
        run_name: Name of the run (auto-generated if None)
        device: GPU device ID or "cpu"
        workers: Number of data loader workers
        augment: Enable data augmentation
        resume: Resume from last checkpoint
        weights_path: Path to pretrained weights (if not using model_name)
    
    Returns:
        dict: Training results including metrics and paths
    """
    
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_yaml}")
    
    # Generate run name with timestamp if not provided
    if run_name is None:
        run_name = f"crater-{model_name}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print("\n" + "="*70)
    print("YOLO CRATER DETECTION TRAINING")
    print("="*70)
    print(f"Model:          {model_name}")
    print(f"Dataset:        {data_yaml}")
    print(f"Image Size:     {imgsz}x{imgsz}")
    print(f"Batch Size:     {batch_size}")
    print(f"Epochs:         {epochs}")
    print(f"Learning Rate:  {learning_rate}")
    print(f"Early Stop Pat: {patience}")
    print(f"Augmentation:   {augment}")
    print(f"Device:         {device}")
    print(f"Workers:        {workers}")
    print(f"Run Name:       {run_name}")
    print("="*70 + "\n")
    
    # Load or create model
    if weights_path:
        print(f"Loading weights from: {weights_path}")
        model = YOLO(weights_path)
    else:
        print(f"Loading pretrained {model_name} model...")
        model = YOLO(f"{model_name}.pt")
    
    # Training configuration
    train_config = {
        "data": str(data_yaml.resolve()),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "project": str(project_dir),
        "name": run_name,
        "patience": patience,
        "save": True,
        "save_period": 10,  # Save every 10 epochs
        "verbose": True,
        "lr0": learning_rate,
        "lrf": 0.01,  # Final learning rate = lr0 * lrf
        "mosaic": 1.0 if augment else 0.0,  # Data augmentation
        "flipud": 0.5 if augment else 0.0,
        "fliplr": 0.5 if augment else 0.0,
        "degrees": 15 if augment else 0.0,
        "translate": 0.1 if augment else 0.0,
        "scale": 0.5 if augment else 0.0,
        "perspective": 0.0,
        "hsv_h": 0.015 if augment else 0.0,
        "hsv_s": 0.7 if augment else 0.0,
        "hsv_v": 0.4 if augment else 0.0,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "box": 7.5,  # Box loss weight
        "cls": 0.5,  # Class loss weight
        "dfl": 1.5,  # DFL loss weight
        "iou": 0.7,  # NMS IoU threshold
        "kobj": 1.0,  # Object detection loss gain
        "resume": resume,
        "plots": True,  # Save training plots
        "exist_ok": False,
    }
    
    print("Starting training...")
    print("-" * 70)
    
    # Train the model
    results = model.train(**train_config)
    
    print("-" * 70)
    print("\n✓ Training completed!")
    
    # Get results directory
    run_dir = project_dir / run_name
    results_dict = {
        "run_name": run_name,
        "model": model_name,
        "epochs": epochs,
        "run_directory": str(run_dir),
        "weights_best": str(run_dir / "weights" / "best.pt"),
        "weights_last": str(run_dir / "weights" / "last.pt"),
        "results_csv": str(run_dir / "results.csv"),
    }
    
    print(f"\nResults saved to: {run_dir}")
    print(f"Best weights:    {run_dir / 'weights' / 'best.pt'}")
    print(f"Last weights:    {run_dir / 'weights' / 'last.pt'}")
    
    return results_dict


def validate_model(
    weights_path: Path,
    data_yaml: Path = Path("yolo/crater.yaml"),
    imgsz: int = 1024,
    device: int | str = 0,
    split: str = "val",
) -> dict:
    """
    Validate trained model on test/val set
    
    Args:
        weights_path: Path to trained model weights
        data_yaml: Path to dataset config
        imgsz: Image size for validation
        device: GPU device ID or "cpu"
        split: Dataset split to validate on (val or test)
    
    Returns:
        dict: Validation metrics
    """
    print("\n" + "="*70)
    print("MODEL VALIDATION")
    print("="*70)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    model = YOLO(str(weights_path))
    
    print(f"Validating on {split} set...")
    print(f"Weights: {weights_path}")
    print(f"Image Size: {imgsz}x{imgsz}")
    print("-" * 70)
    
    results = model.val(
        data=str(data_yaml),
        imgsz=imgsz,
        device=device,
        split=split,
        verbose=True,
    )
    
    print("-" * 70)
    print("✓ Validation completed!")
    
    metrics = {
        "map50": float(results.box.map50) if hasattr(results.box, 'map50') else None,
        "map": float(results.box.map) if hasattr(results.box, 'map') else None,
        "precision": float(results.box.mp) if hasattr(results.box, 'mp') else None,
        "recall": float(results.box.mr) if hasattr(results.box, 'mr') else None,
    }
    
    print("\nValidation Metrics:")
    for key, val in metrics.items():
        if val is not None:
            print(f"  {key}: {val:.4f}")
    
    return metrics


def export_model(
    weights_path: Path,
    export_format: str = "onnx",
    imgsz: int = 1024,
    device: int | str = 0,
) -> Path:
    """
    Export trained model to different formats
    
    Args:
        weights_path: Path to trained model weights
        export_format: Format to export to (onnx, torchscript, tflite, pb, etc.)
        imgsz: Image size
        device: GPU device ID or "cpu"
    
    Returns:
        Path: Path to exported model
    """
    print("\n" + "="*70)
    print(f"EXPORTING MODEL TO {export_format.upper()}")
    print("="*70)
    
    model = YOLO(str(weights_path))
    
    print(f"Exporting {weights_path}...")
    print(f"Format: {export_format}")
    print("-" * 70)
    
    exported_path = model.export(
        format=export_format,
        imgsz=imgsz,
        device=device,
    )
    
    print("-" * 70)
    print(f"✓ Model exported to: {exported_path}")
    
    return Path(exported_path)


def predict_on_images(
    weights_path: Path,
    image_paths: list,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    imgsz: int = 1024,
    device: int | str = 0,
    save_results: bool = True,
    save_dir: Path = Path("yolo/predictions"),
) -> dict:
    """
    Run inference on images
    
    Args:
        weights_path: Path to trained model weights
        image_paths: List of image file paths
        conf_threshold: Confidence threshold for detections
        iou_threshold: IOU threshold for NMS
        imgsz: Image size
        device: GPU device ID or "cpu"
        save_results: Whether to save annotated images
        save_dir: Directory to save predictions
    
    Returns:
        dict: Prediction results
    """
    print("\n" + "="*70)
    print("RUNNING INFERENCE")
    print("="*70)
    
    model = YOLO(str(weights_path))
    
    print(f"Model: {weights_path}")
    print(f"Images: {len(image_paths)}")
    print(f"Confidence: {conf_threshold}")
    print(f"IOU: {iou_threshold}")
    print("-" * 70)
    
    results = model.predict(
        source=image_paths,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        device=device,
        save=save_results,
        project=str(save_dir),
        name="detections",
        verbose=True,
    )
    
    print("-" * 70)
    print(f"✓ Inference completed! Results saved to {save_dir}")
    
    # Summarize detections
    total_detections = sum(len(r.boxes) for r in results)
    print(f"Total detections: {total_detections}")
    
    return {
        "total_images": len(image_paths),
        "total_detections": total_detections,
        "results": results,
        "save_dir": str(save_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train, validate, and test YOLO crater detection model"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train a YOLO model")
    train_parser.add_argument("--data", type=Path, default=Path("yolo/crater.yaml"), help="Dataset config YAML")
    train_parser.add_argument("--model", type=str, default="yolov8n", help="YOLO model (yolov8n/s/m/l/x)")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    train_parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    train_parser.add_argument("--batch", type=int, default=8, help="Batch size")
    train_parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    train_parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    train_parser.add_argument("--project", type=Path, default=Path("yolo/runs"), help="Project directory")
    train_parser.add_argument("--name", type=str, default=None, help="Run name")
    train_parser.add_argument("--device", default="0", help="Device (0 for GPU or 'cpu')")
    train_parser.add_argument("--workers", type=int, default=4, help="Data loader workers")
    train_parser.add_argument("--no-augment", action="store_true", help="Disable augmentation")
    train_parser.add_argument("--resume", action="store_true", help="Resume training")
    
    # Validation command
    val_parser = subparsers.add_parser("validate", help="Validate a trained model")
    val_parser.add_argument("--weights", type=Path, required=True, help="Path to weights file")
    val_parser.add_argument("--data", type=Path, default=Path("yolo/crater.yaml"), help="Dataset config YAML")
    val_parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    val_parser.add_argument("--device", default="0", help="Device (0 for GPU or 'cpu')")
    val_parser.add_argument("--split", type=str, default="val", help="Dataset split (val or test)")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export trained model")
    export_parser.add_argument("--weights", type=Path, required=True, help="Path to weights file")
    export_parser.add_argument("--format", type=str, default="onnx", help="Export format (onnx/torchscript/etc)")
    export_parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    export_parser.add_argument("--device", default="0", help="Device (0 for GPU or 'cpu')")
    
    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Run inference on images")
    pred_parser.add_argument("--weights", type=Path, required=True, help="Path to weights file")
    pred_parser.add_argument("--images", type=str, nargs="+", required=True, help="Image file paths")
    pred_parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    pred_parser.add_argument("--iou", type=float, default=0.45, help="IOU threshold")
    pred_parser.add_argument("--imgsz", type=int, default=1024, help="Image size")
    pred_parser.add_argument("--device", default="0", help="Device (0 for GPU or 'cpu')")
    pred_parser.add_argument("--save-dir", type=Path, default=Path("yolo/predictions"), help="Save directory")
    
    args = parser.parse_args()
    
    if args.command == "train":
        device = get_device()
        try:
            result = train_crater_yolo(
                data_yaml=args.data,
                model_name=args.model,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch_size=args.batch,
                learning_rate=args.lr,
                patience=args.patience,
                project_dir=args.project,
                run_name=args.name,
                device=device,
                workers=args.workers,
                augment=not args.no_augment,
                resume=args.resume,
            )
            print("\nTraining Summary:")
            for key, val in result.items():
                print(f"  {key}: {val}")
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return 1
    
    elif args.command == "validate":
        device = get_device()
        try:
            metrics = validate_model(
                weights_path=args.weights,
                data_yaml=args.data,
                imgsz=args.imgsz,
                device=device,
                split=args.split,
            )
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            return 1
    
    elif args.command == "export":
        device = get_device()
        try:
            exported = export_model(
                weights_path=args.weights,
                export_format=args.format,
                imgsz=args.imgsz,
                device=device,
            )
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return 1
    
    elif args.command == "predict":
        device = get_device()
        try:
            image_list = []
            for img_path in args.images:
                if os.path.isdir(img_path):
                    image_list.extend(Path(img_path).glob("*.png"))
                    image_list.extend(Path(img_path).glob("*.jpg"))
                else:
                    image_list.append(img_path)
            
            results = predict_on_images(
                weights_path=args.weights,
                image_paths=image_list,
                conf_threshold=args.conf,
                iou_threshold=args.iou,
                imgsz=args.imgsz,
                device=device,
                save_dir=args.save_dir,
            )
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return 1
    
    else:
        parser.print_help()
        return 1
    
    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    exit(main())
