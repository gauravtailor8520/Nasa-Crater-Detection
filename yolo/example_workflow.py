"""
Complete example workflow for crater YOLO training
Shows: prepare -> train -> validate -> predict -> export
"""

from pathlib import Path
import subprocess
import sys

def run_command(cmd, description):
    """Run shell command and print status"""
    print(f"\n{'='*70}")
    print(f"STEP: {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"❌ FAILED: {description}")
        return False
    print(f"✓ SUCCESS: {description}")
    return True


def main():
    """Full training workflow example"""
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent.parent
    
    # Step 1: Prepare dataset
    print("\n" + "="*70)
    print("CRATER YOLO COMPLETE WORKFLOW")
    print("="*70)
    
    # Check dataset preparation
    dataset_dir = project_root / "yolo" / "dataset"
    if not (dataset_dir / "images" / "train").exists():
        if not run_command(
            f"cd {project_root} && python yolo/prepare_yolo.py --gt train-sb/train-gt.csv --images-root train/train --out yolo/dataset --train-ratio 0.85",
            "Prepare YOLO Dataset"
        ):
            return 1
    else:
        print("\n✓ Dataset already prepared, skipping preparation")
    
    # Step 2: Train model
    if not run_command(
        f"cd {project_root} && python yolo/train_yolo.py train --model yolov8n --epochs 50 --batch 8 --imgsz 1024 --patience 15",
        "Train YOLOv8n Model"
    ):
        return 1
    
    # Find the latest run
    runs_dir = project_root / "yolo" / "runs" / "detect"
    if runs_dir.exists():
        latest_run = sorted(runs_dir.glob("*"))[-1]
        best_weights = latest_run / "weights" / "best.pt"
        
        # Step 3: Validate model
        if best_weights.exists():
            if not run_command(
                f"cd {project_root} && python yolo/train_yolo.py validate --weights {best_weights} --data yolo/crater.yaml --imgsz 1024",
                "Validate Model"
            ):
                print("⚠ Validation failed but continuing...")
        
        # Step 4: Predict on sample images
        sample_images = list((project_root / "train" / "train" / "altitude08" / "longitude15").glob("orientation*.png"))[:3]
        if sample_images:
            images_str = " ".join(str(img) for img in sample_images)
            if not run_command(
                f"cd {project_root} && python yolo/train_yolo.py predict --weights {best_weights} --images {images_str} --conf 0.25 --imgsz 1024",
                "Run Inference on Sample Images"
            ):
                print("⚠ Inference failed but continuing...")
        
        # Step 5: Export model
        if best_weights.exists():
            if not run_command(
                f"cd {project_root} && python yolo/train_yolo.py export --weights {best_weights} --format onnx --imgsz 1024",
                "Export Model to ONNX"
            ):
                print("⚠ Export failed but continuing...")
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE!")
    print("="*70)
    print(f"""
Results saved to:
  - Training: {project_root / 'yolo' / 'runs' / 'detect'}
  - Predictions: {project_root / 'yolo' / 'predictions'}
  - Exported: {project_root / 'yolo' / 'runs' / 'detect' / '*' / 'weights' / 'best.onnx'}

Next steps:
  1. Check results in the runs directory
  2. Fine-tune hyperparameters if needed
  3. Train on full dataset with optimized parameters
  4. Deploy exported model in your application
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
