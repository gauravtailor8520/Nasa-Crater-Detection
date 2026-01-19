import os
import cv2
import numpy as np
from ultralytics import YOLO
import logging

# Suppress YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Load model once
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../submission/code/best.pt'))
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    model = None

def process_crop(crop_img):
    """
    Fits an ellipse to a crater crop using the logic from ff.ipynb.
    Returns: ((cx, cy), (semi_major, semi_minor), angle) or None
    """
    if crop_img is None or crop_img.size == 0:
        return None

    # Convert to grayscale if needed
    if len(crop_img.shape) == 3:
        gray_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    else:
        gray_crop = crop_img

    try:
        # ---------- PREPROCESS ----------
        blur = cv2.GaussianBlur(gray_crop, (7, 7), 0)
        clahe = cv2.createCLAHE(3.0, (8, 8))
        enhanced = clahe.apply(blur)
        
        # ---------- EDGE DETECTION ----------
        edges = cv2.Canny(enhanced, 50, 140)
        
        # Remove crop borders (10 pixels)
        h, w = edges.shape
        border = 10
        if h > 2 * border and w > 2 * border:
            edges[:border, :] = 0
            edges[-border:, :] = 0
            edges[:, :border] = 0
            edges[:, -border:] = 0
        
        # ---------- DISTANCE TRANSFORM ----------
        inv = cv2.bitwise_not(edges)
        dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
        dist = cv2.normalize(dist, None, 0, 1, cv2.NORM_MINMAX)
        
        # Find rim points (close to edge)
        rim_mask = (dist < 0.12).astype(np.uint8) * 255
        
        # ---------- COLLECT RIM POINTS ----------
        ys, xs = np.where(rim_mask > 0)
        points = np.column_stack((xs, ys))
        
        if len(points) < 30:
            return None
        
        points = points.reshape(-1, 1, 2).astype(np.int32)
        
        # ---------- FIT ELLIPSE ----------
        ellipse = cv2.fitEllipse(points)
        return ellipse

    except Exception:
        return None

def detect_craters(image_path):
    """
    Runs detection on a single image.
    Returns:
        - annotated_image: numpy array (BGR) with drawings
        - detections: list of dicts with crater info
    """
    if model is None:
        return None, []

    original_img = cv2.imread(image_path)
    if original_img is None:
        return None, []

    try:
        results = model.predict(
            source=image_path,
            imgsz=640,
            conf=0.25,
            iou=0.5,
            device="cpu",
            save=False,
            verbose=False
        )
    except Exception as e:
        print(f"Error predicting {image_path}: {e}")
        return original_img, []

    result = results[0]
    detections = []
    
    # Copy for annotation
    annotated_img = original_img.copy()

    if result.boxes is not None:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure coordinates are within image bounds
            h_img, w_img = original_img.shape[:2]
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(w_img, x2); y2 = min(h_img, y2)
            
            if x2 <= x1 or y2 <= y1:
                continue

            crop = original_img[y1:y2, x1:x2].copy()
            ellipse_data = process_crop(crop)
            
            detection_info = {
                "bbox": [x1, y1, x2, y2],
                "class": int(cls),
                "ellipse": None
            }

            if ellipse_data:
                (cx_crop, cy_crop), (W, H), angle = ellipse_data
                
                # Global coordinates
                global_cx = x1 + cx_crop
                global_cy = y1 + cy_crop
                
                detection_info["ellipse"] = {
                    "cx": global_cx,
                    "cy": global_cy,
                    "major": max(W, H),
                    "minor": min(W, H),
                    "angle": angle
                }

                # Draw Ellipse
                # cv2.ellipse takes ((cx, cy), (axes), angle)
                # axes is (half_major, half_minor) but fitEllipse returns full diameters in (W, H)
                # Actually fitEllipse returns (center, axes, angle) where axes are full widths? 
                # Docs: "The function calculates the ellipse that fits... returns box... (center, size, angle)"
                # Size is (width, height). so semi-axes are size/2.
                
                cv2.ellipse(annotated_img, ((global_cx, global_cy), (W, H), angle), (0, 255, 0), 2)
                
            else:
                # Fallbck: Draw BBox
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            detections.append(detection_info)

    return annotated_img, detections
