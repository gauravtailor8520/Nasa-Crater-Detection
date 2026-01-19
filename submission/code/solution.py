import os
import sys
import csv
import numpy as np
import cv2
from ultralytics import YOLO
import logging

# Suppress YOLO logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

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
        (cx, cy), (W, H), angle = ellipse
        
        # Semi-axes
        semi_major = max(W, H) / 2.0
        semi_minor = min(W, H) / 2.0
        
        return (cx, cy), (semi_major, semi_minor), angle

    except Exception:
        return None

def guess_detections(root_dir: str, out_path: str):
    # Load model
    model_path = os.path.join(os.path.dirname(__file__), 'best.pt')
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return

    fields = ['ellipseCenterX(px)','ellipseCenterY(px)','ellipseSemimajor(px)',
               'ellipseSemiminor(px)','ellipseRotation(deg)','inputImage',
               'crater_classification']
    
    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fields)
        writer.writeheader()
        
        cnt = 0
        for dirpath, _, files in os.walk(root_dir):
            if 'truth' in dirpath:
                continue
                
            norm = os.path.normpath(dirpath)
            dirparts = [part for part in norm.split(os.sep) if part]

            for f in files:
                if f.endswith('.png'):
                    # Build ID
                    # Assuming structure matches sample: root/altitudeX/longitudeY/file.png
                    # dirparts[-2] = altitudeX, dirparts[-1] = longitudeY
                    if len(dirparts) >= 2:
                        id = os.path.join(dirparts[-2], dirparts[-1], f[:-4])
                    else:
                        id = f[:-4] # Fallback
                    
                    print(cnt, id)
                    cnt += 1
                    
                    full_path = os.path.join(dirpath, f)
                    
                    # Run prediction
                    try:
                        results = model.predict(
                            source=full_path,
                            imgsz=640,
                            conf=0.25,
                            iou=0.5,
                            device="cpu",
                            save=False,
                            verbose=False
                        )
                    except Exception as e:
                        print(f"Error processing {full_path}: {e}", file=sys.stderr)
                        continue

                    # Process results
                    result = results[0]
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        
                        original_img = cv2.imread(full_path)
                        if original_img is None:
                            continue

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
                            
                            if ellipse_data:
                                (cx_crop, cy_crop), (semi_major, semi_minor), angle = ellipse_data
                                
                                # Global coordinates
                                global_cx = x1 + cx_crop
                                global_cy = y1 + cy_crop
                                
                                out_row = {}
                                out_row['inputImage'] = id.replace(os.sep, '/') # Ensure forward slashes for ID if needed? Sample has forward slashes.
                                out_row['ellipseCenterX(px)'] = round(global_cx, 2)
                                out_row['ellipseCenterY(px)'] = round(global_cy, 2)
                                out_row['ellipseSemimajor(px)'] = round(semi_major, 2)
                                out_row['ellipseSemiminor(px)'] = round(semi_minor, 2)
                                out_row['ellipseRotation(deg)'] = round(angle, 2)
                                out_row['crater_classification'] = str(int(cls))
                                writer.writerow(out_row)
                            else:
                                # Fallback if ellipse fit fails? 
                                # ff.ipynb skips. But maybe we should output the bbox as ellipse circle?
                                # Sample solution generates circles.
                                # For now, we mimic ff.ipynb which skips "insufficient rim points".
                                # But actually strictly speaking we detected a crater, just failed specific fit.
                                # Let's provide a rough ellipse from bbox.
                                w_box = x2 - x1
                                h_box = y2 - y1
                                global_cx = x1 + w_box / 2.0
                                global_cy = y1 + h_box / 2.0
                                semi_major = max(w_box, h_box) / 2.0
                                semi_minor = min(w_box, h_box) / 2.0
                                
                                # Just output it to be safe, maybe better than missing detection.
                                # But user asked to use ff.ipynb logic detaled. ff.ipynb skips. 
                                # "Skipping ... insufficient rim points"
                                # I will skip to strictly follow ff.ipynb logic.
                                pass

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python solution.py <root_dir> <out_path>")
        sys.exit(1)
    guess_detections(sys.argv[1], sys.argv[2])
