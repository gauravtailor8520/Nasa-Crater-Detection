from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import cv2
import time
from model_utils import detect_craters

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save original
    timestamp = int(time.time())
    filename = f"upload_{timestamp}_{file.filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Process
    annotated_img, detections = detect_craters(filepath)
    
    if annotated_img is None:
         return jsonify({'error': 'Model failed to load or process image'}), 500

    # Save processed
    processed_filename = f"processed_{timestamp}_{file.filename}"
    processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
    cv2.imwrite(processed_filepath, annotated_img)

    return jsonify({
        'original_url': f"/static/uploads/{filename}",
        'processed_url': f"/static/uploads/{processed_filename}",
        'detections': detections,
        'count': len(detections)
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
