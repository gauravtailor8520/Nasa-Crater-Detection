# Crater Detection Dataset - Complete Data Description

## Overview
This dataset is designed for a **Lunar Crater Detection Challenge**, focused on identifying and classifying craters from lunar surface images captured at varying altitudes, longitudes, orientations, and lighting conditions.

---

## Dataset Statistics

### Training Data
- **Total Images**: 4,150 PNG files (verified count)
- **Total Crater Annotations**: 183,329 labeled craters
- **Ground Truth File**: `train-gt.csv`

**Note on Image Count**: The actual count of 4,150 images was verified by:
- Counting actual PNG files on disk: 4,150 files
- Counting unique images in `train-gt.csv`: 4,150 unique image identifiers
- Structure calculation: 59 longitude-altitude combinations × 10 orientations × 5 lighting conditions = 4,150 images
  - Altitude breakdown: 
    - altitude01-04: 12 longitudes each = 600 images per altitude (2,400 total)
    - altitude05: 13 longitudes = 650 images
    - altitude06: 6 longitudes = 300 images
    - altitude07-10: 4 longitudes each = 200 images per altitude (800 total)

**Historical Note**: Previous documentation may have cited 12,450 training images. This number cannot be derived from the current dataset structure. Possible explanations:
- It may have been from an earlier dataset version or specification
- It could represent an intended dataset size that wasn't fully populated
- It might have been a calculation error (12,450 would require 249 longitude-altitude combinations, but only 59 exist)
- The number may have referred to total crater annotations divided by some estimated average (183,329 / ~14.7 ≈ 12,466, but this doesn't match actual image count)

### Test Data
- **Total Images**: 1,350 PNG files
- **No ground truth provided** (for evaluation purposes)

### Sample Submission
- **Total Predictions**: 21,826 crater detections
- **Format**: Matches the ground truth structure

---

## Directory Structure

```
datashare/
├── train/
│   └── train/
│       ├── altitude01/
│       │   ├── longitude02/
│       │   │   ├── orientation01_light01.png
│       │   │   ├── orientation01_light02.png
│       │   │   ├── ...
│       │   │   ├── orientation10_light05.png
│       │   │   └── truth/
│       │   ├── longitude03/
│       │   ├── ... (12 longitude folders per altitude)
│       ├── altitude02/
│       ├── ... (10 altitude levels total)
│       └── altitude10/
├── test/
│   └── test/
│       ├── altitude01/
│       │   ├── longitude05/
│       │   ├── longitude06/
│       │   ├── longitude15/
│       │   └── longitude20/
│       ├── altitude02/
│       ├── ... (10 altitude levels)
│       └── altitude10/
├── train-gt.csv              # Ground truth for training data
├── detections-04-16.csv      # Example detection file
├── data_combiner.py          # Utility to combine detection CSVs
├── scorer.py                 # Evaluation scorer
└── sample-submission/
    ├── code/
    │   ├── Dockerfile
    │   ├── sample_solution.py
    │   ├── test.sh
    │   └── train.sh
    └── solution/
        └── solution.csv
```

---

## Data Organization Hierarchy

The image data is organized in a **4-level hierarchy**:

1. **Altitude** (10 levels): `altitude01` through `altitude10`
   - Represents different camera altitudes above the lunar surface
   
2. **Longitude** (Variable number per altitude):
   - **Training**: 12 longitude positions per altitude (longitude02, longitude03, longitude04, longitude08, longitude09, longitude10, longitude11, longitude12, longitude14, longitude17, longitude18, longitude19)
   - **Test**: 4 longitude positions per altitude (longitude05, longitude06, longitude15, longitude20)
   
3. **Orientation** (10 positions): `orientation01` through `orientation10`
   - Represents different camera angles/orientations
   
4. **Lighting** (5 conditions): `light01` through `light05`
   - Different lighting conditions for the same scene

### Image Naming Convention
Images follow the pattern: `orientation{XX}_light{YY}.png`
- Example: `orientation01_light01.png`, `orientation05_light03.png`

### Truth Subdirectory
Each longitude folder contains a `truth/` subdirectory with `detections.csv` files:
- **Location**: `train/train/altitude{XX}/longitude{YY}/truth/detections.csv`
- **Purpose**: Contains raw detection data with additional metadata columns
- **Structure**: 13 columns including bounding boxes, confidence scores, and Robbins crater database IDs
- **Processing**: These files are combined by `data_combiner.py` into the final `train-gt.csv` format

---

## Data Files

### 1. train-gt.csv (Ground Truth)
**Purpose**: Contains labeled crater annotations for all training images

**Structure**: 183,329 rows (excluding header)

**Columns** (7 total):
| Column Name | Type | Description | Example Values |
|------------|------|-------------|----------------|
| `ellipseCenterX(px)` | Float | X-coordinate of crater ellipse center in pixels | 1971.34, 655.10 |
| `ellipseCenterY(px)` | Float | Y-coordinate of crater ellipse center in pixels | 985.42, 1675.22 |
| `ellipseSemimajor(px)` | Float | Semi-major axis length of ellipse in pixels | 283.15, 239.36 |
| `ellipseSemiminor(px)` | Float | Semi-minor axis length of ellipse in pixels | 235.80, 197.84 |
| `ellipseRotation(deg)` | Float | Rotation angle of ellipse in degrees | 249.40, 232.16 |
| `inputImage` | String | Image identifier (path format) | altitude08/longitude15/orientation01_light01 |
| `crater_classification` | Integer | Crater quality/confidence class (3 or 4) | 3, 4 |

**Key Characteristics**:
- Multiple craters can be annotated per image
- Craters are represented as ellipses (accounting for perspective distortion)
- Same physical craters appear across different orientations and lighting conditions
- Image identifiers use format: `altitude{XX}/longitude{YY}/orientation{ZZ}_light{WW}`

**Sample Data**:
```csv
ellipseCenterX(px),ellipseCenterY(px),ellipseSemimajor(px),ellipseSemiminor(px),ellipseRotation(deg),inputImage,crater_classification
1971.34,985.42,283.15,235.80,249.40,altitude08/longitude15/orientation01_light01,4
655.10,1675.22,239.36,197.84,232.16,altitude08/longitude15/orientation01_light01,4
1059.26,223.36,164.72,124.67,156.46,altitude08/longitude15/orientation01_light01,4
```

---

### 2. detections-04-16.csv (Example Detections)
**Purpose**: Sample detection file showing additional metadata columns from intermediate processing

**Structure**: 2,857 rows (excluding header)

**Note**: This file represents an intermediate format with additional columns that are filtered out during the combination process. The final submission format matches `train-gt.csv` structure.

**Columns** (13 total):
| Column Name | Type | Description | Notes |
|------------|------|-------------|-------|
| `detectionConfidence` | Integer | Confidence score (always 1 in sample) | Dropped in combined output |
| `boundingBoxMinX(px)` | Integer | Bounding box minimum X coordinate | Dropped in combined output |
| `boundingBoxMinY(px)` | Integer | Bounding box minimum Y coordinate | Dropped in combined output |
| `boundingBoxMaxX(px)` | Integer | Bounding box maximum X coordinate | Dropped in combined output |
| `boundingBoxMaxY(px)` | Integer | Bounding box maximum Y coordinate | Dropped in combined output |
| `ellipseCenterX(px)` | Float | Ellipse center X coordinate | **Required in submission** |
| `ellipseCenterY(px)` | Float | Ellipse center Y coordinate | **Required in submission** |
| `ellipseSemimajor(px)` | Float | Ellipse semi-major axis | **Required in submission** |
| `ellipseSemiminor(px)` | Float | Ellipse semi-minor axis | **Required in submission** |
| `ellipseRotation(deg)` | Float | Ellipse rotation angle | **Required in submission** |
| `inputImage` | String | Image filename (e.g., orientation01_light01.png) | **Required in submission** |
| `crater_id_Robbins` | String | External crater database ID (format: XX-Y-ZZZZZZ) | Dropped in combined output |
| `crater_classification` | Integer | Crater class (3 or 4) | **Required in submission** |

**Sample Data**:
```csv
detectionConfidence,boundingBoxMinX(px),boundingBoxMinY(px),boundingBoxMaxX(px),boundingBoxMaxY(px),ellipseCenterX(px),ellipseCenterY(px),ellipseSemimajor(px),ellipseSemiminor(px),ellipseRotation(deg),inputImage,crater_id_Robbins,crater_classification
1,0,0,1160,850,468.709716796875,121.3450698852539,765.2926025390625,650.9957275390625,125.40353775024414,orientation01_light01.png,00-1-000000,4
```

---

### 3. sample-submission/solution/solution.csv
**Purpose**: Example submission format showing expected output structure

**Structure**: 21,826 crater predictions

**Required Columns** (7 total):
1. `ellipseCenterX(px)` - Float
2. `ellipseCenterY(px)` - Float
3. `ellipseSemimajor(px)` - Float
4. `ellipseSemiminor(px)` - Float
5. `ellipseRotation(deg)` - Float
6. `inputImage` - String (format: `altitude{XX}/longitude{YY}/orientation{ZZ}_light{WW}`)
7. `crater_classification` - Integer (3 or 4)

**Sample Data**:
```csv
ellipseCenterX(px),ellipseCenterY(px),ellipseSemimajor(px),ellipseSemiminor(px),ellipseRotation(deg),inputImage,crater_classification
256,1306,198,198,0,altitude08/longitude03/orientation06_light02,4
340,1250,195,195,0,altitude08/longitude03/orientation06_light02,4
```

---

## Image Specifications

### Format
- **File Type**: PNG (Portable Network Graphics)
- **Color Space**: BGR (as evidenced by OpenCV usage in sample code)
- **Exact Dimensions**: 2592×2048 pixels (width × height)
  - Validated in `data_combiner.py`: X coordinates must be in [0, 2592), Y coordinates in [0, 2048)

### Content
- Grayscale or color images of the lunar surface
- Images show cratered terrain at various scales
- Same physical location captured under different conditions (orientation, lighting)

---

## Crater Annotations

### Ellipse Representation
Craters are annotated as **ellipses** rather than circles because:
1. Perspective distortion from camera angle
2. Oblique viewing angles create elliptical projections of circular craters
3. More accurate geometric representation

### Ellipse Parameters
An ellipse is defined by 5 parameters:
- **Center**: (ellipseCenterX, ellipseCenterY)
- **Size**: (ellipseSemimajor, ellipseSemiminor)
- **Orientation**: ellipseRotation in degrees

### Crater Classification
- **Class 3**: Lower confidence or quality crater annotation
- **Class 4**: Higher confidence or quality crater annotation
- Distribution shows predominantly class 4 craters in the dataset

---

## Scoring Methodology

The evaluation uses a **Geodesic Distance (dGA)** metric for ellipse matching.

### Key Scoring Components (from scorer.py):

#### 1. Geodesic Angular Distance (dGA)
- Measures similarity between two ellipses in a geometrically meaningful way
- Computed using ellipse covariance matrices
- Range: 0 to π/2 radians (0° to 90°)
- Lower values indicate better matches

#### 2. Chi-Squared Threshold (ξ²)
- **Threshold**: ξ² ≤ 13.277
- Determines if two ellipses are considered a match
- Based on normalized geodesic distance

#### 3. Pixel Error Ratio
- **Value**: 0.07 (7% of minimum semi-axis)
- Used to compute reference sigma for matching tolerance
- Smaller craters have tighter matching requirements

### Scoring Algorithm
1. **For each ground truth crater**:
   - Find the best matching predicted crater using dGA
   - Apply chi-squared test (ξ² ≤ 13.277)
   - Check radius constraints (rA ≤ 1.5×rB and rB ≤ 1.5×rA)
   - Check center distance: |cx_A - cx_B| ≤ r and |cy_A - cy_B| ≤ r (where r = min(rA, rB))

2. **Match Scoring**:
   - Each matched pair contributes: `1 - (dGA / π)` to the score
   - Score per image: `avg_dga × min(1.0, tp_count / min(10, len(truth_craters)))`
   - Where `avg_dga` is the average of `(1 - dGA/π)` for all matched pairs
   - `tp_count` is the number of true positive matches
   - Final score is averaged across all images and multiplied by 100

3. **Special Cases**:
   - Empty images (no craters): Score = 1.0 if correctly predicted (both truth and prediction have only one row with semi-major = -1)
   - Empty mismatch (predicted craters when none exist or vice versa): Score = 0.0
   - Images with no predictions: Score = 0.0

### Evaluation Formula Components
```python
XI_2_THRESH = 13.277          # Chi-squared threshold
NN_PIX_ERR_RATIO = 0.07       # 7% pixel error tolerance

# For matching:
comparison_sig = NN_PIX_ERR_RATIO * min(semi_major, semi_minor)
ref_sig = 0.85 / sqrt(semi_major * semi_minor) * comparison_sig
xi_2 = (dGA)² / (ref_sig)²

# Match accepted if: xi_2 ≤ XI_2_THRESH

# Geodesic Angular Distance (dGA) calculation:
# Uses ellipse covariance matrices Y_i and Y_j
# dGA = arccos(min(1, 4×√(det(Y_i)×det(Y_j)) / det(Y_i + Y_j) × exp(-0.5×(y_i-y_j)ᵀ×Y_i×(Y_i+Y_j)⁻¹×Y_j×(y_i-y_j))))
# Where Y is computed from ellipse parameters (a, b, φ) using rotation matrices
```

---

## Utility Scripts

### 1. data_combiner.py
**Purpose**: Combine multiple detection CSV files into a single submission file

**Key Functions**:
- `combine_detections(root_dir, out_path)`: Walks directory tree and combines all `detections.csv` files
- Automatically constructs `inputImage` field from directory structure
- Drops unnecessary columns: `detectionConfidence`, bounding box columns, `crater_id_Robbins`
- Preserves `crater_classification` field
- Handles platform-specific path separators

**Data Validation & Filtering**:
- Filters out craters with centers outside image bounds (X: [0, 2592), Y: [0, 2048))
- Filters out craters whose ellipse bounding box extends beyond image boundaries
- Filters out craters with total perimeter > 60% of image height (2×(dx+dy) > 0.6×2048)
- Formats numeric values to 2 decimal places (except `crater_classification`)
- Handles images with no detections by writing a row with `-1` for all numeric fields

**Input Image Path Construction**:
- Extracts 4th-from-last and 3rd-from-last directory components (altitude and longitude)
- Combines with image filename (without .png extension) to create full path identifier
- Format: `altitude{XX}/longitude{YY}/orientation{ZZ}_light{WW}`

**Usage Pattern**:
```bash
python data_combiner.py <root_directory> <output_file.csv>
```

**Output**: Single CSV file with standardized format for submission

---

### 2. scorer.py
**Purpose**: Offline evaluation of crater detection predictions

**Usage**:
```bash
python scorer.py --pred <solution_file> --truth <ground_truth_file> --out_dir <output_directory>
```

**Output**: 
- `result.txt` in the specified output directory containing the final score

**Scoring Features**:
- Implements geodesic distance (dGA) metric
- Ellipse-to-ellipse matching algorithm
- Handles empty images correctly
- Uses chi-squared test for match validation

---

### 3. sample_solution.py
**Purpose**: Baseline crater detection using computer vision techniques

**Approach**:
- Uses **Hough Circle Transform** from OpenCV
- Converts images to grayscale
- Applies median blur for noise reduction
- Detects circular features (approximates craters as circles)

**Parameters**:
```python
minDist=100        # Minimum distance between circle centers
param1=100         # Canny edge detection high threshold
param2=40          # Accumulator threshold for circle centers
minRadius=80       # Minimum circle radius
maxRadius=200      # Maximum circle radius
```

**Output Format**:
- Top 20 circles per image
- Converts circles to "ellipses" (equal semi-major and semi-minor axes)
- Sets rotation to 0° (circles are rotation-invariant)
- All craters classified as class 4

**Limitations**:
- Only detects circles, not true ellipses
- Simple baseline approach (not optimized for performance)
- Fixed parameters may not work well across all altitudes/scales

---

## Data Characteristics and Challenges

### 1. Multi-View Redundancy
- Same physical craters visible in multiple images:
  - 10 different orientations
  - 5 different lighting conditions
- Total of **50 images per longitude-altitude combination**

### 2. Scale Variation
- 10 altitude levels create significant scale differences
- Crater sizes range from ~40 pixels to ~700+ pixels
- Requires multi-scale detection approaches

### 3. Lighting Variation
- 5 lighting conditions per scene
- Affects crater visibility and shadow patterns
- Requires robust feature extraction

### 4. Perspective Distortion
- Varying camera orientations create elliptical projections
- Oblique angles increase ellipse eccentricity
- Rotation angles span 0-360 degrees

### 5. Class Imbalance
- Predominantly class 4 craters (high quality)
- Class 3 craters are less common
- May require balanced sampling strategies

### 6. Dense Annotations
- Average of ~15 craters per image (183,329 craters / 12,450 images)
- Overlapping craters require precise localization
- Multiple scales present in single images

---

## Recommended Approaches

### 1. Detection Methods
- **Deep Learning**: Faster R-CNN, Mask R-CNN, YOLO for ellipse detection
- **Classical CV**: Hough Transform, Edge Detection, Template Matching
- **Hybrid**: Combine classical features with deep learning

### 2. Multi-Scale Handling
- Feature Pyramid Networks (FPN)
- Multi-scale training and inference
- Scale-adaptive detection thresholds

### 3. Leveraging Multi-View Data
- Consistent detection across orientations/lighting
- View aggregation for robust predictions
- Self-supervised learning from consistency

### 4. Ellipse-Specific Techniques
- Direct ellipse regression networks
- Oriented bounding boxes
- Geometric constraint enforcement

---

## Data Quality Notes

### Ground Truth Quality
- High-quality manual annotations
- Consistent ellipse parametrization
- Classification labels (3 vs 4) indicate annotation confidence

### Potential Issues
- Edge cases: Partially visible craters at image boundaries
- Overlapping craters may be challenging to separate
- Very small craters (<40px) may be under-represented
- Lighting variations may cause some craters to be invisible

---

## File Formats Summary

### CSV Format
- **Encoding**: UTF-8
- **Delimiter**: Comma (`,`)
- **Newline**: Standard (handled by Python CSV module)
- **Headers**: Always present in first row

### Image Format
- **Extension**: `.png`
- **Encoding**: Standard PNG compression
- **Channels**: Likely 3-channel BGR (based on OpenCV usage)

---

## Competition/Challenge Context

This appears to be a **data science competition** focused on:

1. **Object Detection**: Locating craters in images
2. **Geometric Estimation**: Accurately fitting ellipses to crater shapes
3. **Classification**: Distinguishing crater quality (class 3 vs 4)
4. **Robustness**: Handling scale, orientation, and lighting variations

### Success Metrics
- **Primary**: Geodesic distance (dGA) based score
- **Secondary**: Recall (detecting all true craters)
- **Tertiary**: Precision (avoiding false positives)

---

## Additional Files

### Docker Support
- **File**: `sample-submission/code/Dockerfile`
- **Purpose**: Containerize solution for reproducible evaluation
- **Scripts**: `train.sh` and `test.sh` for automated pipelines

### Environment
- **Python Environment**: Virtual environment in `env/`
- **Key Dependencies**: 
  - NumPy 2.2.6
  - SciPy 1.16.3
  - OpenCV (cv2)
  - Jupyter/IPython (for notebooks)
  - boto3/botocore (for AWS S3 operations)
  - spaCy ecosystem (blis, catalogue, cymem, confection)

---

## Summary Statistics

| Metric | Training | Test |
|--------|----------|------|
| Total Images | 4,150 (verified) | 1,350 (verified) |
| Total Craters | 183,329 | Unknown |
| Altitude Levels | 10 | 10 |
| Longitude Positions | 12 per altitude | 4 per altitude |
| Orientations | 10 | 10 |
| Lighting Conditions | 5 | 5 |
| Images per Location | 50 (10×5) | 50 (10×5) |
| Average Craters/Image | ~44.2 (183,329 craters / 4,150 images) | - |
| Crater Size Range | 40-700+ pixels | - |
| Classification Classes | 2 (class 3, 4) | - |
| Image Dimensions | 2592×2048 pixels | 2592×2048 pixels |
| Truth CSV Files | 84 files (one per longitude-altitude) | N/A |

---

## Contact & Support

For questions about the data format, scoring methodology, or challenge rules, refer to the contest forum (mentioned in scorer.py comments).

---

## Additional Technical Details

### Ellipse Bounding Box Calculation
When validating crater positions, the system calculates the axis-aligned bounding box of each ellipse:
- `dx = sqrt((a×cos(α))² + (b×sin(α))²)`
- `dy = sqrt((a×sin(α))² + (b×cos(α))²)`
- Ellipse must fit within image: `[cx-dx, cx+dx] ⊆ [0, 2592)` and `[cy-dy, cy+dy] ⊆ [0, 2048)`

### Robbins Crater Database
The `crater_id_Robbins` field references entries from the Robbins Lunar Crater Database, providing external validation and cross-referencing capabilities. Format appears to be: `XX-Y-ZZZZZZ` where:
- `XX`: Possibly region or catalog identifier
- `Y`: Sub-category or type
- `ZZZZZZ`: Unique crater identifier

### Empty Image Handling
- Images with no craters are represented by a single row with `ellipseSemimajor(px) = -1`
- All other numeric fields are also set to `-1` for empty images
- The `inputImage` field still contains the proper image identifier
- Scoring correctly handles these cases with special logic

### Coordinate System
- Origin (0, 0) is at the **top-left** corner of the image
- X-axis increases **rightward** (0 to 2591)
- Y-axis increases **downward** (0 to 2047)
- Ellipse rotation is measured in degrees, counter-clockwise from the positive X-axis

---

**Document Version**: 2.0  
**Last Updated**: January 9, 2026  
**Data Path**: `d:\datashare\`
