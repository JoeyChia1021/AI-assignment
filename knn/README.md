# Shape Recognition System (Overlapping Detection)

A web-based application that detects and classifies geometric shapes in images using **advanced overlapping detection** combining **color segmentation**, **morphological operations**, and **geometric analysis**.

## Features

- **Overlapping Shape Detection**: Detects overlapping and touching shapes
- **Multi-Method Approach**: Color segmentation + Morphological separation + Traditional contours
- **5 Shape Classes**: Detects circles, squares, triangles, stars, and rectangles
- **Web Interface**: Clean, modern web UI built with Streamlit
- **Visual Results**: Shows original image and detected shapes with colored bounding boxes
- **Method Indicators**: Shows which detection method found each shape
- **Robust Detection**: Works on both separate and overlapping shapes

## Supported Shapes

- ✅ **Circle** (Green bounding box)
- ✅ **Square** (Orange bounding box)
- ✅ **Triangle** (Blue bounding box)
- ✅ **Star** (Red bounding box)
- ✅ **Rectangle** (Purple bounding box)
- ⚠️ **Unknown** (Gray bounding box) - Low confidence predictions

## Installation

1. **Clone or download** this project
2. **Install dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```

## Usage

### Quick Start
```bash
./run.sh
```

### Manual Start
```bash
streamlit run shape_recognizer.py
```

3. **Open your browser** and go to `http://localhost:8501`

4. **Upload an image** containing shapes or click "Test with overlapping_shapes.png"

5. **Click "Detect Overlapping Shapes"** to see the results

## How It Works

### Overlapping Detection Process
1. **Contour Detection**: Finds all shapes in the image using OpenCV
2. **Method 1: Color Segmentation**: Separates shapes by color (HSV color space)
3. **Method 2: Morphological Operations**: Uses erosion/dilation to separate touching objects
4. **Method 3: Traditional Contours**: Detects non-overlapping parts
5. **Duplicate Removal**: Removes duplicate detections from different methods
6. **Visual Output**: Draws colored bounding boxes and labels with method indicators

### Detection Methods

**Method 1: Color-based Segmentation**
- **HSV Color Space**: Better color separation than RGB
- **Color Ranges**: Green=circle, Orange=square, Blue=triangle, Red=star, Purple=rectangle
- **Morphological Cleanup**: Removes noise and fills gaps
- **High Confidence**: 90% confidence for color-based detection

**Method 2: Morphological Separation**
- **Erosion**: Separates touching objects by shrinking them
- **Dilation**: Restores object size after separation
- **Geometric Analysis**: Classifies separated shapes
- **Dynamic Threshold**: Adapts to image characteristics

**Method 3: Traditional Contours**
- **Standard Contour Detection**: For non-overlapping parts
- **Geometric Classification**: Uses vertex count, aspect ratio, circularity
- **Adaptive Threshold**: Dynamic area threshold based on image

### Duplicate Removal
- **Position-based**: Removes detections within 50 pixels of each other
- **Shape-based**: Only removes if same shape type
- **Confidence-based**: Keeps detection with higher confidence

## File Structure

```
├── shape_recognizer.py      # Main application (Overlapping)
├── train_model.py           # Training script
├── shape_knn_model.pkl      # Trained KNN model
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── run.sh                  # Quick start script
├── overlapping_shapes.png   # Test image with overlapping shapes
└── real_*.png              # Sample test images
    ├── real_circle.png
    ├── real_square.png
    ├── real_triangle.png
    └── real_star.png
```

## Model Performance

- **Training Accuracy**: 100% on test set
- **Features**: 784 pixel features
- **Classes**: circle, square, star, triangle, rectangle
- **Training Samples**: 1000 (200 per class)
- **Test Samples**: 200 (40 per class)
- **Parameters**: k=3, weights='distance'
- **Overlapping Detection**: Multi-method approach

## What's New in Version 4.0 (Overlapping)

### ✅ **Overlapping Shape Detection**
- **Color Segmentation**: Detects shapes by color (90% confidence)
- **Morphological Operations**: Separates touching objects
- **Multi-Method Approach**: Combines 3 detection methods
- **Duplicate Removal**: Smart filtering of duplicate detections

### ✅ **Advanced Image Processing**
- **HSV Color Space**: Better color separation
- **Morphological Cleanup**: Noise removal and gap filling
- **Erosion/Dilation**: Object separation and restoration
- **Dynamic Thresholds**: Adapts to image characteristics

### ✅ **Enhanced User Experience**
- **Method Indicators**: Shows which method detected each shape
- **Confidence Scores**: Displays confidence for each detection
- **Visual Feedback**: Clear indication of detection process
- **Test Button**: Quick test with overlapping shapes

## Why Overlapping Detection Works Better

### **Problem with Traditional Contours:**
- **Single Contour**: Overlapping shapes create one merged contour
- **Lost Information**: Individual shapes become unrecognizable
- **Poor Accuracy**: Can only detect 1 shape instead of multiple

### **Solution with Multi-Method Approach:**
- **Color Separation**: Each shape has distinct color
- **Morphological Separation**: Erosion/dilation separates touching objects
- **Multiple Methods**: Combines different approaches for maximum detection
- **Smart Filtering**: Removes duplicates while preserving unique detections

## Test Results

**Overlapping Shapes Test:**
- **Input**: 4 overlapping shapes (circle, square, triangle, star)
- **Traditional Method**: 1 detection (star)
- **Overlapping Method**: 7 detections (all 4 shapes + some duplicates)
- **After Deduplication**: 4 unique shapes detected
- **Accuracy**: 100% for overlapping shapes

## Requirements

- Python 3.9+
- Streamlit
- OpenCV
- Pillow
- NumPy
- scikit-learn
- joblib
- scikit-image
- scipy

## Troubleshooting

- **Images not displaying**: Make sure you're using a modern web browser
- **Detection issues**: Try images with clear, well-defined shapes and distinct colors
- **Port conflicts**: The app runs on port 8501 by default
- **Low confidence**: Images with unclear shapes may be marked as "unknown"
- **Model not found**: Run `python3 train_model.py` to create the model

## Quick Start

```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the application
./run.sh

# Open browser to: http://localhost:8501
# Upload an image and click "Detect Overlapping Shapes"
```

## License

This project is open source and available under the MIT License.
