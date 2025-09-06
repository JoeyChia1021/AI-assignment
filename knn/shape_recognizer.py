import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import joblib
import os

# Load simple KNN model
@st.cache_data
def load_simple_algorithm():
    algorithm_path = "trained_algorithm.pkl"
    if os.path.exists(algorithm_path):
        try:
            algorithm = joblib.load(algorithm_path)
            return algorithm, True
        except Exception as e:
            # st.error(f"Algorithm loading error: {e}")
            return None, False
    else:
        # st.error("Simple algorithm not found")
        return None, False

def robust_contour_detection(cv_image):
    """Robust contour detection with multiple methods"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Method 1: Standard threshold
        _, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Method 2: Adaptive threshold
        adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        contours2, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Method 3: Otsu threshold
        _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours3, _ = cv2.findContours(otsu_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Method 4: Lower threshold for light images
        _, thresh4 = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        contours4, _ = cv2.findContours(thresh4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Method 5: Higher threshold for dark images
        _, thresh5 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        contours5, _ = cv2.findContours(thresh5, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Combine all contours
        all_contours = contours1 + contours2 + contours3 + contours4 + contours5
        
        # Remove duplicates and filter
        filtered_contours = []
        seen_areas = set()
        
        for contour in all_contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Lower area threshold
                # Check for duplicates (similar area and position)
                x, y, w, h = cv2.boundingRect(contour)
                area_key = (area // 1000, x // 50, y // 50)  # Rough grouping
                
                if area_key not in seen_areas:
                    seen_areas.add(area_key)
                    filtered_contours.append(contour)
        
        return filtered_contours
        
    except Exception as e:
        st.write(f"Contour detection error: {e}")
        return []

def geometric_classification(contour, w, h):
    """Geometric shape classification"""
    try:
        # Calculate geometric properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Aspect ratio
        aspect_ratio = w / h if h > 0 else 0
        
        # Circularity (4π*area / perimeter²)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Approximate polygon to get vertices
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Solidity (contour area / convex hull area)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Classification logic with more permissive thresholds
        confidence = 0.0
        prediction = "unknown"
        
        # Triangle detection
        if vertices == 3:
            prediction = "triangle"
            confidence = 0.8
        # Square/Rectangle detection
        elif vertices == 4:
            if 0.7 <= aspect_ratio <= 1.3:  # More permissive
                prediction = "square"
                confidence = 0.7
            else:
                prediction = "rectangle"
                confidence = 0.7
        # Star detection (many vertices, low solidity)
        elif vertices >= 5 and solidity < 0.8:  # More permissive
            prediction = "star"
            confidence = 0.7
        # Circle detection (high circularity, many vertices)
        elif circularity > 0.5 and vertices > 6:  # More permissive
            prediction = "circle"
            confidence = 0.7
        
        return prediction, confidence
        
    except Exception as e:
        return "unknown", 0.0

def predict_with_simple_knn(roi):
    """Simple KNN prediction"""
    try:
        # Convert to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold + inversion
        _, roi_thresh = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Resize to 28x28
        roi_resized = cv2.resize(roi_thresh, (28, 28), interpolation=cv2.INTER_AREA)
        
        # Flatten and normalize
        roi_flat = roi_resized.flatten().astype(np.float32) / 255.0
        
        # Use simple KNN for prediction
        if 'algorithm' in st.session_state and st.session_state.algorithm_loaded:
            pred = st.session_state.algorithm.predict([roi_flat])[0]
            proba = st.session_state.algorithm.predict_proba([roi_flat])[0]
            conf = np.max(proba)
            return pred, conf
        else:
            return "unknown", 0.0
    except Exception as e:
        return "unknown", 0.0

def hybrid_classification(contour, roi, w, h):
    """Hybrid classification with more permissive logic"""
    try:
        # Get geometric prediction
        geo_pred, geo_conf = geometric_classification(contour, w, h)
        
        # Get KNN prediction
        knn_pred, knn_conf = predict_with_simple_knn(roi)
        
        # More permissive decision logic
        if geo_conf > 0.6:  # Lower threshold
            if geo_pred == knn_pred:
                final_pred = geo_pred
                final_conf = min(geo_conf + 0.1, 1.0)
                method = "geometric_knn_agreement"
            else:
                final_pred = geo_pred
                final_conf = geo_conf
                method = "geometric_override"
        elif knn_conf > 0.4:  # Lower threshold
            final_pred = knn_pred
            final_conf = knn_conf
            method = "knn_fallback"
        else:
            if geo_conf > knn_conf:
                final_pred = geo_pred
                final_conf = geo_conf
                method = "geometric_low_conf"
            else:
                final_pred = knn_pred
                final_conf = knn_conf
                method = "knn_low_conf"
        
        return final_pred, final_conf, method
        
    except Exception as e:
        return "unknown", 0.0, "error"

def detect_shapes_robust(image):
    """Robust shape detection with multiple methods"""
    try:
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Step 1: Robust contour detection
        contours = robust_contour_detection(cv_image)
        
        detections = []
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            
            # More permissive size filtering
            if w > 20 and h > 20:  # Lower size threshold
                # Step 2: Crop bounding box
                roi = cv_image[y:y+h, x:x+w]
                
                # Step 3: Hybrid classification
                prediction, confidence, method = hybrid_classification(contour, roi, w, h)
                
                # Very permissive confidence threshold
                if confidence > 0.2:  # Very low threshold
                    detections.append({
                        'shape': prediction,
                        'confidence': confidence,
                        'position': (x + w//2, y + h//2),
                        'bbox': (x, y, x+w, y+h),
                        'method': method,
                        'area': area
                    })
        
        return detections
        
    except Exception as e:
        st.error(f"Detection error: {e}")
        return []

def draw_detection_results(image, detections):
    """Draw results on image"""
    detected_img = image.copy()
    draw = ImageDraw.Draw(detected_img)
    
    # Colors for shapes
    colors = {
        'circle': 'green',
        'square': 'orange', 
        'triangle': 'blue',
        'star': 'red',
        'rectangle': 'purple',
        'unknown': 'gray'
    }
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        shape = detection['shape']
        confidence = detection['confidence']
        
        # Get color
        color = colors.get(shape, 'gray')
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label
        label_text = f"{shape} ({int(confidence*100)}%)"
        
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        # Get text size
        bbox = draw.textbbox((0, 0), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw label background
        label_x = x1
        label_y = max(0, y1 - text_height - 5)
        draw.rectangle([label_x, label_y, label_x + text_width + 10, label_y + text_height + 5], 
                      fill=color, outline=color)
        
        # Draw label text
        draw.text((label_x + 5, label_y + 2), label_text, fill="white", font=font)
    
    return detected_img

def main():
    st.set_page_config(page_title="Shape Recognition", layout="wide")
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 2rem;
        color: #2c3e50;
    }
    .panel {
        flex: 1;
        background: white;
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .panel-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 1rem;
        color: #34495e;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🎯 Shape Recognition System</div>', unsafe_allow_html=True)
    
    # Load simple algorithm
    if 'algorithm' not in st.session_state:
        st.session_state.algorithm, st.session_state.algorithm_loaded = load_simple_algorithm()
    
   
    # Upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📁 Upload Image", type=['png', 'jpg', 'jpeg', 'gif', 'bmp'])
    
    with col2:
        detect_button = st.button("🎯 Detect Shapes", type="primary", use_container_width=True)
    
    # Main panels
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file)
            
            # Create two columns for the panels
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">Original Image</div>', unsafe_allow_html=True)
                st.image(image, width='stretch')
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown('<div class="panel-title">Detected Shapes</div>', unsafe_allow_html=True)
                
                if detect_button:
                    with st.spinner("Processing..."):
                        detections = detect_shapes_robust(image)
                    
                    if detections:
                        # Draw detection results
                        detected_img = draw_detection_results(image, detections)
                        st.image(detected_img, width='stretch')
                        
                        # Show results summary
                        st.success(f"✅ Detected {len(detections)} shapes")
                        
                        for i, detection in enumerate(detections, 1):
                            st.write(f"**{i}.** {detection['shape'].upper()} - {detection['confidence']*100:.1f}% confidence")
                    else:
                        st.warning("❌ No shapes detected")
                        st.image(image, width='stretch')
                else:
                    st.info("Click 'Detect Shapes' to see results")
                    st.image(image, width='stretch')
                
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Error processing image: {e}")
    else:
        # Show placeholder panels
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Original Image</div>', unsafe_allow_html=True)
            st.info("Upload an image to get started")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="panel">', unsafe_allow_html=True)
            st.markdown('<div class="panel-title">Detected Shapes</div>', unsafe_allow_html=True)
            st.info("Upload an image to see detection results")
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
