import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import joblib
import os

# Load KNN model with label encoder
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

def predict_with_knn(roi):
    """KNN prediction using the trained model"""
    try:
        # Convert to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold + inversion (same as training)
        _, roi_thresh = cv2.threshold(roi_gray, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Resize to 20x20 (same as training)
        roi_resized = cv2.resize(roi_thresh, (20, 20), interpolation=cv2.INTER_AREA)
        
        # Flatten and normalize
        roi_flat = roi_resized.flatten().astype(np.float32) / 255.0
        
        # Use KNN for prediction
        if 'algorithm' in st.session_state and st.session_state.algorithm_loaded:
            # The model expects encoded labels, but we need to decode them
            pred_encoded = st.session_state.algorithm.predict([roi_flat])[0]
            proba = st.session_state.algorithm.predict_proba([roi_flat])[0]
            conf = np.max(proba)
            
            # Decode the prediction back to string
            # The model was trained with encoded labels, so we need to decode
            shape_mapping = {0: 'circle', 1: 'rectangle', 2: 'square', 3: 'star', 4: 'triangle'}
            pred = shape_mapping.get(pred_encoded, 'unknown')
            
            return pred, conf
        else:
            return "unknown", 0.0
    except Exception as e:
        return "unknown", 0.0

def hybrid_classification(contour, roi, w, h):
    """Hybrid classification - prioritize KNN over geometric"""
    try:
        # Get KNN prediction first (more accurate)
        knn_pred, knn_conf = predict_with_knn(roi)
        
        # Get geometric prediction as backup
        geo_pred, geo_conf = geometric_classification(contour, w, h)
        
        # Prioritize KNN if it has reasonable confidence
        if knn_conf > 0.3:  # Lower threshold for KNN
            final_pred = knn_pred
            final_conf = knn_conf
            method = "knn_primary"
        elif geo_conf > 0.6:  # Use geometric if KNN is uncertain
            final_pred = geo_pred
            final_conf = geo_conf
            method = "geometric_fallback"
        else:
            # Use whichever has higher confidence
            if knn_conf > geo_conf:
                final_pred = knn_pred
                final_conf = knn_conf
                method = "knn_low_conf"
            else:
                final_pred = geo_pred
                final_conf = geo_conf
                method = "geometric_low_conf"
        
        return final_pred, final_conf, method
        
    except Exception as e:
        return "unknown", 0.0, "error"

def detect_shapes_robust(image):
    """Robust shape detection with KNN priority"""
    try:
        # Convert PIL to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Step 1: Robust contour detection
        contours = robust_contour_detection(cv_image)
        
        detections = []
        
        for contour in contours:
            try:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Extract ROI
                roi = cv_image[y:y+h, x:x+w]
                
                if roi.size == 0:
                    continue
                
                # Use hybrid classification (KNN priority)
                pred, conf, method = hybrid_classification(contour, roi, w, h)
                
                if pred != "unknown" and conf > 0.3:
                    detections.append({
                        'bbox': (x, y, w, h),
                        'prediction': pred,
                        'confidence': conf,
                        'method': method
                    })
                    
            except Exception as e:
                st.error(f"Detection error: {e}")
                continue
        
        return detections
        
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return []

def draw_detections(image, detections):
    """Draw detection results on image"""
    try:
        # Convert to PIL for drawing
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)
        
        # Colors for different shapes
        colors = {
            'circle': (255, 165, 0),    # Orange
            'square': (0, 255, 0),      # Green
            'rectangle': (0, 0, 255),   # Blue
            'triangle': (255, 0, 255),  # Magenta
            'star': (255, 255, 0)       # Yellow
        }
        
        for detection in detections:
            x, y, w, h = detection['bbox']
            pred = detection['prediction']
            conf = detection['confidence']
            method = detection['method']
            
            # Get color
            color = colors.get(pred, (128, 128, 128))
            
            # Draw bounding box
            draw.rectangle([x, y, x+w, y+h], outline=color, width=3)
            
            # Draw label
            label = f"{pred} ({conf:.0%})"
            try:
                # Try to use a font
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 16)
            except:
                font = ImageFont.load_default()
            
            # Draw text background
            text_bbox = draw.textbbox((x, y-25), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x, y-25), label, fill=(255, 255, 255), font=font)
        
        return pil_image
        
    except Exception as e:
        st.error(f"Error drawing detections: {e}")
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Streamlit app
def main():
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🎯 Shape Recognition System</div>', unsafe_allow_html=True)
    
    # Load KNN algorithm
    if 'algorithm' not in st.session_state:
        st.session_state.algorithm, st.session_state.algorithm_loaded = load_simple_algorithm()
    
    # Upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("📁 Upload Image", type=['png', 'jpg', 'jpeg', 'gif', 'bmp'])
    
    with col2:
        detect_button = st.button("🎯 Detect Shapes", type="primary", use_container_width=True)
    
    # Process uploaded image
    if uploaded_file is not None and detect_button:
        try:
            # Load image
            image = Image.open(uploaded_file)
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Detect shapes
            detections = detect_shapes_robust(image)
            
            if detections:
                # Convert to OpenCV format for drawing
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # Draw detections
                result_image = draw_detections(cv_image, detections)
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Original Image")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Detected Shapes")
                    st.image(result_image, use_column_width=True)
                
                # Show detection details
                st.subheader("Detection Details")
                for i, detection in enumerate(detections):
                    st.write(f"**Shape {i+1}**: {detection['prediction']} (Confidence: {detection['confidence']:.1%}, Method: {detection['method']})")
            else:
                st.warning("No shapes detected in the image.")
                st.image(image, use_column_width=True)
                
        except Exception as e:
            st.error(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
