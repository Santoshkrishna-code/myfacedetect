"""Streamlit web UI for MyFaceDetect.

Run with: streamlit run app.py
"""
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from pathlib import Path

try:
    from myfacedetect import detect_faces, detect_faces_realtime
    HAS_MYFACEDETECT = True
except ImportError:
    HAS_MYFACEDETECT = False


st.set_page_config(page_title="MyFaceDetect", layout="wide")

st.title("🎭 MyFaceDetect - Face Detection & Recognition")
st.markdown("Upload images or use your webcam to detect faces in real-time")

if not HAS_MYFACEDETECT:
    st.error("MyFaceDetect not installed. Please install: `pip install myfacedetect`")
    st.stop()


# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    detection_method = st.selectbox(
        "Detection Method",
        ["mediapipe", "haar", "ensemble"],
        help="Select the face detection method"
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        0.0, 1.0, 0.5,
        help="Minimum confidence to consider a detection"
    )
    
    show_landmarks = st.checkbox("Show Landmarks", value=False)
    draw_stats = st.checkbox("Show Statistics", value=True)


# Main tabs
tab1, tab2, tab3 = st.tabs(["📸 Image Upload", "📹 Webcam", "📊 Batch Processing"])

with tab1:
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp"])
    
    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_column_width=True)
        
        with col2:
            st.subheader("Detection Results")
            
            # Run detection
            with st.spinner("Running face detection..."):
                try:
                    faces = detect_faces(image_bgr, method=detection_method)
                    
                    # Draw detections
                    result_image = image_bgr.copy()
                    for i, face in enumerate(faces):
                        x, y, w, h = face['bbox']
                        conf = face.get('confidence', 1.0)
                        
                        color = (0, 255, 0) if conf > confidence_threshold else (0, 0, 255)
                        cv2.rectangle(result_image, (x, y), (x+w, y+h), color, 2)
                        
                        label = f"Face {i+1} ({conf:.2f})"
                        cv2.putText(result_image, label, (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        if show_landmarks and 'landmarks' in face:
                            for point in face['landmarks']:
                                cv2.circle(result_image, tuple(point), 2, (0, 0, 255), -1)
                    
                    st.image(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB), use_column_width=True)
                    
                    if draw_stats:
                        st.info(f"✅ Detected {len(faces)} face(s)")
                        for i, face in enumerate(faces):
                            st.write(f"**Face {i+1}:** Confidence = {face.get('confidence', 'N/A'):.3f}")
                
                except Exception as e:
                    st.error(f"Error during detection: {str(e)}")


with tab2:
    st.header("Webcam Detection")
    
    if st.button("Start Webcam", key="webcam_start"):
        st.warning("⚠️ Webcam support requires running locally. See documentation for setup.")
        st.code("""
# Run this locally:
python examples/detect_faces_live.py
        """)


with tab3:
    st.header("Batch Processing")
    
    uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    
    if uploaded_files:
        st.info(f"Processing {len(uploaded_files)} images...")
        
        results_summary = []
        progress_bar = st.progress(0)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            image_array = np.array(image)
            
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            
            try:
                faces = detect_faces(image_bgr, method=detection_method)
                results_summary.append({
                    'file': uploaded_file.name,
                    'faces_detected': len(faces),
                    'status': '✅'
                })
            except Exception as e:
                results_summary.append({
                    'file': uploaded_file.name,
                    'faces_detected': 0,
                    'status': f'❌ {str(e)[:50]}'
                })
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        st.subheader("Processing Results")
        
        results_df = st.dataframe(results_summary, use_container_width=True)
        
        total_faces = sum(r['faces_detected'] for r in results_summary)
        st.success(f"✅ Processing complete! Total faces detected: {total_faces}")


# Footer
st.markdown("---")
st.markdown("""
**MyFaceDetect v0.4.0** - CPU-friendly face detection and recognition

- 📚 [Documentation](https://github.com/Santoshkrishna-code/myfacedetect)
- 🐛 [Report Issues](https://github.com/Santoshkrishna-code/myfacedetect/issues)
- ⭐ [GitHub Repository](https://github.com/Santoshkrishna-code/myfacedetect)

Made with ❤️ by Santosh Krishna
""")
