import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import os

# --- Page Configuration and Title ---
st.set_page_config(layout="wide", page_title="Cell Morphometric Analysis")

st.title("🔬 AI-Powered Cell Morphometric Analysis Tool")
st.write("Upload a microscopy image to automatically segment cells and extract quantitative data.")

# --- Analysis Functions and Model Loading ---

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    """Loads the YOLO model from the specified path."""
    return YOLO(model_path)

def calculate_morphometrics(result_object):
    """
    Calculates morphometric data (area, perimeter, circularity, etc.) from the model's segmentation masks.
    """
    masks = result_object.masks
    if not masks:
        return pd.DataFrame()

    morph_data = []
    class_names = result_object.names
    for i, mask_data in enumerate(masks):
        # Extract mask and find contours
        mask_np = mask_data.data[0].cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        # Calculate metrics for the largest contour
        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        
        aspect_ratio = 0
        if len(cnt) >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            aspect_ratio = MA / ma if ma > 0 else 0
            
        class_id = int(result_object.boxes[i].cls)
        confidence = float(result_object.boxes[i].conf)

        morph_data.append({
            'cell_id': i + 1,
            'class_name': class_names[class_id],
            'confidence': round(confidence, 4),
            'area_pixels': area,
            'perimeter_pixels': perimeter,
            'circularity': round(circularity, 4),
            'aspect_ratio': round(aspect_ratio, 4)
        })
        
    return pd.DataFrame(morph_data)

# --- Application Interface (Sidebar and Main Area) ---

MODEL_PATH = "best.pt"

# Sidebar for controls
with st.sidebar:
    st.header("Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    
    st.header("Image Upload")
    uploaded_file = st.file_uploader("Choose a cell image...", type=["png", "jpg", "jpeg"])

# Check if the model file exists before proceeding
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at '{MODEL_PATH}'.")
    st.error("Please ensure the 'best.pt' file is in the same directory as the application script.")
else:
    model = load_model(MODEL_PATH)
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Main area to display results
        st.header("Analysis Results")
        
        with st.spinner('Segmenting cells and extracting metrics...'):
            results = model.predict(source=image, save=False, conf=confidence_threshold)
            result = results[0]
            
            # Two-column layout for visualization
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, caption='Uploaded Image.', use_column_width=True)
            with col2:
                st.subheader("Model Segmentation")
                predicted_plot = result.plot(line_width=2, font_size=12)
                st.image(cv2.cvtColor(predicted_plot, cv2.COLOR_BGR2RGB), caption='Cells segmented by the model.', use_column_width=True)
            
            # Tabs for quantitative analysis
            st.header("Quantitative Analysis")
            tab1, tab2 = st.tabs(["📊 Data Table", "📈 Distribution Plots"])
            
            df_morph = calculate_morphometrics(result)
            
            with tab1:
                if not df_morph.empty:
                    st.dataframe(df_morph)
                else:
                    st.warning("No cells were detected in the image with the current confidence threshold.")
            
            with tab2:
                if not df_morph.empty:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    df_morph['area_pixels'].hist(bins=20, ax=ax1, color='skyblue', edgecolor='black')
                    ax1.set_title('Cell Area Distribution')
                    ax1.set_xlabel('Area (pixels²)')
                    ax1.set_ylabel('Frequency')
                    
                    df_morph['circularity'].hist(bins=20, ax=ax2, color='salmon', edgecolor='black')
                    ax2.set_title('Circularity Distribution')
                    ax2.set_xlabel('Circularity (1 = Perfect Circle)')
                    
                    st.pyplot(fig)
                else:
                    st.warning("No cells detected to generate plots.")

    else:
        st.info("Waiting for an image to be uploaded to start the analysis.")

