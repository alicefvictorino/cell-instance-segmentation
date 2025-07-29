import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from PIL import Image
import os
import io 
import plotly.express as px

# --- Page Configuration and Title ---
st.set_page_config(layout="wide", page_title="Cell Morphometric Analysis")

# --- Application Header ---
st.markdown("""
    <style>
    .big-title {
        font-size: 3em;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 0.5em;
    }
    .subtitle {
        color: var(--text-color);
        font-size: 1.2em;
        text-align: center;
        margin-bottom: 1.5em;
    }
    .upload-container {
        border: 2px dashed #4CAF50;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        margin-bottom: 2em;
        background-color: var(--secondary-background-color);
        color: var(--text-color); 
    }
    .upload-container h3, .upload-container p {
        color: var(--text-color);
    }
    .stSpinner > div > span {
        font-size: 1.2em;
        color: #4CAF50;
    }
    .metric-card {
        background-color: var(--secondary-background-color);
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card h4 {
        color: var(--text-color);
        margin-bottom: 5px;
    }
    .metric-card p {
        font-size: 1.4em;
        font-weight: bold;
        color: #28a745;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<p class='big-title'>🔬 AI-Powered Cell Morphometric Analysis</p>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Upload a microscopy image to automatically segment cells and extract detailed quantitative data.</p>", unsafe_allow_html=True)

# --- Analysis Functions and Model Loading ---

@st.cache_resource
def load_model(model_path):
    """Loads the YOLO model from the specified path."""
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

def calculate_morphometrics(result_object):
    """
    Calculates morphometric data (area, perimeter, circularity, etc.) from the model's segmentation masks.
    """
    masks = result_object.masks
    if masks is None or not masks.data.numel():
        return pd.DataFrame()

    morph_data = []
    class_names = result_object.names
    
    for i in range(len(masks.data)):
        mask_data = masks[i]
        
        mask_np = mask_data.data[0].cpu().numpy().astype(np.uint8)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        
        aspect_ratio = 0
        if len(cnt) >= 5: 
            try:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                aspect_ratio = MA / ma if ma > 0 else 0
            except cv2.error: 
                pass
            
        class_id = int(result_object.boxes[i].cls)
        confidence = float(result_object.boxes[i].conf)

        morph_data.append({
            'Cell ID': i + 1,
            'Class': class_names[class_id],
            'Confidence': round(confidence, 4),
            'Area (pixels)': area,
            'Perimeter (pixels)': perimeter,
            'Circularity': round(circularity, 4),
            'Aspect Ratio': round(aspect_ratio, 4)
        })
        
    return pd.DataFrame(morph_data)

# --- Model Verification and Loading ---
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "best.pt")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file 'best.pt' not found at '{MODEL_PATH}'.")
    st.error("Please ensure the 'best.pt' file is in the same directory as the application script.")
    st.stop()

model = load_model(MODEL_PATH)

# --- User Interface ---

# Sidebar for upload and settings
with st.sidebar:
    st.header("Image Upload")
    st.markdown("Drag and drop or click to select your microscopy image.")
    uploaded_file = st.file_uploader(" ", type=["png", "jpg", "jpeg"], accept_multiple_files=False, key="main_uploader")
    
    st.markdown("---")
    st.header("Analysis Settings")
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05,
                                     help="Adjust this value to include or exclude low-confidence detections.")


# --- Main Results Area ---
if uploaded_file is None:
    st.markdown("""
        <div class="upload-container">
            <h3>No Image Uploaded</h3>
            <p>Upload an image in the side panel to begin the analysis.</p>
            <p>Accepted formats: PNG, JPG, JPEG.</p>
        </div>
        """, unsafe_allow_html=True)
else:
    image = Image.open(uploaded_file).convert("RGB")

    MAX_SIZE = 1024  
    image.thumbnail((MAX_SIZE, MAX_SIZE))

    st.subheader("Uploaded Image Preview")
    st.image(image, caption='Your Uploaded Image.', width=400)

    st.markdown("---")
    st.header("Analysis Results")

    img_np = np.array(image)

    with st.spinner('Segmenting cells and extracting metrics... This may take a moment.'):
        try:
            results = model.predict(
                source=img_np,
                save=False,
                conf=confidence_threshold,
                verbose=False,
                device="cpu"
            )
            result = results[0]
        except Exception as e:
            st.error(f"Error during segmentation: {e}")
            st.warning("Please check if the model is correct and the image is compatible.")
            st.stop()


        col_img_original, col_img_segmented = st.columns(2)
        with col_img_original:
            st.subheader("Original Image")
            st.image(image, caption='Input image.', use_container_width=True)
        with col_img_segmented:
            st.subheader("Model Segmentation")
            if result.masks is not None and len(result.masks) > 0: 
                predicted_plot = result.plot(line_width=2, font_size=12, conf=False, labels=False)
                st.image(predicted_plot, caption=f'Segmented cells ({len(result.masks)} detected).', use_container_width=True)
            else:
                st.info("No cells detected with the current confidence threshold.")
                st.image(image, caption='No segmentation. Adjust the confidence threshold?', use_container_width=True) 


    st.markdown("---")
    st.header("Quantitative Analysis")
    
    df_morph = calculate_morphometrics(result)
    
    if not df_morph.empty:
        total_cells = len(df_morph)
        st.success(f"✔️ {total_cells} Cells Detected!")

        # --- Statistical Summary in Cards ---
        st.subheader("Cellular Statistical Summary")
        
        stats = df_morph[['Area (pixels)', 'Perimeter (pixels)', 'Circularity', 'Aspect Ratio']].describe().loc[['mean', 'std', 'min', 'max']].round(2)
        
        col_area, col_perimeter, col_circularity, col_aspect_ratio = st.columns(4)

        with col_area:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Average Area</h4>
                <p>{stats.loc['mean', 'Area (pixels)']:.2f}</p>
                <small>± {stats.loc['std', 'Area (pixels)']:.2f} px²</small>
            </div>
            """, unsafe_allow_html=True)
        with col_perimeter:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Average Perimeter</h4>
                <p>{stats.loc['mean', 'Perimeter (pixels)']:.2f}</p>
                <small>± {stats.loc['std', 'Perimeter (pixels)']:.2f} px</small>
            </div>
            """, unsafe_allow_html=True)
        with col_circularity:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Average Circularity</h4>
                <p>{stats.loc['mean', 'Circularity']:.2f}</p>
                <small>± {stats.loc['std', 'Circularity']:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
        with col_aspect_ratio:
            st.markdown(f"""
            <div class="metric-card">
                <h4>Average Aspect Ratio</h4>
                <p>{stats.loc['mean', 'Aspect Ratio']:.2f}</p>
                <small>± {stats.loc['std', 'Aspect Ratio']:.2f}</small>
            </div>
            """, unsafe_allow_html=True)
            
        # --- Interactive Distribution Plots (Plotly Express) ---
        st.subheader("Metric Distributions")
        
        metric_hist = st.selectbox(
            "Select a metric for the Histogram:",
            ['Area (pixels)', 'Perimeter (pixels)', 'Circularity', 'Aspect Ratio'],
            key='hist_select'
        )
        if metric_hist:
            fig_hist = px.histogram(df_morph, x=metric_hist, nbins=30, 
                                    title=f'Distribution of {metric_hist}',
                                    template="plotly_white", color_discrete_sequence=px.colors.qualitative.Plotly)
            fig_hist.update_layout(xaxis_title=metric_hist, yaxis_title="Frequency")
            st.plotly_chart(fig_hist, use_container_width=True)

        if df_morph['Class'].nunique() > 1:
            st.markdown("---")
            st.subheader("Metric Comparison by Cell Class")
            metric_box = st.selectbox(
                "Select a metric for the Box Plot:",
                ['Area (pixels)', 'Perimeter (pixels)', 'Circularity', 'Aspect Ratio'],
                key='box_select'
            )
            if metric_box:
                fig_box = px.box(df_morph, x='Class', y=metric_box, 
                                 title=f'Distribution of {metric_box} by Class',
                                 template="plotly_white", color='Class')
                fig_box.update_layout(xaxis_title="Cell Class", yaxis_title=metric_box)
                st.plotly_chart(fig_box, use_container_width=True)
                
        # --- Data Table and Download ---
        st.markdown("---")
        st.subheader("Complete Data Table (for Detailed Analysis)")
        
        tab_data_full, tab_download = st.tabs(["🔎 View Table", "💾 Download Data"])
        
        with tab_data_full:
            st.dataframe(df_morph, use_container_width=True)
        
        with tab_download:
            csv = df_morph.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Morphometric Data (CSV)",
                data=csv,
                file_name="cell_morphometric_data.csv",
                mime="text/csv",
                help="Download all extracted morphometric data as a CSV file."
            )

            
