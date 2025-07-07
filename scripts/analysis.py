import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from pathlib import Path
import argparse
from ultralytics import YOLO

def calculate_morphometrics(result_object):
    """
    Calculates morphometric features from a YOLO result object.

    This function takes the output of a YOLOv11-seg prediction, iterates through
    each detected mask, and computes key shape-based metrics using OpenCV.

    Args:
        result_object: The result object from a `model.predict()` call.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the morphometric data
                      for each detected cell, including area, perimeter,
                      circularity, and aspect ratio.
    """
    masks = result_object.masks
    if not masks:
        return pd.DataFrame()

    morph_data = []
    class_names = result_object.names

    for i, mask_data in enumerate(masks):
        # Convert mask to a binary numpy array for OpenCV
        mask_np = mask_data.data[0].cpu().numpy().astype(np.uint8)
        
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        
        # Assume the largest contour is the cell
        cnt = max(contours, key=cv2.contourArea)
        
        # --- Morphometric Calculations ---
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        
        # Circularity (approaches 1 for a perfect circle)
        circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0
        
        # Aspect Ratio (from fitted ellipse)
        aspect_ratio = 0
        if len(cnt) >= 5:  # fitEllipse requires at least 5 points
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            aspect_ratio = MA / ma if ma > 0 else 0
        
        # Get classification details
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

def main(args):
    """
    Main function to orchestrate the morphometric analysis pipeline.
    
    This function coordinates the entire analysis process:
    1. Loads the trained YOLOv11 model
    2. Runs inference on the specified test image
    3. Extracts morphometric features from detected cells
    4. Displays results in a formatted table
    
    Args:
        args: Command line arguments containing:
            - model_path: Path to the trained model file (best.pt)
            - image_path: Path to the test image for analysis
            - conf: Confidence threshold for predictions
    """
    model_path = Path(args.model_path)
    image_path = Path(args.image_path)

    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    print(f"Running prediction on test image: {image_path.name}")
    results = model.predict(source=str(image_path), conf=args.conf)
    result = results[0]
    
    print(f"Found {len(result.masks) if result.masks else 0} cells.")

    df_morph = calculate_morphometrics(result)
    print("\n--- Morphometric Analysis Table ---")
    print(df_morph.to_string())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Morphometric analysis using a trained YOLO model.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file (best.pt).")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the test image.")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold for predictions.")
    args = parser.parse_args()
    main(args)
