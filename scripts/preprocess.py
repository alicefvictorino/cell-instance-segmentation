import os
import shutil
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
import yaml

def rle_to_mask(rle_string: str, height: int, width: int) -> np.ndarray:
    """
    Convert Run-Length Encoding (RLE) string to binary mask.
    
    Run-Length Encoding is a simple form of data compression where consecutive
    data elements are stored as a single data value and count. This function
    converts an RLE string representing a binary mask into a 2D numpy array.
    
    Args:
        rle_string (str): Run-length encoded string in format "start1 length1 start2 length2 ..."
        height (int): Height of the target mask in pixels
        width (int): Width of the target mask in pixels
        
    Returns:
        np.ndarray: Binary mask as a 2D numpy array with dtype uint8
    """
    if pd.isna(rle_string):
        return np.zeros((height, width), dtype=np.uint8)
    s = rle_string.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    img = np.zeros(width * height, dtype=np.uint8)
    for lo, hi in zip(starts, lengths):
        img[lo:lo + hi] = 1
    return img.reshape((height, width))

def main(args):
    """
    Main preprocessing function to convert Sartorius dataset to YOLO format.
    
    This function orchestrates the entire preprocessing pipeline:
    1. Loads the training CSV file containing RLE annotations
    2. Creates the necessary directory structure for YOLO format
    3. Converts RLE annotations to YOLO polygon format
    4. Copies images to the appropriate directories
    5. Generates the dataset.yaml configuration file
    
    Args:
        args: Command line arguments containing:
            - raw_data_dir: Path to directory containing train.csv and train/ folder
            - output_dir: Path where YOLO-formatted dataset will be saved
    """
    print("--- Starting Preprocessing Pipeline ---")
    
    # Convert string paths to Path objects
    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    
    train_csv_path = raw_data_dir / 'train.csv'
    train_img_dir = raw_data_dir / 'train'

    # Create directory structure
    if output_dir.exists():
        shutil.rmtree(output_dir)
    yolo_img_dir = output_dir / "images" / "train"
    yolo_label_dir = output_dir / "labels" / "train"
    yolo_img_dir.mkdir(parents=True, exist_ok=True)
    yolo_label_dir.mkdir(parents=True, exist_ok=True)

    # Conversion logic
    df = pd.read_csv(train_csv_path)
    class_map = {name: i for i, name in enumerate(df['cell_type'].unique())}
    
    image_groups = df.groupby('id')
    for image_id, group in tqdm(image_groups, desc="Converting images"):
        img_height = group['height'].iloc[0]
        img_width = group['width'].iloc[0]
        label_path = yolo_label_dir / f"{image_id}.txt"
        with open(label_path, 'w') as label_file:
            for _, row in group.iterrows():
                mask = rle_to_mask(row['annotation'], height=img_height, width=img_width)
                if mask.sum() == 0: continue
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours: continue
                contour = max(contours, key=cv2.contourArea)
                if contour.size < 6: continue
                normalized_contour = contour.astype(float).flatten()
                normalized_contour[0::2] /= img_width
                normalized_contour[1::2] /= img_height
                class_id = class_map[row['cell_type']]
                label_file.write(f"{class_id} {' '.join(map(str, normalized_contour))}\n")
        
        src_image_path = train_img_dir / f"{image_id}.png"
        shutil.copy(src_image_path, yolo_img_dir)

    print(f"\nConversion completed. YOLO dataset saved to: '{output_dir}'")
    
    # Create YAML configuration file
    class_names = list(class_map.keys())
    yaml_content = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/train',
        'nc': len(class_names),
        'names': class_names
    }
    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False)
    print(f"Dataset configuration file created at: {yaml_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert Sartorius dataset to YOLO format.")
    parser.add_argument('--raw_data_dir', type=str, required=True, help="Path to raw data directory (contains train.csv and train/ folder).")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory where YOLO dataset will be saved.")
    args = parser.parse_args()
    main(args)
