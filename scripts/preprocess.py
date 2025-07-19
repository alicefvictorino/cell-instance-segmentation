import os
import shutil
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import argparse
import yaml
from sklearn.model_selection import train_test_split 

def rle_to_mask(rle_string: str, height: int, width: int) -> np.ndarray:
    """
    Converts a Run-Length Encoding (RLE) string to a binary mask.
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

def process_group(group, yolo_label_dir, class_map):
    """
    Helper function to process a group of annotations for a single image.
    """
    image_id = group['id'].iloc[0]
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

def main(args):
    """
    Main preprocessing function to convert the Sartorius dataset to YOLO format.
    """
    print("--- Starting Preprocessing Pipeline (with Train/Val Split) ---")
    
    raw_data_dir = Path(args.raw_data_dir)
    output_dir = Path(args.output_dir)
    
    train_csv_path = raw_data_dir / 'train.csv'
    train_img_dir = raw_data_dir / 'train'

    if output_dir.exists():
        shutil.rmtree(output_dir)

    yolo_img_train_dir = output_dir / "images" / "train"
    yolo_label_train_dir = output_dir / "labels" / "train"
    yolo_img_val_dir = output_dir / "images" / "val"
    yolo_label_val_dir = output_dir / "labels" / "val"
    
    yolo_img_train_dir.mkdir(parents=True, exist_ok=True)
    yolo_label_train_dir.mkdir(parents=True, exist_ok=True)
    yolo_img_val_dir.mkdir(parents=True, exist_ok=True)
    yolo_label_val_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(train_csv_path)
    class_map = {name: i for i, name in enumerate(df['cell_type'].unique())}

    all_image_ids = df['id'].unique()
    train_ids, val_ids = train_test_split(all_image_ids, test_size=0.2, random_state=42)
    
    print(f"Total images: {len(all_image_ids)}")
    print(f"Training images: {len(train_ids)}")
    print(f"Validation images: {len(val_ids)}")

    df_train = df[df['id'].isin(train_ids)]
    df_val = df[df['id'].isin(val_ids)]

    train_groups = df_train.groupby('id')
    for image_id, group in tqdm(train_groups, desc="Converting TRAIN images"):
        process_group(group, yolo_label_train_dir, class_map)
        shutil.copy(train_img_dir / f"{image_id}.png", yolo_img_train_dir)

    val_groups = df_val.groupby('id')
    for image_id, group in tqdm(val_groups, desc="Converting VALIDATION images"):
        process_group(group, yolo_label_val_dir, class_map)
        shutil.copy(train_img_dir / f"{image_id}.png", yolo_img_val_dir)

    print(f"\nConversion completed. YOLO dataset saved at: '{output_dir}'")
    
    class_names = list(class_map.keys())
    yaml_content = {
        'path': str(output_dir.resolve()),
        'train': 'images/train',
        'val': 'images/val', 
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
