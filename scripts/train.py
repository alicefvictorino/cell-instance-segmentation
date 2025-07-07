import argparse
from pathlib import Path
from ultralytics import YOLO

def main(args):
    """
    Main training function for YOLOv11 segmentation model.
    
    This function handles the complete training pipeline:
    1. Loads a pre-trained YOLOv11 model
    2. Verifies the dataset configuration file exists
    3. Initiates training with specified parameters
    4. Saves results to the designated project directory
    
    Args:
        args: Command line arguments containing:
            - data_dir: Path to directory containing dataset.yaml
            - project_dir: Directory where training results will be saved
            - model_name: Name of the pre-trained model to use
            - epochs: Number of training epochs
    """
    print("--- Starting YOLOv11 Model Training ---")
    
    # Load pre-trained model
    model = YOLO(args.model_name)

    # Path to dataset configuration file
    yaml_path = Path(args.data_dir) / "dataset.yaml"
    
    if not yaml_path.exists():
        print(f"ERROR: Configuration file not found at '{yaml_path}'")
        return

    print(f"Using configuration file: {yaml_path}")
    print(f"Results will be saved to: {args.project_dir}")

    # Start training
    model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=640,
        project=args.project_dir,
        name=f'{Path(args.model_name).stem}_{args.epochs}_epochs'
    )
    
    print("\n--- TRAINING COMPLETED! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a YOLOv11 segmentation model.")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to processed YOLO dataset directory (contains dataset.yaml).")
    parser.add_argument('--project_dir', type=str, required=True, help="Main directory to save training runs.")
    parser.add_argument('--model_name', type=str, default='yolov11n-seg.pt', help="Name of the pre-trained model to use.")
    parser.add_argument('--epochs', type=int, default=25, help="Number of training epochs.")
    args = parser.parse_args()
    main(args)