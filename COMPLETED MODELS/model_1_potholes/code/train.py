import os
import yaml
import cv2
import argparse
import numpy as np
from cv_model import PotholeCVDetector
from vibration_model import VibrationDetector

def generate_synthetic_yolo_dataset(base_dir="synthetic_dataset", num_train=10, num_val=3):
    """
    Generates a synthetic YOLO formatted dataset for testing the training code.
    Draws dark ellipses on a gray background to represent potholes on a road.
    """
    print(f"[Train] Generating synthetic YOLO dataset at '{base_dir}'...")
    
    # Define directories
    dirs = [
        os.path.join(base_dir, "images", "train"),
        os.path.join(base_dir, "images", "val"),
        os.path.join(base_dir, "labels", "train"),
        os.path.join(base_dir, "labels", "val")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    def create_fake_road_image(img_path, label_path):
        # Create a gray background (road)
        img = np.full((480, 640, 3), 120, dtype=np.uint8)
        # Add some road texture noise
        noise = np.random.randint(-15, 15, img.shape)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Decide how many potholes to draw (1 or 2)
        num_potholes = np.random.randint(1, 3)
        labels = []
        
        for _ in range(num_potholes):
            # Random pothole parameters
            center_x = np.random.randint(100, 540)
            center_y = np.random.randint(100, 380)
            axes_x = np.random.randint(20, 60)
            axes_y = np.random.randint(10, 30)
            angle = np.random.randint(-15, 15)
            
            # Draw dark ellipse (pothole)
            cv2.ellipse(img, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, (50, 50, 50), -1)
            # Add some inner depth shadow
            cv2.ellipse(img, (center_x, center_y), (int(axes_x*0.7), int(axes_y*0.7)), angle, 0, 360, (25, 25, 25), -1)
            
            # YOLO label calculations (normalized center_x, center_y, width, height)
            x_c = center_x / 640.0
            y_c = center_y / 480.0
            w_n = (axes_x * 2) / 640.0
            h_n = (axes_y * 2) / 480.0
            
            # class_id is 0 for pothole
            labels.append(f"0 {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
            
        # Save image and label files
        cv2.imwrite(img_path, img)
        with open(label_path, "w") as f:
            f.write("\n".join(labels))
            
    # Generate train files
    for i in range(num_train):
        img_p = os.path.join(base_dir, "images", "train", f"road_train_{i}.jpg")
        lbl_p = os.path.join(base_dir, "labels", "train", f"road_train_{i}.txt")
        create_fake_road_image(img_p, lbl_p)
        
    # Generate val files
    for i in range(num_val):
        img_p = os.path.join(base_dir, "images", "val", f"road_val_{i}.jpg")
        lbl_p = os.path.join(base_dir, "labels", "val", f"road_val_{i}.txt")
        create_fake_road_image(img_p, lbl_p)
        
    # Create dataset.yaml
    dataset_yaml = {
        "path": os.path.abspath(base_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": {0: "pothole"}
    }
    
    yaml_path = os.path.join(base_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
        
    print(f"[Train] Synthetic dataset successfully created at '{yaml_path}'")
    return yaml_path

def main():
    parser = argparse.ArgumentParser(description="Potholes and Road Crack Model Training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size for training")
    parser.add_argument("--device", type=str, default=None, help="Device to train on (e.g. cpu, cuda, or 0)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic training dataset generation")
    args = parser.parse_args()

    print("=========================================")
    print("  Training Pipeline: Pothole & Road Crack  ")
    print("=========================================\n")
    
    # 1. Check for real dataset
    real_yaml = "/home/leyoking/sih op/complete thing/datasets/potholes_cv/dataset.yaml"
    use_real = os.path.exists(real_yaml) and not args.synthetic
    
    if use_real:
        yaml_path = real_yaml
        print(f"[Train] Found real RDD2022 dataset config: {yaml_path}")
        print("[Train] Training on real road defect imagery...")
    else:
        print("[Train] Training on synthetic mock data...")
        yaml_path = generate_synthetic_yolo_dataset(base_dir="synthetic_dataset")
        
    # 2. Device determination
    if args.device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except:
            device = "cpu"
    else:
        device = args.device
        
    print(f"[Train] Target execution hardware: {device.upper()}")
    print(f"[Train] Hyperparameters -> Epochs: {args.epochs}, Batch size: {args.batch}\n")
    
    # 3. Train CV Model
    print("Step 1: Training CV Object Detection (YOLOv8)...")
    PotholeCVDetector.train_model(
        dataset_yaml_path=yaml_path,
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        device=device,
        save_dir="runs"
    )
    
    # Copy best weights to current directory
    best_weight_src = "runs/pothole_yolov8n/weights/best.pt"
    best_weight_dest = "yolov8n_pothole.pt"
    if os.path.exists(best_weight_src):
        import shutil
        shutil.copy(best_weight_src, best_weight_dest)
        print(f"[Train] Best trained CV model weights copied to: {best_weight_dest}")
    else:
        print("[Train] Warning: Could not find trained weights at standard YOLO output directory.")
        
    print("\n[Train] Training pipeline execution completed successfully!")

if __name__ == "__main__":
    main()
