import os
import cv2
import numpy as np
import yaml
from pathlib import Path
from ultralytics import YOLO

# 1. Create a mock dataset
def create_mock_dataset(base_path="mock_dataset"):
    print("Setting up dataset directories...")
    base = Path(base_path)
    images_train = base / "images" / "train"
    images_val = base / "images" / "val"
    labels_train = base / "labels" / "train"
    labels_val = base / "labels" / "val"
    
    # Create the directories
    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)
        
    print("Generating mock structual damage images...")
    # Generate mock images and labels (10 for train, 4 for val for speed)
    for split, img_dir, lbl_dir, count in [("train", images_train, labels_train, 10), ("val", images_val, labels_val, 4)]:
        for i in range(count):  
            img_path = img_dir / f"{split}_{i}.jpg"
            lbl_path = lbl_dir / f"{split}_{i}.txt"
            
            # create a random image to simulate a building structure
            img = np.random.randint(50, 200, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(str(img_path), img)
            
            # create a label: class x_center y_center width height
            cls_id = np.random.randint(0, 3) # 0: Intact, 1: Crack, 2: Spalling
            with open(lbl_path, "w") as f:
                f.write(f"{cls_id} 0.5 0.5 0.4 0.4\n")
                
    # Create data.yaml required by Ultralytics
    yaml_path = base / "data.yaml"
    data = {
        'path': os.path.abspath(base_path),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,
        'names': {0: 'Intact', 1: 'Crack', 2: 'Spalling/Severe Damage'}
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
        
    print(f"Mock dataset generated successfully at: {yaml_path}")
    return str(yaml_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Automated Edge AI Training Pipeline')
    parser.add_argument('--tune', action='store_true', help='Enable exhaustive hyperparameter sweep')
    args = parser.parse_args()

    yaml_path = create_mock_dataset()
    
    print("\n--- Downloading Base Edge Model (yolov8n.pt) ---")
    model = YOLO('yolov8n.pt')
    
    if args.tune:
        print("\n--- Initiating Intelligent Hyperparameter Sweeping (Ray Tune) ---")
        print("🔍 Searching for optimal learning rates, augmentations, and optimizers.")
        # Uses genetic evolution algorithms embedded in ultralytics to find the peak map50 for edge structures
        model.tune(data=yaml_path, epochs=5, iterations=10, optimizer='AdamW', imgsz=256, project="models_tune")
    else:
        print("\n--- Starting Automatic Base Training ---")
        # Added project tracking parameters
        model.train(data=yaml_path, epochs=1, imgsz=256, project="models", name="exported", exist_ok=True)
    
    # Locate the best model saved by ultralytics
    best_model_path = "models/exported/weights/best.pt"
    print(f"\n--- Training completed. Model weights saved to {best_model_path} ---")
    
    if os.path.exists(best_model_path):
        print("\n--- Exporting Best Model to ONNX for Edge ---")
        best_model = YOLO(best_model_path)
        best_model.export(format="onnx", imgsz=256, opset=12, dynamic=False)
        print("\n🟢 PIPELINE COMPLETED: ONNX model ready for deployment on Edge AI Hardware!")
