import os
import yaml
import kagglehub
from ultralytics import YOLO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Required when running locally: keep all training code inside the __main__ guard
# to avoid PyTorch multiprocessing crashes.
if __name__ == '__main__':
    
    os.chdir(PROJECT_ROOT)
    print("1. Downloading and locating the Kaggle dataset on the local machine...")
    # KaggleHub downloads to the local cache directory (for example: ~/.cache/kagglehub)
    base_path = kagglehub.dataset_download("hoangxuanviet/scut-head")
    print(f"Dataset root: {base_path}")

    # 2. Automatically scan for the train and validation image folders
    train_dir = None
    val_dir = None

    for root, dirs, files in os.walk(base_path):
        if 'train' in root and root.endswith('images'):
            train_dir = root
        if ('val' in root or 'valid' in root) and root.endswith('images'):
            val_dir = root

    # 3. Process the result and start training
    if train_dir and val_dir:
        print("\n2. Found the correct image directories.")
        
        # Write the dataset YAML at the project root
        data_yaml = {
            'train': train_dir,
            'val': val_dir,
            'nc': 1,
            'names': {0: 'head'}
        }
        yaml_path = os.path.join(PROJECT_ROOT, 'scut_data.yaml')
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, sort_keys=False)
        print(f"[INFO] Created dataset config: {yaml_path}")

        print("\n3. Starting training...")
        # Load the base YOLO11s weights from the local project if available
        local_weights = os.path.join(PROJECT_ROOT, 'models', 'pretrained', 'yolo11s.pt')
        model = YOLO(local_weights if os.path.exists(local_weights) else 'yolo11s.pt')

        # Start training
        results = model.train(
            data=yaml_path,
            epochs=100,         
            imgsz=1280,         
            batch=4,            
            device=0,           # Use the discrete NVIDIA GPU
            patience=20,        
            optimizer='auto',   
            cos_lr=True,        
            
            # --- SCUT-HEAD tuning parameters ---
            box=7.5,            
            cls=0.5,            
            dfl=1.5,            
            mosaic=1.0,         
            
            # --- Save outputs in the standard project runs folder ---
            project=os.path.join(PROJECT_ROOT, 'outputs', 'runs'),
            name='train_head'
        )

        print("\n[INFO] Training complete.")
        print("[INFO] Best weights: outputs/runs/train_head/weights/best.pt")
    else:
        print("\n[WARNING] Could not find the 'images' directories in this dataset.")