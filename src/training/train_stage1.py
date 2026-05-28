import os
from ultralytics import YOLO

def train_stage1():
    print("[INFO] Starting Stage 1: Pre-training architecture on SCUT-HEAD dataset...")
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_yaml = os.path.join(PROJECT_ROOT, "configs/scut_data.yaml")
    
    if not os.path.exists(data_yaml):
        print(f"[ERROR] Could not find {data_yaml}. Please run src/data_prep/prep_scut.py first!")
        return

    # Initialize base model
    model = YOLO('yolo11s.pt')
    
    save_dir = os.path.join(PROJECT_ROOT, "models/stage1_scut")
    os.makedirs(save_dir, exist_ok=True)

    # Start training process
    results = model.train(
        data=data_yaml,
        epochs=100,         # Deep training for initial feature extraction
        imgsz=1280,         # High resolution for top-down camera views
        batch=4,            # Safe batch size for T4/RTX4050 GPUs
        device=0,           # Use first GPU
        patience=20,        # Early stopping patience
        optimizer='auto',
        cos_lr=True,        # Smooth learning rate decay
        
        # Augmentation parameters tuned for dense head detection
        box=7.5,            
        cls=0.5,            
        dfl=1.5,            
        mosaic=1.0,         # Maximize mosaic augmentation for dense crowds
        
        project=save_dir,
        name='yolo11s_ultimate_head'
    )
    
    print(f"\n[SUCCESS] Stage 1 completed! Model weights saved at: {save_dir}/yolo11s_ultimate_head/weights/")

if __name__ == "__main__":
    train_stage1()
