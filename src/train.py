"""
Train Head Detection Model
==========================
Purpose:
    Train a YOLO11s model to detect heads using the generated towncentre_head dataset.
    Optimized for RTX 4050 6GB VRAM.
"""

from ultralytics import YOLO
import os

def main():
    # Setup paths
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
    os.chdir(PROJECT_ROOT)
    
    print("[INFO] Starting YOLO11s training for Head Detection...")
    
    # Load the base model (Transfer Learning from COCO)
    # yolo11s is used to avoid CUDA Out Of Memory on 6GB VRAM
    model_path = os.path.join(PROJECT_ROOT, 'models/pretrained/yolo11s.pt')
    model = YOLO(model_path)
    
    # Config and Output paths
    data_cfg = os.path.join(PROJECT_ROOT, 'configs/head_dataset.yaml')
    project_dir = os.path.join(PROJECT_ROOT, 'outputs/runs')
    
    # Train the model
    # Note for RTX 4050 6GB:
    # - imgsz=1280 helps with small heads.
    # - batch=1 is the safest setting to avoid CUDA OOM on this GPU.
    results = model.train(
        data=data_cfg,
        epochs=50,          # 50 epochs is enough for a strong baseline
        imgsz=1280,         # Higher image size for small head objects
        batch=4,            # Keep memory use low at 1280 resolution
        device=0,           # Use GPU 0
        amp=False,          # Disable AMP (Mixed Precision) to prevent CUDA kernel crash on laptop GPU
        project=project_dir,
        name='train_head',  # Save results to outputs/runs/train_head/
        exist_ok=True,
        workers=0,          # Single-thread data loading to prevent pin_memory desync crash
    )
    
    print("\n[INFO] Training complete!")
    best_weights = os.path.join(project_dir, "train_head/weights/best.pt")
    print(f"[INFO] The best model weights are saved at: {best_weights}")
    print("[INFO] You can now update src/app.py to use this new model.")

if __name__ == "__main__":
    main()
