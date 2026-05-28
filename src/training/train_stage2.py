import os
import argparse
from ultralytics import YOLO

def train_stage2(resume=False, model_path=None):
    print("[INFO] Starting Stage 2: Fine-tuning for SAHI-compatibility on CrowdHuman...")
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_yaml = os.path.join(PROJECT_ROOT, "configs/crowdhuman_data.yaml")
    
    if not os.path.exists(data_yaml):
        print(f"[ERROR] Could not find {data_yaml}. Please run src/data_prep/prep_crowd.py first!")
        return

    save_dir = os.path.join(PROJECT_ROOT, "models/stage2_crowdhuman")
    os.makedirs(save_dir, exist_ok=True)

    if resume:
        print("[INFO] Resume Training feature activated...")
        # Locate the last.pt from the previous run
        last_pt = os.path.join(save_dir, "run_1/weights/last.pt")
        if os.path.exists(last_pt):
            model = YOLO(last_pt)
            print(f"[INFO] Resuming training from checkpoint: {last_pt}")
            results = model.train(resume=True)
            print("\n[SUCCESS] Resume process completed!")
            return
        else:
            print(f"[ERROR] Could not find checkpoint {last_pt} to resume from.")
            return

    # Initialize model based on Stage 1 best weights (if available)
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, "models/stage1_scut/yolo11s_ultimate_head/weights/best.pt")
    
    if not os.path.exists(model_path):
        print(f"[WARNING] Stage 1 model not found at {model_path}.")
        print("[INFO] Falling back to the base yolo11s.pt model.")
        model_path = 'yolo11s.pt'
        
    model = YOLO(model_path)
    print(f"[INFO] Loaded model: {model_path}")

    # Start the fine-tuning process
    results = model.train(
        data=data_yaml,
        epochs=50,          # 50 epochs is sufficient for this large dataset
        imgsz=640,          # Resolution 640 to synchronize with SAHI slicing grid
        batch=16,           # Optimized batch size for VRAM efficiency
        rect=False,         # Square images matching SAHI slice geometry
        device=0,           # Use GPU
        
        optimizer='auto',
        cos_lr=True,
        patience=10,        # Faster early stopping
        
        mosaic=1.0,         
        mixup=0.1,          
        
        project=save_dir,
        name='run_1'
    )
    
    print(f"\n[SUCCESS] Stage 2 completed! Model weights saved at: {save_dir}/run_1/weights/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 Training: CrowdHuman Fine-Tuning")
    parser.add_argument("--resume", action="store_true", help="Resume training from the last checkpoint")
    parser.add_argument("--model", type=str, default=None, help="Path to initial .pt file (defaults to Stage 1 best.pt)")
    args = parser.parse_args()
    
    train_stage2(resume=args.resume, model_path=args.model)
