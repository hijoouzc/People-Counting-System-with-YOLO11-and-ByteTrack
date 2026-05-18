"""
Dataset Preparation Script for Head Detection
===========================================
Purpose:
    Extract frames from TownCentreXVID.mp4 and generate YOLO format labels 
    from TownCentre-groundtruth.top (using head coordinates).
    
    To save time and disk space, it extracts 1 frame every 15 frames.
"""

import os
import csv
import cv2
import random
import shutil
from collections import defaultdict

# ==========================================
# Configuration (Paths are relative to project root)
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

VIDEO_PATH = os.path.join(PROJECT_ROOT, "data/raw/TownCentreXVID.mp4")
GT_PATH = os.path.join(PROJECT_ROOT, "data/raw/TownCentre-groundtruth.top")
DATASET_DIR = os.path.join(PROJECT_ROOT, "data/datasets/towncentre_head")

FRAME_SKIP = 5
TRAIN_RATIO = 0.8

def setup_directories():
    # Clean up old dataset if exists
    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
        
    for split in ['train', 'val']:
        os.makedirs(os.path.join(DATASET_DIR, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(DATASET_DIR, 'labels', split), exist_ok=True)

def parse_groundtruth():
    print(f"[INFO] Parsing Ground Truth file: {GT_PATH}")
    # frame_num -> list of YOLO format labels: (class_id, x_center, y_center, width, height)
    labels_by_frame = defaultdict(list)
    
    if not os.path.exists(GT_PATH):
        print(f"[ERROR] Ground truth file not found at: {GT_PATH}")
        return labels_by_frame

    with open(GT_PATH, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 8:
                continue
                
            try:
                frame_num = int(row[1])
                head_valid = int(row[2])
                
                if head_valid == 1:
                    head_left = float(row[4])
                    head_top = float(row[5])
                    head_right = float(row[6])
                    head_bottom = float(row[7])
                    
                    # Normalize coordinates for 1920x1080 resolution
                    # YOLO format: x_center, y_center, width, height (all normalized 0-1)
                    img_w, img_h = 1920.0, 1080.0
                    
                    x_center = ((head_left + head_right) / 2.0) / img_w
                    y_center = ((head_top + head_bottom) / 2.0) / img_h
                    width = (head_right - head_left) / img_w
                    height = (head_bottom - head_top) / img_h
                    
                    # Clip to [0, 1] just to be safe
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    
                    class_id = 0 # 0 for "head"
                    
                    labels_by_frame[frame_num].append((class_id, x_center, y_center, width, height))
            except ValueError:
                continue
                
    print(f"[INFO] Found head annotations for {len(labels_by_frame)} unique frames.")
    return labels_by_frame

def extract_frames(labels_by_frame):
    print(f"[INFO] Extracting frames from {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video {VIDEO_PATH}")
        return
        
    frame_idx = 0
    saved_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        # Process every FRAME_SKIP-th frame
        if frame_idx % FRAME_SKIP == 0:
            # Check if we have annotations for this frame
            if frame_idx in labels_by_frame:
                # Random split 80/20
                split = 'train' if random.random() < TRAIN_RATIO else 'val'
                
                # Format names
                base_name = f"frame_{frame_idx:05d}"
                img_path = os.path.join(DATASET_DIR, 'images', split, f"{base_name}.jpg")
                txt_path = os.path.join(DATASET_DIR, 'labels', split, f"{base_name}.txt")
                
                # Save Image
                cv2.imwrite(img_path, frame)
                
                # Save Labels
                with open(txt_path, 'w') as f:
                    for lbl in labels_by_frame[frame_idx]:
                        # lbl format: (class_id, x, y, w, h)
                        f.write(f"{lbl[0]} {lbl[1]:.6f} {lbl[2]:.6f} {lbl[3]:.6f} {lbl[4]:.6f}\n")
                        
                saved_count += 1
                if saved_count % 50 == 0:
                    print(f"  -> Saved {saved_count} frames...")
                    
        frame_idx += 1
        
    cap.release()
    print(f"[INFO] Extraction complete. Total frames saved: {saved_count}")

def main():
    # Ensure we are running from project root
    os.chdir(PROJECT_ROOT)
    
    setup_directories()
    labels = parse_groundtruth()
    extract_frames(labels)
    
    if os.path.exists(os.path.join(DATASET_DIR, 'images', 'train')):
        train_count = len(os.listdir(os.path.join(DATASET_DIR, 'images', 'train')))
        val_count = len(os.listdir(os.path.join(DATASET_DIR, 'images', 'val')))
        
        print("\n" + "="*40)
        print("  DATASET READY")
        print("="*40)
        print(f"  Train images: {train_count}")
        print(f"  Val images:   {val_count}")
        print(f"  Total:        {train_count + val_count}")
        print("="*40)

if __name__ == "__main__":
    main()
