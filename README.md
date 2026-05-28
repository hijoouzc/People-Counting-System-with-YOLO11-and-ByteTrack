# 🚶 Crowd Tracking System with YOLO11, SAHI, and ByteTrack

A robust pedestrian counting system engineered to handle dense crowds and occlusions in high-resolution video streams. This system leverages **YOLO11s**, **SAHI** (Slicing Aided Hyper Inference) for detecting small objects in high-resolution frames, and **ByteTrack** for stable identity tracking.

The project features a **Clean Architecture** with a fully automated, two-stage training pipeline transitioning from the **SCUT-HEAD** dataset to the massive **CrowdHuman** dataset.

---

## ✨ Key Features
- **SAHI Integration**: Slices high-resolution frames (e.g., 1080p) into smaller grid patches (640x640) to detect tiny heads that standard YOLO would miss.
- **Two-Stage Training**: 
  - *Stage 1*: Pre-train YOLO11s on SCUT-HEAD (imgsz=1280) for robust head detection.
  - *Stage 2*: Fine-tune on CrowdHuman (imgsz=640) to synchronize with SAHI slicing geometry.
- **ByteTrack & Supervision**: Seamless tracking and counting logic using Roboflow's powerful `supervision` library.
- **Automated Data Prep**: Directly fetches and converts datasets from Kaggle using `kagglehub`.
- **Memory Optimized**: Frame-by-frame generator-based video processing to prevent RAM overload.

---

## 📁 Project Structure

```text
Person-Counting/
├── archive/                  # Legacy code and experiments
├── configs/                  # Auto-generated dataset configurations
│   ├── scut_data.yaml        
│   └── crowdhuman_data.yaml  
├── data/                     
│   ├── crowdhuman_yolo/      # Prepared CrowdHuman YOLO format
│   └── raw/                  # Source videos (e.g., TownCentreXVID.mp4)
├── models/                   
│   ├── stage1_scut/          # Stage 1 weights
│   └── stage2_crowdhuman/    # Stage 2 weights
├── outputs/                  # Annotated output videos
├── src/                      # Core Source Code
│   ├── data_prep/            
│   │   ├── prep_scut.py      
│   │   └── prep_crowd.py     
│   ├── training/             
│   │   ├── train_stage1.py   
│   │   └── train_stage2.py   
│   └── inference/            
│       └── track_sahi.py     
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

- Python 3.10+
- CUDA-capable GPU (Tested on NVIDIA RTX 4050 / T4)
- Recommended environment: `conda`

```bash
# Create and activate environment
conda create -n ai python=3.10 -y
conda activate ai

# Install PyTorch with CUDA support (adjust for your hardware)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Pipeline Usage

The pipeline is divided into three distinct phases. Ensure you run them sequentially from the root project directory.

### Phase 1: Data Preparation
Automatically fetch datasets from Kaggle and format them for YOLO.
```bash
# Prepare SCUT-HEAD dataset (Stage 1)
python src/data_prep/prep_scut.py

# Prepare CrowdHuman dataset (Stage 2)
# Note: Downloads ~30GB of data and converts JSON to YOLO format
python src/data_prep/prep_crowd.py
```

### Phase 2: Model Training
Train the head detection architecture across two datasets.
```bash
# Stage 1: Base training on SCUT-HEAD (High-res 1280)
python src/training/train_stage1.py

# Stage 2: Fine-tuning on CrowdHuman (SAHI-compatible 640)
# Supports resuming via --resume flag
python src/training/train_stage2.py
```

### Phase 3: Inference & Tracking
Run the SAHI + ByteTrack inference pipeline on your video. Place your input video in `data/raw/` (e.g., `TownCentreXVID.mp4`).

```bash
# Run tracking with default paths
python src/inference/track_sahi.py

# Run tracking on a custom video
python src/inference/track_sahi.py \
    --video data/raw/my_video.mp4 \
    --model models/stage2_crowdhuman/run_1/weights/best.pt \
    --output outputs/my_output.mp4
```

> **Note on Counting Line:** The counting line (`LineZone`) is hardcoded in `src/inference/track_sahi.py` for the default `TownCentreXVID.mp4` camera angle. Adjust `start_pt` and `end_pt` inside the script to match your specific video angle.

---

## 🛠 Tech Stack
- **Ultralytics**: YOLO11 architecture
- **SAHI**: Slicing Aided Hyper Inference
- **Supervision**: Advanced computer vision tracking (ByteTrack) and annotations
- **Kagglehub**: Dataset acquisition

---

## 🗺 Roadmap
- [x] Integrate SAHI for micro-head detection
- [x] Standardize training pipeline to Clean Architecture
- [x] Integrate Supervision ByteTrack for seamless counting
- [ ] Add RTSP stream processing
- [ ] Implement polygon zones (Region counting)
- [ ] Speed estimation using calibration data

---

## 📄 License
This project is for research and educational purposes.
