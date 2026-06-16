# 🚶 Crowd Tracking System with YOLO11, SAHI, and BoT-SORT

A highly robust pedestrian counting system engineered to handle dense crowds and occlusions in high-resolution video streams. This system leverages **YOLO11s**, **SAHI** (Slicing Aided Hyper Inference) for detecting extremely small objects in 1080p frames, and **BoT-SORT / ByteTrack** for stable identity tracking.

It utilizes an advanced mathematical **Buffer Zone** technique via cross-products to eliminate counting flicker, guaranteeing highly accurate directional counting (IN/OUT) across a custom diagonal counting line.

---

## ✨ Key Features
- **SAHI Integration**: Slices high-resolution frames into smaller grids to detect tiny heads that standard YOLO misses, reducing false negatives drastically.
- **Custom Tracker Tuning**: Implemented customized BoT-SORT and ByteTrack configurations (`track_buffer` tuned to 60-120 frames) to minimize Identity Switches (ID Fragmentation) in crowded occlusions.
- **Buffer Zone Counting**: Prevents multiple counts (flickering) when a person lingers on the counting line. Achieves **97-100% OUT Counting Accuracy**.
- **Automated Ground Truth Evaluation**: A robust evaluation pipeline that automatically parses Oxford TownCentre ground truth `.top` files and compares them frame-by-frame against the AI predictions to compute Total Unique IDs and Directional Accuracy.
- **Two-Stage Training**: Clean architecture transitioning from SCUT-HEAD (imgsz=1280) to CrowdHuman (imgsz=640).

---

## 📁 Project Structure

```text
Person-Counting/
├── archive/                  # Legacy code and experiments
├── configs/                  # Tracker and Dataset configurations
│   ├── scut_data.yaml        
│   ├── crowdhuman_data.yaml  
│   └── custom_tracker.yaml   # Tuned BoT-SORT parameters for dense crowds
├── data/raw/                 # Source videos (e.g., TownCentre_1min.mp4) & GT
├── models/trained/           # Pre-trained YOLO weights (HeadDetect_v1.pt)
├── outputs/count/            # Tracking videos, CSV logs, and detection logs
├── src/                      # Core Source Code
│   ├── data_prep/            # Kaggle data fetchers
│   ├── training/             # Stage 1 and Stage 2 training pipelines
│   ├── inference/            
│   │   ├── track_standard.py # Baseline YOLO + BoT-SORT tracking
│   │   └── track_sahi.py     # Advanced SAHI + ByteTrack tracking
│   └── evaluation/           
│       └── evaluate.py       # Ground truth accuracy validation script
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

- Python 3.10+
- CUDA-capable GPU (Tested on NVIDIA RTX 4050 / T4)

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

## 🚀 Inference & Tracking

You can choose between the Standard YOLO approach and the highly accurate SAHI approach.

### 1. Standard YOLO Tracking (`track_standard.py`)
Uses the tuned BoT-SORT tracker defined in `configs/custom_tracker.yaml`. Fast, but struggles with tiny heads in the distance.
```bash
python src/inference/track_standard.py
```

### 2. SAHI Tracking (`track_sahi.py`)
Uses Slicing Aided Hyper Inference combined with a tuned `sv.ByteTrack` (`lost_track_buffer=60`). This is the recommended approach for dense, high-resolution crowds like the TownCentre dataset.
```bash
python src/inference/track_sahi.py
```

Outputs will be saved in `outputs/count/` including:
- Annotated `.mp4` video with HUD.
- Per-frame counts and detection coordinates `.csv`/`.txt`.

---

## 📊 Evaluation & Metrics

Once inference is complete, evaluate the model against the human-labeled ground truth using the built-in evaluation script:

```bash
python src/evaluation/evaluate.py
```

### Recent Benchmark (TownCentre - 1500 frames)
| Metric | Standard YOLO (`track_standard.py`) | SAHI + YOLO (`track_sahi.py`) |
|--------|-------------------------------------|-------------------------------|
| **Total Unique IDs** | 153 (Error: 63) | **136** (Error: 46) |
| **IN Accuracy** | 80.77% | **92.31%** |
| **OUT Accuracy** | 97.06% | **100.00%** |
| **Overall Accuracy** | 65.43% | **80.40%** |

*Note: The Ground Truth contains 90 distinct people. Object trackers frequently exceed this number due to ID Switches during prolonged occlusions, but our tuned `lost_track_buffer` greatly reduces this fragmentation.*

---

## 🛠 Tech Stack
- **Ultralytics**: YOLO11 architecture & BoT-SORT
- **SAHI**: Slicing Aided Hyper Inference
- **Supervision**: Advanced computer vision tracking (ByteTrack)
- **Kagglehub**: Dataset acquisition

## 📄 License
This project is for research and educational purposes.
