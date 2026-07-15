# 🚶 Crowd Tracking System with YOLO11, SAHI, and ByteTrack

A highly robust pedestrian counting system engineered to handle dense crowds and occlusions in high-resolution video streams. This system leverages **YOLO11**, **SAHI** (Slicing Aided Hyper Inference) for detecting extremely small objects, and **ByteTrack** for stable identity tracking.

It utilizes an advanced mathematical **Buffer Zone** technique via cross-products to eliminate counting flicker, guaranteeing highly accurate directional counting (IN/OUT) across a custom diagonal counting line.

---

## ✨ Key Features
- **Interactive Web Dashboard (NEW)**: A professional, iOS-style GUI built with **Streamlit** to adjust confidence thresholds, track buffers, and counting line coordinates in real-time, accompanied by live dynamic charts.
- **SAHI Integration**: Slices high-resolution frames into smaller grids to detect tiny heads that standard YOLO misses, reducing false negatives drastically.
- **Custom Tracker Tuning**: Implemented customized ByteTrack configurations to minimize Identity Switches (ID Fragmentation) in crowded occlusions.
- **Buffer Zone Counting**: Prevents multiple counts (flickering) when a person lingers on the counting line. 
- **Automated Ground Truth Evaluation**: A robust evaluation pipeline that automatically parses ground truth data and compares them frame-by-frame against the AI predictions to compute Directional Accuracy.
- **Centralized Configuration**: Code architecture utilizes `pathlib` and a centralized `ProjectPaths` class for seamless execution from any directory.

---

## 📁 Project Structure

```text
Person-Counting/
├── configs/                  # Tracker and Dataset configurations
├── data/raw/                 # Source videos (e.g., TownCentre_1min.mp4) & GT
├── models/trained/           # Pre-trained YOLO weights
├── src/                      # Core Source Code
│   ├── app/                  
│   │   └── streamlit_app.py  # Interactive Web Dashboard
│   ├── utils/
│   │   ├── config.py         # ProjectPaths and InferenceConfig 
│   │   └── video_handler.py
│   ├── inference/            
│   │   ├── count_standard.py # Baseline YOLO tracking
│   │   └── count_sahi.py     # Advanced SAHI tracking
│   └── evaluation/           
│       └── evaluate.py       # Ground truth accuracy validation
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

# Install dependencies (includes Streamlit & Plotly)
pip install -r requirements.txt
```

---

## 🌐 Interactive Web Dashboard

To launch the beautiful real-time inference dashboard:
```bash
streamlit run src/app/streamlit_app.py
```
**Features:**
- Upload your own `.mp4` files.
- Live Preview to adjust the Counting Line `(X, Y)` coordinates accurately.
- Real-time tracking overlay with minimized WebSocket payload for zero-lag streaming.
- Auto-generated Analytics Chart upon completion.

---

## 🚀 CLI Inference & Tracking

You can also choose to run inference purely via the command line.

### 1. Standard YOLO Tracking (`count_standard.py`)
Fast inference using standard YOLO11 + ByteTrack.
```bash
python src/inference/count_standard.py
```

### 2. SAHI Tracking (`count_sahi.py`)
Uses Slicing Aided Hyper Inference combined with tuned ByteTrack. Recommended for dense, high-resolution crowds.
```bash
python src/inference/count_sahi.py
```

Outputs will be saved in `outputs/count_standard/` and `outputs/count_sahi/` including:
- Annotated `.mp4` video with HUD.
- Per-frame counts and detection coordinates `.csv`/`.txt`.

---

## 📊 Evaluation & Metrics

Evaluate the model against human-labeled ground truth (TownCentre dataset) using the built-in evaluation script:

```bash
python src/evaluation/evaluate.py
```

### Recent Benchmark (TownCentre - 1500 frames)
| Metric | Standard YOLO (`count_standard.py`) | SAHI + YOLO (`count_sahi.py`) |
|--------|-------------------------------------|-------------------------------|
| **Total Unique IDs** | 163 | **178** |
| **IN Accuracy** | 100.00% | **100.00%** |
| **OUT Accuracy** | 92.31% | **96.15%** |
| **Overall Accuracy** | 70.40% | **66.13%** |

*Note: The Ground Truth contains 90 distinct people. Using `CONFIDENCE_THRESHOLD = 0.1` allows ByteTrack to utilize low-confidence bounding boxes to maintain identities behind occlusions.*

---

## 🛠 Tech Stack
- **Ultralytics**: YOLO11 architecture
- **SAHI**: Slicing Aided Hyper Inference
- **Supervision**: Advanced computer vision tracking (ByteTrack & LineZone)
- **Streamlit & Plotly**: Interactive Data Dashboards

## 📄 License
This project is for research and educational purposes.
