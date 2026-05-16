# 🚶 Person Counting System

A pedestrian counting system using **YOLOv11** and **ByteTrack** on surveillance footage. The system detects and tracks pedestrians crossing a defined line, classifying movement as **IN** (upward) or **OUT** (downward).

Includes a full pipeline: detection → tracking → line counting → accuracy evaluation against the **Oxford Town Centre** ground truth dataset. Also supports **fine-tuning a custom Head Detection model** for improved accuracy in crowded scenes.

---

## ✨ Features

- **Real-time Line Counting** — Count pedestrians crossing a configurable horizontal line
- **ByteTrack Integration** — Stable identity tracking across frames, minimizing ID switches
- **Ground Truth Evaluation** — Compare AI results against Oxford Town Centre annotations
- **Custom Head Detection Training** — Fine-tune YOLO on head annotations to drastically improve accuracy in crowds
- **Automatic Dataset Generation** — Convert `.top` annotation files into YOLO-format labels automatically
- **Modular Project Structure** — Clean `src/`, `configs/`, `models/`, `data/` layout ready for deployment

---

## 📁 Project Structure

```text
Person-Counting/
├── configs/
│   └── head_dataset.yaml        # YOLO dataset config for head detection training
├── data/
│   ├── raw/                     # Source video & ground truth files (gitignored)
│   │   ├── TownCentreXVID.mp4
│   │   ├── TownCentre-groundtruth.top
│   │   └── TownCentre-calibration.ci
│   └── datasets/
│       └── towncentre_head/     # Auto-generated YOLO dataset (gitignored)
│           ├── images/
│           │   ├── train/
│           │   └── val/
│           └── labels/
│               ├── train/
│               └── val/
├── experiments/                 # Exploratory scripts (detection, tracking tests)
│   ├── test_image.py
│   ├── test_tracking.py
│   └── test_video.py
├── models/
│   └── pretrained/              # YOLO base weights (gitignored)
│       ├── yolo11s.pt
│       └── yolo11m.pt
├── outputs/                     # All generated outputs (gitignored)
│   └── count/
│       ├── counting_output.mp4
│       └── summary.csv
├── src/                         # Core source code
│   ├── app.py                   # Main counting script
│   ├── evaluate.py              # Accuracy evaluation vs. ground truth
│   ├── prepare_data.py          # Auto-generate YOLO dataset from .top file
│   └── train.py                 # Fine-tune YOLO for head detection
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Requirements

- Python 3.10+
- CUDA-capable GPU (tested on NVIDIA RTX 4050 6GB VRAM)
- Conda (recommended) or virtualenv

**Python Dependencies:**
```
ultralytics>=8.3.0
opencv-python-headless
numpy
pandas
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/your-username/Person-Counting.git
cd Person-Counting
pip install -r requirements.txt
```

### 2. Download Required Files

Download the following files and place them in `data/raw/`:

| File | Source |
|---|---|
| `TownCentreXVID.mp4` | [Oxford Town Centre Dataset](http://www.robots.ox.ac.uk/~lav/Research/Projects/2009bbenfold_headpose/project.html) |
| `TownCentre-groundtruth.top` | Same page above |
| `yolo11s.pt` / `yolo11m.pt` | [Ultralytics YOLO11](https://github.com/ultralytics/assets/releases) → place in `models/pretrained/` |

### 3. Run Pedestrian Counting

```bash
# Use default video (data/raw/TownCentreXVID.mp4)
python src/app.py

# Or specify a custom video
python src/app.py path/to/your_video.mp4
```

**Output:**
- Annotated video saved to `outputs/count/counting_output.mp4`
- Summary CSV saved to `outputs/count/summary.csv`
- Press **`q`** on the display window to stop early

---

## 📊 Accuracy Evaluation

Compare AI results against the Oxford Town Centre ground truth:

```bash
# Run app.py first to generate outputs/count/summary.csv, then:
python src/evaluate.py
```

**Example output:**
```
=============================================
  GROUND TRUTH COUNTS
=============================================
  Total IN  (Downward): 37
  Total OUT (Upward):   41
=============================================

=============================================
  AI PREDICTED COUNTS
=============================================
  Total IN  (Downward): 95
  Total OUT (Upward):   97
=============================================
```

> **Note:** The default YOLO `person` class overestimates counts due to crowd occlusion causing ID switches. Training a dedicated **Head Detection** model (see below) resolves this significantly.

---

## 🧠 Train a Custom Head Detection Model

Training YOLO to detect **heads** instead of full bodies is the industry-standard solution for accuracy in crowded scenes — heads are almost never occluded by other people.

### Step 1: Generate the Dataset

The ground truth `.top` file already contains head bounding box coordinates for all 157 people. This script converts them automatically into YOLO format labels:

```bash
python src/prepare_data.py
```

This will create the `data/datasets/towncentre_head/` directory with ~207 images (80% train, 20% val) and their corresponding `.txt` label files.

### Step 2: Train the Model

```bash
python src/train.py
```

Training runs for **50 epochs** on your GPU using `yolo11s.pt` as the base (optimized for 6GB VRAM with `batch=4`). Results are saved to `outputs/runs/train_head/`.

### Step 3: Use Your Custom Model

Edit `src/app.py` and update the model path:

```python
# Line ~32 in src/app.py
MODEL_NAME = os.path.join(PROJECT_ROOT, "outputs/runs/train_head/weights/best.pt")
```

Then run `python src/app.py` again to see the improved counting!

---

## ⚙️ Configuration

All key parameters are at the top of each script. You can adjust them without touching the core logic:

### `src/app.py`

| Parameter | Default | Description |
|---|---|---|
| `MODEL_NAME` | `models/pretrained/yolo11m.pt` | Path to YOLO weights |
| `CONFIDENCE_THRESHOLD` | `0.5` | Detection confidence filter |
| `COUNTING_LINE` | `[(0, 500), (1920, 500)]` | Line coordinates `[(x1,y1),(x2,y2)]` |
| `PERSON_CLASS_ID` | `0` | COCO class ID for person |

### `src/evaluate.py`

| Parameter | Default | Description |
|---|---|---|
| `COUNTING_LINE_Y` | `500` | Must match `COUNTING_LINE` Y in `app.py` |

### `src/prepare_data.py`

| Parameter | Default | Description |
|---|---|---|
| `FRAME_SKIP` | `15` | Extract 1 frame every N frames |
| `TRAIN_RATIO` | `0.8` | 80% train, 20% val split |

### `src/train.py`

| Parameter | Default | Description |
|---|---|---|
| `epochs` | `50` | Training epochs |
| `batch` | `4` | Batch size (safe for 6GB VRAM) |
| `imgsz` | `640` | Input image size |

---

## 🔢 IN / OUT Logic

The counting direction is defined as follows:

| Direction | Label | Description |
|---|---|---|
| ↓ Top → Bottom | **IN** | Person moves downward across the line |
| ↑ Bottom → Top | **OUT** | Person moves upward across the line |

This matches the default Ultralytics `ObjectCounter` behavior with a horizontal line.

---

## 🛠 Hardware Tested

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop (6GB VRAM) |
| CUDA | 12.8 |
| PyTorch | 2.10.0+cu128 |
| Ultralytics | 8.3+ |
| OS | Ubuntu 22.04 |

> **⚠️ CUDA Crash Warning:** If you encounter `CUDA error: unspecified launch failure` during training or inference, reduce `batch` to `2` or allow the GPU to cool down before retrying. This is a hardware thermal issue, not a code bug.

---

## 📚 Dataset

This project uses the **Oxford Town Centre** dataset:

- **Video:** 1920×1080, 25 FPS, ~5 minutes, recorded on a UK high street
- **Annotations:** 157 unique pedestrian tracks with head & body bounding boxes, frame-by-frame
- **Format (`.top`):** `person_id, frame, head_valid, body_valid, headL, headT, headR, headB, bodyL, bodyT, bodyR, bodyB`

> The dataset is **not included** in this repository due to file size. Please download it from the [Oxford Visual Geometry Group](http://www.robots.ox.ac.uk/~lav/Research/Projects/2009bbenfold_headpose/project.html).

---

## 🗺 Roadmap

- [ ] Improve accuracy with trained Head Detection model
- [ ] Add multi-line / polygon region support
- [ ] Real-time RTSP stream support
- [ ] Flask/FastAPI web dashboard for live monitoring
- [ ] Speed estimation using `TownCentre-calibration.ci` camera parameters

---

## 📄 License

This project is for research and educational purposes.
