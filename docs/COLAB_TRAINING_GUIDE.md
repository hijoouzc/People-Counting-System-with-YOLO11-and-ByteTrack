# 🚀 Training Head Detection on Google Colab

## Yêu cầu trước khi bắt đầu

Vì dataset và model pretrained bị `.gitignore` nên không có trên GitHub. Bạn cần upload 2 thứ lên **Google Drive** trước:

1. **Video gốc** → `data/raw/TownCentreXVID.mp4`
2. **Model pretrained** → `models/pretrained/yolo11s.pt`

> Hoặc nếu đã có dataset sẵn, nén và upload: `zip -r head_dataset.zip data/datasets/towncentre_head/`

---

## Mở Google Colab

1. Vào [https://colab.research.google.com](https://colab.research.google.com)
2. Tạo **New Notebook**
3. Menu **Runtime → Change runtime type → GPU (T4)**
4. Copy từng cell bên dưới và chạy lần lượt

---

## Cell 1 — Kiểm tra GPU
```python
!nvidia-smi
```

## Cell 2 — Clone project từ GitHub + Cài thư viện
```python
# Clone repo
!git clone https://github.com/hijoouzc/People-Counting-System-with-YOLO11-and-ByteTrack.git
%cd People-Counting-System-with-YOLO11-and-ByteTrack

# Cài thư viện
!pip install -r requirements.txt -q
print("✅ Project cloned & dependencies installed!")
```

## Cell 3 — Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

## Cell 4 — Copy dataset từ Drive vào project
```python
import os, shutil, zipfile

project_dir = '/content/People-Counting-System-with-YOLO11-and-ByteTrack'

# === OPTION A: Nếu bạn upload file zip dataset ===
zip_path = '/content/drive/MyDrive/head_dataset.zip'
if os.path.exists(zip_path):
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(project_dir)
    print("✅ Dataset extracted from zip!")

# === OPTION B: Nếu bạn upload video gốc, chạy prepare_data.py để tạo dataset ===
# Uncomment 3 dòng dưới nếu dùng Option B:
# shutil.copy('/content/drive/MyDrive/TownCentreXVID.mp4', f'{project_dir}/data/raw/')
# shutil.copy('/content/drive/MyDrive/TownCentre-groundtruth.top', f'{project_dir}/data/raw/')
# !python src/prepare_data.py

# Copy model pretrained
os.makedirs(f'{project_dir}/models/pretrained', exist_ok=True)
drive_model = '/content/drive/MyDrive/yolo11s.pt'
if os.path.exists(drive_model):
    shutil.copy(drive_model, f'{project_dir}/models/pretrained/yolo11s.pt')
    print("✅ Model copied from Drive!")
else:
    print("⚠️ yolo11s.pt not found on Drive, will auto-download during training")

# Verify
!ls data/datasets/towncentre_head/images/train/ | wc -l
!echo "images found in train set"
```

## Cell 5 — Train 🔥
```python
from ultralytics import YOLO
import os

# Load model (từ project hoặc auto-download)
model_path = 'models/pretrained/yolo11s.pt'
if os.path.exists(model_path):
    model = YOLO(model_path)
else:
    model = YOLO('yolo11s.pt')

# Train với cấu hình tối ưu cho Colab T4 (16GB VRAM)
results = model.train(
    data='configs/head_dataset.yaml',
    epochs=100,         # Gấp đôi so với local → model học kỹ hơn
    imgsz=1280,         # Gấp đôi so với local → detect đầu nhỏ tốt hơn
    batch=16,           # T4 16GB cho phép batch lớn hơn nhiều
    device=0,
    project='outputs/runs',
    name='train_head',
    exist_ok=True,
    workers=4,
    patience=20,        # Early stopping nếu không cải thiện sau 20 epoch
    cos_lr=True,        # Cosine LR schedule
)

print("\n🎉 Training complete!")
```

## Cell 6 — Xem kết quả
```python
from IPython.display import Image, display

results_dir = 'outputs/runs/train_head'

# Loss & metrics curves
display(Image(filename=f'{results_dir}/results.png', width=800))

# Predictions on validation
if os.path.exists(f'{results_dir}/val_batch0_pred.jpg'):
    display(Image(filename=f'{results_dir}/val_batch0_pred.jpg', width=800))
```

## Cell 7 — Lưu model về Google Drive
```python
import shutil

src = 'outputs/runs/train_head/weights/best.pt'
dst = '/content/drive/MyDrive/best_head_detection.pt'

shutil.copy2(src, dst)
size_mb = os.path.getsize(dst) / 1024 / 1024
print(f"✅ Model saved to Google Drive: {dst}")
print(f"📦 Size: {size_mb:.1f} MB")
```

---

## Sau khi train xong — Đưa model về máy local

1. Vào Google Drive → download `best_head_detection.pt`
2. Đặt vào:
   ```
   Person-Counting/outputs/runs/train_head/weights/best.pt
   ```
3. Sửa `src/app.py` (dòng ~32):
   ```python
   MODEL_NAME = os.path.join(PROJECT_ROOT, "outputs/runs/train_head/weights/best.pt")
   ```
4. Chạy:
   ```bash
   python src/app.py
   ```

---

## Đồng bộ code 2 chiều (Local ↔ Colab)

### Khi sửa code ở local → Muốn cập nhật trên Colab:
```python
# Chạy cell này trong Colab
%cd /content/People-Counting-System-with-YOLO11-and-ByteTrack
!git pull origin main
```

### Khi sửa code trên Colab → Muốn cập nhật về local:
```bash
# Chạy trên terminal máy local
cd ~/Documents/VSCode/Person-Counting
git pull origin main
```

> **Lưu ý:** Chỉ có **code** được đồng bộ qua Git. Dataset và model weights
> phải copy thủ công qua Google Drive vì chúng bị `.gitignore`.

---

## So sánh Local vs Colab

| Tham số | Local (RTX 4050) | Colab (T4) |
|---|---|---|
| VRAM | 6GB | **16GB** |
| `batch` | 4 | **16** |
| `imgsz` | 640 | **1280** |
| `epochs` | 50 | **100** |
| Crash risk | ⚠️ Cao | ✅ Không |
| mAP50 dự kiến | ~50% | **65-75%** |
| Thời gian | ~8 phút | ~15-20 phút |
