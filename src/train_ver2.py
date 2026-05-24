import os
import yaml
import kagglehub
from ultralytics import YOLO

# BẮT BUỘC: Khi chạy local, toàn bộ lệnh train phải nằm trong khối __main__ 
# để tránh lỗi crash do cơ chế Multiprocessing (đa tiến trình) của PyTorch.
if __name__ == '__main__':
    
    print("1. Đang tải và xác định bộ dữ liệu từ Kaggle xuống máy cá nhân...")
    # Tự động tải về thư mục cache cục bộ của máy (VD: ~/.cache/kagglehub)
    base_path = kagglehub.dataset_download("hoangxuanviet/scut-head")
    print(f"Thư mục gốc: {base_path}")

    # 2. Bật "Radar" dò tìm tự động mọi ngóc ngách
    train_dir = None
    val_dir = None

    for root, dirs, files in os.walk(base_path):
        if 'train' in root and root.endswith('images'):
            train_dir = root
        if ('val' in root or 'valid' in root) and root.endswith('images'):
            val_dir = root

    # 3. Xử lý kết quả và Train
    if train_dir and val_dir:
        print(f"\n2. Đã radar thành công thư mục ảnh chính xác!")
        
        # Tạo file YAML lưu ngay tại thư mục hiện tại (cùng chỗ với file train.py)
        data_yaml = {
            'train': train_dir,
            'val': val_dir,
            'nc': 1,
            'names': {0: 'head'}
        }
        yaml_path = 'scut_data.yaml' 
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, sort_keys=False)
        print(f"✅ Đã tạo file {yaml_path}.")

        print("\n3. Bắt đầu quá trình rèn luyện cường độ cao...")
        # Khởi tạo "não bộ" sơ khai của YOLO11s
        model = YOLO('yolo11s.pt')

        # Bắt đầu quá trình rèn luyện
        results = model.train(
            data=yaml_path,
            epochs=100,         
            imgsz=1280,         
            batch=4,            
            device=0,           # Kích hoạt Card đồ họa rời (NVIDIA GPU) trên máy
            patience=20,        
            optimizer='auto',   
            cos_lr=True,        
            
            # --- CÁC THÔNG SỐ ÉP XUNG CHO SCUT-HEAD (Giữ nguyên) ---
            box=7.5,            
            cls=0.5,            
            dfl=1.5,            
            mosaic=1.0,         
            
            # --- LƯU THẲNG VÀO THƯ MỤC LOCAL ---
            project='SCUT_HEAD_RUNS',  # Sẽ tự động tạo thư mục này ngay cạnh file train.py
            name='yolo11s_ultimate_head'
        )

        print("\n🎉 Đã train xong! File best.pt của bạn nằm trong thư mục: SCUT_HEAD_RUNS/yolo11s_ultimate_head/weights/")
    else:
        print("\n❌ Cảnh báo: Không tìm thấy thư mục 'images' trong bộ dataset này.")