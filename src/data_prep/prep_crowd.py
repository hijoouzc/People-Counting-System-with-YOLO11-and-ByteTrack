import json
import os
import cv2
import glob
import shutil
import yaml
import kagglehub
from tqdm import tqdm

def locate_image(img_id, root_dir):
    """Locate the image file given its ID and root directory."""
    paths_to_check = [
        os.path.join(root_dir, f"Images/{img_id}.jpg"),
        os.path.join(root_dir, f"Images_val/{img_id}.jpg"),
        os.path.join(root_dir, f"{img_id}.jpg")
    ]
    for p in paths_to_check:
        if os.path.exists(p):
            return p
    return None

def prepare_crowdhuman(output_dir="data/crowdhuman_yolo", config_path="configs/crowdhuman_data.yaml"):
    print("[INFO] Downloading CrowdHuman dataset via KaggleHub...")
    dataset_path = kagglehub.dataset_download("leducnhuan/crowdhuman")
    print(f"[INFO] Dataset downloaded to: {dataset_path}")

    print("[INFO] Searching for .odgt annotation files...")
    train_odgt_list = glob.glob(f'{dataset_path}/**/annotation_train.odgt', recursive=True)
    val_odgt_list = glob.glob(f'{dataset_path}/**/annotation_val.odgt', recursive=True)

    if not train_odgt_list or not val_odgt_list:
        print("[ERROR] Could not find .odgt annotation files!")
        return

    train_odgt = train_odgt_list[0]
    val_odgt = val_odgt_list[0]
    dataset_root = os.path.dirname(train_odgt)

    print(f"  -> Train ODGT: {train_odgt}")
    print(f"  -> Val ODGT:   {val_odgt}")

    # Prepare directories
    train_img_dir = os.path.join(output_dir, "images/train")
    train_lbl_dir = os.path.join(output_dir, "labels/train")
    val_img_dir = os.path.join(output_dir, "images/val")
    val_lbl_dir = os.path.join(output_dir, "labels/val")

    datasets = [
        {'name': 'Train', 'odgt': train_odgt, 'img_dir': train_img_dir, 'lbl_dir': train_lbl_dir},
        {'name': 'Validation', 'odgt': val_odgt, 'img_dir': val_img_dir, 'lbl_dir': val_lbl_dir}
    ]

    for split in datasets:
        print(f"\n[INFO] Processing {split['name']} set...")
        os.makedirs(split['img_dir'], exist_ok=True)
        os.makedirs(split['lbl_dir'], exist_ok=True)

        with open(split['odgt'], 'r') as f:
            lines = f.readlines()

        count_heads = 0
        count_images = 0

        for line in tqdm(lines, desc=f"Progress {split['name']}"):
            data = json.loads(line)
            image_id = data['ID']

            img_path = locate_image(image_id, dataset_root)
            if not img_path:
                continue

            img = cv2.imread(img_path)
            if img is None: 
                continue
            height, width, _ = img.shape

            valid_boxes = []
            for box in data['gtboxes']:
                if box['tag'] == 'person' and 'hbox' in box:
                    x, y, w, h = box['hbox']
                    if w <= 0 or h <= 0: continue

                    # Normalize bounding box coordinates for YOLO format
                    x_center = max(0.0, min(1.0, (x + w / 2.0) / width))
                    y_center = max(0.0, min(1.0, (y + h / 2.0) / height))
                    w_norm = max(0.0, min(1.0, w / width))
                    h_norm = max(0.0, min(1.0, h / height))

                    valid_boxes.append(f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")
                    count_heads += 1

            if valid_boxes:
                label_path = os.path.join(split['lbl_dir'], f"{image_id}.txt")
                with open(label_path, 'w') as f_out:
                    f_out.write("\n".join(valid_boxes))

                target_img_path = os.path.join(split['img_dir'], f"{image_id}.jpg")
                if not os.path.exists(target_img_path):
                    # Use symlink to save space, fallback to copy if symlink fails
                    try:
                        os.symlink(img_path, target_img_path)
                    except OSError:
                        shutil.copy2(img_path, target_img_path)

                count_images += 1

        print(f"[SUCCESS] Finished {split['name']} set: {count_images} images, {count_heads} head annotations.")

    print("\n[INFO] Generating data.yaml configuration file...")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Use absolute paths for YOLO to avoid relative path resolution issues
    abs_output_dir = os.path.abspath(output_dir)
    data_yaml = {
        'train': os.path.join(abs_output_dir, "images/train"),
        'val': os.path.join(abs_output_dir, "images/val"),
        'nc': 1,
        'names': ['head']
    }
    with open(config_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"[SUCCESS] YAML configuration saved at: {config_path}")

if __name__ == "__main__":
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(PROJECT_ROOT)
    
    out_dir = os.path.join(PROJECT_ROOT, "data/crowdhuman_yolo")
    conf_path = os.path.join(PROJECT_ROOT, "configs/crowdhuman_data.yaml")
    
    prepare_crowdhuman(output_dir=out_dir, config_path=conf_path)
