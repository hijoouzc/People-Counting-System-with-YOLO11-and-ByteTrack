import os
import yaml
import kagglehub

def prepare_scut_head(output_config_path="configs/scut_data.yaml"):
    print("[INFO] Downloading and locating SCUT-HEAD dataset via KaggleHub...")
    base_path = kagglehub.dataset_download("hoangxuanviet/scut-head")
    print(f"[INFO] KaggleHub base directory: {base_path}")

    train_dir = None
    val_dir = None

    for root, dirs, files in os.walk(base_path):
        if 'train' in root and root.endswith('images'):
            train_dir = root
        if ('val' in root or 'valid' in root) and root.endswith('images'):
            val_dir = root

    if train_dir and val_dir:
        print("[INFO] Successfully located image directories!")
        print(f"  -> Train: {train_dir}")
        print(f"  -> Val:   {val_dir}")

        os.makedirs(os.path.dirname(output_config_path), exist_ok=True)
        
        data_yaml = {
            'train': train_dir,
            'val': val_dir,
            'nc': 1,
            'names': {0: 'head'}
        }
        
        with open(output_config_path, 'w') as f:
            yaml.dump(data_yaml, f, sort_keys=False)
        print(f"[SUCCESS] Configuration file created at: {output_config_path}")
    else:
        print("[ERROR] Could not find 'images' directories in the dataset. Downloaded structure:")
        for root, dirs, files in os.walk(base_path):
            level = root.replace(base_path, '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}- {os.path.basename(root)}/")

if __name__ == "__main__":
    # Ensure execution from the project root directory
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.chdir(PROJECT_ROOT)
    config_path = os.path.join(PROJECT_ROOT, "configs/scut_data.yaml")
    prepare_scut_head(config_path)
