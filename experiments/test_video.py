"""
Step 1: Basic Video Detection Test
===================================
Purpose:
    Verify that YOLO can accurately detect pedestrians in the EarthCam video
    using the RTX 4050 GPU. No tracking or counting logic — detection only.

Usage:
    python test_video.py
    python test_video.py data/earthcam_video.mp4

Output:
    - Real-time display window (press 'q' to exit early)
    - Annotated video saved to runs/detect/predict/
"""

import sys
import os
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================
MODEL_NAME = "yolo11s.pt"       # Balanced speed/accuracy model
CONFIDENCE_THRESHOLD = 0.3      # Minimum detection confidence (30%)
GPU_DEVICE = 0                  # GPU index (0 = first GPU, RTX 4050)
PERSON_CLASS_ID = 0             # COCO class index for "person"
DEFAULT_VIDEO_PATH = "data/TownCentreXVID.mp4"


# ============================================================
# Helper Functions
# ============================================================
def get_video_path() -> str:
    """
    Resolve video path from command-line argument or use default.
    Exits with a clear error message if the file does not exist.
    """
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH

    if not os.path.exists(path):
        print(f"[ERROR] Video file not found: {path}")
        print(f"        Please place your EarthCam video at: {DEFAULT_VIDEO_PATH}")
        sys.exit(1)

    return path


def print_run_config(video_path: str) -> None:
    """Print the current run configuration for debugging reference."""
    print("=" * 55)
    print("  Step 1: Basic Detection Test")
    print("=" * 55)
    print(f"  Video   : {video_path}")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Conf    : {CONFIDENCE_THRESHOLD}")
    print(f"  Device  : GPU {GPU_DEVICE}")
    print("=" * 55)
    print()


def run_detection(video_path: str) -> None:
    """
    Run YOLO person detection on the given video.

    - Detects only class 0 (person) from the COCO dataset.
    - Displays annotated frames in a real-time window.
    - Saves the annotated video to runs/detect/predict/.
    """
    model = YOLO(MODEL_NAME)

    print("[INFO] Starting detection... Press 'q' on the video window to exit early.")
    print()

    # Use stream=True and iterate to prevent RAM from accumulating results
    for _ in model.predict(
        source=video_path,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        show=True,          # Display real-time window
        save=True,          # Save annotated video to runs/detect/predict/
        device=GPU_DEVICE,
        stream=True,        # <--- Prevents RAM overflow
    ):
        pass


def print_checklist() -> None:
    """Print post-run evaluation checklist."""
    print()
    print("=" * 55)
    print("  Step 1 Complete!")
    print("  Output saved to: runs/detect/predict/")
    print("=" * 55)
    print()
    print("  Evaluation Checklist:")
    print("  [?] Are bounding boxes accurately wrapping pedestrians?")
    print("  [?] Is inference speed smooth? (check ms/frame in terminal)")
    print()


# ============================================================
# Main Entry Point
# ============================================================
def main():
    video_path = get_video_path()
    print_run_config(video_path)
    run_detection(video_path)
    print_checklist()


if __name__ == "__main__":
    main()
