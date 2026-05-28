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
    - Annotated video saved to outputs/predict/
"""

import sys
import os
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_NAME = os.path.join(PROJECT_ROOT, "/home/hoinguyen/Downloads/best(1).pt")       # Balanced speed/accuracy model
CONFIDENCE_THRESHOLD = 0.15      # Minimum detection confidence
GPU_DEVICE = 0                  # GPU index (0 = first GPU, RTX 4050)
PERSON_CLASS_ID = 0             # COCO class index for "person"
DEFAULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "data/raw/TownCentre_1min.mp4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")


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
    - Saves the annotated video to outputs/predict/.
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
        save=True,          # Save annotated video to outputs/predict/
        project=OUTPUT_DIR,
        name="predict",
        exist_ok=True,
        device=GPU_DEVICE,
        stream=True,        # <--- Prevents RAM overflow
    ):
        pass


def print_checklist() -> None:
    """Print post-run evaluation checklist."""
    print()
    print("=" * 55)
    print("  Step 1 Complete!")
    print("  Output saved to: outputs/predict/")
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
