"""
Step 2: ByteTrack Tracking Test
===================================
Purpose:
    Verify that YOLO can accurately track pedestrians across frames using ByteTrack.
    Each person should be assigned a unique ID that remains stable.

Usage:
    python test_tracking.py
    python test_tracking.py data/earthcam_video.mp4

Output:
    - Real-time display window with IDs (press 'q' to exit early)
    - Annotated video saved to outputs/track/
"""

import sys
import os
from ultralytics import YOLO


# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_NAME = os.path.join(PROJECT_ROOT, "models/trained/HeadDetect_v1.pt")       # Balanced speed/accuracy model
CONFIDENCE_THRESHOLD = 0.30      # Minimum detection confidence
GPU_DEVICE = 0                  # GPU index (0 = first GPU, RTX 4050)
PERSON_CLASS_ID = 0             # COCO class index for "person"
DEFAULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "data/raw/TownCentre_1min.mp4")
TRACKER_TYPE = "bytetrack.yaml" # Use ByteTrack for object tracking
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
        print(f"        Please place your video at: {DEFAULT_VIDEO_PATH}")
        sys.exit(1)

    return path


def print_run_config(video_path: str) -> None:
    """Print the current run configuration for debugging reference."""
    print("=" * 55)
    print("  Step 2: ByteTrack Tracking Test")
    print("=" * 55)
    print(f"  Video   : {video_path}")
    print(f"  Model   : {MODEL_NAME}")
    print(f"  Tracker : {TRACKER_TYPE}")
    print(f"  Conf    : {CONFIDENCE_THRESHOLD}")
    print(f"  Device  : GPU {GPU_DEVICE}")
    print("=" * 55)
    print()


def run_tracking(video_path: str) -> None:
    """
    Run YOLO person tracking on the given video.

    - Tracks class 0 (person) using ByteTrack.
    - Displays annotated frames with IDs in a real-time window.
    - Saves the annotated video to outputs/track/.
    """
    model = YOLO(MODEL_NAME)

    print("[INFO] Starting tracking... Press 'q' on the video window to exit early.")
    print()

    # Use model.track instead of predict
    # persist=True ensures tracking IDs are maintained between frames
    # stream=True prevents RAM from accumulating results
    for _ in model.track(
        source=video_path,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        tracker=TRACKER_TYPE,
        persist=True,       # <--- Crucial for tracking IDs
        show=True,          # Display real-time window
        save=True,          # Save annotated video
        project=OUTPUT_DIR,
        name="track",
        exist_ok=True,
        device=GPU_DEVICE,
        stream=True,        # <--- Prevents RAM overflow
    ):
        pass


def print_checklist() -> None:
    """Print post-run evaluation checklist."""
    print()
    print("=" * 55)
    print("  Step 2 Complete!")
    print("  Output saved to: outputs/track/")
    print("=" * 55)
    print()
    print("  Evaluation Checklist:")
    print("  [?] Does each person have a unique ID?")
    print("  [?] Are the IDs stable (not swapping or dropping frequently)?")
    print("  [?] Is inference speed smooth? (check ms/frame in terminal)")
    print()


# ============================================================
# Main Entry Point
# ============================================================
def main():
    video_path = get_video_path()
    print_run_config(video_path)
    run_tracking(video_path)
    print_checklist()


if __name__ == "__main__":
    main()
