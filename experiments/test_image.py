"""
Step 1a: Single Image Detection Test
=====================================
Purpose:
    Quick sanity check — run YOLO person detection on a single image
    before processing full video. Validates model accuracy and GPU setup.

Usage:
    python test_image.py
    python test_image.py data/image.png

Output:
    - Annotated image saved to runs/detect/predict/
    - Detection summary printed to terminal
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
DEFAULT_IMAGE_PATH = "data/image.png"


# ============================================================
# Helper Functions
# ============================================================
def get_image_path() -> str:
    """
    Resolve image path from command-line argument or use default.
    Exits with a clear error message if the file does not exist.
    """
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE_PATH

    if not os.path.exists(path):
        print(f"[ERROR] Image file not found: {path}")
        print(f"        Please place your test image at: {DEFAULT_IMAGE_PATH}")
        sys.exit(1)

    return path


def print_detection_summary(results) -> None:
    """
    Print a concise summary of detection results.
    Shows total person count and per-detection confidence scores.
    """
    boxes = results[0].boxes
    person_count = len(boxes)

    print()
    print("=" * 55)
    print(f"  Detection Results: {person_count} person(s) found")
    print("=" * 55)

    if person_count > 0:
        confidences = boxes.conf.tolist()
        for i, conf in enumerate(confidences):
            print(f"  [{i+1}] Person — confidence: {conf:.2f}")

    print()
    print(f"  Inference speed: {results[0].speed['inference']:.1f} ms")
    print(f"  Output saved to: runs/detect/predict/")
    print("=" * 55)


# ============================================================
# Main Entry Point
# ============================================================
def main():
    image_path = get_image_path()

    print("=" * 55)
    print("  Step 1a: Single Image Detection Test")
    print("=" * 55)
    print(f"  Image  : {image_path}")
    print(f"  Model  : {MODEL_NAME}")
    print(f"  Conf   : {CONFIDENCE_THRESHOLD}")
    print(f"  Device : GPU {GPU_DEVICE}")
    print("=" * 55)
    print()

    # Load pretrained YOLO model (auto-downloads on first run)
    model = YOLO(MODEL_NAME)

    # Run detection: person class only, save annotated result
    results = model.predict(
        source=image_path,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE_THRESHOLD,
        save=True,              # Save annotated image to runs/detect/predict/
        device=GPU_DEVICE,
    )

    print_detection_summary(results)


if __name__ == "__main__":
    main()
