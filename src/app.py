"""
Step 3: Line Counting
===================================
Purpose:
    Track pedestrians using ByteTrack and count how many cross a defined line.
    Outputs an annotated video and a CSV summary.

Usage:
    python test_counting.py
    python test_counting.py data/earthcam_video.mp4

Output:
    - Real-time display window with line and counts (press 'q' to exit early)
    - Annotated video saved to runs/detect/count/
    - CSV report saved to runs/detect/count/summary.csv
"""

import sys
import os

import cv2
import csv
from ultralytics.solutions.object_counter import ObjectCounter


# ============================================================
# Configuration (Paths are relative to project root)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_NAME = os.path.join(PROJECT_ROOT, "models/pretrained/yolo11m.pt")
CONFIDENCE_THRESHOLD = 0.5
PERSON_CLASS_ID = 0
DEFAULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "data/raw/TownCentreXVID.mp4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/count")

# Define the counting line.
# You can tweak these coordinates based on the camera angle.
# Format: [(x1, y1), (x2, y2)]
COUNTING_LINE = [(0, 500), (1920, 500)]


def get_video_path() -> str:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    if not os.path.exists(path):
        print(f"[ERROR] Video file not found: {path}")
        sys.exit(1)
    return path


def run_counting(video_path: str) -> None:
    # Ensure we are running from project root
    os.chdir(PROJECT_ROOT)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_video_path = os.path.join(OUTPUT_DIR, "counting_output.mp4")
    out_csv_path = os.path.join(OUTPUT_DIR, "summary.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(out_video_path,
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (w, h))

    # Initialize Object Counter for Ultralytics 8.3+ API
    counter = ObjectCounter(
        model=MODEL_NAME,           # Model is passed directly
        classes=[PERSON_CLASS_ID],  # Only track persons
        conf=CONFIDENCE_THRESHOLD,  # Confidence threshold
        region=COUNTING_LINE,       # Line points
        show=True,                  # Show real-time stream
        line_width=2, tracker="bytetrack.yaml",
    )

    print(f"[INFO] Starting counting... Press 'q' on the window to exit early.")
    print(f"[INFO] Counting line set at Y=500. Output will be saved to {OUTPUT_DIR}")
    print()

    frame_count = 0
    results = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # For Ultralytics 8.3+, ObjectCounter handles inference and tracking internally
        results = counter.process(frame)
        
        # Get annotated image
        annotated_frame = results.plot_im
        
        video_writer.write(annotated_frame)
        frame_count += 1
        
        # Optional: Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames... IN: {results.in_count}, OUT: {results.out_count}")

        # The 'show=True' parameter automatically handles cv2.imshow
        # We just need to handle the quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

    # Save to CSV
    final_in = results.in_count if results else 0
    final_out = results.out_count if results else 0

    with open(out_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Count"])
        writer.writerow(["Total IN", final_in])
        writer.writerow(["Total OUT", final_out])
        writer.writerow(["Total Frames Processed", frame_count])

    print("\n" + "=" * 55)
    print("  Step 3 Complete!")
    print(f"  Final Count - IN: {final_in} | OUT: {final_out}")
    print(f"  Output Video: {out_video_path}")
    print(f"  Output CSV:   {out_csv_path}")
    print("=" * 55)


def main():
    video_path = get_video_path()
    run_counting(video_path)


if __name__ == "__main__":
    main()
