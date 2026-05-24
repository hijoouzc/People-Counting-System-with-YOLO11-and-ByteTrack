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

MODEL_NAME = os.path.join(PROJECT_ROOT, "outputs/runs/train_head/weights/best.pt")
CONFIDENCE_THRESHOLD = 0.3
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


def run_counting(video_path: str, max_frames: int | None = None) -> None:
    # Ensure we are running from project root
    os.chdir(PROJECT_ROOT)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    frames_dir = os.path.join(OUTPUT_DIR, 'frames')
    os.makedirs(frames_dir, exist_ok=True)
    out_video_path = os.path.join(OUTPUT_DIR, "counting_output.mp4")
    out_csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
    perframe_csv = os.path.join(OUTPUT_DIR, "counts_per_frame.csv")
    detections_txt = os.path.join(OUTPUT_DIR, "detections_per_frame.txt")

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
    perframe_rows = []
    # open detections file for write
    det_f = open(detections_txt, 'w')
    det_f.write('frame,track_id,class_id,x1,y1,x2,y2,conf\n')

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

        # Save per-frame detections/tracks (best-effort extraction)
        tracked_count = 0
        try:
            # Some versions expose `results.tracks` or `results.tracked_objects`
            tracks = getattr(results, 'tracks', None) or getattr(results, 'tracked_objects', None)
            if tracks is not None:
                for tr in tracks:
                    # attempt common attribute names
                    tid = getattr(tr, 'id', getattr(tr, 'track_id', None))
                    if tid is None:
                        # try dictionary-like
                        tid = tr.get('id') if isinstance(tr, dict) else ''
                    box = getattr(tr, 'box', None) or getattr(tr, 'bbox', None) or None
                    if box is not None:
                        # box may be object with xyxy or list
                        if hasattr(box, 'xyxy'):
                            x1, y1, x2, y2 = box.xyxy
                        elif isinstance(box, (list, tuple)) and len(box) >= 4:
                            x1, y1, x2, y2 = box[:4]
                        else:
                            x1 = y1 = x2 = y2 = 0
                    else:
                        # fallback: try results.boxes
                        boxes = getattr(results, 'boxes', None)
                        if boxes is not None and len(boxes):
                            b = boxes[0]
                            if hasattr(b, 'xyxy'):
                                x1, y1, x2, y2 = b.xyxy
                            else:
                                x1 = y1 = x2 = y2 = 0
                        else:
                            x1 = y1 = x2 = y2 = 0

                    cls = getattr(tr, 'cls', getattr(tr, 'class_id', '0'))
                    conf = getattr(tr, 'conf', getattr(tr, 'confidence', 0.0))
                    det_f.write(f"{frame_count},{tid},{cls},{x1},{y1},{x2},{y2},{conf}\n")
                    tracked_count += 1
            else:
                # fallback: try results.boxes (detections without track ids)
                boxes = getattr(results, 'boxes', None)
                if boxes is not None:
                    for b in boxes:
                        if hasattr(b, 'xyxy'):
                            x1, y1, x2, y2 = b.xyxy
                        else:
                            vals = getattr(b, 'tolist', lambda: [])()
                            if vals:
                                x1, y1, x2, y2 = vals[:4]
                            else:
                                x1 = y1 = x2 = y2 = 0
                        cls = getattr(b, 'cls', 0)
                        conf = getattr(b, 'conf', 0.0)
                        det_f.write(f"{frame_count},, {cls},{x1},{y1},{x2},{y2},{conf}\n")
                        tracked_count += 1
        except Exception:
            # any extraction error should not stop processing
            tracked_count = 0

        # record per-frame counts: frame, in_count, out_count, tracked_count
        in_count = getattr(results, 'in_count', 0)
        out_count = getattr(results, 'out_count', 0)
        perframe_rows.append((frame_count, in_count, out_count, tracked_count))
        
        # Optional: Print progress every 100 frames
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames... IN: {getattr(results, 'in_count', 0)}, OUT: {getattr(results, 'out_count', 0)}")

        # The 'show=True' parameter automatically handles cv2.imshow
        # We just need to handle the quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # optional: stop early if max_frames provided
        if max_frames is not None and frame_count >= max_frames:
            print(f"Reached max_frames={max_frames}, stopping early")
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    det_f.close()

    # write per-frame counts CSV
    try:
        import csv as _csv
        with open(perframe_csv, 'w', newline='') as _f:
            writer = _csv.writer(_f)
            writer.writerow(['frame', 'in_count', 'out_count', 'tracked_count'])
            for r in perframe_rows:
                writer.writerow(r)
    except Exception:
        pass

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
    print(f"Per-frame counts saved to: {perframe_csv}")
    print(f"Per-frame detections saved to: {detections_txt}")


def main():
    video_path = get_video_path()
    run_counting(video_path)


if __name__ == "__main__":
    main()
