"""
Step 3: Line Counting (Optimized for Head Detection)
===================================================
Purpose:
    Track pedestrian heads using ByteTrack and count how many cross a defined
    diagonal line. Outputs an annotated video and a CSV summary.
"""

import sys
import os
import cv2
import csv
from ultralytics import YOLO

# ============================================================
# Configuration (Paths are relative to project root)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_NAME = os.path.join(PROJECT_ROOT, "models/trained/HeadDetect_v1.pt")
CONFIDENCE_THRESHOLD = 0.15
DEFAULT_VIDEO_PATH = os.path.join(PROJECT_ROOT, "data/raw/TownCentre_1min.mp4")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs/count")

# ============================================================
# Counting Line Configuration (Diagonal Line Support)
# ============================================================
# Define two endpoints of the counting line
LINE_START = (0, 250)       # Left endpoint  (x, y)
LINE_END   = (1920, 550)    # Right endpoint (x, y)
BUFFER = 15                 # Buffer distance (in pixels) to prevent flickering


def cross_product_sign(line_start, line_end, point):
    """
    Determine which side of the line a point is on using the cross product.
    Returns:
        > 0 : point is BELOW the line (left-to-right direction)
        < 0 : point is ABOVE the line
        = 0 : point is exactly ON the line
    """
    ax, ay = line_start
    bx, by = line_end
    px, py = point
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def point_to_line_distance(line_start, line_end, point):
    """
    Calculate the perpendicular distance from a point to the counting line.
    Used for the buffer zone check.
    """
    ax, ay = line_start
    bx, by = line_end
    px, py = point
    # Length of the line segment
    line_len = ((bx - ax)**2 + (by - ay)**2) ** 0.5
    if line_len == 0:
        return float('inf')
    # Absolute perpendicular distance
    return abs((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / line_len


def get_video_path() -> str:
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VIDEO_PATH
    if not os.path.exists(path):
        print(f"[ERROR] Video file not found: {path}")
        sys.exit(1)
    return path


def run_counting(video_path: str, max_frames: int | None = None) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
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

    video_writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Initialize YOLO model directly
    model = YOLO(MODEL_NAME)

    print(f"[INFO] Starting tracking... Press 'q' on the window to exit early.")
    print(f"[INFO] Line: {LINE_START} -> {LINE_END}, Buffer = ±{BUFFER}px")
    print(f"[INFO] Output saved to {OUTPUT_DIR}\n")

    # Tracking variables
    track_history = {}  # Stores {track_id: last_cross_product_sign_outside_buffer}
    count_in = 0        # Moving from ABOVE to BELOW the line
    count_out = 0       # Moving from BELOW to ABOVE the line
    frame_count = 0
    perframe_rows = []

    # Open detection log file securely
    with open(detections_txt, 'w') as det_f:
        det_f.write('frame,track_id,class_id,x1,y1,x2,y2,conf\n')

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            tracked_count_this_frame = 0

            # Run Tracking
            results = model.track(frame, persist=True, tracker="bytetrack.yaml", conf=CONFIDENCE_THRESHOLD, verbose=False)[0]

            # 1. Draw diagonal counting line and buffer zone
            cv2.line(frame, LINE_START, LINE_END, (0, 255, 255), 3) # Main line (Yellow)

            # 2. Extract and process Bounding Boxes
            if results.boxes.id is not None:
                boxes = results.boxes.xyxy.cpu().numpy()
                track_ids = results.boxes.id.cpu().numpy().astype(int)
                confs = results.boxes.conf.cpu().numpy()
                clss = results.boxes.cls.cpu().numpy().astype(int)

                for box, track_id, conf, cls in zip(boxes, track_ids, confs, clss):
                    x1, y1, x2, y2 = box
                    xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Log detections
                    det_f.write(f"{frame_count},{track_id},{cls},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{conf:.2f}\n")
                    tracked_count_this_frame += 1

                    # --- PRECISE DIAGONAL LINE COUNTING LOGIC ---
                    center = (xc, yc)
                    dist = point_to_line_distance(LINE_START, LINE_END, center)
                    sign = cross_product_sign(LINE_START, LINE_END, center)

                    # Only process when the point is outside the buffer zone
                    is_outside_buffer = dist > BUFFER

                    if track_id in track_history and is_outside_buffer:
                        old_sign = track_history[track_id]

                        # Crossed from ABOVE to BELOW (sign changed from negative to positive)
                        if old_sign < 0 and sign > 0:
                            count_in += 1

                        # Crossed from BELOW to ABOVE (sign changed from positive to negative)
                        elif old_sign > 0 and sign < 0:
                            count_out += 1

                    # Update history only when outside buffer (or first appearance)
                    if track_id not in track_history or is_outside_buffer:
                        track_history[track_id] = sign

                    # Draw BBox and ID for observation
                    color = (0, 255, 0) if sign < 0 else (0, 165, 255)  # Green=above, Orange=below
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Record this frame's data
            perframe_rows.append((frame_count, count_in, count_out, tracked_count_this_frame))

            # Display total count HUD
            cv2.putText(frame, f"IN (Down): {count_in}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"OUT (Up): {count_out}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            # Save video
            video_writer.write(frame)

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames... IN: {count_in}, OUT: {count_out}")

            if max_frames and frame_count >= max_frames:
                break

    # Cleanup
    cap.release()
    video_writer.release()

    # Write per-frame CSV
    with open(perframe_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'in_count', 'out_count', 'tracked_count'])
        writer.writerows(perframe_rows)

    # Write summary CSV
    with open(out_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Count"])
        writer.writerow(["Total IN (Down)", count_in])
        writer.writerow(["Total OUT (Up)", count_out])
        writer.writerow(["Total Frames Processed", frame_count])

    print("\n" + "=" * 55)
    print("  Step 3 Complete!")
    print(f"  Final Count - IN: {count_in} | OUT: {count_out}")
    print(f"  Output Video: {out_video_path}")
    print("=" * 55)


def main():
    video_path = get_video_path()
    run_counting(video_path)

if __name__ == "__main__":
    main()
