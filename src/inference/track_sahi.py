import os
import cv2
import csv
import argparse
import numpy as np
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# ============================================================
# Counting Line Configuration (Diagonal Line Support)
# MUST match archive/app.py and archive/evaluate.py
# ============================================================
LINE_START = (0, 250)       # Left endpoint  (x, y)
LINE_END   = (1920, 550)    # Right endpoint (x, y)
BUFFER = 15                 # Perpendicular buffer distance (pixels) to prevent flickering


def cross_product_sign(line_start, line_end, point):
    """
    Determine which side of the line a point lies on using the cross product.
    Returns:
        > 0 : point is BELOW the line
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
    Used to determine if a point is inside the buffer zone.
    """
    ax, ay = line_start
    bx, by = line_end
    px, py = point
    line_len = ((bx - ax)**2 + (by - ay)**2) ** 0.5
    if line_len == 0:
        return float('inf')
    return abs((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / line_len


def track_video(video_path, model_path, output_dir):
    print(f"[INFO] Initializing SAHI detection model with weights: {model_path}")
    
    # Load YOLO model via SAHI
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8', 
            model_path=model_path,
            confidence_threshold=0.15,
            device="cuda:0"
        )
        print("[INFO] Model successfully loaded onto GPU.")
    except Exception as e:
        print(f"[WARNING] Failed to load on GPU, falling back to CPU. Error: {e}")
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8', 
            model_path=model_path,
            confidence_threshold=0.15,
            device="cpu"
        )

    # Initialize ByteTrack from supervision
    tracker = sv.ByteTrack()

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    # Setup Output Directories
    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, "tracking_sahi_output.mp4")
    out_csv_path = os.path.join(output_dir, "summary.csv")
    perframe_csv = os.path.join(output_dir, "counts_per_frame.csv")
    detections_txt = os.path.join(output_dir, "detections_per_frame.txt")

    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    video_writer = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f"[INFO] Starting frame-by-frame video processing: {video_path}")
    print(f"[INFO] Counting Line: {LINE_START} -> {LINE_END}, Buffer = ±{BUFFER}px")

    # Tracking Variables
    track_history = {}  # {track_id: last_cross_product_sign_outside_buffer}
    count_in = 0        # Crossed from ABOVE to BELOW the line
    count_out = 0       # Crossed from BELOW to ABOVE the line
    frame_count = 0
    perframe_rows = []

    # Open detections log file safely
    with open(detections_txt, 'w') as det_f:
        det_f.write('frame,track_id,class_id,x1,y1,x2,y2,conf\n')

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            tracked_count_this_frame = 0

            # SAHI expects RGB images, but cv2 uses BGR frames
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 1. SAHI Sliced Prediction: Divide frame into 640x640 grids with overlap
            result = get_sliced_prediction(
                rgb_frame,
                detection_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.2,
                overlap_width_ratio=0.2
            )
            
            # 2. Convert SAHI results to sv.Detections manually (compatible with all sv versions)
            raw_xyxy, raw_conf, raw_cls = [], [], []
            for obj in result.object_prediction_list:
                raw_xyxy.append([obj.bbox.minx, obj.bbox.miny, obj.bbox.maxx, obj.bbox.maxy])
                raw_conf.append(obj.score.value)
                raw_cls.append(obj.category.id)
                
            if len(raw_xyxy) > 0:
                detections = sv.Detections(
                    xyxy=np.array(raw_xyxy),
                    confidence=np.array(raw_conf),
                    class_id=np.array(raw_cls).astype(int)
                )
            else:
                detections = sv.Detections.empty()
                
            detections = tracker.update_with_detections(detections)
            
            # 3. Draw diagonal counting line
            cv2.line(frame, LINE_START, LINE_END, (0, 255, 255), 3) # Main Yellow Line

            # 4. Extract and process tracking boxes for precise counting
            if detections.tracker_id is not None and len(detections) > 0:
                for i in range(len(detections)):
                    x1, y1, x2, y2 = detections.xyxy[i]
                    track_id = int(detections.tracker_id[i])
                    confidence = float(detections.confidence[i]) if detections.confidence is not None else 0.0
                    cls = int(detections.class_id[i]) if detections.class_id is not None else 0
                    xc, yc = int((x1 + x2) / 2), int((y1 + y2) / 2)  # CENTER of the head

                    # Log detections
                    det_f.write(f"{frame_count},{track_id},{cls},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{confidence:.2f}\n")
                    tracked_count_this_frame += 1

                    # --- PRECISE DIAGONAL LINE COUNTING LOGIC ---
                    center = (xc, yc)
                    dist = point_to_line_distance(LINE_START, LINE_END, center)
                    sign = cross_product_sign(LINE_START, LINE_END, center)
                    is_outside_buffer = dist > BUFFER

                    if track_id in track_history and is_outside_buffer:
                        old_sign = track_history[track_id]

                        # Crossed from ABOVE to BELOW (negative -> positive)
                        if old_sign < 0 and sign > 0:
                            count_in += 1

                        # Crossed from BELOW to ABOVE (positive -> negative)
                        elif old_sign > 0 and sign < 0:
                            count_out += 1

                    # Update history only when outside buffer (or first appearance)
                    if track_id not in track_history or is_outside_buffer:
                        track_history[track_id] = sign

                    # Draw BBox, color-coded by side: Green=above, Orange=below
                    color = (0, 255, 0) if sign < 0 else (0, 165, 255)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Record frame statistics
            perframe_rows.append((frame_count, count_in, count_out, tracked_count_this_frame))

            # HUD Display
            cv2.putText(frame, f"IN (Down): {count_in}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"OUT (Up): {count_out}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            video_writer.write(frame)
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames... IN: {count_in}, OUT: {count_out}")

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
    print("  Tracking and Counting Complete!")
    print(f"  Final Count - IN: {count_in} | OUT: {count_out}")
    print(f"  Outputs saved to: {output_dir}")
    print("=" * 55)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SAHI + ByteTrack inference with diagonal line counting")
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    default_video = os.path.join(PROJECT_ROOT, "data/raw/TownCentre_1min.mp4")
    default_model = os.path.join(PROJECT_ROOT, "models/trained/HeadDetect_v1.pt")
    default_output = os.path.join(PROJECT_ROOT, "outputs/count")
    
    parser.add_argument("--video", type=str, default=default_video, help="Path to input video")
    parser.add_argument("--model", type=str, default=default_model, help="Path to model weights (.pt)")
    parser.add_argument("--output", type=str, default=default_output, help="Directory to save outputs")
    
    args = parser.parse_args()
    
    track_video(args.video, args.model, args.output)
