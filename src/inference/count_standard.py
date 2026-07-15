"""
Step 3: Line Counting (Production-Ready)
===================================================
Purpose:
    Track pedestrian heads using ByteTrack and count how many cross a defined
    diagonal line using the robust `supervision` library.
"""

import sys
import os
import csv
import argparse
import cv2
from pathlib import Path
from ultralytics import YOLO
import supervision as sv

# Dynamically resolve project root and add to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import InferenceConfig, setup_parser
from src.utils.video_handler import VideoHandler


def run_counting(video_path: str, model_path: str, output_dir: str):
    """
    Run the main counting pipeline on the specified video using supervision.
    """
    config = InferenceConfig()
    os.makedirs(output_dir, exist_ok=True)
    
    out_video_path = os.path.join(output_dir, "counting_output.mp4")
    out_csv_path = os.path.join(output_dir, "summary.csv")
    perframe_csv = os.path.join(output_dir, "counts_per_frame.csv")
    detections_txt = os.path.join(output_dir, "detections_per_frame.txt")
    
    video = VideoHandler(video_path, out_video_path)
    model = YOLO(model_path)
    
    # Initialize Supervision tools
    tracker = sv.ByteTrack(
        track_activation_threshold=0.4,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8
    )
    line_zone = sv.LineZone(start=config.LINE_START, end=config.LINE_END)
    
    # Annotators
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=2,
        text_thickness=1,
        text_scale=0.5
    )
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    print(f"[INFO] Starting tracking... Press 'q' on the window to exit early.")
    print(f"[INFO] Output saved to {output_dir}\n")

    frame_count = 0
    total_in = 0
    total_out = 0

    # Open CSV files safely inside a context manager
    with open(detections_txt, 'w') as det_f, open(perframe_csv, 'w', newline='') as csv_f:
        det_f.write('frame,track_id,class_id,x1,y1,x2,y2,conf\n')
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(['frame', 'in_count', 'out_count', 'tracked_count'])

        try:
            for frame in video.get_frames():
                frame_count += 1
                
                # 1. Run YOLO detection
                results = model(frame, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]
                
                # 2. Convert to Supervision Detections
                detections = sv.Detections.from_ultralytics(results)
                detections = tracker.update_with_detections(detections)
                
                # 3. Update Line Zone for counting
                line_zone.trigger(detections=detections)
                total_in = line_zone.in_count
                total_out = line_zone.out_count
                
                # 4. Annotation
                labels = [f"ID: {tracker_id}" for tracker_id in detections.tracker_id] if detections.tracker_id is not None else []
                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
                
                # HUD
                tracked_count = len(detections)
                cv2.putText(annotated_frame, f"IN: {total_in}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_HUD_IN, 2)
                cv2.putText(annotated_frame, f"OUT: {total_out}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_HUD_OUT, 2)
                cv2.putText(annotated_frame, f"In Frame: {tracked_count}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_HUD_INFO, 2)
                
                # 5. Log Data continuously
                if detections.tracker_id is not None:
                    for i in range(len(detections)):
                        x1, y1, x2, y2 = detections.xyxy[i]
                        track_id = int(detections.tracker_id[i])
                        conf = float(detections.confidence[i])
                        cls = int(detections.class_id[i])
                        det_f.write(f"{frame_count},{track_id},{cls},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{conf:.2f}\n")
                        
                csv_writer.writerow([frame_count, total_in, total_out, tracked_count])
                
                # Flush occasionally to avoid data loss
                if frame_count % 50 == 0:
                    det_f.flush()
                    csv_f.flush()

                video.write_frame(annotated_frame)
                
                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames... IN: {total_in}, OUT: {total_out}")

        except KeyboardInterrupt:
            print("\n[WARNING] Process interrupted by user. Saving data safely...")
        finally:
            video.release()

    # Save Final Summary CSV
    with open(out_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Count"])
        writer.writerow(["Total IN (Down)", total_in])
        writer.writerow(["Total OUT (Up)", total_out])
        writer.writerow(["Total Frames Processed", frame_count])

    print("\n" + "=" * 55)
    print("  Tracking and Counting Complete!")
    print(f"  Final Count - IN: {total_in} | OUT: {total_out}")
    print(f"  Output Video: {out_video_path}")
    print("=" * 55)


if __name__ == "__main__":
    config = InferenceConfig()
    parser = setup_parser(
        default_video=config.DEFAULT_VIDEO_PATH,
        default_model=config.MODEL_V1_PATH,
        default_output=config.OUTPUT_STANDARD_DIR,
        description="Standard YOLO + ByteTrack line counting"
    )
    args = parser.parse_args()
    
    run_counting(args.video, args.model, args.output)
