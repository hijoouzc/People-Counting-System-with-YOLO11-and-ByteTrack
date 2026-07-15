"""
Step 4: Line Counting (Optimized for SAHI - Production Ready)
===================================================
Purpose:
    Use Slicing Aided Hyper Inference (SAHI) with ByteTrack to detect and track
    small pedestrian heads, counting them as they cross a diagonal line.
"""

import sys
import os
import cv2
import csv
from pathlib import Path
import numpy as np
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

# Dynamically resolve project root and add to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import InferenceConfig, setup_parser
from src.utils.video_handler import VideoHandler


def init_sahi_model(model_path: str, config: InferenceConfig):
    """Initialize the SAHI model on GPU, fallback to CPU on failure."""
    print(f"[INFO] Initializing SAHI detection model with weights: {model_path}")
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8', 
            model_path=model_path,
            confidence_threshold=config.CONFIDENCE_THRESHOLD,
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
    return detection_model


def track_video(video_path: str, model_path: str, output_dir: str):
    """
    Run the SAHI counting pipeline on the specified video.
    """
    config = InferenceConfig()
    os.makedirs(output_dir, exist_ok=True)
    
    out_video_path = os.path.join(output_dir, "tracking_sahi_output.mp4")
    out_csv_path = os.path.join(output_dir, "sahi_summary.csv")
    perframe_csv = os.path.join(output_dir, "sahi_counts_per_frame.csv")
    detections_txt = os.path.join(output_dir, "sahi_detections_per_frame.txt")
    
    video = VideoHandler(video_path, out_video_path)
    detection_model = init_sahi_model(model_path, config)
    
    # Initialize Supervision tools
    tracker = sv.ByteTrack(
        track_activation_threshold=0.4,
        lost_track_buffer=60,
        minimum_matching_threshold=0.8
    )
    line_zone = sv.LineZone(start=config.SAHI_LINE_START, end=config.SAHI_LINE_END)
    
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    print(f"[INFO] Starting frame-by-frame video processing: {video_path}")
    print(f"[INFO] Output saved to {output_dir}\n")

    frame_count = 0
    total_in = 0
    total_out = 0

    with open(detections_txt, 'w') as det_f, open(perframe_csv, 'w', newline='') as csv_f:
        det_f.write('frame,track_id,class_id,x1,y1,x2,y2,conf\n')
        csv_writer = csv.writer(csv_f)
        csv_writer.writerow(['frame', 'in_count', 'out_count', 'tracked_count'])

        try:
            for frame in video.get_frames():
                frame_count += 1
                
                # 1. Prepare image and get SAHI prediction
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = get_sliced_prediction(
                    rgb_frame,
                    detection_model,
                    slice_height=640,
                    slice_width=640,
                    overlap_height_ratio=0.2,
                    overlap_width_ratio=0.2
                )
                
                # 2. Format results for supervision Tracker
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
                    
                # 3. Update Tracker and Line Zone
                detections = tracker.update_with_detections(detections)
                line_zone.trigger(detections=detections)
                
                total_in = line_zone.in_count
                total_out = line_zone.out_count

                # 4. Annotate
                labels = [f"ID: {tracker_id}" for tracker_id in detections.tracker_id] if detections.tracker_id is not None else []
                annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
                
                # HUD
                tracked_count = len(detections)
                cv2.putText(annotated_frame, f"IN: {total_in}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_HUD_IN, 2)
                cv2.putText(annotated_frame, f"OUT: {total_out}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_HUD_OUT, 2)
                cv2.putText(annotated_frame, f"In Frame: {tracked_count}", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, config.COLOR_HUD_INFO, 2)
                
                # 5. Continuous Logging
                if detections.tracker_id is not None:
                    for i in range(len(detections)):
                        x1, y1, x2, y2 = detections.xyxy[i]
                        track_id = int(detections.tracker_id[i])
                        conf = float(detections.confidence[i])
                        cls = int(detections.class_id[i])
                        det_f.write(f"{frame_count},{track_id},{cls},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{conf:.2f}\n")
                        
                csv_writer.writerow([frame_count, total_in, total_out, tracked_count])
                
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
    print(f"  Outputs saved to: {output_dir}")
    print("=" * 55)


if __name__ == '__main__':
    config = InferenceConfig()
    parser = setup_parser(
        default_video=config.DEFAULT_SAHI_VIDEO,
        default_model=config.MODEL_V1_PATH,
        default_output=config.OUTPUT_SAHI_DIR,
        description="SAHI + ByteTrack inference with supervision line counting"
    )
    args = parser.parse_args()
    
    track_video(args.video, args.model, args.output)
