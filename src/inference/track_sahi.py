import os
import cv2
import argparse
import supervision as sv
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def track_video(video_path, model_path, output_path):
    print(f"[INFO] Initializing SAHI detection model with weights: {model_path}")
    
    # Load YOLO model via SAHI (Using yolov8 API which natively supports yolo11 models)
    try:
        detection_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8', 
            model_path=model_path,
            confidence_threshold=0.15,
            device="cuda:0" # Attempt to use GPU for fast inference
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

    # Initialize ByteTrack algorithm from the supervision library
    tracker = sv.ByteTrack()
    
    # Configure LineZone for counting pedestrians
    # Default coordinates tailored for TownCentreXVID.mp4
    start_pt = sv.Point(x=0, y=500)
    end_pt = sv.Point(x=1920, y=500)
    counting_line = sv.LineZone(start=start_pt, end=end_pt)
    
    # Visualization annotation tools
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    line_annotator = sv.LineZoneAnnotator()

    if not os.path.exists(video_path):
        print(f"[ERROR] Video file not found: {video_path}")
        return

    print(f"[INFO] Starting frame-by-frame video processing for: {video_path}")
    
    def process_frame(frame):
        # SAHI expects RGB images, but cv2/supervision uses BGR frames
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. SAHI Sliced Prediction: Divide 1080p frame into 640x640 grids with 20% overlap
        result = get_sliced_prediction(
            rgb_frame,
            detection_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2
        )
        
        # 2. Convert SAHI prediction results to Supervision format
        detections = sv.Detections.from_sahi(result)
        
        # 3. Pass Bounding Boxes into ByteTrack to maintain identity tracking
        detections = tracker.update_with_detections(detections)
        
        # 4. Trigger counting logic when tracked objects cross the predefined line
        counting_line.trigger(detections=detections)
        
        # 5. Visual Rendering: Draw boxes, labels, and the counting line
        labels = [
            f"#{tracker_id} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id, _
            in detections
        ]
        
        annotated_frame = frame.copy()
        
        # Robust annotation calls to support various supervision versions
        try:
            annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
        except TypeError:
            annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections=detections, labels=labels)
            
        try:
            annotated_frame = line_annotator.annotate(annotated_frame, line_counter=counting_line)
        except TypeError:
            annotated_frame = line_annotator.annotate(scene=annotated_frame, line_counter=counting_line)

        return annotated_frame

    # Process the entire video memory-efficiently (frame-by-frame processing)
    sv.process_video(
        source_path=video_path,
        target_path=output_path,
        callback=process_frame
    )
    
    print(f"\n[SUCCESS] Processing complete! Tracked video saved at: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Inference tracking pipeline using SAHI and ByteTrack")
    
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Define default paths based on the project architecture
    default_video = os.path.join(PROJECT_ROOT, "data/raw/TownCentreXVID.mp4")
    default_model = os.path.join(PROJECT_ROOT, "models/stage2_crowdhuman/run_1/weights/best.pt")
    default_output = os.path.join(PROJECT_ROOT, "outputs/tracking_sahi_output.mp4")
    
    parser.add_argument("--video", type=str, default=default_video, help="Path to input video")
    parser.add_argument("--model", type=str, default=default_model, help="Path to model weights (.pt)")
    parser.add_argument("--output", type=str, default=default_output, help="Path to save output video")
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    track_video(args.video, args.model, args.output)
