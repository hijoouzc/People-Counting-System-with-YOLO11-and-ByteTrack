"""
Step 1: Basic Video Detection Test
===================================
Purpose:
    Verify that YOLO can accurately detect pedestrians in the video.
    No tracking or counting logic — detection only.
"""

import sys
import os
from pathlib import Path
from ultralytics import YOLO

# Dynamically resolve project root and add to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import InferenceConfig, setup_parser
from src.utils.video_handler import VideoHandler


def run_detection(video_path: str, model_path: str, output_dir: str) -> None:
    """
    Run YOLO person detection on the given video.
    """
    config = InferenceConfig()
    os.makedirs(output_dir, exist_ok=True)
    out_video_path = os.path.join(output_dir, "detect_output.mp4")
    
    video = VideoHandler(video_path, out_video_path)
    model = YOLO(model_path)

    print("[INFO] Starting detection... Press 'q' on the video window to exit early.")
    print(f"[INFO] Output saved to {output_dir}\n")

    frame_count = 0
    try:
        for frame in video.get_frames():
            frame_count += 1
            
            # Predict and plot directly using ultralytics built-in method for simple detection test
            results = model.predict(
                source=frame,
                classes=[config.PERSON_CLASS_ID],
                conf=config.CONFIDENCE_THRESHOLD,
                imgsz=1024,
                verbose=False
            )[0]
            
            annotated_frame = results.plot()
            video.write_frame(annotated_frame)
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames...")
                
    except KeyboardInterrupt:
        print("\n[WARNING] Process interrupted by user.")
    finally:
        video.release()

    print("\n" + "=" * 55)
    print("  Step 1 Complete!")
    print(f"  Output saved to: {out_video_path}")
    print("=" * 55)


if __name__ == "__main__":
    config = InferenceConfig()
    parser = setup_parser(
        default_video=config.DEFAULT_VIDEO_PATH,
        default_model=config.MODEL_V1_PATH,
        default_output=os.path.join(PROJECT_ROOT, "outputs", "predict"),
        description="Basic Video Detection Test"
    )
    args = parser.parse_args()
    
    run_detection(args.video, args.model, args.output)
