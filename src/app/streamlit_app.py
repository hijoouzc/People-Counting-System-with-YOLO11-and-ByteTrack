import sys
from pathlib import Path
import cv2
import streamlit as st
import pandas as pd
import supervision as sv
from ultralytics import YOLO

# Resolve project root dynamically and add to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.config import InferenceConfig, ProjectPaths

# Configure page
st.set_page_config(page_title=  "Counting System", layout="wide")

# Hide Streamlit UI elements for a cleaner look
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Title
st.title("Counting System")

config = InferenceConfig()

# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.header("Configuration")
conf_thresh = st.sidebar.slider("Confidence", 0.0, 1.0, config.CONFIDENCE_THRESHOLD, 0.05)
track_thresh = st.sidebar.slider("Track Activation", 0.0, 1.0, 0.4, 0.05)
track_buffer = st.sidebar.slider("Track Buffer", 10, 150, 60, 10)

st.sidebar.markdown("---")
st.sidebar.header("Line Coordinates")
col_x, col_y = st.sidebar.columns(2)
line_start_x = int(col_x.number_input("Start X", value=config.LINE_START.x, step=10))
line_start_y = int(col_y.number_input("Start Y", value=config.LINE_START.y, step=10))
line_end_x = int(col_x.number_input("End X", value=config.LINE_END.x, step=10))
line_end_y = int(col_y.number_input("End Y", value=config.LINE_END.y, step=10))

st.sidebar.markdown("---")
st.sidebar.header("Input")
model_type = st.sidebar.selectbox("Model", ["Version 1", "Version 2"], index=1)
model_path = config.MODEL_V1_PATH if model_type == "Version 1" else config.MODEL_V2_PATH

video_file = st.sidebar.file_uploader("Custom Video", type=['mp4', 'avi'])
video_path = config.DEFAULT_VIDEO_PATH

if video_file is not None:
    # Save uploaded file temporarily to data/raw
    temp_path = ProjectPaths.RAW_DATA_DIR / "temp_uploaded.mp4"
    with open(temp_path, "wb") as f:
        f.write(video_file.getbuffer())
    video_path = str(temp_path)

start_btn = st.sidebar.button("Start Inference", type="primary")

# ==========================================
# MAIN PANEL
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Live Video")
    video_placeholder = st.empty()

with col2:
    st.subheader("Analytics")
    
    # Beautiful iOS-like metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    in_metric = metric_col1.empty()
    out_metric = metric_col2.empty()
    current_metric = metric_col3.empty()
    
    st.markdown("---")
    chart_placeholder = st.empty()

# ==========================================
# PREVIEW
# ==========================================
if not start_btn:
    cap_prev = cv2.VideoCapture(video_path)
    if cap_prev.isOpened():
        ret, frame = cap_prev.read()
        if ret:
            start_pt = sv.Point(line_start_x, line_start_y)
            end_pt = sv.Point(line_end_x, line_end_y)
            line_zone_prev = sv.LineZone(start=start_pt, end=end_pt)
            annotator_prev = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
            
            annotated_prev = annotator_prev.annotate(frame.copy(), line_counter=line_zone_prev)
            display_img = cv2.resize(annotated_prev, (1024, 576))
            video_placeholder.image(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB), channels="RGB")
        cap_prev.release()

# ==========================================
# INFERENCE LOOP
# ==========================================
if start_btn:
    st.sidebar.success("Loading Model...")
    model = YOLO(model_path)
    
    st.sidebar.info("Initializing Tracker...")
    tracker = sv.ByteTrack(
        track_activation_threshold=track_thresh,
        lost_track_buffer=track_buffer,
        minimum_matching_threshold=0.8
    )
    start_pt = sv.Point(line_start_x, line_start_y)
    end_pt = sv.Point(line_end_x, line_end_y)
    line_zone = sv.LineZone(start=start_pt, end=end_pt)
    line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_thickness=1, text_scale=0.5)
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video source: {video_path}")
    else:
        # Prepare data structures for charting
        frames_list = []
        in_counts = []
        out_counts = []
        
        frame_idx = 0
        
        # We use a progress text in the sidebar
        status_text = st.sidebar.empty()
        status_text.text("Processing...")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_idx += 1
            
            # 1. Detection
            results = model(frame, conf=conf_thresh, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # 2. Tracking
            detections = tracker.update_with_detections(detections)
            
            # 3. Counting
            line_zone.trigger(detections=detections)
            
            # 4. Annotation
            labels = [f"ID: {id}" for id in detections.tracker_id] if detections.tracker_id is not None else []
            annotated = box_annotator.annotate(scene=frame.copy(), detections=detections)
            annotated = label_annotator.annotate(scene=annotated, detections=detections, labels=labels)
            annotated = line_zone_annotator.annotate(annotated, line_counter=line_zone)
            
            # Streamlit requires RGB for images
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            
            # 5. Render to UI (OPTIMIZATION: Render every 2 frames and resize to reduce WebSocket payload)
            if frame_idx % 2 == 0:
                display_img = cv2.resize(annotated_rgb, (1024, 576))
                video_placeholder.image(display_img, channels="RGB")
                
                total_in = line_zone.in_count
                total_out = line_zone.out_count
                people_in_frame = len(detections)
                
                # Render Clean Metrics
                in_metric.metric("In", total_in)
                out_metric.metric("Out", total_out)
                current_metric.metric("Current", people_in_frame)
            
            # Analytics recording
            frames_list.append(frame_idx)
            in_counts.append(line_zone.in_count)
            out_counts.append(line_zone.out_count)
            
        cap.release()
        status_text.text("Completed")
        st.success("Finished!")
        
        # Render final chart after inference completes to save performance
        if len(frames_list) > 0:
            df = pd.DataFrame({
                'Frame': frames_list,
                'In': in_counts,
                'Out': out_counts
            }).set_index('Frame')
            chart_placeholder.line_chart(df, color=["#0000FF", "#FF0000"])
