"""
Evaluation Script for Person Counting
=====================================
Purpose:
    Read the Oxford Town Centre ground truth data (.top) and simulate diagonal
    line counting using the same cross-product logic as app.py and track_sahi.py.
    Then compare ground truth counts against AI prediction results from summary.csv
    and sahi_summary.csv, automatically restricting the ground truth to the exact
    frame range processed by the AI.

Usage:
    python evaluate.py
"""

import os
import csv
from collections import defaultdict

# ============================================================
# Configuration — MUST match app.py and track_sahi.py exactly
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

GROUND_TRUTH_FILE = os.path.join(PROJECT_ROOT, "data/raw/TownCentre-groundtruth.top")
AI_SUMMARY_FILE = os.path.join(PROJECT_ROOT, "outputs/count/summary.csv")
SAHI_SUMMARY_FILE = os.path.join(PROJECT_ROOT, "outputs/count/sahi_summary.csv")
AI_DETECTIONS_FILE = os.path.join(PROJECT_ROOT, "outputs/count/detections_per_frame.txt")
SAHI_DETECTIONS_FILE = os.path.join(PROJECT_ROOT, "outputs/count/sahi_detections_per_frame.txt")

# Counting line endpoints — must mirror app.py
LINE_START = (0, 250)       # Left endpoint  (x, y)
LINE_END   = (1920, 550)    # Right endpoint (x, y)
BUFFER = 15                 # Perpendicular buffer distance (pixels)


def cross_product_sign(line_start, line_end, point):
    """
    Determine which side of the line a point lies on using the cross product.
    Returns:
        > 0 : point is BELOW the line
        < 0 : point is ABOVE the line
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
    line_len = ((bx - ax)**2 + (by - ay)**2) ** 0.5
    if line_len == 0:
        return float('inf')
    return abs((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / line_len


def read_ai_summary(filepath):
    ai_in, ai_out, max_frames = 0, 0, 999999
    exists = os.path.exists(filepath)
    if exists:
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    if row[0] == "Total IN (Down)":
                        ai_in = int(row[1])
                    elif row[0] == "Total OUT (Up)":
                        ai_out = int(row[1])
                    elif row[0] == "Total Frames Processed":
                        max_frames = int(row[1])
    return exists, ai_in, ai_out, max_frames


def count_unique_ids(filepath, max_frames):
    unique_ids = set()
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # Skip header
            for row in reader:
                if len(row) >= 2:
                    try:
                        frame = int(row[0])
                        track_id = int(row[1])
                        if frame <= max_frames:
                            unique_ids.add(track_id)
                    except ValueError:
                        pass
    return len(unique_ids)


def simulate_gt(trajectories, max_frames):
    gt_in = 0
    gt_out = 0
    last_outside_sign = {}  # {person_id: last cross_product_sign outside buffer}

    for person_id, full_path in trajectories.items():
        # Only evaluate frames up to max_frames
        path = [(f, cx, cy) for f, cx, cy in full_path if f <= max_frames]
        if len(path) < 2:
            continue
            
        counted = False
        for _, cx, cy in path:
            center = (cx, cy)
            dist = point_to_line_distance(LINE_START, LINE_END, center)
            sign = cross_product_sign(LINE_START, LINE_END, center)
            is_outside_buffer = dist > BUFFER

            if person_id in last_outside_sign and not counted and is_outside_buffer:
                old_sign = last_outside_sign[person_id]

                # Crossed from ABOVE to BELOW (negative -> positive)
                if old_sign < 0 and sign > 0:
                    gt_in += 1
                    counted = True

                # Crossed from BELOW to ABOVE (positive -> negative)
                elif old_sign > 0 and sign < 0:
                    gt_out += 1
                    counted = True

            # Update saved position only when outside the buffer zone
            if person_id not in last_outside_sign or is_outside_buffer:
                last_outside_sign[person_id] = sign
                
    return gt_in, gt_out


def evaluate():
    os.chdir(PROJECT_ROOT)

    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"[ERROR] Ground truth file not found: {GROUND_TRUTH_FILE}")
        return

    # ============================================================
    # Phase 1: Read AI Prediction Results & Get Frame Limit
    # ============================================================
    ai_exists, ai_in, ai_out, ai_max_frames = read_ai_summary(AI_SUMMARY_FILE)
    sahi_exists, sahi_in, sahi_out, sahi_max_frames = read_ai_summary(SAHI_SUMMARY_FILE)

    max_eval_frames = 999999
    if ai_exists or sahi_exists:
        max_eval_frames = max([f for f, exists in [(ai_max_frames, ai_exists), (sahi_max_frames, sahi_exists)] if exists])
        print(f"[INFO] Found AI summaries. Max frames processed among them: {max_eval_frames}")
    else:
        print("[WARNING] AI summary files not found. Evaluating Ground Truth on ALL frames.")

    # ============================================================
    # Phase 2: Parse Ground Truth Trajectories (Within Global Limit)
    # ============================================================
    trajectories = defaultdict(list)

    print(f"[INFO] Reading ground truth data from {GROUND_TRUTH_FILE}...")
    with open(GROUND_TRUTH_FILE, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 12:
                continue

            try:
                person_id = int(row[0])
                frame_num = int(row[1])

                if frame_num > max_eval_frames:
                    continue

                head_valid = int(row[2])
                body_valid = int(row[3])

                if head_valid == 1:
                    head_left   = float(row[4])
                    head_top    = float(row[5])
                    head_right  = float(row[6])
                    head_bottom = float(row[7])
                    cx = (head_left + head_right) / 2.0
                    cy = (head_top + head_bottom) / 2.0
                    trajectories[person_id].append((frame_num, cx, cy))

                elif body_valid == 1:
                    body_left   = float(row[8])
                    body_top    = float(row[9])
                    body_right  = float(row[10])
                    body_bottom = float(row[11])
                    cx = (body_left + body_right) / 2.0
                    cy = (body_top + body_bottom) / 2.0
                    trajectories[person_id].append((frame_num, cx, cy))

            except ValueError:
                continue

    # Sort each trajectory chronologically
    for person_id in trajectories:
        trajectories[person_id].sort(key=lambda x: x[0])

    print(f"[INFO] Processed trajectories for {len(trajectories)} distinct people.")

    # ============================================================
    # Phase 3 & 4: Simulate & Evaluate Accuracy
    # ============================================================
    
    def evaluate_results(name, pred_in, pred_out, max_frames, detections_file):
        gt_in, gt_out = simulate_gt(trajectories, max_frames)
        
        pred_total = count_unique_ids(detections_file, max_frames)
        gt_total = len([pid for pid, path in trajectories.items() if any(f <= max_frames for f, cx, cy in path)])
        
        print("\n" + "=" * 55)
        print(f"  {name.upper()} EVALUATION")
        print("=" * 55)
        print(f"  Line: {LINE_START} -> {LINE_END}, Buffer = ±{BUFFER}px")
        print(f"  Evaluation Frame Limit: 1 to {max_frames}")
        print("-" * 55)
        print(f"  Ground Truth - IN: {gt_in}, OUT: {gt_out} | TOTAL UNIQUE: {gt_total}")
        print(f"  AI Predicted - IN: {pred_in}, OUT: {pred_out} | TOTAL UNIQUE: {pred_total}")
        print("-" * 55)
        
        err_in = abs(gt_in - pred_in)
        acc_in = max(0.0, 100.0 - (err_in / max(1, gt_in)) * 100.0) if gt_in > 0 else 0.0

        err_out = abs(gt_out - pred_out)
        acc_out = max(0.0, 100.0 - (err_out / max(1, gt_out)) * 100.0) if gt_out > 0 else 0.0

        err_total = abs(gt_total - pred_total)
        acc_total = max(0.0, 100.0 - (err_total / max(1, gt_total)) * 100.0) if gt_total > 0 else 0.0

        print(f"  IN Accuracy:     {acc_in:.2f}% (Error: {err_in} persons)")
        print(f"  OUT Accuracy:    {acc_out:.2f}% (Error: {err_out} persons)")
        print(f"  Total Unique ID: {acc_total:.2f}% (Error: {err_total} persons)")
        print(f"  Overall Acc:     {(acc_in + acc_out + acc_total) / 3:.2f}%")
        print("=" * 55)

    if not ai_exists and not sahi_exists:
        gt_in, gt_out = simulate_gt(trajectories, 999999)
        print("\n" + "=" * 55)
        print("  GROUND TRUTH COUNTS")
        print(f"  Line: {LINE_START} -> {LINE_END}, Buffer = ±{BUFFER}px")
        print(f"  Evaluation Frame Limit: ALL FRAMES")
        print("=" * 55)
        print(f"  Total IN  (Above -> Below): {gt_in}")
        print(f"  Total OUT (Below -> Above): {gt_out}")
        print("=" * 55)
        print("Run archive/app.py or src/inference/track_sahi.py to generate AI results.")
    
    if ai_exists:
        evaluate_results("Standard YOLO (app.py)", ai_in, ai_out, ai_max_frames, AI_DETECTIONS_FILE)
    else:
        print(f"\n[WARNING] Standard AI summary file not found at {AI_SUMMARY_FILE}.")

    if sahi_exists:
        evaluate_results("SAHI + YOLO (track_sahi.py)", sahi_in, sahi_out, sahi_max_frames, SAHI_DETECTIONS_FILE)
    else:
        print(f"\n[WARNING] SAHI summary file not found at {SAHI_SUMMARY_FILE}.")


if __name__ == "__main__":
    evaluate()
