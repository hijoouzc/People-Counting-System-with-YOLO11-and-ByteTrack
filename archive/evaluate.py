"""
Evaluation Script for Person Counting
=====================================
Purpose:
    Read the Oxford Town Centre ground truth data (.top) and simulate diagonal
    line counting using the same cross-product logic as app.py and track_sahi.py.
    Then compare ground truth counts against AI prediction results from summary.csv,
    automatically restricting the ground truth to the exact frame range processed by the AI.

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
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

GROUND_TRUTH_FILE = os.path.join(PROJECT_ROOT, "data/raw/TownCentre-groundtruth.top")
AI_SUMMARY_FILE = os.path.join(PROJECT_ROOT, "outputs/count/summary.csv")

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


def evaluate():
    os.chdir(PROJECT_ROOT)

    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"[ERROR] Ground truth file not found: {GROUND_TRUTH_FILE}")
        return

    # ============================================================
    # Phase 1: Read AI Prediction Results & Get Frame Limit
    # ============================================================
    ai_in = 0
    ai_out = 0
    max_eval_frames = 999999  # Default to unlimited if AI summary is not found
    ai_summary_exists = os.path.exists(AI_SUMMARY_FILE)

    if ai_summary_exists:
        with open(AI_SUMMARY_FILE, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    if row[0] == "Total IN (Down)":
                        ai_in = int(row[1])
                    elif row[0] == "Total OUT (Up)":
                        ai_out = int(row[1])
                    elif row[0] == "Total Frames Processed":
                        max_eval_frames = int(row[1])
        print(f"[INFO] AI processed {max_eval_frames} frames. Restricting Ground Truth evaluation to these frames.")
    else:
        print("[WARNING] AI summary file not found. Evaluating Ground Truth on ALL frames.")

    # ============================================================
    # Phase 2: Parse Ground Truth Trajectories (Within Frame Limit)
    # ============================================================
    # Format: trajectories[person_id] = [(frame, cx, cy), ...]
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

                # Skip ground truth records beyond what the AI processed
                if frame_num > max_eval_frames:
                    continue

                head_valid = int(row[2])
                body_valid = int(row[3])

                # Prefer HEAD bounding box (matches our Head Detection model)
                if head_valid == 1:
                    head_left   = float(row[4])
                    head_top    = float(row[5])
                    head_right  = float(row[6])
                    head_bottom = float(row[7])
                    cx = (head_left + head_right) / 2.0
                    cy = (head_top + head_bottom) / 2.0
                    trajectories[person_id].append((frame_num, cx, cy))

                # Fallback to BODY centroid if head is not valid
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
    # Phase 3: Simulate Diagonal Buffer-Zone Counting
    # ============================================================
    gt_in = 0
    gt_out = 0
    last_outside_sign = {}  # {person_id: last cross_product_sign outside buffer}

    for person_id, path in trajectories.items():
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

    print("\n" + "=" * 55)
    print("  GROUND TRUTH COUNTS")
    print(f"  Line: {LINE_START} -> {LINE_END}, Buffer = ±{BUFFER}px")
    print(f"  Evaluation Frame Limit: 1 to {max_eval_frames}")
    print("=" * 55)
    print(f"  Total IN  (Above -> Below): {gt_in}")
    print(f"  Total OUT (Below -> Above): {gt_out}")
    print("=" * 55)

    # ============================================================
    # Phase 4: Accuracy Evaluation
    # ============================================================
    if ai_summary_exists:
        print("\n" + "=" * 55)
        print("  AI PREDICTED COUNTS")
        print("=" * 55)
        print(f"  Total IN  (Above -> Below): {ai_in}")
        print(f"  Total OUT (Below -> Above): {ai_out}")
        print("=" * 55)

        # Calculate Accuracy
        err_in = abs(gt_in - ai_in)
        acc_in = max(0.0, 100.0 - (err_in / max(1, gt_in)) * 100.0)

        err_out = abs(gt_out - ai_out)
        acc_out = max(0.0, 100.0 - (err_out / max(1, gt_out)) * 100.0)

        print("\n" + "=" * 55)
        print("  ACCURACY EVALUATION")
        print("=" * 55)
        print(f"  IN Accuracy:  {acc_in:.2f}% (Error: {err_in} persons)")
        print(f"  OUT Accuracy: {acc_out:.2f}% (Error: {err_out} persons)")
        print(f"  Overall Acc:  {(acc_in + acc_out) / 2:.2f}%")
        print("=" * 55)
    else:
        print(f"\n[WARNING] AI summary file not found at {AI_SUMMARY_FILE}.")
        print("Run archive/app.py first to generate AI results.")


if __name__ == "__main__":
    evaluate()
