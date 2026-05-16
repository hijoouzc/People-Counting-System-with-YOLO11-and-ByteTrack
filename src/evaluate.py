"""
Evaluation Script for Person Counting
===================================
Purpose:
    Read the Oxford Town Centre ground truth data (.top) and simulate line counting
    at Y=600. Then compare the ground truth counts against the YOLO+ByteTrack results.

Usage:
    python evaluate_counting.py
"""

import os
import csv
from collections import defaultdict

# ============================================================
# Configuration (Paths are relative to project root)
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

GROUND_TRUTH_FILE = os.path.join(PROJECT_ROOT, "data/raw/TownCentre-groundtruth.top")
COUNTING_LINE_Y = 500
AI_SUMMARY_FILE = os.path.join(PROJECT_ROOT, "outputs/count/summary.csv")


def evaluate():
    # Ensure we are running from project root
    os.chdir(PROJECT_ROOT)
    
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"[ERROR] Ground truth file not found: {GROUND_TRUTH_FILE}")
        return

    # Dictionary to store trajectory of each person
    # Format: trajectories[person_id] = [(frame_num, cy), (frame_num, cy), ...]
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
                body_valid = int(row[3])
                
                # Only use valid body boxes
                if body_valid == 1:
                    body_top = float(row[9])
                    body_bottom = float(row[11])
                    
                    # Calculate centroid Y of the body
                    cy = (body_top + body_bottom) / 2.0
                    trajectories[person_id].append((frame_num, cy))
            except ValueError:
                continue

    # Sort trajectories by frame number to ensure chronological order
    for person_id in trajectories:
        trajectories[person_id].sort(key=lambda x: x[0])

    print(f"[INFO] Processed trajectories for {len(trajectories)} distinct people.")

    # ============================================================
    # Simulate Counting
    # Default Ultralytics Logic:
    # - Downward movement (Y increasing) = IN
    # - Upward movement (Y decreasing) = OUT
    # ============================================================
    gt_in = 0
    gt_out = 0
    counted_ids = set()

    for person_id, path in trajectories.items():
        if len(path) < 2:
            continue
            
        # Traverse the path to find the first crossing of Y=600
        for i in range(1, len(path)):
            prev_y = path[i-1][1]
            curr_y = path[i][1]
            
            # Crossing downwards (Top to Bottom -> Y increases) -> IN
            if prev_y < COUNTING_LINE_Y <= curr_y:
                if person_id not in counted_ids:
                    gt_in += 1
                    counted_ids.add(person_id)
                    break # Count exactly once per person
                    
            # Crossing upwards (Bottom to Top -> Y decreases) -> OUT
            elif prev_y > COUNTING_LINE_Y >= curr_y:
                if person_id not in counted_ids:
                    gt_out += 1
                    counted_ids.add(person_id)
                    break # Count exactly once per person

    print("\n" + "=" * 45)
    print("  GROUND TRUTH COUNTS")
    print("=" * 45)
    print(f"  Total IN  (Downward): {gt_in}")
    print(f"  Total OUT (Upward):   {gt_out}")
    print("=" * 45)

    # ============================================================
    # Compare with AI Results
    # ============================================================
    ai_in = 0
    ai_out = 0
    
    if os.path.exists(AI_SUMMARY_FILE):
        with open(AI_SUMMARY_FILE, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    if row[0] == "Total IN":
                        ai_in = int(row[1])
                    elif row[0] == "Total OUT":
                        ai_out = int(row[1])
                        
        print("\n" + "=" * 45)
        print("  AI PREDICTED COUNTS")
        print("=" * 45)
        print(f"  Total IN  (Downward): {ai_in}")
        print(f"  Total OUT (Upward):   {ai_out}")
        print("=" * 45)
        
        # Calculate Error
        err_in = abs(gt_in - ai_in)
        acc_in = max(0.0, 100.0 - (err_in / max(1, gt_in)) * 100.0)
        
        err_out = abs(gt_out - ai_out)
        acc_out = max(0.0, 100.0 - (err_out / max(1, gt_out)) * 100.0)
        
        print("\n" + "=" * 45)
        print("  ACCURACY EVALUATION")
        print("=" * 45)
        print(f"  IN Accuracy:  {acc_in:.2f}% (Error: {err_in} persons)")
        print(f"  OUT Accuracy: {acc_out:.2f}% (Error: {err_out} persons)")
        print(f"  Overall Acc:  {(acc_in + acc_out) / 2:.2f}%")
        print("=" * 45)
    else:
        print(f"\n[WARNING] AI summary file not found at {AI_SUMMARY_FILE}.")
        print("Run test_counting.py first to generate AI results.")

if __name__ == "__main__":
    evaluate()
