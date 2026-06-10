#!/usr/bin/env python3
"""
Run RTMPose-m (rtmlib/ONNX) on all 4 exercise test sets and classify using Angular DTW.
RTMPose-m is the OpenMMLab successor to HRNet; same COCO-17 keypoint format, no compilation needed.
Saves results to /tmp/hrnet_results.json.

Install dependencies before running:
    pip install rtmlib onnxruntime

Usage:
    python scripts/run_hrnet_all.py
"""
import glob, json, os
import numpy as np
import cv2
from fastdtw import fastdtw
from rtmlib import Body

DATASET = os.path.join(os.path.dirname(__file__), "..",
                       "An_upper_limb_stroke_rehabilitation_exercise_video")
EXERCISES = [
    ("1_Lifting_an_Object",   20.0),
    ("2_Extending_the_Elbow", 10.0),
    ("3_Lifting_the_Wrist",   15.0),
    ("4_Opening_the_Hand",    10.0),
]
SAMPLE      = 3      # use every Nth frame
FRAME_W     = 810.0  # normalisation divisor (pixel → relative coords)
SCORE_THR   = 0.3    # minimum keypoint score for elbow landmarks
# COCO-17 indices
SHOULDER_I, ELBOW_I, WRIST_I = 6, 8, 10


def build_model() -> Body:
    return Body(pose='rtmpose-m', backend='onnxruntime', device='cpu')


def extract_elbow_angles(video_path: str, model: Body) -> list:
    cap = cv2.VideoCapture(video_path)
    angles, frame_idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % SAMPLE == 0:
            # rtmlib takes BGR directly
            keypoints, scores = model(frame)   # (N,17,2), (N,17)
            if keypoints.shape[0] > 0:
                kps = keypoints[0]             # (17, 2) pixel x,y
                sc  = scores[0]                # (17,)
                if sc[ELBOW_I] < SCORE_THR or sc[WRIST_I] < SCORE_THR:
                    frame_idx += 1
                    continue
                kps = kps / FRAME_W            # normalise to relative coords
                a, b, c = kps[SHOULDER_I], kps[ELBOW_I], kps[WRIST_I]
                ba, bc  = a - b, c - b
                denom   = np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10
                cos     = np.dot(ba, bc) / denom
                ang     = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
                if not np.isnan(ang):
                    angles.append(ang)
        frame_idx += 1
    cap.release()
    return angles


def angular_dtw_classify(pat: list, exp: list, threshold: float) -> str:
    if len(pat) < 5 or len(exp) < 5:
        return "incomplete"
    dist, path = fastdtw(pat, exp, dist=lambda x, y: abs(x - y))
    return "complete" if (dist / len(path)) <= threshold else "incomplete"


def run_exercise(ex_folder: str, threshold: float, model: Body) -> dict:
    ex_dir = os.path.join(DATASET, "Exercise", ex_folder)
    complete_vids   = sorted(v for v in glob.glob(os.path.join(ex_dir, "Complete",   "Test", "*.mp4"))
                             if "DASHBOARD" not in v)
    incomplete_vids = sorted(v for v in glob.glob(os.path.join(ex_dir, "Incomplete", "Test", "*.mp4"))
                             if "DASHBOARD" not in v)

    print(f"\n{'─'*60}")
    print(f"Exercise: {ex_folder}")
    print(f"  Test set: {len(complete_vids)} complete, {len(incomplete_vids)} incomplete")

    expert_path  = complete_vids[0]
    patient_vids = [(v, "complete") for v in complete_vids[1:]] + \
                   [(v, "incomplete") for v in incomplete_vids]

    print(f"  Expert : {os.path.basename(expert_path)}")
    print("  Extracting expert angles …")
    expert_angles = extract_elbow_angles(expert_path, model)
    print(f"  Expert angle frames: {len(expert_angles)}")

    correct = total = 0
    for i, (vid_path, true_label) in enumerate(patient_vids, 1):
        angles = extract_elbow_angles(vid_path, model)
        pred   = angular_dtw_classify(angles, expert_angles, threshold)
        correct += pred == true_label
        total   += 1
        mark    = "✓" if pred == true_label else "✗"
        print(f"  [{i:2d}/{len(patient_vids)}] {mark}  {os.path.basename(vid_path):<32s}  "
              f"true={true_label:<10s}  pred={pred}")

    acc = round(correct / total * 100, 1) if total else 0.0
    print(f"  RTMPose-m  accuracy: {correct}/{total} = {acc}%")
    return {"correct": correct, "total": total, "accuracy": acc}


def main() -> None:
    print("Loading RTMPose-m via rtmlib …")
    model = build_model()

    results = {}
    for idx, (ex_folder, threshold) in enumerate(EXERCISES, 1):
        results[f"ex{idx}"] = run_exercise(ex_folder, threshold, model)

    print("\n" + "="*60)
    print("RTMPose-m Summary")
    for idx in range(1, 5):
        r = results[f"ex{idx}"]
        print(f"  Ex{idx}: {r['correct']}/{r['total']} = {r['accuracy']}%")

    json.dump(results, open("/tmp/hrnet_results.json", "w"), indent=2)
    print("\nSaved /tmp/hrnet_results.json")


if __name__ == "__main__":
    main()
