#!/usr/bin/env python3
"""
Run KeypointRCNN (ResNet50-FPN) on Exercise 1 test set and classify
using Angular DTW. Saves result to /tmp/krcnn_ex1_results.json.

Usage:
    /opt/homebrew/bin/python3.10 scripts/run_krcnn_ex1.py
"""
import glob, json, os, time
import numpy as np
import cv2
import torch
import torchvision
from fastdtw import fastdtw

DATASET = os.path.join(os.path.dirname(__file__), "..",
                       "An_upper_limb_stroke_rehabilitation_exercise_video")
EX_DIR  = os.path.join(DATASET, "Exercise", "1_Lifting_an_Object")
SAMPLE  = 3      # use every Nth frame (balances speed vs DTW fidelity)
OPT_THR = 20.0   # calibrated threshold from Section 3

krcnn = torchvision.models.detection.keypointrcnn_resnet50_fpn(
    weights=torchvision.models.detection.KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
krcnn.eval()


def extract_elbow_angles(video_path: str) -> list[float]:
    cap = cv2.VideoCapture(video_path)
    angles, frame_idx = [], 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % SAMPLE == 0:
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                t_in = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
                out  = krcnn(t_in)
                if out and len(out[0]['keypoints']) > 0:
                    best = out[0]['scores'].argmax().item()
                    kps  = out[0]['keypoints'][best].numpy()  # (17, 3)
                    a, b, c = kps[6, :2], kps[8, :2], kps[10, :2]
                    ba, bc  = a - b, c - b
                    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-10)
                    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
                    if not np.isnan(ang):
                        angles.append(ang)
            frame_idx += 1
    cap.release()
    return angles


def angular_dtw_classify(pat: list[float], exp: list[float]) -> str:
    if len(pat) < 5 or len(exp) < 5:
        return "incomplete"
    dist, path = fastdtw(pat, exp, dist=lambda x, y: abs(x - y))
    return "complete" if (dist / len(path)) <= OPT_THR else "incomplete"


def main() -> None:
    complete_vids   = sorted(v for v in glob.glob(os.path.join(EX_DIR, "Complete", "Test", "*.mp4"))
                             if "DASHBOARD" not in v)
    incomplete_vids = sorted(v for v in glob.glob(os.path.join(EX_DIR, "Incomplete", "Test", "*.mp4"))
                             if "DASHBOARD" not in v)

    print(f"Test set: {len(complete_vids)} complete, {len(incomplete_vids)} incomplete")

    expert_path  = complete_vids[0]
    patient_vids = [(v, "complete") for v in complete_vids[1:]] + \
                   [(v, "incomplete") for v in incomplete_vids]

    print(f"Expert : {os.path.basename(expert_path)}")
    print("Extracting expert angles (this takes ~10 s) …")
    expert_angles = extract_elbow_angles(expert_path)
    print(f"  {len(expert_angles)} angle frames")

    correct = total = 0
    for i, (vid_path, true_label) in enumerate(patient_vids, 1):
        angles = extract_elbow_angles(vid_path)
        pred   = angular_dtw_classify(angles, expert_angles)
        correct += pred == true_label
        total   += 1
        mark    = "✓" if pred == true_label else "✗"
        print(f"  [{i:2d}/{len(patient_vids)}] {mark}  {os.path.basename(vid_path):<32s}  "
              f"true={true_label:<10s}  pred={pred}")

    acc = round(correct / total * 100, 1) if total else 0.0
    print(f"\nKeypointRCNN  Ex1 accuracy: {correct}/{total} = {acc}%")

    result = {"correct": correct, "total": total, "accuracy": acc}
    json.dump(result, open("/tmp/krcnn_ex1_results.json", "w"))
    print("Saved /tmp/krcnn_ex1_results.json")


if __name__ == "__main__":
    main()
