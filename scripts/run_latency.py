#!/usr/bin/env python3
"""
Measure per-frame inference latency for MediaPipe, YOLOv8-Pose, and HRNet-W32.
Runs on 50 real frames from Exercise 1; first 5 dropped as warm-up.

Install dependencies before running:
    pip install openmim
    mim install mmengine mmcv mmdet mmpose

Usage:
    /opt/homebrew/bin/python3.10 scripts/run_latency.py
"""
import glob, json, os, time
import numpy as np
import cv2

BASE    = os.path.join(os.path.dirname(__file__), "..")
VID_DIR = os.path.join(BASE, "An_upper_limb_stroke_rehabilitation_exercise_video",
                       "Exercise", "1_Lifting_an_Object", "Complete", "Test")
N_FRAMES = 50
WARMUP   = 5


def load_frames(n: int = N_FRAMES) -> list:
    vids = [v for v in glob.glob(os.path.join(VID_DIR, "*.mp4"))
            if "DASHBOARD" not in v]
    cap, frames = cv2.VideoCapture(vids[0]), []
    while len(frames) < n:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    print(f"Loaded {len(frames)} frames from {os.path.basename(vids[0])}")
    return frames


def benchmark_mediapipe(frames: list) -> dict:
    import mediapipe as mp
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                   min_detection_confidence=0.5, min_tracking_confidence=0.5)
    times = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t0  = time.perf_counter()
        pose.process(rgb)
        times.append((time.perf_counter() - t0) * 1000)
    pose.close()
    times = times[WARMUP:]
    return {"mean": round(np.mean(times), 1), "std": round(np.std(times), 1),
            "p95": round(np.percentile(times, 95), 1)}


def benchmark_yolo(frames: list) -> dict:
    from ultralytics import YOLO
    model = YOLO(os.path.join(BASE, "yolov8n-pose.pt"))
    times = []
    for f in frames:
        t0 = time.perf_counter()
        model.predict(f, verbose=False, device="cpu")
        times.append((time.perf_counter() - t0) * 1000)
    times = times[WARMUP:]
    return {"mean": round(np.mean(times), 1), "std": round(np.std(times), 1),
            "p95": round(np.percentile(times, 95), 1)}


def benchmark_hrnet(frames: list) -> dict:
    from mmpose.apis import MMPoseInferencer
    inferencer = MMPoseInferencer(pose2d="human")
    times = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t0  = time.perf_counter()
        next(inferencer(rgb, return_vis=False))
        times.append((time.perf_counter() - t0) * 1000)
    times = times[WARMUP:]
    return {"mean": round(np.mean(times), 1), "std": round(np.std(times), 1),
            "p95": round(np.percentile(times, 95), 1)}


def main() -> None:
    frames = load_frames()

    print("\nMediaPipe …")
    mp_r = benchmark_mediapipe(frames)
    print(f"  mean={mp_r['mean']} ms  std={mp_r['std']} ms  p95={mp_r['p95']} ms")

    print("YOLOv8-Pose …")
    yolo_r = benchmark_yolo(frames)
    print(f"  mean={yolo_r['mean']} ms  std={yolo_r['std']} ms  p95={yolo_r['p95']} ms")

    print("HRNet-W32 …")
    hrnet_r = benchmark_hrnet(frames)
    print(f"  mean={hrnet_r['mean']} ms  std={hrnet_r['std']} ms  p95={hrnet_r['p95']} ms")

    result = {"MediaPipe": mp_r, "YOLOv8-Pose": yolo_r, "HRNet-W32": hrnet_r}
    json.dump(result, open("/tmp/latency_results.json", "w"))
    print("\nSaved /tmp/latency_results.json")


if __name__ == "__main__":
    main()
