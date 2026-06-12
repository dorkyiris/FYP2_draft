"""
Training pipeline: MediaPipe Pose vs YOLOv8 — binary rehab exercise classification.
Dataset: Nandana et al. 2026 (4 exercises, Complete=1 / Incomplete=0).

Test protocol: each test set is a single held-out subject (subject-independent eval).
  Ex1 → subject 07, Ex2 → subject 08, Ex3 → subject 10, Ex4 → subject 09.

Two classifiers are trained and compared per exercise × detector:
  1. Random Forest on per-video statistical + angle features
  2. Angular DTW nearest-centroid (classify by minimum DTW distance to class mean)

Feature design rationale:
  - Raw pixel coordinates fail subject-independent eval (subject body size/position vary).
  - Joint angles are scale/translation-invariant → primary features.
  - Velocity (frame-to-frame delta stats) captures motion dynamics.
  - Statistical aggregates (mean, std, min, max, Q25, Q75) over the video give a fixed
    feature vector from variable-length sequences without padding artefacts.
"""

import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "An_upper_limb_stroke_rehabilitation_exercise_video"
YOLO_DIR  = DATA_DIR / "YOLO_CSV"
MP_DIR    = DATA_DIR / "MediaPipe_CSV"
MODEL_DIR = BASE_DIR / "assets" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXERCISES = [
    "1_Lifting an Object",
    "2_Extending the Elbow",
    "3_Lifting the Wrist",
    "4_Opening the Hand",
]

# ── Angle computation ─────────────────────────────────────────────────────────

def _angle_at_vertex(a: np.ndarray, v: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle (degrees) at vertex v for each row, given 2-D points a, v, b (shape N×2)."""
    va = a - v
    vb = b - v
    dot   = np.einsum("ij,ij->i", va, vb)
    norms = np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-9
    return np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))


def compute_yolo_angles(df: pd.DataFrame) -> np.ndarray:
    """
    COCO keypoints:
      kp5=L_shoulder  kp6=R_shoulder  kp7=L_elbow   kp8=R_elbow
      kp9=L_wrist    kp10=R_wrist    kp11=L_hip    kp12=R_hip
    Returns (N × 4): [L_elbow, R_elbow, L_shoulder, R_shoulder]
    """
    def kp(i):
        return df[[f"kp{i}_x", f"kp{i}_y"]].values.astype(float)
    return np.column_stack([
        _angle_at_vertex(kp(5),  kp(7),  kp(9)),   # left  elbow: shoulder-elbow-wrist
        _angle_at_vertex(kp(6),  kp(8),  kp(10)),  # right elbow
        _angle_at_vertex(kp(11), kp(5),  kp(7)),   # left  shoulder: hip-shoulder-elbow
        _angle_at_vertex(kp(12), kp(6),  kp(8)),   # right shoulder
    ])


def compute_mp_angles(df: pd.DataFrame) -> np.ndarray:
    """
    MediaPipe Pose landmarks:
      lm11=L_shoulder  lm12=R_shoulder  lm13=L_elbow   lm14=R_elbow
      lm15=L_wrist    lm16=R_wrist     lm23=L_hip     lm24=R_hip
    Returns (N × 4): [L_elbow, R_elbow, L_shoulder, R_shoulder]
    """
    def lm(i):
        return df[[f"lm{i}_x", f"lm{i}_y"]].values.astype(float)
    return np.column_stack([
        _angle_at_vertex(lm(11), lm(13), lm(15)),  # left  elbow
        _angle_at_vertex(lm(12), lm(14), lm(16)),  # right elbow
        _angle_at_vertex(lm(23), lm(11), lm(13)),  # left  shoulder
        _angle_at_vertex(lm(24), lm(12), lm(14)),  # right shoulder
    ])


# ── DTW ───────────────────────────────────────────────────────────────────────

def _dtw_1d(s1: np.ndarray, s2: np.ndarray) -> float:
    n, m = len(s1), len(s2)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            acc[i, j] = abs(s1[i-1] - s2[j-1]) + min(acc[i-1, j], acc[i, j-1], acc[i-1, j-1])
    return float(acc[n, m])


def angular_dtw(seq1: np.ndarray, seq2: np.ndarray) -> float:
    """Mean DTW over all angle channels (degrees), normalized by path length."""
    n = max(len(seq1), len(seq2))
    return float(np.mean([_dtw_1d(seq1[:, k], seq2[:, k]) / n
                           for k in range(seq1.shape[1])]))


def _resample(angles: np.ndarray, n: int = 60) -> np.ndarray:
    idx = np.linspace(0, len(angles) - 1, n, dtype=int)
    return angles[idx]


# ── Feature engineering ───────────────────────────────────────────────────────

def _stat_features(X: np.ndarray) -> np.ndarray:
    """6-statistic summary per column: mean, std, min, max, Q25, Q75."""
    return np.concatenate([
        X.mean(axis=0), X.std(axis=0), X.min(axis=0), X.max(axis=0),
        np.percentile(X, 25, axis=0), np.percentile(X, 75, axis=0),
    ])


def _velocity_features(X: np.ndarray) -> np.ndarray:
    """Mean absolute frame-to-frame delta and its std per channel."""
    delta = np.abs(np.diff(X, axis=0))
    return np.concatenate([delta.mean(axis=0), delta.std(axis=0)])


def build_features(df: pd.DataFrame, angle_fn) -> np.ndarray:
    """
    Construct subject-invariant feature vector:
      - 4-angle trajectory stats  (4 × 6 = 24 dims)
      - 4-angle velocity stats    (4 × 2 =  8 dims)
      Total: 32 dims — small, interpretable, scale-invariant.
    """
    angles = angle_fn(df)  # (N × 4)
    # Remove rows where all angles collapsed to 0 (detection failure)
    valid = np.abs(angles).sum(axis=1) > 0
    angles = angles[valid]
    if len(angles) < 5:
        return None
    angle_stats = _stat_features(angles)
    angle_vel   = _velocity_features(angles)
    return np.concatenate([angle_stats, angle_vel])


# ── Data loading ──────────────────────────────────────────────────────────────

def load_exercise(root_dir: Path, exercise: str, angle_fn) -> list[dict]:
    records = []
    for label_name, label_val in [("Complete", 1), ("Incomplete", 0)]:
        for split in ("Train", "Test"):
            split_dir = root_dir / exercise / label_name / split
            if not split_dir.exists():
                continue
            for csv_path in sorted(split_dir.glob("*.csv")):
                try:
                    df = pd.read_csv(csv_path)
                    feats = build_features(df, angle_fn)
                    if feats is None:
                        continue
                    angles_raw = angle_fn(df)
                    valid = np.abs(angles_raw).sum(axis=1) > 0
                    angles_clean = angles_raw[valid]
                    records.append({
                        "video_id": csv_path.stem,
                        "split":    split,
                        "label":    label_val,
                        "features": feats,
                        "angles":   angles_clean,
                    })
                except Exception as exc:
                    print(f"    [WARN] {csv_path.name}: {exc}")
    return records


# ── Classifiers ───────────────────────────────────────────────────────────────

def _build_rf() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])


def _dtw_nearest_centroid_predict(test_rec: list[dict],
                                   train_complete: list[dict],
                                   train_incomplete: list[dict],
                                   dtw_n: int = 60) -> np.ndarray:
    """
    Classify each test sample by minimum Angular DTW distance to:
      - mean Complete    angle sequence (class centroid)
      - mean Incomplete  angle sequence (class centroid)
    """
    ref_c = np.mean([_resample(r["angles"], dtw_n) for r in train_complete],  axis=0)
    ref_i = np.mean([_resample(r["angles"], dtw_n) for r in train_incomplete], axis=0)

    preds = []
    for r in test_rec:
        seq = _resample(r["angles"], dtw_n)
        d_c = angular_dtw(seq, ref_c)
        d_i = angular_dtw(seq, ref_i)
        preds.append(1 if d_c <= d_i else 0)
    return np.array(preds)


# ── Reporting ─────────────────────────────────────────────────────────────────

def _report_metrics(y_true, y_pred, label: str) -> dict:
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    print(f"\n    [{label}]")
    print(f"      Accuracy : {acc:.4f}  ({int(acc*len(y_true))}/{len(y_true)})")
    print(f"      Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(classification_report(y_true, y_pred,
          target_names=["Incomplete", "Complete"], digits=4, zero_division=0))
    return {"accuracy": round(acc,4), "precision": round(prec,4),
            "recall": round(rec,4), "f1": round(f1,4)}


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(detector: str, root_dir: Path, angle_fn) -> list[dict]:
    print(f"\n{'='*62}")
    print(f"  DETECTOR: {detector}")
    print(f"{'='*62}")

    rows = []

    for exercise in EXERCISES:
        print(f"\n  ── {exercise}")
        records = load_exercise(root_dir, exercise, angle_fn)

        train_rec = [r for r in records if r["split"] == "Train"]
        test_rec  = [r for r in records if r["split"] == "Test"]

        if not train_rec or not test_rec:
            print("    [SKIP] insufficient data")
            continue

        X_train = np.stack([r["features"] for r in train_rec])
        y_train = np.array([r["label"]    for r in train_rec])
        X_test  = np.stack([r["features"] for r in test_rec])
        y_test  = np.array([r["label"]    for r in test_rec])

        train_c = [r for r in train_rec if r["label"] == 1]
        train_i = [r for r in train_rec if r["label"] == 0]

        print(f"    Train: {len(train_rec)} (C={len(train_c)}, I={len(train_i)})  "
              f"Test: {len(test_rec)} (C={int((y_test==1).sum())}, I={int((y_test==0).sum())})")
        print(f"    Feature dims: {X_train.shape[1]}  "
              f"(4 angles × 6 stats + 4 vel × 2 stats)")

        # ── Random Forest ──
        rf_model = _build_rf()
        rf_model.fit(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        rf_metrics = _report_metrics(y_test, y_pred_rf, "Random Forest")

        # ── Angular DTW Nearest-Centroid ──
        y_pred_dtw = _dtw_nearest_centroid_predict(test_rec, train_c, train_i)
        dtw_metrics = _report_metrics(y_test, y_pred_dtw, "Angular DTW Nearest-Centroid")

        # ── Angular DTW distances (for reporting) ──
        dtw_n = 60
        ref_c = np.mean([_resample(r["angles"], dtw_n) for r in train_c], axis=0)
        ref_i = np.mean([_resample(r["angles"], dtw_n) for r in train_i], axis=0)
        dtw_c_vals = [angular_dtw(_resample(r["angles"], dtw_n), ref_c)
                      for r in test_rec if r["label"] == 1]
        dtw_i_vals = [angular_dtw(_resample(r["angles"], dtw_n), ref_i)
                      for r in test_rec if r["label"] == 0]
        mean_dtw_c = float(np.mean(dtw_c_vals)) if dtw_c_vals else float("nan")
        mean_dtw_i = float(np.mean(dtw_i_vals)) if dtw_i_vals else float("nan")
        print(f"    Angular DTW (mean, normalized, n={dtw_n} frames):")
        print(f"      Complete   → dist to Complete ref  : {mean_dtw_c:.4f}°")
        print(f"      Incomplete → dist to Incomplete ref: {mean_dtw_i:.4f}°")

        # ── Save RF model ──
        ex_slug = exercise.replace(" ", "_")
        model_path = MODEL_DIR / f"{detector}_{ex_slug}_rf.pkl"
        joblib.dump(rf_model, model_path)
        print(f"    Model saved → {model_path.relative_to(BASE_DIR)}")

        rows.append({
            "detector": detector, "exercise": exercise,
            "train_n": len(train_rec), "test_n": len(test_rec),
            **{f"rf_{k}": v for k, v in rf_metrics.items()},
            **{f"dtw_{k}": v for k, v in dtw_metrics.items()},
            "mean_dtw_complete": round(mean_dtw_c, 4),
            "mean_dtw_incomplete": round(mean_dtw_i, 4),
        })

    return rows


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*62)
    print("  DATA SUMMARY (space-named folders, duplicates excluded)")
    print("="*62)
    print(f"  {'Detector':<11} {'Exercise':<24} {'Class':<11} {'Train':>6}  {'Test':>5}")
    print(f"  {'-'*60}")
    for ex in EXERCISES:
        for det, det_dir in [("YOLO", YOLO_DIR), ("MediaPipe", MP_DIR)]:
            for cls in ("Complete", "Incomplete"):
                tr = len(list((det_dir / ex / cls / "Train").glob("*.csv")))
                te = len(list((det_dir / ex / cls / "Test").glob("*.csv")))
                print(f"  {det:<11} {ex[:24]:<24} {cls:<11} {tr:>6}  {te:>5}")
        print()

    print("\n  Feature design: 4 joint angles × (6 stats + 2 velocity stats) = 32 dims")
    print("  Angles: L_elbow, R_elbow, L_shoulder, R_shoulder")
    print("  Velocity: mean/std of |frame[t] - frame[t-1]|")
    print("  Scale/translation invariant → suited for subject-independent eval\n")

    all_rows = []
    all_rows += run_pipeline("YOLO",      YOLO_DIR, compute_yolo_angles)
    all_rows += run_pipeline("MediaPipe", MP_DIR,   compute_mp_angles)

    df = pd.DataFrame(all_rows)
    col_order = [
        "detector", "exercise", "train_n", "test_n",
        "rf_accuracy", "rf_precision", "rf_recall", "rf_f1",
        "dtw_accuracy", "dtw_precision", "dtw_recall", "dtw_f1",
        "mean_dtw_complete", "mean_dtw_incomplete",
    ]
    df = df[col_order]

    print(f"\n{'='*62}")
    print("  FINAL SUMMARY")
    print(f"{'='*62}")
    print(df.to_string(index=False))

    out = DATA_DIR / "training_results_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nSummary → {out.relative_to(BASE_DIR)}")
    print(f"Models  → {MODEL_DIR.relative_to(BASE_DIR)}/")


if __name__ == "__main__":
    main()
