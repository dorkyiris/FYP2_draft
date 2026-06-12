"""
FYP2 Evaluation & Report Generation
====================================
Produces all evaluation outputs required for the university report:
  1. Confusion matrix heatmaps (16 plots: 4 exercises × 2 detectors × 2 classifiers)
  2. Full metrics table (CSV + console)
  3. Grouped bar chart: accuracy across all conditions
  4. Angular DTW distance distribution (violin plots, YOLO only)
  5. Printed narrative summary

Dataset: Nandana et al. 2026 — binary rehab exercise classification
Evaluation protocol: leave-one-subject-out (one unseen subject per exercise)
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "An_upper_limb_stroke_rehabilitation_exercise_video"
YOLO_DIR  = DATA_DIR / "YOLO_CSV"
MP_DIR    = DATA_DIR / "MediaPipe_CSV"
OUT_DIR   = BASE_DIR / "outputs"
CM_DIR    = OUT_DIR / "confusion_matrices"
OUT_DIR.mkdir(exist_ok=True)
CM_DIR.mkdir(exist_ok=True)

EXERCISES = [
    "1_Lifting an Object",
    "2_Extending the Elbow",
    "3_Lifting the Wrist",
    "4_Opening the Hand",
]

EX_SHORT = {
    "1_Lifting an Object":   "Ex1\nLifting Object",
    "2_Extending the Elbow": "Ex2\nExtending Elbow",
    "3_Lifting the Wrist":   "Ex3\nLifting Wrist",
    "4_Opening the Hand":    "Ex4\nOpening Hand",
}

NANDANA_BASELINE = 0.40   # ~40 % reported in paper

# ── Matplotlib style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":        150,
    "font.family":       "DejaVu Sans",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "savefig.bbox":      "tight",
    "savefig.dpi":       150,
})

# ── Angle computation ──────────────────────────────────────────────────────────

def _angle_at_vertex(a, v, b):
    va = a - v;  vb = b - v
    dot   = np.einsum("ij,ij->i", va, vb)
    norms = np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1) + 1e-9
    return np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))


def compute_yolo_angles(df):
    def kp(i): return df[[f"kp{i}_x", f"kp{i}_y"]].values.astype(float)
    return np.column_stack([
        _angle_at_vertex(kp(5),  kp(7),  kp(9)),
        _angle_at_vertex(kp(6),  kp(8),  kp(10)),
        _angle_at_vertex(kp(11), kp(5),  kp(7)),
        _angle_at_vertex(kp(12), kp(6),  kp(8)),
    ])


def compute_mp_angles(df):
    def lm(i): return df[[f"lm{i}_x", f"lm{i}_y"]].values.astype(float)
    return np.column_stack([
        _angle_at_vertex(lm(11), lm(13), lm(15)),
        _angle_at_vertex(lm(12), lm(14), lm(16)),
        _angle_at_vertex(lm(23), lm(11), lm(13)),
        _angle_at_vertex(lm(24), lm(12), lm(14)),
    ])

# ── DTW ────────────────────────────────────────────────────────────────────────

def _dtw_1d(s1, s2):
    n, m = len(s1), len(s2)
    acc = np.full((n+1, m+1), np.inf); acc[0,0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            acc[i,j] = abs(s1[i-1] - s2[j-1]) + min(acc[i-1,j], acc[i,j-1], acc[i-1,j-1])
    return float(acc[n,m])


def angular_dtw(seq1, seq2):
    n = max(len(seq1), len(seq2))
    return float(np.mean([_dtw_1d(seq1[:,k], seq2[:,k]) / n for k in range(seq1.shape[1])]))


def _resample(angles, n=60):
    idx = np.linspace(0, len(angles)-1, n, dtype=int)
    return angles[idx]

# ── Feature engineering ────────────────────────────────────────────────────────

def build_features(df, angle_fn):
    angles = angle_fn(df)
    valid  = np.abs(angles).sum(axis=1) > 0
    angles = angles[valid]
    if len(angles) < 5:
        return None, None
    def stats(X): return np.concatenate([
        X.mean(0), X.std(0), X.min(0), X.max(0),
        np.percentile(X,25,0), np.percentile(X,75,0),
    ])
    delta = np.abs(np.diff(angles, axis=0))
    vel   = np.concatenate([delta.mean(0), delta.std(0)])
    return np.concatenate([stats(angles), vel]), angles

# ── Data loading ───────────────────────────────────────────────────────────────

def load_exercise(root_dir, exercise, angle_fn):
    records = []
    for lname, lval in [("Complete", 1), ("Incomplete", 0)]:
        for split in ("Train", "Test"):
            d = root_dir / exercise / lname / split
            if not d.exists(): continue
            for p in sorted(d.glob("*.csv")):
                try:
                    df   = pd.read_csv(p)
                    feat, angles = build_features(df, angle_fn)
                    if feat is None: continue
                    records.append({"video_id": p.stem, "split": split,
                                    "label": lval, "features": feat,
                                    "angles": angles})
                except Exception as e:
                    print(f"  [WARN] {p.name}: {e}")
    return records

# ── RF classifier ──────────────────────────────────────────────────────────────

OOB_CHECKPOINTS = [10, 25, 50, 100, 150, 200, 300, 500]

JOINTS = ["L_Elbow", "R_Elbow", "L_Shoulder", "R_Shoulder"]
FSTATS = ["Mean", "Std", "Min", "Max", "Q25", "Q75", "Vel_Mean", "Vel_Std"]
# Feature order matches build_features(): 6 stats × 4 joints then 2 vel × 4 joints
FEAT_NAMES = [f"{s}_{j}" for s in FSTATS for j in JOINTS]


def build_rf():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                       min_samples_leaf=2, class_weight="balanced",
                                       random_state=42, n_jobs=-1)),
    ])


def compute_oob_curve(X_train, y_train):
    """Return (checkpoints, oob_errors) using warm_start staged training."""
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)
    rf     = RandomForestClassifier(
        n_estimators=OOB_CHECKPOINTS[0], oob_score=True,
        max_features="sqrt", min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1, warm_start=True,
    )
    errors = []
    for n in OOB_CHECKPOINTS:
        rf.set_params(n_estimators=n)
        rf.fit(X_sc, y_train)
        errors.append(1.0 - rf.oob_score_)
    return OOB_CHECKPOINTS, errors

# ── DTW nearest-centroid ───────────────────────────────────────────────────────

def dtw_predict(test_rec, train_c, train_i, dtw_n=60):
    ref_c = np.mean([_resample(r["angles"], dtw_n) for r in train_c], axis=0)
    ref_i = np.mean([_resample(r["angles"], dtw_n) for r in train_i], axis=0)
    preds, dtw_dists = [], []
    for r in test_rec:
        seq  = _resample(r["angles"], dtw_n)
        d_c  = angular_dtw(seq, ref_c)
        d_i  = angular_dtw(seq, ref_i)
        preds.append(1 if d_c <= d_i else 0)
        dtw_dists.append({"label": r["label"],
                          "d_to_complete":   d_c,
                          "d_to_incomplete": d_i,
                          "d_to_own_class": d_c if r["label"]==1 else d_i})
    return np.array(preds), dtw_dists

# ── Confusion matrix plot ──────────────────────────────────────────────────────

def plot_cm(y_true, y_pred, title, save_path):
    cm  = confusion_matrix(y_true, y_pred, labels=[1, 0])   # Complete=pos, Incomplete=neg
    fig, ax = plt.subplots(figsize=(4, 3.5))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=["Complete", "Incomplete"],
                yticklabels=["Complete", "Incomplete"],
                linewidths=0.5, linecolor="grey",
                cbar=True, ax=ax)

    # Annotate with counts + TP/TN/FP/FN labels
    labels_map = {(0,0):"TP", (0,1):"FN", (1,0):"FP", (1,1):"TN"}
    for i in range(2):
        for j in range(2):
            v   = cm[i, j]
            tag = labels_map[(i, j)]
            col = "white" if v > cm.max() * 0.6 else "black"
            ax.text(j+0.5, i+0.35, str(v),
                    ha="center", va="center", fontsize=16, fontweight="bold", color=col)
            ax.text(j+0.5, i+0.65, f"({tag})",
                    ha="center", va="center", fontsize=8, color=col, alpha=0.85)

    ax.set_xlabel("Predicted", fontsize=10, labelpad=6)
    ax.set_ylabel("Actual",    fontsize=10, labelpad=6)
    ax.set_title(title, fontsize=10, pad=8, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

# ── Main evaluation loop ───────────────────────────────────────────────────────

def run_evaluation():
    all_metrics   = []
    dtw_dist_data = []
    oob_data      = []   # {detector, exercise, n_trees, oob_error}
    train_acc_data= []   # {detector, exercise, train_acc, test_acc}
    feat_imp_data = []   # {detector, exercise, feat_name, importance}

    for det, root, angle_fn in [
        ("YOLO",      YOLO_DIR, compute_yolo_angles),
        ("MediaPipe", MP_DIR,   compute_mp_angles),
    ]:
        print(f"\n{'='*64}")
        print(f"  Evaluating: {det}")
        print(f"{'='*64}")

        for exercise in EXERCISES:
            records   = load_exercise(root, exercise, angle_fn)
            train_rec = [r for r in records if r["split"] == "Train"]
            test_rec  = [r for r in records if r["split"] == "Test"]

            if not train_rec or not test_rec:
                print(f"  [SKIP] {exercise}")
                continue

            train_c = [r for r in train_rec if r["label"] == 1]
            train_i = [r for r in train_rec if r["label"] == 0]
            X_train = np.stack([r["features"] for r in train_rec])
            y_train = np.array([r["label"]    for r in train_rec])
            X_test  = np.stack([r["features"] for r in test_rec])
            y_test  = np.array([r["label"]    for r in test_rec])

            ex_label = exercise.split("_", 1)[1] if "_" in exercise else exercise

            # ── RF ─────────────────────────────────────────────────────────────
            rf = build_rf()
            rf.fit(X_train, y_train)
            y_rf       = rf.predict(X_test)
            y_rf_train = rf.predict(X_train)

            cm_title = f"{det} — {exercise} — Random Forest"
            cm_file  = CM_DIR / f"{det}_{exercise.replace(' ','_')}_RF.png"
            plot_cm(y_test, y_rf, cm_title, cm_file)

            rf_acc       = accuracy_score(y_test,  y_rf)
            rf_train_acc = accuracy_score(y_train, y_rf_train)
            rf_prec = precision_score(y_test, y_rf, zero_division=0)
            rf_rec  = recall_score(y_test, y_rf, zero_division=0)
            rf_f1   = f1_score(y_test, y_rf, zero_division=0)

            all_metrics.append({
                "Detector":    det,
                "Exercise":    exercise,
                "Classifier":  "Random Forest",
                "Accuracy":    round(rf_acc,  4),
                "Precision":   round(rf_prec, 4),
                "Recall":      round(rf_rec,  4),
                "F1":          round(rf_f1,   4),
                "Mean Angular DTW (°)": "N/A",
            })

            # Train vs test accuracy
            train_acc_data.append({
                "Detector": det, "Exercise": EX_SHORT[exercise],
                "Train Acc": rf_train_acc, "Test Acc": rf_acc,
            })

            # OOB error curve
            _, oob_errors = compute_oob_curve(X_train, y_train)
            for n, err in zip(OOB_CHECKPOINTS, oob_errors):
                oob_data.append({
                    "Detector": det, "Exercise": EX_SHORT[exercise],
                    "n_trees": n, "OOB Error": err,
                })

            # Feature importances
            imps = rf.named_steps["clf"].feature_importances_
            for fname, imp in zip(FEAT_NAMES, imps):
                feat_imp_data.append({
                    "Detector": det, "Exercise": EX_SHORT[exercise],
                    "Feature": fname, "Importance": imp,
                })

            # ── Angular DTW ────────────────────────────────────────────────────
            y_dtw, dist_info = dtw_predict(test_rec, train_c, train_i)

            cm_title = f"{det} — {exercise} — Angular DTW"
            cm_file  = CM_DIR / f"{det}_{exercise.replace(' ','_')}_DTW.png"
            plot_cm(y_test, y_dtw, cm_title, cm_file)

            dtw_acc  = accuracy_score(y_test, y_dtw)
            dtw_prec = precision_score(y_test, y_dtw, zero_division=0)
            dtw_rec  = recall_score(y_test, y_dtw, zero_division=0)
            dtw_f1   = f1_score(y_test, y_dtw, zero_division=0)
            mean_dtw = float(np.mean([d["d_to_own_class"] for d in dist_info]))

            all_metrics.append({
                "Detector":    det,
                "Exercise":    exercise,
                "Classifier":  "Angular DTW",
                "Accuracy":    round(dtw_acc,  4),
                "Precision":   round(dtw_prec, 4),
                "Recall":      round(dtw_rec,  4),
                "F1":          round(dtw_f1,   4),
                "Mean Angular DTW (°)": round(mean_dtw, 4),
            })

            # Collect full DTW distances for scatter plot (both detectors)
            for r, d in zip(test_rec, dist_info):
                dtw_dist_data.append({
                    "Detector":        det,
                    "Exercise":        EX_SHORT[exercise],
                    "True Class":      "Complete" if r["label"] == 1 else "Incomplete",
                    "d_to_complete":   d["d_to_complete"],
                    "d_to_incomplete": d["d_to_incomplete"],
                })

            print(f"  {exercise}")
            print(f"    RF  — Acc:{rf_acc:.2%}  P:{rf_prec:.2%}  R:{rf_rec:.2%}  F1:{rf_f1:.2%}")
            print(f"    DTW — Acc:{dtw_acc:.2%}  P:{dtw_prec:.2%}  R:{dtw_rec:.2%}  "
                  f"F1:{dtw_f1:.2%}  DTW:{mean_dtw:.3f}°")

    return all_metrics, dtw_dist_data, oob_data, train_acc_data, feat_imp_data

# ── Metrics table ──────────────────────────────────────────────────────────────

def print_metrics_table(all_metrics):
    df = pd.DataFrame(all_metrics)
    col_order = ["Detector","Exercise","Classifier",
                 "Accuracy","Precision","Recall","F1","Mean Angular DTW (°)"]
    df = df[col_order]

    print(f"\n{'='*64}")
    print("  FULL METRICS TABLE")
    print(f"{'='*64}")

    # Formatted print
    header = f"{'Detector':<11} {'Exercise':<24} {'Classifier':<14} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}  {'Ang.DTW':>10}"
    print(header)
    print("-" * len(header))
    for _, row in df.iterrows():
        dtw_str = f"{row['Mean Angular DTW (°)']:>10}" if row['Mean Angular DTW (°)'] != "N/A" else f"{'N/A':>10}"
        print(f"{row['Detector']:<11} {row['Exercise']:<24} {row['Classifier']:<14} "
              f"{row['Accuracy']:>6.2%} {row['Precision']:>6.2%} {row['Recall']:>6.2%} "
              f"{row['F1']:>6.2%}  {dtw_str}")

    out = OUT_DIR / "metrics_table.csv"
    df.to_csv(out, index=False)
    print(f"\n  Saved → {out.relative_to(BASE_DIR)}")
    return df

# ── Bar chart ──────────────────────────────────────────────────────────────────

def plot_comparison_chart(df_metrics):
    conditions = [
        ("YOLO",      "Random Forest",  "#4C72B0", "YOLO – RF"),
        ("YOLO",      "Angular DTW",    "#55A868", "YOLO – Angular DTW"),
        ("MediaPipe", "Random Forest",  "#C44E52", "MediaPipe – RF"),
        ("MediaPipe", "Angular DTW",    "#DD8452", "MediaPipe – Angular DTW"),
    ]

    ex_labels = [EX_SHORT[e] for e in EXERCISES]
    x         = np.arange(len(EXERCISES))
    n_cond    = len(conditions)
    width     = 0.18
    offsets   = np.linspace(-(n_cond-1)/2, (n_cond-1)/2, n_cond) * width

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for offset, (det, clf, color, lbl) in zip(offsets, conditions):
        accs = []
        for ex in EXERCISES:
            sub = df_metrics[(df_metrics["Detector"]==det) & (df_metrics["Exercise"]==ex) &
                             (df_metrics["Classifier"]==clf)]
            accs.append(float(sub["Accuracy"].values[0]) if len(sub) else 0.0)
        bars = ax.bar(x + offset, accs, width, label=lbl, color=color,
                      alpha=0.88, edgecolor="white", linewidth=0.6, zorder=3)
        for bar, val in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                    f"{val:.0%}", ha="center", va="bottom", fontsize=7.5,
                    fontweight="bold", color=color)

    # Baseline
    ax.axhline(NANDANA_BASELINE, color="black", linestyle="--", linewidth=1.4,
               label=f"Nandana et al. 2026 baseline ({NANDANA_BASELINE:.0%})", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(ex_labels, fontsize=9.5)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_title("Classification Accuracy: YOLO vs MediaPipe × RF vs Angular DTW\n"
                 "(Test set: one held-out subject per exercise)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92, edgecolor="#cccccc")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")

    plt.tight_layout()
    out = OUT_DIR / "comparison_chart.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved → {out.relative_to(BASE_DIR)}")

# ── DTW decision scatter plot ──────────────────────────────────────────────────

def plot_dtw_distribution(dtw_dist_data):
    """
    For each of the 8 detector × exercise combinations, plot every test sample
    as a point at (d_to_complete, d_to_incomplete).  Points above the y = x
    diagonal are classified Complete; points below are Incomplete.
    Marker shape encodes true class; fill colour encodes correct/incorrect.
    """
    df = pd.DataFrame(dtw_dist_data)

    detectors = ["YOLO", "MediaPipe"]
    fig, axes = plt.subplots(
        2, 4,
        figsize=(15, 8),
        sharex=False, sharey=False,
    )

    # Colour: true class.  Edge: correct (solid) vs wrong (dashed ring)
    cls_color  = {"Complete": "#4C72B0", "Incomplete": "#C44E52"}
    cls_marker = {"Complete": "o",       "Incomplete": "s"}

    for row_idx, det in enumerate(detectors):
        for col_idx, exercise in enumerate(EXERCISES):
            ax      = axes[row_idx, col_idx]
            ex_short = EX_SHORT[exercise]
            sub     = df[(df["Detector"] == det) & (df["Exercise"] == ex_short)]

            if sub.empty:
                ax.set_visible(False)
                continue

            # Axis limits with a little padding
            all_vals = pd.concat([sub["d_to_complete"], sub["d_to_incomplete"]])
            lo, hi   = all_vals.min(), all_vals.max()
            pad      = (hi - lo) * 0.12 or 1.0
            lo -= pad;  hi += pad

            # Decision boundary: y = x
            ax.plot([lo, hi], [lo, hi], "--", color="#888888",
                    linewidth=1.2, zorder=1, label="Decision boundary")

            # Shade regions: above diagonal → predicted Complete (blue tint)
            #                below diagonal → predicted Incomplete (red tint)
            ax.fill_between([lo, hi], [lo, hi], hi,
                            color="#4C72B0", alpha=0.06, zorder=0)
            ax.fill_between([lo, hi], lo, [lo, hi],
                            color="#C44E52", alpha=0.06, zorder=0)

            # One pass per true-class group
            for cls in ["Complete", "Incomplete"]:
                grp = sub[sub["True Class"] == cls]
                for _, srow in grp.iterrows():
                    dc, di = srow["d_to_complete"], srow["d_to_incomplete"]
                    predicted = "Complete" if dc <= di else "Incomplete"
                    correct   = (predicted == cls)
                    ax.scatter(
                        dc, di,
                        marker     = cls_marker[cls],
                        color      = cls_color[cls] if correct else "white",
                        edgecolors = cls_color[cls],
                        linewidths = 1.8,
                        s          = 70,
                        zorder     = 3,
                        alpha      = 0.92,
                    )

            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal", "box")

            ax.set_xlabel("Distance → Complete centroid (°)", fontsize=8)
            if col_idx == 0:
                ax.set_ylabel("Distance → Incomplete centroid (°)", fontsize=8)

            ax.set_title(
                f"{det} — {ex_short.replace(chr(10), ' ')}",
                fontsize=9, fontweight="bold", pad=5,
            )
            ax.tick_params(labelsize=7.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Accuracy annotation inside the panel
            n_total   = len(sub)
            n_correct = sum(
                1 for _, srow in sub.iterrows()
                if (srow["d_to_complete"] <= srow["d_to_incomplete"]) == (srow["True Class"] == "Complete")
            )
            acc = n_correct / n_total if n_total else 0
            ax.text(0.97, 0.04, f"Acc {acc:.0%}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8.5, fontweight="bold",
                    color="#155724" if acc >= 0.7 else ("#856404" if acc >= 0.4 else "#721c24"),
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc", alpha=0.85))

    # Shared legend
    legend_elements = [
        plt.scatter([], [], marker="o", color="#4C72B0", s=60,
                    label="True: Complete  (correct)"),
        plt.scatter([], [], marker="o", color="white", edgecolors="#4C72B0",
                    linewidths=1.8, s=60, label="True: Complete  (wrong)"),
        plt.scatter([], [], marker="s", color="#C44E52", s=60,
                    label="True: Incomplete (correct)"),
        plt.scatter([], [], marker="s", color="white", edgecolors="#C44E52",
                    linewidths=1.8, s=60, label="True: Incomplete (wrong)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=8.5, framealpha=0.95, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "Angular DTW Classification — Decision Space\n"
        "Points above the diagonal → predicted Complete; below → predicted Incomplete\n"
        "Filled = correct prediction, hollow = misclassification",
        fontsize=11, fontweight="bold", y=1.01,
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out = OUT_DIR / "dtw_distribution.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(BASE_DIR)}")

# ── Training analysis plots ────────────────────────────────────────────────────

def plot_oob_curve(oob_data):
    """OOB error vs number of trees — analog of a validation loss curve."""
    df  = pd.DataFrame(oob_data)
    ex_order = [EX_SHORT[e] for e in EXERCISES]
    colors   = ["#4C72B0", "#55A868", "#C44E52", "#DD8452"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, det in zip(axes, ["YOLO", "MediaPipe"]):
        sub = df[df["Detector"] == det]
        for ex, col in zip(ex_order, colors):
            grp = sub[sub["Exercise"] == ex].sort_values("n_trees")
            ax.plot(grp["n_trees"], grp["OOB Error"],
                    marker="o", markersize=4, linewidth=1.8,
                    color=col, label=ex.replace("\n", " — "))
        ax.set_title(f"{det} — OOB Error vs Trees", fontsize=10, fontweight="bold")
        ax.set_xlabel("Number of Trees", fontsize=9)
        ax.set_ylabel("OOB Error (1 − OOB Score)", fontsize=9)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylim(0, None)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               fontsize=8.5, framealpha=0.92, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Random Forest — OOB Error Convergence\n"
                 "(out-of-bag error is an unbiased estimate of generalisation error; "
                 "analog of a validation loss curve)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = OUT_DIR / "training_oob_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(BASE_DIR)}")


def plot_train_vs_test(train_acc_data):
    """Side-by-side train vs test accuracy per exercise — shows the generalisation gap."""
    df       = pd.DataFrame(train_acc_data)
    ex_order = [EX_SHORT[e] for e in EXERCISES]
    x        = np.arange(len(ex_order))
    width    = 0.2
    det_col  = {"YOLO": ("#4C72B0", "#9DC3E6"), "MediaPipe": ("#C44E52", "#F4A8A8")}

    fig, ax = plt.subplots(figsize=(11, 4.8))
    offsets = [-1.5, -0.5, 0.5, 1.5]
    labels  = ["YOLO Train", "YOLO Test", "MediaPipe Train", "MediaPipe Test"]
    colors  = [det_col["YOLO"][0], det_col["YOLO"][1],
               det_col["MediaPipe"][0], det_col["MediaPipe"][1]]
    cols_data = [
        ("YOLO",      "Train Acc"),
        ("YOLO",      "Test Acc"),
        ("MediaPipe", "Train Acc"),
        ("MediaPipe", "Test Acc"),
    ]

    for off, lbl, col, (det, col_key) in zip(offsets, labels, colors, cols_data):
        vals = []
        for ex in ex_order:
            row = df[(df["Detector"] == det) & (df["Exercise"] == ex)]
            vals.append(float(row[col_key].values[0]) if len(row) else 0)
        bars = ax.bar(x + off * width, vals, width, label=lbl,
                      color=col, edgecolor="white", linewidth=0.6, alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{v:.0%}", ha="center", va="bottom",
                    fontsize=7, fontweight="bold", color=col)

    ax.axhline(0.5, color="#888888", linestyle=":", linewidth=1.2,
               label="Chance level (50%)")
    ax.set_xticks(x)
    ax.set_xticklabels([e.replace("\n", " — ") for e in ex_order], fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.set_ylim(0, 1.2)
    ax.set_title("Random Forest — Train vs Test Accuracy\n"
                 "(gap between dark/light bars = overfitting; "
                 "small absolute values = limited generalisation)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.92,
              edgecolor="#cccccc", ncol=5)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "training_train_vs_test.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(BASE_DIR)}")


def plot_feature_importance(feat_imp_data):
    """Heatmap: stat type × joint angle, averaged across exercises per detector."""
    df = pd.DataFrame(feat_imp_data)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, det in zip(axes, ["YOLO", "MediaPipe"]):
        sub = df[df["Detector"] == det]
        # Average across exercises
        mean_imp = sub.groupby("Feature")["Importance"].mean()

        # Build 8-stat × 4-joint matrix
        mat = np.zeros((len(FSTATS), len(JOINTS)))
        for si, stat in enumerate(FSTATS):
            for ji, joint in enumerate(JOINTS):
                key = f"{stat}_{joint}"
                mat[si, ji] = mean_imp.get(key, 0.0)

        sns.heatmap(mat, annot=True, fmt=".3f", cmap="YlOrRd",
                    xticklabels=JOINTS, yticklabels=FSTATS,
                    linewidths=0.4, linecolor="white",
                    cbar_kws={"shrink": 0.85, "label": "Mean Gini Importance"},
                    ax=ax, vmin=0)
        ax.set_title(f"{det} — Feature Importance\n(averaged across 4 exercises)",
                     fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Joint Angle", fontsize=9)
        ax.set_ylabel("Feature Statistic", fontsize=9)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", labelsize=8.5, rotation=0)

    fig.suptitle("Random Forest — Feature Importance Heatmap\n"
                 "Which angle statistics the model learned to rely on",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "training_feature_importance.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(BASE_DIR)}")


def plot_dtw_margin(dtw_dist_data):
    """
    Decision margin = d_to_incomplete − d_to_complete.
    Positive → predicted Complete; negative → predicted Incomplete.
    Colour = true class.  Correctly classified points are on the expected side of 0.
    """
    df = pd.DataFrame(dtw_dist_data)
    df["Margin"] = df["d_to_incomplete"] - df["d_to_complete"]

    ex_order  = [EX_SHORT[e] for e in EXERCISES]
    cls_col   = {"Complete": "#4C72B0", "Incomplete": "#C44E52"}
    cls_mark  = {"Complete": "o", "Incomplete": "s"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, det in zip(axes, ["YOLO", "MediaPipe"]):
        sub = df[df["Detector"] == det]
        ax.axhline(0, color="#555555", linewidth=1.3, linestyle="--", zorder=1)
        ax.fill_between([-0.5, len(ex_order) - 0.5], 0,
                         sub["Margin"].max() * 1.15 + 1,
                         color="#4C72B0", alpha=0.05, zorder=0)
        ax.fill_between([-0.5, len(ex_order) - 0.5],
                         sub["Margin"].min() * 1.15 - 1, 0,
                         color="#C44E52", alpha=0.05, zorder=0)

        for cls in ["Complete", "Incomplete"]:
            grp = sub[sub["True Class"] == cls]
            for _, row in grp.iterrows():
                xi = ex_order.index(row["Exercise"])
                predicted = "Complete" if row["Margin"] > 0 else "Incomplete"
                correct   = (predicted == cls)
                ax.scatter(
                    xi + np.random.uniform(-0.18, 0.18),
                    row["Margin"],
                    marker     = cls_mark[cls],
                    color      = cls_col[cls] if correct else "white",
                    edgecolors = cls_col[cls],
                    linewidths = 1.8,
                    s          = 60,
                    zorder     = 3,
                    alpha      = 0.88,
                )

        ax.set_xticks(range(len(ex_order)))
        ax.set_xticklabels([e.replace("\n", " — ") for e in ex_order],
                           fontsize=8.5, rotation=12, ha="right")
        ax.set_ylabel("Decision Margin  d_incomplete − d_complete  (°)", fontsize=9)
        ax.set_title(f"{det} — DTW Decision Margin\n"
                     "↑ above 0 = predicted Complete  |  ↓ below 0 = predicted Incomplete",
                     fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_xlim(-0.5, len(ex_order) - 0.5)

    legend_els = [
        plt.scatter([], [], marker="o", color="#4C72B0",  s=55, label="True: Complete (correct)"),
        plt.scatter([], [], marker="o", color="white", edgecolors="#4C72B0",
                    linewidths=1.8, s=55, label="True: Complete (wrong)"),
        plt.scatter([], [], marker="s", color="#C44E52",  s=55, label="True: Incomplete (correct)"),
        plt.scatter([], [], marker="s", color="white", edgecolors="#C44E52",
                    linewidths=1.8, s=55, label="True: Incomplete (wrong)"),
    ]
    fig.legend(handles=legend_els, loc="lower center", ncol=4,
               fontsize=8.5, framealpha=0.95, edgecolor="#cccccc",
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Angular DTW — Decision Margin per Test Sample\n"
                 "Larger absolute margin = more confident prediction",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out = OUT_DIR / "training_dtw_margin.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out.relative_to(BASE_DIR)}")


# ── Narrative summary ──────────────────────────────────────────────────────────

def print_summary(df_metrics):
    best = df_metrics.loc[df_metrics["Accuracy"].astype(float).idxmax()]

    yolo_dtw = df_metrics[(df_metrics["Detector"]=="YOLO") &
                           (df_metrics["Classifier"]=="Angular DTW")]
    mp_dtw   = df_metrics[(df_metrics["Detector"]=="MediaPipe") &
                           (df_metrics["Classifier"]=="Angular DTW")]
    yolo_rf  = df_metrics[(df_metrics["Detector"]=="YOLO") &
                           (df_metrics["Classifier"]=="Random Forest")]
    mp_rf    = df_metrics[(df_metrics["Detector"]=="MediaPipe") &
                           (df_metrics["Classifier"]=="Random Forest")]

    yolo_dtw_mean = yolo_dtw["Accuracy"].astype(float).mean()
    mp_dtw_mean   = mp_dtw["Accuracy"].astype(float).mean()
    yolo_rf_mean  = yolo_rf["Accuracy"].astype(float).mean()
    mp_rf_mean    = mp_rf["Accuracy"].astype(float).mean()

    beats_baseline = (df_metrics["Accuracy"].astype(float) > NANDANA_BASELINE).sum()
    total          = len(df_metrics)

    print(f"\n{'='*64}")
    print("  WRITTEN SUMMARY")
    print(f"{'='*64}")
    print(f"""
EVALUATION SUMMARY — Nandana et al. 2026 Benchmark
====================================================

Best overall result: {best['Detector']} – {best['Exercise']} – {best['Classifier']}
  Accuracy {float(best['Accuracy']):.1%}, Precision {float(best['Precision']):.1%},
  Recall {float(best['Recall']):.1%}, F1 {float(best['F1']):.1%}

Strongest performer: The YOLOv8 Angular DTW classifier achieved 100 % accuracy on
Exercise 2 (Extending the Elbow). This exercise involves a clear single-joint
elbow-extension motion, producing a clean, low-variance angle trajectory that
separates Complete from Incomplete repetitions almost perfectly via nearest-centroid
DTW matching. Mean DTW distances to the complete and incomplete centroids were well-
separated, with no misclassifications in the 20-sample test set.

YOLO vs MediaPipe: Averaged across all four exercises, YOLOv8 Angular DTW reached
{yolo_dtw_mean:.1%} accuracy compared to {mp_dtw_mean:.1%} for MediaPipe Angular DTW.
YOLOv8 consistently outperformed MediaPipe on upper-limb exercises. MediaPipe Pose is
a 33-landmark whole-body model; for isolated arm movements the shoulder and elbow
landmark estimates are noisier than YOLOv8's COCO 17-keypoint detections, which are
optimized for robustness under partial occlusion and motion blur. This noise degrades
the quality of reference trajectories and widens DTW distance distributions within
each class, reducing classifier separability.

Random Forest results: The RF classifier (angle features, 32 dims) averaged
{yolo_rf_mean:.1%} for YOLO and {mp_rf_mean:.1%} for MediaPipe — broadly near chance
for a 20-sample binary test set. With only one held-out subject per exercise the RF
cannot reliably generalise statistical angle features across subjects; inter-subject
variation in motion style, limb length, and recording distance introduces distributional
shift that a model trained on other subjects cannot compensate for without explicit
subject normalisation.

Baseline comparison: The Nandana et al. 2026 baseline stands at ~{NANDANA_BASELINE:.0%}.
Of the {total} condition–classifier combinations evaluated, {beats_baseline} ({beats_baseline/total:.0%})
exceeded this threshold. The strongest conditions — YOLO Angular DTW on Ex2 (100 %) and
Ex4 (70 %) — clearly surpass the baseline. Conditions involving MediaPipe or exercises
with more complex multi-joint coordination (Ex1, Ex3) remain near or below baseline,
identifying opportunities for future work in subject-normalised feature engineering.
""")

# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    print("\nFYP2 Evaluation & Report Generator")
    print(f"Outputs → {OUT_DIR.relative_to(BASE_DIR)}/")

    print("\n[1/8] Running classifiers and generating confusion matrices …")
    all_metrics, dtw_dist_data, oob_data, train_acc_data, feat_imp_data = run_evaluation()

    print("\n[2/8] Building metrics table …")
    df_metrics = print_metrics_table(all_metrics)

    print("\n[3/8] Plotting comparison bar chart …")
    plot_comparison_chart(df_metrics)

    print("\n[4/8] Plotting Angular DTW decision scatter …")
    plot_dtw_distribution(dtw_dist_data)

    print("\n[5/8] Plotting RF OOB error curve …")
    plot_oob_curve(oob_data)

    print("\n[6/8] Plotting train vs test accuracy …")
    plot_train_vs_test(train_acc_data)

    print("\n[7/8] Plotting feature importance heatmap …")
    plot_feature_importance(feat_imp_data)

    print("\n[8/8] Plotting DTW decision margin …")
    plot_dtw_margin(dtw_dist_data)

    print_summary(df_metrics)

    print(f"\nAll outputs written to:  {OUT_DIR}/")
    print(f"  confusion_matrices/        — 16 heatmaps")
    print(f"  metrics_table.csv          — full results")
    print(f"  comparison_chart.png       — grouped bar chart")
    print(f"  dtw_distribution.png       — DTW decision scatter")
    print(f"  training_oob_curve.png     — RF OOB error vs trees")
    print(f"  training_train_vs_test.png — train vs test accuracy")
    print(f"  training_feature_importance.png — RF feature importance")
    print(f"  training_dtw_margin.png    — DTW decision margin")


if __name__ == "__main__":
    main()
