#!/usr/bin/env python3
"""
Model Validation for MEG Stroke Intervention System.

Loads the trained MEGStrokeNet and evaluates it against test data,
computing regression metrics (MSE, MAE, R2) for each output dimension,
classification metrics (accuracy, sensitivity, specificity, FPR) using a
threshold on valve_extension, and generates diagnostic plots.

Results are broken down by stroke type: healthy, acute, and chronic.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_curve,
)

from models.meg_stroke_net import MEGStrokeNet


# ---------------------------------------------------------------------------
# Synthetic test data generator (matches real model's input shape)
# ---------------------------------------------------------------------------

STROKE_TYPES: List[str] = ["healthy", "acute", "chronic"]
STROKE_TYPE_TO_IDX: Dict[str, int] = {s: i for i, s in enumerate(STROKE_TYPES)}

# Typical ground-truth profiles per stroke type
#   (valve_extension_mean, force_mean, delay_mean)
STROKE_PROFILES: Dict[str, Tuple[float, float, float]] = {
    "healthy": (0.03, 0.03, 0.01),    # near-zero intervention
    "acute":   (0.85, 0.80, 0.35),    # strong intervention
    "chronic": (0.48, 0.42, 0.16),    # moderate intervention
}

OUTPUT_NAMES: List[str] = ["valve_extension", "force_magnitude", "trigger_delay"]
VALVE_THRESHOLD: float = 0.15  # stroke vs no-stroke decision boundary


def generate_test_data(
    n_per_type: int = 200,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Generate synthetic MEG test data with known ground-truth outputs.

    Produces data in the model's native format: (N, 6, 100).

    Returns:
        X: (N, 6, 100) tensor
        y: (N, 3) tensor
        stroke_labels: (N,) string array
    """
    rng = np.random.RandomState(seed)
    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []

    n_ch = MEGStrokeNet.NUM_CHANNELS      # 6
    n_tp = MEGStrokeNet.NUM_TIMEPOINTS    # 100

    for stype in STROKE_TYPES:
        v_mean, f_mean, d_mean = STROKE_PROFILES[stype]
        n = n_per_type

        t = np.linspace(0, 0.5, n_tp)

        signals = np.zeros((n, n_ch, n_tp), dtype=np.float32)
        for i in range(n):
            for ch in range(n_ch):
                # Base oscillations in mu/beta bands
                freq1 = rng.uniform(8, 12)
                freq2 = rng.uniform(12, 32)
                phase1 = rng.uniform(0, 2 * np.pi)
                phase2 = rng.uniform(0, 2 * np.pi)

                amp_mu = 1.0
                amp_beta = 0.5

                if stype == "acute":
                    # Ipsilesional suppression on right channels (1,3,5)
                    if ch in [1, 3, 5]:
                        amp_mu *= 0.2
                        amp_beta *= 0.15
                    else:
                        amp_mu *= 1.1
                elif stype == "chronic":
                    if ch in [1, 3, 5]:
                        amp_mu *= 0.55
                        amp_beta *= 0.35

                signals[i, ch] = (
                    amp_mu * np.sin(2 * np.pi * freq1 * t + phase1)
                    + amp_beta * np.sin(2 * np.pi * freq2 * t + phase2)
                    + rng.randn(n_tp) * 0.1
                ).astype(np.float32)

            # Normalize each channel
            for ch in range(n_ch):
                mu = signals[i, ch].mean()
                std = signals[i, ch].std() + 1e-8
                signals[i, ch] = (signals[i, ch] - mu) / std

        targets = np.column_stack([
            np.clip(rng.normal(v_mean, 0.06, n), 0, 1),
            np.clip(rng.normal(f_mean, 0.06, n), 0, 1),
            np.clip(rng.normal(d_mean, 0.04, n), 0, 0.5),
        ]).astype(np.float32)

        X_parts.append(signals)
        y_parts.append(targets)
        label_parts.append(np.full(n, stype))

    X = torch.from_numpy(np.concatenate(X_parts, axis=0))
    y = torch.from_numpy(np.concatenate(y_parts, axis=0))
    labels = np.concatenate(label_parts, axis=0)

    return X, y, labels


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

@dataclass
class RegressionMetrics:
    mse: float = 0.0
    mae: float = 0.0
    r2: float = 0.0


@dataclass
class ClassificationMetrics:
    accuracy: float = 0.0
    sensitivity: float = 0.0
    specificity: float = 0.0
    fpr: float = 0.0
    conf_matrix: np.ndarray = field(default_factory=lambda: np.zeros((2, 2), dtype=int))
    fpr_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    tpr_curve: np.ndarray = field(default_factory=lambda: np.array([]))
    auc_score: float = 0.0


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray,
) -> Dict[str, RegressionMetrics]:
    metrics: Dict[str, RegressionMetrics] = {}
    for idx, name in enumerate(OUTPUT_NAMES):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        metrics[name] = RegressionMetrics(
            mse=float(mean_squared_error(yt, yp)),
            mae=float(mean_absolute_error(yt, yp)),
            r2=float(r2_score(yt, yp)) if np.std(yt) > 1e-8 else 0.0,
        )
    return metrics


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, threshold: float = VALVE_THRESHOLD,
) -> ClassificationMetrics:
    true_labels = (y_true[:, 0] >= threshold).astype(int)
    pred_labels = (y_pred[:, 0] >= threshold).astype(int)
    pred_scores = y_pred[:, 0]

    # Need at least 2 classes for confusion matrix
    if len(np.unique(true_labels)) < 2:
        return ClassificationMetrics(accuracy=accuracy_score(true_labels, pred_labels))

    cm = confusion_matrix(true_labels, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(true_labels, pred_labels)
    sens = recall_score(true_labels, pred_labels, pos_label=1, zero_division=0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    fpr_val = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    fpr_c, tpr_c, _ = roc_curve(true_labels, pred_scores)
    auc_val = float(auc(fpr_c, tpr_c))

    return ClassificationMetrics(
        accuracy=acc, sensitivity=sens, specificity=spec,
        fpr=fpr_val, conf_matrix=cm,
        fpr_curve=fpr_c, tpr_curve=tpr_c, auc_score=auc_val,
    )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

PLOT_DIR: Path = Path(__file__).parent / "plots"


def plot_confusion_matrix(cm: np.ndarray, save_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    labels = ["No Stroke", "Stroke"]
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
           ylabel="True Label", xlabel="Predicted Label", title="Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)
    fig.tight_layout()
    path = save_path or PLOT_DIR / "confusion_matrix.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_roc_curve(fpr_c: np.ndarray, tpr_c: np.ndarray, auc_val: float,
                   save_path: Optional[Path] = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr_c, tpr_c, color="darkorange", lw=2, label=f"ROC (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random")
    ax.set(xlim=[-0.02, 1.02], ylim=[-0.02, 1.02],
           xlabel="False Positive Rate", ylabel="True Positive Rate",
           title="ROC Curve -- Stroke Detection")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = save_path or PLOT_DIR / "roc_curve.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_prediction_scatter(y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: Optional[Path] = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, (ax, name) in enumerate(zip(axes, OUTPUT_NAMES)):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        ax.scatter(yt, yp, alpha=0.35, s=10, edgecolors="none")
        lims = [0, 1]
        ax.plot(lims, lims, "r--", lw=1, label="Ideal")
        ax.set(xlim=lims, ylim=lims, xlabel="Actual", ylabel="Predicted", title=name)
        ax.legend(fontsize=8)
    fig.suptitle("Prediction vs Actual", fontsize=13)
    fig.tight_layout()
    path = save_path or PLOT_DIR / "prediction_scatter.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                            save_path: Optional[Path] = None) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for idx, (ax, name) in enumerate(zip(axes, OUTPUT_NAMES)):
        errors = y_pred[:, idx] - y_true[:, idx]
        ax.hist(errors, bins=50, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", lw=1)
        ax.set(xlabel="Prediction Error", ylabel="Count", title=f"{name} Error Dist")
    fig.suptitle("Error Distributions", fontsize=13)
    fig.tight_layout()
    path = save_path or PLOT_DIR / "error_distribution.png"
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Results printer
# ---------------------------------------------------------------------------

def print_results_table(
    overall_reg: Dict[str, RegressionMetrics],
    overall_cls: ClassificationMetrics,
    per_type_reg: Dict[str, Dict[str, RegressionMetrics]],
    per_type_cls: Dict[str, ClassificationMetrics],
) -> None:
    sep = "=" * 80
    thin = "-" * 80

    print(f"\n{sep}")
    print("  MEG STROKE INTERVENTION -- MODEL VALIDATION RESULTS")
    print(sep)

    print("\n  OVERALL REGRESSION METRICS")
    print(thin)
    print(f"  {'Output':<20} {'MSE':>10} {'MAE':>10} {'R2':>10}")
    print(thin)
    for name, m in overall_reg.items():
        print(f"  {name:<20} {m.mse:>10.6f} {m.mae:>10.6f} {m.r2:>10.4f}")

    print(f"\n  OVERALL CLASSIFICATION METRICS  (threshold={VALVE_THRESHOLD})")
    print(thin)
    print(f"  Accuracy:            {overall_cls.accuracy:.4f}")
    print(f"  Sensitivity:         {overall_cls.sensitivity:.4f}")
    print(f"  Specificity:         {overall_cls.specificity:.4f}")
    print(f"  False Positive Rate: {overall_cls.fpr:.4f}  "
          f"{'*** SAFETY CRITICAL ***' if overall_cls.fpr > 0.05 else '(OK)'}")
    print(f"  AUC:                 {overall_cls.auc_score:.4f}")

    if overall_cls.conf_matrix.sum() > 0:
        print(f"\n  Confusion Matrix:")
        print(f"                 Pred No-Stroke   Pred Stroke")
        print(f"  True No-Stroke   {overall_cls.conf_matrix[0, 0]:>8}        "
              f"{overall_cls.conf_matrix[0, 1]:>8}")
        print(f"  True Stroke      {overall_cls.conf_matrix[1, 0]:>8}        "
              f"{overall_cls.conf_matrix[1, 1]:>8}")

    for stype in STROKE_TYPES:
        if stype not in per_type_reg:
            continue
        print(f"\n  STROKE TYPE: {stype.upper()}")
        print(thin)
        print(f"  {'Output':<20} {'MSE':>10} {'MAE':>10} {'R2':>10}")
        print(thin)
        for name, m in per_type_reg[stype].items():
            print(f"  {name:<20} {m.mse:>10.6f} {m.mae:>10.6f} {m.r2:>10.4f}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

def load_model(model_path: Path, device: torch.device) -> MEGStrokeNet:
    """Load a trained MEGStrokeNet from a .pth checkpoint."""
    model = MEGStrokeNet().to(device)
    if model_path.exists():
        state = torch.load(str(model_path), map_location=device, weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"  Loaded model from {model_path}")
    else:
        print(f"  WARNING: {model_path} not found -- using untrained model for demo")
    model.eval()
    return model


def run_validation(
    model_path: Optional[Path] = None,
    n_per_type: int = 200,
    seed: int = 42,
    device_name: str = "cpu",
) -> Dict:
    """Execute the full validation pipeline."""
    project_root = Path(__file__).parent.parent
    if model_path is None:
        model_path = project_root / "models" / "trained_model.pth"

    device = torch.device(device_name)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading model ...")
    model = load_model(model_path, device)

    print("[2/5] Generating test data ...")
    X, y_true_t, stroke_labels = generate_test_data(n_per_type=n_per_type, seed=seed)
    X = X.to(device)
    y_true = y_true_t.numpy()

    print("[3/5] Running inference ...")
    with torch.no_grad():
        y_pred_t = model(X)
    y_pred = y_pred_t.cpu().numpy()

    print("[4/5] Computing metrics ...")
    overall_reg = compute_regression_metrics(y_true, y_pred)
    overall_cls = compute_classification_metrics(y_true, y_pred)

    per_type_reg: Dict[str, Dict[str, RegressionMetrics]] = {}
    per_type_cls: Dict[str, ClassificationMetrics] = {}
    for stype in STROKE_TYPES:
        mask = stroke_labels == stype
        if mask.sum() < 2:
            continue
        per_type_reg[stype] = compute_regression_metrics(y_true[mask], y_pred[mask])
        per_type_cls[stype] = compute_classification_metrics(y_true[mask], y_pred[mask])

    print("[5/5] Generating plots ...")
    plot_confusion_matrix(overall_cls.conf_matrix)
    plot_roc_curve(overall_cls.fpr_curve, overall_cls.tpr_curve, overall_cls.auc_score)
    plot_prediction_scatter(y_true, y_pred)
    plot_error_distribution(y_true, y_pred)

    print_results_table(overall_reg, overall_cls, per_type_reg, per_type_cls)

    summary: Dict = {
        "overall_regression": {
            name: {"mse": m.mse, "mae": m.mae, "r2": m.r2}
            for name, m in overall_reg.items()
        },
        "overall_classification": {
            "accuracy": overall_cls.accuracy,
            "sensitivity": overall_cls.sensitivity,
            "specificity": overall_cls.specificity,
            "fpr": overall_cls.fpr,
            "auc": overall_cls.auc_score,
        },
    }
    summary_path = PLOT_DIR / "validation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved metrics summary: {summary_path}\n")

    return summary


if __name__ == "__main__":
    run_validation()
