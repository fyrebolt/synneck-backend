"""
Training pipeline for MEGStrokeNet.

Provides:
  - Custom weighted MSE + safety-penalty loss function
  - Full training loop with early stopping, checkpointing,
    gradient clipping, and cosine-annealing LR schedule
  - Optional k-fold cross-validation
  - Training-curve visualisation saved to PNG
  - Standalone __main__ that generates synthetic data, trains,
    and saves the model.

Usage:
    python models/training.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so sibling packages resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import copy
import json
import math
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend -- safe for headless servers.
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from models.meg_stroke_net import MEGStrokeNet


# ---------------------------------------------------------------------------
# Synthetic dataset (self-contained fallback)
# ---------------------------------------------------------------------------

class MEGStrokeDataset(Dataset):
    """
    Synthetic MEG stroke dataset.

    Each sample is a 6-channel x 100-timepoint window paired with a
    3-element target vector [valve_extension, force_magnitude, trigger_delay].

    The generator tries to import ``data.synthetic_generator`` and
    ``data.data_loader`` from the project.  If those modules are not yet
    implemented it falls back to a built-in generator that produces
    plausible training data.

    Args:
        num_samples: Number of samples to generate.
        seed: Random seed for reproducibility.
        augment: Whether to apply random data augmentation.
    """

    def __init__(
        self,
        num_samples: int = 5000,
        seed: int = 42,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.augment = augment

        data, targets = self._generate_or_load(num_samples, seed)
        self.data = torch.as_tensor(data, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Data loading / generation
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_or_load(
        num_samples: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Try project loaders first, fall back to built-in synthetic data."""
        try:
            from data.synthetic_generator import generate_synthetic_data  # type: ignore[import-untyped]

            data, targets = generate_synthetic_data(num_samples=num_samples, seed=seed)
            print("[data] Loaded synthetic data from data.synthetic_generator")
            return np.asarray(data), np.asarray(targets)
        except (ImportError, AttributeError):
            pass

        try:
            from data.data_loader import load_meg_data  # type: ignore[import-untyped]

            data, targets = load_meg_data(num_samples=num_samples)
            print("[data] Loaded data from data.data_loader")
            return np.asarray(data), np.asarray(targets)
        except (ImportError, AttributeError):
            pass

        # Built-in fallback generator
        print("[data] Using built-in synthetic data generator")
        return MEGStrokeDataset._builtin_generate(num_samples, seed)

    @staticmethod
    def _builtin_generate(
        num_samples: int, seed: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate plausible synthetic MEG stroke data.

        Three stroke-severity classes are sampled roughly equally:
          - *no stroke*   : low-amplitude signals, target ~ [0, 0, 0]
          - *mild stroke* : moderate spike patterns, target ~ [0.3, 0.4, 0.5]
          - *severe stroke*: large correlated bursts, target ~ [0.8, 0.9, 0.2]

        Returns:
            (data, targets) arrays of shapes (N, 6, 100) and (N, 3).
        """
        rng = np.random.RandomState(seed)

        n_channels = MEGStrokeNet.NUM_CHANNELS
        n_timepoints = MEGStrokeNet.NUM_TIMEPOINTS

        data = np.zeros((num_samples, n_channels, n_timepoints), dtype=np.float32)
        targets = np.zeros((num_samples, MEGStrokeNet.NUM_OUTPUTS), dtype=np.float32)

        for i in range(num_samples):
            severity = rng.choice(3)  # 0=none, 1=mild, 2=severe

            # Base signal: band-limited noise resembling MEG background.
            t = np.linspace(0, 0.5, n_timepoints)
            base_freqs = rng.uniform(4, 30, size=(n_channels, 3))
            signal = np.zeros((n_channels, n_timepoints), dtype=np.float32)
            for ch in range(n_channels):
                for freq in base_freqs[ch]:
                    phase = rng.uniform(0, 2 * np.pi)
                    signal[ch] += np.sin(2 * np.pi * freq * t + phase).astype(
                        np.float32
                    )

            noise_scale = 0.1 + 0.1 * severity
            signal += rng.randn(n_channels, n_timepoints).astype(np.float32) * noise_scale

            if severity == 0:
                # Normal -- no stroke signature
                targets[i] = [
                    rng.uniform(0.0, 0.05),
                    rng.uniform(0.0, 0.05),
                    rng.uniform(0.0, 0.1),
                ]
            elif severity == 1:
                # Mild stroke -- moderate burst
                onset = rng.randint(30, 70)
                width = rng.randint(5, 15)
                amplitude = rng.uniform(2.0, 4.0)
                for ch in range(n_channels):
                    signal[ch, onset : onset + width] += amplitude * rng.randn(
                        min(width, n_timepoints - onset)
                    ).astype(np.float32)
                targets[i] = [
                    rng.uniform(0.2, 0.5),
                    rng.uniform(0.3, 0.6),
                    rng.uniform(0.3, 0.7),
                ]
            else:
                # Severe stroke -- large correlated burst across channels
                onset = rng.randint(20, 60)
                width = rng.randint(15, 30)
                amplitude = rng.uniform(5.0, 10.0)
                shared_burst = (
                    amplitude
                    * np.exp(-0.5 * ((np.arange(n_timepoints) - onset) / (width / 3)) ** 2)
                ).astype(np.float32)
                for ch in range(n_channels):
                    signal[ch] += shared_burst * (1 + 0.2 * rng.randn())
                targets[i] = [
                    rng.uniform(0.7, 1.0),
                    rng.uniform(0.7, 1.0),
                    rng.uniform(0.05, 0.3),
                ]

            # Normalise each channel to roughly zero mean, unit variance.
            for ch in range(n_channels):
                mu = signal[ch].mean()
                std = signal[ch].std() + 1e-8
                signal[ch] = (signal[ch] - mu) / std

            data[i] = signal

        return data, targets

    # ------------------------------------------------------------------
    # Augmentation
    # ------------------------------------------------------------------

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply simple online augmentations to a single sample."""
        # Additive Gaussian noise
        if torch.rand(1).item() < 0.5:
            x = x + 0.05 * torch.randn_like(x)
        # Random time shift (circular)
        if torch.rand(1).item() < 0.5:
            shift = torch.randint(-10, 11, (1,)).item()
            x = torch.roll(x, shifts=int(shift), dims=-1)
        # Random channel dropout
        if torch.rand(1).item() < 0.3:
            ch = torch.randint(0, x.shape[0], (1,)).item()
            x[ch] = 0.0
        # Random amplitude scaling
        if torch.rand(1).item() < 0.5:
            scale = 0.8 + 0.4 * torch.rand(1).item()
            x = x * scale
        return x

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx]
        if self.augment:
            x = self._augment(x.clone())
        return x, self.targets[idx]


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class StrokeInterventionLoss(nn.Module):
    """
    Weighted MSE loss with a safety penalty for false positives.

    Components:
      1. **Weighted MSE** -- ``valve_extension`` gets ``valve_weight`` (2x)
         importance; the other two outputs are weighted at 1x.
      2. **Safety penalty** -- When the true ``valve_extension`` is near zero
         (< ``safety_threshold``) but the prediction is high, an extra
         quadratic penalty is applied.  This discourages the model from
         actuating the solenoid valve when no stroke is occurring.

    Args:
        valve_weight: Extra weight for the valve_extension output.
        safety_penalty_scale: Multiplier for the false-positive penalty.
        safety_threshold: Target valve_extension values below this are
                          considered "should not actuate".
    """

    def __init__(
        self,
        valve_weight: float = 2.0,
        safety_penalty_scale: float = 5.0,
        safety_threshold: float = 0.1,
    ) -> None:
        super().__init__()
        self.valve_weight = valve_weight
        self.safety_penalty_scale = safety_penalty_scale
        self.safety_threshold = safety_threshold

    def forward(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the composite loss.

        Args:
            predictions: (batch, 3) model outputs.
            targets:     (batch, 3) ground-truth labels.

        Returns:
            Scalar loss tensor.
        """
        # Per-output weights: [valve_extension=2, force_magnitude=1, trigger_delay=1]
        weights = torch.tensor(
            [self.valve_weight, 1.0, 1.0],
            device=predictions.device,
            dtype=predictions.dtype,
        )

        # Weighted MSE
        mse = ((predictions - targets) ** 2) * weights.unsqueeze(0)
        weighted_mse = mse.mean()

        # Safety penalty: penalise high valve predictions when target is ~0.
        valve_pred = predictions[:, 0]
        valve_target = targets[:, 0]

        # Mask samples where target indicates "no actuation needed".
        no_stroke_mask = (valve_target < self.safety_threshold).float()

        # Penalty proportional to predicted valve_extension^2 for those cases.
        false_positive_penalty = (no_stroke_mask * valve_pred ** 2).mean()

        total_loss = weighted_mse + self.safety_penalty_scale * false_positive_penalty
        return total_loss


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """
    Build training and validation DataLoaders.

    Args:
        train_dataset: Training split.
        val_dataset: Validation split.
        batch_size: Mini-batch size.
        num_workers: Parallel data loading workers.

    Returns:
        (train_loader, val_loader) tuple.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(
    model: MEGStrokeNet,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> float:
    """
    Execute one training epoch.

    Args:
        model: The network.
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimiser instance.
        device: Torch device.
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        Mean training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: MEGStrokeNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """
    Evaluate on the validation set.

    Args:
        model: The network.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: Torch device.

    Returns:
        Mean validation loss.
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        preds = model(x_batch)
        loss = criterion(preds, y_batch)

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Main training driver
# ---------------------------------------------------------------------------

def train(
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 15,
    max_grad_norm: float = 1.0,
    num_samples: int = 5000,
    val_fraction: float = 0.2,
    seed: int = 42,
    save_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Full training run: generate data, train, checkpoint, plot.

    Args:
        num_epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        learning_rate: Initial learning rate for Adam.
        patience: Early-stopping patience (epochs without improvement).
        max_grad_norm: Gradient clipping threshold.
        num_samples: Number of synthetic training samples to generate.
        val_fraction: Fraction of data used for validation.
        seed: Random seed.
        save_dir: Directory for model and plot artefacts.
        device: Torch device; auto-detected if ``None``.

    Returns:
        Dictionary of final metrics and paths.
    """
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---- Data ----
    print("\n[1/4] Generating synthetic data ...")
    full_dataset = MEGStrokeDataset(num_samples=num_samples, seed=seed, augment=False)
    n_val = int(len(full_dataset) * val_fraction)
    n_train = len(full_dataset) - n_val

    indices = torch.randperm(len(full_dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = MEGStrokeDataset(num_samples=num_samples, seed=seed, augment=True)
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)

    train_loader, val_loader = create_dataloaders(
        train_subset, val_subset, batch_size=batch_size
    )
    print(f"  Train samples: {len(train_subset)}  |  Val samples: {len(val_subset)}")

    # ---- Model / Criterion / Optimiser ----
    print("\n[2/4] Building model ...")
    model = MEGStrokeNet().to(device)
    print(f"  Parameters: {model.count_parameters():,}")

    criterion = StrokeInterventionLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # ---- Training loop ----
    print(f"\n[3/4] Training for up to {num_epochs} epochs (patience={patience}) ...")
    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    train_losses: list[float] = []
    val_losses: list[float] = []

    progress = tqdm(range(1, num_epochs + 1), desc="Training", unit="epoch")
    for epoch in progress:
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, max_grad_norm
        )
        val_loss = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        current_lr = optimizer.param_groups[0]["lr"]
        progress.set_postfix(
            train_loss=f"{train_loss:.4f}",
            val_loss=f"{val_loss:.4f}",
            best_val=f"{best_val_loss:.4f}",
            lr=f"{current_lr:.2e}",
        )

        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\n  Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
            break

    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ---- Save artefacts ----
    print("\n[4/4] Saving artefacts ...")
    model_path = save_dir / "trained_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "architecture": "MEGStrokeNet",
            "num_parameters": model.count_parameters(),
            "best_val_loss": best_val_loss,
            "num_epochs_trained": len(train_losses),
            "training_config": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "patience": patience,
                "max_grad_norm": max_grad_norm,
                "num_samples": num_samples,
                "val_fraction": val_fraction,
                "seed": seed,
            },
        },
        model_path,
    )
    print(f"  Model saved to: {model_path}")

    # Training curves
    plot_path = save_dir / "training_curves.png"
    _plot_training_curves(train_losses, val_losses, plot_path)
    print(f"  Training curves saved to: {plot_path}")

    # ---- Final metrics ----
    final_metrics = _compute_final_metrics(model, val_loader, criterion, device)
    print("\n--- Final Metrics ---")
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.6f}")

    return {
        "model_path": str(model_path),
        "plot_path": str(plot_path),
        "best_val_loss": best_val_loss,
        "num_epochs_trained": len(train_losses),
        "final_metrics": final_metrics,
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate(
    k: int = 5,
    num_epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    patience: int = 15,
    max_grad_norm: float = 1.0,
    num_samples: int = 5000,
    seed: int = 42,
    device: Optional[torch.device] = None,
) -> dict:
    """
    K-fold cross-validation.

    Args:
        k: Number of folds.
        num_epochs: Max epochs per fold.
        batch_size: Mini-batch size.
        learning_rate: Initial LR.
        patience: Early-stopping patience.
        max_grad_norm: Gradient clip norm.
        num_samples: Synthetic dataset size.
        seed: Random seed.
        device: Torch device.

    Returns:
        Dictionary with per-fold and aggregate results.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"\n{'=' * 60}")
    print(f"  {k}-Fold Cross-Validation")
    print(f"{'=' * 60}")

    full_dataset = MEGStrokeDataset(num_samples=num_samples, seed=seed, augment=False)
    augmented_dataset = MEGStrokeDataset(num_samples=num_samples, seed=seed, augment=True)
    n = len(full_dataset)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    fold_size = n // k

    fold_results: list[dict] = []

    for fold_idx in range(k):
        print(f"\n--- Fold {fold_idx + 1}/{k} ---")

        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < k - 1 else n
        val_indices = indices[val_start:val_end]
        train_indices = indices[:val_start] + indices[val_end:]

        train_subset = Subset(augmented_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        train_loader, val_loader = create_dataloaders(
            train_subset, val_subset, batch_size=batch_size
        )

        model = MEGStrokeNet().to(device)
        criterion = StrokeInterventionLoss()
        optimizer = Adam(model.parameters(), lr=learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

        best_val_loss = float("inf")
        best_state = None
        wait = 0

        progress = tqdm(
            range(1, num_epochs + 1),
            desc=f"Fold {fold_idx + 1}",
            unit="epoch",
            leave=False,
        )
        for epoch in progress:
            t_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, max_grad_norm
            )
            v_loss = validate(model, val_loader, criterion, device)
            scheduler.step()

            if v_loss < best_val_loss:
                best_val_loss = v_loss
                best_state = copy.deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1

            progress.set_postfix(val=f"{v_loss:.4f}", best=f"{best_val_loss:.4f}")

            if wait >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        metrics = _compute_final_metrics(model, val_loader, criterion, device)
        metrics["best_val_loss"] = best_val_loss
        fold_results.append(metrics)

        print(f"  Fold {fold_idx + 1} best val loss: {best_val_loss:.6f}")

    # Aggregate
    all_val_losses = [r["best_val_loss"] for r in fold_results]
    summary = {
        "k": k,
        "fold_results": fold_results,
        "mean_val_loss": float(np.mean(all_val_losses)),
        "std_val_loss": float(np.std(all_val_losses)),
    }

    print(f"\n{'=' * 60}")
    print(f"  CV Summary: val_loss = {summary['mean_val_loss']:.6f} "
          f"+/- {summary['std_val_loss']:.6f}")
    print(f"{'=' * 60}")

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_final_metrics(
    model: MEGStrokeNet,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Compute loss and per-output MAE on a DataLoader."""
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0.0
    count = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        preds = model(x_batch)
        total_loss += criterion(preds, y_batch).item()
        count += 1

        all_preds.append(preds.cpu())
        all_targets.append(y_batch.cpu())

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    mae_per_output = (preds_cat - targets_cat).abs().mean(dim=0)

    metrics: dict[str, float] = {"loss": total_loss / max(count, 1)}
    for i, name in enumerate(MEGStrokeNet.OUTPUT_NAMES):
        metrics[f"mae_{name}"] = mae_per_output[i].item()

    return metrics


def _plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Path,
) -> None:
    """Save a training / validation loss plot to *save_path*."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = list(range(1, len(train_losses) + 1))
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=2, color="#2196F3")
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=2, color="#F44336")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("MEGStrokeNet Training Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  MEGStrokeNet Training Pipeline")
    print("=" * 60)

    results = train(
        num_epochs=100,
        batch_size=64,
        learning_rate=1e-3,
        patience=15,
        max_grad_norm=1.0,
        num_samples=5000,
        val_fraction=0.2,
        seed=42,
    )

    print("\n" + "=" * 60)
    print("  Training Complete")
    print("=" * 60)
    print(f"  Model:          {results['model_path']}")
    print(f"  Curves:         {results['plot_path']}")
    print(f"  Best val loss:  {results['best_val_loss']:.6f}")
    print(f"  Epochs trained: {results['num_epochs_trained']}")
    print(f"  Final metrics:")
    for k, v in results["final_metrics"].items():
        print(f"    {k}: {v:.6f}")
    print("=" * 60)
