"""
PyTorch DataLoader for the MEG Stroke Intervention project.

Provides a custom Dataset class that loads MEG data from synthetic and/or real
sources, applies data augmentation, and returns tensors in the expected format:
    - Input:  (batch, channels=6, timepoints=100)
    - Labels: (batch, 3)  [valve_extension, force_magnitude, trigger_delay]

Supports train/val/test splits with configurable augmentation for training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# Expected tensor dimensions
NUM_CHANNELS: int = 6
NUM_TIMEPOINTS: int = 100
NUM_LABELS: int = 3

# Project paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR: Path = PROJECT_ROOT / "data" / "synthetic"
PROCESSED_DIR: Path = PROJECT_ROOT / "data" / "processed"


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation applied during training.

    Attributes:
        enabled: Master switch for all augmentation.
        time_shift_max: Maximum samples to shift in time (both directions).
        amplitude_scale_range: (min, max) scaling factor for amplitude.
        gaussian_noise_std: Standard deviation of additive Gaussian noise.
        channel_dropout_prob: Probability of zeroing out a random channel.
        time_mask_max: Maximum timepoints to mask (set to zero).
    """

    enabled: bool = True
    time_shift_max: int = 5
    amplitude_scale_range: Tuple[float, float] = (0.8, 1.2)
    gaussian_noise_std: float = 0.05
    channel_dropout_prob: float = 0.05
    time_mask_max: int = 10


class MEGStrokeDataset(Dataset):
    """PyTorch Dataset for MEG stroke intervention data.

    Loads data from .npz files (synthetic or real) and applies optional
    augmentation. Each sample consists of a 6-channel, 100-timepoint
    MEG window and a 3-element valve control label.

    Args:
        data: MEG data array, shape (n_samples, 6, 100).
        labels: Valve control labels, shape (n_samples, 3).
        conditions: Condition indices, shape (n_samples,). Optional.
        augmentation: Augmentation config. Set enabled=False for val/test.
        transform: Optional callable applied to each (data, label) pair.
    """

    def __init__(
        self,
        data: NDArray[np.float64],
        labels: NDArray[np.float64],
        conditions: Optional[NDArray[np.int64]] = None,
        augmentation: Optional[AugmentationConfig] = None,
        transform: Optional[callable] = None,
    ) -> None:
        super().__init__()

        assert data.ndim == 3, f"Expected 3D data, got shape {data.shape}"
        assert data.shape[1] == NUM_CHANNELS, (
            f"Expected {NUM_CHANNELS} channels, got {data.shape[1]}"
        )
        assert data.shape[2] == NUM_TIMEPOINTS, (
            f"Expected {NUM_TIMEPOINTS} timepoints, got {data.shape[2]}"
        )
        assert labels.shape[0] == data.shape[0], (
            f"Data/label count mismatch: {data.shape[0]} vs {labels.shape[0]}"
        )
        assert labels.shape[1] == NUM_LABELS, (
            f"Expected {NUM_LABELS} labels, got {labels.shape[1]}"
        )

        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.float32)
        self.conditions = conditions
        self.augmentation = augmentation or AugmentationConfig(enabled=False)
        self.transform = transform
        self._rng = np.random.default_rng()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of:
                - data: tensor of shape (6, 100)
                - labels: tensor of shape (3,)
        """
        x = self.data[idx].copy()  # (6, 100)
        y = self.labels[idx].copy()  # (3,)

        # Apply augmentation if enabled
        if self.augmentation.enabled:
            x = self._augment(x)

        # Apply custom transform if provided
        if self.transform is not None:
            x, y = self.transform(x, y)

        return torch.from_numpy(x), torch.from_numpy(y)

    def _augment(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply random augmentations to a single sample.

        Args:
            x: Input data, shape (6, 100).

        Returns:
            Augmented data, same shape.
        """
        rng = self._rng

        # 1. Time shifting: circular shift along time axis
        if self.augmentation.time_shift_max > 0:
            shift = rng.integers(
                -self.augmentation.time_shift_max,
                self.augmentation.time_shift_max + 1,
            )
            if shift != 0:
                x = np.roll(x, shift, axis=-1)

        # 2. Amplitude scaling: multiply all channels by a random factor
        scale_lo, scale_hi = self.augmentation.amplitude_scale_range
        scale = rng.uniform(scale_lo, scale_hi)
        x = x * scale

        # 3. Gaussian noise: add independent noise to each sample
        if self.augmentation.gaussian_noise_std > 0:
            noise = rng.normal(
                0.0, self.augmentation.gaussian_noise_std, size=x.shape
            ).astype(np.float32)
            x = x + noise

        # 4. Channel dropout: zero out a random channel
        if rng.random() < self.augmentation.channel_dropout_prob:
            ch = rng.integers(0, NUM_CHANNELS)
            x[ch, :] = 0.0

        # 5. Time masking: zero out a contiguous block of timepoints
        if self.augmentation.time_mask_max > 0 and rng.random() < 0.1:
            mask_len = rng.integers(1, self.augmentation.time_mask_max + 1)
            start = rng.integers(0, NUM_TIMEPOINTS - mask_len)
            x[:, start : start + mask_len] = 0.0

        return x

    def get_condition(self, idx: int) -> Optional[int]:
        """Get the condition index for a sample.

        Args:
            idx: Sample index.

        Returns:
            Condition index (0=healthy, 1=acute, 2=chronic) or None.
        """
        if self.conditions is not None:
            return int(self.conditions[idx])
        return None


def load_split_from_npz(
    filepath: Path,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], Optional[NDArray[np.int64]]]:
    """Load a single data split from an .npz file.

    Args:
        filepath: Path to the .npz file.

    Returns:
        Tuple of (data, labels, conditions). Conditions may be None.
    """
    loaded = np.load(filepath)
    data = loaded["data"]
    labels = loaded["labels"]
    conditions = loaded.get("conditions", None)
    if conditions is not None:
        conditions = conditions.astype(np.int64) if conditions.dtype != np.int64 else conditions

    logger.info(
        f"Loaded {filepath.name}: data={data.shape}, labels={labels.shape}"
    )
    return data, labels, conditions


def _merge_data_sources(
    sources: List[Tuple[NDArray, NDArray, Optional[NDArray]]],
) -> Tuple[NDArray, NDArray, Optional[NDArray]]:
    """Merge multiple data sources into a single dataset.

    Args:
        sources: List of (data, labels, conditions) tuples.

    Returns:
        Merged (data, labels, conditions).
    """
    all_data = []
    all_labels = []
    all_conditions = []
    has_conditions = True

    for data, labels, conds in sources:
        all_data.append(data)
        all_labels.append(labels)
        if conds is not None:
            all_conditions.append(conds)
        else:
            has_conditions = False

    merged_data = np.concatenate(all_data, axis=0)
    merged_labels = np.concatenate(all_labels, axis=0)
    merged_conditions = (
        np.concatenate(all_conditions, axis=0) if has_conditions else None
    )

    return merged_data, merged_labels, merged_conditions


def create_datasets(
    synthetic_dir: Optional[Path] = None,
    real_dir: Optional[Path] = None,
    augmentation: Optional[AugmentationConfig] = None,
) -> Dict[str, MEGStrokeDataset]:
    """Create train/val/test datasets from available data sources.

    Loads data from synthetic and/or real (MNE-derived) sources and creates
    PyTorch datasets with appropriate augmentation settings.

    Args:
        synthetic_dir: Path to directory with synthetic train/val/test .npz files.
        real_dir: Path to directory with real (processed) .npz files.
        augmentation: Augmentation config for training set. Val/test are never
            augmented.

    Returns:
        Dictionary mapping 'train', 'val', 'test' to MEGStrokeDataset instances.

    Raises:
        FileNotFoundError: If no data sources are available.
    """
    synthetic_dir = synthetic_dir or SYNTHETIC_DIR
    real_dir = real_dir or PROCESSED_DIR

    if augmentation is None:
        augmentation = AugmentationConfig(enabled=True)

    no_aug = AugmentationConfig(enabled=False)

    datasets: Dict[str, MEGStrokeDataset] = {}

    for split in ["train", "val", "test"]:
        sources: List[Tuple[NDArray, NDArray, Optional[NDArray]]] = []

        # Try to load synthetic data
        synth_path = synthetic_dir / f"{split}.npz"
        if synth_path.exists():
            sources.append(load_split_from_npz(synth_path))

        # Try to load real data
        real_path = real_dir / f"{split}.npz"
        if real_path.exists():
            sources.append(load_split_from_npz(real_path))

        if not sources:
            raise FileNotFoundError(
                f"No data found for split '{split}'. "
                f"Checked: {synth_path}, {real_path}. "
                f"Run download_data.py first."
            )

        data, labels, conditions = _merge_data_sources(sources)

        # Only augment training data
        split_aug = augmentation if split == "train" else no_aug

        datasets[split] = MEGStrokeDataset(
            data=data,
            labels=labels,
            conditions=conditions,
            augmentation=split_aug,
        )

        logger.info(
            f"Created {split} dataset: {len(datasets[split])} samples "
            f"(augmentation={'ON' if split_aug.enabled else 'OFF'})"
        )

    return datasets


def create_dataloaders(
    datasets: Optional[Dict[str, MEGStrokeDataset]] = None,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = True,
    synthetic_dir: Optional[Path] = None,
    real_dir: Optional[Path] = None,
    augmentation: Optional[AugmentationConfig] = None,
) -> Dict[str, DataLoader]:
    """Create PyTorch DataLoaders for train/val/test splits.

    Either pass pre-created datasets or provide directories to load from.

    Args:
        datasets: Pre-created datasets dict. If None, creates from directories.
        batch_size: Number of samples per batch.
        num_workers: Number of parallel data loading workers.
        pin_memory: Pin memory for GPU transfer efficiency.
        synthetic_dir: Path to synthetic data (used if datasets is None).
        real_dir: Path to real data (used if datasets is None).
        augmentation: Augmentation config (used if datasets is None).

    Returns:
        Dictionary mapping 'train', 'val', 'test' to DataLoader instances.
    """
    if datasets is None:
        datasets = create_datasets(
            synthetic_dir=synthetic_dir,
            real_dir=real_dir,
            augmentation=augmentation,
        )

    loaders: Dict[str, DataLoader] = {}

    for split, dataset in datasets.items():
        shuffle = split == "train"
        drop_last = split == "train" and len(dataset) > batch_size

        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=drop_last,
        )

        logger.info(
            f"Created {split} DataLoader: "
            f"{len(dataset)} samples, "
            f"batch_size={batch_size}, "
            f"shuffle={shuffle}, "
            f"num_workers={num_workers}, "
            f"batches={len(loaders[split])}"
        )

    return loaders


def create_dataloaders_from_generated(
    n_samples: int = 5400,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
    augmentation: Optional[AugmentationConfig] = None,
) -> Dict[str, DataLoader]:
    """Generate synthetic data on-the-fly and create DataLoaders.

    Convenience function for quick prototyping that does not require
    pre-saved data files.

    Args:
        n_samples: Total number of samples to generate.
        batch_size: Batch size for DataLoaders.
        num_workers: Number of data loading workers.
        seed: Random seed.
        augmentation: Augmentation config for training.

    Returns:
        Dictionary of DataLoaders for train/val/test.
    """
    from data.synthetic_generator import generate_dataset

    dataset = generate_dataset(n_samples=n_samples, seed=seed)

    # Split indices
    rng = np.random.default_rng(seed)
    n = len(dataset["data"])
    indices = rng.permutation(n)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    splits_idx = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    if augmentation is None:
        augmentation = AugmentationConfig(enabled=True)

    no_aug = AugmentationConfig(enabled=False)

    datasets: Dict[str, MEGStrokeDataset] = {}
    for split, idx in splits_idx.items():
        split_aug = augmentation if split == "train" else no_aug
        datasets[split] = MEGStrokeDataset(
            data=dataset["data"][idx],
            labels=dataset["labels"][idx],
            conditions=dataset["conditions"][idx],
            augmentation=split_aug,
        )

    return create_dataloaders(
        datasets=datasets,
        batch_size=batch_size,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== MEG Stroke DataLoader Test ===")

    # Test 1: Create dataloaders from on-the-fly generated data
    logger.info("\n--- Test 1: On-the-fly generation ---")
    loaders = create_dataloaders_from_generated(
        n_samples=1000,
        batch_size=32,
        num_workers=0,
        seed=42,
    )

    for split, loader in loaders.items():
        batch_data, batch_labels = next(iter(loader))
        logger.info(
            f"  {split}: data={batch_data.shape}, labels={batch_labels.shape}, "
            f"dtype={batch_data.dtype}"
        )
        assert batch_data.shape == (32 if split == "train" else min(32, len(loader.dataset)), NUM_CHANNELS, NUM_TIMEPOINTS), (
            f"Unexpected data shape: {batch_data.shape}"
        )
        assert batch_labels.shape[1] == NUM_LABELS, (
            f"Unexpected label shape: {batch_labels.shape}"
        )

    # Test 2: Check augmentation effects
    logger.info("\n--- Test 2: Augmentation verification ---")
    train_dataset = loaders["train"].dataset
    assert isinstance(train_dataset, MEGStrokeDataset)

    # Get the same sample twice -- augmentation should make them different
    x1, y1 = train_dataset[0]
    x2, y2 = train_dataset[0]
    are_different = not torch.allclose(x1, x2, atol=1e-6)
    logger.info(f"  Same index, different outputs (augmentation): {are_different}")

    # Val/test should be deterministic
    val_dataset = loaders["val"].dataset
    assert isinstance(val_dataset, MEGStrokeDataset)
    v1, _ = val_dataset[0]
    v2, _ = val_dataset[0]
    are_same = torch.allclose(v1, v2, atol=1e-6)
    logger.info(f"  Val deterministic (no augmentation): {are_same}")

    # Test 3: Check label ranges
    logger.info("\n--- Test 3: Label range verification ---")
    all_labels = []
    for _, labels in loaders["train"]:
        all_labels.append(labels)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    logger.info(f"  Label ranges:")
    label_names = ["valve_extension", "force_magnitude", "trigger_delay"]
    for i, name in enumerate(label_names):
        col = all_labels_tensor[:, i]
        logger.info(
            f"    {name}: [{col.min():.4f}, {col.max():.4f}], "
            f"mean={col.mean():.4f}"
        )

    # Test 4: Try loading from saved files (if they exist)
    logger.info("\n--- Test 4: File-based loading ---")
    try:
        file_datasets = create_datasets()
        logger.info("  Successfully loaded from saved files")
        for split, ds in file_datasets.items():
            logger.info(f"    {split}: {len(ds)} samples")
    except FileNotFoundError as e:
        logger.info(f"  No saved files found (expected if first run): {e}")
        logger.info("  Run download_data.py first to generate saved data")

    logger.info("\n=== DataLoader Test Complete ===")
