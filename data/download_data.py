"""
MEG dataset downloader for the MEG Stroke Intervention project.

Attempts to download real MEG data from the MNE sample dataset (which is small
and reliable). Falls back to synthetic data generation if downloads fail.
All downloads are cached locally in data/raw/ to avoid repeated transfers.

Usage:
    python download_data.py              # Download MNE sample + generate synthetic
    python download_data.py --synthetic  # Only generate synthetic data
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
SYNTHETIC_DIR: Path = DATA_DIR / "synthetic"


def _ensure_dirs() -> None:
    """Create all required data directories."""
    for d in [RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured directory: {d}")


def download_mne_sample(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Optional[Path]:
    """Download and extract motor-cortex channels from the MNE sample dataset.

    The MNE sample dataset (~1.5GB) contains MEG recordings with motor tasks.
    We extract the relevant motor cortex channels and convert to our standard
    format (6 channels, 200Hz, 500ms windows).

    Args:
        output_dir: Directory to cache the raw download. Defaults to data/raw/.
        force: If True, re-download even if cached.

    Returns:
        Path to the extracted .npz file, or None if download failed.
    """
    output_dir = output_dir or RAW_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_file = output_dir / "mne_sample_motor.npz"

    if cache_file.exists() and not force:
        logger.info(f"MNE sample data already cached: {cache_file}")
        return cache_file

    try:
        import mne

        logger.info("Downloading MNE sample dataset (this may take a few minutes)...")
        logger.info("This is a ~1.5GB download but only needs to happen once.")

        # Download the sample dataset
        sample_data_path = mne.datasets.sample.data_path(
            verbose=True
        )
        raw_fname = sample_data_path / "MEG" / "sample" / "sample_audvis_raw.fif"

        if not raw_fname.exists():
            logger.error(f"MNE sample raw file not found: {raw_fname}")
            return None

        logger.info(f"Loading raw data from: {raw_fname}")
        raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

        # Find motor cortex-adjacent channels
        # MNE sample dataset uses Neuromag system; we pick channels near
        # the motor strip (C3/C4 equivalent positions)
        all_ch_names = raw.ch_names
        meg_channels = [ch for ch in all_ch_names if ch.startswith("MEG")]

        # Select 6 channels distributed across left and right motor areas
        # Neuromag channels near motor cortex: ~0241-0243 (left), ~1311-1313 (right)
        target_channels = []
        left_candidates = [
            ch for ch in meg_channels
            if any(prefix in ch for prefix in ["MEG 023", "MEG 024", "MEG 025"])
        ]
        right_candidates = [
            ch for ch in meg_channels
            if any(prefix in ch for prefix in ["MEG 130", "MEG 131", "MEG 132"])
        ]

        # Pick 3 from each hemisphere; fall back to first 6 MEG if not found
        if len(left_candidates) >= 3 and len(right_candidates) >= 3:
            target_channels = left_candidates[:3] + right_candidates[:3]
        else:
            logger.warning(
                "Could not find motor cortex channels, using first 6 MEG channels"
            )
            target_channels = meg_channels[:6]

        logger.info(f"Selected channels: {target_channels}")

        # Pick these channels
        raw_picked = raw.copy().pick(target_channels)

        # Resample to 200Hz
        logger.info("Resampling to 200 Hz...")
        raw_picked.resample(200, verbose=False)

        # Get data as numpy array
        data_array = raw_picked.get_data()  # (n_channels, n_timepoints)
        srate = int(raw_picked.info["sfreq"])

        logger.info(
            f"Extracted data: {data_array.shape}, srate={srate}Hz"
        )

        # Segment into 500ms (100-sample) windows
        window_samples = int(0.5 * srate)
        n_windows = data_array.shape[1] // window_samples
        n_channels = data_array.shape[0]

        windows = np.zeros(
            (n_windows, n_channels, window_samples), dtype=np.float64
        )
        for i in range(n_windows):
            start = i * window_samples
            end = start + window_samples
            windows[i] = data_array[:, start:end]

        # Scale to reasonable amplitude range (MEG data is in Tesla, very small)
        # Convert to a normalized range similar to our synthetic data
        windows = windows / (np.std(windows) + 1e-15)

        logger.info(f"Segmented into {n_windows} windows of {window_samples} samples")

        # Save as .npz
        np.savez_compressed(
            cache_file,
            data=windows,
            channel_names=np.array(target_channels),
            srate=np.array(srate),
            source=np.array("mne_sample_audvis"),
        )

        logger.info(f"Saved MNE sample motor data: {cache_file}")
        return cache_file

    except ImportError:
        logger.warning(
            "MNE-Python not installed. Install with: pip install mne\n"
            "Falling back to synthetic data generation."
        )
        return None
    except Exception as e:
        logger.error(f"Failed to download/process MNE sample data: {e}")
        logger.info("Falling back to synthetic data generation.")
        return None


def generate_synthetic_fallback(
    output_dir: Optional[Path] = None,
    n_samples: int = 5400,
    seed: int = 42,
    force: bool = False,
) -> Path:
    """Generate synthetic stroke MEG data as a fallback.

    This is guaranteed to succeed and produces data in the same format
    as the real MEG data pipeline.

    Args:
        output_dir: Directory to save synthetic data. Defaults to data/synthetic/.
        n_samples: Total number of samples to generate.
        seed: Random seed for reproducibility.
        force: If True, regenerate even if data exists.

    Returns:
        Path to the directory containing train/val/test .npz files.
    """
    # Import here to avoid circular imports at module level
    from data.synthetic_generator import generate_dataset, save_dataset

    output_dir = output_dir or SYNTHETIC_DIR

    # Check if already generated
    train_file = output_dir / "train.npz"
    if train_file.exists() and not force:
        logger.info(f"Synthetic data already exists: {output_dir}")
        return output_dir

    logger.info(f"Generating {n_samples} synthetic MEG samples (seed={seed})...")
    start_time = time.time()

    dataset = generate_dataset(n_samples=n_samples, seed=seed)
    paths = save_dataset(dataset, output_dir, seed=seed)

    elapsed = time.time() - start_time
    logger.info(
        f"Synthetic data generated in {elapsed:.1f}s: "
        f"{len(paths)} files in {output_dir}"
    )

    return output_dir


def convert_mne_to_training_format(
    mne_data_path: Path,
    output_dir: Optional[Path] = None,
    seed: int = 42,
) -> Optional[Path]:
    """Convert downloaded MNE data into the training format with splits.

    Since real MNE data does not have stroke labels, we assign pseudo-labels
    based on signal characteristics. This is primarily useful for testing the
    pipeline with real MEG signal properties.

    Args:
        mne_data_path: Path to the .npz file from download_mne_sample.
        output_dir: Directory to save processed splits.
        seed: Random seed for splits and label generation.

    Returns:
        Path to the output directory, or None on failure.
    """
    output_dir = output_dir or PROCESSED_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        loaded = np.load(mne_data_path, allow_pickle=True)
        data = loaded["data"]  # (n_windows, 6, 100)
        n_windows = data.shape[0]

        logger.info(f"Converting {n_windows} MNE windows to training format...")

        rng = np.random.default_rng(seed)

        # Generate pseudo-labels: since this is real data without stroke annotations,
        # we create labels based on inter-hemispheric power asymmetry
        labels = np.zeros((n_windows, 3), dtype=np.float64)
        conditions = np.zeros(n_windows, dtype=np.int64)

        for i in range(n_windows):
            window = data[i]
            left_power = np.mean(window[:3] ** 2)
            right_power = np.mean(window[3:] ** 2)
            asymmetry = abs(left_power - right_power) / (
                left_power + right_power + 1e-10
            )

            # Assign condition based on asymmetry + randomness
            # This creates a diverse training set from real signal characteristics
            noise = rng.uniform(0, 0.3)
            score = asymmetry + noise

            if score < 0.3:
                conditions[i] = 0  # "healthy-like"
                labels[i] = [
                    rng.uniform(0.0, 0.05),
                    rng.uniform(0.0, 0.05),
                    rng.uniform(0.0, 0.02),
                ]
            elif score < 0.6:
                conditions[i] = 2  # "chronic-like"
                labels[i] = [
                    rng.uniform(0.3, 0.65),
                    rng.uniform(0.25, 0.6),
                    rng.uniform(0.08, 0.25),
                ]
            else:
                conditions[i] = 1  # "acute-like"
                labels[i] = [
                    rng.uniform(0.7, 1.0),
                    rng.uniform(0.6, 0.95),
                    rng.uniform(0.2, 0.5),
                ]

        # Create train/val/test splits
        indices = rng.permutation(n_windows)
        n_train = int(n_windows * 0.7)
        n_val = int(n_windows * 0.15)

        splits = {
            "train": indices[:n_train],
            "val": indices[n_train : n_train + n_val],
            "test": indices[n_train + n_val :],
        }

        for split_name, split_idx in splits.items():
            filepath = output_dir / f"{split_name}.npz"
            np.savez_compressed(
                filepath,
                data=data[split_idx],
                labels=labels[split_idx],
                conditions=conditions[split_idx],
            )
            logger.info(f"Saved {split_name}: {len(split_idx)} samples -> {filepath}")

        return output_dir

    except Exception as e:
        logger.error(f"Failed to convert MNE data: {e}")
        return None


def run_download_pipeline(
    force: bool = False,
    synthetic_only: bool = False,
    n_synthetic: int = 5400,
    seed: int = 42,
) -> Dict[str, Optional[Path]]:
    """Run the complete data download and generation pipeline.

    Attempts to:
    1. Download MNE sample data and convert to training format
    2. Generate synthetic data as primary training data (always)
    3. Fall back to synthetic-only if MNE download fails

    Args:
        force: Re-download/regenerate even if data exists.
        synthetic_only: Skip MNE download, only generate synthetic.
        n_synthetic: Number of synthetic samples to generate.
        seed: Random seed.

    Returns:
        Dictionary with paths to available data sources:
            - 'synthetic': Path to synthetic data directory
            - 'mne_raw': Path to raw MNE data file (or None)
            - 'mne_processed': Path to processed MNE data directory (or None)
    """
    _ensure_dirs()

    result: Dict[str, Optional[Path]] = {
        "synthetic": None,
        "mne_raw": None,
        "mne_processed": None,
    }

    # Step 1: Always generate synthetic data (it is our primary, reliable source)
    logger.info("=" * 60)
    logger.info("Step 1: Generating synthetic training data")
    logger.info("=" * 60)
    result["synthetic"] = generate_synthetic_fallback(
        n_samples=n_synthetic, seed=seed, force=force
    )

    # Step 2: Optionally download real MNE data
    if not synthetic_only:
        logger.info("")
        logger.info("=" * 60)
        logger.info("Step 2: Downloading MNE sample dataset")
        logger.info("=" * 60)

        mne_path = download_mne_sample(force=force)
        result["mne_raw"] = mne_path

        if mne_path is not None:
            logger.info("")
            logger.info("=" * 60)
            logger.info("Step 3: Converting MNE data to training format")
            logger.info("=" * 60)
            result["mne_processed"] = convert_mne_to_training_format(
                mne_path, seed=seed
            )
    else:
        logger.info("Skipping MNE download (synthetic-only mode)")

    # Summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("Download Pipeline Summary")
    logger.info("=" * 60)
    for source, path in result.items():
        status = f"READY at {path}" if path else "NOT AVAILABLE"
        logger.info(f"  {source}: {status}")

    return result


def verify_data_integrity(data_dir: Path) -> bool:
    """Verify that a data directory has valid train/val/test splits.

    Args:
        data_dir: Directory containing .npz split files.

    Returns:
        True if all required files exist and have valid shapes.
    """
    required_files = ["train.npz", "val.npz", "test.npz"]

    for fname in required_files:
        fpath = data_dir / fname
        if not fpath.exists():
            logger.warning(f"Missing required file: {fpath}")
            return False

        try:
            loaded = np.load(fpath)
            data = loaded["data"]
            labels = loaded["labels"]

            if data.ndim != 3 or data.shape[1] != 6 or data.shape[2] != 100:
                logger.warning(
                    f"Invalid data shape in {fpath}: {data.shape} "
                    f"(expected (N, 6, 100))"
                )
                return False

            if labels.ndim != 2 or labels.shape[1] != 3:
                logger.warning(
                    f"Invalid labels shape in {fpath}: {labels.shape} "
                    f"(expected (N, 3))"
                )
                return False

            if data.shape[0] != labels.shape[0]:
                logger.warning(
                    f"Data/label count mismatch in {fpath}: "
                    f"{data.shape[0]} vs {labels.shape[0]}"
                )
                return False

        except Exception as e:
            logger.warning(f"Error loading {fpath}: {e}")
            return False

    logger.info(f"Data integrity verified: {data_dir}")
    return True


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download and prepare MEG data for stroke intervention model"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Only generate synthetic data (skip MNE download)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download/regeneration of all data",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5400,
        help="Number of synthetic samples to generate (default: 5400)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    logger.info("=== MEG Stroke Data Download Pipeline ===")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Data directory: {DATA_DIR}")

    result = run_download_pipeline(
        force=args.force,
        synthetic_only=args.synthetic,
        n_synthetic=args.n_samples,
        seed=args.seed,
    )

    # Verify generated data
    logger.info("")
    logger.info("=== Verifying Data Integrity ===")
    if result["synthetic"]:
        verify_data_integrity(result["synthetic"])
    if result["mne_processed"]:
        verify_data_integrity(result["mne_processed"])

    logger.info("\nDone.")
