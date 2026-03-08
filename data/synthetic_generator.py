"""
Synthetic MEG stroke data generator for the MEG Stroke Intervention project.

Generates realistic synthetic MEG signals with motor cortex ERD/ERS patterns
for three conditions: healthy, acute_stroke, and chronic_stroke. Produces
corresponding solenoid valve control labels for rehabilitation intervention.

Channels: C3, C4, FC3, FC4, CP3, CP4 (bilateral motor cortex coverage)
Sampling: 1000Hz generation -> 200Hz downsampled output
Window: 500ms (100 samples at 200Hz)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# Channel configuration
CHANNEL_NAMES: List[str] = ["C3", "C4", "FC3", "FC4", "CP3", "CP4"]
NUM_CHANNELS: int = 6

# Sampling configuration
GENERATION_SRATE: int = 1000  # Hz, for realistic signal generation
OUTPUT_SRATE: int = 200       # Hz, downsampled for model input
WINDOW_SEC: float = 0.5       # seconds
WINDOW_SAMPLES: int = int(WINDOW_SEC * OUTPUT_SRATE)  # 100 samples

# Frequency bands of interest
MU_BAND: Tuple[float, float] = (8.0, 12.0)
BETA_BAND: Tuple[float, float] = (12.0, 32.0)

# Condition types
CONDITIONS: List[str] = ["healthy", "acute_stroke", "chronic_stroke"]


@dataclass
class StrokeProfile:
    """Defines the neurophysiological profile for a stroke condition.

    Attributes:
        condition: One of 'healthy', 'acute_stroke', 'chronic_stroke'.
        affected_side: 'left' or 'right' hemisphere affected. For healthy, this
            is set to 'none' but channels are generated symmetrically.
        mu_amplitude_ipsi: Mu rhythm amplitude on the ipsilesional (affected) side.
        mu_amplitude_contra: Mu rhythm amplitude on the contralesional (healthy) side.
        beta_amplitude_ipsi: Beta rhythm amplitude on ipsilesional side.
        beta_amplitude_contra: Beta rhythm amplitude on contralesional side.
        erd_depth_ipsi: Event-related desynchronization depth (0-1) ipsilesional.
        erd_depth_contra: Event-related desynchronization depth (0-1) contralesional.
        latency_shift_ms: Additional latency in ms for ipsilesional response.
        noise_level: Background noise amplitude.
        valve_extension_range: (min, max) for valve extension labels.
        force_magnitude_range: (min, max) for force magnitude labels.
        trigger_delay_range: (min, max) for trigger delay labels.
    """

    condition: str
    affected_side: str = "right"
    mu_amplitude_ipsi: float = 1.0
    mu_amplitude_contra: float = 1.0
    beta_amplitude_ipsi: float = 0.5
    beta_amplitude_contra: float = 0.5
    erd_depth_ipsi: float = 0.6
    erd_depth_contra: float = 0.6
    latency_shift_ms: float = 0.0
    noise_level: float = 0.1
    valve_extension_range: Tuple[float, float] = (0.0, 0.0)
    force_magnitude_range: Tuple[float, float] = (0.0, 0.0)
    trigger_delay_range: Tuple[float, float] = (0.0, 0.0)


# Pre-defined profiles for each condition
CONDITION_PROFILES: Dict[str, StrokeProfile] = {
    "healthy": StrokeProfile(
        condition="healthy",
        affected_side="none",
        mu_amplitude_ipsi=1.0,
        mu_amplitude_contra=1.0,
        beta_amplitude_ipsi=0.5,
        beta_amplitude_contra=0.5,
        erd_depth_ipsi=0.6,
        erd_depth_contra=0.6,
        latency_shift_ms=0.0,
        noise_level=0.08,
        valve_extension_range=(0.0, 0.05),
        force_magnitude_range=(0.0, 0.05),
        trigger_delay_range=(0.0, 0.02),
    ),
    "acute_stroke": StrokeProfile(
        condition="acute_stroke",
        affected_side="right",
        mu_amplitude_ipsi=0.25,        # severely reduced
        mu_amplitude_contra=1.1,       # slight compensatory increase
        beta_amplitude_ipsi=0.15,      # severely reduced
        beta_amplitude_contra=0.6,     # slight compensatory increase
        erd_depth_ipsi=0.15,           # minimal desynchronization
        erd_depth_contra=0.7,          # hyper-activation compensating
        latency_shift_ms=80.0,         # significant delay
        noise_level=0.15,              # more neural noise
        valve_extension_range=(0.7, 1.0),
        force_magnitude_range=(0.6, 0.95),
        trigger_delay_range=(0.2, 0.5),
    ),
    "chronic_stroke": StrokeProfile(
        condition="chronic_stroke",
        affected_side="right",
        mu_amplitude_ipsi=0.55,        # partially recovered
        mu_amplitude_contra=1.05,      # mild compensation
        beta_amplitude_ipsi=0.35,      # partially recovered
        beta_amplitude_contra=0.55,    # mild compensation
        erd_depth_ipsi=0.35,           # partial recovery of ERD
        erd_depth_contra=0.65,         # slight over-activation
        latency_shift_ms=35.0,         # moderate delay
        noise_level=0.11,
        valve_extension_range=(0.3, 0.65),
        force_magnitude_range=(0.25, 0.6),
        trigger_delay_range=(0.08, 0.25),
    ),
}


def _generate_oscillation(
    n_samples: int,
    srate: int,
    freq_band: Tuple[float, float],
    amplitude: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate a band-limited oscillatory signal.

    Uses superposition of sinusoids at random frequencies within the band
    with random phases to create a naturalistic oscillatory signal.

    Args:
        n_samples: Number of samples to generate.
        srate: Sampling rate in Hz.
        freq_band: (low, high) frequency limits in Hz.
        amplitude: Peak amplitude of the oscillation.
        rng: NumPy random generator instance.

    Returns:
        1-D array of shape (n_samples,) containing the oscillatory signal.
    """
    t = np.arange(n_samples) / srate
    n_components = rng.integers(3, 7)
    freqs = rng.uniform(freq_band[0], freq_band[1], size=n_components)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=n_components)
    amplitudes = rng.uniform(0.5, 1.0, size=n_components)
    amplitudes = amplitudes / amplitudes.sum()  # normalize so they sum to 1

    sig = np.zeros(n_samples, dtype=np.float64)
    for freq, phase, amp in zip(freqs, phases, amplitudes):
        sig += amp * np.sin(2.0 * np.pi * freq * t + phase)

    return sig * amplitude


def _apply_erd_envelope(
    sig: NDArray[np.float64],
    srate: int,
    erd_depth: float,
    latency_shift_ms: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Apply an event-related desynchronization (ERD) amplitude envelope.

    Simulates the characteristic power decrease during motor planning/execution,
    with optional latency shift for perilesional dysfunction.

    Args:
        sig: Input oscillatory signal, shape (n_samples,).
        srate: Sampling rate in Hz.
        erd_depth: Depth of desynchronization (0 = no ERD, 1 = full suppression).
        latency_shift_ms: Additional latency in milliseconds before ERD onset.
        rng: NumPy random generator.

    Returns:
        Signal with ERD envelope applied, same shape as input.
    """
    n_samples = len(sig)
    t = np.arange(n_samples) / srate

    # ERD typically begins ~200ms into a 500ms window for motor tasks
    erd_onset = 0.15 + (latency_shift_ms / 1000.0)
    erd_onset += rng.normal(0, 0.02)  # jitter
    erd_onset = np.clip(erd_onset, 0.05, 0.45)

    # Create smooth ERD envelope using a sigmoid transition
    steepness = rng.uniform(15.0, 30.0)
    envelope = 1.0 - erd_depth * (1.0 / (1.0 + np.exp(-steepness * (t - erd_onset))))

    return sig * envelope


def _generate_single_channel(
    n_samples: int,
    srate: int,
    mu_amplitude: float,
    beta_amplitude: float,
    erd_depth: float,
    latency_shift_ms: float,
    noise_level: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Generate a single MEG channel signal with mu and beta oscillations.

    Args:
        n_samples: Number of samples.
        srate: Sampling rate in Hz.
        mu_amplitude: Amplitude of mu (8-12 Hz) oscillation.
        beta_amplitude: Amplitude of beta (12-32 Hz) oscillation.
        erd_depth: Depth of event-related desynchronization.
        latency_shift_ms: Latency shift for ERD onset in ms.
        noise_level: Standard deviation of background noise.
        rng: NumPy random generator.

    Returns:
        1-D signal array of shape (n_samples,).
    """
    # Generate mu rhythm
    mu = _generate_oscillation(n_samples, srate, MU_BAND, mu_amplitude, rng)
    mu = _apply_erd_envelope(mu, srate, erd_depth, latency_shift_ms, rng)

    # Generate beta rhythm
    beta = _generate_oscillation(n_samples, srate, BETA_BAND, beta_amplitude, rng)
    beta = _apply_erd_envelope(beta, srate, erd_depth * 0.8, latency_shift_ms, rng)

    # Add 1/f background noise (pink noise approximation)
    white = rng.normal(0.0, noise_level, n_samples)
    # Simple 1/f filter: integrate and leak
    pink = np.zeros(n_samples, dtype=np.float64)
    alpha = 0.98
    pink[0] = white[0]
    for i in range(1, n_samples):
        pink[i] = alpha * pink[i - 1] + white[i]
    pink = pink * (noise_level / (np.std(pink) + 1e-8))

    # Add occasional transient artifacts (muscle, eye blinks) at low probability
    if rng.random() < 0.05:
        artifact_pos = rng.integers(0, n_samples)
        artifact_width = rng.integers(5, 30)
        artifact_amp = rng.uniform(2.0, 5.0) * noise_level
        start = max(0, artifact_pos - artifact_width // 2)
        end = min(n_samples, artifact_pos + artifact_width // 2)
        artifact = artifact_amp * np.exp(
            -0.5 * ((np.arange(start, end) - artifact_pos) / (artifact_width / 4)) ** 2
        )
        combined = mu + beta + pink
        combined[start:end] += artifact
        return combined

    return mu + beta + pink


def _downsample(
    sig: NDArray[np.float64], original_srate: int, target_srate: int
) -> NDArray[np.float64]:
    """Downsample a signal using scipy's decimate for anti-aliased resampling.

    Args:
        sig: Input signal, shape (..., n_samples).
        original_srate: Original sampling rate in Hz.
        target_srate: Target sampling rate in Hz.

    Returns:
        Downsampled signal.
    """
    factor = original_srate // target_srate
    if factor <= 1:
        return sig
    return scipy_signal.decimate(sig, factor, axis=-1, zero_phase=True)


def generate_sample(
    profile: StrokeProfile,
    rng: np.random.Generator,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate a single MEG data sample with corresponding valve control labels.

    Produces a 6-channel, 500ms window of synthetic MEG data at the generation
    sampling rate, then downsamples to the output rate.

    Args:
        profile: StrokeProfile defining the neurophysiological parameters.
        rng: NumPy random generator for reproducibility.

    Returns:
        Tuple of:
            - data: array of shape (NUM_CHANNELS, WINDOW_SAMPLES) = (6, 100)
            - labels: array of shape (3,) = [valve_extension, force_magnitude, trigger_delay]
    """
    n_gen = int(WINDOW_SEC * GENERATION_SRATE)
    channels: List[NDArray[np.float64]] = []

    # Channel mapping: C3, FC3, CP3 are left hemisphere; C4, FC4, CP4 are right
    # If affected_side is 'right', ipsilesional channels are right (C4, FC4, CP4)
    left_channels = [0, 2, 4]   # C3, FC3, CP3
    right_channels = [1, 3, 5]  # C4, FC4, CP4

    if profile.affected_side == "right":
        ipsi_indices = right_channels
        contra_indices = left_channels
    elif profile.affected_side == "left":
        ipsi_indices = left_channels
        contra_indices = right_channels
    else:
        # Healthy: symmetric
        ipsi_indices = list(range(NUM_CHANNELS))
        contra_indices = []

    for ch_idx in range(NUM_CHANNELS):
        if ch_idx in ipsi_indices and profile.affected_side != "none":
            mu_amp = profile.mu_amplitude_ipsi
            beta_amp = profile.beta_amplitude_ipsi
            erd = profile.erd_depth_ipsi
            lat = profile.latency_shift_ms
        elif ch_idx in contra_indices:
            mu_amp = profile.mu_amplitude_contra
            beta_amp = profile.beta_amplitude_contra
            erd = profile.erd_depth_contra
            lat = 0.0
        else:
            # Healthy symmetric case
            mu_amp = profile.mu_amplitude_ipsi
            beta_amp = profile.beta_amplitude_ipsi
            erd = profile.erd_depth_ipsi
            lat = 0.0

        # Add slight per-channel variation
        mu_amp *= rng.uniform(0.9, 1.1)
        beta_amp *= rng.uniform(0.9, 1.1)
        erd *= rng.uniform(0.9, 1.1)
        erd = np.clip(erd, 0.0, 1.0)

        ch_signal = _generate_single_channel(
            n_gen, GENERATION_SRATE, mu_amp, beta_amp, erd, lat,
            profile.noise_level, rng,
        )
        channels.append(ch_signal)

    # Stack channels: (6, n_gen)
    data = np.stack(channels, axis=0)

    # Downsample from GENERATION_SRATE to OUTPUT_SRATE
    data = _downsample(data, GENERATION_SRATE, OUTPUT_SRATE)

    # Ensure exact window size (handle rounding)
    if data.shape[1] > WINDOW_SAMPLES:
        data = data[:, :WINDOW_SAMPLES]
    elif data.shape[1] < WINDOW_SAMPLES:
        pad = WINDOW_SAMPLES - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad)), mode="edge")

    # Generate valve control labels based on condition
    valve_ext = rng.uniform(*profile.valve_extension_range)
    force_mag = rng.uniform(*profile.force_magnitude_range)
    trigger_del = rng.uniform(*profile.trigger_delay_range)
    labels = np.array([valve_ext, force_mag, trigger_del], dtype=np.float64)

    return data, labels


def generate_dataset(
    n_samples: int = 5400,
    condition_ratios: Optional[Dict[str, float]] = None,
    seed: int = 42,
    vary_affected_side: bool = True,
) -> Dict[str, NDArray[np.float64]]:
    """Generate a complete synthetic dataset with multiple conditions.

    Args:
        n_samples: Total number of samples to generate.
        condition_ratios: Dict mapping condition names to their proportions.
            Defaults to equal distribution across conditions.
        seed: Random seed for reproducibility.
        vary_affected_side: If True, randomly assign affected side for stroke
            conditions to increase data diversity.

    Returns:
        Dictionary with keys:
            - 'data': shape (n_samples, 6, 100) float64
            - 'labels': shape (n_samples, 3) float64
            - 'conditions': shape (n_samples,) int64 (0=healthy, 1=acute, 2=chronic)
            - 'condition_names': list of condition name strings
    """
    if condition_ratios is None:
        condition_ratios = {c: 1.0 / len(CONDITIONS) for c in CONDITIONS}

    # Normalize ratios
    total = sum(condition_ratios.values())
    condition_ratios = {k: v / total for k, v in condition_ratios.items()}

    rng = np.random.default_rng(seed)

    all_data: List[NDArray[np.float64]] = []
    all_labels: List[NDArray[np.float64]] = []
    all_conditions: List[int] = []

    for cond_idx, cond_name in enumerate(CONDITIONS):
        n_cond = int(n_samples * condition_ratios.get(cond_name, 0.0))
        if cond_idx == len(CONDITIONS) - 1:
            # Ensure we hit exactly n_samples
            n_cond = n_samples - sum(
                int(n_samples * condition_ratios.get(c, 0.0))
                for c in CONDITIONS[:-1]
            )

        logger.info(f"Generating {n_cond} samples for condition: {cond_name}")

        base_profile = CONDITION_PROFILES[cond_name]

        for _ in range(n_cond):
            # Optionally vary the affected side for stroke conditions
            profile = StrokeProfile(
                condition=base_profile.condition,
                affected_side=(
                    rng.choice(["left", "right"])
                    if vary_affected_side and base_profile.affected_side != "none"
                    else base_profile.affected_side
                ),
                mu_amplitude_ipsi=base_profile.mu_amplitude_ipsi,
                mu_amplitude_contra=base_profile.mu_amplitude_contra,
                beta_amplitude_ipsi=base_profile.beta_amplitude_ipsi,
                beta_amplitude_contra=base_profile.beta_amplitude_contra,
                erd_depth_ipsi=base_profile.erd_depth_ipsi,
                erd_depth_contra=base_profile.erd_depth_contra,
                latency_shift_ms=base_profile.latency_shift_ms,
                noise_level=base_profile.noise_level,
                valve_extension_range=base_profile.valve_extension_range,
                force_magnitude_range=base_profile.force_magnitude_range,
                trigger_delay_range=base_profile.trigger_delay_range,
            )

            data, labels = generate_sample(profile, rng)
            all_data.append(data)
            all_labels.append(labels)
            all_conditions.append(cond_idx)

    dataset = {
        "data": np.stack(all_data, axis=0),          # (N, 6, 100)
        "labels": np.stack(all_labels, axis=0),       # (N, 3)
        "conditions": np.array(all_conditions, dtype=np.int64),  # (N,)
        "condition_names": CONDITIONS,
    }

    logger.info(
        f"Generated dataset: data={dataset['data'].shape}, "
        f"labels={dataset['labels'].shape}, "
        f"conditions distribution={np.bincount(dataset['conditions']).tolist()}"
    )

    return dataset


def save_dataset(
    dataset: Dict[str, NDArray[np.float64]],
    output_dir: str | Path,
    split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, Path]:
    """Save dataset to disk with train/val/test splits.

    Args:
        dataset: Dictionary from generate_dataset().
        output_dir: Directory to save .npz files.
        split_ratios: (train, val, test) proportions.
        seed: Random seed for shuffling.

    Returns:
        Dictionary mapping split names to their file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n = len(dataset["data"])
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)

    n_train = int(n * split_ratios[0])
    n_val = int(n * split_ratios[1])

    splits = {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }

    saved_paths: Dict[str, Path] = {}

    for split_name, split_indices in splits.items():
        filepath = output_dir / f"{split_name}.npz"
        np.savez_compressed(
            filepath,
            data=dataset["data"][split_indices],
            labels=dataset["labels"][split_indices],
            conditions=dataset["conditions"][split_indices],
        )
        saved_paths[split_name] = filepath
        logger.info(
            f"Saved {split_name} split: {len(split_indices)} samples -> {filepath}"
        )

    # Also save metadata
    metadata_path = output_dir / "metadata.npz"
    np.savez(
        metadata_path,
        channel_names=np.array(CHANNEL_NAMES),
        condition_names=np.array(CONDITIONS),
        srate=np.array(OUTPUT_SRATE),
        window_samples=np.array(WINDOW_SAMPLES),
    )
    saved_paths["metadata"] = metadata_path

    return saved_paths


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info("=== Synthetic MEG Stroke Data Generator ===")
    logger.info(f"Channels: {CHANNEL_NAMES}")
    logger.info(f"Output sample rate: {OUTPUT_SRATE} Hz")
    logger.info(f"Window: {WINDOW_SEC}s = {WINDOW_SAMPLES} samples")

    # Generate dataset
    dataset = generate_dataset(n_samples=5400, seed=42)

    logger.info(f"Data shape: {dataset['data'].shape}")
    logger.info(f"Labels shape: {dataset['labels'].shape}")
    logger.info(f"Conditions: {np.bincount(dataset['conditions']).tolist()}")

    # Print label statistics per condition
    for cond_idx, cond_name in enumerate(CONDITIONS):
        mask = dataset["conditions"] == cond_idx
        cond_labels = dataset["labels"][mask]
        logger.info(
            f"  {cond_name}: "
            f"valve_ext={cond_labels[:, 0].mean():.3f}+/-{cond_labels[:, 0].std():.3f}, "
            f"force_mag={cond_labels[:, 1].mean():.3f}+/-{cond_labels[:, 1].std():.3f}, "
            f"trigger_del={cond_labels[:, 2].mean():.3f}+/-{cond_labels[:, 2].std():.3f}"
        )

    # Save to disk
    project_root = Path(__file__).resolve().parent.parent
    output_dir = project_root / "data" / "synthetic"
    paths = save_dataset(dataset, output_dir)
    logger.info(f"Saved splits to: {output_dir}")
    for name, path in paths.items():
        logger.info(f"  {name}: {path}")
