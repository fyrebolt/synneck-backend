"""
MEG signal preprocessing pipeline for the MEG Stroke Intervention project.

Provides real-time compatible preprocessing for 6-channel motor cortex MEG data,
including bandpass filtering, notch filtering, artifact removal, feature extraction
(PSD in mu/beta bands), laterality index computation, and normalization.

All functions operate on single windows to support real-time inference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal

logger = logging.getLogger(__name__)

# Default filter parameters
DEFAULT_BANDPASS: Tuple[float, float] = (1.0, 45.0)
DEFAULT_NOTCH_FREQS: List[float] = [50.0, 60.0]
DEFAULT_SRATE: int = 200
DEFAULT_ARTIFACT_THRESHOLD: float = 5.0  # standard deviations


@dataclass
class PreprocessingConfig:
    """Configuration for the MEG preprocessing pipeline.

    Attributes:
        srate: Sampling rate of input data in Hz.
        bandpass: (low, high) cutoff frequencies for bandpass filter in Hz.
        notch_freqs: List of power line frequencies to notch filter (Hz).
        notch_quality: Quality factor for notch filters.
        artifact_threshold: Threshold in standard deviations for artifact rejection.
        mu_band: (low, high) frequency range for mu rhythm in Hz.
        beta_band: (low, high) frequency range for beta rhythm in Hz.
        normalize: Whether to z-score normalize features.
        filter_order: Order of the Butterworth bandpass filter.
    """

    srate: int = DEFAULT_SRATE
    bandpass: Tuple[float, float] = DEFAULT_BANDPASS
    notch_freqs: List[float] = field(default_factory=lambda: list(DEFAULT_NOTCH_FREQS))
    notch_quality: float = 30.0
    artifact_threshold: float = DEFAULT_ARTIFACT_THRESHOLD
    mu_band: Tuple[float, float] = (8.0, 12.0)
    beta_band: Tuple[float, float] = (12.0, 32.0)
    normalize: bool = True
    filter_order: int = 4


class MEGPreprocessor:
    """Real-time compatible MEG signal preprocessor.

    Designed to process single 500ms windows of 6-channel motor cortex data.
    Can be used both in batch mode (full dataset) and streaming mode (one
    window at a time for real-time BCI).

    Attributes:
        config: PreprocessingConfig with all filter parameters.
        running_mean: Running mean for online normalization, shape (n_features,).
        running_var: Running variance for online normalization, shape (n_features,).
        n_seen: Number of samples seen for running statistics.
    """

    # Channel groupings: left hemisphere (indices 0, 2, 4) and right (1, 3, 5)
    LEFT_CHANNELS: List[int] = [0, 2, 4]   # C3, FC3, CP3
    RIGHT_CHANNELS: List[int] = [1, 3, 5]  # C4, FC4, CP4

    def __init__(self, config: Optional[PreprocessingConfig] = None) -> None:
        self.config = config or PreprocessingConfig()
        self._build_filters()

        # Running statistics for online normalization
        self.running_mean: Optional[NDArray[np.float64]] = None
        self.running_var: Optional[NDArray[np.float64]] = None
        self.n_seen: int = 0

    def _build_filters(self) -> None:
        """Pre-compute filter coefficients for efficiency."""
        nyquist = self.config.srate / 2.0

        # Bandpass filter coefficients (Butterworth)
        low = self.config.bandpass[0] / nyquist
        high = self.config.bandpass[1] / nyquist
        # Clamp to valid Butterworth range (0, 1)
        low = max(low, 1e-5)
        high = min(high, 1.0 - 1e-5)
        self._bp_sos = scipy_signal.butter(
            self.config.filter_order, [low, high], btype="bandpass", output="sos"
        )

        # Notch filter coefficients
        self._notch_coeffs: List[Tuple[NDArray, NDArray]] = []
        for freq in self.config.notch_freqs:
            if freq < nyquist:  # Only create notch if freq is below Nyquist
                b, a = scipy_signal.iirnotch(
                    freq, self.config.notch_quality, fs=self.config.srate
                )
                self._notch_coeffs.append((b, a))

        logger.debug(
            f"Built filters: bandpass={self.config.bandpass}, "
            f"notch={[f for f in self.config.notch_freqs if f < nyquist]}"
        )

    def bandpass_filter(
        self, data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply bandpass filter to multi-channel data.

        Args:
            data: Input data, shape (channels, timepoints).

        Returns:
            Filtered data, same shape as input.
        """
        return scipy_signal.sosfiltfilt(self._bp_sos, data, axis=-1).astype(
            np.float64
        )

    def notch_filter(
        self, data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Apply notch filters to remove power line interference.

        Args:
            data: Input data, shape (channels, timepoints).

        Returns:
            Filtered data with power line artifacts removed.
        """
        result = data.copy()
        for b, a in self._notch_coeffs:
            result = scipy_signal.filtfilt(b, a, result, axis=-1)
        return result.astype(np.float64)

    def remove_artifacts(
        self, data: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.bool_]]:
        """Remove artifacts using threshold-based detection.

        Channels exceeding the threshold (in standard deviations from the mean)
        are interpolated from neighboring channels. If all channels exceed the
        threshold, the window is clipped.

        Args:
            data: Input data, shape (channels, timepoints).

        Returns:
            Tuple of:
                - Cleaned data, same shape as input.
                - Boolean mask indicating which channels had artifacts, shape (channels,).
        """
        cleaned = data.copy()
        n_channels = data.shape[0]

        # Compute per-channel statistics
        ch_means = np.mean(data, axis=-1, keepdims=True)
        ch_stds = np.std(data, axis=-1, keepdims=True)
        ch_stds = np.maximum(ch_stds, 1e-8)  # avoid division by zero

        # Detect artifact samples: points exceeding threshold
        z_scores = np.abs((data - ch_means) / ch_stds)
        artifact_mask_samples = z_scores > self.config.artifact_threshold

        # Per-channel artifact flag: channel has more than 10% artifact samples
        artifact_fraction = artifact_mask_samples.mean(axis=-1)
        channel_artifact_mask = artifact_fraction > 0.10

        for ch_idx in range(n_channels):
            if not channel_artifact_mask[ch_idx]:
                continue

            # Find artifact timepoints in this channel
            bad_samples = artifact_mask_samples[ch_idx]

            # Interpolate from non-artifactual channels
            good_channels = [
                i for i in range(n_channels)
                if i != ch_idx and not channel_artifact_mask[i]
            ]

            if good_channels:
                # Replace artifact samples with mean of good channels
                replacement = np.mean(data[good_channels], axis=0)
                cleaned[ch_idx, bad_samples] = replacement[bad_samples]
            else:
                # All channels bad: clip to threshold
                limit = (
                    ch_means[ch_idx]
                    + self.config.artifact_threshold * ch_stds[ch_idx]
                )
                cleaned[ch_idx] = np.clip(
                    cleaned[ch_idx], -limit, limit
                )

        return cleaned, channel_artifact_mask

    def compute_psd(
        self,
        data: NDArray[np.float64],
        band: Tuple[float, float],
    ) -> NDArray[np.float64]:
        """Compute average power spectral density in a frequency band.

        Uses Welch's method for robust PSD estimation on short windows.

        Args:
            data: Input data, shape (channels, timepoints).
            band: (low, high) frequency band in Hz.

        Returns:
            Band power per channel, shape (channels,).
        """
        n_samples = data.shape[-1]
        # Use nperseg that fits within our window
        nperseg = min(n_samples, max(32, n_samples // 2))

        freqs, psd = scipy_signal.welch(
            data,
            fs=self.config.srate,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            axis=-1,
        )

        # Find frequency indices within the band
        band_mask = (freqs >= band[0]) & (freqs <= band[1])
        if not np.any(band_mask):
            return np.zeros(data.shape[0], dtype=np.float64)

        # Average power in band
        band_power = np.mean(psd[:, band_mask], axis=-1)
        return band_power.astype(np.float64)

    def extract_laterality_index(
        self, left_power: NDArray[np.float64], right_power: NDArray[np.float64]
    ) -> float:
        """Compute laterality index from bilateral power measurements.

        LI = (Right - Left) / (Right + Left)

        Returns value in [-1, 1] where:
            - Positive values indicate right hemisphere dominance
            - Negative values indicate left hemisphere dominance
            - Zero indicates bilateral symmetry

        Args:
            left_power: Power values for left hemisphere channels, shape (n_left,).
            right_power: Power values for right hemisphere channels, shape (n_right,).

        Returns:
            Scalar laterality index.
        """
        left_avg = np.mean(left_power)
        right_avg = np.mean(right_power)
        denom = left_avg + right_avg

        if denom < 1e-10:
            return 0.0

        return float((right_avg - left_avg) / denom)

    def extract_features(
        self, data: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Extract a feature vector from a single window of MEG data.

        Features include:
            - Mu band power per channel (6 values)
            - Beta band power per channel (6 values)
            - Mu laterality index (1 value)
            - Beta laterality index (1 value)
            - Total power per channel (6 values)
            - Temporal variance per channel (6 values)
        Total: 26 features

        Args:
            data: Preprocessed data, shape (6, timepoints).

        Returns:
            Feature vector, shape (26,).
        """
        # PSD in mu and beta bands
        mu_power = self.compute_psd(data, self.config.mu_band)       # (6,)
        beta_power = self.compute_psd(data, self.config.beta_band)   # (6,)

        # Laterality indices
        mu_li = self.extract_laterality_index(
            mu_power[self.LEFT_CHANNELS], mu_power[self.RIGHT_CHANNELS]
        )
        beta_li = self.extract_laterality_index(
            beta_power[self.LEFT_CHANNELS], beta_power[self.RIGHT_CHANNELS]
        )

        # Total broadband power
        total_power = np.mean(data ** 2, axis=-1)  # (6,)

        # Temporal variance (captures signal dynamics)
        temporal_var = np.var(data, axis=-1)  # (6,)

        features = np.concatenate([
            mu_power,                          # 6
            beta_power,                        # 6
            np.array([mu_li, beta_li]),        # 2
            total_power,                       # 6
            temporal_var,                      # 6
        ])

        return features.astype(np.float64)

    def update_normalization_stats(
        self, features: NDArray[np.float64]
    ) -> None:
        """Update running mean and variance for online normalization.

        Uses Welford's online algorithm for numerically stable computation.

        Args:
            features: Feature vector, shape (n_features,).
        """
        self.n_seen += 1

        if self.running_mean is None:
            self.running_mean = features.copy()
            self.running_var = np.zeros_like(features)
            return

        delta = features - self.running_mean
        self.running_mean += delta / self.n_seen
        delta2 = features - self.running_mean
        self.running_var += delta * delta2

    def normalize_features(
        self, features: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Normalize features using running statistics or simple z-scoring.

        Args:
            features: Raw feature vector, shape (n_features,) or (batch, n_features).

        Returns:
            Normalized feature vector, same shape as input.
        """
        if self.running_mean is not None and self.n_seen > 1:
            variance = self.running_var / (self.n_seen - 1)
            std = np.sqrt(variance + 1e-8)
            return (features - self.running_mean) / std

        # Fallback: per-feature normalization within the vector
        if features.ndim == 1:
            std = np.std(features)
            if std < 1e-8:
                return features
            return (features - np.mean(features)) / std
        else:
            mean = np.mean(features, axis=0, keepdims=True)
            std = np.std(features, axis=0, keepdims=True)
            std = np.maximum(std, 1e-8)
            return (features - mean) / std

    def process_window(
        self,
        data: NDArray[np.float64],
        extract_features: bool = False,
    ) -> Dict[str, NDArray[np.float64] | float | NDArray[np.bool_]]:
        """Process a single window through the complete preprocessing pipeline.

        This is the main entry point for real-time processing. Applies all
        preprocessing steps in order and optionally extracts features.

        Args:
            data: Raw MEG data, shape (6, timepoints).
            extract_features: If True, also compute and return feature vector.

        Returns:
            Dictionary with keys:
                - 'cleaned': Preprocessed data, shape (6, timepoints)
                - 'artifact_mask': Boolean channel artifact flags, shape (6,)
                - 'features': Feature vector, shape (26,) (only if extract_features=True)
                - 'features_normalized': Normalized features (only if extract_features=True
                    and self.config.normalize=True)
        """
        result: Dict[str, NDArray[np.float64] | float | NDArray[np.bool_]] = {}

        # Step 1: Bandpass filter
        filtered = self.bandpass_filter(data)

        # Step 2: Notch filter
        filtered = self.notch_filter(filtered)

        # Step 3: Artifact removal
        cleaned, artifact_mask = self.remove_artifacts(filtered)

        result["cleaned"] = cleaned
        result["artifact_mask"] = artifact_mask

        # Step 4: Feature extraction (optional)
        if extract_features:
            features = self.extract_features(cleaned)
            result["features"] = features

            if self.config.normalize:
                self.update_normalization_stats(features)
                result["features_normalized"] = self.normalize_features(features)

        return result

    def process_batch(
        self,
        data: NDArray[np.float64],
        extract_features: bool = False,
    ) -> Dict[str, NDArray[np.float64]]:
        """Process a batch of windows through the preprocessing pipeline.

        Args:
            data: Batch of MEG data, shape (batch, 6, timepoints).
            extract_features: If True, also compute and return features.

        Returns:
            Dictionary with keys:
                - 'cleaned': shape (batch, 6, timepoints)
                - 'features': shape (batch, 26) (if extract_features=True)
                - 'features_normalized': shape (batch, 26) (if extract_features
                    and normalize)
        """
        batch_size = data.shape[0]
        cleaned_list: List[NDArray[np.float64]] = []
        features_list: List[NDArray[np.float64]] = []

        for i in range(batch_size):
            result = self.process_window(data[i], extract_features=extract_features)
            cleaned_list.append(result["cleaned"])
            if extract_features and "features" in result:
                features_list.append(result["features"])

        output: Dict[str, NDArray[np.float64]] = {
            "cleaned": np.stack(cleaned_list, axis=0),
        }

        if extract_features and features_list:
            raw_features = np.stack(features_list, axis=0)
            output["features"] = raw_features
            if self.config.normalize:
                output["features_normalized"] = self.normalize_features(raw_features)

        return output

    def fit_normalization(self, data: NDArray[np.float64]) -> None:
        """Fit normalization statistics from a batch of training data.

        Processes all windows and computes global mean/variance for feature
        normalization. Call this on the training set before inference.

        Args:
            data: Training data, shape (n_samples, 6, timepoints).
        """
        logger.info(f"Fitting normalization on {data.shape[0]} samples...")

        # Reset running stats
        self.running_mean = None
        self.running_var = None
        self.n_seen = 0

        for i in range(data.shape[0]):
            self.process_window(data[i], extract_features=True)

        logger.info(
            f"Normalization fitted: mean range=[{self.running_mean.min():.4f}, "
            f"{self.running_mean.max():.4f}], n_seen={self.n_seen}"
        )

    def get_normalization_params(
        self,
    ) -> Optional[Dict[str, NDArray[np.float64]]]:
        """Get current normalization parameters for serialization.

        Returns:
            Dict with 'mean', 'var', 'n_seen' or None if not fitted.
        """
        if self.running_mean is None:
            return None
        return {
            "mean": self.running_mean.copy(),
            "var": self.running_var.copy() / max(self.n_seen - 1, 1),
            "n_seen": np.array(self.n_seen),
        }

    def set_normalization_params(
        self, params: Dict[str, NDArray[np.float64]]
    ) -> None:
        """Load normalization parameters from a saved state.

        Args:
            params: Dict with 'mean', 'var', 'n_seen' keys.
        """
        self.running_mean = params["mean"].copy()
        self.running_var = params["var"].copy() * int(params["n_seen"])
        self.n_seen = int(params["n_seen"])
        logger.info(f"Loaded normalization params (n_seen={self.n_seen})")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from synthetic_generator import generate_dataset

    logger.info("=== MEG Preprocessing Pipeline Test ===")

    # Generate a small test dataset
    dataset = generate_dataset(n_samples=100, seed=123)
    data = dataset["data"]
    logger.info(f"Input data shape: {data.shape}")

    # Create preprocessor
    config = PreprocessingConfig(srate=200, normalize=True)
    preprocessor = MEGPreprocessor(config)

    # Process single window
    logger.info("\n--- Single window processing ---")
    single = data[0]  # (6, 100)
    result = preprocessor.process_window(single, extract_features=True)
    logger.info(f"Cleaned shape: {result['cleaned'].shape}")
    logger.info(f"Artifact mask: {result['artifact_mask']}")
    logger.info(f"Features shape: {result['features'].shape}")
    logger.info(f"Features (first 10): {result['features'][:10]}")

    # Process batch
    logger.info("\n--- Batch processing ---")
    batch_result = preprocessor.process_batch(data[:20], extract_features=True)
    logger.info(f"Batch cleaned shape: {batch_result['cleaned'].shape}")
    logger.info(f"Batch features shape: {batch_result['features'].shape}")

    # Fit normalization on full training set
    logger.info("\n--- Fitting normalization ---")
    preprocessor_fitted = MEGPreprocessor(config)
    preprocessor_fitted.fit_normalization(data)

    # Process with fitted normalization
    result_norm = preprocessor_fitted.process_window(data[50], extract_features=True)
    if "features_normalized" in result_norm:
        feats = result_norm["features_normalized"]
        logger.info(
            f"Normalized features: mean={feats.mean():.4f}, std={feats.std():.4f}"
        )

    # Test serialization
    params = preprocessor_fitted.get_normalization_params()
    if params is not None:
        logger.info(f"Normalization params: mean shape={params['mean'].shape}")
        new_preprocessor = MEGPreprocessor(config)
        new_preprocessor.set_normalization_params(params)
        logger.info("Successfully loaded normalization params into new preprocessor")

    logger.info("\n=== Preprocessing Pipeline Test Complete ===")
