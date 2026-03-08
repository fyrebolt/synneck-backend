"""
MEG Stroke Intervention - Data Pipeline

Provides synthetic data generation, preprocessing, downloading, and
PyTorch data loading for 6-channel motor cortex MEG stroke data.
"""

from data.data_loader import (
    AugmentationConfig,
    MEGStrokeDataset,
    create_dataloaders,
    create_dataloaders_from_generated,
    create_datasets,
)
from data.preprocessing import MEGPreprocessor, PreprocessingConfig
from data.synthetic_generator import (
    CHANNEL_NAMES,
    CONDITIONS,
    CONDITION_PROFILES,
    NUM_CHANNELS,
    OUTPUT_SRATE,
    WINDOW_SAMPLES,
    generate_dataset,
    generate_sample,
    save_dataset,
)

__all__ = [
    # Synthetic generation
    "generate_dataset",
    "generate_sample",
    "save_dataset",
    "CHANNEL_NAMES",
    "CONDITIONS",
    "CONDITION_PROFILES",
    "NUM_CHANNELS",
    "OUTPUT_SRATE",
    "WINDOW_SAMPLES",
    # Preprocessing
    "MEGPreprocessor",
    "PreprocessingConfig",
    # Data loading
    "MEGStrokeDataset",
    "AugmentationConfig",
    "create_datasets",
    "create_dataloaders",
    "create_dataloaders_from_generated",
]
