from setuptools import setup, find_packages

setup(
    name="meg-stroke-intervention",
    version="1.0.0",
    description="MEG-based stroke intervention neural network for solenoid valve control",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "mne>=1.4.0",
        "scikit-learn>=1.2.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "h5py>=3.8.0",
        "pandas>=2.0.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.15.0",
    ],
    entry_points={
        "console_scripts": [
            "meg-stroke=run_complete_pipeline:main",
        ],
    },
)
