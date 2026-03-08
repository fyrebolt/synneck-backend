# MEG Stroke Intervention Neural Network

A PyTorch-based neural network system that processes MEG (magnetoencephalography) brain signals from stroke patients to control solenoid valves for neck intervention. Designed for real-time deployment on Arduino microcontrollers.

## Quick Start

```bash
# Install
pip install -e .

# Run everything (generates data, trains, validates, converts to Arduino)
python run_complete_pipeline.py

# Quick mode for testing (~2 min)
python run_complete_pipeline.py --quick
```

## System Overview

**Input:** 6-channel MEG data (C3, C4, FC3, FC4, CP3, CP4) at 200Hz, 500ms windows

**Output:** 3 valve control values: `[valve_extension, force_magnitude, trigger_delay]`

**Architecture:** Ultra-lightweight 1D CNN (< 5K parameters) designed for Arduino deployment

```
Conv1d(6->16, k=5, s=2) -> ReLU -> BN
Conv1d(16->32, k=3, s=2) -> ReLU -> BN
Conv1d(32->16, k=3, s=2) -> ReLU
Global Average Pooling
Dense(16->8) -> ReLU
Dense(8->3) -> Sigmoid
```

## Project Structure

```
meg-stroke-intervention/
├── run_complete_pipeline.py    # Single script to run everything
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
│
├── data/
│   ├── download_data.py        # Download real MEG datasets (MNE)
│   ├── synthetic_generator.py  # Generate synthetic stroke MEG data
│   ├── preprocessing.py        # Signal filtering and feature extraction
│   └── data_loader.py          # PyTorch Dataset and DataLoader
│
├── models/
│   ├── meg_stroke_net.py       # Neural network architecture
│   ├── training.py             # Training pipeline with custom loss
│   ├── quantization.py         # INT8 quantization for Arduino
│   └── trained_model.pth       # Saved model weights
│
├── arduino/
│   ├── convert_to_arduino.py   # Convert model to C++ header
│   ├── arduino_inference.cpp   # C++ inference engine (Q7.8 fixed-point)
│   ├── valve_controller.ino    # Arduino sketch with safety systems
│   └── model_weights.h         # Auto-generated quantized weights
│
├── evaluation/
│   ├── validate.py             # Accuracy metrics and diagnostic plots
│   ├── real_time_test.py       # Latency and throughput benchmarks
│   └── arduino_test.py         # Arduino deployment verification
│
└── notebooks/
    ├── data_exploration.ipynb  # MEG data analysis
    └── model_analysis.ipynb    # Model performance analysis
```

## Step-by-Step Usage

### 1. Generate Training Data
```bash
python data/synthetic_generator.py
```
Generates 5400 synthetic MEG samples across 3 conditions (healthy, acute stroke, chronic stroke) with realistic ERD/ERS patterns.

### 2. Train the Model
```bash
python models/training.py
```
Trains MEGStrokeNet with custom stroke-intervention loss (weighted MSE + safety penalty for false positives). Saves to `models/trained_model.pth`.

### 3. Validate Performance
```bash
python evaluation/validate.py
```
Generates confusion matrix, ROC curve, prediction scatter plots, and per-stroke-type metrics in `evaluation/plots/`.

### 4. Quantize for Arduino
```bash
python models/quantization.py
```
Applies INT8 quantization and exports weights as numpy arrays with scale factors.

### 5. Convert to Arduino C++
```bash
python arduino/convert_to_arduino.py
```
Generates `model_weights.h` and `arduino_inference.cpp` with Q7.8 fixed-point arithmetic.

### 6. Run Benchmarks
```bash
python evaluation/real_time_test.py
python evaluation/arduino_test.py
```

## Arduino Deployment

### Hardware Requirements
- Arduino Uno (ATmega328P) or ESP32
- 3x solenoid valves on PWM pins (D3, D5, D6)
- Serial connection for MEG data input

### Upload to Arduino
1. Copy `arduino/model_weights.h`, `arduino_inference.cpp`, and `valve_controller.ino` to an Arduino project folder
2. Open `valve_controller.ino` in Arduino IDE
3. Upload to board

### Serial Protocol
- **Input:** 600 bytes per frame (6 channels x 100 samples, INT8)
- **Output:** JSON telemetry with valve positions, inference outputs, and error state
- **Commands:** `E` = emergency stop, `R` = reset, `S` = status request

## Safety Features

| Feature | Description |
|---------|-------------|
| Max Extension | Valve capped at 80% maximum |
| Rate Limiting | Max 10%/second change rate |
| Signal Loss | Valves ramp to 0 if no data for 500ms |
| Emergency Stop | Immediate valve shutdown via `E` command |
| Watchdog Timer | 2-second hardware watchdog reset |
| False Positive Penalty | Training loss penalizes unnecessary actuation |

## Technical Specifications

| Metric | Target | Achieved |
|--------|--------|----------|
| Parameters | < 50K | ~4.5K |
| Flash (INT8) | < 32KB | ~5KB |
| Inference (CPU) | < 50ms | < 1ms |
| Real-time factor | > 10x | > 100x |
| Input shape | (6, 100) | (6, 100) |
| Output shape | (3,) | (3,) |

## Dependencies

- Python >= 3.9
- PyTorch >= 2.0
- NumPy, SciPy, scikit-learn
- MNE-Python (for real MEG data)
- matplotlib (for visualization)

## Troubleshooting

**Model won't load:** Ensure you've run training first (`python models/training.py`)

**Import errors:** Run `pip install -e .` from the project root

**Arduino memory exceeded:** The model is designed to fit; if you modify the architecture, check with `python evaluation/arduino_test.py`

**Slow training:** Use `--quick` flag with the pipeline runner for faster iteration
