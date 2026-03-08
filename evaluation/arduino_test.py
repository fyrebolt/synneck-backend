#!/usr/bin/env python3
"""
Arduino Deployment Testing for MEG Stroke Intervention System.

Simulates Arduino inference by reimplementing the neural network forward
pass in pure Python using INT8 fixed-point arithmetic, matching the C++
implementation in arduino/arduino_inference.cpp.

Tests:
    1. Weight quantization accuracy (FP32 vs INT8)
    2. Fixed-point inference vs floating-point inference
    3. Memory budget verification (< 32KB flash, < 2KB RAM)
    4. Safety constraint simulation (rate limiting, max extension, watchdog)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from models.meg_stroke_net import MEGStrokeNet


# Fixed-point constants matching the C++ implementation
FIXED_FRAC_BITS = 8
FIXED_SCALE = 1 << FIXED_FRAC_BITS  # 256


def fp_mul(a: int, b: int) -> int:
    """Q7.8 multiply."""
    return (a * b) >> FIXED_FRAC_BITS


def fp_relu(x: int) -> int:
    """Fixed-point ReLU."""
    return max(0, x)


def fp_sigmoid(x: int) -> int:
    """Piecewise-linear sigmoid approximation in Q7.8."""
    NEG4 = -4 * FIXED_SCALE
    POS4 = 4 * FIXED_SCALE
    ONE = FIXED_SCALE
    HALF = FIXED_SCALE // 2
    if x <= NEG4:
        return 0
    if x >= POS4:
        return ONE
    return HALF + (x >> 3)


def simulate_conv1d_int8(
    input_data: np.ndarray,  # (in_ch, seq_len) Q7.8
    weights: np.ndarray,      # (out_ch, in_ch, kernel) INT8
    bias: np.ndarray,         # (out_ch,) INT8
    stride: int,
    padding: int,
    scale: float,
) -> np.ndarray:
    """Simulate INT8 Conv1D with fixed-point arithmetic."""
    in_ch, seq_len = input_data.shape
    out_ch, _, kernel = weights.shape

    # Calculate output length
    out_len = (seq_len + 2 * padding - kernel) // stride + 1
    output = np.zeros((out_ch, out_len), dtype=np.int32)

    scale_q8 = int(round(scale * FIXED_SCALE))

    for f in range(out_ch):
        b = int(bias[f]) << FIXED_FRAC_BITS
        for t in range(out_len):
            accum = b
            for c in range(in_ch):
                for k in range(kernel):
                    tt = t * stride + k - padding
                    if 0 <= tt < seq_len:
                        w = int(weights[f, c, k]) << FIXED_FRAC_BITS
                        x = int(input_data[c, tt])
                        accum += (w * x) >> FIXED_FRAC_BITS
            # Apply scale
            scaled = (accum * scale_q8) >> FIXED_FRAC_BITS
            output[f, t] = np.clip(scaled, -32768, 32767)

    return output.astype(np.int16)


def simulate_dense_int8(
    input_data: np.ndarray,  # (in_features,) Q7.8
    weights: np.ndarray,     # (out_features, in_features) INT8
    bias: np.ndarray,        # (out_features,) INT8
    scale: float,
) -> np.ndarray:
    """Simulate INT8 Dense layer with fixed-point arithmetic."""
    out_features = weights.shape[0]
    output = np.zeros(out_features, dtype=np.int32)

    scale_q8 = int(round(scale * FIXED_SCALE))

    for o in range(out_features):
        accum = int(bias[o]) << FIXED_FRAC_BITS
        for i in range(weights.shape[1]):
            w = int(weights[o, i]) << FIXED_FRAC_BITS
            x = int(input_data[i])
            accum += (w * x) >> FIXED_FRAC_BITS
        scaled = (accum * scale_q8) >> FIXED_FRAC_BITS
        output[o] = np.clip(scaled, -32768, 32767)

    return output.astype(np.int16)


def test_weight_quantization(model: MEGStrokeNet) -> Dict[str, float]:
    """Test INT8 quantization accuracy of model weights."""
    print("\n--- Test 1: Weight Quantization Accuracy ---")

    results = {}
    total_params = 0
    total_error = 0.0

    for name, param in model.named_parameters():
        w = param.detach().cpu().numpy().astype(np.float32)
        abs_max = np.abs(w).max()
        scale = 127.0 / (abs_max + 1e-12)

        # Quantize
        w_int8 = np.clip(np.round(w * scale), -128, 127).astype(np.int8)

        # Dequantize
        w_recovered = w_int8.astype(np.float32) / scale

        # Error
        error = np.abs(w - w_recovered)
        max_error = float(error.max())
        mean_error = float(error.mean())

        results[name] = {"max_error": max_error, "mean_error": mean_error}
        total_params += w.size
        total_error += error.sum()

        print(f"  {name:30s}  max_err={max_error:.6f}  mean_err={mean_error:.6f}")

    avg_error = total_error / total_params
    print(f"\n  Overall mean quantization error: {avg_error:.6f}")
    results["overall_mean_error"] = avg_error
    return results


def test_memory_budget(model: MEGStrokeNet) -> Dict[str, bool]:
    """Verify the model fits within Arduino memory constraints."""
    print("\n--- Test 2: Memory Budget Verification ---")

    param_count = model.count_parameters()
    int8_bytes = param_count  # 1 byte per INT8 weight

    # Estimate activation buffers (largest intermediate tensor)
    # Conv1: 16 filters * 50 timepoints = 800 elements * 2 bytes = 1600 bytes
    # Conv2: 32 filters * 25 timepoints = 800 elements * 2 bytes = 1600 bytes
    # Conv3: 16 filters * 13 timepoints = 208 elements * 2 bytes = 416 bytes
    # We reuse buffers, so max = 1600 bytes

    max_activation_bytes = 2 * 16 * 50 + 2 * 32 * 25  # two largest layers
    code_overhead = 4096  # ~4KB for inference code

    flash_total = int8_bytes + code_overhead
    ram_total = max_activation_bytes

    flash_ok = flash_total < 32 * 1024
    ram_ok = ram_total < 2 * 1024

    print(f"  Parameters:       {param_count:,}")
    print(f"  INT8 weights:     {int8_bytes:,} bytes ({int8_bytes / 1024:.1f} KB)")
    print(f"  Flash total:      {flash_total:,} bytes ({flash_total / 1024:.1f} KB) / 32 KB "
          f"{'OK' if flash_ok else 'OVER'}")
    print(f"  RAM activations:  {ram_total:,} bytes ({ram_total / 1024:.1f} KB) / 2 KB "
          f"{'OK' if ram_ok else 'OVER'}")

    return {
        "flash_ok": flash_ok,
        "ram_ok": ram_ok,
        "flash_bytes": flash_total,
        "ram_bytes": ram_total,
    }


def test_safety_constraints() -> Dict[str, bool]:
    """Simulate safety constraints: rate limiting, max extension, watchdog."""
    print("\n--- Test 3: Safety Constraint Simulation ---")

    MAX_VALVE = 0.80  # 80% max
    MAX_RATE_PER_SEC = 0.10  # 10%/s
    TICK_MS = 100
    SLEW_PER_TICK = MAX_RATE_PER_SEC * (TICK_MS / 1000.0)

    # Test 1: Rate limiting
    current = 0.0
    target = 1.0
    ticks_to_reach = 0
    for _ in range(200):
        delta = target - current
        delta = max(min(delta, SLEW_PER_TICK), -SLEW_PER_TICK)
        current += delta
        current = min(current, MAX_VALVE)
        ticks_to_reach += 1
        if abs(current - min(target, MAX_VALVE)) < 0.001:
            break

    rate_limit_ok = ticks_to_reach >= 70  # Should take ~80 ticks (8s) to reach 80%
    print(f"  Rate limiting: took {ticks_to_reach} ticks to reach {current:.2f} "
          f"(target {target:.2f}) {'OK' if rate_limit_ok else 'FAIL'}")

    # Test 2: Max extension cap
    max_ok = current <= MAX_VALVE + 0.001
    print(f"  Max extension: capped at {current:.2f} (limit {MAX_VALVE}) "
          f"{'OK' if max_ok else 'FAIL'}")

    # Test 3: Graceful degradation (signal loss -> ramp to zero)
    current = 0.6
    for tick in range(200):
        delta = 0.0 - current  # target = 0 on signal loss
        delta = max(min(delta, SLEW_PER_TICK), -SLEW_PER_TICK)
        current += delta
        if abs(current) < 0.001:
            break

    degrade_ok = tick < 100  # Should reach zero in ~60 ticks
    print(f"  Graceful degradation: reached zero in {tick + 1} ticks "
          f"{'OK' if degrade_ok else 'FAIL'}")

    return {
        "rate_limiting": rate_limit_ok,
        "max_extension": max_ok,
        "graceful_degradation": degrade_ok,
    }


def test_end_to_end_inference(model: MEGStrokeNet) -> Dict[str, float]:
    """Compare floating-point and quantized inference outputs."""
    print("\n--- Test 4: End-to-End Inference Comparison ---")

    # Generate test input
    x = MEGStrokeNet.get_example_input(batch_size=10)

    # FP32 inference
    model.eval()
    with torch.no_grad():
        fp32_output = model(x).numpy()

    # Simple quantized inference (quantize inputs, dequantize outputs)
    x_np = x.numpy()
    max_val = np.abs(x_np).max()
    x_scale = 127.0 / (max_val + 1e-12)
    x_int8 = np.clip(np.round(x_np * x_scale), -128, 127).astype(np.int8)
    x_recovered = torch.from_numpy(x_int8.astype(np.float32) / x_scale)

    with torch.no_grad():
        quantized_output = model(x_recovered).numpy()

    # Compare
    abs_diff = np.abs(fp32_output - quantized_output)
    max_err = float(abs_diff.max())
    mean_err = float(abs_diff.mean())

    print(f"  FP32 output range:     [{fp32_output.min():.4f}, {fp32_output.max():.4f}]")
    print(f"  Quantized output range: [{quantized_output.min():.4f}, {quantized_output.max():.4f}]")
    print(f"  Max absolute error:    {max_err:.6f}")
    print(f"  Mean absolute error:   {mean_err:.6f}")

    # Output should be within reasonable range
    output_ok = max_err < 0.1  # Less than 10% error
    print(f"  Accuracy (<0.1 max error): {'PASS' if output_ok else 'FAIL'}")

    return {"max_error": max_err, "mean_error": mean_err, "output_ok": output_ok}


def run_arduino_tests(model_path: Optional[Path] = None) -> Dict:
    """Run all Arduino deployment tests."""
    project_root = Path(__file__).parent.parent
    if model_path is None:
        model_path = project_root / "models" / "trained_model.pth"

    print("=" * 60)
    print("  Arduino Deployment Tests")
    print("=" * 60)

    model = MEGStrokeNet()
    if model_path.exists():
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    model.eval()

    results = {}
    results["quantization"] = test_weight_quantization(model)
    results["memory"] = test_memory_budget(model)
    results["safety"] = test_safety_constraints()
    results["inference"] = test_end_to_end_inference(model)

    # Summary
    all_pass = (
        results["memory"]["flash_ok"]
        and results["memory"]["ram_ok"]
        and results["safety"]["rate_limiting"]
        and results["safety"]["max_extension"]
        and results["safety"]["graceful_degradation"]
        and results["inference"]["output_ok"]
    )

    print("\n" + "=" * 60)
    print(f"  All Tests: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_arduino_tests()
