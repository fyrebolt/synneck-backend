#!/usr/bin/env python3
"""
Real-Time Performance Benchmarking for MEG Stroke Intervention System.

Loads the trained model and measures inference latency, throughput, memory
usage, and streaming performance to validate real-time constraints.

Target: < 50ms inference, > 10x real-time factor.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import time
from typing import Dict, List, Optional

import numpy as np
import torch

from models.meg_stroke_net import MEGStrokeNet


def load_model(model_path: Optional[Path] = None) -> MEGStrokeNet:
    """Load the trained model for benchmarking."""
    project_root = Path(__file__).parent.parent
    if model_path is None:
        model_path = project_root / "models" / "trained_model.pth"

    model = MEGStrokeNet()
    if model_path.exists():
        state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
    model.eval()
    return model


def benchmark_single_inference(model: MEGStrokeNet, n_warmup: int = 50,
                                n_trials: int = 500) -> Dict[str, float]:
    """Measure single-sample inference latency.

    Args:
        model: The model to benchmark.
        n_warmup: Warmup iterations.
        n_trials: Measurement iterations.

    Returns:
        Dict with latency stats in ms.
    """
    x = MEGStrokeNet.get_example_input(batch_size=1)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)

    # Measure
    latencies: List[float] = []
    with torch.no_grad():
        for _ in range(n_trials):
            start = time.perf_counter()
            model(x)
            end = time.perf_counter()
            latencies.append((end - start) * 1000.0)

    arr = np.array(latencies)
    return {
        "mean_ms": float(arr.mean()),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "std_ms": float(arr.std()),
    }


def benchmark_throughput(model: MEGStrokeNet, batch_sizes: List[int] = None,
                          n_trials: int = 100) -> Dict[int, float]:
    """Measure throughput (samples/sec) at various batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32, 64]

    results: Dict[int, float] = {}
    for bs in batch_sizes:
        x = MEGStrokeNet.get_example_input(batch_size=bs)

        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(x)

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_trials):
                model(x)
        elapsed = time.perf_counter() - start
        total_samples = bs * n_trials
        results[bs] = total_samples / elapsed

    return results


def benchmark_streaming(model: MEGStrokeNet, duration_sec: float = 5.0,
                         window_ms: float = 500.0) -> Dict[str, float]:
    """Simulate real-time streaming: process windows at 500ms intervals.

    Returns:
        Dict with real-time factor and missed deadlines count.
    """
    window_interval = window_ms / 1000.0  # 0.5s
    n_windows = int(duration_sec / window_interval)

    latencies: List[float] = []
    missed_deadlines = 0
    target_ms = 50.0  # 50ms target

    with torch.no_grad():
        for _ in range(n_windows):
            x = MEGStrokeNet.get_example_input(batch_size=1)
            start = time.perf_counter()
            model(x)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(elapsed_ms)
            if elapsed_ms > target_ms:
                missed_deadlines += 1

    arr = np.array(latencies)
    # Real-time factor: how many times faster than real-time
    processing_time_per_window = arr.mean() / 1000.0
    real_time_factor = window_interval / processing_time_per_window

    return {
        "n_windows": n_windows,
        "mean_latency_ms": float(arr.mean()),
        "max_latency_ms": float(arr.max()),
        "real_time_factor": real_time_factor,
        "missed_deadlines": missed_deadlines,
        "missed_pct": 100.0 * missed_deadlines / n_windows,
        "target_ms": target_ms,
    }


def estimate_memory_usage(model: MEGStrokeNet) -> Dict[str, int]:
    """Estimate memory usage of the model."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.nelement() * b.element_size() for b in model.buffers())

    # Estimate activation memory for single sample
    x = MEGStrokeNet.get_example_input(batch_size=1)
    input_bytes = x.nelement() * x.element_size()

    return {
        "parameter_bytes": param_bytes,
        "buffer_bytes": buffer_bytes,
        "input_bytes": input_bytes,
        "total_model_bytes": param_bytes + buffer_bytes,
    }


def run_realtime_benchmarks(model_path: Optional[Path] = None) -> Dict:
    """Run all real-time performance benchmarks."""
    print("=" * 60)
    print("  Real-Time Performance Benchmarks")
    print("=" * 60)

    model = load_model(model_path)
    param_count = model.count_parameters()
    print(f"\n  Model: MEGStrokeNet ({param_count:,} parameters)")

    # 1. Single inference latency
    print("\n--- Single Inference Latency ---")
    latency = benchmark_single_inference(model)
    for k, v in latency.items():
        print(f"  {k:>15}: {v:.4f}")
    meets_target = latency["p95_ms"] < 50.0
    print(f"  Target (<50ms p95): {'PASS' if meets_target else 'FAIL'}")

    # 2. Throughput
    print("\n--- Throughput (samples/sec) ---")
    throughput = benchmark_throughput(model)
    for bs, sps in throughput.items():
        print(f"  batch_size={bs:>3}: {sps:>10.1f} samples/sec")

    # 3. Streaming simulation
    print("\n--- Streaming Simulation (5s) ---")
    streaming = benchmark_streaming(model)
    print(f"  Windows processed:  {streaming['n_windows']}")
    print(f"  Mean latency:       {streaming['mean_latency_ms']:.4f} ms")
    print(f"  Max latency:        {streaming['max_latency_ms']:.4f} ms")
    print(f"  Real-time factor:   {streaming['real_time_factor']:.1f}x")
    print(f"  Missed deadlines:   {streaming['missed_deadlines']} ({streaming['missed_pct']:.1f}%)")
    rt_ok = streaming["real_time_factor"] > 10.0
    print(f"  Target (>10x RT):   {'PASS' if rt_ok else 'FAIL'}")

    # 4. Memory
    print("\n--- Memory Usage ---")
    mem = estimate_memory_usage(model)
    for k, v in mem.items():
        print(f"  {k:>25}: {v:>8,} bytes  ({v / 1024:.1f} KB)")

    # Arduino estimate
    timing = model.inference_time_estimate()
    print(f"\n--- Arduino Estimate (16 MHz ATmega328P) ---")
    print(f"  Estimated FLOPs:    {timing['estimated_flops']:,}")
    print(f"  Estimated latency:  {timing['estimated_latency_ms']} ms")

    print("\n" + "=" * 60)
    print("  Benchmarks Complete")
    print("=" * 60)

    return {
        "latency": latency,
        "throughput": throughput,
        "streaming": streaming,
        "memory": mem,
        "arduino_estimate": timing,
    }


if __name__ == "__main__":
    run_realtime_benchmarks()
