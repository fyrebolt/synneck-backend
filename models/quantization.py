"""
Model quantization pipeline for MEG stroke intervention neural network.

Converts a trained PyTorch MEGStrokeNet to INT8 quantized format suitable for
deployment on resource-constrained microcontrollers (ATmega328P / ESP32).

Pipeline:
    1. Load trained model from models/trained_model.pth
    2. Apply dynamic quantization (linear layers)
    3. Export INT8 numpy weights for Arduino conversion
    4. Generate architecture JSON for Arduino conversion
    5. Validate quantized accuracy vs original
    6. Print memory usage comparison
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from models.meg_stroke_net import MEGStrokeNet


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _model_size_bytes(model: nn.Module) -> int:
    """Return total bytes consumed by model parameters."""
    total = 0
    for p in model.parameters():
        total += p.nelement() * p.element_size()
    return total


def _generate_calibration_data(
    n_samples: int = 200,
    seed: int = 42,
) -> List[torch.Tensor]:
    """Generate synthetic calibration data that mimics MEG signal statistics.

    Uses the same shape as the real model input: (1, 6, 100).
    """
    rng = np.random.RandomState(seed)
    samples: List[torch.Tensor] = []
    for _ in range(n_samples):
        raw = rng.randn(
            1, MEGStrokeNet.NUM_CHANNELS, MEGStrokeNet.NUM_TIMEPOINTS
        ).astype(np.float32)
        # Crude temporal smoothing to approximate MEG autocorrelation
        smoothed = np.cumsum(raw, axis=2)
        smoothed = smoothed / (np.abs(smoothed).max() + 1e-8)
        samples.append(torch.from_numpy(smoothed))
    return samples


def _evaluate_model(
    model: nn.Module,
    data: List[torch.Tensor],
) -> Tuple[np.ndarray, float]:
    """Run inference on calibration data and return (predictions, latency_ms)."""
    model.eval()
    outputs: List[np.ndarray] = []
    start = time.perf_counter()
    with torch.no_grad():
        for x in data:
            out = model(x)
            outputs.append(out.cpu().numpy())
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return np.concatenate(outputs, axis=0), elapsed_ms


# ---------------------------------------------------------------------------
# Core quantization pipeline
# ---------------------------------------------------------------------------

def load_trained_model(model_path: str) -> MEGStrokeNet:
    """Load a trained MEGStrokeNet from disk.

    Args:
        model_path: Path to the saved .pth file.

    Returns:
        The loaded (or fresh) MEGStrokeNet in eval mode.
    """
    model = MEGStrokeNet()
    path = Path(model_path)
    if path.exists():
        state = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"[quantization] Loaded trained model from {path}")
    else:
        print(f"[quantization] WARNING: {path} not found -- using random weights")
    model.eval()
    return model


def apply_dynamic_quantization(model: MEGStrokeNet) -> Optional[nn.Module]:
    """Apply dynamic (weight-only) INT8 quantization to Linear layers.

    Args:
        model: The trained floating-point model.

    Returns:
        A dynamically-quantized copy of the model, or None if the
        quantization engine is not available (e.g. on some Python/PyTorch
        builds missing the ``fbgemm`` or ``qnnpack`` backends).
    """
    try:
        dyn_model = torch.quantization.quantize_dynamic(
            copy.deepcopy(model),
            qconfig_spec={nn.Linear},
            dtype=torch.qint8,
        )
        print("[quantization] Dynamic quantization applied (Linear layers -> INT8)")
        return dyn_model
    except RuntimeError as exc:
        if "NoQEngine" in str(exc) or "quantized" in str(exc).lower():
            print(f"[quantization] WARNING: PyTorch quantization engine not available "
                  f"({exc.__class__.__name__}). Skipping dynamic quantization.")
            print("[quantization] Manual INT8 weight export will still proceed.")
            return None
        raise  # re-raise unexpected RuntimeErrors


def validate_quantized_model(
    original: nn.Module,
    quantized: nn.Module,
    data: List[torch.Tensor],
) -> Dict[str, Any]:
    """Compare quantized model outputs against the original.

    Args:
        original: The floating-point reference model.
        quantized: The quantized model.
        data: Evaluation tensors.

    Returns:
        Dictionary with comparison metrics.
    """
    orig_preds, orig_ms = _evaluate_model(original, data)
    quant_preds, quant_ms = _evaluate_model(quantized, data)

    abs_diff = np.abs(orig_preds - quant_preds)
    max_abs_error = float(abs_diff.max())
    mean_abs_error = float(abs_diff.mean())
    cos_sim = float(
        np.dot(orig_preds.ravel(), quant_preds.ravel())
        / (np.linalg.norm(orig_preds.ravel()) * np.linalg.norm(quant_preds.ravel()) + 1e-12)
    )
    results = {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "cosine_similarity": cos_sim,
        "original_latency_ms": orig_ms,
        "quantized_latency_ms": quant_ms,
        "speedup": orig_ms / (quant_ms + 1e-12),
    }

    print("\n=== Quantization Accuracy Validation ===")
    print(f"  Max absolute error  : {max_abs_error:.6f}")
    print(f"  Mean absolute error : {mean_abs_error:.6f}")
    print(f"  Cosine similarity   : {cos_sim:.6f}")
    print(f"  Original latency    : {orig_ms:.2f} ms  ({len(data)} samples)")
    print(f"  Quantized latency   : {quant_ms:.2f} ms  ({len(data)} samples)")
    print(f"  Speedup             : {results['speedup']:.2f}x")
    return results


# ---------------------------------------------------------------------------
# Weight export
# ---------------------------------------------------------------------------

def export_weights_numpy(model: nn.Module, output_dir: str) -> Dict[str, str]:
    """Export every parameter tensor as a .npy file with INT8 values.

    Floating-point weights are quantized to INT8 using per-tensor
    min/max scaling:
        scale = 127.0 / max(abs(w))
        q     = round(w * scale)

    A companion ``scale_factors.json`` is written so that the Arduino
    converter can reconstruct fixed-point values.

    Args:
        output_dir: Directory to write .npy files into.

    Returns:
        Mapping from parameter name to saved file path.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    saved: Dict[str, str] = {}
    scales: Dict[str, float] = {}

    for name, param in model.named_parameters():
        w = param.detach().cpu().numpy().astype(np.float32)
        abs_max = np.abs(w).max()
        scale = 127.0 / (abs_max + 1e-12)
        w_int8 = np.clip(np.round(w * scale), -128, 127).astype(np.int8)

        safe_name = name.replace(".", "_")
        npy_path = str(out / f"{safe_name}.npy")
        np.save(npy_path, w_int8)
        saved[name] = npy_path
        scales[safe_name] = float(1.0 / scale)  # de-quantization factor

    # Save scale factors alongside the arrays
    scales_path = str(out / "scale_factors.json")
    with open(scales_path, "w") as f:
        json.dump(scales, f, indent=2)
    saved["_scale_factors"] = scales_path

    print(f"[quantization] Exported {len(saved) - 1} weight arrays + scales to {out}")
    return saved


def generate_architecture_json(
    model: MEGStrokeNet,
    output_path: str,
) -> str:
    """Write a JSON file describing the model architecture.

    This file is consumed by ``arduino/convert_to_arduino.py`` to generate
    the C++ inference code and weight header.

    Args:
        model: The (unquantized) model.
        output_path: Where to write the JSON.

    Returns:
        The path that was written.
    """
    layers: List[Dict[str, Any]] = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            layers.append({
                "name": name,
                "type": "Conv1d",
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size[0],
                "stride": module.stride[0],
                "padding": module.padding[0],
                "has_bias": module.bias is not None,
            })
        elif isinstance(module, nn.BatchNorm1d):
            layers.append({
                "name": name,
                "type": "BatchNorm1d",
                "num_features": module.num_features,
            })
        elif isinstance(module, nn.Linear):
            layers.append({
                "name": name,
                "type": "Linear",
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
            })

    arch = {
        "model_name": "MEGStrokeNet",
        "num_channels": MEGStrokeNet.NUM_CHANNELS,
        "num_timepoints": MEGStrokeNet.NUM_TIMEPOINTS,
        "num_outputs": MEGStrokeNet.NUM_OUTPUTS,
        "layers": layers,
        "total_params": sum(p.numel() for p in model.parameters()),
        "total_params_bytes_fp32": _model_size_bytes(model),
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(arch, f, indent=2)
    print(f"[quantization] Architecture JSON written to {output_path}")
    return output_path


def _validate_input_quantization(
    model: nn.Module,
    data: List[torch.Tensor],
) -> Dict[str, Any]:
    """Fallback validation when dynamic quantization is unavailable.

    Quantizes the *inputs* to INT8, dequantizes back, and compares model
    outputs against the original floating-point inputs.  This measures the
    end-to-end accuracy impact of input quantization only.
    """
    model.eval()
    fp32_outputs: List[np.ndarray] = []
    quant_outputs: List[np.ndarray] = []

    with torch.no_grad():
        for x in data:
            fp32_outputs.append(model(x).cpu().numpy())
            # Quantize input
            x_np = x.numpy()
            abs_max = np.abs(x_np).max()
            scale = 127.0 / (abs_max + 1e-12)
            x_int8 = np.clip(np.round(x_np * scale), -128, 127).astype(np.int8)
            x_recovered = torch.from_numpy(x_int8.astype(np.float32) / scale)
            quant_outputs.append(model(x_recovered).cpu().numpy())

    fp32_arr = np.concatenate(fp32_outputs, axis=0)
    quant_arr = np.concatenate(quant_outputs, axis=0)
    abs_diff = np.abs(fp32_arr - quant_arr)

    results: Dict[str, Any] = {
        "max_abs_error": float(abs_diff.max()),
        "mean_abs_error": float(abs_diff.mean()),
        "cosine_similarity": float(
            np.dot(fp32_arr.ravel(), quant_arr.ravel())
            / (np.linalg.norm(fp32_arr.ravel()) * np.linalg.norm(quant_arr.ravel()) + 1e-12)
        ),
        "mode": "input_quantization_only",
    }

    print("\n=== Input Quantization Validation ===")
    print(f"  Max absolute error  : {results['max_abs_error']:.6f}")
    print(f"  Mean absolute error : {results['mean_abs_error']:.6f}")
    print(f"  Cosine similarity   : {results['cosine_similarity']:.6f}")
    return results


# ---------------------------------------------------------------------------
# Memory comparison
# ---------------------------------------------------------------------------

def print_memory_comparison(original_model: nn.Module) -> None:
    """Print model size and Arduino fit estimates."""
    orig_bytes = _model_size_bytes(original_model)
    param_count = sum(p.numel() for p in original_model.parameters())
    int8_weight_bytes = param_count  # 1 byte per weight

    print("\n=== Memory Usage Comparison ===")
    print(f"  Original model (FP32) : {orig_bytes:>8,} bytes  ({orig_bytes / 1024:.1f} KB)")
    print(f"  INT8 weight estimate  : {int8_weight_bytes:>8,} bytes  ({int8_weight_bytes / 1024:.1f} KB)")
    print(f"  Compression ratio     : {orig_bytes / (int8_weight_bytes + 1):.1f}x")

    flash_limit_kb = 32
    print(f"\n  Flash budget (32 KB)  : {'OK' if int8_weight_bytes < flash_limit_kb * 1024 else 'EXCEEDS LIMIT'}")
    print(f"  RAM budget  ( 2 KB)   : (activations checked at C++ level)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_quantization_pipeline(
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute the full quantization pipeline end-to-end.

    Args:
        model_path: Path to the trained .pth checkpoint.
        output_dir: Base directory for all outputs (defaults to ``models/``).

    Returns:
        Dictionary with validation results.
    """
    project_root = Path(__file__).parent.parent
    if model_path is None:
        model_path = str(project_root / "models" / "trained_model.pth")
    if output_dir is None:
        output_dir = str(project_root / "models")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  MEG Stroke Intervention -- Model Quantization Pipeline")
    print("=" * 60)

    # Step 1: Load trained model
    print("\n--- Step 1: Load trained model ---")
    original_model = load_trained_model(model_path)

    # Step 2: Generate calibration data
    print("\n--- Step 2: Generate calibration data ---")
    cal_data = _generate_calibration_data(n_samples=200)
    print(f"  Generated {len(cal_data)} calibration samples  "
          f"shape={cal_data[0].shape}")

    # Step 3: Dynamic quantization
    print("\n--- Step 3: Dynamic quantization ---")
    dyn_model = apply_dynamic_quantization(original_model)

    # Step 4: Validate quantized model accuracy
    print("\n--- Step 4: Validate quantized model ---")
    eval_data = _generate_calibration_data(n_samples=50, seed=99)
    if dyn_model is not None:
        validation_results = validate_quantized_model(original_model, dyn_model, eval_data)
    else:
        # Fall back to input-quantization comparison (quantize inputs only)
        print("  Dynamic quantization unavailable -- validating input quantization only")
        validation_results = _validate_input_quantization(original_model, eval_data)

    # Step 5: Export INT8 numpy weights
    print("\n--- Step 5: Export INT8 weights ---")
    weights_dir = str(out / "quantized_weights")
    export_weights_numpy(original_model, weights_dir)

    # Step 6: Generate architecture JSON
    print("\n--- Step 6: Generate architecture JSON ---")
    arch_json_path = str(out / "model_architecture.json")
    generate_architecture_json(original_model, arch_json_path)

    # Step 7: Memory comparison
    print("\n--- Step 7: Memory comparison ---")
    print_memory_comparison(original_model)

    # Save validation results
    val_path = str(out / "quantization_validation.json")
    with open(val_path, "w") as f:
        json.dump(validation_results, f, indent=2)
    print(f"\n  Validation results saved to {val_path}")

    print("\n" + "=" * 60)
    print("  Quantization pipeline complete.")
    print("=" * 60)

    return validation_results


if __name__ == "__main__":
    run_quantization_pipeline()
