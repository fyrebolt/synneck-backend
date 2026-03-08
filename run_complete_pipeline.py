#!/usr/bin/env python3
"""
MEG Stroke Intervention -- Complete Pipeline Runner

Single script that executes the entire system end-to-end:
    1. Generate synthetic training data
    2. Train the neural network
    3. Validate model performance
    4. Quantize for Arduino deployment
    5. Convert to Arduino C++ code
    6. Run real-time performance benchmarks
    7. Run Arduino deployment tests
    8. Generate final report

Usage:
    python run_complete_pipeline.py
    python run_complete_pipeline.py --quick     # fast mode for testing
    python run_complete_pipeline.py --skip-data  # skip data generation
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import time
from typing import Dict, Any


def step_banner(step_num: int, total: int, title: str) -> None:
    """Print a formatted step banner."""
    print(f"\n{'=' * 70}")
    print(f"  STEP {step_num}/{total}: {title}")
    print(f"{'=' * 70}\n")


def run_pipeline(quick: bool = False, skip_data: bool = False) -> Dict[str, Any]:
    """Execute the complete pipeline.

    Args:
        quick: If True, use smaller dataset and fewer epochs.
        skip_data: If True, skip data generation (use existing data).

    Returns:
        Dictionary with all results.
    """
    total_steps = 7
    results: Dict[str, Any] = {}
    pipeline_start = time.time()

    n_samples = 1000 if quick else 5400
    n_epochs = 20 if quick else 100
    patience = 5 if quick else 15

    print("=" * 70)
    print("  MEG STROKE INTERVENTION -- COMPLETE PIPELINE")
    print("=" * 70)
    print(f"  Mode:       {'QUICK' if quick else 'FULL'}")
    print(f"  Samples:    {n_samples}")
    print(f"  Max epochs: {n_epochs}")
    print(f"  Patience:   {patience}")
    print("=" * 70)

    # ---------------------------------------------------------------
    # Step 1: Generate synthetic data
    # ---------------------------------------------------------------
    step_banner(1, total_steps, "GENERATE SYNTHETIC DATA")

    if skip_data and (PROJECT_ROOT / "data" / "synthetic" / "train.npz").exists():
        print("  Skipping data generation (--skip-data flag, existing data found)")
        results["data"] = {"status": "skipped"}
    else:
        from data.synthetic_generator import generate_dataset, save_dataset

        dataset = generate_dataset(n_samples=n_samples, seed=42)
        output_dir = PROJECT_ROOT / "data" / "synthetic"
        paths = save_dataset(dataset, output_dir)

        results["data"] = {
            "status": "generated",
            "n_samples": n_samples,
            "data_shape": list(dataset["data"].shape),
            "labels_shape": list(dataset["labels"].shape),
        }
        print(f"\n  Generated {n_samples} samples -> {output_dir}")

    # ---------------------------------------------------------------
    # Step 2: Train the neural network
    # ---------------------------------------------------------------
    step_banner(2, total_steps, "TRAIN NEURAL NETWORK")

    from models.training import train

    train_results = train(
        num_epochs=n_epochs,
        batch_size=64,
        learning_rate=1e-3,
        patience=patience,
        max_grad_norm=1.0,
        num_samples=n_samples,
        val_fraction=0.2,
        seed=42,
        save_dir=PROJECT_ROOT / "models",
    )

    results["training"] = {
        "model_path": train_results["model_path"],
        "best_val_loss": train_results["best_val_loss"],
        "num_epochs_trained": train_results["num_epochs_trained"],
        "final_metrics": train_results["final_metrics"],
    }

    # ---------------------------------------------------------------
    # Step 3: Validate model
    # ---------------------------------------------------------------
    step_banner(3, total_steps, "VALIDATE MODEL")

    from evaluation.validate import run_validation

    val_results = run_validation(
        model_path=PROJECT_ROOT / "models" / "trained_model.pth",
        n_per_type=100 if quick else 200,
        seed=42,
    )
    results["validation"] = val_results

    # ---------------------------------------------------------------
    # Step 4: Quantize model
    # ---------------------------------------------------------------
    step_banner(4, total_steps, "QUANTIZE MODEL FOR ARDUINO")

    from models.quantization import run_quantization_pipeline

    quant_results = run_quantization_pipeline(
        model_path=str(PROJECT_ROOT / "models" / "trained_model.pth"),
        output_dir=str(PROJECT_ROOT / "models"),
    )
    results["quantization"] = quant_results

    # ---------------------------------------------------------------
    # Step 5: Convert to Arduino C++
    # ---------------------------------------------------------------
    step_banner(5, total_steps, "CONVERT TO ARDUINO C++")

    from arduino.convert_to_arduino import run_conversion

    try:
        run_conversion(
            weights_dir=str(PROJECT_ROOT / "models" / "quantized_weights"),
            arch_path=str(PROJECT_ROOT / "models" / "model_architecture.json"),
            output_dir=str(PROJECT_ROOT / "arduino"),
        )
        results["arduino_conversion"] = {"status": "success"}
    except Exception as e:
        print(f"  Arduino conversion warning: {e}")
        results["arduino_conversion"] = {"status": "warning", "error": str(e)}

    # ---------------------------------------------------------------
    # Step 6: Real-time performance benchmarks
    # ---------------------------------------------------------------
    step_banner(6, total_steps, "REAL-TIME PERFORMANCE BENCHMARKS")

    from evaluation.real_time_test import run_realtime_benchmarks

    rt_results = run_realtime_benchmarks(
        model_path=PROJECT_ROOT / "models" / "trained_model.pth"
    )
    results["realtime"] = {
        "mean_latency_ms": rt_results["latency"]["mean_ms"],
        "p95_latency_ms": rt_results["latency"]["p95_ms"],
        "real_time_factor": rt_results["streaming"]["real_time_factor"],
    }

    # ---------------------------------------------------------------
    # Step 7: Arduino deployment tests
    # ---------------------------------------------------------------
    step_banner(7, total_steps, "ARDUINO DEPLOYMENT TESTS")

    from evaluation.arduino_test import run_arduino_tests

    arduino_results = run_arduino_tests(
        model_path=PROJECT_ROOT / "models" / "trained_model.pth"
    )
    results["arduino_tests"] = {
        "flash_ok": arduino_results["memory"]["flash_ok"],
        "ram_ok": arduino_results["memory"]["ram_ok"],
        "safety_ok": all(arduino_results["safety"].values()),
        "inference_ok": arduino_results["inference"]["output_ok"],
    }

    # ---------------------------------------------------------------
    # Final Report
    # ---------------------------------------------------------------
    pipeline_elapsed = time.time() - pipeline_start

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE -- FINAL REPORT")
    print("=" * 70)
    print(f"\n  Total time: {pipeline_elapsed:.1f}s ({pipeline_elapsed / 60:.1f} min)")

    print("\n  --- Training ---")
    print(f"  Best val loss:     {results['training']['best_val_loss']:.6f}")
    print(f"  Epochs trained:    {results['training']['num_epochs_trained']}")

    if "overall_classification" in results.get("validation", {}):
        cls = results["validation"]["overall_classification"]
        print(f"\n  --- Validation ---")
        print(f"  Accuracy:          {cls.get('accuracy', 'N/A')}")
        print(f"  AUC:               {cls.get('auc', 'N/A')}")
        print(f"  FPR:               {cls.get('fpr', 'N/A')}")

    print(f"\n  --- Real-Time Performance ---")
    print(f"  Mean latency:      {results['realtime']['mean_latency_ms']:.4f} ms")
    print(f"  P95 latency:       {results['realtime']['p95_latency_ms']:.4f} ms")
    print(f"  Real-time factor:  {results['realtime']['real_time_factor']:.1f}x")

    print(f"\n  --- Arduino Deployment ---")
    print(f"  Flash budget:      {'PASS' if results['arduino_tests']['flash_ok'] else 'FAIL'}")
    print(f"  RAM budget:        {'PASS' if results['arduino_tests']['ram_ok'] else 'FAIL'}")
    print(f"  Safety checks:     {'PASS' if results['arduino_tests']['safety_ok'] else 'FAIL'}")
    print(f"  Inference check:   {'PASS' if results['arduino_tests']['inference_ok'] else 'FAIL'}")

    # Save report
    report_path = PROJECT_ROOT / "pipeline_report.json"
    # Convert non-serializable values
    serializable_results = json.loads(
        json.dumps(results, default=lambda o: str(o) if not isinstance(o, (int, float, str, bool, list, dict, type(None))) else o)
    )
    with open(report_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n  Report saved: {report_path}")

    print("\n" + "=" * 70)
    print("  HACKATHON DEMO READY!")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="MEG Stroke Intervention -- Complete Pipeline"
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: smaller dataset, fewer epochs"
    )
    parser.add_argument(
        "--skip-data", action="store_true",
        help="Skip data generation (use existing data)"
    )
    args = parser.parse_args()

    run_pipeline(quick=args.quick, skip_data=args.skip_data)


if __name__ == "__main__":
    main()
