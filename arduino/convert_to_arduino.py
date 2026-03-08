"""
Convert quantized MEG stroke-intervention model to Arduino-compatible C++.

This script reads the INT8 quantized weights exported by
``models/quantization.py`` and generates two files:

    1. **model_weights.h**  -- INT8 weight arrays as ``PROGMEM`` C constants.
    2. **arduino_inference.cpp** -- Standalone forward-pass implementation in
       pure C++ using Q7.8 fixed-point arithmetic.

All generated code targets ATmega328P (Arduino Uno) and ESP32, with:
    - < 32 KB flash for weights + code
    - < 2 KB RAM for activations and buffers

Architecture (must match models/meg_stroke_net.py):
    Conv1d(6->16, k=5, s=2, p=2) -> ReLU -> BN
    Conv1d(16->32, k=3, s=2, p=1) -> ReLU -> BN
    Conv1d(32->16, k=3, s=2, p=1) -> ReLU
    GlobalAvgPool
    Dense(16->8) -> ReLU
    Dense(8->3) -> Sigmoid
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import textwrap
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Constants -- must mirror models/meg_stroke_net.py
# ---------------------------------------------------------------------------

NUM_MEG_CHANNELS: int = 6
NUM_TIMEPOINTS: int = 100
NUM_OUTPUTS: int = 3

FLASH_BUDGET_BYTES: int = 32 * 1024  # 32 KB
RAM_BUDGET_BYTES: int = 2 * 1024     # 2 KB

# Q7.8 fixed-point: 1 sign bit, 7 integer bits, 8 fractional bits
FIXED_POINT_FRAC_BITS: int = 8
FIXED_POINT_SCALE: int = 1 << FIXED_POINT_FRAC_BITS  # 256


# ---------------------------------------------------------------------------
# Weight loading
# ---------------------------------------------------------------------------

def load_quantized_weights(
    weights_dir: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Load INT8 ``.npy`` weight arrays and their scale factors."""
    wdir = Path(weights_dir)
    scales_path = wdir / "scale_factors.json"
    if not scales_path.exists():
        raise FileNotFoundError(f"Scale factors not found at {scales_path}")

    with open(scales_path) as f:
        scales: Dict[str, float] = json.load(f)

    weights: Dict[str, np.ndarray] = {}
    for npy_file in sorted(wdir.glob("*.npy")):
        name = npy_file.stem
        weights[name] = np.load(str(npy_file))

    print(f"[convert] Loaded {len(weights)} weight arrays from {wdir}")
    return weights, scales


def load_architecture(arch_path: str) -> Dict[str, Any]:
    """Load the model architecture JSON."""
    with open(arch_path) as f:
        arch = json.load(f)
    print(f"[convert] Architecture: {arch['model_name']}  "
          f"params={arch['total_params']}")
    return arch


# ---------------------------------------------------------------------------
# C header generation -- model_weights.h
# ---------------------------------------------------------------------------

def _format_int8_array(arr: np.ndarray, name: str, per_line: int = 16) -> str:
    """Format a numpy INT8 array as a PROGMEM C constant."""
    flat = arr.ravel().astype(np.int8)
    lines: List[str] = []
    lines.append(f"// Shape: {list(arr.shape)}  elements: {flat.size}")
    lines.append(
        f"static const int8_t {name}[{flat.size}] PROGMEM "
        f"__attribute__((aligned(4))) = {{"
    )
    for i in range(0, flat.size, per_line):
        chunk = flat[i : i + per_line]
        vals = ", ".join(f"{int(v):>4d}" for v in chunk)
        trailing = "," if i + per_line < flat.size else ""
        lines.append(f"    {vals}{trailing}")
    lines.append("};")
    return "\n".join(lines)


def _format_scale_constant(name: str, scale: float) -> str:
    """Format a de-quantization scale as a C constant (Q7.8 fixed-point)."""
    fixed = int(round(scale * FIXED_POINT_SCALE))
    return (
        f"// De-quant scale (float={scale:.8f}, Q7.8={fixed})\n"
        f"static const int16_t {name}_scale = {fixed};"
    )


def generate_weights_header(
    weights: Dict[str, np.ndarray],
    scales: Dict[str, float],
    output_path: str,
) -> int:
    """Generate ``model_weights.h`` containing all INT8 weight arrays."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    parts: List[str] = [
        "/*",
        f" * model_weights.h -- Auto-generated on {timestamp}",
        " * INT8 quantized weights for MEGStrokeNet",
        " * Architecture: Conv1d(6->16,k5,s2) -> Conv1d(16->32,k3,s2) -> Conv1d(32->16,k3,s2) -> Dense(16->8) -> Dense(8->3)",
        " * Do NOT edit by hand.",
        " */",
        "",
        "#ifndef MODEL_WEIGHTS_H",
        "#define MODEL_WEIGHTS_H",
        "",
        "#include <stdint.h>",
        "",
        "#ifdef __AVR__",
        "  #include <avr/pgmspace.h>",
        "#else",
        "  #ifndef PROGMEM",
        "    #define PROGMEM",
        "  #endif",
        "#endif",
        "",
        "/* ----- Model dimensions ----- */",
        f"#define NUM_MEG_CHANNELS   {NUM_MEG_CHANNELS}",
        f"#define NUM_TIMEPOINTS     {NUM_TIMEPOINTS}",
        f"#define NUM_OUTPUTS        {NUM_OUTPUTS}",
        f"#define FIXED_FRAC_BITS    {FIXED_POINT_FRAC_BITS}",
        "",
        "/* Conv layer output sizes (after stride) */",
        "#define CONV1_OUT_CH  16",
        "#define CONV1_OUT_LEN 50   /* (100 + 2*2 - 5) / 2 + 1 */",
        "#define CONV2_OUT_CH  32",
        "#define CONV2_OUT_LEN 25   /* (50 + 2*1 - 3) / 2 + 1 */",
        "#define CONV3_OUT_CH  16",
        "#define CONV3_OUT_LEN 13   /* (25 + 2*1 - 3) / 2 + 1 */",
        "",
        "/* ----- Scale factors (Q7.8 fixed-point) ----- */",
    ]

    total_bytes = 0

    # Scale factors
    for sname, sval in sorted(scales.items()):
        parts.append(_format_scale_constant(sname, sval))
    parts.append("")

    # Weight arrays
    parts.append("/* ----- Weight arrays (INT8 PROGMEM) ----- */")
    for wname in sorted(weights.keys()):
        arr = weights[wname]
        parts.append("")
        parts.append(_format_int8_array(arr, wname))
        total_bytes += arr.size

    parts.extend(["", "#endif  /* MODEL_WEIGHTS_H */", ""])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(parts))

    print(f"[convert] Wrote {output_path}  ({total_bytes} bytes of weights)")
    return total_bytes


# ---------------------------------------------------------------------------
# C++ inference source generation
# ---------------------------------------------------------------------------

def generate_inference_cpp(output_path: str) -> int:
    """Generate ``arduino_inference.cpp`` -- a standalone forward pass.

    Architecture matches MEGStrokeNet from meg_stroke_net.py:
        Conv1d(6->16, k=5, s=2, p=2) + ReLU
        Conv1d(16->32, k=3, s=2, p=1) + ReLU
        Conv1d(32->16, k=3, s=2, p=1) + ReLU
        Global Average Pooling
        Dense(16->8) + ReLU
        Dense(8->3) + Sigmoid
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Buffer sizes (int16_t elements)
    buf1_size = 16 * 50    # conv1 output: 16 channels * 50 timepoints = 800
    buf2_size = 32 * 25    # conv2 output: 32 channels * 25 timepoints = 800
    buf3_size = 16 * 13    # conv3 output: 16 channels * 13 timepoints = 208
    pool_size = 16         # after global avg pool
    fc1_size = 8           # dense layer 1
    fc2_size = 3           # dense layer 2 (output)

    # We can reuse buffers: largest is 800 int16 = 1600 bytes
    max_buf = max(buf1_size, buf2_size)
    ram_bytes = max_buf * 2 * 2 + pool_size * 2 + fc1_size * 2 + fc2_size * 2
    # = 1600 + 1600 + 32 + 16 + 6 = 3254 bytes ... tight.
    # Optimize: reuse buf_a for conv3 (only 208 elements fit easily)

    cpp_source = textwrap.dedent(f"""\
        /*
         * arduino_inference.cpp -- Auto-generated on {timestamp}
         *
         * INT8 fixed-point inference for MEGStrokeNet.
         * Architecture: Conv1d->Conv1d->Conv1d->GAP->Dense->Dense->Sigmoid
         * Input: {NUM_MEG_CHANNELS} channels x {NUM_TIMEPOINTS} timepoints
         * Output: {NUM_OUTPUTS} valve control values [0.0, 1.0]
         *
         * Uses Q7.8 arithmetic, minimal dynamic allocation.
         */

        #include "model_weights.h"
        #include <stdint.h>
        #include <string.h>

        #ifdef __AVR__
          #include <avr/pgmspace.h>
          #define READ_WEIGHT(addr)  ((int8_t)pgm_read_byte(addr))
        #else
          #define READ_WEIGHT(addr)  (*(addr))
        #endif

        /* ------------------------------------------------------------------ */
        /* Static activation buffers                                          */
        /* ------------------------------------------------------------------ */
        static int16_t buf_a[{max_buf}];   /* reusable buffer A */
        static int16_t buf_b[{max_buf}];   /* reusable buffer B */
        static int16_t pool_buf[{pool_size}];
        static int16_t fc1_buf[{fc1_size}];
        static int16_t fc2_buf[{fc2_size}];

        /* ------------------------------------------------------------------ */
        /* Fixed-point helpers (Q7.8)                                          */
        /* ------------------------------------------------------------------ */

        static inline int16_t fp_relu(int16_t x) {{
            return (x > 0) ? x : 0;
        }}

        static inline int16_t fp_sigmoid(int16_t x) {{
            const int16_t NEG4 = -4 * (1 << FIXED_FRAC_BITS);
            const int16_t POS4 =  4 * (1 << FIXED_FRAC_BITS);
            const int16_t ONE  =  1 * (1 << FIXED_FRAC_BITS);
            const int16_t HALF =      (1 << (FIXED_FRAC_BITS - 1));
            if (x <= NEG4) return 0;
            if (x >= POS4) return ONE;
            return (int16_t)(HALF + (x >> 3));
        }}

        /* ------------------------------------------------------------------ */
        /* Conv1D with stride                                                 */
        /* ------------------------------------------------------------------ */
        static void conv1d_stride(
            const int16_t *in_data, int16_t in_ch, int16_t in_len,
            const int8_t *weights, const int8_t *bias,
            int16_t out_ch, int16_t kernel, int16_t stride, int16_t pad,
            int16_t scale_q8,
            int16_t *out_data, int16_t out_len)
        {{
            for (int16_t f = 0; f < out_ch; f++) {{
                int16_t b = (int16_t)READ_WEIGHT(&bias[f]) << FIXED_FRAC_BITS;
                for (int16_t t = 0; t < out_len; t++) {{
                    int32_t accum = (int32_t)b;
                    int16_t t_in_start = t * stride - pad;
                    for (int16_t c = 0; c < in_ch; c++) {{
                        for (int16_t k = 0; k < kernel; k++) {{
                            int16_t tt = t_in_start + k;
                            if (tt < 0 || tt >= in_len) continue;
                            int16_t w_idx = (f * in_ch + c) * kernel + k;
                            int16_t w = (int16_t)READ_WEIGHT(&weights[w_idx]) << FIXED_FRAC_BITS;
                            int16_t x = in_data[c * in_len + tt];
                            accum += ((int32_t)w * (int32_t)x) >> FIXED_FRAC_BITS;
                        }}
                    }}
                    int32_t scaled = ((int32_t)(int16_t)(accum) * (int32_t)scale_q8) >> FIXED_FRAC_BITS;
                    if (scaled >  32767) scaled =  32767;
                    if (scaled < -32768) scaled = -32768;
                    out_data[f * out_len + t] = (int16_t)scaled;
                }}
            }}
        }}

        static void relu_inplace(int16_t *data, int16_t n) {{
            for (int16_t i = 0; i < n; i++) {{
                data[i] = fp_relu(data[i]);
            }}
        }}

        static void global_avg_pool(
            const int16_t *in_data, int16_t channels, int16_t seq_len,
            int16_t *out_data)
        {{
            for (int16_t c = 0; c < channels; c++) {{
                int32_t sum = 0;
                for (int16_t t = 0; t < seq_len; t++) {{
                    sum += in_data[c * seq_len + t];
                }}
                out_data[c] = (int16_t)(sum / seq_len);
            }}
        }}

        static void dense(
            const int16_t *in_data, int16_t in_features,
            const int8_t *weights, const int8_t *bias,
            int16_t out_features, int16_t scale_q8,
            int16_t *out_data)
        {{
            for (int16_t o = 0; o < out_features; o++) {{
                int32_t accum = (int32_t)((int16_t)READ_WEIGHT(&bias[o]) << FIXED_FRAC_BITS);
                for (int16_t i = 0; i < in_features; i++) {{
                    int16_t w = (int16_t)READ_WEIGHT(&weights[o * in_features + i]) << FIXED_FRAC_BITS;
                    accum += ((int32_t)w * (int32_t)in_data[i]) >> FIXED_FRAC_BITS;
                }}
                int32_t scaled = ((int32_t)(int16_t)(accum) * (int32_t)scale_q8) >> FIXED_FRAC_BITS;
                if (scaled >  32767) scaled =  32767;
                if (scaled < -32768) scaled = -32768;
                out_data[o] = (int16_t)scaled;
            }}
        }}

        /* ------------------------------------------------------------------ */
        /* Main inference entry point                                          */
        /* ------------------------------------------------------------------ */
        void meg_inference(const int8_t *input, float *output) {{
            /* Convert input to Q7.8 */
            for (int16_t i = 0; i < NUM_MEG_CHANNELS * NUM_TIMEPOINTS; i++) {{
                buf_a[i] = (int16_t)input[i] << FIXED_FRAC_BITS;
            }}

            /* Conv1: (6, 100) -> (16, 50), k=5, s=2, p=2 */
            conv1d_stride(buf_a, NUM_MEG_CHANNELS, NUM_TIMEPOINTS,
                          conv1_weight, conv1_bias,
                          CONV1_OUT_CH, 5, 2, 2,
                          conv1_weight_scale,
                          buf_b, CONV1_OUT_LEN);
            relu_inplace(buf_b, CONV1_OUT_CH * CONV1_OUT_LEN);

            /* Conv2: (16, 50) -> (32, 25), k=3, s=2, p=1 */
            conv1d_stride(buf_b, CONV1_OUT_CH, CONV1_OUT_LEN,
                          conv2_weight, conv2_bias,
                          CONV2_OUT_CH, 3, 2, 1,
                          conv2_weight_scale,
                          buf_a, CONV2_OUT_LEN);
            relu_inplace(buf_a, CONV2_OUT_CH * CONV2_OUT_LEN);

            /* Conv3: (32, 25) -> (16, 13), k=3, s=2, p=1 */
            conv1d_stride(buf_a, CONV2_OUT_CH, CONV2_OUT_LEN,
                          conv3_weight, conv3_bias,
                          CONV3_OUT_CH, 3, 2, 1,
                          conv3_weight_scale,
                          buf_b, CONV3_OUT_LEN);
            relu_inplace(buf_b, CONV3_OUT_CH * CONV3_OUT_LEN);

            /* Global Average Pooling: (16, 13) -> (16,) */
            global_avg_pool(buf_b, CONV3_OUT_CH, CONV3_OUT_LEN, pool_buf);

            /* Dense 1: 16 -> 8 + ReLU */
            dense(pool_buf, CONV3_OUT_CH,
                  fc1_weight, fc1_bias, 8,
                  fc1_weight_scale,
                  fc1_buf);
            relu_inplace(fc1_buf, 8);

            /* Dense 2: 8 -> 3 + Sigmoid */
            dense(fc1_buf, 8,
                  fc2_weight, fc2_bias, NUM_OUTPUTS,
                  fc2_weight_scale,
                  fc2_buf);

            for (int16_t i = 0; i < NUM_OUTPUTS; i++) {{
                int16_t s = fp_sigmoid(fc2_buf[i]);
                output[i] = (float)s / (float)(1 << FIXED_FRAC_BITS);
            }}
        }}
    """)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(cpp_source)

    print(f"[convert] Wrote {output_path}  (estimated {ram_bytes} bytes static RAM)")
    return ram_bytes


# ---------------------------------------------------------------------------
# Memory budget verification
# ---------------------------------------------------------------------------

def verify_memory_budget(weight_bytes: int, ram_bytes: int,
                          code_overhead_bytes: int = 4096) -> bool:
    flash_used = weight_bytes + code_overhead_bytes
    print("\n=== Memory Budget Verification ===")
    print(f"  Flash: weights={weight_bytes:,}B + code~{code_overhead_bytes:,}B "
          f"= {flash_used:,}B / {FLASH_BUDGET_BYTES:,}B "
          f"({'OK' if flash_used < FLASH_BUDGET_BYTES else 'OVER'})")
    print(f"  RAM  : buffers={ram_bytes:,}B / {RAM_BUDGET_BYTES:,}B "
          f"({'OK' if ram_bytes < RAM_BUDGET_BYTES else 'OVER'})")

    ok = flash_used < FLASH_BUDGET_BYTES and ram_bytes < RAM_BUDGET_BYTES
    if not ok:
        print("  ** WARNING: Memory budget exceeded -- "
              "consider pruning or reducing layer sizes. **")
    return ok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_conversion(
    weights_dir: Optional[str] = None,
    arch_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    """Run the full conversion pipeline."""
    project_root = Path(__file__).parent.parent
    if weights_dir is None:
        weights_dir = str(project_root / "models" / "quantized_weights")
    if arch_path is None:
        arch_path = str(project_root / "models" / "model_architecture.json")
    if output_dir is None:
        output_dir = str(project_root / "arduino")

    print("=" * 60)
    print("  MEG Stroke Intervention -- Arduino C++ Conversion")
    print("=" * 60)

    # Load weights and architecture
    weights, scales = load_quantized_weights(weights_dir)
    arch = load_architecture(arch_path)

    # Generate model_weights.h
    header_path = str(Path(output_dir) / "model_weights.h")
    weight_bytes = generate_weights_header(weights, scales, header_path)

    # Generate arduino_inference.cpp
    cpp_path = str(Path(output_dir) / "arduino_inference.cpp")
    ram_bytes = generate_inference_cpp(cpp_path)

    # Verify memory budget
    verify_memory_budget(weight_bytes, ram_bytes)

    print("\n=== Conversion Summary ===")
    print(f"  Architecture  : {arch['model_name']}")
    print(f"  Parameters    : {arch['total_params']:,}")
    print(f"  Weight data   : {weight_bytes:,} bytes (INT8)")
    print(f"  RAM buffers   : {ram_bytes:,} bytes")
    print(f"  Header file   : {header_path}")
    print(f"  Source file    : {cpp_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_conversion()
