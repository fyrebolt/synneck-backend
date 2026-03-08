"""
MEG Stroke Pattern Detection Network.

Ultra-lightweight CNN designed for real-time stroke intervention
via solenoid valve control. Architecture constrained to < 50K parameters
for deployment on Arduino-class microcontrollers.

Input:  (batch, 6 channels, 100 timepoints)  -- 500ms window at 200Hz
Output: (batch, 3) -> [valve_extension, force_magnitude, trigger_delay]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


class MEGStrokeNet(nn.Module):
    """
    Ultra-lightweight CNN for MEG stroke pattern detection.
    Designed for Arduino deployment (< 50K parameters).

    Input: (batch, 6 channels, 100 timepoints)
    Output: (batch, 3) -> [valve_extension, force_magnitude, trigger_delay]

    Architecture:
        Conv1D(6->16, k=5, s=2) -> ReLU -> BN
        Conv1D(16->32, k=3, s=2) -> ReLU -> BN
        Conv1D(32->16, k=3, s=2) -> ReLU
        Global Average Pooling
        Dense(16->8) -> ReLU
        Dense(8->3) -> Sigmoid
    """

    # Class constants
    NUM_CHANNELS: int = 6
    NUM_TIMEPOINTS: int = 100
    NUM_OUTPUTS: int = 3
    MAX_PARAMETERS: int = 50_000
    OUTPUT_NAMES: list[str] = ["valve_extension", "force_magnitude", "trigger_delay"]

    def __init__(self) -> None:
        super().__init__()

        # --- Convolutional backbone ---
        # Layer 1: 6 -> 16 filters, kernel=5, stride=2
        self.conv1 = nn.Conv1d(
            in_channels=6, out_channels=16, kernel_size=5, stride=2, padding=2
        )
        self.bn1 = nn.BatchNorm1d(16)

        # Layer 2: 16 -> 32 filters, kernel=3, stride=2
        self.conv2 = nn.Conv1d(
            in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1
        )
        self.bn2 = nn.BatchNorm1d(32)

        # Layer 3: 32 -> 16 filters, kernel=3, stride=2
        self.conv3 = nn.Conv1d(
            in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1
        )

        # --- Classifier head ---
        self.fc1 = nn.Linear(16, 8)
        self.fc2 = nn.Linear(8, 3)

        # Validate parameter budget
        param_count = self.count_parameters()
        if param_count >= self.MAX_PARAMETERS:
            raise RuntimeError(
                f"Model has {param_count} parameters, exceeding the "
                f"{self.MAX_PARAMETERS} budget for Arduino deployment."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, 6, 100).

        Returns:
            Tensor of shape (batch, 3) with values in [0, 1].
        """
        # Layer 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        # Layer 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)

        # Layer 3
        x = self.conv3(x)
        x = F.relu(x)

        # Global Average Pooling: (batch, 16, T') -> (batch, 16)
        x = x.mean(dim=2)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def inference_time_estimate(
        self,
        clock_speed_mhz: float = 16.0,
        ops_per_cycle: float = 1.0,
    ) -> dict[str, float]:
        """
        Estimate inference time on an Arduino-class microcontroller.

        Uses a simple roofline model: each MAC (multiply-accumulate) counts
        as 2 floating-point operations.  We assume fixed-point quantisation
        brings the effective cost to ~1 op/cycle on an 8-bit ALU.

        Args:
            clock_speed_mhz: Processor clock in MHz (default 16 for ATmega328P).
            ops_per_cycle: Effective operations per clock cycle after
                           quantisation (default 1.0).

        Returns:
            Dictionary with estimated FLOPs, total ops, and latency in ms.
        """
        flops = 0

        # Conv layers: FLOPs = 2 * out_channels * out_length * in_channels * kernel_size
        # (factor of 2 for multiply + add)
        input_length = self.NUM_TIMEPOINTS

        # Layer 1: Conv1d(6->16, k=5, s=2, p=2) on length 100
        out_len_1 = (input_length + 2 * 2 - 5) // 2 + 1  # = 50
        flops += 2 * 16 * out_len_1 * 6 * 5

        # Layer 2: Conv1d(16->32, k=3, s=2, p=1) on length 50
        out_len_2 = (out_len_1 + 2 * 1 - 3) // 2 + 1  # = 25
        flops += 2 * 32 * out_len_2 * 16 * 3

        # Layer 3: Conv1d(32->16, k=3, s=2, p=1) on length 25
        out_len_3 = (out_len_2 + 2 * 1 - 3) // 2 + 1  # = 13
        flops += 2 * 16 * out_len_3 * 32 * 3

        # FC1: 16 -> 8
        flops += 2 * 16 * 8

        # FC2: 8 -> 3
        flops += 2 * 8 * 3

        clock_speed_hz = clock_speed_mhz * 1e6
        total_cycles = flops / ops_per_cycle
        latency_s = total_cycles / clock_speed_hz
        latency_ms = latency_s * 1000.0

        return {
            "estimated_flops": flops,
            "clock_speed_mhz": clock_speed_mhz,
            "ops_per_cycle": ops_per_cycle,
            "estimated_cycles": total_cycles,
            "estimated_latency_ms": round(latency_ms, 2),
            "conv_output_lengths": [out_len_1, out_len_2, out_len_3],
        }

    @classmethod
    def get_example_input(cls, batch_size: int = 1) -> torch.Tensor:
        """
        Return a sample input tensor for testing / tracing.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            Random tensor of shape (batch_size, 6, 100).
        """
        return torch.randn(batch_size, cls.NUM_CHANNELS, cls.NUM_TIMEPOINTS)


# ---------------------------------------------------------------------------
# Standalone sanity checks
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("MEGStrokeNet  --  Architecture Summary")
    print("=" * 60)

    model = MEGStrokeNet()

    # Parameter count
    total_params = model.count_parameters()
    print(f"\nTotal trainable parameters: {total_params:,}")
    print(f"Parameter budget:           {MEGStrokeNet.MAX_PARAMETERS:,}")
    print(f"Under budget:               {'YES' if total_params < MEGStrokeNet.MAX_PARAMETERS else 'NO'}")

    # Layer-by-layer breakdown
    print("\n--- Layer-by-layer parameter count ---")
    for name, param in model.named_parameters():
        print(f"  {name:30s}  {str(list(param.shape)):20s}  {param.numel():>6,}")

    # Forward pass test
    x = MEGStrokeNet.get_example_input(batch_size=4)
    y = model(x)
    print(f"\nInput  shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Output range: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"Output names: {MEGStrokeNet.OUTPUT_NAMES}")

    # Inference time estimate
    timing = model.inference_time_estimate()
    print(f"\n--- Arduino Inference Estimate ({timing['clock_speed_mhz']} MHz) ---")
    print(f"  Estimated FLOPs:        {timing['estimated_flops']:,}")
    print(f"  Estimated cycles:       {timing['estimated_cycles']:,.0f}")
    print(f"  Estimated latency:      {timing['estimated_latency_ms']} ms")
    print(f"  Conv output lengths:    {timing['conv_output_lengths']}")

    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)
