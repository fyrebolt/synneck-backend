"""
Multi-Modal MEG Stroke Intervention Network.

Fuses three input streams to predict solenoid valve control outputs:
    1. MEG signal     -- (6 channels, 100 timepoints) 1D CNN backbone
    2. Valve feedback -- current solenoid state [extension, force, delay]
    3. Patient data   -- tabular characteristics (age, severity, etc.)

Output: (3,) -> [valve_extension, force_magnitude, trigger_delay] in [0, 1]

Default patient feature vector (p=8):
    [age_norm, sex, bmi_norm, stroke_type, nihss_norm,
     affected_hemisphere, time_since_onset_norm, muscle_tone_norm]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Default patient feature configuration
# ---------------------------------------------------------------------------

PATIENT_FEATURE_NAMES: list[str] = [
    "age_norm",               # age / 100
    "sex",                    # 0 = female, 1 = male
    "bmi_norm",               # BMI / 50
    "stroke_type",            # 0 = ischemic, 1 = hemorrhagic
    "nihss_norm",             # NIHSS score / 42  (0 = no deficit, 1 = max)
    "affected_hemisphere",    # 0 = left, 1 = right
    "time_since_onset_norm",  # hours since onset / 720  (capped at 30 days)
    "muscle_tone_norm",       # modified Ashworth scale / 4
]

NUM_PATIENT_FEATURES: int = len(PATIENT_FEATURE_NAMES)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class MEGEncoder(nn.Module):
    """1D CNN backbone for MEG signal stream.

    Input:  (batch, 6, 100)
    Output: (batch, 16)  -- MEG embedding
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(6,  16, kernel_size=5, stride=2, padding=2)
        self.bn1   = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2   = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1)
        self.proj  = nn.Linear(32, 16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))   # (B, 16, 50)
        x = F.relu(self.bn2(self.conv2(x)))   # (B, 32, 25)
        x = F.relu(self.conv3(x))             # (B, 32, 13)
        x = x.mean(dim=2)                     # global avg pool -> (B, 32)
        return F.relu(self.proj(x))            # (B, 16)


class ValveFeedbackEncoder(nn.Module):
    """MLP encoder for solenoid valve feedback stream.

    Input:  (batch, 3 * feedback_window)  -- flattened [ext, force, delay] history
    Output: (batch, 8)                    -- feedback embedding

    Args:
        feedback_window: Number of recent feedback readings to include.
            1 means only the current reading; N means a rolling history of N frames.
    """

    def __init__(self, feedback_window: int = 1) -> None:
        super().__init__()
        self.feedback_window = feedback_window
        in_dim = 3 * feedback_window
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3 * feedback_window)
        return self.net(x)


class PatientEncoder(nn.Module):
    """MLP encoder for tabular patient characteristics.

    Input:  (batch, num_patient_features)
    Output: (batch, 8)  -- patient embedding

    Args:
        num_features: Number of patient feature dimensions.
    """

    def __init__(self, num_features: int = NUM_PATIENT_FEATURES) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Top-level model
# ---------------------------------------------------------------------------

class MEGMultiModalNet(nn.Module):
    """Multi-modal valve controller fusing MEG, valve feedback, and patient data.

    Architecture:
        MEG (6,100)    ──── MEGEncoder          ──── (16,) ──┐
        Valve (3*W,)   ──── ValveFeedbackEncoder ──── (8,)  ──┼── concat(32) ── FC(32→16) ── FC(16→3) ── Sigmoid
        Patient (p,)   ──── PatientEncoder       ──── (8,)  ──┘

    Output: (batch, 3) -> [valve_extension, force_magnitude, trigger_delay]

    Args:
        num_patient_features: Dimensionality of the patient feature vector.
        feedback_window: How many recent valve feedback frames to use (default 1).
    """

    OUTPUT_NAMES: list[str] = ["valve_extension", "force_magnitude", "trigger_delay"]

    def __init__(
        self,
        num_patient_features: int = NUM_PATIENT_FEATURES,
        feedback_window: int = 1,
    ) -> None:
        super().__init__()

        self.meg_encoder      = MEGEncoder()
        self.valve_encoder    = ValveFeedbackEncoder(feedback_window)
        self.patient_encoder  = PatientEncoder(num_patient_features)

        # Fusion head: 16 + 8 + 8 = 32
        self.fusion = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid(),
        )

    def forward(
        self,
        meg: torch.Tensor,
        valve_feedback: torch.Tensor,
        patient: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            meg:           (batch, 6, 100)              MEG signal window
            valve_feedback:(batch, 3 * feedback_window) Solenoid feedback history
            patient:       (batch, num_patient_features) Patient characteristics

        Returns:
            (batch, 3) tensor with values in [0, 1]:
                [:, 0] valve_extension
                [:, 1] force_magnitude
                [:, 2] trigger_delay
        """
        meg_emb     = self.meg_encoder(meg)            # (B, 16)
        valve_emb   = self.valve_encoder(valve_feedback)  # (B, 8)
        patient_emb = self.patient_encoder(patient)    # (B, 8)

        fused = torch.cat([meg_emb, valve_emb, patient_emb], dim=1)  # (B, 32)
        return self.fusion(fused)                      # (B, 3)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MEGMultiModalNet  --  Architecture Summary")
    print("=" * 60)

    model = MEGMultiModalNet(
        num_patient_features=NUM_PATIENT_FEATURES,
        feedback_window=1,
    )

    B = 4
    meg     = torch.randn(B, 6, 100)
    valve   = torch.randn(B, 3)
    patient = torch.randn(B, NUM_PATIENT_FEATURES)

    out = model(meg, valve, patient)

    print(f"\nMEG input shape:     {meg.shape}")
    print(f"Valve input shape:   {valve.shape}")
    print(f"Patient input shape: {patient.shape}")
    print(f"Output shape:        {out.shape}")
    print(f"Output range:        [{out.min().item():.4f}, {out.max().item():.4f}]")
    print(f"Output names:        {MEGMultiModalNet.OUTPUT_NAMES}")
    print(f"\nTotal parameters:    {model.count_parameters():,}")

    print("\n--- Patient features ---")
    for i, name in enumerate(PATIENT_FEATURE_NAMES):
        print(f"  [{i}] {name}")

    print("\n--- Sub-module parameter counts ---")
    for name, module in [
        ("MEGEncoder",          model.meg_encoder),
        ("ValveFeedbackEncoder",model.valve_encoder),
        ("PatientEncoder",      model.patient_encoder),
        ("Fusion head",         model.fusion),
    ]:
        n = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"  {name:30s}  {n:>5,}")

    print("\n" + "=" * 60)
    print("All checks passed.")
    print("=" * 60)
