"""
app/schema.py
=============
Input: raw EEG time-series for a single subject.
Output: ADHD prediction label + confidence + real computed EEG metrics.
"""

from pydantic import BaseModel, Field
from typing import List

# 19 EEG channels
EEG_CHANNELS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz'
]


class EEGInput(BaseModel):
    """
    Raw EEG data for one subject.
    eeg_data: 2-D array of shape [n_timepoints, 19], values in microvolts (µV).
    Minimum recommended length: 750 samples (3 seconds at 250 Hz).
    Channel order: Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T7 T8 P7 P8 Fz Cz Pz
    """
    eeg_data: List[List[float]] = Field(
        ...,
        description=(
            "2-D array of shape [n_timepoints, 19]. "
            "Each inner list = one timepoint with 19 EEG channel values in µV. "
            "Channel order: Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 "
            "F7 F8 T7 T8 P7 P8 Fz Cz Pz"
        ),
        example=[
            [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.1, -0.3, 0.4,
             -0.1, 0.2, -0.2, 0.3, 0.1, -0.4, 0.2, 0.0, 0.1, -0.1]
        ] * 10
    )


class ADHDPrediction(BaseModel):
    prediction: int = Field(
        ..., description="1 = ADHD likely, 0 = ADHD unlikely"
    )
    label: str = Field(
        ..., description="Human-readable result: 'ADHD' or 'Non-ADHD'"
    )
    confidence: float = Field(
        ..., description="Model confidence score (0.0 – 1.0)"
    )
    confidence_pct: str = Field(
        ..., description="Confidence as a percentage string, e.g. '78.34%'"
    )
    threshold_used: float = Field(
        ..., description="Decision threshold applied (default 0.45)"
    )

    # ── Real computed EEG metrics ────────────────────────────────────────────
    theta_power: float = Field(..., description="Mean theta band power (µV²)")
    alpha_power: float = Field(..., description="Mean alpha band power (µV²)")
    beta_power: float  = Field(..., description="Mean beta band power (µV²)")
    delta_power: float = Field(..., description="Mean delta band power (µV²)")
    gamma_power: float = Field(..., description="Mean gamma band power (µV²)")

    theta_beta_ratio: float = Field(..., description="Frontal theta/beta power ratio")
    alpha_coherence: float  = Field(..., description="Mean pairwise alpha-band coherence across channels (0-1)")
    sample_entropy: float   = Field(..., description="Mean sample entropy across channels")

    band_power_distribution: List[float] = Field(
        ..., description="[delta, theta, alpha, beta, gamma] mean power values for charting"
    )
    entropy_trend: List[float] = Field(
        ..., description="Sample entropy per epoch, for trend chart"
    )
