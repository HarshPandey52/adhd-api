"""
app/schema.py
=============
Input: raw EEG time-series for a single subject.
Output: ADHD prediction label + confidence + real computed EEG metrics.

EEG_CHANNELS is the single source of truth for channel count/order —
predictor.py imports it from here rather than redefining its own copy,
so the two can't drift out of sync again.
"""

from pydantic import BaseModel, Field
from typing import List

# All 32 recorded channels: 26 scalp EEG + 6 EOG/reference auxiliary channels.
# Order matters — must exactly match the column order final_model.py trained on.
EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4',
    'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz', 'CP4',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2',
    'VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'Mass',
]


class EEGInput(BaseModel):
    """
    Raw EEG data for one subject.
    eeg_data: 2-D array of shape [n_timepoints, 32], values in the same
    units/scale as the training CSVs (no unit conversion is applied).
    Minimum recommended length: 1500 samples (3 seconds at 500 Hz native rate).
    Channel order: Fp1 Fp2 F7 F3 Fz F4 F8 FC3 FCz FC4 T7 C3 Cz C4 T8
    CP3 CPz CP4 P7 P3 Pz P4 P8 O1 Oz O2 VPVA VNVB HPHL HNHR Erbs Mass
    """
    eeg_data: List[List[float]] = Field(
        ...,
        description=(
            "2-D array of shape [n_timepoints, 32]. "
            "Each inner list = one timepoint with 32 channel values. "
            "Channel order: Fp1 Fp2 F7 F3 Fz F4 F8 FC3 FCz FC4 T7 C3 Cz C4 T8 "
            "CP3 CPz CP4 P7 P3 Pz P4 P8 O1 Oz O2 VPVA VNVB HPHL HNHR Erbs Mass"
        ),
        example=[
            [0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 0.1, -0.3, 0.4, -0.1,
             0.2, -0.2, 0.3, 0.1, -0.4, 0.2, 0.0, 0.1, -0.1, 0.2,
             -0.3, 0.1, 0.0, -0.2, 0.3, -0.1, 0.4, -0.2, 0.1, -0.3, 0.2, 0.0]
        ] * 10
    )


class ADHDPrediction(BaseModel):
    prediction: int = Field(..., description="1 = ADHD likely, 0 = ADHD unlikely")
    label: str = Field(..., description="Human-readable result: 'ADHD' or 'Non-ADHD'")
    confidence: float = Field(..., description="Model confidence score (0.0 – 1.0)")
    confidence_pct: str = Field(..., description="Confidence as a percentage string")
    threshold_used: float = Field(..., description="Decision threshold applied")

    theta_power: float = Field(..., description="Mean theta band power")
    alpha_power: float = Field(..., description="Mean alpha band power")
    beta_power: float  = Field(..., description="Mean beta band power")
    delta_power: float = Field(..., description="Mean delta band power")
    gamma_power: float = Field(..., description="Mean gamma band power")

    theta_beta_ratio: float = Field(..., description="Frontal theta/beta power ratio")
    alpha_coherence: float  = Field(..., description="Mean pairwise alpha-band coherence (0-1)")
    sample_entropy: float   = Field(..., description="Mean sample entropy across sampled channels")

    band_power_distribution: List[float] = Field(
        ..., description="[delta, theta, alpha, beta, gamma] for charting"
    )
    entropy_trend: List[float] = Field(
        ..., description="Sample entropy per epoch, for trend chart"
    )
