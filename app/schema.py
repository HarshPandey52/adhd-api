"""
app/schema.py
=============
Input: raw EEG time-series for a single subject.
Output: ADHD prediction label + confidence.
"""

from pydantic import BaseModel, Field
from typing import List

# 19 EEG channels defined in your notebook
EEG_CHANNELS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz'
]


class EEGInput(BaseModel):
    """
    Raw EEG data for one subject.

    eeg_data: list of samples, each sample is a list of 19 channel values.
              Shape → [n_timepoints, 19]
              Values must be in microvolts (µV).
              Minimum recommended length: 750 samples (3 seconds at 250 Hz).

    Example (tiny 3-sample snippet for documentation purposes):
        {
          "eeg_data": [
            [0.1, -0.2, 0.3, ...],   ← 19 values per timepoint
            [0.2, -0.1, 0.4, ...],
            ...
          ]
        }
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
