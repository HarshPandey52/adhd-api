"""
app/predictor.py
================
Loads all saved pipeline components once at startup and exposes
a single predict() function used by the API endpoint.

The preprocessing here is IDENTICAL to your notebook's feature
extraction — same frequency bands, same epoch settings, same
Riemannian + power feature fusion.
"""

import os
import numpy as np
import mne
import joblib
from xgboost import XGBClassifier
from functools import lru_cache

from app.schema import EEGInput, ADHDPrediction

# ── Constants (mirror notebook exactly) ──────────────────────────────────────
EEG_CHANNELS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz'
]
SFREQ             = 250
FREQ_BANDS        = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
FRONTAL_CHANNELS  = ['Fp1', 'Fp2', 'Fz']
EPOCH_DURATION    = 3.0
EPOCH_OVERLAP     = 1.5
OPTIMAL_THRESHOLD = 0.45
MODEL_DIR         = os.path.join(os.path.dirname(__file__), "..", "model")


# ── Load model components once ───────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_components():
    model_path   = os.path.join(MODEL_DIR, "xgb_model.json")
    scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
    riemann_path = os.path.join(MODEL_DIR, "riemann_pipeline.pkl")

    for p in [model_path, scaler_path, riemann_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Model file not found: {p}\n"
                "Run train_and_save_model.py first to generate model files."
            )

    model   = XGBClassifier()
    model.load_model(model_path)
    scaler  = joblib.load(scaler_path)
    riemann = joblib.load(riemann_path)
    return model, scaler, riemann


# ── Feature extraction for a single subject ──────────────────────────────────
def _extract_features_single(eeg_array: np.ndarray):
    """
    eeg_array: shape [n_timepoints, 19] in µV
    Returns:
        X_epochs: shape [n_epochs, 57, epoch_len]  ← 3 bands × 19 ch
        X_power:  shape [n_epochs, power_feat_dim]
    """
    frontal_idx = [EEG_CHANNELS.index(ch) for ch in FRONTAL_CHANNELS]

    # Convert to (channels, timepoints) and µV → V
    data = eeg_array.T * 1e-6

    info = mne.create_info(EEG_CHANNELS, SFREQ, ch_types="eeg")
    raw  = mne.io.RawArray(data, info, verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    band_epochs_dict, band_power_dict = {}, {}

    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        filtered = raw.copy().filter(fmin, fmax, verbose=False)
        epochs   = mne.make_fixed_length_epochs(
            filtered, duration=EPOCH_DURATION,
            overlap=EPOCH_OVERLAP, preload=True, verbose=False
        )
        ep_data = epochs.get_data()
        band_epochs_dict[band_name] = ep_data
        band_power_dict[band_name]  = np.mean(ep_data**2, axis=2)

    stacked = np.concatenate(
        [band_epochs_dict['theta'],
         band_epochs_dict['alpha'],
         band_epochs_dict['beta']],
        axis=1
    )

    theta_frontal    = np.mean(band_power_dict['theta'][:, frontal_idx], axis=1)
    beta_frontal     = np.mean(band_power_dict['beta'][:, frontal_idx],  axis=1)
    theta_beta_ratio = theta_frontal / (beta_frontal + 1e-10)

    X_epochs_list, X_power_list = [], []

    for i in range(stacked.shape[0]):
        X_epochs_list.append(stacked[i])

        total_power = (band_power_dict['theta'][i] +
                       band_power_dict['alpha'][i] +
                       band_power_dict['beta'][i])

        rel_theta = band_power_dict['theta'][i] / (total_power + 1e-10)
        rel_alpha = band_power_dict['alpha'][i] / (total_power + 1e-10)
        rel_beta  = band_power_dict['beta'][i]  / (total_power + 1e-10)

        power_feat = np.concatenate([
            band_power_dict['theta'][i], band_power_dict['alpha'][i],
            band_power_dict['beta'][i],  rel_theta, rel_alpha,
            rel_beta, [theta_beta_ratio[i]]
        ])
        X_power_list.append(power_feat)

    return np.array(X_epochs_list), np.array(X_power_list)


# ── Main prediction function ──────────────────────────────────────────────────
def predict_adhd(data: EEGInput) -> ADHDPrediction:
    model, scaler, riemann = _load_components()

    eeg_array = np.array(data.eeg_data)  # [n_timepoints, 19]

    # Validate shape
    if eeg_array.ndim != 2 or eeg_array.shape[1] != len(EEG_CHANNELS):
        raise ValueError(
            f"eeg_data must have shape [n_timepoints, 19]. "
            f"Got {eeg_array.shape}"
        )
    min_samples = int(SFREQ * EPOCH_DURATION)
    if eeg_array.shape[0] < min_samples:
        raise ValueError(
            f"Need at least {min_samples} timepoints "
            f"({EPOCH_DURATION}s × {SFREQ}Hz). Got {eeg_array.shape[0]}."
        )

    # Extract features
    X_epochs, X_power = _extract_features_single(eeg_array)

    # Riemannian transform (inference only — no fitting)
    X_riemann = riemann.transform(X_epochs)

    # Fuse features
    X_combined = np.hstack([X_riemann, X_power])

    # Scale
    X_scaled = scaler.transform(X_combined)

    # Predict probabilities per epoch, then aggregate (median — same as notebook)
    epoch_probs = model.predict_proba(X_scaled)[:, 1]
    mean_prob   = float(np.median(epoch_probs))

    prediction = 1 if mean_prob >= OPTIMAL_THRESHOLD else 0
    label      = "ADHD" if prediction == 1 else "Non-ADHD"

    return ADHDPrediction(
        prediction       = prediction,
        label            = label,
        confidence       = round(mean_prob, 4),
        confidence_pct   = f"{round(mean_prob * 100, 2)}%",
        threshold_used   = OPTIMAL_THRESHOLD,
    )
