"""
app/predictor.py
================
Loads saved pipeline components and runs inference + real EEG metric
calculations (theta/alpha/beta/delta/gamma power, theta/beta ratio,
alpha coherence, sample entropy) for display on the frontend.
"""

import os
import numpy as np
import mne
import joblib
from xgboost import XGBClassifier
from functools import lru_cache
from itertools import combinations
from scipy.signal import hilbert

from app.schema import EEGInput, ADHDPrediction

EEG_CHANNELS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz'
]
SFREQ             = 250
FREQ_BANDS        = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
DELTA_BAND        = (1, 4)
GAMMA_BAND        = (30, 45)

FRONTAL_CHANNELS  = ['Fp1', 'Fp2', 'Fz']
EPOCH_DURATION    = 3.0
EPOCH_OVERLAP     = 1.5
OPTIMAL_THRESHOLD = 0.45
MODEL_DIR         = os.path.join(os.path.dirname(__file__), "..", "model")


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


def _extract_prediction_features(raw):
    frontal_idx = [EEG_CHANNELS.index(ch) for ch in FRONTAL_CHANNELS]
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

    return (np.array(X_epochs_list), np.array(X_power_list),
            band_power_dict, theta_beta_ratio, band_epochs_dict['alpha'])


def _band_power(raw, fmin, fmax):
    filtered = raw.copy().filter(fmin, fmax, verbose=False)
    epochs   = mne.make_fixed_length_epochs(
        filtered, duration=EPOCH_DURATION,
        overlap=EPOCH_OVERLAP, preload=True, verbose=False
    )
    ep_data = epochs.get_data()
    return np.mean(ep_data**2, axis=2)


def _alpha_coherence(alpha_epochs):
    n_epochs, n_channels, _ = alpha_epochs.shape
    pairs = list(combinations(range(n_channels), 2))
    coh_values = []

    for ep in range(n_epochs):
        phases = np.angle(hilbert(alpha_epochs[ep], axis=-1))
        for i, j in pairs:
            phase_diff = phases[i] - phases[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            coh_values.append(plv)

    return float(np.mean(coh_values)) if coh_values else 0.0


def _sample_entropy(signal, m=2, r=None):
    n = len(signal)
    if r is None:
        r = 0.2 * np.std(signal)
    if r == 0:
        return 0.0

    def _phi(m_):
        templates = np.array([signal[i:i + m_] for i in range(n - m_)])
        count = 0
        total = 0
        for i in range(len(templates)):
            dists = np.max(np.abs(templates - templates[i]), axis=1)
            count += np.sum(dists <= r) - 1
            total += len(templates) - 1
        return count / total if total > 0 else 0.0

    phi_m  = _phi(m)
    phi_m1 = _phi(m + 1)

    if phi_m == 0 or phi_m1 == 0:
        return 0.0
    return float(-np.log(phi_m1 / phi_m))


def _sample_entropy_trend(epochs_data, max_epochs=20, max_samples=300):
    n_epochs = min(epochs_data.shape[0], max_epochs)
    trend = []

    for ep in range(n_epochs):
        ch_entropies = []
        sample_channels = [0, 1, 4, 5, 8, 9, 16]  # Fp1,Fp2,C3,C4,O1,O2,Fz
        for ch in sample_channels:
            sig = epochs_data[ep, ch, :max_samples]
            ch_entropies.append(_sample_entropy(sig))
        trend.append(round(float(np.mean(ch_entropies)), 4))

    return trend


def predict_adhd(data: EEGInput) -> ADHDPrediction:
    model, scaler, riemann = _load_components()

    eeg_array = np.array(data.eeg_data)

    if eeg_array.ndim != 2 or eeg_array.shape[1] != len(EEG_CHANNELS):
        raise ValueError(
            f"eeg_data must have shape [n_timepoints, 19]. Got {eeg_array.shape}"
        )
    min_samples = int(SFREQ * EPOCH_DURATION)
    if eeg_array.shape[0] < min_samples:
        raise ValueError(
            f"Need at least {min_samples} timepoints "
            f"({EPOCH_DURATION}s × {SFREQ}Hz). Got {eeg_array.shape[0]}."
        )

    data_v = eeg_array.T * 1e-6
    info = mne.create_info(EEG_CHANNELS, SFREQ, ch_types="eeg")
    raw  = mne.io.RawArray(data_v, info, verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    X_epochs, X_power, band_power_dict, theta_beta_ratio, alpha_epochs = \
        _extract_prediction_features(raw)

    X_riemann  = riemann.transform(X_epochs)
    X_combined = np.hstack([X_riemann, X_power])
    X_scaled   = scaler.transform(X_combined)

    epoch_probs = model.predict_proba(X_scaled)[:, 1]
    mean_prob   = float(np.median(epoch_probs))

    prediction = 1 if mean_prob >= OPTIMAL_THRESHOLD else 0
    label      = "ADHD" if prediction == 1 else "Non-ADHD"

    theta_power = float(np.mean(band_power_dict['theta']) * 1e12)
    alpha_power = float(np.mean(band_power_dict['alpha']) * 1e12)
    beta_power  = float(np.mean(band_power_dict['beta'])  * 1e12)

    delta_power_arr = _band_power(raw, *DELTA_BAND)
    gamma_power_arr = _band_power(raw, *GAMMA_BAND)
    delta_power = float(np.mean(delta_power_arr) * 1e12)
    gamma_power = float(np.mean(gamma_power_arr) * 1e12)

    tb_ratio = float(np.mean(theta_beta_ratio))
    alpha_coh = _alpha_coherence(alpha_epochs)

    entropy_trend = _sample_entropy_trend(alpha_epochs)
    sample_ent = float(np.mean(entropy_trend)) if entropy_trend else 0.0

    band_power_distribution = [
        round(delta_power, 4), round(theta_power, 4), round(alpha_power, 4),
        round(beta_power, 4), round(gamma_power, 4),
    ]

    return ADHDPrediction(
        prediction       = prediction,
        label            = label,
        confidence       = round(mean_prob, 4),
        confidence_pct   = f"{round(mean_prob * 100, 2)}%",
        threshold_used   = OPTIMAL_THRESHOLD,
        theta_power      = round(theta_power, 4),
        alpha_power      = round(alpha_power, 4),
        beta_power       = round(beta_power, 4),
        delta_power      = round(delta_power, 4),
        gamma_power      = round(gamma_power, 4),
        theta_beta_ratio = round(tb_ratio, 4),
        alpha_coherence  = round(alpha_coh, 4),
        sample_entropy   = round(sample_ent, 4),
        band_power_distribution = band_power_distribution,
        entropy_trend            = entropy_trend,
    )
