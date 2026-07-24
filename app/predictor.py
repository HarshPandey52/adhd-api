"""
app/predictor.py
================
Loads saved pipeline components and runs inference + EEG metric
calculations for display on the frontend.

REWRITTEN to match final_model.py's actual training pipeline exactly:
  - No mne. Signal processing is scipy-based (resample_poly + Butterworth
    sosfiltfilt), same as training.
  - Native sample rate is 500 Hz; data is downsampled to 125 Hz before
    filtering/epoching, same as training.
  - Epochs are 3.0s, NON-overlapping (overlap=0.0), same as training.
  - Feature vector = Riemannian tangent-space vector (from an OAS
    covariance per epoch, computed on the 96-channel theta+alpha+beta
    stack) + power features (raw + relative band power + frontal
    theta/beta ratio), same as training.
  - Epoch-level probabilities are aggregated with the 75th percentile
    (AGG_PERCENTILE), matching final_model.py's aggregate_probs(), NOT
    the median used by the old 19-channel predictor.
  - All 32 recorded columns (26 scalp EEG + VPVA/VNVB/HPHL/HNHR/Erbs/Mass)
    are used identically, matching training — nothing is dropped or
    given a different channel type.

⚠️ THINGS YOU MUST CONFIRM/UPDATE BEFORE DEPLOYING:
  1. riemann_pipeline.pkl / scaler.pkl must be re-saved AFTER final_model.py
     is updated to import RiemannTangentSpace from app.riemann_utils
     instead of defining it inline. Otherwise joblib.load() will raise
     "Can't get attribute 'RiemannTangentSpace' on <module '__main__'>".
  2. OPTIMAL_THRESHOLD below is a placeholder (0.45, carried over from the
     old model). final_model.py's tune_threshold() prints the actual
     tuned threshold for your New32 model — read it from that training
     run's console output and update the constant below to match.
  3. AGG_PERCENTILE is set to 75 to match final_model.py's AGG_PERCENTILE.
     If you change that constant in training, change it here too.
  4. Input units: final_model.py does NOT convert the CSV values to volts
     (no `* 1e-6` anywhere) — it uses the raw CSV scale as-is. This code
     does the same. If your /predict callers were sending µV expecting
     the old µV->V conversion, that assumption no longer applies — send
     raw values on the same scale as the training CSVs.
  5. schema.py's EEG_CHANNELS must be the 32-channel list in the exact
     order below (imported from schema.py, not redefined here, to avoid
     the two-copies-that-can-drift problem from the old code).
"""

import os
import time
import numpy as np
import joblib
from xgboost import XGBClassifier
from functools import lru_cache
from itertools import combinations
from scipy.signal import hilbert
from scipy.spatial.distance import cdist

from app.schema import EEGInput, ADHDPrediction, EEG_CHANNELS
from app.riemann_utils import (
    resample_signal, bandpass_filter, make_fixed_length_epochs,
    epoch_covariance,
)

# ── Constants — mirror final_model.py exactly ────────────────────────────────
SFREQ              = 500   # native sample rate the model was trained on
PROCESS_SFREQ       = 125   # data is downsampled to this before filtering/epoching
FREQ_BANDS          = {'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30)}
DELTA_BAND          = (1, 4)     # display-only metric, not a model input feature
GAMMA_BAND          = (30, 45)   # display-only metric, not a model input feature

FRONTAL_CHANNELS    = ['Fp1', 'Fp2', 'Fz']
EPOCH_DURATION       = 3.0
EPOCH_OVERLAP        = 0.0   # non-overlapping, matches final_model.py
OPTIMAL_THRESHOLD    = 0.45  # PLACEHOLDER — replace with the tuned value from
                              # final_model.py's tune_threshold() output for New32
AGG_PERCENTILE       = 75    # matches final_model.py's AGG_PERCENTILE
MODEL_DIR            = os.path.join(os.path.dirname(__file__), "..", "model")

# Hard cap on epochs run through the expensive per-epoch stages (covariance,
# riemann transform, coherence, entropy). Matches final_model.py's
# MAX_EPOCHS_PER_SUBJECT so inference never sees more epochs than training
# did per subject.
MAX_EPOCHS_FOR_INFERENCE = 15

# Channels used for the sample-entropy display trend (subset, for speed).
# Recomputed from EEG_CHANNELS by name so it can't silently point at the
# wrong columns if the channel order ever changes.
_ENTROPY_TARGET_NAMES = ['Fp1', 'Fp2', 'C3', 'C4', 'O1', 'O2', 'Fz']


def _log(stage: str, t0: float):
    print(f"[predict timing] {stage}: {time.time() - t0:.2f}s", flush=True)


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


def _select_epoch_indices(n_epochs: int, max_epochs: int) -> np.ndarray:
    """Evenly-spaced subset of epoch indices, capped at max_epochs.
    Deterministic (unlike training's random subsample) so inference
    returns the same result for the same upload every time."""
    if n_epochs <= max_epochs:
        return np.arange(n_epochs)
    return np.linspace(0, n_epochs - 1, max_epochs).round().astype(int)


def _band_epochs_and_power(data: np.ndarray, fmin: float, fmax: float):
    """data: (n_channels, n_times) at PROCESS_SFREQ.
    Returns (epochs, power) for one band, unfiltered by the epoch cap yet."""
    filtered = bandpass_filter(data, PROCESS_SFREQ, fmin, fmax)
    ep_data  = make_fixed_length_epochs(
        filtered, PROCESS_SFREQ, EPOCH_DURATION, EPOCH_OVERLAP
    )
    power = np.mean(ep_data ** 2, axis=2)
    return ep_data, power


def _sample_entropy(signal_, m=2, r=None):
    """Vectorized sample entropy via scipy cdist (Chebyshev distance)."""
    n = len(signal_)
    if r is None:
        r = 0.2 * np.std(signal_)
    if r == 0:
        return 0.0

    def _phi(m_):
        n_templates = n - m_
        if n_templates < 2:
            return 0.0
        templates = np.array([signal_[i:i + m_] for i in range(n_templates)])
        dists = cdist(templates, templates, metric='chebyshev')
        count = np.sum(dists <= r) - n_templates  # exclude self-matches
        total = n_templates * (n_templates - 1)
        return count / total if total > 0 else 0.0

    phi_m  = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m == 0 or phi_m1 == 0:
        return 0.0
    return float(-np.log(phi_m1 / phi_m))


def _sample_entropy_trend(alpha_epochs: np.ndarray, target_idx: list[int], max_samples=300):
    t0 = time.time()
    trend = []
    for ep in range(alpha_epochs.shape[0]):
        ch_entropies = [
            _sample_entropy(alpha_epochs[ep, ch, :max_samples]) for ch in target_idx
        ]
        trend.append(round(float(np.mean(ch_entropies)), 4))
    _log(f"_sample_entropy_trend ({alpha_epochs.shape[0]} epochs)", t0)
    return trend


def _alpha_coherence(alpha_epochs: np.ndarray) -> float:
    t0 = time.time()
    n_epochs, n_channels, _ = alpha_epochs.shape
    pairs = list(combinations(range(n_channels), 2))
    coh_values = []
    for ep in range(n_epochs):
        phases = np.angle(hilbert(alpha_epochs[ep], axis=-1))
        for i, j in pairs:
            phase_diff = phases[i] - phases[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            coh_values.append(plv)
    _log(f"_alpha_coherence ({n_epochs} epochs)", t0)
    return float(np.mean(coh_values)) if coh_values else 0.0


def predict_adhd(data: EEGInput) -> ADHDPrediction:
    request_t0 = time.time()
    model, scaler, riemann = _load_components()

    eeg_array = np.array(data.eeg_data, dtype=np.float32)

    if eeg_array.ndim != 2 or eeg_array.shape[1] != len(EEG_CHANNELS):
        raise ValueError(
            f"eeg_data must have shape [n_timepoints, {len(EEG_CHANNELS)}]. "
            f"Got {eeg_array.shape}"
        )
    min_samples = int(SFREQ * EPOCH_DURATION)
    if eeg_array.shape[0] < min_samples:
        raise ValueError(
            f"Need at least {min_samples} timepoints "
            f"({EPOCH_DURATION}s x {SFREQ}Hz native rate). Got {eeg_array.shape[0]}."
        )

    # ── Preprocessing: matches final_model.py's extract_subject_features exactly ──
    t0 = time.time()
    raw_data = eeg_array.T  # (n_channels, n_times) — NOT scaled to volts; raw CSV units
    raw_data = raw_data - raw_data.mean(axis=0, keepdims=True)  # average reference
    data = resample_signal(raw_data, SFREQ, PROCESS_SFREQ)      # 500 -> 125 Hz
    _log("average-reference + resample", t0)

    t0 = time.time()
    band_epochs, band_power = {}, {}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        ep, pw = _band_epochs_and_power(data, fmin, fmax)
        band_epochs[band_name] = ep
        band_power[band_name]  = pw
    _log("filter+epoch theta/alpha/beta", t0)

    # Cap epoch count once, consistently across bands, before any of the
    # expensive per-epoch work (covariance, riemann, coherence, entropy) runs.
    n_epochs = min(band_epochs[b].shape[0] for b in FREQ_BANDS)
    if n_epochs < 1:
        raise ValueError(
            "Recording is too short to produce a single 3-second epoch "
            "after resampling. Upload a longer recording."
        )
    keep = _select_epoch_indices(n_epochs, MAX_EPOCHS_FOR_INFERENCE)
    for b in FREQ_BANDS:
        band_epochs[b] = band_epochs[b][keep]
        band_power[b]  = band_power[b][keep]

    # ── Feature extraction: mirrors extract_subject_features ──────────────────
    t0 = time.time()
    stacked = np.concatenate(
        [band_epochs['theta'], band_epochs['alpha'], band_epochs['beta']], axis=1
    )  # (n_kept, 96, n_samples)

    frontal_idx = [EEG_CHANNELS.index(ch) for ch in FRONTAL_CHANNELS]
    theta_frontal    = np.mean(band_power['theta'][:, frontal_idx], axis=1)
    beta_frontal     = np.mean(band_power['beta'][:, frontal_idx],  axis=1)
    theta_beta_ratio = theta_frontal / (beta_frontal + 1e-10)

    covariances, power_feats = [], []
    for i in range(stacked.shape[0]):
        covariances.append(epoch_covariance(stacked[i]))

        total_power = (band_power['theta'][i] + band_power['alpha'][i] + band_power['beta'][i])
        rel_theta = band_power['theta'][i] / (total_power + 1e-10)
        rel_alpha = band_power['alpha'][i] / (total_power + 1e-10)
        rel_beta  = band_power['beta'][i]  / (total_power + 1e-10)

        power_feat = np.concatenate([
            band_power['theta'][i], band_power['alpha'][i], band_power['beta'][i],
            rel_theta, rel_alpha, rel_beta, [theta_beta_ratio[i]]
        ]).astype(np.float32)
        power_feats.append(power_feat)

    X_power = np.array(power_feats)
    _log(f"covariance+power features ({stacked.shape[0]} epochs)", t0)

    # ── Riemannian tangent space + scaling + model ─────────────────────────────
    t0 = time.time()
    X_riemann  = riemann.transform(covariances)
    X_combined = np.hstack([X_riemann, X_power])
    X_scaled   = scaler.transform(X_combined)
    _log("riemann.transform + scale", t0)

    t0 = time.time()
    epoch_probs = model.predict_proba(X_scaled)[:, 1]
    # Percentile aggregation, matching final_model.py's aggregate_probs() —
    # a subject only needs a cluster of high-prob epochs to flag, not a
    # majority of them. (The old 19-channel predictor used the median here;
    # that no longer matches how this model's threshold was tuned.)
    mean_prob = float(np.percentile(epoch_probs, AGG_PERCENTILE))
    _log("model.predict_proba", t0)

    prediction = 1 if mean_prob >= OPTIMAL_THRESHOLD else 0
    label      = "ADHD" if prediction == 1 else "Non-ADHD"

    # ── Display-only metrics (not fed into the model) ──────────────────────────
    theta_power = float(np.mean(band_power['theta']))
    alpha_power = float(np.mean(band_power['alpha']))
    beta_power  = float(np.mean(band_power['beta']))

    delta_epochs, delta_power_arr = _band_epochs_and_power(data, *DELTA_BAND)
    gamma_epochs, gamma_power_arr = _band_epochs_and_power(data, *GAMMA_BAND)
    delta_power = float(np.mean(delta_power_arr[keep]))
    gamma_power = float(np.mean(gamma_power_arr[keep]))

    tb_ratio  = float(np.mean(theta_beta_ratio))
    alpha_epochs_kept = band_epochs['alpha']
    alpha_coh = _alpha_coherence(alpha_epochs_kept)

    entropy_target_idx = [EEG_CHANNELS.index(ch) for ch in _ENTROPY_TARGET_NAMES]
    entropy_trend = _sample_entropy_trend(alpha_epochs_kept, entropy_target_idx)
    sample_ent = float(np.mean(entropy_trend)) if entropy_trend else 0.0

    band_power_distribution = [
        round(delta_power, 4), round(theta_power, 4), round(alpha_power, 4),
        round(beta_power, 4), round(gamma_power, 4),
    ]

    _log("TOTAL /predict", request_t0)

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
