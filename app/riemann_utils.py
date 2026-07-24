"""
app/riemann_utils.py
=====================
Shared signal-processing + Riemannian-geometry code used by BOTH the
training script (final_model.py) and the inference code (predictor.py).

WHY THIS FILE HAS TO EXIST:
joblib/pickle does not save a class's code — it saves a *reference* to
where that class lives (its module path) and expects the same class,
importable from the same path, to exist when you unpickle later. The
original final_model.py defined RiemannTangentSpace inline and was run
as `__main__`, so `joblib.dump(riemann, ...)` pickled a reference to
`__main__.RiemannTangentSpace`. That reference does not exist inside the
FastAPI app process, so `joblib.load()` would fail there.

Fix: define everything here, import it from BOTH final_model.py and
predictor.py, and re-run training once so the saved .pkl references
`app.riemann_utils.RiemannTangentSpace` instead of `__main__`.

⚠️ ACTION REQUIRED: update final_model.py to
`from app.riemann_utils import (...)` instead of defining these inline,
then re-run it to regenerate riemann_pipeline.pkl against this module
path. The existing riemann_pipeline.pkl (pickled from __main__) will
NOT load correctly until you do this.
"""

from math import gcd

import numpy as np
from scipy import signal
from sklearn.covariance import OAS

RANDOM_STATE = 42
MEAN_SUBSAMPLE_SIZE = 2000  # covariances used to estimate the Riemannian mean

RNG = np.random.default_rng(RANDOM_STATE)


# ===============================
# SIGNAL PROCESSING (no mne — matches final_model.py exactly)
# ===============================
def resample_signal(data, orig_sfreq, target_sfreq):
    """data: (n_channels, n_times) -> resampled to target_sfreq using
    polyphase filtering (scipy.signal.resample_poly)."""
    g = gcd(int(orig_sfreq), int(target_sfreq))
    up, down = int(target_sfreq) // g, int(orig_sfreq) // g
    return signal.resample_poly(data, up, down, axis=1).astype(np.float32)


def bandpass_filter(data, sfreq, fmin, fmax, order=4):
    """data: (n_channels, n_times) float32 -> filtered float32."""
    nyq = sfreq / 2.0
    sos = signal.butter(order, [fmin / nyq, fmax / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, data, axis=1).astype(np.float32)


def make_fixed_length_epochs(data, sfreq, duration, overlap):
    """data: (n_channels, n_times) -> (n_epochs, n_channels, win_samples)."""
    n_channels, n_times = data.shape
    win = int(round(duration * sfreq))
    step = win - int(round(overlap * sfreq))
    if step <= 0:
        raise ValueError("overlap must be smaller than duration")
    starts = range(0, n_times - win + 1, step)
    epochs = np.stack([data[:, s:s + win] for s in starts], axis=0)
    return epochs


# ===============================
# RIEMANNIAN GEOMETRY (replaces pyriemann)
# ===============================
def _eig_sym(M):
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals = np.clip(eigvals, 1e-10, None)
    return eigvals, eigvecs


def sqrtm_sym(M):
    w, v = _eig_sym(M)
    return (v @ np.diag(np.sqrt(w)) @ v.T).astype(np.float32)


def invsqrtm_sym(M):
    w, v = _eig_sym(M)
    return (v @ np.diag(1.0 / np.sqrt(w)) @ v.T).astype(np.float32)


def logm_sym(M):
    w, v = _eig_sym(M)
    return (v @ np.diag(np.log(w)) @ v.T).astype(np.float32)


def expm_sym(M):
    w, v = np.linalg.eigh(M)
    return (v @ np.diag(np.exp(w)) @ v.T).astype(np.float32)


def epoch_covariance(epoch):
    """epoch: (n_channels, n_times) float32 -> (n_channels, n_channels) float32
    OAS-shrunk covariance. Equivalent to pyriemann Covariances(estimator='oas')."""
    est = OAS(assume_centered=False)
    est.fit(epoch.T.astype(np.float64))  # sklearn's OAS wants float64 internally
    return est.covariance_.astype(np.float32)


def riemannian_mean(covs, max_iter=25, tol=1e-6):
    """Karcher/Frechet mean of a set of SPD matrices (affine-invariant metric)."""
    C = np.mean(covs, axis=0).astype(np.float32)
    for _ in range(max_iter):
        C_sqrt = sqrtm_sym(C)
        C_invsqrt = invsqrtm_sym(C)
        S = np.zeros_like(C)
        for cov in covs:
            S += logm_sym(C_invsqrt @ cov @ C_invsqrt)
        S /= len(covs)
        C_new = (C_sqrt @ expm_sym(S) @ C_sqrt).astype(np.float32)
        crit = np.linalg.norm(C_new - C, ord="fro")
        C = C_new
        if crit < tol:
            break
    return C


def tangent_space_vector(cov, mean_invsqrt):
    """Project one SPD covariance into the tangent space and vectorize
    (upper triangle, off-diagonals scaled by sqrt(2))."""
    S = logm_sym(mean_invsqrt @ cov @ mean_invsqrt)
    n = S.shape[0]
    coeffs = np.full((n, n), np.sqrt(2), dtype=np.float32)
    np.fill_diagonal(coeffs, 1.0)
    iu = np.triu_indices(n)
    return (S * coeffs)[iu]


class RiemannTangentSpace:
    """Fit the Frechet mean from a subsample of covariances (fast), then
    project covariances into tangent space.

    NOTE: this class must stay import-identical (same module path, same
    attribute names) between training and inference, or joblib.load()
    will break. Don't move or rename it without re-pickling.
    """

    def __init__(self, subsample_size=MEAN_SUBSAMPLE_SIZE):
        self.subsample_size = subsample_size
        self.mean_ = None
        self.mean_invsqrt_ = None

    def fit(self, covs):
        if len(covs) > self.subsample_size:
            idx = RNG.choice(len(covs), size=self.subsample_size, replace=False)
            sample = [covs[i] for i in idx]
        else:
            sample = covs
        self.mean_ = riemannian_mean(sample)
        self.mean_invsqrt_ = invsqrtm_sym(self.mean_)
        return self

    def transform(self, covs):
        return np.stack([tangent_space_vector(c, self.mean_invsqrt_) for c in covs])

    def fit_transform(self, covs):
        self.fit(covs)
        return self.transform(covs)
