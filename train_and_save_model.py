import gc
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.covariance import OAS
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ===============================
# CONFIG
# ===============================
DATA_DIR = "D:\Out"   # <-- set to your real path (see os.walk snippet)
FILE_PATTERN = "sub-*.csv"                     # excludes manifest.csv

SFREQ = 500                             # native TDBRAIN sampling rate
PROCESS_SFREQ = 125                     # downsample to this before filtering/epoching
                                         # (still >4x the 30 Hz beta ceiling — plenty of
                                         # headroom — and cuts filter+covariance cost 4x)
EPOCH_DURATION = 3.0                    # seconds
EPOCH_OVERLAP = 0.0                     # non-overlapping — half the epochs of 1.5s overlap,
                                         # and each epoch is now statistically independent
MAX_EPOCHS_PER_SUBJECT = 15             # cap per subject; set to None to disable

# All 32 recorded channels: 26 scalp EEG + 6 EOG/ECG/EMG auxiliary channels
EEG_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4',
    'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz', 'CP4',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2',
    'VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'Mass',
]

FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
}

LABEL_COLUMN = "indication"             # raw diagnosis field: ADHD / HEALTHY / MDD / OCD / SMC / ...

TEST_SIZE = 0.10
RANDOM_STATE = 42
MEAN_SUBSAMPLE_SIZE = 2000              # covariances used to estimate the Riemannian mean

# class-imbalance handling
POS_WEIGHT_MULTIPLIER = 1.8             # multiply natural scale_pos_weight (n_neg/n_pos) by
                                         # this to push the model harder on ADHD errors.
                                         # 1.0 = natural ratio (what was used before).
AGG_PERCENTILE = 75                     # subject-level score = this percentile of its epoch
                                         # probabilities, not the median. A subject only needs
                                         # a cluster of high-prob epochs to flag, not a majority.

RNG = np.random.default_rng(RANDOM_STATE)


# ===============================
# SIGNAL PROCESSING (replaces mne)
# ===============================
def resample_signal(data, orig_sfreq, target_sfreq):
    """data: (n_channels, n_times) -> resampled to target_sfreq using
    polyphase filtering (scipy.signal.resample_poly), which is fast and
    avoids aliasing. 500 -> 125 Hz reduces to up=1, down=4 exactly."""
    from math import gcd
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
    project the FULL set of covariances into tangent space (one pass, no
    iteration needed for that part)."""

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



# ===============================
# STEP 1 — DISCOVER SUBJECT FILES + LABELS
# ===============================
def load_subject_index(data_dir):
    """One row per file: subject id + label, read cheaply (just first row).
    Label rule: ADHD -> 1, everything else -> 0."""
    files = sorted(glob.glob(os.path.join(data_dir, FILE_PATTERN)))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {data_dir} matching {FILE_PATTERN}")

    records = []
    for f in files:
        head = pd.read_csv(f, nrows=1)
        if LABEL_COLUMN not in head.columns:
            print(f"  skipping {f}: no '{LABEL_COLUMN}' column")
            continue
        indication = str(head[LABEL_COLUMN].iloc[0]).strip().upper()
        label = 1 if indication == "ADHD" else 0
        records.append({
            "path": f,
            "subject_id": os.path.splitext(os.path.basename(f))[0],
            "label": label,
        })

    idx = pd.DataFrame(records)
    print(f"Usable subjects: {len(idx)}")
    print(idx["label"].value_counts().rename({1: "ADHD", 0: "non-ADHD"}))
    return idx


# ===============================
# STEP 2 — FEATURE EXTRACTION (per subject, memory-lean)
# ===============================
def extract_subject_features(path, label, subject_id):
    """Returns per-epoch COVARIANCES (not raw waveforms), power features,
    labels, and group ids for one subject."""
    df = pd.read_csv(path)
    missing = [c for c in EEG_CHANNELS if c not in df.columns]
    if missing:
        print(f"  {subject_id}: missing channels {missing}, skipping")
        return None

    data = df[EEG_CHANNELS].values.T.astype(np.float32)  # (n_channels, n_times)
    del df
    data = data - data.mean(axis=0, keepdims=True)         # average reference
    data = resample_signal(data, SFREQ, PROCESS_SFREQ)     # 500 -> 125 Hz, ~4x less work below

    band_power = {}
    band_epochs_for_cov = {}
    for band_name, (fmin, fmax) in FREQ_BANDS.items():
        filtered = bandpass_filter(data, PROCESS_SFREQ, fmin, fmax)
        ep_data = make_fixed_length_epochs(filtered, PROCESS_SFREQ, EPOCH_DURATION, EPOCH_OVERLAP)
        del filtered
        band_epochs_for_cov[band_name] = ep_data
        band_power[band_name] = np.mean(ep_data ** 2, axis=2)
    del data

    n_epochs = min(band_epochs_for_cov[b].shape[0] for b in FREQ_BANDS)
    n_samples = min(band_epochs_for_cov[b].shape[2] for b in FREQ_BANDS)

    if MAX_EPOCHS_PER_SUBJECT is not None and n_epochs > MAX_EPOCHS_PER_SUBJECT:
        keep = RNG.choice(n_epochs, size=MAX_EPOCHS_PER_SUBJECT, replace=False)
        keep.sort()
    else:
        keep = np.arange(n_epochs)

    for b in FREQ_BANDS:
        band_epochs_for_cov[b] = band_epochs_for_cov[b][keep, :, :n_samples]
        band_power[b] = band_power[b][keep]

    stacked = np.concatenate(
        [band_epochs_for_cov['theta'], band_epochs_for_cov['alpha'], band_epochs_for_cov['beta']],
        axis=1,
    )  # (n_epochs_kept, 96, n_samples) — only lives for this one subject, then discarded
    del band_epochs_for_cov

    frontal_idx = [EEG_CHANNELS.index(ch) for ch in ['Fp1', 'Fp2', 'Fz']]
    theta_frontal = np.mean(band_power['theta'][:, frontal_idx], axis=1)
    beta_frontal = np.mean(band_power['beta'][:, frontal_idx], axis=1)
    theta_beta_ratio = theta_frontal / (beta_frontal + 1e-10)

    cov_subj, power_subj = [], []
    for i in range(stacked.shape[0]):
        cov_subj.append(epoch_covariance(stacked[i]))   # <- collapse to covariance HERE

        total_power = band_power['theta'][i] + band_power['alpha'][i] + band_power['beta'][i]
        power_feat = np.concatenate([
            band_power['theta'][i], band_power['alpha'][i], band_power['beta'][i],
            band_power['theta'][i] / (total_power + 1e-10),
            band_power['alpha'][i] / (total_power + 1e-10),
            band_power['beta'][i] / (total_power + 1e-10),
            [theta_beta_ratio[i]],
        ]).astype(np.float32)
        power_subj.append(power_feat)

    del stacked  # the only thing kept per epoch from here on is the small covariance

    y_subj = [label] * len(cov_subj)
    g_subj = [subject_id] * len(cov_subj)
    return cov_subj, power_subj, y_subj, g_subj



def build_dataset(subject_index, log_every=50):
    X_cov, X_power, y_all, groups = [], [], [], []
    for i, (_, row) in enumerate(subject_index.iterrows()):
        result = extract_subject_features(row["path"], row["label"], row["subject_id"])
        if result is not None:
            cv, xp, y, g = result
            X_cov.extend(cv)
            X_power.extend(xp)
            y_all.extend(y)
            groups.extend(g)
        gc.collect()
        if (i + 1) % log_every == 0:
            print(f"  ...{i + 1}/{len(subject_index)} subjects processed, "
                  f"{len(X_cov)} epochs so far")

    return X_cov, np.array(X_power, dtype=np.float32), np.array(y_all), np.array(groups)

# ===============================
# STEP 3 — THRESHOLD TUNING (train-only)
# ===============================
def aggregate_probs(probs):
    """Subject-level score from its epoch probabilities. Percentile instead of
    median so a subject only needs a cluster of high-confidence epochs to flag,
    not a majority of them."""
    return np.percentile(probs, AGG_PERCENTILE)


def tune_threshold(model, X_val, y_val, groups_val, metric="balanced_accuracy"):
    """Sweep thresholds and pick the best by `metric` (default: balanced_accuracy,
    not raw accuracy — raw accuracy is biased toward specificity on an imbalanced
    set like this one, which is why sensitivity was stuck at 0.25 before).
    Prints the full curve so you can eyeball the sensitivity/specificity tradeoff
    and override the pick manually if you want a different operating point."""
    probs = model.predict_proba(X_val)[:, 1]
    best_t, best_score = 0.5, -1
    rows = []
    for t in np.arange(0.30, 0.71, 0.01):
        preds, trues = [], []
        for subj in np.unique(groups_val):
            idx = np.where(groups_val == subj)[0]
            p = aggregate_probs(probs[idx])
            preds.append(1 if p >= t else 0)
            trues.append(y_val[idx][0])

        acc = accuracy_score(trues, preds)
        bal_acc = balanced_accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(trues, preds, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        rows.append((t, acc, bal_acc, f1, sens, spec))

        score = {"accuracy": acc, "balanced_accuracy": bal_acc, "f1": f1}[metric]
        if score > best_score:
            best_score, best_t = score, t

    print(f"\n{'t':>5} {'acc':>7} {'bal_acc':>8} {'f1':>7} {'sens':>7} {'spec':>7}")
    for t, acc, bal_acc, f1, sens, spec in rows:
        marker = "  <-- picked" if abs(t - best_t) < 1e-9 else ""
        print(f"{t:5.2f} {acc:7.3f} {bal_acc:8.3f} {f1:7.3f} {sens:7.3f} {spec:7.3f}{marker}")

    return best_t



# ===============================
# MAIN
# ===============================
def main():
    subject_index = load_subject_index(DATA_DIR)

    print("\nExtracting features for all subjects (covariances only, memory-lean)...")
    X_cov, X_power, y_epochs, groups = build_dataset(subject_index)
    print("Total epochs:", len(X_cov))
    print("Total subjects used:", len(np.unique(groups)))

    X_cov = np.stack(X_cov)  # (n_epochs, 96, 96) float32 — this is the big array now,
                              # and it's small: n_epochs * 96*96*4 bytes

    # ---- subject-wise 90/10 split ----
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X_cov, y_epochs, groups))

    val_splitter = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    tr_sub_idx, val_sub_idx = next(val_splitter.split(
        np.zeros(len(train_idx)), y_epochs[train_idx], groups[train_idx]
    ))
    fit_idx = train_idx[tr_sub_idx]
    val_idx = train_idx[val_sub_idx]

    print(f"\nSubjects -> fit: {len(np.unique(groups[fit_idx]))}, "
          f"val: {len(np.unique(groups[val_idx]))}, "
          f"test: {len(np.unique(groups[test_idx]))}")

    # ---- Riemannian tangent space (mean estimated on FIT only, subsampled) ----
    riemann = RiemannTangentSpace()
    X_fit_r = riemann.fit_transform(X_cov[fit_idx])
    X_val_r = riemann.transform(X_cov[val_idx])
    X_test_r = riemann.transform(X_cov[test_idx])
    del X_cov
    gc.collect()

    X_fit = np.hstack([X_fit_r, X_power[fit_idx]])
    X_val = np.hstack([X_val_r, X_power[val_idx]])
    X_test = np.hstack([X_test_r, X_power[test_idx]])

    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    y_fit = y_epochs[fit_idx]
    y_val = y_epochs[val_idx]
    y_test = y_epochs[test_idx]

    # ---- model ----
    n_pos = (y_fit == 1).sum()
    n_neg = (y_fit == 0).sum()
    pos_weight = (n_neg / max(n_pos, 1)) * POS_WEIGHT_MULTIPLIER
    print(f"\nTrain epochs -> ADHD: {n_pos}, non-ADHD: {n_neg}, "
          f"scale_pos_weight: {pos_weight:.2f} (natural ratio x{POS_WEIGHT_MULTIPLIER})")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        scale_pos_weight=pos_weight,
        random_state=RANDOM_STATE,
    )
    model.fit(X_fit, y_fit)

    threshold = tune_threshold(model, X_val, y_val, groups[val_idx])
    print(f"\nTuned decision threshold (from validation subjects): {threshold:.2f}")

    test_probs = model.predict_proba(X_test)[:, 1]
    subj_preds, subj_true, subj_probs = [], [], []
    for subj in np.unique(groups[test_idx]):
        idx = np.where(groups[test_idx] == subj)[0]
        p = aggregate_probs(test_probs[idx])
        subj_preds.append(1 if p >= threshold else 0)
        subj_true.append(y_test[idx][0])
        subj_probs.append(p)

    acc = accuracy_score(subj_true, subj_preds)
    auc = roc_auc_score(subj_true, subj_probs)
    tn, fp, fn, tp = confusion_matrix(subj_true, subj_preds, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")

    print("\n===== FINAL HELD-OUT TEST RESULTS (10% of subjects) =====")
    print("Subjects in test:", len(subj_true))
    print("Accuracy:", acc)
    print("AUC:", auc)
    print("Sensitivity:", sens)
    print("Specificity:", spec)

    importance = model.feature_importances_
    riemann_features = [f"Riemann_{i}" for i in range(X_fit_r.shape[1])]
    power_features = [f"Power_{i}" for i in range(X_power.shape[1])]
    feat_df = pd.DataFrame({
        "Feature": riemann_features + power_features,
        "Importance": importance,
    }).sort_values("Importance", ascending=False)

    plt.figure()
    plt.barh(feat_df["Feature"][:20][::-1], feat_df["Importance"][:20][::-1])
    plt.xlabel("Importance")
    plt.title("Top 20 Features")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("\nSaved feature_importance.png")

    # ===============================
    # SAVE DEPLOYMENT ARTIFACTS
    # ===============================
    import joblib

    joblib.dump(riemann, "riemann_pipeline.pkl")
    joblib.dump(scaler, "scaler.pkl")
    model.get_booster().save_model("xgb_model.json")
    print("\nSaved riemann_pipeline.pkl, scaler.pkl, xgb_model.json")

if __name__ == "__main__":
    main()