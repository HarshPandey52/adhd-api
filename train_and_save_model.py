"""
train_and_save_model.py
=======================
Run this script ONCE on your dataset to train the model and save all
pipeline components to the /model directory.

Usage:
    python train_and_save_model.py --data /path/to/your_dataset.csv

The script saves:
    model/xgb_model.json        ← XGBoost model
    model/scaler.pkl            ← StandardScaler
    model/riemann_pipeline.pkl  ← Covariances + TangentSpace pipeline
"""

import argparse
import numpy as np
import mne
import pandas as pd
import joblib
import os

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from xgboost import XGBClassifier
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

# ── Constants (must match your notebook exactly) ──────────────────────────────
EEG_CHANNELS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz'
]
SFREQ = 250
FREQ_BANDS = {
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta':  (13, 30)
}
FRONTAL_CHANNELS = ['Fp1', 'Fp2', 'Fz']
EPOCH_DURATION   = 3.0
EPOCH_OVERLAP    = 1.5
OPTIMAL_THRESHOLD = 0.45
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")


# ── Feature extraction (identical to your notebook) ──────────────────────────
def extract_features(df_raw):
    X_epochs, X_power, y_epochs, groups = [], [], [], []
    frontal_idx = [EEG_CHANNELS.index(ch) for ch in FRONTAL_CHANNELS]

    for subj_id, subj_df in df_raw.groupby("ID"):
        label = 1 if subj_df["Class"].iloc[0] == "ADHD" else 0
        data  = subj_df[EEG_CHANNELS].values.T * 1e-6

        info = mne.create_info(EEG_CHANNELS, SFREQ, ch_types="eeg")
        raw  = mne.io.RawArray(data, info, verbose=False)
        raw.set_eeg_reference("average", verbose=False)

        band_epochs, band_power = {}, {}
        for band_name, (fmin, fmax) in FREQ_BANDS.items():
            filtered = raw.copy().filter(fmin, fmax, verbose=False)
            epochs   = mne.make_fixed_length_epochs(
                filtered, duration=EPOCH_DURATION,
                overlap=EPOCH_OVERLAP, preload=True, verbose=False
            )
            ep_data = epochs.get_data()
            band_epochs[band_name] = ep_data
            band_power[band_name]  = np.mean(ep_data**2, axis=2)

        stacked = np.concatenate(
            [band_epochs['theta'], band_epochs['alpha'], band_epochs['beta']],
            axis=1
        )

        theta_frontal     = np.mean(band_power['theta'][:, frontal_idx], axis=1)
        beta_frontal      = np.mean(band_power['beta'][:, frontal_idx],  axis=1)
        theta_beta_ratio  = theta_frontal / (beta_frontal + 1e-10)

        for i in range(stacked.shape[0]):
            X_epochs.append(stacked[i])

            total_power = (band_power['theta'][i] +
                           band_power['alpha'][i] +
                           band_power['beta'][i])

            rel_theta = band_power['theta'][i] / (total_power + 1e-10)
            rel_alpha = band_power['alpha'][i] / (total_power + 1e-10)
            rel_beta  = band_power['beta'][i]  / (total_power + 1e-10)

            power_feat = np.concatenate([
                band_power['theta'][i], band_power['alpha'][i],
                band_power['beta'][i],  rel_theta, rel_alpha,
                rel_beta, [theta_beta_ratio[i]]
            ])

            X_power.append(power_feat)
            y_epochs.append(label)
            groups.append(subj_id)

    return (np.array(X_epochs), np.array(X_power),
            np.array(y_epochs),  np.array(groups))


# ── Train on ALL data and save ────────────────────────────────────────────────
def train_and_save(data_path: str):
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("Loading dataset...")
    df_raw = pd.read_csv(data_path)
    print(f"Subjects: {df_raw['ID'].nunique()}")

    print("Extracting features (this may take a few minutes)...")
    X_epochs, X_power, y, groups = extract_features(df_raw)

    print("Fitting Riemannian pipeline on all data...")
    riemann = Pipeline([
        ('cov', Covariances(estimator='oas')),
        ('ts',  TangentSpace(metric='riemann'))
    ])
    X_riemann = riemann.fit_transform(X_epochs)

    X_all = np.hstack([X_riemann, X_power])

    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    print("Training XGBoost on all data...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_scaled, y)

    # ── Save all components ───────────────────────────────────────────────────
    model.save_model(os.path.join(MODEL_DIR, "xgb_model.json"))
    joblib.dump(scaler,  os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(riemann, os.path.join(MODEL_DIR, "riemann_pipeline.pkl"))

    print("\n✅ Saved:")
    print(f"   model/xgb_model.json")
    print(f"   model/scaler.pkl")
    print(f"   model/riemann_pipeline.pkl")
    print("\nNow run:  uvicorn app.main:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    args = parser.parse_args()
    train_and_save(args.data)
