# ADHD Prediction API — v1.1 (with real EEG metrics)

## What's new in this version

The `/predict` endpoint now returns real computed metrics alongside the
prediction, instead of placeholder values:

```json
{
  "prediction": 1,
  "label": "ADHD",
  "confidence": 0.7609,
  "confidence_pct": "76.09%",
  "threshold_used": 0.45,

  "theta_power": 2.41,
  "alpha_power": 1.18,
  "beta_power": 0.87,
  "delta_power": 3.05,
  "gamma_power": 0.32,

  "theta_beta_ratio": 2.77,
  "alpha_coherence": 0.41,
  "sample_entropy": 1.62,

  "band_power_distribution": [3.05, 2.41, 1.18, 0.87, 0.32],
  "entropy_trend": [1.5, 1.6, 1.7, 1.55, ...]
}
```

## How metrics are computed

- **theta_power / alpha_power / beta_power**: mean band power (µV²) across
  all epochs and channels, computed from the same filtered epochs used by
  the prediction model.
- **delta_power / gamma_power**: same approach, additional bandpass filters
  (1-4 Hz and 30-45 Hz) — display-only, not fed into the model.
- **theta_beta_ratio**: frontal (Fp1, Fp2, Fz) theta/beta ratio, averaged
  across epochs — same calculation used as a model feature.
- **alpha_coherence**: mean pairwise phase-locking value (PLV) across all
  channel pairs in the alpha band, averaged over epochs. Range 0-1.
- **sample_entropy / entropy_trend**: sample entropy (m=2) computed per
  epoch on a subset of channels (Fp1, Fp2, C3, C4, O1, O2, Fz), averaged
  for the overall value and reported per-epoch for the trend chart.

## Replace your existing deployment

1. Replace `app/` folder contents with this version's `app/` folder
2. Update `requirements.txt` (adds `scipy`)
3. Keep your existing `model/` folder — **no retraining needed**,
   `train_and_save_model.py` is unchanged
4. Push to GitHub → Render auto-redeploys

## Deploy (same as before)

```bash
git add .
git commit -m "Add real EEG metrics to API response"
git push origin main
```

Render will rebuild automatically.

⚠️ Note: sample entropy computation adds a few seconds to response time
for long recordings due to its O(n²) nature — this is normal.
