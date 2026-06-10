"""
app/main.py
===========
FastAPI application entry point.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schema import EEGInput, ADHDPrediction
from app.predictor import predict_adhd

app = FastAPI(
    title="ADHD Prediction API",
    description=(
        "Predicts ADHD from raw EEG signals using a "
        "Riemannian geometry + XGBoost pipeline."
    ),
    version="1.0.0",
)

# ── CORS: allow your frontend to call this API ────────────────────────────────
# Replace "*" with your actual frontend URL in production, e.g.:
# allow_origins=["https://your-site.com"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "ADHD Prediction API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=ADHDPrediction, tags=["Prediction"])
def predict(data: EEGInput):
    """
    ### Input
    Raw EEG time-series for one subject:
    - `eeg_data`: 2-D array of shape **[n_timepoints, 19]**
    - Values in **µV** (microvolts)
    - Minimum **750 samples** (3 seconds at 250 Hz)
    - Channel order: `Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T7 T8 P7 P8 Fz Cz Pz`

    ### Output
    - `prediction`: `1` = ADHD, `0` = Non-ADHD
    - `label`: `"ADHD"` or `"Non-ADHD"`
    - `confidence`: float 0–1
    - `confidence_pct`: e.g. `"78.34%"`
    - `threshold_used`: decision threshold (default 0.45)
    """
    try:
        result = predict_adhd(data)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
