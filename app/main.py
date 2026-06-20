"""
app/main.py
===========
FastAPI application entry point.
Includes /predict (ADHD EEG prediction) and /chat (AI Companion, proxied
to Gemini so the API key never touches the browser).
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schema import EEGInput, ADHDPrediction
from app.predictor import predict_adhd
from app.chat_schema import ChatRequest, ChatResponse
from app.chat_service import get_chat_reply

app = FastAPI(
    title="ADHD Prediction API",
    description=(
        "Predicts ADHD from raw EEG signals using a "
        "Riemannian geometry + XGBoost pipeline, with real EEG metrics "
        "and an AI Companion chat endpoint."
    ),
    version="1.2.0",
)

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
    - `eeg_data`: 2-D array of shape **[n_timepoints, 19]**, values in µV
    - Minimum **750 samples** (3 seconds at 250 Hz)
    - Channel order: `Fp1 Fp2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T7 T8 P7 P8 Fz Cz Pz`

    ### Output
    - `prediction` / `label` / `confidence` / `confidence_pct` / `threshold_used`
    - `theta_power`, `alpha_power`, `beta_power`, `delta_power`, `gamma_power` (µV²)
    - `theta_beta_ratio`, `alpha_coherence`, `sample_entropy`
    - `band_power_distribution`: [delta, theta, alpha, beta, gamma] for charts
    - `entropy_trend`: per-epoch sample entropy values for trend chart
    """
    try:
        return predict_adhd(data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    AI Companion chat endpoint. Proxies to Google Gemini server-side —
    the API key is read from the GEMINI_API_KEY environment variable
    and never sent to or exposed in the browser.

    ### Input
    - `message`: the user's latest message
    - `history`: prior conversation turns (optional)
    - `eeg_context`: latest /predict response dict, for grounding (optional)

    ### Output
    - `reply`: the assistant's text reply
    """
    try:
        reply = await get_chat_reply(req)
        return ChatResponse(reply=reply)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
