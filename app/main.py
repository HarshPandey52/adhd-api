"""
app/main.py
===========
FastAPI application entry point.

Endpoints:
  GET  /                    health check
  GET  /health               health check
  POST /predict               ADHD EEG prediction
  POST /chat                  agentic AI Companion chat (tool-calling enabled)
  GET  /patients/{id}/history  patient visit history (for direct UI display)
  POST /patients/visit         save a visit record directly (non-chat path)
  PUT  /patients/visit          update notes/prescription on an existing visit
  GET  /patients                list all known patient IDs
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schema import EEGInput, ADHDPrediction
from app.predictor import predict_adhd
from app.chat_schema import ChatRequest, ChatResponse
from app.chat_service import get_chat_reply
from app.database import (
    init_db, save_visit, get_patient_history,
    update_visit_notes, list_all_patients,
)
from app.patient_schema import (
    SaveVisitRequest, UpdateVisitRequest,
    PatientHistoryResponse, PatientListResponse, VisitRecord,
)

app = FastAPI(
    title="ADHD Prediction API",
    description=(
        "EEG-based ADHD screening with real computed metrics, an agentic "
        "AI Companion (tool-calling chat), and patient visit history. "
        "⚠️ PROTOTYPE: no authentication — do not use with real patient data."
    ),
    version="1.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "ADHD Prediction API is running."}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}


# ── Prediction ──────────────────────────────────────────────────────────────
@app.post("/predict", response_model=ADHDPrediction, tags=["Prediction"])
def predict(data: EEGInput):
    try:
        return predict_adhd(data)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Agentic chat ──────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Agentic chat endpoint. The model can autonomously call tools to:
      - look up a patient's visit history
      - save the current visit's notes/prescription
      - list known patients
    Proxied server-side to Gemini — API key never touches the browser.
    """
    try:
        reply, tools_called = await get_chat_reply(req)
        return ChatResponse(reply=reply, tool_calls_made=tools_called)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Patient history (direct REST path, for UI rendering) ─────────────────────
@app.get("/patients/{patient_id}/history", response_model=PatientHistoryResponse, tags=["Patients"])
def get_history(patient_id: str):
    """Returns all visits for a patient, most recent first."""
    visits = get_patient_history(patient_id)
    return PatientHistoryResponse(
        patient_id=patient_id,
        visit_count=len(visits),
        visits=[VisitRecord(**v) for v in visits],
    )


@app.post("/patients/visit", response_model=VisitRecord, tags=["Patients"])
def create_visit(req: SaveVisitRequest):
    """Saves a new visit record directly (used by the upload-analyze flow,
    independent of the chat agent)."""
    record = save_visit(
        patient_id=req.patient_id,
        prediction=req.prediction,
        doctor_notes=req.doctor_notes,
        prescription=req.prescription,
    )
    return VisitRecord(**record)


@app.put("/patients/visit", response_model=VisitRecord, tags=["Patients"])
def edit_visit(req: UpdateVisitRequest):
    """Updates notes/prescription on an existing visit record."""
    record = update_visit_notes(req.visit_id, req.doctor_notes, req.prescription)
    if not record:
        raise HTTPException(status_code=404, detail=f"Visit {req.visit_id} not found")
    return VisitRecord(**record)


@app.get("/patients", response_model=PatientListResponse, tags=["Patients"])
def list_patients():
    """Returns all patient IDs that have at least one visit on record."""
    return PatientListResponse(patient_ids=list_all_patients())
