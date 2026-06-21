"""
app/main.py
===========
FastAPI application entry point.

Endpoints:
  GET  /                       health check
  GET  /health                 health check
  POST /predict                ADHD EEG prediction
  POST /chat                   agentic AI Companion chat (ADK + ArmorIQ enforced)
  POST /mcp                    the patient-tools MCP server (JSON-RPC/SSE)
  GET  /patients/{id}/history  patient visit history (for direct UI display)
  POST /patients/visit         save a visit record directly (non-chat path)
  PUT  /patients/visit         update notes/prescription on an existing visit
  GET  /patients               list all known patient IDs

⚠️ PROTOTYPE: no doctor authentication yet. Tool calls made by the agent
ARE now verified via ArmorIQ (real SDK, not a placeholder), but the HTTP
endpoints below them are still open. Add auth before using with real
patient data.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.schema import EEGInput, ADHDPrediction
from app.predictor import predict_adhd
from app.chat_schema import ChatRequest, ChatResponse
from app.armoriq_chat_service import get_chat_reply
from app.mcp_server import router as mcp_router
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
        "EEG-based ADHD screening with real computed metrics, a Google ADK "
        "agent wrapped in ArmorIQ policy enforcement (real SDK), and a "
        "patient-history MCP server. ⚠️ PROTOTYPE: no doctor auth yet."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the MCP server (JSON-RPC/SSE) at /mcp — this is what the ADK agent's
# McpToolset connects to, and what gets registered in ArmorIQ's MCP Registry.
app.include_router(mcp_router)


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


# ── Agentic chat (ADK + ArmorIQ) ──────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(req: ChatRequest):
    """
    Runs the NeuroSakhi AI Companion as a real Google ADK agent. Every tool
    call it makes (get_patient_history, save_visit_record, list_known_patients)
    is verified by ArmorIQ against a signed intent plan before execution —
    blocking unplanned/injected tool calls at the proxy level.
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
    visits = get_patient_history(patient_id)
    return PatientHistoryResponse(
        patient_id=patient_id,
        visit_count=len(visits),
        visits=[VisitRecord(**v) for v in visits],
    )


@app.post("/patients/visit", response_model=VisitRecord, tags=["Patients"])
def create_visit(req: SaveVisitRequest):
    record = save_visit(
        patient_id=req.patient_id,
        prediction=req.prediction,
        doctor_notes=req.doctor_notes,
        prescription=req.prescription,
    )
    return VisitRecord(**record)


@app.put("/patients/visit", response_model=VisitRecord, tags=["Patients"])
def edit_visit(req: UpdateVisitRequest):
    record = update_visit_notes(req.visit_id, req.doctor_notes, req.prescription)
    if not record:
        raise HTTPException(status_code=404, detail=f"Visit {req.visit_id} not found")
    return VisitRecord(**record)


@app.get("/patients", response_model=PatientListResponse, tags=["Patients"])
def list_patients():
    return PatientListResponse(patient_ids=list_all_patients())
