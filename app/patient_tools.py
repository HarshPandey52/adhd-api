"""
app/patient_tools.py
=====================
Tool definitions + dispatcher for the chat agent to read/write patient
history during a conversation. This is what makes the chatbot an agent
rather than a static Q&A bot — the LLM decides when to call these.

Wired into chat_service.py via Gemini's function-calling (tools) feature.
"""

from app.database import save_visit, get_patient_history, list_all_patients

# ── Tool definitions (Gemini function-calling schema) ─────────────────────────
PATIENT_TOOLS = [
    {
        "name": "get_patient_history",
        "description": (
            "Retrieves the full visit history for a patient by their ID, "
            "including past EEG predictions, doctor notes, and prescriptions "
            "from previous visits. Use this when the doctor asks about a "
            "patient's history, past results, previous notes, or wants to "
            "compare current findings to prior visits."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The patient identifier, e.g. 'P001'"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "save_visit_record",
        "description": (
            "Saves the current visit's EEG analysis, doctor notes, and/or "
            "prescription to a patient's record. Use this when the doctor "
            "explicitly asks to save, record, log, or store notes/prescription "
            "for the current patient and visit."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "The patient identifier, e.g. 'P001'"
                },
                "doctor_notes": {
                    "type": "string",
                    "description": "The doctor's observations/notes to save"
                },
                "prescription": {
                    "type": "string",
                    "description": "The doctor's prescription or recommendation to save"
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "list_known_patients",
        "description": (
            "Lists all patient IDs that have at least one visit on record. "
            "Use this if the doctor asks 'which patients do we have records for' "
            "or similar."
        ),
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
]


def execute_tool(tool_name: str, args: dict, current_prediction: dict | None = None) -> dict:
    """
    Dispatches a tool call by name. Returns a JSON-serializable result
    to be sent back to the model as the tool's output.
    """
    if tool_name == "get_patient_history":
        patient_id = args.get("patient_id", "").strip()
        if not patient_id:
            return {"error": "patient_id is required"}
        visits = get_patient_history(patient_id)
        return {
            "patient_id": patient_id,
            "visit_count": len(visits),
            "visits": visits
        }

    elif tool_name == "save_visit_record":
        patient_id = args.get("patient_id", "").strip()
        if not patient_id:
            return {"error": "patient_id is required"}
        doctor_notes = args.get("doctor_notes")
        prescription = args.get("prescription")
        # Attach the current session's prediction if available and not already provided
        prediction = current_prediction
        record = save_visit(patient_id, prediction, doctor_notes, prescription)
        return {"status": "saved", "record": record}

    elif tool_name == "list_known_patients":
        return {"patient_ids": list_all_patients()}

    else:
        return {"error": f"Unknown tool: {tool_name}"}
