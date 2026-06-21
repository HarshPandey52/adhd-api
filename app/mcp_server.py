"""
app/mcp_server.py
==================
A real Model Context Protocol (MCP) server exposing the patient-history
tools (get_patient_history, save_visit_record, list_known_patients) over
JSON-RPC 2.0 + Server-Sent Events, per ArmorIQ's MCP Format Requirements:
https://docs.armoriq.ai/sdk/mcp-directory/mcp-format/index

This replaces the old in-process patient_tools.py dispatcher. It is now a
standalone protocol server that:
  - Google ADK's McpToolset can connect to as a tool source
  - ArmorIQ can register in its MCP Registry and enforce policy on

Mounted into the main FastAPI app at /mcp (see app/main.py).
"""

import json
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.database import save_visit, get_patient_history, list_all_patients

router = APIRouter()

MCP_SERVER_NAME = "neurosakhi-patient-mcp"
MCP_SERVER_VERSION = "1.0.0"

# ── Tool definitions, per MCP tools/list schema ───────────────────────────────
TOOLS = [
    {
        "name": "get_patient_history",
        "description": (
            "Retrieves the full visit history for a patient by their ID, "
            "including past EEG predictions, doctor notes, and prescriptions "
            "from previous visits. Use this when the doctor asks about a "
            "patient's history, past results, previous notes, or wants to "
            "compare current findings to prior visits."
        ),
        "inputSchema": {
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
        "inputSchema": {
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
                },
                "prediction_json": {
                    "type": "string",
                    "description": (
                        "JSON-stringified prediction result from /predict for this "
                        "visit, if available (optional)"
                    )
                }
            },
            "required": ["patient_id"]
        }
    },
    {
        "name": "list_known_patients",
        "description": (
            "Lists all patient IDs that have at least one visit on record. "
            "Use this if the doctor asks which patients have records."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
]


def _sse(data: dict) -> str:
    """Wraps a JSON-RPC response in the required SSE envelope."""
    return f"event: message\ndata: {json.dumps(data)}\n\n"


def _execute_tool(tool_name: str, args: dict) -> dict:
    """Runs the actual tool logic against the SQLite-backed patient DB."""
    if tool_name == "get_patient_history":
        patient_id = (args.get("patient_id") or "").strip()
        if not patient_id:
            return {"error": "patient_id is required"}
        visits = get_patient_history(patient_id)
        return {"patient_id": patient_id, "visit_count": len(visits), "visits": visits}

    elif tool_name == "save_visit_record":
        patient_id = (args.get("patient_id") or "").strip()
        if not patient_id:
            return {"error": "patient_id is required"}
        prediction = None
        if args.get("prediction_json"):
            try:
                prediction = json.loads(args["prediction_json"])
            except (json.JSONDecodeError, TypeError):
                prediction = None
        record = save_visit(
            patient_id,
            prediction,
            args.get("doctor_notes"),
            args.get("prescription"),
        )
        return {"status": "saved", "record": record}

    elif tool_name == "list_known_patients":
        return {"patient_ids": list_all_patients()}

    else:
        return {"error": f"Unknown tool: {tool_name}"}


async def _handle_jsonrpc(request_data: dict) -> dict:
    method = request_data.get("method")
    msg_id = request_data.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": MCP_SERVER_NAME, "version": MCP_SERVER_VERSION},
            },
        }

    elif method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {"tools": TOOLS}}

    elif method == "tools/call":
        params = request_data.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        result_data = _execute_tool(tool_name, arguments)

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [
                    {"type": "text", "text": json.dumps(result_data)}
                ]
            },
        }

    else:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


@router.get("/mcp")
async def mcp_health_check():
    """
    Some MCP registries/validators (e.g. ArmorIQ's onboarding flow) probe
    the endpoint with a plain GET before sending JSON-RPC POST requests.
    Respond with basic server info so that check doesn't 405.
    """
    return {
        "status": "ok",
        "server": MCP_SERVER_NAME,
        "version": MCP_SERVER_VERSION,
        "protocol": "JSON-RPC 2.0 over HTTP, POST only, SSE response",
    }


@router.post("/mcp")
async def mcp_endpoint(request: Request):
    """
    JSON-RPC 2.0 over HTTP, SSE response — per ArmorIQ MCP format spec.
    Handles initialize / tools/list / tools/call.
    """
    request_data = await request.json()
    response_data = await _handle_jsonrpc(request_data)

    async def stream():
        yield _sse(response_data)

    return StreamingResponse(stream(), media_type="text/event-stream")
