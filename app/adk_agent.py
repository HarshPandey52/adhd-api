"""
app/adk_agent.py
=================
Defines the NeuroSakhi AI Companion as a real Google ADK LlmAgent, with
the patient-history MCP server (app/mcp_server.py) wired in as its tool
source via McpToolset over SSE.

This replaces the old hand-rolled Gemini function-calling loop
(the previous app/chat_service.py). The agent itself is now framework-
native, which is what lets ArmorIQ's ArmorIQADK wrap it (see
app/armoriq_chat_service.py) — ArmorIQ only integrates with ADK agents
today, not arbitrary LLM call loops.

GEMINI_API_KEY must be set as an environment variable — ADK's Gemini
model reads it directly (GOOGLE_API_KEY or GEMINI_API_KEY).
"""

import os
from google.adk.agents import LlmAgent
from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams

# Ensure ADK's Gemini integration can find the key under either name it checks.
if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

ADK_MODEL = os.environ.get("ADK_MODEL", "gemini-2.5-flash")

# The MCP server is mounted at /mcp on this same FastAPI app (see app/main.py).
# In production on Render, the service calls itself over its own public URL.
SELF_BASE_URL = os.environ.get("SELF_BASE_URL", "http://127.0.0.1:8000")
MCP_URL = f"{SELF_BASE_URL}/mcp"

SYSTEM_INSTRUCTION = """You are NeuroSakhi's AI Companion — a neuroscience-informed assistant for doctors using an EEG-based ADHD screening platform.

YOUR ROLE:
- Explain EEG parameters (band powers, coherence, entropy) in clear language
- Discuss how ADHD-related EEG patterns relate to memory, attention, and behavior
- Help the doctor retrieve and record patient visit history using your tools
- Be warm, factual, supportive, and scientifically grounded

TOOLS — use them proactively when relevant, don't just describe what you'd do:
- get_patient_history: call this whenever the doctor mentions a patient ID, asks about history, past visits, prior notes, or wants to compare current results to previous ones
- save_visit_record: call this when the doctor asks you to save, record, log, or store notes/prescription for the current patient
- list_known_patients: call this if asked which patients have records

HARD CONSTRAINTS — always enforce:
1. NEVER claim to diagnose ADHD. Always frame as "screening indicators" or "patterns consistent with"
2. NEVER recommend specific medications, dosages, or treatment regimens on your own — you may relay what a doctor explicitly dictates to save, but don't suggest dosages yourself
3. NEVER predict long-term outcomes or prognoses with certainty
4. ALWAYS frame outputs as decision support for the doctor, not as the final word
5. Keep responses concise (3-5 sentences per point). No walls of text.
6. This is a PROTOTYPE system — if asked about data security, be honest that this demo has no doctor authentication yet (though tool calls are now verified via ArmorIQ) and real patient data handling needs further review.

TONE: Warm, grounded, evidence-informed. Like a knowledgeable research colleague.

When discussing EEG results, ground your answer in the actual values provided. End clinical-judgment responses with a reminder that NeuroSakhi is a screening/decision-support tool, not a diagnostic authority."""


def build_root_agent() -> LlmAgent:
    """
    Constructs the ADK LlmAgent wired to the patient MCP server.
    Called once per request in armoriq_chat_service.py (cheap to construct;
    the McpToolset opens its SSE connection lazily on first use).
    """
    patient_toolset = McpToolset(
        connection_params=SseConnectionParams(
            url=MCP_URL,
            timeout=15,
        ),
    )

    return LlmAgent(
        model=ADK_MODEL,
        name="neurosakhi_companion",
        description="NeuroSakhi AI Companion — EEG/ADHD decision support with patient history tools.",
        instruction=SYSTEM_INSTRUCTION,
        tools=[patient_toolset],
    )
