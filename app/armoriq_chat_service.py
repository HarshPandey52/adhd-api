"""
app/armoriq_chat_service.py
=============================
Runs the NeuroSakhi ADK agent (app/adk_agent.py) per request, wrapped by
ArmorIQ's policy/audit layer (ArmorIQADK). Every tool call the agent makes
to the patient MCP server is verified against a signed intent token before
it executes — this is the real ArmorIQ integration, not a regex filter.

Pattern follows https://docs.armoriq.ai/sdk/integrations/google-adk exactly:
  1. ArmorIQADK is constructed once per process (module-level singleton)
  2. Per request: armoriq.for_user(email, goal=message) opens a scope
  3. scope.install(root_agent) attaches the before/after tool callbacks
  4. Run the ADK runner as normal
  5. scope.uninstall(root_agent) in `finally` — always

ARMORIQ_API_KEY and GEMINI_API_KEY must be set as environment variables
on the server (Render → Environment). Neither is ever sent to the browser.
"""

import os
import logging

from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from armoriq_sdk.integrations.google_adk import ArmorIQADK

from app.adk_agent import build_root_agent
from app.chat_schema import ChatRequest

logger = logging.getLogger(__name__)

ARMORIQ_API_KEY = os.environ.get("ARMORIQ_API_KEY", "")
ARMORIQ_USE_PRODUCTION = os.environ.get("ARMORIQ_USE_PRODUCTION", "true").lower() == "true"

# Fixed placeholder identity for the prototype stage — every doctor session
# is scoped under this email until real doctor accounts/login are added.
# ArmorIQ's policy/audit trail is keyed off this value.
DEFAULT_DOCTOR_EMAIL = os.environ.get("DEFAULT_DOCTOR_EMAIL", "doctor@neurosakhi.demo")

_armoriq_client: "ArmorIQADK | None" = None


def _get_armoriq_client() -> ArmorIQADK:
    global _armoriq_client
    if _armoriq_client is None:
        if not ARMORIQ_API_KEY:
            raise RuntimeError(
                "ARMORIQ_API_KEY is not set on the server. "
                "Add it under Render → Environment Variables. "
                "Get a key at https://platform.armoriq.ai"
            )
        _armoriq_client = ArmorIQADK(
            api_key=ARMORIQ_API_KEY,
            use_production=ARMORIQ_USE_PRODUCTION,
        )
    return _armoriq_client


async def get_chat_reply(req: ChatRequest) -> tuple[str, list[str]]:
    """
    Runs one chat turn through the ArmorIQ-wrapped ADK agent.
    Returns (reply_text, tool_names_called).

    Note: ADK's runner manages its own multi-turn session internally per
    session_id. For simplicity in this prototype, each HTTP request opens
    a fresh session and we replay `req.history` as context via the prompt;
    a production version would persist ADK sessions across requests.
    """
    if not os.environ.get("GEMINI_API_KEY") and not os.environ.get("GOOGLE_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set on the server. "
            "Add it under Render → Environment Variables."
        )

    armoriq = _get_armoriq_client()
    root_agent = build_root_agent()

    # Fold prior turns into the message ADK sees, since each HTTP request
    # is a fresh ADK session in this prototype.
    history_text = ""
    if req.history:
        lines = []
        for turn in req.history:
            speaker = "Doctor" if turn.role == "user" else "Assistant"
            lines.append(f"{speaker}: {turn.content}")
        history_text = "Conversation so far:\n" + "\n".join(lines) + "\n\n"

    eeg_context_text = _build_eeg_context_text(req.eeg_context)
    full_message = f"{history_text}{eeg_context_text}\n\nDoctor's latest message: {req.message}"

    tools_called: list[str] = []
    reply_text = ""

    scope = armoriq.for_user(DEFAULT_DOCTOR_EMAIL, goal=req.message)
    scope.install(root_agent)

    try:
        runner = InMemoryRunner(agent=root_agent, app_name="neurosakhi")
        session = await runner.session_service.create_session(
            app_name=runner.app_name, user_id=DEFAULT_DOCTOR_EMAIL
        )

        async for event in runner.run_async(
            user_id=DEFAULT_DOCTOR_EMAIL,
            session_id=session.id,
            new_message=genai_types.Content(
                role="user", parts=[genai_types.Part(text=full_message)]
            ),
        ):
            # Collect tool-call names for transparency in the API response
            if hasattr(event, "content") and event.content:
                for part in getattr(event.content, "parts", []) or []:
                    fn_call = getattr(part, "function_call", None)
                    if fn_call is not None:
                        tools_called.append(fn_call.name)
                    text_piece = getattr(part, "text", None)
                    if text_piece:
                        reply_text += text_piece

    except Exception as e:
        logger.exception("ADK agent run failed")
        raise RuntimeError(f"Agent run failed: {e}") from e

    finally:
        scope.uninstall(root_agent)

    reply_text = reply_text.strip() or "Sorry, I had trouble responding. Please try again."
    return reply_text, tools_called


def _build_eeg_context_text(eeg_context: dict | None) -> str:
    if not eeg_context:
        return "No EEG analysis has been run yet in this session."

    lines = ["CURRENT SESSION'S EEG ANALYSIS (not yet saved unless explicitly saved via a tool call):"]
    label_map = {
        "label": "Prediction", "confidence_pct": "Confidence",
        "theta_power": "Theta Power (µV²)", "alpha_power": "Alpha Power (µV²)",
        "beta_power": "Beta Power (µV²)", "delta_power": "Delta Power (µV²)",
        "gamma_power": "Gamma Power (µV²)", "theta_beta_ratio": "Theta/Beta Ratio",
        "alpha_coherence": "Alpha Coherence", "sample_entropy": "Sample Entropy",
    }
    for key, display_name in label_map.items():
        if key in eeg_context and eeg_context[key] is not None:
            lines.append(f"- {display_name}: {eeg_context[key]}")
    return "\n".join(lines)
