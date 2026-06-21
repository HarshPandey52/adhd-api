"""
app/chat_service.py
====================
Server-side agent loop: calls Gemini with function-calling tools enabled.
When Gemini decides to call a tool (e.g. get_patient_history), this code
executes it locally and feeds the result back to Gemini, which then
produces the final natural-language reply. This loop is what makes the
chatbot an "agent" rather than a single-shot Q&A bot.

GEMINI_API_KEY is read from an environment variable — never exposed
to the browser. Set it on Render under Environment Variables.
"""

import os
import httpx

from app.chat_schema import ChatRequest
from app.patient_tools import PATIENT_TOOLS, execute_tool

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

MAX_TOOL_ITERATIONS = 4  # safety cap on agent loop turns

SYSTEM_PROMPT = """You are NeuroSakhi's AI Companion — a neuroscience-informed assistant for doctors using an EEG-based ADHD screening platform.

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
6. This is a PROTOTYPE system — if asked about data security, be honest that this demo has no authentication or encryption yet and should not be used with real patient data until that is added.

TONE: Warm, grounded, evidence-informed. Like a knowledgeable research colleague.

When discussing EEG results, ground your answer in the actual values provided. End clinical-judgment responses with a reminder that NeuroSakhi is a screening/decision-support tool, not a diagnostic authority."""


def _build_eeg_context_text(eeg_context: dict | None) -> str:
    if not eeg_context:
        return "No EEG analysis has been run yet in this session."

    lines = ["CURRENT SESSION'S EEG ANALYSIS (not yet saved to any patient record unless explicitly saved):"]
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


def _to_gemini_contents(req: ChatRequest) -> list:
    contents = []
    for turn in req.history:
        role = "model" if turn.role == "assistant" else "user"
        contents.append({"role": role, "parts": [{"text": turn.content}]})
    contents.append({"role": "user", "parts": [{"text": req.message}]})
    return contents


async def get_chat_reply(req: ChatRequest) -> tuple[str, list[str]]:
    """
    Runs the agent loop: calls Gemini, executes any requested tool calls,
    feeds results back, repeats until Gemini gives a final text answer
    (or MAX_TOOL_ITERATIONS is hit).
    Returns (reply_text, list_of_tool_names_called).
    """
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set on the server. "
            "Add it under Render → Environment Variables."
        )

    eeg_context_text = _build_eeg_context_text(req.eeg_context)
    full_system_prompt = f"{SYSTEM_PROMPT}\n\n{eeg_context_text}"

    contents = _to_gemini_contents(req)
    tools_called = []

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        for _ in range(MAX_TOOL_ITERATIONS):
            payload = {
                "contents": contents,
                "systemInstruction": {"parts": [{"text": full_system_prompt}]},
                "tools": [{"functionDeclarations": PATIENT_TOOLS}],
                "generationConfig": {"maxOutputTokens": 700, "temperature": 0.4},
            }

            response = await client.post(GEMINI_URL, headers=headers, json=payload)

            if response.status_code != 200:
                raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")

            data = response.json()

            try:
                candidate = data["candidates"][0]
                parts = candidate["content"]["parts"]
            except (KeyError, IndexError):
                return "Sorry, I had trouble responding. Please try again.", tools_called

            function_calls = [p["functionCall"] for p in parts if "functionCall" in p]

            if not function_calls:
                # Final text answer reached
                text = "".join(p.get("text", "") for p in parts).strip()
                return text or "Sorry, I had trouble responding. Please try again.", tools_called

            # Model wants to call one or more tools — execute them and feed results back
            contents.append({"role": "model", "parts": parts})

            function_response_parts = []
            for fc in function_calls:
                tool_name = fc.get("name", "")
                tool_args = fc.get("args", {})
                tools_called.append(tool_name)

                result = execute_tool(tool_name, tool_args, current_prediction=req.eeg_context)

                function_response_parts.append({
                    "functionResponse": {
                        "name": tool_name,
                        "response": result
                    }
                })

            contents.append({"role": "user", "parts": function_response_parts})

    # Hit the iteration cap without a final answer
    return ("I gathered some information but need more steps than allowed to finish. "
            "Could you rephrase or narrow your request?"), tools_called
