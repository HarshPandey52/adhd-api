"""
app/chat_service.py
====================
Server-side proxy to the Google Gemini API. The Gemini API key lives
only in an environment variable on the server (Render) and is never
exposed to the browser.

Set the env var GEMINI_API_KEY in your Render dashboard:
  Settings → Environment → Add Environment Variable
    Key:   GEMINI_API_KEY
    Value: <your key from aistudio.google.com/app/apikey>
"""

import os
import httpx

from app.chat_schema import ChatRequest

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL   = "gemini-2.5-flash"
GEMINI_URL     = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"

SYSTEM_PROMPT = """You are NeuroSakhi's AI Companion — a neuroscience-informed support assistant embedded in an EEG-based ADHD screening platform.

YOUR ROLE:
- Explain EEG parameters (band powers, coherence, entropy) in clear, non-technical language
- Discuss how ADHD-related EEG patterns relate to memory, attention, working memory, impulse control, and behavior
- Offer psychoeducation about ADHD: coping strategies, cognitive patterns, daily life impacts
- Be warm, factual, supportive, and scientifically grounded

HARD CONSTRAINTS — always enforce:
1. NEVER claim to diagnose ADHD. Always frame as "screening indicators" or "patterns consistent with"
2. NEVER recommend specific medications, dosages, or treatment regimens
3. NEVER predict long-term outcomes or prognoses with certainty
4. ALWAYS recommend professional consultation for any clinical decision
5. If someone shows distress, acknowledge emotions and provide crisis resources (iCall India: 9152987821)
6. Keep responses concise (3-5 sentences max per point). No walls of text.
7. Use simple, accessible language — the user may not have a neuroscience background

TONE: Warm, grounded, evidence-informed. Like a knowledgeable research mentor — not a therapist, not a doctor.

When discussing EEG results, always ground your answer in the actual values provided if available. End every response with a gentle reminder that NeuroSakhi is a screening tool and a licensed clinician should be consulted for any clinical decision."""


def _build_eeg_context_text(eeg_context: dict | None) -> str:
    if not eeg_context:
        return "No EEG analysis has been run yet in this session."

    lines = ["LATEST EEG ANALYSIS RESULTS:"]
    label_map = {
        "label": "Prediction",
        "confidence_pct": "Confidence",
        "theta_power": "Theta Power (µV²)",
        "alpha_power": "Alpha Power (µV²)",
        "beta_power": "Beta Power (µV²)",
        "delta_power": "Delta Power (µV²)",
        "gamma_power": "Gamma Power (µV²)",
        "theta_beta_ratio": "Theta/Beta Ratio",
        "alpha_coherence": "Alpha Coherence",
        "sample_entropy": "Sample Entropy",
    }
    for key, display_name in label_map.items():
        if key in eeg_context and eeg_context[key] is not None:
            lines.append(f"- {display_name}: {eeg_context[key]}")

    return "\n".join(lines)


async def get_chat_reply(req: ChatRequest) -> str:
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set on the server. "
            "Add it under Render → Environment Variables."
        )

    eeg_context_text = _build_eeg_context_text(req.eeg_context)
    full_system_prompt = f"{SYSTEM_PROMPT}\n\n{eeg_context_text}"

    # Build Gemini "contents" — alternating user/model turns
    contents = []
    for turn in req.history:
        gemini_role = "model" if turn.role == "assistant" else "user"
        contents.append({"role": gemini_role, "parts": [{"text": turn.content}]})
    contents.append({"role": "user", "parts": [{"text": req.message}]})

    payload = {
        "contents": contents,
        "systemInstruction": {"parts": [{"text": full_system_prompt}]},
        "generationConfig": {
            "maxOutputTokens": 700,
            "temperature": 0.6,
        },
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(GEMINI_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(f"Gemini API error {response.status_code}: {response.text}")

    data = response.json()

    try:
        candidates = data["candidates"]
        parts = candidates[0]["content"]["parts"]
        reply_text = "".join(p.get("text", "") for p in parts)
    except (KeyError, IndexError):
        reply_text = "Sorry, I had trouble responding. Please try again."

    return reply_text.strip() or "Sorry, I had trouble responding. Please try again."
