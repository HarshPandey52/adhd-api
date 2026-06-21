"""
app/chat_schema.py
===================
Pydantic models for the /chat endpoint.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="The user's latest message")
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Prior turns in the conversation, oldest first"
    )
    eeg_context: Optional[dict] = Field(
        default=None,
        description="Optional dict of the latest prediction result for grounding."
    )


class ChatResponse(BaseModel):
    reply: str = Field(..., description="The assistant's reply text")
    tool_calls_made: List[str] = Field(
        default_factory=list,
        description="Names of tools the agent invoked while answering, for transparency/debugging"
    )
