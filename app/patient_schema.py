"""
app/patient_schema.py
======================
Request/response models for patient history endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any


class SaveVisitRequest(BaseModel):
    patient_id: str = Field(..., description="Doctor-entered patient identifier, e.g. 'P001'")
    prediction: Optional[dict] = Field(
        default=None,
        description="The /predict response for this visit (optional if notes-only update)"
    )
    doctor_notes: Optional[str] = Field(default=None, description="Doctor's observations for this visit")
    prescription: Optional[str] = Field(default=None, description="Doctor's prescription/recommendation")


class UpdateVisitRequest(BaseModel):
    visit_id: int = Field(..., description="ID of the visit record to update")
    doctor_notes: Optional[str] = None
    prescription: Optional[str] = None


class VisitRecord(BaseModel):
    id: int
    patient_id: str
    visit_date: str
    prediction: Optional[dict] = None
    doctor_notes: Optional[str] = None
    prescription: Optional[str] = None


class PatientHistoryResponse(BaseModel):
    patient_id: str
    visit_count: int
    visits: List[VisitRecord]


class PatientListResponse(BaseModel):
    patient_ids: List[str]
