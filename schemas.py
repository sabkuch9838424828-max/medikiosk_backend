from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict

class PatientLoginInput(BaseModel):
    mobile: str
    full_name: Optional[str] = None
    dob: Optional[str] = None


class PatientAuthPasswordInput(BaseModel):
    mobile: str
    password: str
    full_name: Optional[str] = None
    dob: Optional[str] = None


class LinkAbhaInput(BaseModel):
    patient_id: int
    abha_number: str
    abha_address: str


class ResetPasswordInput(BaseModel):
    current_password: str
    new_password: str

class DoctorLoginInput(BaseModel):
    staff_id: str
    password: str


class PatientIntakeInput(BaseModel):
    full_name: str
    phone_number: str
    gender: str
    age: int
    symptoms: str
    medical_history: Optional[str] = ""
    full_transcript: Optional[str] = ""
    department: Optional[str] = None
    urgency: str = "ROUTINE"
    priority_level: int = 3
    allergies: List[str] = []
    chronic: List[str] = []
    patient_id: Optional[int] = None
    consultation_type: str = "FRESH" 
    follow_up_of: Optional[int] = None

class DoctorReviewInput(BaseModel):
    doctor_id: int
    patient_id: Optional[int] = None
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class PatientReviewInput(BaseModel):
    doctor_id: int
    patient_id: int
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    
class ChatTurn(BaseModel):
    role: str 
    text: str

class ChatRequest(BaseModel):
    message: str
    language: str
    history: List[ChatTurn] = []

class ChatResponse(BaseModel):
    reply: str
    symptom_type: Optional[str] = None
    symptom_summary: Optional[str] = None
    history_summary: Optional[str] = None
    ready_for_triage: bool = False
    matched_department: Optional[str] = None
    doctor_name: Optional[str] = None
    urgency: Optional[str] = None
    priority_level: Optional[int] = None


class DocumentVerifyInput(BaseModel):
    is_verified: bool
    doctor_notes: Optional[str] = None


class PrescriptionMedicine(BaseModel):
    name: str
    dose: str
    frequency: str
    duration: str


class CompleteConsultationInput(BaseModel):
    diagnosis: Optional[str] = None
    medicines: List[PrescriptionMedicine] = []

class LabLoginInput(BaseModel):
    staff_id: str
    password: str
