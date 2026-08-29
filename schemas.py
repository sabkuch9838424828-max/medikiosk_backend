from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict

class PatientLoginInput(BaseModel):
    """Used in /api/patient/login (OTP flow)"""
    mobile: str
    full_name: Optional[str] = None
    dob: Optional[str] = None


class PatientAuthPasswordInput(BaseModel):
    """Used in /api/patient/authenticate (Password flow)"""
    mobile: str
    password: str
    full_name: Optional[str] = None
    dob: Optional[str] = None


class LinkAbhaInput(BaseModel):
    """Used in /api/patient/link-abha"""
    patient_id: int
    abha_number: str
    abha_address: str


class ResetPasswordInput(BaseModel):
    """Used in /api/patient/{id}/reset-password & /api/doctor/{id}/reset-password"""
    current_password: str
    new_password: str


# ==========================================
# 2. DOCTOR AUTHENTICATION SCHEMAS
# ==========================================

class DoctorLoginInput(BaseModel):
    """Used in /api/doctor/authenticate"""
    staff_id: str
    password: str


# ==========================================
# 3. PATIENT INTAKE & TRIAGE SCHEMAS
# ==========================================

class PatientIntakeInput(BaseModel):
    """Used in /api/intake to generate token and queue entry"""
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
    consultation_type: str = "FRESH"  # 'FRESH' | 'FOLLOWUP'
    follow_up_of: Optional[int] = None


# ==========================================
# 4. REVIEWS & RATINGS SCHEMAS
# ==========================================

class DoctorReviewInput(BaseModel):
    """Used in /api/reviews (Patient reviewing Doctor)"""
    doctor_id: int
    patient_id: Optional[int] = None
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class PatientReviewInput(BaseModel):
    """Used in /api/patient-reviews (Doctor reviewing Patient)"""
    doctor_id: int
    patient_id: int
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


# ==========================================
# 5. AI CHAT & TRIAGE SCHEMAS
# ==========================================
    
class ChatTurn(BaseModel):
    role: str          # "user" | "model"
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
    """Used in /api/documents/{document_id}/verify"""
    is_verified: bool
    doctor_notes: Optional[str] = None


class PrescriptionMedicine(BaseModel):
    name: str
    dose: str
    frequency: str
    duration: str


class CompleteConsultationInput(BaseModel):
    """Used in /api/doctor/queue/{queue_id}/complete"""
    diagnosis: Optional[str] = None
    medicines: List[PrescriptionMedicine] = []

class LabLoginInput(BaseModel):
    staff_id: str
    password: str