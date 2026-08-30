from sqlalchemy import Column, Integer, String, Text, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
from database import base

class Patient(base):
    __tablename__ = "patients"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    phone_number = Column(String(20), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    dob = Column(String(50), nullable=True)
    gender = Column(String(20), nullable=True)
    password = Column(String(255), nullable=True)
    abha_number = Column(String(50), nullable=True)
    abha_address = Column(String(100), nullable=True)
    known_allergies = Column(JSON, default=list)
    chronic_conditions = Column(JSON, default=list)
    past_surgeries = Column(JSON, default=list)
    created_at = Column(DateTime, default=datetime.utcnow)
    queues = relationship("Queue", back_populates="patient", cascade="all, delete-orphan")
    consultations = relationship("Consultation", back_populates="patient")
    documents = relationship("MedicalDocument", back_populates="patient")

class Doctor(base):
    __tablename__ = "doctors"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    staff_id = Column(String(50), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=False)
    password = Column(String(255), nullable=False)
    department = Column(String(100), default="General Medicine")
    created_at = Column(DateTime, default=datetime.utcnow)
    consultations = relationship("Consultation", back_populates="doctor")

class Queue(base):
    __tablename__ = "queues"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id", ondelete="CASCADE"), nullable=False)
    token_number = Column(Integer, nullable=False)
    status = Column(String(50), default="Waiting")
    consultation_type = Column(String(50), default="FRESH")
    speech_to_text_transcript = Column(Text, nullable=True)
    ai_structured_summary = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    patient = relationship("Patient", back_populates="queues")

class Consultation(base):
    __tablename__ = "consultations"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=True)
    triage_session_id = Column(Integer, nullable=True)
    consultation_type = Column(String(50), default="FRESH")
    ai_structured_summary = Column(JSON, nullable=True)
    prescriptions = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    patient = relationship("Patient", back_populates="consultations")
    doctor = relationship("Doctor", back_populates="consultations")

class Review(base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    doctor_id = Column(Integer, ForeignKey("doctors.id"), nullable=False)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=True)
    reviewer_type = Column(String(50), nullable=False)  # 'PATIENT_TO_DOCTOR' | 'DOCTOR_TO_PATIENT'
    rating = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class Lab(base):
    __tablename__ = "labs"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    staff_id = Column(String(50), unique=True, index=True, nullable=False)
    lab_name = Column(String(255), nullable=False)
    password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class MedicalDocument(base):
    __tablename__ = "medical_documents"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    patient_id = Column(Integer, ForeignKey("patients.id"), nullable=False)
    source = Column(String(20), default="PATIENT_UPLOAD")
    file_path = Column(String(500), nullable=True)
    extracted_data = Column(JSON, default=dict)
    is_doctor_verified = Column(Boolean, default=False)
    doctor_notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    patient = relationship("Patient", back_populates="documents")
