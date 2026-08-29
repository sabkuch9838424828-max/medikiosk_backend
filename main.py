import os
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import schemas
import models
from extractor import extract_medical_report
from database import session, engine, get_db
from gemini import router as ai_router
from typing import Optional

UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()
app.include_router(ai_router)
# Database tables create karne ke liye
models.base.metadata.create_all(bind=engine)


# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/api/patient/check')
def check_patient_existence(mobile: str, db: Session = Depends(get_db)):
    """Check karta hai ki patient already registered hai ya naya."""
    patient = db.query(models.Patient).filter(models.Patient.phone_number == mobile).first()
    return {"exists": patient is not None}


@app.post('/api/patient/login')
def patient_otp_login(payload: schemas.PatientLoginInput, db: Session = Depends(get_db)):
    """OTP verification ke baad patient profile fetch ya create karta hai."""
    patient = db.query(models.Patient).filter(models.Patient.phone_number == payload.mobile).first()

    if not patient:
        patient = models.Patient(
            phone_number=payload.mobile,
            full_name=payload.full_name or "New Patient",
            dob=payload.dob
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

    return {
        "status": "success",
        "patient_id": patient.id,
        "full_name": patient.full_name,
        "abha_linked": bool(patient.abha_number)
    }







@app.post('/api/patient/authenticate')
def patient_password_login(payload: schemas.PatientAuthPasswordInput, db: Session = Depends(get_db)):
    """Mobile number aur password se patient sign-in."""
    patient = db.query(models.Patient).filter(models.Patient.phone_number == payload.mobile).first()

    if not patient:
        # Naya user password ke sath direct register hota hai
        patient = models.Patient(
            phone_number=payload.mobile,
            full_name=payload.full_name or "New Patient",
            dob=payload.dob,
            password=payload.password
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)
    else:
        if patient.password and patient.password != payload.password:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password.")

    return {
        "status": "success",
        "patient_id": patient.id,
        "full_name": patient.full_name,
        "abha_linked": bool(patient.abha_number)
    }
@app.post('/api/patient/link-abha')
def link_abha(payload: schemas.LinkAbhaInput, db: Session = Depends(get_db)):
    """ABHA ID profile mein link aur save karta hai."""
    patient = db.query(models.Patient).filter(models.Patient.id == payload.patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found.")

    patient.abha_number = payload.abha_number
    patient.abha_address = payload.abha_address
    db.commit()
    return {"status": "success", "message": "ABHA ID linked successfully"}

@app.post('/api/patient/{patient_id}/delink-abha')
def delink_abha(patient_id: int, db: Session = Depends(get_db)):
    """ABHA ID delink karta hai."""
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found.")

    patient.abha_number = None
    patient.abha_address = None
    db.commit()
    return {"status": "success", "message": "ABHA ID delinked."}

@app.post('/api/patient/{patient_id}/reset-password')
def reset_patient_password(patient_id: int, payload: schemas.ResetPasswordInput, db: Session = Depends(get_db)):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found.")

    if patient.password and patient.password != payload.current_password:
        raise HTTPException(status_code=400, detail="Current password does not match.")

    patient.password = payload.new_password
    db.commit()
    return {"status": "success", "message": "Password updated successfully."}

@app.post('/api/intake')
def register_intake(payload: schemas.PatientIntakeInput, db: Session = Depends(get_db)):
    """Patient intake submit karta hai aur naya OPD token generate karta hai."""
    patient = None
    if payload.patient_id:
        patient = db.query(models.Patient).filter(models.Patient.id == payload.patient_id).first()

    if not patient:
        patient = db.query(models.Patient).filter(models.Patient.phone_number == payload.phone_number).first()

    if not patient:
        patient = models.Patient(
            phone_number=payload.phone_number,
            full_name=payload.full_name,
            gender=payload.gender,
            known_allergies=payload.allergies,
            chronic_conditions=payload.chronic
        )
        db.add(patient)
        db.commit()
        db.refresh(patient)

    # Dynamic Token Generation
    queue_count = db.query(models.Queue).count()
    token_num = 101 + queue_count

    ai_summary = {
        "primary_complaint": payload.symptoms,
        "history_notes": payload.medical_history,
        "department": payload.department or "General Medicine",
        "urgency": payload.urgency
    }

    # Queue Entry
    new_queue_entry = models.Queue(
        patient_id=patient.id,
        token_number=token_num,
        status="Waiting",
        consultation_type=payload.consultation_type,
        speech_to_text_transcript=payload.full_transcript or payload.symptoms,
        ai_structured_summary=ai_summary
    )
    db.add(new_queue_entry)
    db.commit()
    db.refresh(new_queue_entry)

    return {
        "status": "success",
        "token": token_num,
        "patient_id": patient.id,
        "queue_id": new_queue_entry.id
    }

@app.get('/api/patient/{patient_id}/consultations')
def get_patient_consultations(patient_id: int, db: Session = Depends(get_db)):
    """Follow-up picker ke liye patient ke purane visits fetch karta hai."""
    consultations = db.query(models.Consultation).filter(models.Consultation.patient_id == patient_id).order_by(models.Consultation.created_at.desc()).all()
    return {
        "consultations": [
            {
                "triage_session_id": c.id,
                "ai_structured_summary": c.ai_structured_summary,
                "consultation_type": c.consultation_type,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in consultations
        ]
    }


@app.get('/api/patient/{patient_id}/medical-history')
def get_full_patient_history(patient_id: int, db: Session = Depends(get_db)):
    """Doctor dashboard ke liye patient ka complete historical record."""
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found.")

    documents = db.query(models.MedicalDocument).filter(models.MedicalDocument.patient_id == patient_id).order_by(models.MedicalDocument.created_at.desc()).all()
    consultations = db.query(models.Consultation).filter(models.Consultation.patient_id == patient_id).order_by(models.Consultation.created_at.desc()).all()

    return {
        "patient": {
            "full_name": patient.full_name,
            "gender": patient.gender,
            "dob": patient.dob,
            "phone_number": patient.phone_number,
            "known_allergies": patient.known_allergies or [],
            "chronic_conditions": patient.chronic_conditions or [],
            "past_surgeries": patient.past_surgeries or [],
        },
        "documents": [
            {
                "id": d.id,
                "created_at": d.created_at.isoformat() if d.created_at else None,
                "extracted_data": d.extracted_data,
                "is_doctor_verified": d.is_doctor_verified,
                "doctor_notes": d.doctor_notes,
                # frontend loads this straight into an <img src="..."> for the prescription viewer
                "file_url": f"http://127.0.0.1:8000/api/documents/{d.id}/file" if d.file_path else None,
            }
            for d in documents
        ],
        "consultations": [
            {
                "ai_structured_summary": c.ai_structured_summary,
                "consultation_type": c.consultation_type,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in consultations
        ],
    }


@app.post('/api/doctor/authenticate')
def doctor_login(payload: schemas.DoctorLoginInput, db: Session = Depends(get_db)):
    doctor = db.query(models.Doctor).filter(models.Doctor.staff_id == payload.staff_id).first()
    
    # Auto-seed demo doctor agar database empty ho
    if not doctor and payload.staff_id == "DOC-101" and payload.password == "doctor123":
        doctor = models.Doctor(
            staff_id="DOC-101",
            full_name="Dr. Shubh Sharma",
            password="doctor123",
            department="Cardiology"
        )
        db.add(doctor)
        db.commit()
        db.refresh(doctor)

    if not doctor or doctor.password != payload.password:
        raise HTTPException(status_code=401, detail="Invalid Staff ID or password.")

    return {
        "status": "success",
        "doctor_id": doctor.id,
        "full_name": doctor.full_name,
        "department": doctor.department
    }


@app.post('/api/doctor/{doctor_id}/reset-password')
def reset_doctor_password(doctor_id: int, payload: schemas.ResetPasswordInput, db: Session = Depends(get_db)):
    doctor = db.query(models.Doctor).filter(models.Doctor.id == doctor_id).first()
    if not doctor:
        raise HTTPException(status_code=404, detail="Doctor profile not found.")

    if doctor.password != payload.current_password:
        raise HTTPException(status_code=400, detail="Current password does not match.")

    doctor.password = payload.new_password
    db.commit()
    return {"status": "success", "message": "Password updated."}


@app.get('/api/doctor/queue')
def get_live_queue(db: Session = Depends(get_db)):
    """Live doctor dashboard queue."""
    waiting_items = db.query(models.Queue).filter(models.Queue.status != "Completed").order_by(models.Queue.id.asc()).all()

    queue_list = []
    for q in waiting_items:
        p = q.patient
        queue_list.append({
            "queue_id": q.id,
            "patient_id": q.patient_id,
            "token_number": q.token_number,
            "consultation_type": q.consultation_type,
            "full_name": p.full_name if p else "Unknown",
            "gender": p.gender if p else "—",
            "age_meta": f"Age: 25",
            "speech_to_text_transcript": q.speech_to_text_transcript,
            "ai_structured_summary": q.ai_structured_summary,
            "known_allergies": p.known_allergies if p else [],
            "chronic_conditions": p.chronic_conditions if p else []
        })

    return {"queue": queue_list}


@app.delete('/api/doctor/queue/{queue_id}')
def remove_from_queue(queue_id: int, db: Session = Depends(get_db)):
    queue_item = db.query(models.Queue).filter(models.Queue.id == queue_id).first()
    if not queue_item:
        raise HTTPException(status_code=404, detail="Queue item not found.")

    new_consult = models.Consultation(
        patient_id=queue_item.patient_id,
        doctor_id=1,
        consultation_type=queue_item.consultation_type,
        ai_structured_summary=queue_item.ai_structured_summary,
        prescriptions={"completed": True}
    )
    db.add(new_consult)
    db.delete(queue_item)
    db.commit()

    return {"status": "success", "message": "Patient removed and marked completed."}


@app.post('/api/reviews')
def create_doctor_review(payload: schemas.DoctorReviewInput, db: Session = Depends(get_db)):
    review = models.Review(
        doctor_id=payload.doctor_id,
        patient_id=payload.patient_id,
        reviewer_type="PATIENT_TO_DOCTOR",
        rating=payload.rating,
        comment=payload.comment
    )
    db.add(review)
    db.commit()
    return {"status": "success", "message": "Doctor review submitted."}


@app.post('/api/patient-reviews')
def create_patient_review(payload: schemas.PatientReviewInput, db: Session = Depends(get_db)):
    review = models.Review(
        doctor_id=payload.doctor_id,
        patient_id=payload.patient_id,
        reviewer_type="DOCTOR_TO_PATIENT",
        rating=payload.rating,
        comment=payload.comment
    )
    db.add(review)
    db.commit()
    return {"status": "success", "message": "Internal patient review saved."}








# @app.post('/api/patient_details')
# def login_input(payload: schemas.patient_details, db: Session = Depends(get_db)):





# parsed_json = extractor.extract_medical_report()
# def save_cbc_record(db, patient_id: int, parsed_json: dict):
#     cbc_entry = models.cbc_reports(
#         patient_id=patient_id,
#         report_date=parsed_json.get("report_date"),
#         lab_name=parsed_json.get("lab_name"),
#         hemoglobin=parsed_json.get("hemoglobin"),
#         rbc_count=parsed_json.get("rbc_count"),
#         hematocrit_pcv=parsed_json.get("hematocrit_pcv"),
#         mcv=parsed_json.get("mcv"),
#         mch=parsed_json.get("mch"),
#         mchc=parsed_json.get("mchc"),
#         rdw=parsed_json.get("rdw"),
#         wbc_count=parsed_json.get("wbc_count"),
#         neutrophils=parsed_json.get("neutrophils"),
#         lymphocytes=parsed_json.get("lymphocytes"),
#         monocytes=parsed_json.get("monocytes"),
#         eosinophils=parsed_json.get("eosinophils"),
#         basophils=parsed_json.get("basophils"),
#         platelet_count=parsed_json.get("platelet_count"),
#         mpv=parsed_json.get("mpv"),
#         is_abnormal=bool(parsed_json.get("abnormal_flags")),
#         abnormal_summary=", ".join(parsed_json.get("abnormal_flags", []))
#     )
#     db.add(cbc_entry)
#     db.commit()
#     db.refresh(cbc_entry)
#     return cbc_entry

def _merge_into_permanent_history(patient: "models.Patient", extracted_json: dict) -> None:
    """
    OCR se mile allergies/chronic conditions ko patient ke PERMANENT record
    (known_allergies, chronic_conditions) me merge karta hai — duplicate check
    ke saath (case-insensitive), taaki wahi cheez baar baar na jud jaye.
    """
    existing_allergies = list(patient.known_allergies or [])
    existing_allergies_lower = {a.lower() for a in existing_allergies}
    for new_allergy in extracted_json.get("extracted_allergies", []) or []:
        if new_allergy and new_allergy.lower() not in existing_allergies_lower:
            existing_allergies.append(new_allergy)
            existing_allergies_lower.add(new_allergy.lower())
    patient.known_allergies = existing_allergies

    existing_chronic = list(patient.chronic_conditions or [])
    existing_chronic_lower = {c.lower() for c in existing_chronic}
    for new_condition in extracted_json.get("extracted_chronic_conditions", []) or []:
        if new_condition and new_condition.lower() not in existing_chronic_lower:
            existing_chronic.append(new_condition)
            existing_chronic_lower.add(new_condition.lower())
    patient.chronic_conditions = existing_chronic


@app.post("/api/ocr/upload")
async def upload_prescription_gemini(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found.")

    # --- 1. File ko disk par save karo (baad me "View Original Prescription
    #        Image" modal me isi file ko serve kiya jayega) ---
    contents = await file.read()
    safe_filename = f"{patient_id}_{file.filename}".replace(" ", "_")
    saved_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(saved_path, "wb") as f:
        f.write(contents)

    # --- 2. Gemini se extraction (extractor.py ka function reuse) ---
    try:
        extracted_json = extract_medical_report(saved_path)
        if not extracted_json:
            raise ValueError("Empty extraction result from Gemini.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini OCR processing failed: {str(e)}")

    # --- 3. Allergies / chronic conditions ko PERMANENT history me merge karo ---
    _merge_into_permanent_history(patient, extracted_json)

    # --- 4. Document record save karo (file path + extracted JSON dono) ---
    doc_record = models.MedicalDocument(
        patient_id=patient_id,
        file_path=saved_path,
        extracted_data=extracted_json,
        is_doctor_verified=False,
        doctor_notes=extracted_json.get("clinical_summary", "Extracted via MediKiosk AI Vision Scanner")
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)
    db.refresh(patient)

    return {
        "status": "success",
        "document_id": doc_record.id,
        "extracted_data": extracted_json,
        "file_url": f"http://127.0.0.1:8000/api/documents/{doc_record.id}/file",
        # updated permanent record — frontend turant UI refresh kar sakta hai
        "known_allergies": patient.known_allergies,
        "chronic_conditions": patient.chronic_conditions,
    }


@app.get("/api/documents/{document_id}/file")
def get_document_file(document_id: int, db: Session = Depends(get_db)):
    """Prescription-view modal ke liye asli scanned image serve karta hai."""
    doc = db.query(models.MedicalDocument).filter(models.MedicalDocument.id == document_id).first()
    if not doc or not doc.file_path or not os.path.exists(doc.file_path):
        raise HTTPException(status_code=404, detail="Document file not found.")
    return FileResponse(doc.file_path)


# ==========================================
# DOCUMENT VERIFICATION (doctor verifies OCR-scanned past history)
# ==========================================

@app.post('/api/documents/{document_id}/verify')
def verify_document(document_id: int, payload: schemas.DocumentVerifyInput, db: Session = Depends(get_db)):
    """Doctor ye confirm karta hai ki OCR se extract hua data sahi hai."""
    doc = db.query(models.MedicalDocument).filter(models.MedicalDocument.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found.")

    doc.is_doctor_verified = payload.is_verified
    if payload.doctor_notes is not None:
        doc.doctor_notes = payload.doctor_notes

    db.commit()
    return {
        "status": "success",
        "document_id": doc.id,
        "is_doctor_verified": doc.is_doctor_verified,
        "doctor_notes": doc.doctor_notes,
    }


# ==========================================
# COMPLETE CONSULTATION (real prescribed medicines save karta hai)
# ==========================================

@app.post('/api/doctor/queue/{queue_id}/complete')
def complete_consultation_with_prescription(
    queue_id: int, payload: schemas.CompleteConsultationInput, db: Session = Depends(get_db)
):
    """Doctor 'Complete Consultation' dabata hai — actual diagnosis + medicines save hote hain."""
    queue_item = db.query(models.Queue).filter(models.Queue.id == queue_id).first()
    if not queue_item:
        raise HTTPException(status_code=404, detail="Queue item not found.")

    new_consult = models.Consultation(
        patient_id=queue_item.patient_id,
        doctor_id=1,
        consultation_type=queue_item.consultation_type,
        ai_structured_summary=queue_item.ai_structured_summary,
        prescriptions={
            "diagnosis": payload.diagnosis,
            "medicines": [m.dict() for m in payload.medicines],
        },
    )
    db.add(new_consult)
    db.delete(queue_item)
    db.commit()

    return {"status": "success", "message": "Consultation completed and prescription saved."}


# ==========================================
# PATIENT — apni prescriptions dekhne ke liye
# ==========================================

@app.get('/api/patient/{patient_id}/prescriptions')
def get_patient_prescriptions(patient_id: int, db: Session = Depends(get_db)):
    """Patient login karke apni saari purani prescriptions dekh sakta hai."""
    consultations = (
        db.query(models.Consultation)
        .filter(
            models.Consultation.patient_id == patient_id,
            models.Consultation.prescriptions.isnot(None),
        )
        .order_by(models.Consultation.created_at.desc())
        .all()
    )
    return {
        "prescriptions": [
            {
                "created_at": c.created_at.isoformat() if c.created_at else None,
                "diagnosis": (c.prescriptions or {}).get("diagnosis"),
                "medicines": (c.prescriptions or {}).get("medicines", []),
            }
            for c in consultations
        ]
    }


# ==========================================
# LAB STAFF AUTH + REPORT PUSH
# ==========================================

@app.post('/api/lab/authenticate')
def lab_login(payload: schemas.LabLoginInput, db: Session = Depends(get_db)):
    lab = db.query(models.Lab).filter(models.Lab.staff_id == payload.staff_id).first()

    # Demo lab auto-seed (Doctor login jaisa hi pattern)
    if not lab and payload.staff_id == "LAB-101" and payload.password == "lab123":
        lab = models.Lab(staff_id="LAB-101", lab_name="AIIMS Central Diagnostic Lab", password="lab123")
        db.add(lab)
        db.commit()
        db.refresh(lab)

    if not lab or lab.password != payload.password:
        raise HTTPException(status_code=401, detail="Invalid Lab Staff ID or password.")

    return {"status": "success", "lab_id": lab.id, "lab_name": lab.lab_name}


@app.get('/api/lab/patient-lookup')
def lab_patient_lookup(mobile: Optional[str] = None, abha: Optional[str] = None, db: Session = Depends(get_db)):
    """Lab technician patient ko dhoondhta hai — mobile number ya ABHA ID se."""
    if not mobile and not abha:
        raise HTTPException(status_code=400, detail="Provide mobile number or ABHA ID to search.")

    query = db.query(models.Patient)
    patient = (
        query.filter(models.Patient.phone_number == mobile).first() if mobile
        else query.filter(
            (models.Patient.abha_number == abha) | (models.Patient.abha_address == abha)
        ).first()
    )

    if not patient:
        raise HTTPException(status_code=404, detail="No patient found with this mobile number / ABHA ID.")

    return {
        "patient_id": patient.id,
        "full_name": patient.full_name,
        "phone_number": patient.phone_number,
        "abha_number": patient.abha_number,
    }


@app.post('/api/lab/upload-report')
async def lab_upload_report(
    patient_id: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Lab report ko seedha patient ki ABHA-linked history me push karta hai."""
    patient = db.query(models.Patient).filter(models.Patient.id == patient_id).first()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient profile not found.")

    contents = await file.read()
    safe_filename = f"lab_{patient_id}_{file.filename}".replace(" ", "_")
    saved_path = os.path.join(UPLOAD_DIR, safe_filename)
    with open(saved_path, "wb") as f:
        f.write(contents)

    try:
        extracted_json = extract_medical_report(saved_path)
        if not extracted_json:
            raise ValueError("Empty extraction result from Gemini.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini OCR processing failed: {str(e)}")

    # Wahi merge-into-permanent-history function reuse karo jo OCR upload ke liye pehle banaya tha
    _merge_into_permanent_history(patient, extracted_json)

    doc_record = models.MedicalDocument(
        patient_id=patient_id,
        source="LAB",
        file_path=saved_path,
        extracted_data=extracted_json,
        is_doctor_verified=False,
        doctor_notes=extracted_json.get("clinical_summary", "Report uploaded by diagnostic lab"),
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)
    db.refresh(patient)

    return {
        "status": "success",
        "document_id": doc_record.id,
        "extracted_data": extracted_json,
        "known_allergies": patient.known_allergies,
        "chronic_conditions": patient.chronic_conditions,
    }