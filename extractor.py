import os
import json
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# API key ab .env se aayegi — kabhi bhi key seedhe code me mat likho.
# .env file me: GEMINI_API_KEY=your_actual_key_here
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set — add it to your .env file.")
genai.configure(api_key=API_KEY)

# Generation config to force strict JSON output
generation_config = {
    "response_mime_type": "application/json",
    "temperature": 0.1
}

# NOTE: "gemini-3.6-flash" is not a real/valid model name — Google does not
# use that versioning scheme. Using "gemini-2.5-flash": stable, fast, cheap,
# and supports image (vision) input — exactly what OCR extraction needs.
model = genai.GenerativeModel(
    model_name="gemini-3.6-flash",
    generation_config=generation_config
)

PROMPT = """
You are an expert clinical data extraction engine for MediKiosk.
Analyze the provided medical document (prescription, lab report, radiology scan, or discharge note) and output strict JSON matching this schema:

{
  "report_type": "Specific type (e.g., Complete Blood Count, Lipid Profile, Chest X-Ray, General Prescription)",
  "report_date": "YYYY-MM-DD or string date if available",
  "doctor_or_facility": "Doctor or Lab/Hospital name if available",
  "clinical_summary": "1-2 sentence concise clinical takeaway for the physician dashboard",
  "blood_group": "Blood group if mentioned anywhere on the document (e.g. 'O+'), else null",
  "extracted_allergies": ["Any allergies explicitly mentioned in the document, e.g. 'Penicillin' — empty list if none mentioned"],
  "extracted_chronic_conditions": ["Any chronic/long-term conditions explicitly mentioned, e.g. 'Type 2 Diabetes', 'Hypertension' — empty list if none mentioned"],
  "dynamic_data": {
    "abnormal_flags": ["List of critical/abnormal findings (e.g., 'Low Hemoglobin', 'High Triglycerides')"],
    "test_metrics": [
      {
        "name": "Parameter name (e.g. Hemoglobin, Total Cholesterol, SpO2)",
        "value": "Value as number/string",
        "unit": "Measurement unit (e.g. g/dL, mg/dL)",
        "reference_range": "Normal range (e.g. 13.0 - 17.0)",
        "status": "NORMAL | HIGH | LOW | ABNORMAL"
      }
    ],
    "medications": [
      {
        "name": "Medicine name",
        "dosage": "e.g. 500mg",
        "frequency": "e.g. TDS / 1-0-1",
        "duration": "e.g. 5 days"
      }
    ],
    "radiology_findings": "Detailed impressions/findings if document is an image/scan",
    "additional_notes": "Any other critical diagnostic notes"
  }
}
"""

def extract_medical_report(image_path: str) -> dict:
    try:
        img = Image.open(image_path)
        response = model.generate_content([PROMPT, img])
        
        # Clean response if markdown fences exist
        cleaned_text = response.text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
            
        return json.loads(cleaned_text.strip())
    except Exception as e:
        print(f"Extraction failed: {e}")
        return {}