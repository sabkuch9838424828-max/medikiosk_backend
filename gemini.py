from fastapi import FastAPI, HTTPException, UploadFile, File, Form, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import os
import logging
import traceback
import threading
import time
import asyncio
from google import genai
from google.genai import types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medikiosk")

router = APIRouter()
GEMINI_MODEL = "gemini-3.6-flash"

GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv(
        "GEMINI_FALLBACK_MODELS",
        "gemini-3.7-flash,gemini-3.5-flash,gemini-3.1-flash-lite,"
        "gemini-2.5-pro,gemini-2.5-flash,gemini-2.5-flash-lite"
    ).split(",") if m.strip()
]

KEY_COOLDOWN_SECONDS = 65

def _load_gemini_keys() -> list:
    raw = os.getenv("GEMINI_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        single = os.getenv("GEMINI_API_KEY")
        if single:
            keys = [single.strip()]
    return keys

class GeminiKeyManager:
    def __init__(self, keys: list, cooldown_seconds: int = KEY_COOLDOWN_SECONDS):
        if not keys:
            raise RuntimeError(
                "No Gemini API keys configured. Set GEMINI_API_KEYS as a comma-separated "
                "list (or GEMINI_API_KEY for a single key)."
            )
        self._keys = keys
        self._cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._cooldown_until = {k: 0.0 for k in self._keys}
        self._next_start = 0
        self._clients = {k: genai.Client(api_key=k) for k in self._keys}
        logger.info("Gemini key pool initialized with %d key(s).", len(self._keys))

    def _label(self, key: str) -> str:
        return f"...{key[-4:]}" if len(key) > 4 else "key"

    def keys_in_order(self) -> list:
        with self._lock:
            n = len(self._keys)
            start = self._next_start
            self._next_start = (self._next_start + 1) % n
            order = [self._keys[(start + i) % n] for i in range(n)]
            now = time.monotonic()
            ready = [k for k in order if self._cooldown_until[k] <= now]
            cooling = sorted(
                (k for k in order if self._cooldown_until[k] > now),
                key=lambda k: self._cooldown_until[k],
            )
        return ready + cooling  # cooling keys last, soonest-available first

    def mark_cooldown(self, key: str):
        with self._lock:
            self._cooldown_until[key] = time.monotonic() + self._cooldown_seconds
        logger.warning("Gemini key %s hit a rate limit; cooling down for %ss.", self._label(key), self._cooldown_seconds)

    def client_for(self, key: str) -> genai.Client:
        return self._clients[key]

key_manager = GeminiKeyManager(_load_gemini_keys())

TRANSIENT_STATUS_CODES = {500, 503, 504}
TRANSIENT_MESSAGE_HINTS = ("503", "unavailable", "500", "internal error", "504", "deadline_exceeded", "overloaded")

def _classify_gemini_error(exc: Exception) -> str:
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    message = str(exc).lower()

    if status_code == 429 or "429" in message or "resource_exhausted" in message or "rate limit" in message:
        return "rate_limit"
    if status_code in TRANSIENT_STATUS_CODES or any(hint in message for hint in TRANSIENT_MESSAGE_HINTS):
        return "transient"
    return "fatal"

def _generate_across_keys(model: str, max_rounds: int, base_backoff_seconds: float, **generate_kwargs):    
    last_error = None
    for round_num in range(1, max_rounds + 1):
        for key in key_manager.keys_in_order():
            client_for_key = key_manager.client_for(key)
            try:
                return client_for_key.models.generate_content(model=model, **generate_kwargs)
            except Exception as e:
                kind = _classify_gemini_error(e)
                last_error = e
                if kind == "rate_limit":
                    key_manager.mark_cooldown(key)
                    logger.warning("Gemini key %s rate-limited on %s (round %d/%d); trying next key.",
                                    key_manager._label(key), model, round_num, max_rounds)
                    continue
                elif kind == "transient":
                    logger.warning("Gemini key %s hit a transient error on %s (round %d/%d): %s; trying next key.",
                                    key_manager._label(key), model, round_num, max_rounds, e)
                    continue
                raise  
        if round_num < max_rounds:
            backoff = base_backoff_seconds * (2 ** (round_num - 1))
            logger.warning("All %d Gemini keys failed round %d/%d on %s; backing off %.1fs before next round.",
                            len(key_manager._keys), round_num, max_rounds, model, backoff)
            time.sleep(backoff)
    raise last_error

def generate_with_key_pool(model: str = GEMINI_MODEL, max_rounds: int = 2, base_backoff_seconds: float = 1.5,
                            **generate_kwargs):
    
    models_to_try = [model] + [m for m in GEMINI_FALLBACK_MODELS if m != model]
    last_error = None
    for attempt_model in models_to_try:
        try:
            response = _generate_across_keys(attempt_model, max_rounds, base_backoff_seconds, **generate_kwargs)
            if attempt_model != model:
                logger.warning("Served request with fallback model %s (primary %s was unavailable).",
                                attempt_model, model)
            return response
        except Exception as e:
            last_error = e
            logger.warning("Model %s exhausted across the whole key pool; trying next model.", attempt_model)
            continue
    raise HTTPException(
        status_code=503,
        detail="Gemini is currently rate-limited or overloaded across all configured keys and models. Please try again shortly."
    ) from last_error


class ChatTurn(BaseModel):
    role: str  
    text: str

class AIChatQuery(BaseModel):
    message: str
    language: Optional[str] = "Hindi"
    history: Optional[List[ChatTurn]] = [] 



TRIAGE_SYSTEM_INSTRUCTION = """
You are an advanced AI clinical intake and triage assistant for a multi-specialty hospital kiosk.

Each request includes the full prior conversation (as alternating user/model turns) plus the
patient's newest message. Read the ENTIRE conversation before deciding what to do — you must
never ask the patient for information they already gave earlier in the conversation. If you
find yourself about to ask something already covered above, stop and move to the next missing
piece of information (or finalize, if nothing is missing) instead.

You conduct a short, multi-turn intake conversation before recommending a doctor. Do NOT match
a department/doctor on the very first message — gather information first.

Your responsibilities:
1. Systematically prompt and guide the patient to provide, across multiple turns:
   a) Their current symptoms / reason for today's visit
   b) Past medical history or chronic conditions (e.g. diabetes, asthma, prior surgeries)
   c) Known drug or food allergies
   Ask ONE clear follow-up question at a time in the "reply" field, and ONLY about whichever of
   (a)/(b)/(c) is still genuinely missing from the conversation so far (a brief "no allergies" /
   "no history" counts as covered). Never repeat a question the patient has already answered,
   even if they answered it several turns ago or phrased it differently than expected.
2. Accurately transcribe audio or text input (supporting English, Hindi, or Hinglish).
3. Classify EACH patient message as either:
   - "CURRENT_SYMPTOM": what the patient is here for today (e.g. "I have a headache", "chest pain since morning")
   - "MEDICAL_HISTORY": pre-existing/background conditions, allergies, or past diagnoses the patient mentions
     (e.g. "I have asthma", "I'm allergic to dust", "I had surgery 2 years ago")
   If a single message contains both, classify it by whichever is the primary content of that message.
4. Maintain two running, cumulative clinical summaries built from the ENTIRE conversation so far
   (not just the latest message), always written in clear, professional, concise clinical English
   regardless of what language the patient is speaking — these go directly into the doctor's chart:
   - "symptom_summary": current visit's symptoms as a short clinical phrase, e.g.
     "Headache, nasal congestion, and abdominal pain, onset this morning."
   - "history_summary": relevant past history and allergies as a short clinical phrase, e.g.
     "History of hypertension, type 2 diabetes, and asthma. Reports dust allergy."
   Update both of these on every turn to reflect everything gathered so far, deduplicated and
   condensed — never a raw copy-paste of the patient's own wording, and never just the latest
   message alone.
5. Set "ready_for_triage":
   - false, while you are still actively gathering (a), (b), or (c) above — in this case you may
     leave "department"/"doctor"/"room"/"urgency"/"priority_level" as your best current guess, but
     the frontend will NOT act on them yet, so focus "reply" on asking the next genuinely-missing
     intake question.
   - true, once you have gathered enough about symptoms, history, and allergies to safely
     recommend a department. At that point, determine the correct hospital department and doctor:
     - Cardiology -> Dr. Amit Sharma (Cardiology - Room 102) [Priority Level 1 if chest pain/emergency]
     - Orthopedics -> Dr. Rajesh Nair (Orthopedics - Room 305) [Priority Level 2]
     - Gastroenterology -> Dr. Neha Gupta (Gastroenterology - Room 401) [Priority Level 2]
     - General Medicine -> Dr. Priya Varma (General Medicine - Room 204) [Priority Level 3]
   - true immediately, without delay, if EITHER: (i) the patient describes a clear emergency
     (e.g. severe chest pain, difficulty breathing), or (ii) the patient signals frustration or
     repetition — e.g. "I already told you", "I said this already", "just refer me" — in which
     case finalize using whatever information has been gathered so far rather than asking again.
6. Provide a compassionate response back to the patient in their preferred language.

You MUST respond strictly in valid JSON format using these exact keys:
{
  "transcript": "The cleaned text or speech transcription",
  "symptom_type": "CURRENT_SYMPTOM" or "MEDICAL_HISTORY",
  "reply": "Your conversational response — a follow-up intake question, or the final triage summary once ready",
  "ready_for_triage": true or false,
  "symptom_summary": "Cumulative professional clinical summary of current symptoms, in English",
  "history_summary": "Cumulative professional clinical summary of past history/allergies, in English",
  "department": "Cardiology or Orthopedics or Gastroenterology or General Medicine",
  "doctor": "Dr. Amit Sharma (Cardiology - Room 102)",
  "room": "Room 102",
  "urgency": "EMERGENCY" or "URGENT" or "ROUTINE",
  "priority_level": 1 or 2 or 3
}
"""


def extract_gemini_text(response) -> str:

    candidates = getattr(response, "candidates", None)
    if not candidates:
        raise ValueError("Gemini returned no candidates (empty or fully blocked response).")

    candidate = candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)

    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) if content else None

    if not parts:
        
        raise ValueError(
            f"Gemini returned no usable text (finish_reason={finish_reason}). "
            f"This usually means the audio was blocked, empty, or unintelligible."
        )

    text = "".join(getattr(p, "text", "") or "" for p in parts)
    if not text.strip():
        raise ValueError(f"Gemini returned an empty response (finish_reason={finish_reason}).")

    return text


def parse_gemini_json(response):
    raw_text = extract_gemini_text(response)

    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    try:
        data = json.loads(cleaned.strip())
    except json.JSONDecodeError as e:
        logger.error("Gemini did not return valid JSON. Raw text was:\n%s", raw_text)
        raise ValueError(f"Gemini response was not valid JSON: {e}")

    return {
        "transcript": data.get("transcript", "Audio processed successfully."),
        "symptom_type": data.get("symptom_type", "CURRENT_SYMPTOM"),
        "reply": data.get("reply", "Symptoms noted."),
        "ready_for_triage": bool(data.get("ready_for_triage", False)),
        "symptom_summary": data.get("symptom_summary", ""),
        "history_summary": data.get("history_summary", ""),
        "matched_department": data.get("department", "General Medicine"),
        "doctor_name": data.get("doctor", "Dr. Priya Varma (General Medicine - Room 204)"),
        "room": data.get("room", "Room 204"),
        "urgency": data.get("urgency", "ROUTINE"),
        "priority_level": data.get("priority_level", 3)
    }

def build_history_contents(history):
    contents = []
    for turn in history or []:
        role = turn.get("role") if isinstance(turn, dict) else getattr(turn, "role", None)
        text = turn.get("text") if isinstance(turn, dict) else getattr(turn, "text", None)
        if not text:
            continue
        role = "model" if role == "model" else "user"
        contents.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
    return contents

@router.post("/api/ai/chat")
def multilingual_ai_triage(data: AIChatQuery):
    history_contents = build_history_contents(data.history)
    latest_content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=f"Patient Message: '{data.message}'\nPreferred Language: {data.language}")]
    )
    try:
        response = generate_with_key_pool(
            model=GEMINI_MODEL,
            contents=history_contents + [latest_content],
            config=types.GenerateContentConfig(
                system_instruction=TRIAGE_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return parse_gemini_json(response)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Text chat failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Gemini API Error: {str(e)}")

@router.post("/api/ai/voice-chat")
async def process_voice_chat(file: UploadFile = File(...), language: str = Form(...), history: str = Form(default="[]")):
    try:
        audio_bytes = await file.read()
        prompt = f"Listen to this patient audio recording. Accurately transcribe speech, extract current medical issues, past history, and allergies. Preferred language for reply: {language}"

        if not audio_bytes:
            raise ValueError("Received an empty audio file from the browser (0 bytes).")

        mime_type = file.content_type or "audio/wav"
        logger.info("Voice upload: %d bytes, content_type=%s", len(audio_bytes), mime_type)

        try:
            history_list = json.loads(history) if history else []
        except json.JSONDecodeError:
            history_list = []

        history_contents = build_history_contents(history_list)
        latest_content = types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                types.Part.from_text(text=prompt),
            ]
        )
        response = await asyncio.to_thread(
            generate_with_key_pool,
            model=GEMINI_MODEL,
            contents=history_contents + [latest_content],
            config=types.GenerateContentConfig(
                system_instruction=TRIAGE_SYSTEM_INSTRUCTION,
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )

        return parse_gemini_json(response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Voice chat failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Gemini Voice Processing Error: {str(e)}")
