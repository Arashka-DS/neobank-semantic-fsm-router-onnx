import os
import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Response, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from src.normalizer import PersianFinancialNormalizer
from src.state_machine import DialogFSM

app = FastAPI(title="Neobank Semantic Router & Dialog FSM", version="1.1.0")

normalizer = PersianFinancialNormalizer()
fsm = DialogFSM()

MODEL_PATH = "models/router_int8.onnx"
TOKENIZER_DIR = "models/tokenizer"

tokenizer = None
ort_session = None

def init_runtime():
    global tokenizer, ort_session
    if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_DIR):
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 2
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        ort_session = ort.InferenceSession(MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])

init_runtime()

INTENT_MAP = {0: "TRANSFER", 1: "BALANCE_INQUIRY", 2: "BILL_PAYMENT", 3: "OOD"}

class CommandPayload(BaseModel):
    session_id: str = Field(..., example="sess_usr_99182")
    text: str = Field(..., example="پونصد هزار تومن به علی کارت به کارت کن")

def compute_energy_score(logits: np.ndarray, temperature: float = 1.0) -> float:
    shifted = logits / temperature
    max_logit = np.max(shifted)
    logsumexp = max_logit + np.log(np.sum(np.exp(shifted - max_logit)))
    return float(-temperature * logsumexp)

@app.post("/route-command")
def route_command(payload: CommandPayload, response: Response):
    global ort_session, tokenizer
    if ort_session is None:
        init_runtime()
        if ort_session is None:
            raise HTTPException(status_code=503, detail="Model compiling. Retry shortly.")

    start_time = time.perf_counter()
    clean_text = normalizer.normalize(payload.text)
    
    # 1. Tokenize & ONNX INT8 Inference
    encoded = tokenizer(clean_text, return_tensors="np", truncation=True, max_length=32, padding=False)
    ort_inputs = {
        "input_ids": encoded["input_ids"].astype(np.int64),
        "attention_mask": encoded["attention_mask"].astype(np.int64)
    }
    intent_logits, _ = ort_session.run(None, ort_inputs)
    
    # 2. Thermodynamic OOD Gating
    raw_intent_logits = intent_logits[0]
    energy_score = compute_energy_score(raw_intent_logits, temperature=1.0)
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
    
    response.headers["X-Inference-Time-MS"] = str(latency_ms)
    response.headers["X-Energy-OOD-Score"] = f"{energy_score:.4f}"
    
    if energy_score > -2.5:
        return {
            "status": "REJECTED_OOD",
            "message": "Query outside banking capability domain.",
            "energy_score": energy_score,
            "session_id": payload.session_id,
            "latency_ms": latency_ms
        }

    # 3. Intent & Canonical Slot Extraction
    intent_id = int(np.argmax(raw_intent_logits))
    predicted_intent = INTENT_MAP.get(intent_id, "TRANSFER")
    
    extracted_slots = {}
    
    # Canonical Toman/Rial Extraction
    amount_obj = normalizer.parse_financial_amount(clean_text)
    if amount_obj:
        extracted_slots["AMOUNT"] = amount_obj
        
    for name in ["علی", "رضا", "مریم", "سارا", "محمد", "پدر", "مادر"]:
        if name in clean_text:
            extracted_slots["RECIPIENT"] = name
            break
            
    # 4. FSM Multi-Turn Step
    updated_state = fsm.step(payload.session_id, predicted_intent, extracted_slots)
    
    return {
        "session_id": payload.session_id,
        "normalized_query": clean_text,
        "intent": predicted_intent,
        "energy_score": energy_score,
        "extracted_slots": extracted_slots,
        "dialogue_state": updated_state,
        "latency_ms": latency_ms
    }

@app.get("/health")
def health():
    return {"status": "ACTIVE", "model_loaded": ort_session is not None}
