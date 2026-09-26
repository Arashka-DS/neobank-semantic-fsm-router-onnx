import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from src.normalizer import PersianFinancialNormalizer
from src.state_machine import DialogStateTracker
from src.data_generator import SLOT_LABELS, INTENT_MAP

app = FastAPI(title="Neobank Semantic Router & Dialog FSM", version="2.0.0")

REV_INTENT_MAP = {v: k for k, v in INTENT_MAP.items()}
MODEL_PATH = "models/semantic_router_int8.onnx"
TOKENIZER_NAME = "distilbert-base-multilingual-cased"

normalizer = PersianFinancialNormalizer()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
dialog_manager = DialogStateTracker()

class CommandRequest(BaseModel):
    session_id: str = Field(..., example="SESSION-5992")
    user_query: str = Field(..., example="می‌خوام به علی پول بفرستم")

def calculate_energy_score(logits: np.ndarray, temperature: float = 1.0) -> float:
    """Calculates Energy Score for Out-Of-Distribution (OOD) detection."""
    # Negative log-sum-exp of scaled logits
    return -temperature * np.log(np.sum(np.exp(logits / temperature)))

@app.post("/route-command")
def parse_and_route_command(request: CommandRequest, response: Response):
    start_time = time.perf_counter()
    
    cleaned_query = normalizer.normalize(request.user_query)
    inputs = tokenizer(cleaned_query, padding=True, truncation=True, max_length=64, return_tensors="np")
    
    # ONNX Inference
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64)
    }
    intent_logits, slot_logits = ort_session.run(None, ort_inputs)

    # 1. Energy-Based OOD & Ambiguity Rejection
    energy = calculate_energy_score(intent_logits[0])
    ENERGY_THRESHOLD = -5.0 # Tuned hyperparameter
    
    if energy > ENERGY_THRESHOLD:
        return {
            "status": "REJECTED_OOD",
            "reason": "Query is out of domain or highly ambiguous.",
            "energy_score": round(energy, 3)
        }

    # 2. Intent & Confidence
    intent_probs = np.exp(intent_logits[0]) / np.sum(np.exp(intent_logits[0]))
    top_intent_id = int(np.argmax(intent_probs))
    confidence = float(intent_probs[top_intent_id])
    detected_intent = REV_INTENT_MAP.get(top_intent_id, "unknown")

    # 3. Extract BIO Slots
    slot_preds = np.argmax(slot_logits[0], axis=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    
    extracted_slots = {}
    current_slot_name = None
    current_slot_value = []

    for token, slot_idx in zip(tokens, slot_preds):
        if token in tokenizer.all_special_tokens:
            continue
        tag = SLOT_LABELS[slot_idx]
        if tag.startswith("B-"):
            if current_slot_name:
                extracted_slots[current_slot_name] = " ".join(current_slot_value).replace(" ##", "")
            current_slot_name = tag[2:]
            current_slot_value = [token]
        elif tag.startswith("I-") and current_slot_name == tag[2:]:
            current_slot_value.append(token)
            
    if current_slot_name:
        extracted_slots[current_slot_name] = " ".join(current_slot_value).replace(" ##", "")

    # 4. Multi-Turn Dialogue State Machine (FSM) Integration
    dialog_response = dialog_manager.process_turn(
        session_id=request.session_id,
        current_intent=detected_intent,
        extracted_slots=extracted_slots,
        confidence=confidence
    )

    # Transparent Audit Headers
    latency_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Inference-Time-MS"] = f"{latency_ms:.2f}"
    response.headers["X-Energy-OOD-Score"] = f"{energy:.3f}"

    return {
        "nlu_layer": {
            "intent": detected_intent,
            "confidence": round(confidence, 4),
            "raw_slots": extracted_slots
        },
        "state_machine": dialog_response
    }
