import time
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from src.normalizer import PersianFinancialNormalizer
from src.data_generator import SLOT_LABELS, INTENT_MAP

app = FastAPI(title="Neobank Semantic Router & Slot Filling Engine", version="1.0.0")

# Inverted Lookups
REV_INTENT_MAP = {v: k for k, v in INTENT_MAP.items()}
MODEL_PATH = "models/semantic_router_int8.onnx"
TOKENIZER_NAME = "distilbert-base-multilingual-cased"

normalizer = PersianFinancialNormalizer()
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

# Load ONNX Inference Session
ort_session = ort.InferenceSession(
    MODEL_PATH if ort.get_device() == 'CPU' else "models/semantic_router.onnx",
    providers=['CPUExecutionProvider']
)

class CommandRequest(BaseModel):
    user_query: str = Field(..., example="مبلغ ۵۰۰ هزار تومان به علی کارت به کارت کن")
    account_id: str = Field(..., example="ACC-99210")

class FinancialActionPayload(BaseModel):
    intent: str
    confidence: float
    slots: dict
    execution_status: str
    requires_confirmation: bool

@app.post("/route-command", response_model=FinancialActionPayload)
def parse_and_route_command(request: CommandRequest, response: Response):
    start_time = time.perf_counter()
    
    # 1. Normalize Persian text
    cleaned_query = normalizer.normalize(request.user_query)
    if not cleaned_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 2. Tokenize
    inputs = tokenizer(
        cleaned_query,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="np"
    )
    
    input_ids = inputs["input_ids"].astype(np.int64)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # 3. ONNX Session Run
    ort_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask
    }
    intent_logits, slot_logits = ort_session.run(None, ort_inputs)

    # 4. Softmax & Confidence Calculation
    intent_probs = np.exp(intent_logits[0]) / np.sum(np.exp(intent_logits[0]))
    top_intent_id = int(np.argmax(intent_probs))
    confidence = float(intent_probs[top_intent_id])
    detected_intent = REV_INTENT_MAP.get(top_intent_id, "unknown")

    # 5. Extract BIO Slots
    slot_preds = np.argmax(slot_logits[0], axis=-1)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
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
        else:
            if current_slot_name:
                extracted_slots[current_slot_name] = " ".join(current_slot_value).replace(" ##", "")
                current_slot_name = None
                current_slot_value = []
                
    if current_slot_name:
        extracted_slots[current_slot_name] = " ".join(current_slot_value).replace(" ##", "")

    # Calculate latency in ms
    latency_ms = (time.perf_counter() - start_time) * 1000
    response.headers["X-Inference-Time-MS"] = f"{latency_ms:.2f}"

    # 6. Safety Check & Financial Fallback Guardrail
    CONFIDENCE_THRESHOLD = 0.75
    if confidence < CONFIDENCE_THRESHOLD:
        return FinancialActionPayload(
            intent=detected_intent,
            confidence=round(confidence, 4),
            slots=extracted_slots,
            execution_status="DISAMBIGUATION_REQUIRED",
            requires_confirmation=True
        )

    return FinancialActionPayload(
        intent=detected_intent,
        confidence=round(confidence, 4),
        slots=extracted_slots,
        execution_status="READY_FOR_EXECUTION",
        requires_confirmation=False
    )

@app.get("/health")
def health():
    return {"status": "ACTIVE", "runtime": "ONNX-CPU-INT8"}
