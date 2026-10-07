import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from src.data_generator import (
    generate_synthetic_corpus, 
    SLOT_LABELS, 
    INTENT_MAP, 
    SLOT_MAP
)

BASE_MODEL_NAME = "distilbert-base-multilingual-cased"

class JointIntentSlotClassifier(nn.Module):
    def __init__(self, base_model_name, num_intents, num_slots):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        hidden_dim = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        
        # Head 1: Sequence Classification (Intent)
        self.intent_classifier = nn.Linear(hidden_dim, num_intents)
        # Head 2: Token Classification (Slot BIO tags)
        self.slot_classifier = nn.Linear(hidden_dim, num_slots)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_token_output = sequence_output[:, 0, :]
        
        cls_token_output = self.dropout(cls_token_output)
        sequence_output = self.dropout(sequence_output)
        
        intent_logits = self.intent_classifier(cls_token_output)
        slot_logits = self.slot_classifier(sequence_output)
        
        return intent_logits, slot_logits

def train_and_export():
    print(f"Initializing Tokenizer & Model: {BASE_MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME):
    tokenizer.save_pretrained("models/tokenizer")
    model = JointIntentSlotClassifier(
        BASE_MODEL_NAME, 
        num_intents=len(INTENT_MAP), 
        num_slots=len(SLOT_LABELS)
    )
    model.eval()

    # Create dummy inputs for ONNX tracing
    sample_text = "پونصد هزار تومن به علی کارت به کارت کن"
    encoded = tokenizer(
        sample_text,
        padding="max_length",
        max_length=32,
        truncation=True,
        return_tensors="pt"
    )
    dummy_input_ids = encoded['input_ids']
    dummy_attention_mask = encoded['attention_mask']

    os.makedirs("models", exist_ok=True)
    onnx_path = "models/semantic_router.onnx"
    quantized_onnx_path = "models/router_int8.onnx"

    print("Exporting Model to ONNX with Dynamic Axes...")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['intent_logits', 'slot_logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'attention_mask': {0: 'batch_size', 1: 'seq_len'},
            'intent_logits': {0: 'batch_size'},
            'slot_logits': {0: 'batch_size', 1: 'seq_len'}
        },
        opset_version=14
    )
    print(f"Base ONNX model exported to {onnx_path}")

    print("Executing INT8 Dynamic Quantization (CPU Optimization)...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized Model saved to {quantized_onnx_path}")

if __name__ == "__main__":
    train_and_export()
