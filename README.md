# Neobank Semantic Router & Dialog FSM

An ultra-low latency, non-generative Natural Language Understanding (NLU) pipeline designed for mobile banking apps. This architecture converts conversational commands (e.g., *"Transfer 5 million to Ali"*) into strict, execution-ready JSON payloads.

By utilizing discriminative deep learning and strict state machines, this system achieves single-digit millisecond latency while physically guaranteeing zero LLM hallucination risk on financial endpoints.

## 🚀 Architectural Pillars
1. **Joint Intent & Slot Transformer:** A custom PyTorch head built on a distilled transformer. It simultaneously predicts the overall sentence intent and tags token boundaries (`B-AMOUNT`, `I-RECIPIENT`) using the BIO tagging scheme.
2. **Dynamic CPU Quantization:** The model is exported to an ONNX graph with dynamic sequence axes and quantized to INT8, enabling 10-15ms inference times on bare-metal CPUs without requiring GPU infrastructure.
3. **Multi-Turn Dialogue State Tracker (DST):** Incorporates a Finite State Machine (FSM). If a user provides an incomplete command, the FSM tracks the active session, parks the missing state (`AWAITING_SLOT_AMOUNT`), and merges contexts across conversational turns.
4. **Thermodynamic Out-Of-Distribution (OOD) Gate:** Rejects jailbreaks, adversarial text, and non-financial chat by calculating the Energy Score over the raw logits:
   $$E(x) = -T \cdot \log \sum_{i} \exp\left(\frac{f_i(x)}{T}\right)$$
   This prevents the Softmax overconfidence problem inherent in standard classification models.

## ⚙️ Quick Start
1. **Spin up the services:**
   ```bash
   docker-compose up -d --build
   ```
2. **Test a complete single-turn command:**
   ```bash
   curl -X POST http://localhost:8000/route-command \
     -H "Content-Type: application/json" \
     -d '{"session_id": "test_1", "text": "پونصد هزار تومن به علی کارت به کارت کن"}'
   ```  
3. **Test a partial multi-turn command (FSM slot parking):**
   ```bash
   curl -X POST http://localhost:8000/route-command \
     -H "Content-Type: application/json" \
     -d '{"session_id": "test_2", "text": "می‌خوام به علی پول بفرستم"}'
   ```
4. **Inspect Response Headers:**
Check headers for `X-Inference-Time-MS` and `X-Energy-OOD-Score` observability telemetry.
5. **Interactive UI:**
Open the Streamlit dialogue visualizer at `http://localhost:8501`.
