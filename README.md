# Neobank Semantic Router & Dialog FSM

An ultra-low latency, non-generative Natural Language Understanding (NLU) pipeline designed for mobile banking apps. This architecture converts conversational commands (e.g., *"Transfer 5 million to Ali"*) into strict, execution-ready JSON payloads.

By utilizing discriminative deep learning and strict state machines, this system achieves single-digit millisecond latency while physically guaranteeing zero LLM hallucination risk on financial endpoints.

## 🚀 Architectural Pillars
1. **Joint Intent & Slot Transformer:** A custom PyTorch head built on a distilled transformer. It simultaneously predicts the overall sentence intent and tags token boundaries (`B-AMOUNT`, `I-RECIPIENT`) using the BIO tagging scheme.
2. **Dynamic CPU Quantization:** The model is exported to an ONNX graph with dynamic sequence axes and quantized to INT8, enabling 10-15ms inference times on bare-metal CPUs without requiring GPU infrastructure.
3. **Multi-Turn Dialogue State Tracker (DST):** Incorporates an in-memory Finite State Machine (FSM). If a user provides an incomplete command, the FSM tracks the active session, parks the missing state (`AWAITING_SLOT_AMOUNT`), and merges contexts across conversational turns.
4. **Thermodynamic Out-Of-Distribution (OOD) Gate:** Rejects jailbreaks, adversarial text, and non-financial chat by calculating the Energy Score over the raw logits. This prevents the Softmax overconfidence problem inherent in standard classification models.

## ⚙️ Quick Start
1. `docker-compose up -d --build`
2. Test a complete turn: POST to `http://localhost:8000/route-command` with `"پونصد هزار تومن به علی کارت به کارت کن"`. The FSM will return `READY_FOR_EXECUTION`.
3. Test a partial turn: POST with `"می‌خوام به علی پول بفرستم"`. The FSM will return `status: INCOMPLETE` and request the missing `AMOUNT` slot.
4. Check the Response Headers for `X-Inference-Time-MS` and `X-Energy-OOD-Score` observability metrics.
