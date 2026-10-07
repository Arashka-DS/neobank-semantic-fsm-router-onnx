# Neobank Semantic Router & Dialog FSM

An ultra-low latency, non-generative Natural Language Understanding (NLU) pipeline designed for mobile banking applications. This architecture converts conversational Persian commands (e.g., *"Transfer 500k Tomans to Ali"*) into strict, execution-ready JSON payloads.

By utilizing discriminative deep learning and deterministic state machines, this system achieves single-digit millisecond latency while eliminating LLM hallucination risks on core financial endpoints.

## 🚀 Architectural Pillars
1. **Joint Intent & Slot Transformer:** A custom PyTorch head built on a distilled transformer (`ParsBERT`). Simultaneously classifies intent and extracts conversational slot entities.
2. **Canonical Banking Currency Normalization:** Handles the Toman vs. Rial discrepancy. Converts informal colloquial user phrasing (*"تومن"*, *"میلیون"*, *"پونصد هزار"*) into explicit, dual-layer objects: user-facing Toman formatting and canonical ISO **IRR (Rials)** required by central payment gateways (Shetab/Shaparak).
3. **Dynamic CPU Quantization:** The PyTorch computational graph is exported to ONNX and quantized dynamically to INT8, achieving 10–15ms inference latencies on bare-metal CPUs without dedicated GPU hardware.
4. **Multi-Turn Dialogue State Tracker (DST):** Uses a Finite State Machine (FSM). When a command is partially provided, the FSM tracks the active session, parks the missing slot state (`AWAITING_SLOT_AMOUNT`), and merges context across turns.
5. **Thermodynamic Out-Of-Distribution (OOD) Gate:** Rejects jailbreaks, adversarial input, and off-topic chat by calculating the free energy score over the raw logit distribution:
   $$E(x) = -T \cdot \log \sum_{i} \exp\left(\frac{f_i(x)}{T}\right)$$
   This overcomes the Softmax overconfidence limitation common in conventional classifiers.

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
Open the Streamlit dialogue visualizer at `http://localhost:7501`.
