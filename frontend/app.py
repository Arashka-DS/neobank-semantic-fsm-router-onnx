import streamlit as st
import requests
import uuid
import json

st.set_page_config(page_title="Neobank NLU Engine", layout="wide", initial_sidebar_state="collapsed")

# Initialize Session State
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_data" not in st.session_state:
    st.session_state.debug_data = {}

st.title("⚡ Neobank Semantic Router & FSM")

col1, col2 = st.columns([1.2, 1])

# --- LEFT COLUMN: Chat Interface ---
with col1:
    st.subheader("📱 Banking Assistant")
    chat_container = st.container(height=500)
    
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
    user_input = st.chat_input("Type a command (e.g., '۵۰۰ هزار تومن به علی کارت به کارت کن')")
    
    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
                
        # Call API
        payload = {"session_id": st.session_state.session_id, "user_query": user_input}
        try:
            res = requests.post("http://semantic_router:8000/route-command", json=payload)
            res.raise_for_status()
            data = res.json()
            
            # Extract headers for telemetry
            latency = res.headers.get("X-Inference-Time-MS", "0")
            energy = res.headers.get("X-Energy-OOD-Score", "0")
            
            # Handle API Response
            if data.get("status") == "REJECTED_OOD":
                bot_reply = "⚠️ I didn't understand that as a banking command. Please try again."
                st.session_state.debug_data = {"OOD": True, "Energy": energy, "Latency": latency}
            else:
                fsm = data.get("state_machine", {})
                nlu = data.get("nlu_layer", {})
                
                st.session_state.debug_data = {
                    "OOD": False,
                    "Energy": energy,
                    "Latency": latency,
                    "Intent": nlu.get("intent"),
                    "Confidence": nlu.get("confidence"),
                    "Slots": nlu.get("raw_slots"),
                    "FSM_State": fsm.get("dialog_state")
                }
                
                if fsm.get("status") == "INCOMPLETE":
                    missing = ", ".join(fsm.get("missing_slots", []))
                    bot_reply = f"I can help with that. Please provide the missing detail: **{missing}**"
                else:
                    bot_reply = f"✅ Transaction payload ready! Transferring {fsm['execution_payload']['collected_slots']}."
                    # Reset session ID after successful execution
                    st.session_state.session_id = str(uuid.uuid4())[:8]

        except Exception as e:
            bot_reply = f"System Error: {e}"
            
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(bot_reply)
        st.rerun()

# --- RIGHT COLUMN: Developer Observability ---
with col2:
    st.subheader("⚙️ NLU Telemetry & FSM State")
    st.markdown("Monitor real-time ONNX extraction and State Machine routing.")
    
    debug = st.session_state.debug_data
    if debug:
        m1, m2 = st.columns(2)
        m1.metric("ONNX Inference Latency", f"{debug.get('Latency')} ms")
        
        # Energy Score styling
        energy_val = float(debug.get('Energy', 0))
        energy_color = "normal" if energy_val < -5.0 else "inverse"
        m2.metric("OOD Energy Score", f"{energy_val:.2f}", delta="Rejection > -5.0", delta_color=energy_color)
        
        st.divider()
        
        if debug.get("OOD"):
            st.error("🚨 QUERY REJECTED: Out-Of-Distribution (OOD) detected by Thermodynamics Energy threshold. Stopped before hitting FSM.")
        else:
            st.success(f"**NLU Intent:** {debug.get('Intent')} (Conf: {debug.get('Confidence')})")
            st.info(f"**Active FSM State:** `{debug.get('FSM_State')}`")
            
            st.markdown("**Extracted BIO Slots:**")
            st.json(debug.get('Slots', {}))
    else:
        st.info("Awaiting input to generate telemetry...")
