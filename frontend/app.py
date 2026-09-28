import streamlit as st
import requests

st.set_page_config(page_title="Neobank NLU Router Simulator", layout="wide")

API_URL = "http://router_api:8000"

st.title("⚡ Neobank Semantic Router & Dialog FSM")
st.caption("Sub-15ms INT8 ONNX Inference, Thermodynamic Energy OOD Gating, and Multi-Turn FSM Tracking")

if "session_id" not in st.session_state:
    st.session_state.session_id = "sess_demo_1001"

if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([2, 1])

with col1:
    user_input = st.text_input("Enter natural language banking command (Persian):", 
                               value="می‌خوام به علی پول بفرستم")
    
    c_btn1, c_btn2 = st.columns([1, 4])
    with c_btn1:
        submit = st.button("Send Command", type="primary")
    with c_btn2:
        if st.button("Reset Session"):
            st.session_state.session_id = f"sess_demo_{np.random.randint(1000, 9999)}"
            st.session_state.history = []
            st.rerun()

    if submit and user_input:
        payload = {
            "session_id": st.session_state.session_id,
            "text": user_input
        }
        try:
            resp = requests.post(f"{API_URL}/route-command", json=payload)
            data = resp.json()
            headers = resp.headers
            
            st.session_state.history.append({
                "query": user_input,
                "response": data,
                "headers": headers
            })
        except Exception as e:
            st.error(f"Failed to communicate with Router API: {e}")

    # Render History
    st.subheader("Turn-by-Turn Dialogue Stream")
    for item in reversed(st.session_state.history):
        with st.chat_message("user"):
            st.write(item["query"])
        with st.chat_message("assistant"):
            res = item["response"]
            if res.get("status") == "REJECTED_OOD":
                st.error(f"🚨 Out-Of-Distribution Rejected: {res.get('message')}")
            else:
                st.json(res)

with col2:
    st.subheader("Observability & Headers")
    if st.session_state.history:
        latest = st.session_state.history[-1]
        hdrs = latest["headers"]
        res = latest["response"]
        
        latency = hdrs.get("X-Inference-Time-MS", res.get("latency_ms", "N/A"))
        energy = hdrs.get("X-Energy-OOD-Score", f"{res.get('energy_score', 0):.4f}")
        
        st.metric("X-Inference-Time-MS", f"{latency} ms", delta="Sub-15ms Target", delta_color="normal")
        st.metric("X-Energy-OOD-Score", energy, delta="Threshold: -2.5", delta_color="normal")
        
        st.subheader("Current FSM Machine State")
        st.info(f"FSM Status: **{res.get('dialogue_state', {}).get('status', 'IDLE')}**")
