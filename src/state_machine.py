import os
import json
import redis

class DialogFSM:
    def __init__(self):
        self.redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=6379,
            decode_responses=True
        )
        self.TTL_SECONDS = 300 # 5-minute session timeout

    def get_state(self, session_id: str) -> dict:
        state = self.redis_client.get(session_id)
        if state:
            return json.loads(state)
        return {"current_intent": None, "slots": {}, "status": "IDLE"}

    def step(self, session_id: str, intent: str, extracted_slots: dict) -> dict:
        state = self.get_state(session_id)
        
        # If a new valid intent is detected, override current state context
        if intent not in ["UNKNOWN", "OOD"] and state["current_intent"] != intent:
            state["current_intent"] = intent
            state["slots"] = extracted_slots
        else:
            # Merge slots across conversational turns
            state["slots"].update(extracted_slots)
            
        current_intent = state["current_intent"]

        # -----------------------------------------
        # FSM Routing & Slot Validation Logic
        # -----------------------------------------
        if current_intent == "TRANSFER":
            has_dest = "RECIPIENT" in state["slots"] or "DESTINATION_CARD" in state["slots"]
            if "AMOUNT" in state["slots"] and has_dest:
                state["status"] = "READY_FOR_EXECUTION"
            elif "AMOUNT" not in state["slots"]:
                state["status"] = "AWAITING_SLOT_AMOUNT"
            elif not has_dest:
                state["status"] = "AWAITING_SLOT_DESTINATION"
                
        elif current_intent == "BALANCE_INQUIRY":
            # Balance checks require no slots. Instantly ready.
            state["status"] = "READY_FOR_EXECUTION"
            
        elif current_intent == "BILL_PAYMENT":
            if "BILL_ID" in state["slots"]:
                state["status"] = "READY_FOR_EXECUTION"
            else:
                state["status"] = "AWAITING_SLOT_BILL_ID"
                
        else:
            state["status"] = "IDLE"
            
        self.redis_client.set(session_id, json.dumps(state), ex=self.TTL_SECONDS)
        return state
