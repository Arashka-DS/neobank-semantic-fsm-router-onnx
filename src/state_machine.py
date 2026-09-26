import time

class PersianValueParser:
    """Parses colloquial Persian amounts into exact integers."""
    def parse_amount(self, text_slot: str) -> int:
        if not text_slot:
            return 0
            
        multiplier = 1
        text = text_slot.replace("تومن", "").replace("تومان", "").replace("ریال", "").strip()
        
        if "هزار" in text:
            multiplier = 1000
            text = text.replace("هزار", "").strip()
        elif "میلیون" in text:
            multiplier = 1000000
            text = text.replace("میلیون", "").strip()
        elif "میلیارد" in text:
            multiplier = 1000000000
            text = text.replace("میلیارد", "").strip()
            
        try:
            # Handle standard numbers parsed by earlier regex/normalizer
            numeric_val = float(text)
            return int(numeric_val * multiplier)
        except ValueError:
            return 0 # Fallback for complex unhandled slang

class DialogStateTracker:
    def __init__(self):
        # In production, this dictionary is replaced by Redis with a TTL of 300 seconds
        self.session_store = {}
        self.parser = PersianValueParser()
        
        # Schema definition: what slots are required for which intents to execute
        self.required_slots = {
            "card_to_card": ["AMOUNT", "RECIPIENT"],
            "pay_bill": ["BILL_TYPE"],
            "check_balance": []
        }

    def process_turn(self, session_id: str, current_intent: str, extracted_slots: dict, confidence: float):
        # Initialize or retrieve session
        if session_id not in self.session_store:
            self.session_store[session_id] = {
                "state": "IDLE",
                "active_intent": current_intent,
                "collected_slots": {},
                "created_at": time.time()
            }
            
        session = self.session_store[session_id]
        
        # Merge newly extracted slots into session memory
        for key, value in extracted_slots.items():
            if key == "AMOUNT":
                session["collected_slots"][key] = self.parser.parse_amount(value)
            else:
                session["collected_slots"][key] = value

        # Update intent if confidence is extremely high (User changed their mind)
        if confidence > 0.90 and current_intent != session["active_intent"] and session["state"] != "IDLE":
            session["active_intent"] = current_intent
            session["collected_slots"] = {} # Clear slots on context switch
            
        # State Machine Validation Logic
        required = self.required_slots.get(session["active_intent"], [])
        missing = [req for req in required if req not in session["collected_slots"]]
        
        if missing:
            session["state"] = f"AWAITING_{missing[0]}"
            return {
                "session_id": session_id,
                "status": "INCOMPLETE",
                "dialog_state": session["state"],
                "missing_slots": missing,
                "system_prompt": f"Please provide the {missing[0]}."
            }
            
        # All slots collected
        session["state"] = "READY_FOR_EXECUTION"
        final_payload = session.copy()
        
        # Cleanup session after successful collection
        del self.session_store[session_id]
        
        return {
            "session_id": session_id,
            "status": "READY",
            "dialog_state": "READY_FOR_EXECUTION",
            "execution_payload": final_payload
        }
