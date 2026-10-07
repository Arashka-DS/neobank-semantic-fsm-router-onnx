import re

class PersianFinancialNormalizer:
    def __init__(self):
        self.char_map = {
            'ي': 'ی', 'ك': 'ک', 'دِ': 'د', 'بِ': 'ب', 'زِ': 'ز',
            'ذِ': 'ذ', 'شِ': 'ش', 'سِ': 'س', 'ة': 'ه', 'ۀ': 'ه',
        }
        self.digit_map = {
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }
        self.word_multipliers = {
            "میلیارد": 1_000_000_000,
            "همت": 1_000_000_000_000, # Hezār Milliard Toman (Enterprise scale)
            "میلیون": 1_000_000,
            "ملیون": 1_000_000,
            "هزار": 1_000,
            "پونصد": 500,
            "پانصد": 500,
            "دویست": 200,
            "سیصد": 300,
            "چهارصد": 400,
            "ششصد": 600,
            "هفتصد": 700,
            "هشتصد": 800,
            "نهصد": 900
        }

    def normalize(self, text: str) -> str:
        """Cleans Arabic loan-letters, normalizes whitespace, and half-spaces."""
        if not text:
            return ""
        for k, v in self.char_map.items():
            text = text.replace(k, v)
        for k, v in self.digit_map.items():
            text = text.replace(k, v)
        text = re.sub(r'[\u200c\u200b]+', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def parse_financial_amount(self, text: str) -> dict:
        """
        Extracts numerical amount, handles Persian compound multipliers,
        and canonicalizes to ISO IRR (Iranian Rial = Toman * 10).
        """
        clean_text = self.normalize(text)
        
        if "ریال" in clean_text:
            detected_unit = "RIAL"
        elif any(unit in clean_text for unit in ["تومان", "تومن", "ت"]):
            detected_unit = "TOMAN"
        else:
            detected_unit = "TOMAN"

        # 1. Map textual numbers to digits for easier parsing
        num_map = {
            "یک": 1, "دو": 2, "سه": 3, "چهار": 4, "پنج": 5, "شش": 6, "هفت": 7, "هشت": 8, "نه": 9,
            "ده": 10, "بیست": 20, "سی": 30, "چهل": 40, "پنجاه": 50, "شصت": 60, "هفتاد": 70, "هشتاد": 80, "نود": 90,
            "صد": 100, "دویست": 200, "سیصد": 300, "چهارصد": 400, "پانصد": 500, "پونصد": 500, "ششصد": 600, "هفتصد": 700, "هشتصد": 800, "نهصد": 900
        }
        mult_map = {
            "هزار": 1_000, "میلیون": 1_000_000, "ملیون": 1_000_000, "میلیارد": 1_000_000_000, "همت": 1_000_000_000_000
        }

        tokens = clean_text.split()
        base_value = 0
        current_val = 0
        
        # 2. Token-based aggregation
        for token in tokens:
            if token.isdigit():
                current_val += int(token)
            elif token in num_map:
                current_val += num_map[token]
            elif token in mult_map:
                if current_val == 0:
                    current_val = 1 # Implicit "یک هزار"
                current_val *= mult_map[token]
                base_value += current_val
                current_val = 0
        
        base_value += current_val

        if base_value == 0:
            return None

        # 3. Handle Conversational Shorthand
        if base_value < 1000 and ("تومن" in clean_text or "تومان" in clean_text):
            if base_value < 10:
                base_value *= 1_000_000  # e.g., "پنج تومن" -> 5 Million Toman
            else:
                base_value *= 1_000      # e.g., "پنجاه تومن" -> 50 Thousand Toman

        # 4. Canonicalize to Core Banking Standard (IRR)
        canonical_irr = base_value * 10 if detected_unit == "TOMAN" else base_value
        toman_value = base_value if detected_unit == "TOMAN" else base_value // 10

        return {
            "parsed_value": base_value,
            "detected_unit": detected_unit,
            "canonical_amount_irr": canonical_irr,
            "amount_toman": toman_value,
            "formatted_display": f"{toman_value:,} تومان ({canonical_irr:,} ریال)"
        }
