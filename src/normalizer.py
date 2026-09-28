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
        Extracts numerical amount, detects currency unit (Toman vs Rial),
        and canonicalizes to ISO IRR (Iranian Rial = Toman * 10).
        """
        clean_text = self.normalize(text)
        
        # 1. Detect Explicit Currency Unit
        if "ریال" in clean_text:
            detected_unit = "RIAL"
        elif any(unit in clean_text for unit in ["تومان", "تومن", "ت"]):
            detected_unit = "TOMAN"
        else:
            # Neobank default convention: conversational payments are assumed Tomans
            detected_unit = "TOMAN"

        # 2. Extract Base Value (Digits + Word Scale Multipliers)
        digits = re.findall(r'\d+', clean_text)
        base_value = int(digits[0]) if digits else 0
        
        # Handle compound colloquial phrasing (e.g., "500 هزار", "2 میلیون")
        multiplier = 1
        for word, factor in self.word_multipliers.items():
            if word in clean_text:
                if base_value > 0 and base_value < factor:
                    base_value *= factor
                elif base_value == 0:
                    base_value = factor
                break

        if base_value == 0:
            return None

        # 3. Canonicalize to Core Banking Standard (IRR)
        if detected_unit == "TOMAN":
            canonical_irr = base_value * 10
            toman_value = base_value
        else: # Explicitly RIAL
            canonical_irr = base_value
            toman_value = base_value // 10

        return {
            "parsed_value": base_value,
            "detected_unit": detected_unit,
            "canonical_amount_irr": canonical_irr,
            "amount_toman": toman_value,
            "formatted_display": f"{toman_value:,} تومان ({canonical_irr:,} ریال)"
        }
