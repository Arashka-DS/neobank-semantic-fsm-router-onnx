import re

class PersianFinancialNormalizer:
    def __init__(self):
        # Character translation tables
        self.char_map = {
            'ي': 'ی',
            'ك': 'ک',
            'دِ': 'د',
            'بِ': 'ب',
            'زِ': 'ز',
            'ذِ': 'ذ',
            'شِ': 'ش',
            'سِ': 'س',
            'ة': 'ه',
            'ۀ': 'ه',
        }
        self.digit_map = {
            '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
            '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
            '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
            '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
        }

    def normalize(self, text: str) -> str:
        """Cleans Arabic loan-letters, normalizes whitespace, and handles half-spaces."""
        if not text:
            return ""
            
        # Character substitutions
        for k, v in self.char_map.items():
            text = text.replace(k, v)
            
        # Convert Eastern/Persian digits to standard Western digits
        for k, v in self.digit_map.items():
            text = text.replace(k, v)
            
        # Standardize zero-width non-joiner (half-space)
        text = re.sub(r'[\u200c\u200b]+', ' ', text)
        
        # Clean repetitive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_numerical_amount(self, text: str) -> int:
        """Heuristic extractor for spoken/typed amounts in Tomans/Rials."""
        normalized = self.normalize(text)
        
        # Match explicit digit sequences
        digits = re.findall(r'\d+', normalized)
        if digits:
            val = int(digits[0])
            if "میلیون" in text:
                val *= 1_000_000
            elif "هزار" in text:
                val *= 1_000
            return val
            
        # Text-based word patterns
        word_multipliers = {
            "میلیون": 1_000_000,
            "هزار": 1_000,
            "پونصد": 500,
            "دویست": 200,
            "صد": 100
        }
        total = 0
        for word, factor in word_multipliers.items():
            if word in text:
                total += factor
        return total if total > 0 else 0
