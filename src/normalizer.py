import re

class PersianFinancialNormalizer:
    def __init__(self):
        self.persian_digits = "۰۱۲۳۴۵۶۷۸۹"
        self.arabic_digits = "٠١٢٣٤٥٦٧٨٩"
        self.ascii_digits = "0123456789"
        
        self.digit_trans = str.maketrans(
            self.persian_digits + self.arabic_digits,
            self.ascii_digits * 2
        )
        
        # Character harmonization
        self.char_map = {
            'ي': 'ی',
            'ك': 'ک',
            'إ': 'ا',
            'أ': 'ا',
            'ة': 'ه',
            'ؤ': 'و',
            '\u200c': ' ' # Map ZWNJ to single space for uniform whitespace tokenization
        }

    def normalize(self, text: str) -> str:
        if not text:
            return ""
        
        # Transliterate digits
        text = text.translate(self.digit_trans)
        
        # Normalize specific Persian/Arabic characters
        for src, dst in self.char_map.items():
            text = text.replace(src, dst)
            
        # Remove repeated punctuation and trim extra whitespaces
        text = re.sub(r'[!?.,،؛]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
