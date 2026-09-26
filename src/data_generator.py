import random
from src.normalizer import PersianFinancialNormalizer

normalizer = PersianFinancialNormalizer()

# Slot vocabulary
NAMES = ["علی", "سارا", "رضا", "محمد", "نیما", "زهرا", "امیر", "مریم", "فاطمه"]
BANKS = ["ملی", "بلو", "سامان", "پاسارگاد", "سپه", "صادرات", "ملت", "کشاورزی", "رفاه"]
BILL_TYPES = ["برق", "گاز", "آب", "تلفن", "اینترنت"]
AMOUNTS = [
    ("50000", "پنجاه هزار تومن"),
    ("100000", "صد هزار تومن"),
    ("500000", "پونصد هزار تومن"),
    ("1000000", "یک تومن"),
    ("2000000", "دو تومن"),
    ("5000000", "پنج تومن")
]

INTENT_CARD_TO_CARD = "card_to_card"
INTENT_PAY_BILL = "pay_bill"
INTENT_CHECK_BALANCE = "check_balance"

INTENT_MAP = {
    INTENT_CARD_TO_CARD: 0,
    INTENT_PAY_BILL: 1,
    INTENT_CHECK_BALANCE: 2
}

SLOT_LABELS = [
    "O",
    "B-AMOUNT", "I-AMOUNT",
    "B-RECIPIENT", "I-RECIPIENT",
    "B-DEST_BANK", "I-DEST_BANK",
    "B-BILL_TYPE", "I-BILL_TYPE"
]
SLOT_MAP = {tag: i for i, tag in enumerate(SLOT_LABELS)}

def generate_synthetic_corpus(n_samples=2500):
    samples = []
    
    for _ in range(n_samples):
        intent_type = random.choice([INTENT_CARD_TO_CARD, INTENT_PAY_BILL, INTENT_CHECK_BALANCE])
        
        if intent_type == INTENT_CARD_TO_CARD:
            amt_val, amt_str = random.choice(AMOUNTS)
            name = random.choice(NAMES)
            bank = random.choice(BANKS)
            
            # Templates with slot placement
            templates = [
                f"{amt_str} تومن کارت به کارت کن به {name} بانک {bank}",
                f"{amt_str} تومن به حساب بانک {bank} {name} کارت به کارت کن",
                f"{amt_str} بریز به حساب {bank} {name}",
                f"انتقال بده {amt_str} تومن به کارت {name}",
                f"{amt_str} تومن به کارت {name} انتقال بده",
                f"بزن {amt_str} به حساب {name} بانک {bank}",
                f"{amt_str} به حساب بانک {bank} {name} بزن"
            ]
            text = normalizer.normalize(random.choice(templates))
            tokens = text.split()
            tags = ["O"] * len(tokens)
            
            # Simple word-boundary slot assignment
            for i, w in enumerate(tokens):
                if w in amt_str.split():
                    tags[i] = "B-AMOUNT" if tags[max(0, i-1)] != "B-AMOUNT" else "I-AMOUNT"
                elif w == name:
                    tags[i] = "B-RECIPIENT"
                elif w == bank:
                    tags[i] = "B-DEST_BANK"
                    
            samples.append((tokens, tags, INTENT_MAP[INTENT_CARD_TO_CARD]))
            
        elif intent_type == INTENT_PAY_BILL:
            bill = random.choice(BILL_TYPES)
            templates = [
                f"قبض {bill} رو پرداخت کن",
                f"تسویه حساب قبض {bill} از کیف پول",
                f"پرداخت سریع قبض {bill}"
            ]
            text = normalizer.normalize(random.choice(templates))
            tokens = text.split()
            tags = ["O"] * len(tokens)
            for i, w in enumerate(tokens):
                if w == bill:
                    tags[i] = "B-BILL_TYPE"
            samples.append((tokens, tags, INTENT_MAP[INTENT_PAY_BILL]))
            
        elif intent_type == INTENT_CHECK_BALANCE:
            bank = random.choice(BANKS)
            templates = [
                "موجودی حسابم چقدره",
                f"استعلام موجودی کارت {bank}",
                "چقدر تو کیف پولم پول دارم",
                "چقدر توی کیف پولم دارم"
                "موجودی فعلی من رو نشون بده"
            ]
            text = normalizer.normalize(random.choice(templates))
            tokens = text.split()
            tags = ["O"] * len(tokens)
            for i, w in enumerate(tokens):
                if w == bank:
                    tags[i] = "B-DEST_BANK"
            samples.append((tokens, tags, INTENT_MAP[INTENT_CHECK_BALANCE]))
            
    return samples
