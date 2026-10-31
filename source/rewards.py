"""Strict, deterministic numeric scoring; no generated code is executed."""
import re
from decimal import Decimal

SYSTEM_PROMPT = (
    'Solve the math problem. Show a concise calculation, then end with exactly '
    'one final answer line in the form: #### 42. '
    'Replace 42 with your numeric answer; do not put units after it.'
)
NUMBER = r'[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
FINAL = re.compile(rf'^####\s*({NUMBER})\s*$', re.MULTILINE)


def extract_answer(text):
    text = text.strip()
    matches = list(FINAL.finditer(text))
    if len(matches) != 1 or matches[0].end() != len(text):
        return None
    return Decimal(matches[0].group(1).replace(',', ''))


def correct(text, answer):
    value = extract_answer(text)
    return value is not None and value == Decimal(str(answer))


def accuracy_reward(completions, answer, **kwargs):
    texts = [c if isinstance(c, str) else c[0]['content'] for c in completions]
    if len(texts) != len(answer):
        raise ValueError('Completion and answer counts differ')
    return [float(correct(text, gold)) for text, gold in zip(texts, answer)]
