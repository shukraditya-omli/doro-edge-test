"""Cross-encoder reranker: retrieval step, ONNX int8-quantized (~571MB vs
2.1GB fp32 original). Tried gte-multilingual-reranker-base (306M, ~341MB) as
a smaller swap-in, but it doesn't discriminate off-topic from genuine
matches on this task (off-topic scored 0.13 while irrelevant conditions
scored 0.27-0.37) — reverted to bge-reranker-v2-m3, which cleanly separates
off-topic (~0.00002) from genuine matches (0.008+).
"""

import re

from sentence_transformers import CrossEncoder

from golden_data import NONE_CONDITION, list_conditions

MODEL_NAME = "onnx-community/bge-reranker-v2-m3-ONNX"
ONNX_FILE = "onnx/model_quantized.onnx"
SCORE_THRESHOLD = 0.005  # genuine matches score 0.008-0.9+, off-topic scores ~0.00002 across the board

YES_HINT = " (haan, ha, yes, ji haan)"
NO_HINT = " (nahi, nahin, na, no)"
NUMBER_HINTS = {
    "one": "1, वन, एक",
    "two": "2, टू, दो",
    "three": "3, थ्री, तीन",
    "four": "4, फ़ोर, चार",
    "1": "one, वन, एक",
    "2": "two, टू, दो",
    "3": "three, थ्री, तीन",
    "4": "four, फ़ोर, चार",
}
FRUIT_HINT = ", such as mango, banana, aam, or kela"

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(
            MODEL_NAME, backend="onnx", model_kwargs={"file_name": ONNX_FILE}, max_length=256
        )
    return _reranker


def scoring_text(condition):
    lowered = condition.lower()
    hints = []
    if "yes" in lowered:
        hints.append(YES_HINT)
    if "no" in lowered and "off-topic" not in lowered:
        hints.append(NO_HINT)
    if "names a fruit" in lowered:
        hints.append(FRUIT_HINT)
    for word, hint in NUMBER_HINTS.items():
        if re.search(rf"\b{word}\b", lowered):
            hints.append(f" ({hint})")
    return condition + "".join(hints)


def rerank(transcript, step):
    if not transcript.strip():
        conditions = list_conditions(step)
        for c in conditions:
            if "silent" in c.lower():
                return c, None
        return NONE_CONDITION, None

    conditions = [
        c for c in list_conditions(step) if c != NONE_CONDITION and "silent" not in c.lower()
    ]
    pairs = [(transcript, scoring_text(c)) for c in conditions]
    scores = get_reranker().predict(pairs)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    if scores[best_idx] < SCORE_THRESHOLD:
        return NONE_CONDITION, scores
    return conditions[best_idx], scores
