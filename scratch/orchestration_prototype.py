"""Local prototype: cross-encoder retrieval (Planner) + template-fill reply (Speaker).

Retrieval uses a quantized ONNX cross-encoder reranker (see reranker.py) —
brought back after a pure-regex phase (rule_matcher.py, still in the repo)
per explicit request. The SLM proved too unreliable at free-text rewriting
to guarantee coherent output — golden replies are already correct, warm,
hand-written Hindi, so the Speaker step just fills placeholders instead of
generating; no SLM in the live pipeline at all currently.

Run: python scratch/orchestration_prototype.py
"""

import re
import time

from acknowledge import build_off_topic_reply
from golden_data import NONE_CONDITION, get_reply, load_steps
from reranker import get_reranker, rerank

PLACEHOLDER_RE = re.compile(r"\{[^}]*child['’]?s?\s*name[^}]*\}", re.IGNORECASE)


def fill_placeholders(golden_reply, child_name="dost"):
    """No SLM here: template-fill only, guarantees coherent output since the
    golden reply is already correct, warm, hand-written Hindi."""
    return PLACEHOLDER_RE.sub(child_name, golden_reply)


def run_turn(steps, lesson_id, section_num, step_idx, transcript, history, child_name="dost"):
    step = steps[(lesson_id, section_num, step_idx)]

    t0 = time.monotonic()
    condition, scores = rerank(transcript, step)
    t1 = time.monotonic()

    golden_reply = get_reply(step, condition) or step["canonical_prompt"]
    filled_reply = fill_placeholders(golden_reply, child_name)
    # golden_reply falls back to canonical_prompt itself when off-topic, so
    # filled_reply IS the redirect line here — just prepend an acknowledgment.
    text = build_off_topic_reply(transcript, filled_reply) if condition == NONE_CONDITION else filled_reply
    t2 = time.monotonic()

    return {
        "condition": condition,
        "golden_reply": golden_reply,
        "text": text,
        "retrieve_latency": t1 - t0,
        "generate_latency": t2 - t1,
        "total_latency": t2 - t0,
    }


DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")


def is_hindi_dominant(text):
    return bool(DEVANAGARI_RE.search(text))


if __name__ == "__main__":
    get_reranker()  # warm up
    steps = load_steps()
    key = ("lesson_20260603_105025", 4, 0)  # Magic Age Guessing Game
    history = []
    result = run_turn(steps, *key, transcript="nahi", history=history)
    print(result)
