"""Test harness: retrieve+generate accuracy/latency over hand-written cases.

Run: python scratch/test_prototype.py
"""

import re

from orchestration_prototype import is_hindi_dominant, run_turn
from golden_data import load_steps
from reranker import get_reranker

AGE_STEP = ("lesson_20260603_105025", 4, 0)  # Magic Age Guessing Game
STAR_STEP = ("lesson_20260603_105025", 5, 1)  # star counting step 1
BYE_STEP = ("lesson_20260603_105025", 10, 4)  # closing, "child says bye or anything close"
MARS_STEP = ("lesson_20260603_110725", 1, 0)  # "child says Mars"
FRUIT_STEP = ("lesson_20260603_111108", 4, 0)  # "child names a fruit"
AMBIG_STEP = ("lesson_20260603_105025", 5, 2)  # star step 2: only "child tries" / "silent", no keyword family

CASES = [
    (AGE_STEP, "nahi", "child says no or anything similar"),
    (AGE_STEP, "haan", "child says yes"),
    (AGE_STEP, "मैं 3 साल का हूँ", "child says 3"),
    (AGE_STEP, "", "silent"),
    (AGE_STEP, "mujhe chips khana hai", "none/off-topic"),
    (STAR_STEP, "one", "child says one or close"),
    (STAR_STEP, "वन", "child says one or close"),
    (STAR_STEP, "", "silent"),
    (BYE_STEP, "bye bye doro", "child says bye or anything close"),
    (MARS_STEP, "Mars", "child says Mars"),
    (FRUIT_STEP, "mango pasand hai", "child names a fruit"),
    (AMBIG_STEP, "do", "child tries"),
    (AMBIG_STEP, "mujhe bhookh lagi hai", "none/off-topic"),
]

NUMBER_RE = re.compile(r"\d+")


def numbers_preserved(golden_reply, text):
    expected = NUMBER_RE.findall(golden_reply)
    if not expected:
        return True
    return all(n in text for n in expected)


def main():
    get_reranker()  # warm up: exclude model load time from per-case latency
    steps = load_steps()

    results = []
    for step_key, transcript, expected in CASES:
        r = run_turn(steps, *step_key, transcript=transcript, history=[])
        passed = r["condition"] == expected
        hindi_ok = is_hindi_dominant(r["text"])
        nums_ok = numbers_preserved(r["golden_reply"], r["text"])
        results.append((step_key, transcript, expected, r, passed, hindi_ok, nums_ok))

    n = len(results)
    n_pass = sum(1 for r in results if r[4])
    n_hindi = sum(1 for r in results if r[5])
    n_nums = sum(1 for r in results if r[6])
    avg_latency = sum(r[3]["total_latency"] for r in results) / n

    print(f"{'transcript':<25} {'expected':<35} {'got':<35} {'pass':<5} {'hindi':<6} {'nums':<5} lat(s)")
    for step_key, transcript, expected, r, passed, hindi_ok, nums_ok in results:
        print(
            f"{transcript!r:<25} {expected:<35} {r['condition']:<35} "
            f"{str(passed):<5} {str(hindi_ok):<6} {str(nums_ok):<5} {r['total_latency']:.2f}"
        )

    print()
    print(f"Retrieval accuracy: {n_pass}/{n} ({100 * n_pass / n:.0f}%)")
    print(f"Hindi-output rate: {n_hindi}/{n} ({100 * n_hindi / n:.0f}%)")
    print(f"Number-preservation rate: {n_nums}/{n} ({100 * n_nums / n:.0f}%)")
    print(f"Avg total latency: {avg_latency:.2f}s (target < 2.0s)")


if __name__ == "__main__":
    main()
