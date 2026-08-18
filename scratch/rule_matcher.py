"""Rule-based retrieval: replaces the cross-encoder reranker entirely.

The condition vocabulary across all 3 lessons is closed and small (26
unique conditions, mostly yes/no/number/keyword/silence/generic-attempt/
off-topic) — no embedding model needed. See golden_data.load_steps().
"""

import re

from golden_data import NONE_CONDITION, list_conditions

YES_WORDS = {"yes", "haan", "ha", "han", "y"}
NO_WORDS = {"no", "nahi", "nahin", "na", "n"}

NUMBER_WORDS = {
    "1": {"1", "one", "वन", "एक", "ek"},
    "2": {"2", "two", "टू", "दो", "do"},
    "3": {"3", "three", "थ्री", "तीन", "teen", "tin"},
    "4": {"4", "four", "फ़ोर", "चार", "char", "chaar"},
    "8": {"8", "eight", "aath", "aat"},
    "10": {"10", "ten", "das", "dus"},
}

GENERIC_CONDITIONS = {"child repeats", "child responds", "child tries"}
FRUIT_WORDS = {"mango", "banana", "aam", "kela", "आम", "केला"}

PLACEHOLDER_RE = re.compile(r"\{[^}]*\}")


def _tokens(text):
    return set(re.findall(r"[\wऀ-ॿ]+", text.lower()))


def _expected_phrase_words(canonical_prompt):
    """Most canonical prompts end with the exact phrase Doro wants echoed
    back. "बोलो" ("say") reliably precedes it throughout this script — more
    reliable than the last comma, since some prompts have a mid-sentence
    example before the real instruction (e.g. '...I am Doro. ...बोलो, I am
    {child's name}', where the last comma sits before "I am Doro", not
    before the actual target phrase)."""
    if not canonical_prompt:
        return None
    stripped = canonical_prompt.strip().rstrip("?.")
    idx = stripped.rfind("बोलो")
    if idx != -1:
        tail = stripped[idx + len("बोलो") :].lstrip(", ")
    else:
        comma_idx = stripped.rfind(",")
        if comma_idx == -1:
            return None
        tail = stripped[comma_idx + 1 :]
    phrase = PLACEHOLDER_RE.sub(" ", tail)
    words = _tokens(phrase)
    return words or None


def _find_condition(conditions, predicate):
    for c in conditions:
        if predicate(c.lower()):
            return c
    return None


def match(transcript, step):
    """Returns (condition, confident). confident=False means the regex layer
    couldn't decide and the caller should escalate to the SLM fallback."""
    conditions = [c for c in list_conditions(step) if c != NONE_CONDITION]
    stripped = transcript.strip()

    if not stripped:
        c = _find_condition(conditions, lambda lc: "silent" in lc)
        return (c or NONE_CONDITION), True

    tokens = _tokens(stripped)

    if tokens & YES_WORDS:
        c = _find_condition(conditions, lambda lc: "yes" in lc or lc == "child says Yes".lower())
        if c:
            return c, True
    if tokens & NO_WORDS:
        c = _find_condition(
            conditions, lambda lc: "no" in lc and "off-topic" not in lc and "one" not in lc
        )
        if c:
            return c, True

    for digit, words in NUMBER_WORDS.items():
        if tokens & {w.lower() for w in words}:
            c = _find_condition(conditions, lambda lc, d=digit: re.search(rf"\b{d}\b", lc))
            if c:
                return c, True
            # "one" -> "child says one or close" style wording
            c = _find_condition(conditions, lambda lc, d=digit: NUMBER_WORDS[d] & _tokens(lc))
            if c:
                return c, True
            # no numbered condition exists here (e.g. counting steps that
            # only have "child tries"/"silent") — saying any number word is
            # still strong evidence of a genuine attempt at the task.
            generic = _find_condition(conditions, lambda lc: lc in GENERIC_CONDITIONS)
            if generic:
                return generic, True

    if tokens & FRUIT_WORDS:
        c = _find_condition(conditions, lambda lc: "names a fruit" in lc)
        if c:
            return c, True

    # literal keyword conditions, e.g. "child says Mars" / "child says Bye"
    for c in conditions:
        lc = c.lower()
        m = re.match(r"child says (\w+)( or anything close)?$", lc)
        if m and m.group(1) not in ("yes", "no"):
            if m.group(1) in tokens:
                return c, True

    # "child echoed the requested phrase (fully or close)" — e.g. canonical
    # prompt ends "...बोलो, I am {child's name}", child says "I am Riya, i
    # love chocolate": extra chatter shouldn't break the match on the
    # required words actually being present.
    expected_words = _expected_phrase_words(step.get("canonical_prompt"))
    if expected_words:
        full_c = _find_condition(conditions, lambda lc: "full or close" in lc)
        partial_c = _find_condition(
            conditions, lambda lc: lc.startswith("child says only") or "only their name" in lc
        )
        if full_c and expected_words <= tokens:
            return full_c, True
        if partial_c and (tokens & expected_words):
            return partial_c, True

    # No specific rule fired: this is the ambiguous zone. If the step has a
    # generic "any attempt" bucket and no specific family, hand off to the
    # SLM to decide between that and off-topic rather than guessing here.
    generic = _find_condition(conditions, lambda lc: lc in GENERIC_CONDITIONS)
    has_specific_family = any(
        ("yes" in c.lower() or "no" in c.lower() or re.search(r"\d", c.lower())) for c in conditions
    )
    if generic and not has_specific_family:
        return generic, False

    # No generic bucket to weigh against off-topic, and every specific family
    # already failed to match: nothing ambiguous left to escalate.
    return NONE_CONDITION, True
