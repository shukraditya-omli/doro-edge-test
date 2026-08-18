"""Off-topic acknowledgment: no generation, keyword lexicon only.

Free generation for open-ended acknowledgment was tested and failed badly
(gibberish, dropped the redirect, stray emoji — see slm_architecture notes).
This stays deterministic: match known topic keywords to a canned warm Hindi
phrase, fall back to one generic phrase, then always redirect back to the
current question — matching GENERAL_RECOVERY_GUIDANCE in the script data
("acknowledge warmly and bring them back to the current step").
"""

import re

TOPIC_ACKNOWLEDGMENTS = [
    ({"cartoon", "chhota", "bheem", "doraemon", "tv", "shinchan", "motu", "patlu"},
     "वाह, ये तो मज़ेदार है!"),
    ({"chips", "khana", "bhookh", "chocolate", "icecream", "cream", "paratha", "mango", "banana"},
     "यम्मी! मुझे भी खाना पसंद है!"),
    ({"toy", "khilona", "game", "khelna", "ball", "gadi"},
     "अरे वाह, मज़ेदार लगता है!"),
    ({"tired", "sona", "neend", "sleepy"},
     "अच्छा, थोड़ा rest भी ज़रूरी है!"),
]

GENERIC_ACKNOWLEDGMENT = "हाहा, अच्छा!"


def _tokens(text):
    return set(re.findall(r"[\wऀ-ॿ]+", text.lower()))


def acknowledge(transcript):
    tokens = _tokens(transcript)
    for keywords, phrase in TOPIC_ACKNOWLEDGMENTS:
        if tokens & keywords:
            return phrase
    return GENERIC_ACKNOWLEDGMENT


def build_off_topic_reply(transcript, canonical_prompt):
    return f"{acknowledge(transcript)} {canonical_prompt}"
