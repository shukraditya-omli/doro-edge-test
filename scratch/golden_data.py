"""Loads chapter1-golden-turns.json into (lesson_id, section_num, step_idx) steps."""

import json
from pathlib import Path

DATA_PATH = Path(__file__).parent.parent / "chapter1-golden-turns.json"

NONE_CONDITION = "none/off-topic"


def load_steps(path=DATA_PATH):
    rows = json.loads(Path(path).read_text())
    steps = {}
    for row in rows:
        key = (row["lesson_id"], row["section_num"], row.get("step_idx"))
        if row["condition"] == "canonical_prompt":
            step = steps.setdefault(key, {"canonical_prompt": None, "guidance_text": None, "conditions": []})
            step["canonical_prompt"] = row["doro_line"]
        elif row["condition"] == "off_topic_or_unclear_input":
            # lesson-level guidance row (section_num == 0), not a real step
            steps.setdefault(key, {"canonical_prompt": None, "guidance_text": None, "conditions": []})
            steps[key]["guidance_text"] = row["guidance_text"]
        else:
            step = steps.setdefault(key, {"canonical_prompt": None, "guidance_text": None, "conditions": []})
            step["conditions"].append({"condition": row["condition"], "reply": row["reply"]})
    return steps


def get_step(steps, lesson_id, section_num, step_idx):
    return steps[(lesson_id, section_num, step_idx)]


def list_conditions(step):
    return [c["condition"] for c in step["conditions"]] + [NONE_CONDITION]


def get_reply(step, condition):
    for c in step["conditions"]:
        if c["condition"] == condition:
            return c["reply"]
    return None


if __name__ == "__main__":
    steps = load_steps()
    key = ("lesson_20260603_105025", 4, 0)  # Magic Age Guessing Game
    step = get_step(steps, *key)
    print("canonical_prompt:", step["canonical_prompt"])
    print("conditions:", list_conditions(step))
