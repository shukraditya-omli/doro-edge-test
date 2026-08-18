"""Gradio chat UI to walk the Doro script turn-by-turn.

Run: python scratch/app.py
"""

import gradio as gr

from golden_data import load_steps
from orchestration_prototype import fill_placeholders, run_turn
from reranker import get_reranker

STEPS = load_steps()

ORDERED_KEYS = sorted(
    (key for key, step in STEPS.items() if step["canonical_prompt"] is not None),
    key=lambda k: (k[0], k[1], k[2] if k[2] is not None else -1),
)


def start_chat(child_name):
    get_reranker()  # warm up on first load
    first_prompt = fill_placeholders(STEPS[ORDERED_KEYS[0]]["canonical_prompt"], child_name or "dost")
    history = [{"role": "assistant", "content": first_prompt}]
    return history, 0, history, "step 1/" + str(len(ORDERED_KEYS))


def respond(message, history, pointer, child_name):
    if pointer >= len(ORDERED_KEYS):
        return history, pointer, "Lesson finished — hit Restart to go again."

    step_key = ORDERED_KEYS[pointer]
    step = STEPS[step_key]
    history = history + [{"role": "user", "content": message}]

    if not step["conditions"]:
        # this step is Doro speaking (e.g. asking a free-text question like
        # the child's name) with no scripted branch to match against — no
        # reply to generate, just advance to the next line.
        debug = f"no branch conditions on this step, advancing  |  step {pointer + 1}/{len(ORDERED_KEYS)}"
    else:
        result = run_turn(
            STEPS, *step_key, transcript=message, history=[], child_name=child_name or "dost"
        )
        history = history + [{"role": "assistant", "content": result["text"]}]
        debug = (
            f"matched condition: {result['condition']}  |  "
            f"latency: {result['total_latency']:.3f}s  |  "
            f"step {pointer + 1}/{len(ORDERED_KEYS)}"
        )

    pointer += 1
    if pointer < len(ORDERED_KEYS):
        next_prompt = fill_placeholders(
            STEPS[ORDERED_KEYS[pointer]]["canonical_prompt"], child_name or "dost"
        )
        history = history + [{"role": "assistant", "content": next_prompt}]
    else:
        history = history + [{"role": "assistant", "content": "🎉 Lesson finished! Hit Restart to go again."}]

    return history, pointer, debug


with gr.Blocks(title="Doro script tester") as demo:
    gr.Markdown("# Doro conversational flow tester\nWalks the fixed lesson script turn-by-turn.")

    with gr.Row():
        child_name = gr.Textbox(label="Child's name", value="Riya", scale=1)
        restart_btn = gr.Button("Restart", scale=0)

    chatbot = gr.Chatbot(label="Conversation", height=450)
    debug_info = gr.Markdown()
    msg = gr.Textbox(label="Child's reply", placeholder="Type what the child says...")

    pointer_state = gr.State(0)

    def on_restart(name):
        history, pointer, _, _ = start_chat(name)
        return history, pointer, ""

    demo.load(fn=start_chat, inputs=[child_name], outputs=[chatbot, pointer_state, chatbot, debug_info])
    restart_btn.click(fn=on_restart, inputs=[child_name], outputs=[chatbot, pointer_state, debug_info])

    msg.submit(
        fn=respond,
        inputs=[msg, chatbot, pointer_state, child_name],
        outputs=[chatbot, pointer_state, debug_info],
    ).then(lambda: "", outputs=msg)

if __name__ == "__main__":
    demo.launch()
