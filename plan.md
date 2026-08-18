# Implementation Plan - SLM Orchestration (SmolLM2-135M)

This plan implements a "Planner & Speaker" orchestration pattern using the 135M SmolLM2 model. This pattern replaces a single 600M model with two specialized calls to a 135M model to achieve higher precision and lower latency on-device.

## Technical Handoff for Claude

### Goal
Implement a robust on-device RAG system where a tiny SLM (SmolLM2-135M) acts as both an **Intent Classifier** (The Planner) and a **Creative Generator** (The Speaker).

### Architecture
1. **Input**: Hindi/Hinglish transcript from child.
2. **Phase 1 (The Planner)**:
   - **Action**: Call LLM with `CLASSIFIER_PROMPT`.
   - **Input**: User transcript + List of Scenarios.
   - **Output**: Exactly one Scenario ID (e.g., `AGE_GAME_NO`).
3. **Phase 2 (The Retriever)**:
   - **Action**: JS-side lookup in `src/goldenSet.js`.
   - **Output**: The "Golden Response" string (expert-written Hindi).
4. **Phase 3 (The Speaker)**:
   - **Action**: Call LLM with `GLOBAL_MSG` + `GOLDEN_REFERENCE`.
   - **Input**: User transcript + Conversation History + Golden Response.
   - **Output**: Streaming child-friendly Hindi response.

### Component Specs
- **Model**: `SmolLM2-135M-Instruct-Q4_K_M.gguf` (served via llama-server on 127.0.0.1:8080).
- **Data**: `src/goldenSet.js` (array of objects with `context` and `response`).
- **Orchestrator**: `src/main.js` (JavaScript).

---

## Proposed Changes

### Phase 0: Local Python Prototyping (RECOMMENDED)
- Create `scratch/orchestration_prototype.py` using `llama-cpp-python`.
- Iterate on `CLASSIFIER_PROMPT` to ensure SmolLM2-135M can reliably output Scenario IDs from Hindi input.
- Verify `GLOBAL_MSG` adherence to the golden reference.

### Phase 1: Documentation
- Create `slm_architecture.md` in the project root to formalize this pattern.

### Phase 2: Android Implementation
- **AppConfig.java**: Update to SmolLM2-135M URL and stop sequences.
- **config.js**: Define `CLASSIFIER_PROMPT`.
- **main.js**:
  - Overhaul `sendMessage` to implement the two-call sequence.
  - Call 1: Non-streaming classification.
  - Call 2: Streaming generation with reference.

## Verification Plan
1. **Latency**: Total turn-around (Classification + Generation) < 2.0s.
2. **Accuracy**: >90% correct scenario identification in Python tests.
3. **Persona**: 100% Hindi (Devanagari) output.
