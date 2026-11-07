# OpenRouter Tiered LLM (GLM Flash + Nemotron) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Encapsulate OpenRouter tiered inference in Tier 2 of the CourtListener Case-Identity Triage system, dispatching simple repetitive pairs to GLM 5.3 Flash and escalating complex/risk pairs to Nemotron Ultra with seamless fallback.

**Architecture:** Encapsulate OpenRouter API calling, response parsing, and fallback logic completely inside `triage/agent.py`. A structural complexity analyzer (`is_complex_pair`) routes candidate pairs: clean pairs go to GLM Flash, and high-risk pairs (or low-confidence GLM returns) escalate to Nemotron Ultra. An offline deterministic analyzer handles failures or missing credentials.

**Tech Stack:** Python 3.11/3.12, `requests`, `pytest`, `unittest.mock`.

**Spec:** `docs/superpowers/specs/2026-11-07-openrouter-tiered-llm-design.md`

## Global Constraints
- `OPENROUTER_MODEL_FAST` defaults to `z-ai/glm-5.3-flash`
- `OPENROUTER_MODEL_COMPLEX` defaults to `nvidia/nemotron-4-340b-instruct`
- `analyze_with_agent(row)` is the single unified public interface; API details and HTTP mechanics are strictly hidden
- Never hardcode candidate pair IDs, UIDs, or cluster IDs (per `AGENTS.md`)
- The held-out `test` split remains strictly locked

## Review Focus
1. Missing `OPENROUTER_API_KEY`: Must fall back gracefully to `run_offline_falsification` without raising exceptions.
2. Malformed JSON from LLM: Must escalate from fast model to complex model, then to offline fallback if still unparseable.
3. GLM Flash confidence < 0.80: Must automatically escalate to Nemotron Ultra.
4. HTTP 429/timeout: Must handle gracefully and fall back to offline analyzer with clear maintainer note.
5. Preserving test-set lock: No test-set data ever passed or evaluated during unit tests.

---

### Task 1: Structural Complexity Classifier & OpenRouter Client in `triage/agent.py`

**Files:**
- Modify: `triage/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `extract_enriched_features(row)` from `triage/features.py`, `run_offline_falsification(row)` from `triage/agent.py`
- Produces:
  - `is_complex_pair(row: Dict[str, Any]) -> Tuple[bool, str]`
  - `call_openrouter(prompt: str, model: str, api_key: str, timeout: int = 20) -> Optional[Dict[str, Any]]`
  - Updated `analyze_with_agent(row: Dict[str, Any]) -> Dict[str, Any]` returning `model_used` and `tier`

- [ ] **Step 1: Write failing tests in `tests/test_agent.py`**
  Cover `is_complex_pair` detection of docket conflicts, multi-cluster page spans, length disparity, and standard simple pairs. Cover mocked OpenRouter success and fallback ladder.
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_agent.py -v`
- [ ] **Step 3: Implement `is_complex_pair`, `call_openrouter`, and updated `analyze_with_agent` in `triage/agent.py`**
  Add OpenRouter payload construction with system prompt and JSON format, complexity gating, escalation ladder, and offline fallback.
- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv/bin/pytest tests/test_agent.py -v`
- [ ] **Step 5: Commit**
  ```bash
  git add triage/agent.py tests/test_agent.py
  git commit -m "feat(agent): implement OpenRouter tiered inference with GLM Flash and Nemotron"
  ```

---

### Task 2: CLI Integration for OpenRouter Agent Mode

**Files:**
- Modify: `triage/cli.py`
- Test: `tests/test_cli_agent.py`

**Interfaces:**
- Consumes: `analyze_with_agent` from `triage/agent.py`
- Produces: CLI argument `--agent {offline,openrouter,gemini,none}` and proper `TRIAGE_LLM` environment handling in `run_classification_pipeline`

- [ ] **Step 1: Write failing CLI test in `tests/test_cli_agent.py`**
  Verify that `--agent openrouter` invokes the agent tier properly and records `agent_finding` and `model_used` without leaking API keys.
- [ ] **Step 2: Run test to verify it fails**
  Run: `.venv/bin/pytest tests/test_cli_agent.py -v`
- [ ] **Step 3: Update `triage/cli.py`**
  Add `"openrouter"` to the `--agent` choices and ensure `os.environ["TRIAGE_LLM"] = args.agent` is respected when executing the pipeline.
- [ ] **Step 4: Run test to verify it passes**
  Run: `.venv/bin/pytest tests/test_cli_agent.py -v`
- [ ] **Step 5: Commit**
  ```bash
  git add triage/cli.py tests/test_cli_agent.py
  git commit -m "feat(cli): add openrouter agent support in classify command"
  ```

---

### Task 3: Streamlit Workbench Co-Pilot Integration

**Files:**
- Modify: `app/streamlit_app.py`
- Test: `tests/test_anti_leakage.py` (verification)

**Interfaces:**
- Consumes: `analyze_with_agent(current_pair)` from `triage/agent.py`
- Produces: Streamlit UI display showing agent tier (`⚡ GLM Flash (Fast)` vs `🧠 Nemotron Ultra (Deep Reasoning)` vs `Offline`), model used, and falsification evidence.

- [ ] **Step 1: Update Streamlit button callback in `app/streamlit_app.py`**
  Change the manual button from `run_offline_falsification` to `analyze_with_agent`, and render badge/indicator of the model tier used.
- [ ] **Step 2: Verify existing tests pass and anti-leakage guards pass**
  Run: `.venv/bin/pytest tests/test_anti_leakage.py -v`
- [ ] **Step 3: Commit**
  ```bash
  git add app/streamlit_app.py
  git commit -m "feat(ui): update maintainer workbench to use tiered agent falsification"
  ```

---

### Task 4: Documentation & API Key Walkthrough

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: Configuration environment variables `OPENROUTER_API_KEY`, `OPENROUTER_MODEL_FAST`, `OPENROUTER_MODEL_COMPLEX`
- Produces: Comprehensive user walkthrough on how to obtain, set, and use the API key in local `.env`, shell, and CLI.

- [ ] **Step 1: Update `README.md` with Tier 2 OpenRouter Configuration section**
  Include exact commands for `export OPENROUTER_API_KEY=sk-or-v1-...` and running with `--agent openrouter`.
- [ ] **Step 2: Verify full pytest suite**
  Run: `.venv/bin/pytest -v`
- [ ] **Step 3: Commit**
  ```bash
  git add README.md
  git commit -m "docs: add OpenRouter tiered agent configuration and walkthrough"
  ```
