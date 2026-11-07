# Design Doc: Tiered OpenRouter LLM Routing (GLM Flash + Nemotron)

## Intent & Why
Optimize cost efficiency and classification quality in the CourtListener Case-Identity Triage system by routing Tier 2 LLM falsification queries through OpenRouter. High-volume, standard candidate pairs run through a fast and ultra-cheap model (GLM Flash), while complex, ambiguous, or high-risk pairs escalate to a deep reasoning model (Nemotron Ultra).

## Goals, Non-Goals, and Anti-Goals
- **Goals**:
  - Encapsulate all OpenRouter API handling behind the existing `analyze_with_agent(row)` interface.
  - Automatically classify candidate pairs into **simple** (GLM Flash) vs. **complex** (Nemotron Ultra) using precomputed structural features and conflict signals.
  - Fallback ladder: If GLM returns low confidence (<0.80) or invalid JSON, escalate to Nemotron; if OpenRouter is unreachable or unconfigured, fall back seamlessly to the deterministic `run_offline_falsification()`.
  - Expose clean environment variables (`OPENROUTER_API_KEY`, `OPENROUTER_MODEL_FAST`, `OPENROUTER_MODEL_COMPLEX`, `TRIAGE_LLM=openrouter`).
- **Non-Goals**:
  - Replacing Tier 1 (`HistGradientBoostingClassifier` + deterministic domain rules). Tier 1 remains the primary fast gate.
  - Modifying the held-out test split or changing the 8-class taxonomy.
- **Anti-Goals**:
  - Leaking API details, raw prompt plumbing, or network calls into CLI commands, Streamlit UI, or benchmarks.
  - Breaking test benches or offline execution when no API key is provided.

## Approaches Considered
1. **Direct Model Parameter in CLI/UI**: Require users or callers to specify `--model glm` or `--model nemotron`.
   - *Trade-off*: Exposes internal model details to users and makes automated batch processing inflexible. Rejected.
2. **Confidence-Only Cascade**: Run every single pair through GLM Flash first, only calling Nemotron on failure.
   - *Trade-off*: Slower for obviously complex/disputed pairs (e.g., page spans >= 2 or cross-court cert bug pairs) that need deep analysis anyway. Rejected.
3. **Feature & Risk-Based Routing with Fallback (Selected)**:
   - High-risk / complex pairs route directly to Nemotron.
   - Clean / repetitive merge pairs route to GLM Flash, escalating to Nemotron if GLM confidence is low or abstains.
   - All network calls encapsulated inside `triage/agent.py`.

## Architecture & Encapsulation
- **`is_complex_pair(row: Dict[str, Any]) -> Tuple[bool, str]`**:
  Checks structural indicators:
  - Disputed docket or court mismatch (`doc_conflict`, non-appellate different court)
  - Multi-cluster reporter page spans (`max_page_span >= 2`)
  - Substantial length disparity (`len_ratio < 0.35`) or large filing delta (`delta_days > 45`)
  - Explicit dispute / adjudicate flag or graph component contradiction
- **`OpenRouterClient` (Internal to `triage/agent.py`)**:
  - Endpoint: `https://openrouter.ai/api/v1/chat/completions`
  - Defaults:
    - Fast: `z-ai/glm-5.3-flash` (configurable via `OPENROUTER_MODEL_FAST`)
    - Complex: `nvidia/nemotron-4-340b-instruct` (configurable via `OPENROUTER_MODEL_COMPLEX`)
  - Handles JSON mode, headers (`HTTP-Referer`, `X-Title: CourtListener-Triage`), retries, and schema parsing.
- **Unified Public Interface**:
  - `analyze_with_agent(row: Dict[str, Any]) -> Dict[str, Any]` remains the sole entry point. It records which model tier handled the inference (`tier: "fast" | "complex" | "offline"` and `model_used: str`).
- **CLI & UI Wiring**:
  - CLI `classify` command respects `--agent openrouter` (in addition to `gemini`, `offline`, `none`).
  - Streamlit UI co-pilot invokes `analyze_with_agent` and displays which model tier provided the assessment.
