# Claude Code / Agent Instructions (`CLAUDE.md`)

Please read [`AGENTS.md`](./AGENTS.md) in full before making any edits or running commands.

## Quick Summary of Mandatory Guardrails
1. **Never touch `split == "test"`**: Only inspect, train on, or evaluate against `"training"` and `"validation"` splits. Use `triage.io.load_development_splits()` in Python and `python -m triage.cli bench --role <1|2|3|all>` on the CLI. Never pass `--unlock-test-set`.
2. **Never hardcode IDs or case names**: Do not put `pair_id`, `pair_uid` (`dev-*`, `cand-*`, `hn-*`, `adj-*`), or `cluster_id` literals in `triage/*.py`.
3. **Check `training` AND `validation` metrics on every change**:
   - Run `.venv/bin/pytest -v` (which runs `tests/test_anti_leakage.py` plus all 3 role test benches).
   - Run `.venv/bin/python3 -m triage.cli bench --role all` and verify that validation accuracy/F1 improves, generalization gap (`train_acc - val_acc`) stays `< 25%`, and hard-negative false merges do not increase.
4. **Role-Specific Entry Points**:
   - **Role 1 (Normalizer & Versions)**: Read [`AGENT_PROMPT_ROLE1_INGESTION_NORMALIZER.md`](./AGENT_PROMPT_ROLE1_INGESTION_NORMALIZER.md)
   - **Role 2 (Citations & Shared Pages)**: Read [`AGENT_PROMPT_ROLE2_CITATION_TOPOLOGY.md`](./AGENT_PROMPT_ROLE2_CITATION_TOPOLOGY.md)
   - **Role 3 (Graph Consistency & Lifecycle)**: Read [`AGENT_PROMPT_ROLE3_GRAPH_LIFECYCLE.md`](./AGENT_PROMPT_ROLE3_GRAPH_LIFECYCLE.md)
