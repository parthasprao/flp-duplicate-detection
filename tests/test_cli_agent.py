"""Tests for CLI integration with OpenRouter agent mode."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from triage.cli import main
from triage.constants import LABEL_PROBABLE_DUPLICATE


def test_cli_classify_supports_openrouter_agent(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Synthetic candidate input
    input_file = tmp_path / "test_candidates.jsonl"
    out_file = tmp_path / "preds.jsonl"

    candidate_line = (
        '{"pair_id": "test-cli-1-2", "features": {"same_court": 1, "same_docket_number": 1, '
        '"docket_numbers_conflict": 0, "title_exact": 1, "text_length_ratio": 0.98, "min_hamming": 0, '
        '"shared_sha1": 1}, "cluster_low": {"cluster_id": 100, "case_name": "Case A v. B", "court_id": "ca9", '
        '"docket": {"docket_number": "12-345"}}, "cluster_high": {"cluster_id": 200, "case_name": "Case A v. B", '
        '"court_id": "ca9", "docket": {"docket_number": "12-345"}}}\n'
    )
    input_file.write_text(candidate_line, encoding="utf-8")

    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-openrouter-key")

    mock_llm_response = {
        "label": LABEL_PROBABLE_DUPLICATE,
        "confidence": 0.96,
        "falsification_finding": "None found after checking court, docket, dates, page span, and text",
        "supporting_evidence": ["Same case duplicate"],
        "contradicting_evidence": [],
        "maintainer_summary": "True duplicates confirmed.",
    }

    with patch("triage.agent.call_openrouter", return_value=mock_llm_response) as mock_openrouter:
        with patch(
            "sys.argv",
            [
                "triage",
                "classify",
                str(input_file),
                "--out",
                str(out_file),
                "--agent",
                "openrouter",
            ],
        ):
            main()

        assert out_file.exists()
        content = out_file.read_text(encoding="utf-8")
        assert "test-cli-1-2" in content
        assert os.environ.get("TRIAGE_LLM") == "openrouter"
        assert mock_openrouter.call_count >= 1
