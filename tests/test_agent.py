"""Tests for Tier 2 OpenRouter tiered LLM falsification agent."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from triage.agent import (
    analyze_with_agent,
    call_openrouter,
    is_complex_pair,
    run_offline_falsification,
)
from triage.constants import (
    LABEL_CITATION_TOPOLOGY_UNVERIFIED,
    LABEL_PROBABLE_DUPLICATE,
    LABEL_PROBABLE_VERSION,
    LABEL_RELATED_DECISION,
    LABEL_UNRELATED,
)


def _make_pair(
    pair_id: str = "synthetic-1-2",
    doc_conflict: int = 0,
    same_doc: int = 1,
    same_court: int = 1,
    max_page_span: int = 1,
    len_ratio: float = 0.95,
    delta_days: int = 0,
) -> dict:
    return {
        "pair_id": pair_id,
        "features": {
            "docket_numbers_conflict": doc_conflict,
            "same_docket_number": same_doc,
            "same_court": same_court,
            "max_page_cluster_span": max_page_span,
            "min_page_cluster_span": max_page_span,
            "text_length_ratio": len_ratio,
            "filing_date_delta_days": delta_days,
            "min_hamming": 4,
            "shared_citations": 1,
        },
        "cluster_low": {
            "cluster_id": 101,
            "case_name": "Acme Corp v. Beta Inc",
            "court_id": "ca9",
            "docket": {"docket_number": "21-1234"},
            "date_filed": "2021-01-01",
        },
        "cluster_high": {
            "cluster_id": 102,
            "case_name": "Acme Corp v. Beta Inc",
            "court_id": "ca9",
            "docket": {"docket_number": "21-1234"},
            "date_filed": "2021-01-01",
        },
        "excerpt_low": "Affirmed. Opinion of the court.",
        "excerpt_high": "Affirmed. Opinion of the court.",
    }


def test_is_complex_pair_clean_pair() -> None:
    pair = _make_pair()
    is_complex, reason = is_complex_pair(pair)
    assert not is_complex
    assert "standard" in reason.lower() or "clean" in reason.lower()


def test_is_complex_pair_docket_conflict() -> None:
    pair = _make_pair(doc_conflict=1, same_doc=0)
    is_complex, reason = is_complex_pair(pair)
    assert is_complex
    assert "docket" in reason.lower()


def test_is_complex_pair_multi_page_span() -> None:
    pair = _make_pair(max_page_span=3, doc_conflict=1, same_doc=0)
    is_complex, reason = is_complex_pair(pair)
    assert is_complex
    assert "page span" in reason.lower()


def test_is_complex_pair_large_length_disparity() -> None:
    pair = _make_pair(len_ratio=0.15)
    is_complex, reason = is_complex_pair(pair)
    assert is_complex
    assert "length" in reason.lower()


def test_is_complex_pair_large_date_delta() -> None:
    pair = _make_pair(delta_days=180)
    is_complex, reason = is_complex_pair(pair)
    assert is_complex
    assert "date" in reason.lower()


def test_analyze_with_agent_offline_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_LLM", "offline")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    pair = _make_pair()
    res = analyze_with_agent(pair)
    assert res["tier"] == "offline"
    assert res["model_used"] == "offline_analyzer"
    assert "falsification_finding" in res


def test_offline_falsifier_routes_unverified_shared_citation_for_review() -> None:
    pair = _make_pair(doc_conflict=1, same_doc=0)
    result = run_offline_falsification(pair)
    assert result["label"] == LABEL_CITATION_TOPOLOGY_UNVERIFIED
    assert "page-occupancy" in result["falsification_finding"]


def test_analyze_with_agent_openrouter_fast_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_LLM", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")
    pair = _make_pair()

    mock_llm_response = {
        "label": LABEL_PROBABLE_DUPLICATE,
        "confidence": 0.95,
        "falsification_finding": "None found after checking court, docket, dates, page span, and text",
        "supporting_evidence": ["Exact same docket and court"],
        "contradicting_evidence": [],
        "maintainer_summary": "Records are true duplicates.",
    }

    with patch("triage.agent.call_openrouter", return_value=mock_llm_response) as mock_call:
        res = analyze_with_agent(pair)
        assert res["label"] == LABEL_PROBABLE_DUPLICATE
        assert res["tier"] == "fast"
        assert res["model_used"] == "z-ai/glm-5.3-flash"
        assert mock_call.call_count == 1
        assert (mock_call.call_args.kwargs.get("model") or mock_call.call_args[0][1]) == "z-ai/glm-5.3-flash"


def test_analyze_with_agent_openrouter_complex_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_LLM", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")
    pair = _make_pair(max_page_span=3, doc_conflict=1, same_doc=0)  # Complex

    mock_llm_response = {
        "label": LABEL_CITATION_TOPOLOGY_UNVERIFIED,
        "confidence": 0.92,
        "falsification_finding": "Page count is unavailable for the shared citation",
        "supporting_evidence": ["Page span ambiguity"],
        "contradicting_evidence": [],
        "maintainer_summary": "Distinct rulings printed on same page.",
    }

    with patch("triage.agent.call_openrouter", return_value=mock_llm_response) as mock_call:
        res = analyze_with_agent(pair)
        assert res["label"] == LABEL_CITATION_TOPOLOGY_UNVERIFIED
        assert res["tier"] == "complex"
        assert res["model_used"] == "nvidia/nemotron-4-340b-instruct"
        assert mock_call.call_count == 1
        assert (mock_call.call_args.kwargs.get("model") or mock_call.call_args[0][1]) == "nvidia/nemotron-4-340b-instruct"


def test_analyze_with_agent_openrouter_escalation_from_low_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRIAGE_LLM", "openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "mock-key")
    pair = _make_pair()  # Initially simple

    # Fast model returns low confidence (e.g. 0.65)
    fast_response = {
        "label": LABEL_PROBABLE_VERSION,
        "confidence": 0.65,
        "falsification_finding": "Uncertain date gap",
        "supporting_evidence": [],
        "contradicting_evidence": [],
        "maintainer_summary": "Uncertain version.",
    }
    # Nemotron escalates with high confidence
    complex_response = {
        "label": LABEL_PROBABLE_VERSION,
        "confidence": 0.89,
        "falsification_finding": "Slip opinion revised into published opinion",
        "supporting_evidence": ["Same case, revised text"],
        "contradicting_evidence": [],
        "maintainer_summary": "Confirmed version difference.",
    }

    with patch("triage.agent.call_openrouter", side_effect=[fast_response, complex_response]) as mock_call:
        res = analyze_with_agent(pair)
        assert res["label"] == LABEL_PROBABLE_VERSION
        assert res["confidence"] == 0.89
        assert res["tier"] == "complex"
        assert res["model_used"] == "nvidia/nemotron-4-340b-instruct"
        assert mock_call.call_count == 2
