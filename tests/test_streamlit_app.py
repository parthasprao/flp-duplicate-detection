"""UI integration tests with synthetic data only; no held-out corpus or network."""

import csv
import io
import json
from copy import deepcopy
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.streamlit_app import (
    DEMO_PATH,
    courtlistener_url,
    display,
    export_bytes,
    yes_no,
)

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def widget(elements, label):
    return next(element for element in elements if element.label == label)


@pytest.fixture
def app(monkeypatch, tmp_path):
    # An empty, explicitly selected directory forces the checked-in demo fallback.
    monkeypatch.setenv("TRIAGE_DATA_DIR", str(tmp_path))
    return AppTest.from_file(str(APP), default_timeout=30).run()


def assert_no_errors(app):
    assert not list(app.exception), [e.message for e in app.exception]


def test_demo_starts_without_private_data(app):
    assert_no_errors(app)
    assert any("Synthetic demo" in e.value for e in app.info)
    assert len(app.get("link_button")) == 0
    assert len(app.get("download_button")) == 2
    assert all(e.proto.disabled for e in app.get("download_button"))
    assert widget(app.button, "Previous").disabled
    assert widget(app.selectbox, "Candidate pair").value == "1001-1002"


def test_navigation_and_filters(app):
    widget(app.button, "Next").click().run()
    assert_no_errors(app)
    assert widget(app.selectbox, "Candidate pair").value != "1001-1002"
    widget(app.button, "Previous").click().run()
    assert widget(app.selectbox, "Candidate pair").value == "1001-1002"
    widget(app.selectbox, "Priority").select("CRITICAL").run()
    assert_no_errors(app)
    assert any("No pairs match" in e.value for e in app.info)


def test_decision_records_notes_survivor_and_downloads(app):
    widget(app.text_input, "Reviewer name").input("Carla").run()
    widget(app.text_area, "Review notes").input('Reviewed excerpt, "practice only".').run()
    widget(app.text_input, "Evidence cited (source link or passage)").input("Demo excerpt").run()
    widget(app.radio, "Proposed surviving cluster (for duplicate/version proposals only)").set_value(1002).run()
    widget(app.button, "Confirm Duplicate").click().run()
    assert_no_errors(app)
    decisions = app.session_state["decisions"]
    assert set(decisions) == {"1001-1002"}
    decision = decisions["1001-1002"]
    assert decision["reviewer"] == "Carla"
    assert decision["keep_cluster_id"] == 1002
    assert decision["delete_cluster_id"] == 1001
    assert any("Recorded: Probable duplicate" in e.value for e in app.success)
    assert all(not e.proto.disabled for e in app.get("download_button"))
    csv_data, json_data = export_bytes(list(decisions.values()))
    rows = list(csv.DictReader(io.StringIO(csv_data.decode())))
    assert len(rows) == 1
    assert rows[0]["reason"] == 'Reviewed excerpt, "practice only".'
    assert rows[0]["evidence_cited"] == "Demo excerpt"
    proposals = json.loads(json_data)
    assert len(proposals) == 1
    assert proposals[0]["dry_run"] is True
    assert proposals[0]["django_command_spec"]["target_cluster_id"] == 1002
    assert proposals[0]["audit_evidence"]["synthetic_demo"] is True
    widget(app.button, "Next").click().run()
    widget(app.button, "Previous").click().run()
    assert_no_errors(app)
    assert widget(app.text_area, "Review notes").value == 'Reviewed excerpt, "practice only".'
    assert widget(app.radio, "Proposed surviving cluster (for duplicate/version proposals only)").value == 1002


def test_blind_mode_hides_machine_outputs_and_keeps_initial_decision(app):
    widget(app.radio, "Review mode").set_value("Blind Review").run()
    assert_no_errors(app)
    assert widget(app.selectbox, "Priority").disabled
    assert widget(app.selectbox, "Predicted relationship").disabled
    assert not any(e.label == "Run Falsification Agent Check" for e in app.button)
    assert not any(e.label == "Machine recommendation and evidence" for e in app.expander)
    widget(app.button, "Defer").click().run()
    initial = app.session_state["initial_decisions"]["1001-1002"]
    assert initial["decision"] == "insufficient_evidence"
    assert initial["evidence"]["recommendation_visible"] is False
    assert initial["evidence"]["machine_evidence"] == {}
    widget(app.button, "Reveal machine recommendation").click().run()
    assert_no_errors(app)
    assert any(e.label == "Machine recommendation and evidence" for e in app.expander)
    widget(app.button, "Confirm Duplicate").click().run()
    assert app.session_state["initial_decisions"]["1001-1002"] == initial
    assert app.session_state["decisions"]["1001-1002"]["decision"] == "probable_duplicate"


def test_blind_mode_ignores_preexisting_prediction_filters(app):
    widget(app.selectbox, "Priority").select("CRITICAL").run()
    widget(app.radio, "Review mode").set_value("Blind Review").run()
    assert_no_errors(app)
    assert widget(app.selectbox, "Candidate pair").options == [
        "1001-1002", "1003-1004", "1005-1006",
    ]


def test_agent_output_persists_for_correct_pair(app):
    widget(app.button, "Run Falsification Agent Check").click().run()
    assert_no_errors(app)
    assert "1001-1002" in app.session_state["agent_results"]
    finding = app.session_state["agent_results"]["1001-1002"]["falsification_finding"]
    widget(app.text_area, "Review notes").input("Still inspecting").run()
    assert any(finding in e.value for e in app.info)
    widget(app.button, "Next").click().run()
    assert not any(finding in e.value for e in app.info)
    widget(app.button, "Previous").click().run()
    assert any(finding in e.value for e in app.info)


def test_unreviewed_filter_and_updated_decision(app):
    widget(app.button, "Defer").click().run()
    widget(app.button, "Keep Separate").click().run()
    assert_no_errors(app)
    assert app.session_state["decisions"]["1001-1002"]["decision"] == "related_decision"
    assert any("Recorded: Related but distinct decision" in e.value for e in app.success)
    widget(app.radio, "Review status").set_value("Unreviewed").run()
    assert "1001-1002" not in widget(app.selectbox, "Candidate pair").options
    assert_no_errors(app)


def test_live_file_family_and_dataset_switch(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIAGE_DATA_DIR", str(tmp_path))
    rows = [json.loads(line) for line in DEMO_PATH.read_text().splitlines()]
    extra = deepcopy(rows[0])
    extra["pair_id"] = "1002-1007"
    extra["cluster_low"]["cluster_id"] = 1002
    extra["cluster_high"]["cluster_id"] = 1007
    extra["features"].update(cluster_id_low=1002, cluster_id_high=1007)
    rows.append(extra)
    (tmp_path / "candidates.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert_no_errors(app)
    assert any(e.label == "Connected case family" for e in app.expander)
    family = app.dataframe[0].value
    assert not family["Prediction"].isna().any()
    widget(app.button, "Defer").click().run()
    assert app.session_state["decisions"]
    widget(app.selectbox, "Dataset").select(DEMO_PATH).run()
    assert_no_errors(app)
    assert not app.session_state["decisions"]
    assert any("Synthetic demo" in e.value for e in app.info)


def test_missing_metadata_and_empty_excerpts(monkeypatch, tmp_path):
    monkeypatch.setenv("TRIAGE_DATA_DIR", str(tmp_path))
    row = {
        "pair_id": "201-202",
        "cluster_low": {"cluster_id": 201},
        "cluster_high": {"cluster_id": 202},
        "features": {"cluster_id_low": 201, "cluster_id_high": 202},
    }
    (tmp_path / "candidates.jsonl").write_text(json.dumps(row))
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert_no_errors(app)
    assert next(e for e in app.metric if e.label == "Same court").value == "Not available"
    assert next(e for e in app.metric if e.label == "Docket status").value == "Not available"
    assert widget(app.text_area, "Record A excerpt").value == "No text excerpt available"


def test_exports_empty_and_keep_separate_only():
    csv_data, json_data = export_bytes([])
    assert list(csv.DictReader(io.StringIO(csv_data.decode()))) == []
    assert json.loads(json_data) == []
    csv_data, json_data = export_bytes([{
        "pair_id": "1001-1002", "decision": "related_decision",
    }])
    assert len(list(csv.DictReader(io.StringIO(csv_data.decode())))) == 1
    assert json.loads(json_data) == []


def test_safe_display_and_links():
    assert display(None) == display("") == "Not available"
    assert display(0) == "0"
    assert yes_no(None) == yes_no("") == "Not available"
    assert yes_no(0) == "No"
    assert yes_no("False") == "No"
    assert yes_no(1) == "Yes"
    assert courtlistener_url({}) is None
    assert courtlistener_url({"cluster_id": 123}, "javascript:alert(1)") == (
        "https://www.courtlistener.com/opinion/123/"
    )
    assert courtlistener_url({}, "https://evil.example/") is None
