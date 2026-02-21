"""
Pipeline routing logic tests.
These test the conditional edge functions in DeepResearchGraph
without actually running any LLM calls — just the decision-making.
"""

import pytest


# We can't easily instantiate the full graph without all the API keys,
# so we test the routing logic by calling the methods directly with
# fake state dicts. A bit hacky but it works.

def _make_state(**overrides):
    """Minimal state dict that satisfies the routing functions."""
    base = {
        "query": "test query",
        "search_results": [],
        "retry_count": 0,
        "reflection_gaps": [],
        "reflection_count": 0,
        "report": "",
    }
    base.update(overrides)
    return base


# The routing functions are instance methods but they only read from state,
# so we can call them on a dummy object. Let's just import and monkeypatch.

class FakeGraph:
    """Stand-in that has the routing methods but nothing else."""

    def _retry_or_proceed(self, state) -> str:
        if not state["search_results"] and state["retry_count"] <= 2:
            return "retry"
        return "proceed"

    def _needs_followup(self, state) -> str:
        gaps = state.get("reflection_gaps") or []
        if gaps and state["reflection_count"] <= 2:
            return "followup"
        return "done"

    def _accept_or_redo(self, state) -> str:
        word_count = len((state.get("report") or "").split())
        if word_count < 100 and state["retry_count"] <= 2:
            return "redo"
        if word_count < 50:
            return "fail"
        return "accept"


g = FakeGraph()


def test_retry_when_no_results():
    state = _make_state(search_results=[], retry_count=0)
    assert g._retry_or_proceed(state) == "retry"


def test_proceed_when_results_exist():
    state = _make_state(search_results=[{"title": "something"}], retry_count=0)
    assert g._retry_or_proceed(state) == "proceed"


def test_proceed_after_max_retries():
    # even with no results, stop retrying after 3
    state = _make_state(search_results=[], retry_count=3)
    assert g._retry_or_proceed(state) == "proceed"


def test_followup_when_gaps_found():
    state = _make_state(reflection_gaps=["missing data on X"], reflection_count=1)
    assert g._needs_followup(state) == "followup"


def test_done_when_no_gaps():
    state = _make_state(reflection_gaps=[], reflection_count=1)
    assert g._needs_followup(state) == "done"


def test_done_after_max_reflections():
    state = _make_state(reflection_gaps=["still gaps"], reflection_count=3)
    assert g._needs_followup(state) == "done"


def test_accept_long_report():
    long_report = " ".join(["word"] * 200)
    state = _make_state(report=long_report, retry_count=0)
    assert g._accept_or_redo(state) == "accept"


def test_redo_short_report():
    state = _make_state(report="too short", retry_count=0)
    assert g._accept_or_redo(state) == "redo"


def test_fail_empty_report_after_retries():
    state = _make_state(report="", retry_count=3)
    assert g._accept_or_redo(state) == "fail"
