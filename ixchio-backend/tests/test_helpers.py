"""
Tests for the JSON extraction + query sanitizer.
These are the most-used helpers and the extract_json function
has to handle some truly cursed LLM output, so worth testing well.
"""

from core.helpers import extract_json, sanitize_query


# ---- extract_json ----

def test_plain_json():
    raw = '{"name": "test", "value": 42}'
    result = extract_json(raw)
    assert result["name"] == "test"
    assert result["value"] == 42


def test_json_wrapped_in_markdown():
    raw = 'Sure! Here is the result:\n```json\n{"status": "ok"}\n```\nHope that helps!'
    result = extract_json(raw)
    assert result["status"] == "ok"


def test_json_buried_in_text():
    # LLMs love to ramble before giving you the actual JSON
    raw = 'The analysis shows that {"score": 8.5, "valid": true} based on the data.'
    result = extract_json(raw)
    assert result["score"] == 8.5


def test_totally_broken_json():
    # worst case — regex scraping should still pull something useful
    raw = 'relevancy: 7, consistency: 8, overall good'
    result = extract_json(raw)
    # might not parse perfectly but shouldn't crash
    assert isinstance(result, dict)


def test_empty_string():
    result = extract_json("")
    assert result == {}


def test_nested_json():
    raw = '{"search_queries": [{"query": "AI safety", "type": "technical"}]}'
    result = extract_json(raw)
    assert len(result["search_queries"]) == 1
    assert result["search_queries"][0]["type"] == "technical"


# ---- sanitize_query ----

def test_sanitize_strips_newlines():
    assert "\n" not in sanitize_query("hello\nworld\r\n")


def test_sanitize_caps_length():
    long = "x" * 1000
    assert len(sanitize_query(long)) <= 500


def test_sanitize_preserves_normal_input():
    q = "impact of quantum computing on RSA encryption"
    assert sanitize_query(q) == q
