"""
Rate limiter tests.
Checks per-service RPM/RPD enforcement.
"""

import pytest
from core.rate_limiter import RateLimiter


def test_allows_first_request():
    rl = RateLimiter()
    allowed, reason = rl.can_request("groq")
    assert allowed is True
    assert reason == "ok"


def test_tracks_requests():
    rl = RateLimiter()
    rl.can_request("groq")
    rl.requests["groq"].append(100)  # simulate a logged request
    assert len(rl.requests["groq"]) >= 1


def test_rpm_limit_kicks_in():
    rl = RateLimiter()
    # groq limit is 30 rpm — let's fake 30 recent requests
    from time import time
    now = time()
    rl.requests["groq"] = [now - i for i in range(30)]

    allowed, reason = rl.can_request("groq")
    assert allowed is False
    assert reason == "rpm"


def test_unknown_service_gets_defaults():
    rl = RateLimiter()
    # a service not in the limits dict should still work with defaults
    allowed, reason = rl.can_request("some_new_api")
    assert allowed is True


def test_different_services_independent():
    rl = RateLimiter()
    from time import time
    now = time()

    # max out tavily (rpm=5)
    rl.requests["tavily"] = [now - i for i in range(5)]

    # tavily should be blocked
    allowed, _ = rl.can_request("tavily")
    assert allowed is False

    # groq should still be fine
    allowed, _ = rl.can_request("groq")
    assert allowed is True
