"""
Circuit breaker tests.
The breaker should trip after N failures, block calls while open,
and let one through when the timeout expires (half-open state).
"""

import pytest
import asyncio
from time import time
from core.circuit_breaker import CircuitBreaker


@pytest.mark.asyncio
async def test_starts_closed():
    cb = CircuitBreaker(failure_threshold=3, timeout=1)
    assert cb.state == "closed"


@pytest.mark.asyncio
async def test_success_stays_closed():
    cb = CircuitBreaker(failure_threshold=3)

    async def ok():
        return "fine"

    result = await cb.call(ok)
    assert result == "fine"
    assert cb.state == "closed"
    assert cb.failures == 0


@pytest.mark.asyncio
async def test_opens_after_threshold():
    cb = CircuitBreaker(failure_threshold=2, timeout=60)

    async def kaboom():
        raise ValueError("nope")

    for _ in range(2):
        with pytest.raises(ValueError):
            await cb.call(kaboom)

    assert cb.state == "open"


@pytest.mark.asyncio
async def test_open_rejects_calls():
    cb = CircuitBreaker(failure_threshold=1, timeout=60)

    async def kaboom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await cb.call(kaboom)

    assert cb.state == "open"

    # next call should be rejected immediately without even calling the function
    with pytest.raises(Exception, match="Circuit breaker OPEN"):
        async def anything():
            return "should not reach here"
        await cb.call(anything)


@pytest.mark.asyncio
async def test_half_open_after_timeout():
    cb = CircuitBreaker(failure_threshold=1, timeout=0)  # 0s timeout = instant recovery

    async def kaboom():
        raise ValueError("nope")

    with pytest.raises(ValueError):
        await cb.call(kaboom)

    assert cb.state == "open"

    # wait a tiny bit so time() moves past last_failure_time + timeout
    await asyncio.sleep(0.05)

    async def ok():
        return "back online"

    # should transition to half_open and then back to closed on success
    result = await cb.call(ok)
    assert result == "back online"
    assert cb.state == "closed"
