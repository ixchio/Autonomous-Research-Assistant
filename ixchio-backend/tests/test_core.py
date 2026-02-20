import pytest
import asyncio
from main import SemanticCache, CircuitBreaker

@pytest.mark.asyncio
async def test_semantic_cache_basic():
    # threshold 0.99 so it only hits on exact match
    cache = SemanticCache(similarity_threshold=0.99)
    
    compute_calls = 0
    async def mock_compute():
        nonlocal compute_calls
        compute_calls += 1
        return {"result": 42}
        
    # 1st call -> miss
    res1, status1 = await cache.get_or_compute("Exact Query", mock_compute)
    assert status1 == "cache_miss"
    assert compute_calls == 1
    
    # 2nd call -> hit
    res2, status2 = await cache.get_or_compute("Exact Query", mock_compute)
    assert status2 == "cache_hit"
    assert compute_calls == 1  # No new compute

@pytest.mark.asyncio
async def test_circuit_breaker_transitions():
    cb = CircuitBreaker(failure_threshold=2, timeout=1)
    
    async def fail():
        raise ValueError("API Error")
        
    async def succeed():
        return "OK"
        
    # Error 1
    with pytest.raises(ValueError):
        await cb.call(fail)
    assert cb.state == "closed"
    
    # Error 2 -> OPEN
    with pytest.raises(ValueError):
        await cb.call(fail)
    assert cb.state == "open"
    
    # Error 3 -> Fast fail
    with pytest.raises(Exception, match="Circuit breaker OPEN"):
        await cb.call(succeed)
        
    # Wait to become half-open
    await asyncio.sleep(1.1)
    
    # Should enter half_open and succeed
    res = await cb.call(succeed)
    assert res == "OK"
    assert cb.state == "closed"
