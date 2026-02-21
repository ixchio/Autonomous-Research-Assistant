import pytest
import asyncio


@pytest.fixture
def event_loop():
    """fresh loop for each test — avoids the 'loop already running' nonsense"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()
