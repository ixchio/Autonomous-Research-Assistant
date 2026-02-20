"""
Circuit breaker — stops hammering a dead API.
States: closed (normal) → open (failing) → half_open (testing).
"""

import asyncio
from time import time


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.state = "closed"
        self.last_failure_time = 0

    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
            else:
                raise Exception(f"Circuit breaker OPEN — waiting {self.timeout}s cooldown")

        try:
            result = await func(*args, **kwargs)
            # success resets everything
            if self.state == "half_open":
                self.state = "closed"
            self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time()
            if self.failures >= self.failure_threshold:
                self.state = "open"
            raise e
