"""
Tavily — AI-native search engine
Best for news, general queries, recent events.
"""

import os
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential


class TavilyClient:
    def __init__(self, rate_limiter, circuit_breaker):
        self.api_key = os.getenv("TAVILY_API_KEY")
        self.base_url = "https://api.tavily.com/search"
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def search(self, query: str, max_results: int = 5) -> dict:
        await self.rate_limiter.wait_if_needed("tavily")

        async def _hit_api():
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    self.base_url,
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                        "search_depth": "basic",
                    },
                )
                if resp.status != 200:
                    raise Exception(f"Tavily {resp.status}")
                return await resp.json()

        return await self.circuit_breaker.call(_hit_api)
