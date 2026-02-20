"""
Jina AI — two tools in one:
  r.jina.ai  →  Reader (gives you clean markdown from any URL)
  s.jina.ai  →  Search Grounding (real-time facts with citations)
10M free tokens, 200 RPM.
"""

import os
import aiohttp
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential


class JinaClient:
    def __init__(self, rate_limiter, circuit_breaker):
        self.api_key = os.getenv("JINA_API_KEY")
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def read_url(self, url: str) -> str:
        """Scrape a page and get back clean markdown. No parsing headaches."""
        await self.rate_limiter.wait_if_needed("jina")

        async def _hit_api():
            async with aiohttp.ClientSession() as session:
                resp = await session.get(
                    f"https://r.jina.ai/{url}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "text/markdown",
                    },
                )
                if resp.status != 200:
                    raise Exception(f"Jina Reader {resp.status}")
                text = await resp.text()
                # cap it — some pages are absurdly long
                return text[:5000]

        return await self.circuit_breaker.call(_hit_api)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=8))
    async def search(self, query: str) -> List[Dict]:
        """Real-time web search with grounded facts."""
        await self.rate_limiter.wait_if_needed("jina")

        async def _hit_api():
            async with aiohttp.ClientSession() as session:
                resp = await session.get(
                    f"https://s.jina.ai/{query}",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "application/json",
                    },
                )
                if resp.status != 200:
                    raise Exception(f"Jina Search {resp.status}")

                data = await resp.json()
                hits = []
                for item in data.get("data", [])[:5]:
                    hits.append({
                        "title": item.get("title", ""),
                        "url": item.get("url", ""),
                        "content": item.get("content", "")[:500],
                        "score": item.get("score", 0),
                    })
                return hits

        return await self.circuit_breaker.call(_hit_api)
