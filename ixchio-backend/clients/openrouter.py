"""
OpenRouter — free Llama 3.2 3B
Used as the third vote in consensus validation.
"""

import os
import aiohttp
from typing import List, Dict
from tenacity import retry, stop_after_attempt, wait_exponential


class OpenRouterClient:
    def __init__(self, rate_limiter, circuit_breaker):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.rate_limiter = rate_limiter
        self.circuit_breaker = circuit_breaker
        self.model = "meta-llama/llama-3.2-3b-instruct:free"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 2000) -> str:
        await self.rate_limiter.wait_if_needed("openrouter")

        async def _hit_api():
            async with aiohttp.ClientSession() as session:
                resp = await session.post(
                    self.base_url,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    },
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "https://ixchio.dev",
                        "X-Title": "ixchio Research",
                    },
                )
                if resp.status != 200:
                    body = await resp.text()
                    raise Exception(f"OpenRouter {resp.status}: {body}")
                data = await resp.json()
                return data["choices"][0]["message"]["content"]

        return await self.circuit_breaker.call(_hit_api)
