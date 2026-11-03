"""Bounded Tavily Search adapter for the agent tool interface."""
from __future__ import annotations

from collections.abc import Mapping
import math
import time

import httpx


WEB_SEARCH = {"type": "function", "function": {
    "name": "web_search",
    "description": "Search the public web for source titles, URLs, and short excerpts. Cite source URLs when using results.",
    "parameters": {"type": "object", "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 500,
                  "description": "Search query preserving the question's entities and time bounds"}},
        "required": ["query"], "additionalProperties": False}}}


class WebTools:
    tools = [WEB_SEARCH]

    def __init__(self, api_key: str, *, client: httpx.AsyncClient | None = None):
        if not api_key:
            raise ValueError("TAVILY_API_KEY is required for web search")
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(base_url="https://api.tavily.com")
        self._owns_client = client is None

    async def __aenter__(self):
        if self._owns_client:
            await self.client.__aenter__()
        return self

    async def __aexit__(self, *args):
        if self._owns_client:
            await self.client.__aexit__(*args)

    async def call(self, name: str, args: Mapping, *, deadline: float) -> object:
        if name != "web_search":
            raise ValueError("unknown web tool")
        query = args.get("query") if isinstance(args, Mapping) else None
        if not isinstance(query, str) or not query.strip() or len(query) > 500:
            raise ValueError("query must be a nonempty string of at most 500 characters")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("agent time budget exhausted")
        try:
            response = await self.client.post(
                "/search", headers={"Authorization": f"Bearer {self.api_key}"},
                json={"query": query, "topic": "general", "search_depth": "basic",
                      "max_results": 5, "include_answer": False,
                      "include_raw_content": False, "auto_parameters": False},
                timeout=min(30.0, remaining))
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(f"Tavily search failed (HTTP {exc.response.status_code})") from None
        except httpx.TimeoutException:
            raise TimeoutError("Tavily search timed out") from None
        except (httpx.RequestError, ValueError) as exc:
            raise RuntimeError(f"Tavily search failed ({type(exc).__name__})") from None
        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise RuntimeError("Tavily search returned an invalid results payload")
        results = []
        for item in payload["results"][:5]:
            if not isinstance(item, dict):
                continue
            score = item.get("score")
            results.append({"title": str(item.get("title") or "")[:300],
                            "url": str(item.get("url") or "")[:2048],
                            "content": str(item.get("content") or "")[:1600],
                            "score": score if isinstance(score, (int, float)) and not isinstance(score, bool)
                            and math.isfinite(score) else None})
        return {"results": results}
