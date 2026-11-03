"""Small synchronous client for Strata's OpenAI-compatible local HTTP API."""

from __future__ import annotations

import json
from urllib.request import Request, urlopen


def _chat_payload(messages, model, tools, max_tokens, reasoning_effort, temperature, top_p, top_k):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens}
    for name, value in (("reasoning_effort", reasoning_effort), ("temperature", temperature),
                        ("top_p", top_p), ("top_k", top_k)):
        if value is not None:
            payload[name] = value
    if tools is not None:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    return payload


class StrataClient:
    def __init__(self, base_url: str, api_key: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def get(self, path: str) -> dict:
        return self._request(path)

    def chat(self, messages: list[dict], model: str, *, tools: list[dict] | None = None,
             max_tokens: int = 1024, reasoning_effort: str | None = None,
             temperature: float | None = None, top_p: float | None = None,
             top_k: int | None = None) -> dict:
        payload = _chat_payload(messages, model, tools, max_tokens, reasoning_effort,
                                temperature, top_p, top_k)
        return self._request("/v1/chat/completions", payload)

    def _request(self, path: str, payload: dict | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = None
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(self.base_url + path, data=body, headers=headers,
                          method="POST" if payload is not None else "GET")
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)


class AsyncStrataClient:
    """Cancellable HTTP client for bounded agent sessions; sync client remains stable."""

    def __init__(self, base_url: str, api_key: str, *, timeout: float = 180):
        import httpx
        self.api_key = api_key
        self.client = httpx.AsyncClient(base_url=base_url.rstrip("/"),
                                        headers={"Authorization": f"Bearer {api_key}"},
                                        timeout=timeout)

    async def __aenter__(self):
        await self.client.__aenter__()
        return self

    async def __aexit__(self, *args):
        await self.client.__aexit__(*args)

    async def get(self, path: str) -> dict:
        response = await self.client.get(path)
        response.raise_for_status()
        return response.json()

    async def chat(self, messages: list[dict], model: str, *, tools: list[dict] | None = None,
                   max_tokens: int = 1024, reasoning_effort: str | None = None,
                   temperature: float | None = None, top_p: float | None = None,
                   top_k: int | None = None) -> dict:
        payload = _chat_payload(messages, model, tools, max_tokens, reasoning_effort,
                                temperature, top_p, top_k)
        response = await self.client.post("/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()
