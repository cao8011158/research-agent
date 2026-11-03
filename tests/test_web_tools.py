"""Offline Tavily adapter contract tests."""
import asyncio
import time

import httpx
import pytest

from research_agent.web_tools import WebTools


def _run(handler):
    async def scenario():
        async with httpx.AsyncClient(base_url="https://api.tavily.com",
                                     transport=httpx.MockTransport(handler)) as client:
            return await WebTools("secret", client=client).call(
                "web_search", {"query": "named topic 2025"}, deadline=time.monotonic() + 10)
    return asyncio.run(scenario())


def test_search_projects_ranked_bounded_results_and_fixed_request():
    def handler(request):
        assert request.method == "POST" and str(request.url) == "https://api.tavily.com/search"
        assert request.headers["Authorization"] == "Bearer secret"
        import json
        body = json.loads(request.content)
        assert body == {"query": "named topic 2025", "topic": "general", "search_depth": "basic",
                        "max_results": 5, "include_answer": False, "include_raw_content": False,
                        "auto_parameters": False}
        return httpx.Response(200, json={"results": [
            {"title": "First", "url": "https://one.example", "content": "a" * 2000, "score": .9,
             "raw_content": "omit"},
            {"title": "Second", "url": "https://two.example", "content": "b", "score": .8}]})
    result = _run(handler)
    assert [x["title"] for x in result["results"]] == ["First", "Second"]
    assert len(result["results"][0]["content"]) == 1600
    assert "raw_content" not in result["results"][0]


def test_empty_results_stay_empty():
    assert _run(lambda request: httpx.Response(200, json={"results": []})) == {"results": []}


def test_http_failure_has_actionable_status_without_credential_or_body():
    with pytest.raises(RuntimeError, match="Tavily search failed \\(HTTP 429\\)") as error:
        _run(lambda request: httpx.Response(429, text="secret internal failure"))
    assert "secret" not in str(error.value)
