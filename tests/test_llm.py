"""Run in Colab; mock HTTP only."""

import json
from unittest.mock import patch

from research_agent.llm import StrataClient


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return b'{"choices": [{"message": {"content": "ok"}}]}'


def test_chat_request_shape():
    client = StrataClient("http://127.0.0.1:8080/", "private")
    with patch("research_agent.llm.urlopen", return_value=Response()) as open_url:
        reply = client.chat([{"role": "user", "content": "hello"}], "strata")
    request = open_url.call_args.args[0]
    assert request.full_url == "http://127.0.0.1:8080/v1/chat/completions"
    assert json.loads(request.data)["messages"][0]["content"] == "hello"
    assert reply["choices"][0]["message"]["content"] == "ok"
