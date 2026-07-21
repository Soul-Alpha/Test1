from __future__ import annotations

from urllib.parse import parse_qs

import pytest

from scripts.notify_telegram_links import build_message, parse_link, send_message


class _Response:
    def read(self) -> bytes:
        return b'{"ok": true, "result": {"message_id": 1}}'


def test_parse_link_requires_named_https_url() -> None:
    assert parse_link("Prometheus=https://example.trycloudflare.com") == (
        "Prometheus",
        "https://example.trycloudflare.com",
    )
    with pytest.raises(ValueError):
        parse_link("http://example.test")
    with pytest.raises(ValueError):
        parse_link("Prometheus=http://example.test")


def test_build_message_contains_all_links() -> None:
    message = build_message(
        [
            ("Prometheus", "https://prometheus.example"),
            ("Hermes", "https://hermes.example"),
        ]
    )
    assert "Olympus dashboards are online" in message
    assert "Prometheus: https://prometheus.example" in message
    assert "Hermes: https://hermes.example" in message


def test_send_message_posts_chat_and_text_without_logging_token() -> None:
    captured = {}

    def opener(request, *, timeout):
        captured["url"] = request.full_url
        captured["body"] = parse_qs(request.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    send_message(
        token="secret-token",
        chat_id="12345",
        message="dashboard links",
        opener=opener,
    )

    assert captured["url"].endswith("/botsecret-token/sendMessage")
    assert captured["body"]["chat_id"] == ["12345"]
    assert captured["body"]["text"] == ["dashboard links"]
    assert captured["body"]["disable_web_page_preview"] == ["true"]
