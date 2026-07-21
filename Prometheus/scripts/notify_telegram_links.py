"""Send newly created dashboard URLs to an operator-owned Telegram chat.

Credentials are read exclusively from ``TELEGRAM_BOT_TOKEN`` and
``TELEGRAM_CHAT_ID``. They are never accepted as command-line arguments so they
do not appear in process listings or committed configuration.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from typing import Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen


def parse_link(value: str) -> tuple[str, str]:
    """Parse and validate one ``NAME=HTTPS_URL`` argument."""
    name, separator, url = value.partition("=")
    name = name.strip()
    url = url.strip()
    parsed = urlparse(url)
    if not separator or not name:
        raise ValueError("link must use NAME=HTTPS_URL format")
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("dashboard link must be an absolute HTTPS URL")
    return name, url


def build_message(links: Iterable[tuple[str, str]]) -> str:
    rows = list(links)
    if not rows:
        raise ValueError("at least one dashboard link is required")
    lines = ["Olympus dashboards are online", ""]
    lines.extend(f"{name}: {url}" for name, url in rows)
    lines.extend(["", f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}"])
    return "\n".join(lines)


def send_message(
    *,
    token: str,
    chat_id: str,
    message: str,
    opener: Callable[..., object] = urlopen,
    timeout: float = 15.0,
) -> None:
    """Send one plain-text Telegram message and validate the API response."""
    endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
    body = urlencode({"chat_id": chat_id, "text": message, "disable_web_page_preview": "true"}).encode()
    request = Request(endpoint, data=body, method="POST")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    response = opener(request, timeout=timeout)
    payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError("Telegram API did not confirm message delivery")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send dashboard URLs to Telegram")
    parser.add_argument("--link", action="append", default=[], metavar="NAME=HTTPS_URL")
    parser.add_argument("--dry-run", action="store_true", help="Print the message without contacting Telegram")
    args = parser.parse_args(argv)

    try:
        links = [parse_link(value) for value in args.link]
        message = build_message(links)
    except ValueError as exc:
        print(f"Telegram notification configuration error: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        print(message)
        return 0

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        print("Telegram notification skipped: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required.", file=sys.stderr)
        return 2

    try:
        send_message(token=token, chat_id=chat_id, message=message)
    except HTTPError as exc:
        print(f"Telegram request failed with HTTP status {exc.code}.", file=sys.stderr)
        return 1
    except (URLError, TimeoutError):
        print("Telegram request failed because the Telegram API was unreachable.", file=sys.stderr)
        return 1
    except (UnicodeError, json.JSONDecodeError, RuntimeError) as exc:
        print(f"Telegram delivery was not confirmed: {exc}", file=sys.stderr)
        return 1

    print(f"Telegram notification sent with {len(links)} dashboard link(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
