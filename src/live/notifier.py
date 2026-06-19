"""Telegram notifier — sends messages via the Bot API (plain HTTP, no SDK)."""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger("notifier")


class TelegramNotifier:
    def __init__(self, token: str | None = None, chat_id: str | None = None, timeout: int = 15):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self.timeout = timeout
        self.enabled = bool(self.token and self.chat_id)
        if not self.enabled:
            log.warning("Telegram disabled (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set) "
                        "- messages will be printed to stdout instead.")

    def send(self, text: str) -> bool:
        if not self.enabled:
            print("[TELEGRAM-DRYRUN]\n" + text + "\n")
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            r = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=self.timeout,
            )
            if r.status_code != 200:
                log.error("Telegram send failed %s: %s", r.status_code, r.text[:300])
                return False
            return True
        except requests.RequestException as e:
            log.error("Telegram send error: %s", e)
            return False
