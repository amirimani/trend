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

    def _api(self, method: str) -> str:
        return f"https://api.telegram.org/bot{self.token}/{method}"

    def send(self, text: str) -> bool:
        if not self.enabled:
            print("[TELEGRAM-DRYRUN]\n" + text + "\n")
            return False
        try:
            r = requests.post(
                self._api("sendMessage"),
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

    def get_updates(self, offset=None, timeout: int = 25) -> list[dict]:
        """Long-poll for incoming updates (commands). Returns the result list."""
        if not self.enabled:
            return []
        params = {"timeout": timeout}
        if offset is not None:
            params["offset"] = offset
        # HTTP timeout must exceed the long-poll timeout.
        r = requests.get(self._api("getUpdates"), params=params, timeout=timeout + 10)
        if r.status_code != 200:
            log.error("getUpdates failed %s: %s", r.status_code, r.text[:200])
            return []
        return r.json().get("result", [])

    def drain_updates(self):
        """Skip any backlog at startup so old commands aren't replayed. Returns
        the next offset to use."""
        backlog = self.get_updates(offset=None, timeout=0)
        if backlog:
            return backlog[-1]["update_id"] + 1
        return None

    def set_my_commands(self, menu: list[tuple[str, str]]) -> bool:
        """Register the command menu shown in the Telegram UI."""
        if not self.enabled:
            return False
        cmds = [{"command": c, "description": d} for c, d in menu]
        try:
            r = requests.post(self._api("setMyCommands"), json={"commands": cmds},
                              timeout=self.timeout)
            return r.status_code == 200
        except requests.RequestException as e:
            log.error("setMyCommands error: %s", e)
            return False
