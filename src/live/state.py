"""
Persistent state for the multi-symbol live engine.

Schema (v2)
-----------
{
  "version": 2,
  "watchlist": {
     "BTC/USDT": {
        "enabled": true,
        "params":  {<Params fields>},
        "analyzed_at": "<iso>" | null,   # when /analyze last tuned it
        "analysis":    {<summary>} | null,
        "last_bar":    "<iso>" | null,   # last processed closed candle
        "position":    {<open position>} | null,
        "history":     [<closed trades>]
     },
     ...
  }
}

The file lives on the mounted Docker volume so the watchlist, per-symbol tuned
parameters, open positions and trade history all survive restarts.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict

from src.strategy import Params

STATE_FILE = os.getenv("STATE_FILE", "/data/state.json")


# --------------------------------------------------------------------------- #
# Params <-> dict
# --------------------------------------------------------------------------- #
def params_to_dict(p: Params) -> dict:
    return asdict(p)


def dict_to_params(d: dict) -> Params:
    fields = set(asdict(Params()))
    return Params(**{k: v for k, v in d.items() if k in fields})


# --------------------------------------------------------------------------- #
# Symbols
# --------------------------------------------------------------------------- #
def normalize_symbol(sym: str, default_quote: str = "USDT") -> str:
    """ 'sol' -> 'SOL/USDT', 'eth/usdt' -> 'ETH/USDT'. """
    s = (sym or "").strip().upper().replace("\\", "/")
    if not s:
        return s
    if "/" not in s:
        s = f"{s}/{default_quote}"
    return s


def new_entry(params: Params, enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "params": params_to_dict(params),
        "analyzed_at": None,
        "analysis": None,
        "last_bar": None,
        "position": None,
        "history": [],
    }


# --------------------------------------------------------------------------- #
# Load / save / migrate
# --------------------------------------------------------------------------- #
def load_state(path: str = STATE_FILE) -> dict:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_state(state: dict, path: str = STATE_FILE) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, default=str)
    os.replace(tmp, path)


def migrate(state: dict, default_symbol: str, default_params: Params,
            seed_symbols: list[str] | None = None) -> dict:
    """Bring any older/empty state up to v2 and seed the initial watchlist."""
    if state.get("version") == 2 and "watchlist" in state:
        wl = state["watchlist"]
    else:
        wl = {}
        # v1 had flat last_bar/position/history for a single (default) symbol.
        if any(k in state for k in ("last_bar", "position", "history")):
            entry = new_entry(default_params, enabled=True)
            entry["last_bar"] = state.get("last_bar")
            entry["position"] = state.get("position")
            entry["history"] = state.get("history", [])
            wl[default_symbol] = entry
        state = {"version": 2, "watchlist": wl}

    # Seed any requested symbols that aren't present yet.
    for sym in (seed_symbols or []):
        sym = normalize_symbol(sym)
        if sym and sym not in wl:
            wl[sym] = new_entry(default_params, enabled=True)
    return state


# --------------------------------------------------------------------------- #
# Watchlist operations (operate in place; caller persists)
# --------------------------------------------------------------------------- #
def watchlist(state: dict) -> dict:
    return state.setdefault("watchlist", {})


def add_symbol(state: dict, sym: str, params: Params) -> tuple[bool, str]:
    sym = normalize_symbol(sym)
    wl = watchlist(state)
    if not sym:
        return False, "نماد نامعتبر است."
    if sym in wl:
        return False, f"`{sym}` از قبل در واچ‌لیست هست."
    wl[sym] = new_entry(params, enabled=True)
    return True, sym


def remove_symbol(state: dict, sym: str) -> tuple[bool, str]:
    sym = normalize_symbol(sym)
    wl = watchlist(state)
    if sym in wl:
        del wl[sym]
        return True, sym
    return False, f"`{sym}` در واچ‌لیست نیست."


def set_enabled(state: dict, sym: str, enabled: bool) -> tuple[bool, str]:
    sym = normalize_symbol(sym)
    wl = watchlist(state)
    if sym not in wl:
        return False, f"`{sym}` در واچ‌لیست نیست."
    wl[sym]["enabled"] = enabled
    return True, sym
