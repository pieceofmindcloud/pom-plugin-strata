"""The conversation sessions the bridge keeps for prompt prefixes: the names of a conversation's files, which saved
session starts a prompt, which slot holds a turn, and whether a slot's delta base is still the conversation's.

Plain data and no engine import, so it is tested without a GPU (runtime/test_pom_sessions.py). The bridge
(pom_strata_server.py) does the engine calls with these answers.

A conversation key K has these files in the session folder:
    K.strsess        the whole session (the state after its last turn, as this machine last saved it)
    K.ids.json       the tokens that whole session holds, as a JSON list
    K.strsdlt        the turn's delta: the K/V added after the base the peers hold (the base is the previous whole)
    K.dlt.json       {"base": [...], "ids": [...]}: the base's tokens and the tokens the delta brings
"""

from __future__ import annotations

import json
import os
from pathlib import Path

FULL = ".strsess"
DELTA = ".strsdlt"
IDS = ".ids.json"
DELTA_META = ".dlt.json"
# what a transfer names each file by: the peers ask for and send these kinds
KINDS = {"full": FULL, "delta": DELTA, "full_ids": IDS, "delta_meta": DELTA_META}
MAX_KEY = 96


def key_problem(key) -> str | None:
    """Why `key` cannot name a session (None: it can). A key is a file name stem: letters, digits, '-', '_' and '.'."""
    if not isinstance(key, str) or not key:
        return "must be a non-empty name"
    if len(key) > MAX_KEY:
        return f"must be at most {MAX_KEY} characters"
    if key.startswith(".") or any(not (c.isascii() and (c.isalnum() or c in "-_.")) for c in key):
        return "may hold letters, digits, '-', '_' and '.', and must not start with '.'"
    return None


def file_name(key: str, kind: str) -> str:
    """The file name of `key` for a transfer kind (see KINDS)."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    return key + KINDS[kind]


def prefix_len(prompt: list[int], ids: list[int]) -> int:
    """`len(ids)` when `ids` is a proper start of `prompt` (the prompt keeps a token to read), else 0."""
    n = len(ids)
    if 0 < n < len(prompt) and prompt[:n] == ids:
        return n
    return 0


def choose_restore(prompt: list[int], sessions: dict[str, list[int]], held: list[list[int]]) -> str | None:
    """The session key to restore for `prompt`: the one whose tokens start it most, when that is longer than what
    any slot already holds of it (`held`, the slots' tokens). Ties go to the lower key. None: nothing to restore."""
    best_key, best = None, max((prefix_len(prompt, h) for h in held), default=0)
    for key in sorted(sessions):
        n = prefix_len(prompt, sessions[key])
        if n > best:
            best_key, best = key, n
    return best_key


def slot_for_turn(prompt: list[int], held: list[list[int]]) -> int | None:
    """The slot that holds a turn after it ran: the one whose tokens start with the prompt, the most of them. None:
    no slot holds it (the turn was not kept)."""
    best, slot = None, None
    for index, tokens in enumerate(held):
        if len(tokens) >= len(prompt) and tokens[:len(prompt)] == prompt and (best is None or len(tokens) > best):
            best, slot = len(tokens), index
    return slot


def base_holds(tokens: list[int], base: list[int] | None) -> bool:
    """Whether a slot holding `tokens` still has the delta base `base`: the base's tokens start them. A slot that a
    different conversation has taken since fails this, so its engine base is never used for a delta."""
    return bool(base) and tokens[:len(base)] == base


def is_ids(value) -> bool:
    """Whether `value` is a list of token ids."""
    return isinstance(value, list) and all(type(x) is int and x >= 0 for x in value)


def read_ids(path: Path) -> list[int] | None:
    """The token list in a JSON file, or None when it is missing or not a list of token ids."""
    try:
        ids = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return ids if is_ids(ids) else None


def read_delta_meta(path: Path) -> tuple[list[int], list[int]] | None:
    """(base, ids) of a delta's metadata file, or None when it is missing or malformed."""
    try:
        meta = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict) or not is_ids(meta.get("base")) or not is_ids(meta.get("ids")):
        return None
    return meta["base"], meta["ids"]


def write_json(path: Path, value) -> None:
    """Writes `value` to `path` whole: a temporary file, then a rename, so a reader never sees half of it."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value))
    os.replace(tmp, path)


def read_sessions(folder: Path) -> dict[str, list[int]]:
    """Every whole session in `folder` that has both its file and its tokens: key -> tokens."""
    sessions: dict[str, list[int]] = {}
    if not folder.is_dir():
        return sessions
    for path in folder.glob("*" + IDS):
        key = path.name[:-len(IDS)]
        if key_problem(key) or not (folder / (key + FULL)).is_file():
            continue
        ids = read_ids(path)
        if ids:
            sessions[key] = ids
    return sessions
