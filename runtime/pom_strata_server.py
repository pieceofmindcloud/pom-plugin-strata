"""Strata's own server, plus the route the POM pipeline uses.

The plugin starts this instead of `serve/server.py`. Everything Strata serves
stays as it is - its web app and monitor, `/health`, `/v1/*` - and one route is
added, without editing Strata's sources:

    POST /pom/generate
    {"prompt": "...", "chat_template": true, "thinking": true,
     "max_tokens": 512, "stop": ["..."], "temperature": 0.7, ...}

The POM gateway renders a conversation into one prompt text, the same text it
hands to the first peer of a llama.cpp pipeline, and that peer applies the
model's chat template to it as a single user turn after the fixed system turn.
This route does the same with Strata's template and tokenizer, runs the request
through Strata's own request path (the FIFO, the monitor, the reasoning and
stop handling) and streams the generated text back, one JSON object per line:

    {"t": "<text>"}               generated answer text, in order
    {"r": "<text>"}               generated reasoning, in order
    {"p": [read, total]}          prompt progress while the prompt is read
    {"done": {...}}               the end: finish, token counts (including reasoning_tokens),
                                  timings (with MTP draft_n / draft_n_accepted), stop_sequence
    {"error": "..."}              the request failed

Reasoning stays distinct from answer text all the way to the POM adapter, so
OpenAI-compatible clients receive `reasoning_content` instead of literal
`<think>` markup in `content`. Tools are not given to Strata's parser: the POM
renders its own tool block into the prompt and parses the calls out of answer
text. Strata's parser still takes the calls out of the output; they go back
into the text in the template's own form.

    GET /pom/status               loaded, model, context, busy (no prompts)

    POST /pom/session             a session file in the session folder (no paths: one file name); 501 unless the
                                  deployment's session_save setting is on (and then the session routes below too)
    {"action": "save" | "restore" | "save_delta" | "restore_delta", "filename": "conv-1.strsdlt"}
    -> {"tokens", "bytes", "ms"}; a delta is cut from, and applied to, the base the engine holds.

    GET /pom/session/offer?key=conv-1    -> {"tokens", "bytes", "ram_tokens", "layers", "pending"}: what this machine can
                                  give a peer (its file on disk, its layer split); a peer fetches the file in pieces with
                                  GET /pom/session/file?key=&kind=&offset=&len= (kind full, full_ids), and only when the
                                  layers match its own
    GET /pom/session/pending?key=conv-1   -> {"pending": bool}: the conversation's save is still to be written here
                                  (a turn's `done` line says `session.deferred`); the node syncs it after that

The POM's prefill/decode split (the deployment's `inference_role`, which keeps the sessions on): a prefill machine
gets the turn with `"prefill_only": true` and `"decode_holds"` (the tokens the decode machine already holds of the
conversation). It reads the prompt into a batch slot and keeps the session like any turn, sends no text, and its
`done` carries `"prefill": {"tokens"}`; or, when the part the decode machine lacks is shorter than
PREFILL_MIN_TOKENS, it reads nothing and answers `"prefill": {"skipped"}`. The decode machine then fetches the
session and restores it before its own turn (restore_for_prompt).

With `"pieces": true` the prefill machine reads the prompt POM_STRATA_PIECE_TOKENS tokens at a time and saves each
piece as it is read - the first whole, the next ones as deltas - under the keys `<session>.p1`, `.p2`, ...; it writes
`{"piece": {"key", "kind": "full" | "delta", "tokens", "bytes"}}` for each one once its files are written, and its
`done` carries `"prefill": {"tokens", "pieces"}`. The last piece stops one token short of the prompt. The decode
machine fetches each piece and applies it to one batch slot as it comes (POST /pom/session {"key", "action": "piece",
"piece", "kind"}, apply_piece), so only the last piece's transfer waits for the prompt to end.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
for extra in (HERE / "strata", HERE / "pylib"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from serve import server as S  # noqa: E402

import pom_sessions as P  # noqa: E402

# The system turn the POM's llama.cpp path puts in front of every prompt
# (llama-engine `apply_chat_template`). The same here, so a model answers the
# same prompt the same way whichever engine serves it.
SYSTEM_PROMPT = "You are a helpful assistant."
SESSION_ACTIONS = ("save", "restore", "save_delta", "restore_delta")


def session_dir() -> Path:
    """The folder session files live in: POM_STRATA_SESSIONS, else /data/strata/sessions."""
    path = Path(os.environ.get("POM_STRATA_SESSIONS", "/data/strata/sessions"))
    path.mkdir(parents=True, exist_ok=True)
    return path


# per batch slot: the conversation whose K/V the slot holds as its delta base, and the tokens of that base
slot_bases: dict[int, tuple[str, list[int]]] = {}

# The deployment's `session_save` setting (apply_pom_settings). Off, no session is saved or restored: the engine's
# session calls are not made, the session routes answer 501 and a generation runs as it did without sessions.
SESSIONS_ON = False
# the largest piece of a session file a peer reads in one request (the node relays pieces of this size)
SESSION_PIECE = 64 << 20

# A turn's save, deferred (SESSIONS_ON): per batch slot, the conversation, its prompt, the tokens the slot held when the
# turn ended, and when it was deferred. The slot keeps the turn until save_deferred writes it (guard_pick). A save
# not written within DEFER_MAX_S is dropped: a session is a cache, and the next turn re-reads its prompt.
deferred_saves: dict[int, tuple[str, list[int], list[int], float]] = {}
DEFER_MAX_S = 10.0
_saver_started = False
# the conversations whose deferred save is being written now (save_deferred); with deferred_saves, what
# session_pending reports
_writing: set[str] = set()

# The requests this bridge serves now (a generation or a session step, queued or running).  Counted here rather than
# read from Strata's status, which a request that ends early can leave saying busy.
_in_flight_lock = threading.Lock()
_in_flight = 0


@contextlib.contextmanager
def in_flight():
    global _in_flight
    with _in_flight_lock:
        _in_flight += 1
    try:
        yield
    finally:
        with _in_flight_lock:
            _in_flight -= 1


def in_flight_count() -> int:
    with _in_flight_lock:
        return _in_flight


# A conversation's session is kept (saved, and synced to the peers) only from this many prompt tokens: a shorter one
# re-reads its prompt faster than the save and the sync cost.
MIN_SESSION_TOKENS = 8000

# The deployment's prefill/decode role (`inference_role`: "prefill", "decode", or None for both; apply_pom_settings).
ROLE = None
# A prefill machine reads a prompt for the decode machine only when the part that machine lacks (the prompt less what
# it already holds of the conversation, `decode_holds`) has this many tokens: below it, the decode machine reads it
# itself, since sending the session (~118 MB of recurrent state, whatever the prompt) costs more than reading them.
# POM_STRATA_PREFILL_MIN_TOKENS in the prefill node's environment changes it (0: every turn with anything new).
PREFILL_MIN_TOKENS = int(os.environ.get("POM_STRATA_PREFILL_MIN_TOKENS", "4096"))
# A prefill machine's deferred save waits this long for an idle engine (DEFER_MAX_S otherwise): the decode machine
# waits for that save, and a queue of prompts keeps the engine busy.
PREFILL_DEFER_MAX_S = 300.0

DEFAULT_CACHE_MB = 5120      # the session folder's size limit on this machine (POM_STRATA_SESSION_CACHE_MB)
DEFAULT_MAX_AGE_H = 24       # a conversation not used for this long leaves the folder (POM_STRATA_SESSION_MAX_AGE_H)


def prune_cache() -> None:
    """Keeps this machine's session folder within its size limit and drops the conversations not used for a day.  A
    conversation a slot holds as its delta base is kept.  Runs after each session file is written here."""
    limit = int(float(os.environ.get("POM_STRATA_SESSION_CACHE_MB", DEFAULT_CACHE_MB)) * (1 << 20))
    max_age = float(os.environ.get("POM_STRATA_SESSION_MAX_AGE_H", DEFAULT_MAX_AGE_H)) * 3600
    keep = {key for key, _ in slot_bases.values()}
    try:
        removed = P.prune(session_dir(), limit, max_age, keep, time.time())
    except OSError as e:
        print(f"pom: session cache not pruned: {e}", file=sys.stderr, flush=True)
        return
    if removed:
        print(f"pom: session cache: removed {', '.join(removed)}", flush=True)


def session_path(key: str, kind: str) -> str:
    return str(session_dir() / P.file_name(key, kind))


def engine_session(svc, action: str, key: str, slot: int, ids: list[int] | None = None, locked: bool = True,
                   in_turn: bool = False) -> dict:
    """One keyed session step on a batch slot: the whole session, or its delta (`*_delta`).  `locked`: under the service
    FIFO.  A turn's own save runs inside the turn's `run`, which holds the FIFO, so it passes `locked=False`: the
    engine has finished the turn by then (its slot is free), and nothing else is sent meanwhile."""
    kind = "delta" if action.endswith("_delta") else "full"
    path = session_path(key, kind)
    extra = {"in_turn": True} if in_turn else {}
    if not locked:
        return svc.engine.session_file(action, path, slot=slot, ids=ids, **extra)
    with svc.fifo:
        return svc.engine.session_file(action, path, slot=slot, ids=ids, **extra)


def batch_engine(svc):
    """The engine that owns the batch slots (`svc.engine`: StrataEngine keeps batch, slot_cv, slot_busy, slot_held and
    pick_slot), or None when it runs no batch. The service itself has none of them."""
    engine = getattr(svc, "engine", None)
    return engine if engine is not None and getattr(engine, "batch", 0) else None


def restore_for_prompt(svc, ids: list[int]) -> None:
    """Before a turn: a saved session that starts the prompt further than any slot holds it is restored into a free
    slot, so the engine reads only the rest.  A session that cannot be restored leaves the slots as they were."""
    engine = batch_engine(svc)
    if engine is None or not svc.loaded():
        return
    sessions = P.read_sessions(session_dir())
    if not sessions:
        return
    with engine.slot_cv:
        key = P.choose_restore(ids, sessions, engine.slot_held)
        slot = engine.pick_slot(ids) if key is not None else None
    if key is None or slot is None:
        # the test's marker: the prompt is read again from its first token (no saved start of it on disk, or no slot)
        print(f"pom: session restore: none of {len(sessions)} saved sessions starts this prompt "
              f"({len(ids)} tokens): the prompt is read again", file=sys.stderr, flush=True)
        return
    try:
        engine_session(svc, "restore", key, slot, ids=sessions[key])
    except (S.SessionRefused, S.EngineDied, ValueError) as e:
        print(f"pom: session {key} not restored into slot {slot}: {e}", file=sys.stderr, flush=True)
        return
    slot_bases[slot] = (key, sessions[key])
    # the test's marker: the saved start came from this machine's disk into a batch slot; the engine reads the rest
    print(f"pom: session {key} restored from disk into slot {slot}: {P.reuse_len(ids, sessions[key])} of {len(ids)} "
          f"prompt tokens", file=sys.stderr, flush=True)


def save_turn(svc, key: str, prompt: list[int], locked: bool = True, held: list[int] | None = None,
              slot: int | None = None, in_turn: bool = False, wait: bool = True) -> dict | None:
    """A turn that ran in a batch slot: the slot saves the conversation under `key`.  When the slot's base is still this
    conversation's, the delta from that base is cut first (the peers apply it to their whole session), then the whole
    session is saved, which becomes the next base.  `slot`/`held` come from the engine's turn hook (`in_turn`: the
    slot is still the turn's own, its window is open).  Without them the slot is found by its held tokens.
    `wait=False`: a save the engine answers SAVING returns at once, with `pending` (its ticket) and `finish` (the
    bookkeeping, to call once the file is written); the slot is free by then.
    None: no slot holds the turn."""
    engine = batch_engine(svc)
    if engine is None:
        print(f"pom: session {key}: the engine runs no batch slots, not saved", file=sys.stderr, flush=True)
        return None
    deadline = time.monotonic() + 5
    with engine.slot_cv:
        while time.monotonic() < deadline and any(b for x, b in enumerate(engine.slot_busy) if x != slot):
            engine.slot_cv.wait(0.05)
        if slot is None:
            slot = P.slot_for_turn(prompt, engine.slot_held)
            held = list(engine.slot_held[slot]) if slot is not None else []
    if slot is None:
        return None
    held = list(held or [])
    base = slot_bases.get(slot)
    base_ids = base[1] if base and base[0] == key and P.base_holds(held, base[1]) else None
    folder = session_dir()
    delta = False
    if base_ids is not None:
        try:
            engine_session(svc, "save_delta", key, slot, locked=locked, in_turn=in_turn)
            delta = True
        except S.SessionRefused as e:
            print(f"pom: session {key}: no delta this turn: {e}", file=sys.stderr, flush=True)
    whole = engine_session(svc, "save", key, slot, locked=locked, in_turn=in_turn)
    ticket = whole.get("pending")                   # SAVING: the engine copied the state, its file is still written
    finish = lambda: _finish_turn_save(svc, key, slot, delta, base_ids, folder, whole["ids"], ticket)
    if ticket is not None and not wait:
        return {"ids": whole["ids"], "delta": delta, "pending": ticket, "finish": finish}
    return finish()


def _finish_turn_save(svc, key: str, slot: int, delta: bool, base_ids, folder, ids: list[int], ticket) -> dict:
    """What a turn's save leaves behind, once its file is written: the ids and the base the peers hold. A file the
    engine did not write raises (SessionRefused / EngineDied), and then the base stays as it was."""
    if ticket is not None:
        batch_engine(svc).await_session_file(ticket)
    P.write_json(folder / P.file_name(key, "full_ids"), ids)
    if delta:
        P.write_json(folder / P.file_name(key, "delta_meta"), {"base": base_ids, "ids": ids})
    else:
        for kind in ("delta", "delta_meta"):
            (folder / P.file_name(key, kind)).unlink(missing_ok=True)
    slot_bases[slot] = (key, ids)
    prune_cache()
    return {"ids": ids, "delta": delta}


def guard_pick(engine) -> None:
    """Keeps a deferred save's slot for its conversation: while the save is pending, the engine's pick_slot does not
    see the slot as free, unless the conversation's next prompt starts with the tokens the slot holds (then the slot
    is taken, and that save is superseded by the turn's own). Installed once on the engine; the caller holds slot_cv,
    as the caller of pick_slot does."""
    if getattr(engine, "_pom_guarded", False):
        return
    original = engine.pick_slot

    def pick_slot(prompt: list[int]) -> int | None:
        marked = []
        for slot, (_, _, held, _) in deferred_saves.items():
            if engine.slot_busy[slot]:
                continue
            if len(held) < len(prompt) and list(prompt[:len(held)]) == held:
                continue
            engine.slot_busy[slot] = True
            marked.append(slot)
        try:
            slot = original(prompt)
        finally:
            for kept in marked:
                engine.slot_busy[kept] = False
        if slot is not None and slot in deferred_saves:
            key = deferred_saves.pop(slot)[0]
            print(f"pom: session {key}: its slot takes the next turn, its save is superseded", file=sys.stderr,
                  flush=True)
        return slot

    engine.pick_slot = pick_slot
    engine._pom_guarded = True


def keep_prefill(svc, key: str, prompt: list[int], wait_s: float = 10.0) -> bool:
    """A prefill turn's session when the engine's turn hook did not run: its one token is all the turn asks for, so the
    consumer stops before the slot's BDONE and the engine frees the slot in the background, holding the prompt. That
    slot is found once it is free and its save is deferred as a turn's would be. False: no slot holds the prompt."""
    engine = batch_engine(svc)
    if engine is None:
        return False
    deadline = time.monotonic() + wait_s
    with engine.slot_cv:
        while True:
            slot = P.slot_for_turn(list(prompt), engine.slot_held)
            if slot is not None and not engine.slot_busy[slot]:
                held = list(engine.slot_held[slot])
                break
            left = deadline - time.monotonic()
            if left <= 0:
                # what each slot held when no slot held the prompt: busy (still being released), and how many of its
                # tokens agree with the prompt from the start
                def agree(tokens: list[int]) -> int:
                    n = 0
                    for a, b in zip(tokens, prompt):
                        if a != b:
                            break
                        n += 1
                    return n
                state = "; ".join(f"slot {s}: busy={bool(engine.slot_busy[s])} held={len(h)} agree={agree(list(h))}"
                                  for s, h in enumerate(engine.slot_held))
                print(f"pom: prefill of session {key}: no free slot holds its {len(prompt)} prompt tokens after "
                      f"{wait_s:.0f} s ({state})", file=sys.stderr, flush=True)
                return False
            engine.slot_cv.wait(min(0.1, left))
    return defer_save(svc, key, list(prompt), slot, held)


def defer_save(svc, key: str, prompt: list[int], slot: int, held: list[int]) -> bool:
    """A turn that ended in a batch slot, saved later: the slot keeps the turn (guard_pick) and save_deferred writes it
    while the engine is idle. Nothing is read or written here, so the turn returns at once. True: deferred."""
    global _saver_started
    engine = batch_engine(svc)
    if engine is None:
        return False
    with engine.slot_cv:
        deferred_saves[slot] = (key, list(prompt), list(held), time.monotonic())
        guard_pick(engine)
        engine.slot_cv.notify_all()
    if not _saver_started:
        _saver_started = True
        threading.Thread(target=_saver_loop, args=(svc,), name="pom-session-saver", daemon=True).start()
    return True


# The prefill/decode split's pieces: a prefill machine asked for `pieces` reads the prompt this many tokens at a time and
# saves each piece as it is read, so the decode machine fetches and restores it while the next one is read.
PIECE_TOKENS = int(os.environ.get("POM_STRATA_PIECE_TOKENS", "4096"))
# the slot each conversation's pieces are restored into on the decode machine, and the tokens it holds
piece_slots: dict[str, tuple[int, list[int]]] = {}


def save_live(svc, action: str, key: str) -> dict:
    """The engine's main session (the one a request just read, outside the batch slots) saved under `key`: whole, or as
    the delta from what it saved last (`save_delta`). Returns once the file is written: a save answered SAVING is
    awaited here, on this thread, so no other reader of the engine's session answers races for its SAVED."""
    path = session_path(key, "delta" if action.endswith("_delta") else "full")
    with svc.fifo:
        result = svc.engine.session_file(action, path)
        if result.get("pending"):
            ticket = {"path": path, "slot": None, "result": None, "error": None, "done": False}
            svc.engine.save_pending = ticket
            result = svc.engine.await_session_file(ticket)
    return result


def prefill_pieces(svc, key: str, ids: list[int], sampling: dict, cancel, line) -> dict:
    """A prefill machine's turn read in pieces (the prefill/decode split, `"pieces": true`): each piece is the prompt up
    to one of P.piece_ends, read as a request of one token (the engine's main session then holds exactly that start),
    and saved under the conversation's piece key - the first whole, every later one as the delta from the one before.
    Each saved piece is announced at once (`line({"piece": ...})`), so the decode machine fetches it while the next one
    is read. The last piece stops one token short of the prompt: the decode machine reads that token itself."""
    folder = session_dir()
    ends = P.piece_ends(len(ids), PIECE_TOKENS)
    if not ends:
        raise ValueError("the prompt is too short to be read in pieces")
    base: list[int] | None = None
    reused, prompt_ms = None, 0.0
    for index, end in enumerate(ends, 1):
        piece = P.piece_key(key, index)
        if piece is None:
            raise ValueError(f"session {key}: its piece keys would be too long")
        prompt = list(ids[:end])
        done = None
        for kind, x in svc.run(prompt, False, None, 1, sampling, cancel):
            if kind == "ping":
                progress = getattr(svc.engine, "progress", None)
                line({"p": list(progress)} if progress else {"p": None})
            elif kind == "done":
                done = x
        if cancel.is_set() or done is None:
            raise ValueError(f"session {key}: the prefill stopped at piece {index}")
        if reused is None:
            reused = done.get("reused") or 0
        prompt_ms += float((done.get("timings") or {}).get("prompt_ms") or 0.0)
        whole = base is None
        result = save_live(svc, "save" if whole else "save_delta", piece)
        if whole:
            P.write_json(folder / P.file_name(piece, "full_ids"), prompt)
        else:
            P.write_json(folder / P.file_name(piece, "delta_meta"), {"base": base, "ids": prompt})
        line({"piece": {"key": piece, "kind": "full" if whole else "delta", "tokens": end,
                        "bytes": result.get("bytes")}})
        base = prompt
    print(f"pom: prefill of session {key}: {len(ids)} prompt tokens read in {len(ends)} pieces "
          f"({reused} already held)", file=sys.stderr, flush=True)
    prune_cache()
    return {"pieces": len(ends), "reused": reused or 0, "prompt_ms": prompt_ms}


def apply_piece(svc, key: str, piece: str, kind: str) -> dict:
    """The decode machine's side of a piece (fetched into the session folder): the first one (`full`) is restored into a
    free batch slot, every later one (`delta`) onto that slot, which must still hold the piece before it. The piece's
    files are removed once the slot holds it. -> {"tokens", "slot"}."""
    if P.key_problem(piece) or not piece.startswith(key + ".p"):
        raise ValueError("the piece is not one of this session's")
    engine = batch_engine(svc)
    if engine is None:
        raise ValueError("the engine runs no batch slots")
    folder = session_dir()
    if kind == "full":
        ids = P.read_ids(folder / P.file_name(piece, "full_ids"))
        if ids is None:
            raise ValueError(f"no tokens for piece {piece}")
        with engine.slot_cv:
            slot = engine.pick_slot(ids)
        if slot is None:
            raise ValueError("no free batch slot for the piece")
        engine_session(svc, "restore", piece, slot, ids=ids)
    elif kind == "delta":
        meta = P.read_delta_meta(folder / P.file_name(piece, "delta_meta"))
        held = piece_slots.get(key)
        if meta is None:
            raise ValueError(f"no tokens for piece {piece}")
        base, ids = meta
        if held is None or held[1] != base or list(engine.slot_held[held[0]]) != base:
            raise ValueError(f"no slot holds the piece before {piece}")
        slot = held[0]
        engine_session(svc, "restore_delta", piece, slot, ids=ids)
    else:
        raise ValueError("a piece is full or delta")
    piece_slots[key] = (slot, list(ids))
    slot_bases.pop(slot, None)          # the slot's next save is whole: no peer holds a base of it
    for name in (P.file_name(piece, k) for k in ("full", "full_ids", "delta", "delta_meta")):
        (folder / name).unlink(missing_ok=True)
    return {"tokens": len(ids), "slot": slot}


ENGINE_ARGS: list[str] = []     # the engine's arguments as this bridge wrote them (session_config)


def session_config(svc) -> str:
    """The engine settings a saved session is bound to, as one string: Strata refuses to restore a file saved with
    another configuration (its config fingerprint: engine version, backend, --kv, --max-context, --kv-resident, the
    MTP window, STRATA_KV_ROT; conversation_file.cpp). A peer compares it before it fetches a session. The model's own
    settings (rope) and the arithmetic switches are not in it: the same model and runtime give the same ones."""
    def arg(name: str, default: str) -> str:
        if name in ENGINE_ARGS:
            at = ENGINE_ARGS.index(name)
            if at + 1 < len(ENGINE_ARGS):
                return ENGINE_ARGS[at + 1]
        return default
    info = getattr(batch_engine(svc), "info", None) or {}
    try:
        backend = json.loads((Path(__file__).parent / "RUNTIME.json").read_text()).get("backend") or "cuda"
    except (OSError, ValueError):
        backend = "cuda"
    mtp = arg("--mtp-window", "32768") if "--mtp" in ENGINE_ARGS else "-1"
    rot = "1" if os.environ.get("STRATA_KV_ROT", "").startswith("1") else "0"
    return (f"engine={info.get('engine') or info.get('version') or '?'} backend={backend} kv={arg('--kv', info.get('kv') or '?')} "
            f"max_context={arg('--max-context', str(info.get('context') or '?'))} "
            f"kv_resident={arg('--kv-resident', str(info.get('kv_resident') or '?'))} mtp_window={mtp} kv_rot={rot}")


def session_offer(svc, key: str) -> dict:
    """What this machine can give a peer for a conversation: `tokens` of its saved session on disk (the only copy a
    peer can fetch), `bytes` of that file, `ram_tokens` the batch slot holds in RAM, `layers` the layer split the engine
    runs ("whole" on a whole model: a session is only usable on the same split), `config` the engine settings its
    sessions are bound to (session_config), and `pending` while its save is still
    being written. Nothing is read beyond the token count and the file's size."""
    engine = batch_engine(svc)
    ram = 0
    if engine is not None:
        with engine.slot_cv:
            for k, ids in slot_bases.values():
                if k == key:
                    ram = max(ram, len(ids))
    folder = session_dir()
    ids_path, full = folder / P.file_name(key, "full_ids"), folder / P.file_name(key, "full")
    ids = P.read_ids(ids_path) if ids_path.is_file() else None
    on_disk = bool(ids) and full.is_file()
    info = getattr(engine, "info", None) or {}
    return {"tokens": len(ids) if on_disk else 0, "bytes": full.stat().st_size if on_disk else 0,
            "ram_tokens": ram, "layers": str(info.get("layers") or "whole"),
            "config": session_config(svc), "pending": session_pending(svc, key)}


def session_pending(svc, key: str) -> bool:
    """Whether the save of `key` is still to be written here: deferred, or being written now. The node's sync waits for
    it to end (GET /pom/session/pending) before it sends the conversation to the other machines."""
    engine = batch_engine(svc)
    if engine is None:
        return False
    with engine.slot_cv:
        return key in _writing or any(item[0] == key for item in deferred_saves.values())


def save_deferred(svc) -> bool:
    """Writes one deferred save while every slot is idle (the engine's session_file needs that), or drops the ones
    past DEFER_MAX_S. Returns whether a save was written or dropped (False: nothing was ready)."""
    engine = batch_engine(svc)
    if engine is None:
        return False
    with engine.slot_cv:
        now = time.monotonic()
        limit = PREFILL_DEFER_MAX_S if ROLE == "prefill" else DEFER_MAX_S
        for slot in [s for s, item in deferred_saves.items() if now - item[3] > limit]:
            key = deferred_saves.pop(slot)[0]
            print(f"pom: session {key} not saved: no idle engine within {limit:.0f} s", file=sys.stderr,
                  flush=True)
        # a request in flight (queued or generating) may already own a slot the engine has not marked busy yet, and
        # the engine answers a SAVE in the same stream as its generations: the save waits for the bridge to be idle
        if not deferred_saves or any(engine.slot_busy) or in_flight_count():
            return False
        slot = min(deferred_saves, key=lambda s: deferred_saves[s][3])
        key, prompt, held, _ = deferred_saves.pop(slot)
        engine.slot_busy[slot] = True            # the save's own reservation: the slot is not picked meanwhile
        _writing.add(key)
    out = None
    try:
        out = save_turn(svc, key, prompt, locked=True, held=held, slot=slot, in_turn=True, wait=False)
    except Exception as e:                       # a save that fails is a session not kept, never a broken turn
        print(f"pom: session {key} not saved: {e}", file=sys.stderr, flush=True)
    finally:
        # SAVING: the state is in RAM, the slot is free; the file is still being written by the engine. Its end is
        # finished by a thread of its own, so the slot takes the next turn meanwhile.
        with engine.slot_cv:
            engine.slot_busy[slot] = False
            if not (out and out.get("pending")):
                _writing.discard(key)
            engine.slot_cv.notify_all()
    if out and out.get("pending"):
        threading.Thread(target=_finish_writing, args=(key, out, engine), name="pom-session-finish",
                         daemon=True).start()
    return True


def _finish_writing(key: str, out: dict, engine) -> None:
    """The end of a deferred save whose file the engine is still writing: the bookkeeping once it is written, or the
    session dropped with its reason. `key` is pending (session_pending) until then."""
    try:
        out["finish"]()
    except Exception as e:                       # the file was not written: the session is not kept
        print(f"pom: session {key} not saved: {e}", file=sys.stderr, flush=True)
    finally:
        with engine.slot_cv:
            _writing.discard(key)
            engine.slot_cv.notify_all()


def _saver_loop(svc) -> None:
    engine = batch_engine(svc)
    while True:
        with engine.slot_cv:
            if not deferred_saves:
                engine.slot_cv.wait(1.0)
                continue
        if not save_deferred(svc):
            with engine.slot_cv:
                engine.slot_cv.wait(0.2)


def apply_delta(svc, key: str) -> dict:
    """The receiving side: the peer's delta applied to the whole session held here.  The whole session (the base) goes
    into a free slot, the delta on top of it, and the result is saved again as the whole session.  Refused unless the
    whole session's tokens are the delta's base, so a delta is never applied twice or to another base."""
    folder = session_dir()
    meta = P.read_delta_meta(folder / P.file_name(key, "delta_meta"))
    base = P.read_ids(folder / P.file_name(key, "full_ids"))
    if meta is None or base is None or meta[0] != base:
        raise ValueError("the whole session here is not the base of the delta")
    _, ids = meta
    engine = batch_engine(svc)
    if engine is None:
        raise ValueError("the engine runs no batch slots")
    with engine.slot_cv:
        slot = engine.pick_slot(ids)
    if slot is None:
        raise ValueError("no free batch slot to apply the delta in")
    engine_session(svc, "restore", key, slot, ids=base)
    try:
        engine_session(svc, "restore_delta", key, slot, ids=ids)
    except Exception:
        slot_bases.pop(slot, None)
        raise
    whole = engine_session(svc, "save", key, slot)
    if whole.get("pending") is not None:             # SAVING: the file must be written before it is this peer's base
        batch_engine(svc).await_session_file(whole["pending"])
    P.write_json(folder / P.file_name(key, "full_ids"), whole["ids"])
    slot_bases[slot] = (key, whole["ids"])
    prune_cache()
    return {"ids": whole["ids"]}
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "frequency_penalty",
                 "repeat_penalty", "seed")


def generation_request(req: dict) -> tuple[str, bool, bool, int, dict]:
    """(prompt, chat_template, thinking, max_tokens, sampling) of a /pom/generate body; ValueError if malformed."""
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("'prompt' must be a non-empty string")
    max_tokens = req.get("max_tokens") or 0
    if not isinstance(max_tokens, int) or max_tokens < 0:
        raise ValueError("'max_tokens' must be a non-negative integer")
    sampling = {k: req[k] for k in SAMPLING_KEYS if req.get(k) is not None}
    stop = req.get("stop")
    if stop:
        if not isinstance(stop, list) or not all(isinstance(s, str) for s in stop):
            raise ValueError("'stop' must be a list of strings")
        sampling["stop"] = stop
    return prompt, req.get("chat_template", True) is not False, req.get("thinking", True) is not False, \
        max_tokens, sampling


def prefill_skip(ids: list[int], decode_holds) -> str | None:
    """Why a prefill_only turn is not read here (None: it is): the part of the prompt the decode machine lacks is
    shorter than PREFILL_MIN_TOKENS. `decode_holds` is what that machine holds of the conversation (0 when unknown)."""
    holds = decode_holds if isinstance(decode_holds, int) and not isinstance(decode_holds, bool) and decode_holds > 0 \
        else 0
    new = len(ids) - min(holds, len(ids))
    if new < max(PREFILL_MIN_TOKENS, 1):
        return f"{new} new tokens of {len(ids)}, under {PREFILL_MIN_TOKENS}"
    return None


def tool_call_text(name: str, arguments: dict) -> str:
    """A parsed tool call as the model wrote it (the qwen4exp template's XML form), which the POM parses."""
    out = [f"<tool_call>\n<function={name}>\n"]
    for key, value in (arguments or {}).items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        out.append(f"<parameter={key}>\n{value}\n</parameter>\n")
    out.append("</function>\n</tool_call>")
    return "".join(out)


def pom_event(ev) -> tuple[str, str] | None:
    """Maps one parsed Strata event to the POM NDJSON field and text.

    A tool call is whole only in its final `tool_call` event; the streamed
    `tool_start` / `tool_args` pieces come again inside it and are ignored.
    Reasoning deliberately uses a distinct field instead of being flattened
    into literal `<think>` markup.
    """
    if ev.kind == "tool_call" and ev.call is not None:
        return "t", "\n\n" + tool_call_text(ev.call.name, ev.call.arguments)
    text = ev.text or ""
    if not text:
        return None
    if ev.kind == "reasoning":
        return "r", text
    if ev.kind == "content":
        return "t", text
    return None


def fit_budget(max_tokens: int, room: int) -> int:
    """The tokens to generate: the POM's ceiling (0 = none) within the room the prompt leaves. The POM fits
    max_tokens to the context with its own count of the prompt; Strata counts it after its chat template, so
    the ceiling can be a few tokens over, and Strata never shortens a request by itself (#545)."""
    return room if max_tokens <= 0 else min(max_tokens, room)


def probe_groups(svc, labels: list[str]) -> list[list[int]]:
    """The token ids that stand for each label, as the POM's decision probe takes them: the first token of the label and
    of " " + label (without special tokens), each id once."""
    groups = []
    for label in labels:
        ids: list[int] = []
        for text in (label, f" {label}"):
            tokens = svc.tok.encode(text)
            if tokens and tokens[0] not in ids:
                ids.append(tokens[0])
        groups.append(ids)
    return groups


def raw_ids(svc, prompt: str, max_tokens: int) -> tuple[list[int], int]:
    """A completion prompt as it is (no template), with Strata's context check."""
    ids = svc.tok.encode(prompt, parse_special=True)
    ctx = svc.engine.max_context or getattr(svc.engine, "known_ctx", 0)
    if ctx <= 0:
        raise S.EngineStarting("the engine is starting (a minute or two); try again shortly")
    room = ctx - S.CTX_SLACK - len(ids)
    if room < 1:
        raise ValueError(f"prompt ({len(ids)} tokens) leaves no room to answer in the context ({ctx})")
    return ids, fit_budget(max_tokens, room)


def make_handler(original):
    def factory(svc):
        base = original(svc)

        class PomHandler(base):
            def _sessions_off(self) -> bool:
                """The session routes with `session_save` off: 501, and nothing is read or written."""
                if SESSIONS_ON:
                    return False
                self._json(501, {"error": "sessions are off on this deployment (the session_save setting)"})
                return True

            def do_GET(self):
                if self.path.split("?")[0].rstrip("/") == "/pom/status":
                    if self._authorized():
                        with svc.status_lock:
                            queued = int(svc.status.get("queued") or 0)
                        active = in_flight_count()
                        self._json(200, {"loaded": svc.loaded(), "model": svc.model,
                                         "max_context": svc.reported_ctx(), "busy": active > 0,
                                         "in_flight": active, "queued": queued})
                    return
                if self.path.split("?")[0].rstrip("/") == "/pom/session/file":
                    if self._authorized() and not self._sessions_off():
                        self._pom_file_get()
                    return
                if self.path.split("?")[0].rstrip("/") == "/pom/session/offer":
                    if self._authorized() and not self._sessions_off():
                        self._pom_file_offer()
                    return
                if self.path.split("?")[0].rstrip("/") == "/pom/session/pending":
                    if self._authorized() and not self._sessions_off():
                        key = (parse_qs(urlparse(self.path).query).get("key") or [""])[0]
                        why = P.key_problem(key)
                        if why:
                            self._json(400, {"error": f"key {why}"})
                        else:
                            self._json(200, {"pending": session_pending(svc, key)})
                    return
                super().do_GET()

            def do_PUT(self):
                if self.path.split("?")[0].rstrip("/") == "/pom/session/chunk":
                    if self._authorized() and not self._sessions_off():
                        try:
                            self._pom_chunk_put()
                        except ValueError as e:
                            self._json(400, {"error": str(e)})
                    return
                if self.path.split("?")[0].rstrip("/") == "/pom/session/file":
                    if self._authorized() and not self._sessions_off():
                        try:
                            self._pom_file_put()
                        except ValueError as e:
                            self._json(400, {"error": str(e)})
                    return
                self._json(405, {"error": "PUT is for /pom/session/file"})

            def do_POST(self):
                if self.path.split("?")[0].rstrip("/") == "/pom/session":
                    if self._authorized() and not self._sessions_off():
                        try:
                            req = json.loads(self._body() or b"{}")
                            if not isinstance(req, dict):
                                raise ValueError("send a JSON object")
                            with in_flight():
                                self._pom_session(req)
                        except ValueError as e:
                            self._json(400, {"error": str(e)})
                    return
                if self.path.split("?")[0].rstrip("/") != "/pom/generate":
                    return super().do_POST()
                if not self._authorized():
                    return
                try:
                    req = json.loads(self._body() or b"{}")
                    if not isinstance(req, dict):
                        raise ValueError("send a JSON object")
                    with in_flight():
                        self._pom_generate(req)
                except S.BadBody as e:
                    self._json(e.status, {"error": str(e)})
                except ValueError as e:
                    if not self.answer_started:
                        self._json(400, {"error": str(e)})
                except (S.GpuBusy, S.EngineStarting, S.EngineDied, S.EngineStuck, S.ModelBusy) as e:
                    if not self.answer_started:
                        self._json(503, {"error": str(e)})
                finally:
                    svc.drop_embeddings()
                    if self.watch_done is not None:
                        self.watch_done.set()
                    record = self.record
                    if record is not None:
                        with svc.status_lock:
                            record["finished_at"] = S.time.time()
                            record["state"] = "error" if record.get("error") else record.get("outcome",
                                                                                               "completed")
                        svc.request_trace.record = None

            def _file_query(self) -> tuple[str, str]:
                query = parse_qs(urlparse(self.path).query)
                key, kind = (query.get("key") or [""])[0], (query.get("kind") or [""])[0]
                why = P.key_problem(key)
                if why:
                    raise ValueError(f"key {why}")
                if kind not in P.KINDS:
                    raise ValueError(f"kind must be one of {', '.join(P.KINDS)}")
                return key, kind

            def _pom_file_offer(self) -> None:
                """GET /pom/session/offer?key=: what this machine can give a peer for the conversation (session_offer)."""
                key = (parse_qs(urlparse(self.path).query).get("key") or [""])[0]
                why = P.key_problem(key)
                if why:
                    self._json(400, {"error": f"key {why}"})
                    return
                self._json(200, session_offer(svc, key))

            def _pom_file_get(self) -> None:
                """One session file, streamed, under the service FIFO (no save or restore writes it meanwhile). With
                `offset` and `len` a piece of it, read without the FIFO: a published file never changes (its name is
                renamed into place only when it is whole), so a fetch does not hold the generations."""
                try:
                    key, kind = self._file_query()
                    query = parse_qs(urlparse(self.path).query)
                    ranged = "offset" in query or "len" in query
                    offset = int((query.get("offset") or ["0"])[0])
                    length = int((query.get("len") or [str(SESSION_PIECE)])[0])
                    if offset < 0 or not 0 < length <= SESSION_PIECE:
                        raise ValueError("a piece is at most 64 MiB from a nonnegative offset")
                except ValueError as e:
                    self._json(400, {"error": str(e)})
                    return
                path = session_dir() / P.file_name(key, kind)
                if ranged:
                    if not path.is_file():
                        self._json(404, {"error": f"no {kind} file for {key}"})
                        return
                    size = path.stat().st_size
                    with open(path, "rb") as f:
                        f.seek(offset)
                        piece = f.read(length)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(piece)))
                    self.send_header("X-File-Size", str(size))          # the whole file, for the fetch's end
                    self.end_headers()
                    self.wfile.write(piece)
                    return
                with svc.fifo:
                    if not path.is_file():
                        self._json(404, {"error": f"no {kind} file for {key}"})
                        return
                    size = path.stat().st_size
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(size))
                    self.end_headers()
                    with open(path, "rb") as f:
                        while chunk := f.read(16 << 20):
                            self.wfile.write(chunk)

            def _pom_chunk_put(self) -> None:
                """One piece of a session file from a peer (pom_sessions.write_chunk). The file is published when its last
                piece is in; a JSON kind is checked first, so a bad file is never published."""
                query = parse_qs(urlparse(self.path).query)
                key, kind = (query.get("key") or [""])[0], (query.get("kind") or [""])[0]
                why = P.key_problem(key)
                if why:
                    raise ValueError(f"key {why}")
                if kind not in P.KINDS:
                    raise ValueError(f"kind must be one of {', '.join(P.KINDS)}")
                try:
                    offset = int((query.get("offset") or [""])[0])
                    total = int((query.get("total") or [""])[0])
                except ValueError:
                    raise ValueError("offset and total are numbers") from None
                size = int(self.headers.get("Content-Length") or 0)
                if not 0 <= size <= P.CHUNK:
                    raise ValueError(f"a piece is at most {P.CHUNK} bytes")
                data = self.rfile.read(size) if size else b""
                if len(data) != size:
                    raise ValueError("the body ended early")
                path = session_dir() / P.file_name(key, kind)
                with svc.fifo:
                    P.write_chunk(path, offset, total, data)
                    if offset + size == total and kind in ("full_ids", "delta_meta"):
                        try:
                            json.loads(path.read_text())
                        except ValueError:
                            path.unlink(missing_ok=True)
                            raise ValueError(f"{kind} is not JSON") from None
                    if offset + size == total:
                        prune_cache()
                self._json(200, {"received": offset + size})

            def _pom_file_put(self) -> None:
                """One session file received from a peer: written whole, under the service FIFO.  The JSON kinds are
                checked before they are written; the file kinds are written to a temporary name and renamed."""
                key, kind = self._file_query()
                size = int(self.headers.get("Content-Length") or 0)
                if size <= 0:
                    raise ValueError("send the file with a Content-Length")
                path = session_dir() / P.file_name(key, kind)
                if kind == "full_ids" or kind == "delta_meta":
                    body = self.rfile.read(size)
                    if len(body) != size:
                        raise ValueError("the body ended early")
                    try:
                        value = json.loads(body)
                    except ValueError:
                        raise ValueError(f"{kind} must be JSON") from None
                    if (kind == "full_ids" and not P.is_ids(value)) or (kind == "delta_meta" and not (
                            isinstance(value, dict) and P.is_ids(value.get("base")) and P.is_ids(value.get("ids")))):
                        raise ValueError(f"{kind} does not hold what it should")
                    with svc.fifo:
                        P.write_json(path, value)
                else:
                    tmp = path.with_name(path.name + ".part")
                    with svc.fifo:
                        left = size
                        with open(tmp, "wb") as f:
                            while left > 0:
                                chunk = self.rfile.read(min(left, 16 << 20))
                                if not chunk:
                                    tmp.unlink(missing_ok=True)
                                    raise ValueError("the body ended early")
                                f.write(chunk)
                                left -= len(chunk)
                        os.replace(tmp, path)
                self._json(200, {"bytes": size, "file": path.name})

            def _pom_session(self, req: dict) -> None:
                if "key" in req:
                    return self._pom_keyed(req)
                action = req.get("action")
                if action not in SESSION_ACTIONS:
                    raise ValueError("action must be save, restore, save_delta or restore_delta")
                why = S.slot_filename_problem(req.get("filename"))
                if why:
                    raise ValueError(f"filename {why}")
                path = str(session_dir() / req["filename"])
                if action.startswith("restore") and not os.path.lexists(path):
                    self._json(404, {"error": f"no session named {req['filename']}"})
                    return
                with svc.status_lock:
                    svc.status["queued"] += 1
                waiting = True
                try:
                    with svc.fifo:
                        with svc.status_lock:
                            svc.status["queued"] -= 1
                        waiting = False
                        if not svc.loaded():
                            self._json(503, {"error": "the model is not loaded"})
                            return
                        with svc.status_lock:
                            svc.status.update(busy=True, phase=action.replace("_", " "), started=S.time.time())
                        try:
                            result = svc.engine.session_file(action, path)
                        except S.SessionRefused as e:
                            self._json(e.status, {"error": str(e), "kind": e.kind, "published": bool(e.published)})
                            return
                        finally:
                            with svc.status_lock:
                                svc.status["busy"] = False
                finally:
                    if waiting:
                        with svc.status_lock:
                            svc.status["queued"] -= 1
                self._json(200, {"tokens": result.get("tokens"), "bytes": result.get("bytes"),
                                 "ms": result.get("ms")})

            def _pom_keyed(self, req: dict) -> None:
                """A keyed session step: `apply`, the receiving side of a delta the peer sent (see apply_delta), or
                `piece`, a piece of a prefill machine's turn fetched here (see apply_piece)."""
                if req.get("action") not in ("apply", "piece"):
                    raise ValueError("a keyed session action is apply or piece")
                why = P.key_problem(req.get("key"))
                if why:
                    raise ValueError(f"key {why}")
                if not svc.loaded():
                    self._json(503, {"error": "the model is not loaded"})
                    return
                try:
                    if req["action"] == "piece":
                        if not isinstance(req.get("piece"), str):
                            raise ValueError("a piece needs its key")
                        result = apply_piece(svc, req["key"], req["piece"], req.get("kind"))
                    else:
                        result = apply_delta(svc, req["key"])
                except S.SessionRefused as e:
                    self._json(e.status, {"error": str(e), "kind": e.kind, "published": bool(e.published)})
                    return
                self._json(200, result)

            def _line(self, obj) -> None:
                self.wfile.write(json.dumps(obj, ensure_ascii=False).encode() + b"\n")
                self.wfile.flush()

            def _pom_generate(self, req: dict) -> None:
                prompt, chat, thinking, max_tokens, sampling = generation_request(req)
                # sessions off: the request's session is not read, and the turn runs as it did before sessions
                session = req.get("session") if SESSIONS_ON else None
                if session is not None and P.key_problem(session):
                    raise ValueError(f"session {P.key_problem(session)}")
                # the prefill/decode split: this machine reads the prompt for another one, which decodes
                prefill_only = req.get("prefill_only") is True
                if prefill_only and session is None:
                    raise ValueError("'prefill_only' needs a session, and the sessions on (inference_role or "
                                     "session_save)")
                self.record = svc.begin_request("/pom/generate", {"model": svc.model, "stream": True,
                                                                  "max_tokens": max_tokens})
                # a decision probe (POM /v1/systemone): the label distribution of the first generated token
                probe = req.get("probe")
                labels = probe.get("labels") if isinstance(probe, dict) else None
                if labels is not None and (not isinstance(labels, list) or not labels or
                                           not all(isinstance(x, str) and x for x in labels)):
                    raise ValueError("'probe.labels' must be a non-empty list of strings")
                # the question's reusable prefix (bytes of the prompt): pinned so the next questions read only their part
                prefix_bytes = probe.get("prefix_bytes") if isinstance(probe, dict) else None
                if prefix_bytes is not None and (not isinstance(prefix_bytes, int) or isinstance(prefix_bytes, bool)
                                                 or prefix_bytes < 1):
                    raise ValueError("'probe.prefix_bytes' must be a positive whole number")
                svc.load()
                if chat:
                    # the images of the turn (data: URLs) come before its text; the engine reads them through the
                    # deployment's projector (prepare() encodes them and run() takes the embeddings)
                    images = req.get("images") or []
                    if not isinstance(images, list) or not all(isinstance(x, str) and x.startswith("data:image/")
                                                              for x in images):
                        raise ValueError("'images' must be a list of data:image/... URLs")
                    content = ([{"type": "image", "source": src} for src in images]
                               + [{"type": "text", "text": prompt}]) if images else prompt
                    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": content}]
                    # No limit asked: prepare answers the room the templated prompt leaves.
                    ids, thinking, room = svc.prepare(messages, None, {"enable_thinking": thinking}, None)
                    max_new = fit_budget(max_tokens, room)
                    if prefix_bytes and labels:
                        # the engine pins the checkpoint at the end of the question's prefix (`strata_prefix`, resolved
                        # against the rendered prompt the same way a client's own would be)
                        head = prompt.encode()[:prefix_bytes].decode("utf-8", "ignore")
                        resolved = svc.resolve_prefix({"message": 1, "chars": len(head)}, messages, None,
                                                      {"enable_thinking": thinking}, ids)
                        if resolved:
                            sampling = {**sampling, "strata_prefix": resolved}
                else:
                    if req.get("images"):
                        raise ValueError("images need the chat template (chat_template true)")
                    ids, max_new = raw_ids(svc, prompt, max_tokens)
                    thinking = False
                skipped = prefill_skip(ids, req.get("decode_holds")) if prefill_only else None
                if prefill_only and skipped is None:
                    # its tokens are dropped: the decode machine samples the answer itself. Two, not one: a turn of one
                    # token ends at its admission (Strata's generate_batched returns before the slot holds it, without
                    # the turn hook), so its session could not be kept. No thinking budget: with two tokens it caps the
                    # thinking at one, and that ends the turn at its first token as well
                    max_new = 2
                    sampling = {**sampling, "reasoning_budget_tokens": 0}
                # a short prompt is not kept: its turn is not saved in the slot nor synced to the peers. A machine of a
                # prefill/decode split keeps every turn: the prefill machine hands it over, and the decode machine
                # runs it in the slot its session was restored into (and offers what it holds to the next turn)
                if session is not None and len(ids) < MIN_SESSION_TOKENS and ROLE is None:
                    session = None
                if skipped is not None:
                    print(f"pom: prefill of session {session} skipped: {skipped}", file=sys.stderr, flush=True)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    self._line({"done": {"prefill": {"tokens": 0, "skipped": skipped}, "finish": "stop",
                                         "prompt_tokens": len(ids), "completion_tokens": 0,
                                         "reasoning_tokens": 0, "cached_tokens": 0}})
                    return
                svc.engine.last_probe = None
                svc.engine.probe_request = probe_groups(svc, labels) if labels and not prefill_only else None
                if SESSIONS_ON:
                    restore_for_prompt(svc, list(ids))
                cancel = threading.Event()
                self._watch_client(cancel)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                if prefill_only and req.get("pieces") is True:
                    # the prefill/decode split's pieces: each piece is announced as it is saved (prefill_pieces)
                    try:
                        read = prefill_pieces(svc, session, list(ids), sampling, cancel, self._line)
                    except (S.EngineDied, S.SessionRefused, ValueError) as e:
                        self._note(error={"message": str(e)})
                        self._line({"error": str(e)})
                        return
                    except OSError:
                        self._note(outcome="disconnected")
                        cancel.set()
                        return
                    self._line({"done": {"prefill": {"tokens": len(ids) - read["reused"], "pieces": read["pieces"]},
                                         "finish": "stop", "prompt_tokens": len(ids), "completion_tokens": 0,
                                         "reasoning_tokens": 0, "cached_tokens": read["reused"],
                                         "timings": {"prompt_ms": read["prompt_ms"]}}})
                    return
                turn: dict = {}

                def turn_hook(slot, held):
                    # the engine's turn is ending in `slot`, which still holds it: its save is deferred to an idle
                    # engine (defer_save), so the turn's answer does not wait for the disk. The done line says so
                    # (`session.deferred`), and the node syncs the conversation once session_pending is false
                    turn["ran"] = True
                    turn["deferred"] = defer_save(svc, session, list(ids), slot, held)
                pending_done = None

                def write_done(x) -> None:
                    # the turn's save is deferred in its slot window (the hook above); the done line says so
                    if session is not None and "ran" not in turn:
                        print(f"pom: session {session}: the turn did not run in a batch slot, not saved",
                              file=sys.stderr, flush=True)
                    if prefill_only:
                        print(f"pom: prefill of session {session}: {len(ids)} prompt tokens read "
                              f"({x.get('reused') or 0} already in a slot), kept: {bool(turn.get('deferred'))}",
                              file=sys.stderr, flush=True)
                    self._line({"done": {
                        **({"session": {"key": session, "saved": None, "deferred": True}}
                           if turn.get("deferred") else {}),
                        **({"prefill": {"tokens": len(ids) - (x.get("reused") or 0)}} if prefill_only else {}),
                        **({"probe": svc.engine.last_probe} if svc.engine.probe_request else {}),
                        "finish": x.get("finish"),
                        "prompt_tokens": x.get("prompt_tokens", len(ids)),
                        "completion_tokens": x.get("completion_tokens", 0),
                        "reasoning_tokens": x.get("reasoning_tokens", 0),
                        "cached_tokens": x.get("reused") or 0,
                        "timings": x.get("timings"),
                        "stop_sequence": x.get("stop_sequence"),
                    }})

                # A prefill turn runs without the thinking budget: its one token would reach a budget capped at 1, and
                # the budget's wrap-up leaves the pass before the engine ends the turn in its slot (the slot then holds
                # nothing to keep). The prompt's tokens are the same: the template was applied above (prepare), with
                # the request's own thinking flag, as on the decode machine.
                run_thinking = thinking and not prefill_only
                try:
                    for kind, x in svc.run(ids, run_thinking, None, max_new, sampling, cancel,
                                           keep_slot=session is not None,
                                           turn_hook=turn_hook if session is not None else None):
                        if kind == "ping":
                            progress = getattr(svc.engine, "progress", None)
                            self._line({"p": list(progress)} if progress else {"p": None})
                        elif kind == "event":
                            event = None if prefill_only else pom_event(x)
                            if event is not None:
                                field, text = event
                                self._line({field: text})
                        elif kind == "done":
                            if prefill_only:
                                # its slot is freed by the engine's generator after this yield: the session is kept,
                                # and the done line written, once the generator has ended (below)
                                pending_done = x
                                continue
                            write_done(x)
                    if prefill_only and pending_done is not None:
                        if "ran" not in turn:
                            turn["ran"] = True
                            turn["deferred"] = keep_prefill(svc, session, list(ids))
                        write_done(pending_done)
                except OSError:
                    self._note(outcome="disconnected")
                    cancel.set()
                except (S.EngineDied, ValueError) as e:
                    self._note(error={"message": str(e)})
                    try:
                        self._line({"error": str(e)})
                    except OSError:
                        pass
                finally:
                    svc.engine.probe_request = None

        return PomHandler

    return factory


def apply_pom_settings(argv: list[str]) -> None:
    """The deployment's engine settings (`pom_settings` in the run config, written by the plugin) applied with
    Strata's own rules (serve/runconfig.py); a value Strata refuses stops the start with Strata's message."""
    global SESSIONS_ON, ROLE, PREFILL_MIN_TOKENS
    if "--config" not in argv[:-1]:
        return
    path = Path(argv[argv.index("--config") + 1])
    cfg = json.loads(path.read_text(encoding="utf-8"))
    settings = cfg.pop("pom_settings", None)
    if not settings:
        return
    # session_save is the bridge's own setting, not an engine one: it is taken out before Strata's rules see the rest
    SESSIONS_ON = bool(settings.pop("session_save", False))
    # the prefill/decode role is the bridge's own too; the split hands the prompt's state over as a session
    role = settings.pop("inference_role", None)
    ROLE = role if role in ("prefill", "decode") else None
    # the deployment's threshold of the split (POM settings, prefill_min_tokens), over the environment's
    threshold = settings.pop("prefill_min_tokens", None)
    if isinstance(threshold, int) and not isinstance(threshold, bool) and threshold >= 0:
        PREFILL_MIN_TOKENS = threshold
    if ROLE == "prefill":
        print(f"[pom] prefill machine: a turn is read here from {PREFILL_MIN_TOKENS} new tokens", flush=True)
    if ROLE:
        SESSIONS_ON = True
        print(f"[pom] inference role: {ROLE} (the prefill/decode split keeps the sessions on)", flush=True)
        # A machine of the split holds the whole model's experts (~47 GB on qwen4exp IQ3_XXS): asking transparent huge
        # pages for that arena (MADV_HUGEPAGE, with the distro default defrag=madvise) made every page fault compact
        # memory on a machine whose RAM it nearly fills, and the load never ended (100% system time). Strata's own
        # check (#2126) looks only at the start, when the memory is still free. The engine inherits this; an
        # operator's STRATA_ARENA_THP=1 still asks for them.
        if "STRATA_ARENA_THP" not in os.environ:
            os.environ.setdefault("STRATA_NO_ARENA_THP", "1")
            print("[pom] expert arena without transparent huge pages (a whole model; STRATA_ARENA_THP=1 asks for them)",
                  flush=True)
    print(f"[pom] session save (KV cache on the shared disk): {'on' if SESSIONS_ON else 'off'}", flush=True)
    if settings:
        from serve import runconfig
        try:
            cfg, changed = runconfig.apply(cfg, settings)
        except ValueError as e:
            print(f"[pom] engine setting refused: {e}", flush=True)
            raise SystemExit(2)
        print(f"[pom] engine settings: {', '.join(changed) or 'no change'}", flush=True)
    # the same setting turns on the engine's asynchronous session write (SAVING, then the file on a thread)
    args = cfg.get("args") if isinstance(cfg.get("args"), list) else []
    if SESSIONS_ON and "--session-async-save" not in args:
        cfg["args"] = [*args, "--session-async-save"]
    ENGINE_ARGS[:] = [str(a) for a in cfg.get("args") or []]
    path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def main() -> int:
    apply_pom_settings(sys.argv)
    S.make_handler = make_handler(S.make_handler)
    return S.main()


if __name__ == "__main__":
    sys.exit(main())
