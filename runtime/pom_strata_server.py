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

    {"t": "<text>"}               generated text, in order
    {"p": [read, total]}          prompt progress while the prompt is read
    {"done": {...}}               the end: finish, token counts, timings (with the
                                  MTP drafts, draft_n / draft_n_accepted), stop_sequence
    {"error": "..."}              the request failed

Reasoning comes back inside <think> ... </think>, the way the POM chat and the
POM tool-call parser read a llama.cpp pipeline's raw output. Tools are not
given to Strata's parser: the POM renders its own tool block into the prompt
and parses the calls out of the text. Strata's parser still takes the calls
out of the output; they go back into the text in the template's own form.

    GET /pom/status               loaded, model, context, busy (no prompts)

    POST /pom/session             a session file in the session folder (no paths: one file name)
    {"action": "save" | "restore" | "save_delta" | "restore_delta", "filename": "conv-1.strsdlt"}
    -> {"tokens", "bytes", "ms"}; a delta is cut from, and applied to, the base the engine holds.
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
        return
    try:
        engine_session(svc, "restore", key, slot, ids=sessions[key])
    except (S.SessionRefused, S.EngineDied, ValueError) as e:
        print(f"pom: session {key} not restored into slot {slot}: {e}", file=sys.stderr, flush=True)
        return
    slot_bases[slot] = (key, sessions[key])


def save_turn(svc, key: str, prompt: list[int], locked: bool = True, held: list[int] | None = None,
              slot: int | None = None, in_turn: bool = False) -> dict | None:
    """A turn that ran in a batch slot: the slot saves the conversation under `key`.  When the slot's base is still this
    conversation's, the delta from that base is cut first (the peers apply it to their whole session), then the whole
    session is saved, which becomes the next base.  `slot`/`held` come from the engine's turn hook (`in_turn`: the
    slot is still the turn's own, its window is open).  Without them the slot is found by its held tokens.
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
    P.write_json(folder / P.file_name(key, "full_ids"), whole["ids"])
    if delta:
        P.write_json(folder / P.file_name(key, "delta_meta"), {"base": base_ids, "ids": whole["ids"]})
    else:
        for kind in ("delta", "delta_meta"):
            (folder / P.file_name(key, kind)).unlink(missing_ok=True)
    slot_bases[slot] = (key, whole["ids"])
    prune_cache()
    return {"ids": whole["ids"], "delta": delta}


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


def tool_call_text(name: str, arguments: dict) -> str:
    """A parsed tool call as the model wrote it (the qwen4exp template's XML form), which the POM parses."""
    out = [f"<tool_call>\n<function={name}>\n"]
    for key, value in (arguments or {}).items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        out.append(f"<parameter={key}>\n{value}\n</parameter>\n")
    out.append("</function>\n</tool_call>")
    return "".join(out)


class ThinkWrapper:
    """Turns Strata's parsed events back into the raw text a llama.cpp pipeline returns."""

    def __init__(self):
        self.open = False

    def event(self, ev) -> str:
        """One of Strata's events as raw text. A tool call is whole only in its final `tool_call` event; the
        streamed `tool_start` / `tool_args` pieces come again inside it."""
        if ev.kind == "tool_call" and ev.call is not None:
            return self.text("content", "\n\n" + tool_call_text(ev.call.name, ev.call.arguments))
        return self.text(ev.kind, ev.text or "")

    def text(self, kind: str, text: str) -> str:
        if not text:
            return ""
        if kind == "reasoning":
            if not self.open:
                self.open = True
                return "<think>" + text
            return text
        if kind == "content":
            if self.open:
                self.open = False
                return "</think>" + text
            return text
        return ""

    def close(self) -> str:
        if self.open:
            self.open = False
            return "</think>"
        return ""


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
                    if self._authorized():
                        self._pom_file_get()
                    return
                super().do_GET()

            def do_PUT(self):
                if self.path.split("?")[0].rstrip("/") == "/pom/session/chunk":
                    if self._authorized():
                        try:
                            self._pom_chunk_put()
                        except ValueError as e:
                            self._json(400, {"error": str(e)})
                    return
                if self.path.split("?")[0].rstrip("/") == "/pom/session/file":
                    if self._authorized():
                        try:
                            self._pom_file_put()
                        except ValueError as e:
                            self._json(400, {"error": str(e)})
                    return
                self._json(405, {"error": "PUT is for /pom/session/file"})

            def do_POST(self):
                if self.path.split("?")[0].rstrip("/") == "/pom/session":
                    if self._authorized():
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

            def _pom_file_get(self) -> None:
                """One session file, streamed, under the service FIFO (no save or restore writes it meanwhile)."""
                try:
                    key, kind = self._file_query()
                except ValueError as e:
                    self._json(400, {"error": str(e)})
                    return
                path = session_dir() / P.file_name(key, kind)
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
                """A keyed session step: `apply`, the receiving side of a delta the peer sent (see apply_delta)."""
                if req.get("action") != "apply":
                    raise ValueError("a keyed session action is apply")
                why = P.key_problem(req.get("key"))
                if why:
                    raise ValueError(f"key {why}")
                if not svc.loaded():
                    self._json(503, {"error": "the model is not loaded"})
                    return
                try:
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
                session = req.get("session")
                if session is not None and P.key_problem(session):
                    raise ValueError(f"session {P.key_problem(session)}")
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
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
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
                    ids, max_new = raw_ids(svc, prompt, max_tokens)
                    thinking = False
                # a short prompt is not kept: its turn is not saved in the slot nor synced to the peers
                if session is not None and len(ids) < MIN_SESSION_TOKENS:
                    session = None
                svc.engine.last_probe = None
                svc.engine.probe_request = probe_groups(svc, labels) if labels else None
                restore_for_prompt(svc, list(ids))
                cancel = threading.Event()
                self._watch_client(cancel)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                think = ThinkWrapper()
                turn: dict = {}

                def turn_hook(slot, held):
                    # the engine's turn is ending in `slot`, which still holds it: save it there
                    turn["ran"] = True
                    try:
                        turn["saved"] = save_turn(svc, session, list(ids), held=held, slot=slot, in_turn=True)
                    except (S.SessionRefused, S.EngineDied, ValueError) as e:
                        print(f"pom: session {session} not saved: {e}", file=sys.stderr, flush=True)
                try:
                    for kind, x in svc.run(ids, thinking, None, max_new, sampling, cancel,
                                           keep_slot=session is not None,
                                           turn_hook=turn_hook if session is not None else None):
                        if kind == "ping":
                            progress = getattr(svc.engine, "progress", None)
                            self._line({"p": list(progress)} if progress else {"p": None})
                        elif kind == "event":
                            text = think.event(x)
                            if text:
                                self._line({"t": text})
                        elif kind == "done":
                            tail = think.close()
                            if tail:
                                self._line({"t": tail})
                            # the turn was saved in its slot window (the hook below); the done line reports it
                            saved = turn.get("saved")
                            if session is not None and "ran" not in turn:
                                print(f"pom: session {session}: the turn did not run in a batch slot, not saved",
                                      file=sys.stderr, flush=True)
                            self._line({"done": {
                                **({"session": {"key": session, "saved": saved}} if saved else {}),
                                **({"probe": svc.engine.last_probe} if svc.engine.probe_request else {}),
                                "finish": x.get("finish"),
                                "prompt_tokens": x.get("prompt_tokens", len(ids)),
                                "completion_tokens": x.get("completion_tokens", 0),
                                "cached_tokens": x.get("reused") or 0,
                                "timings": x.get("timings"),
                                "stop_sequence": x.get("stop_sequence"),
                            }})
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
    if "--config" not in argv[:-1]:
        return
    path = Path(argv[argv.index("--config") + 1])
    cfg = json.loads(path.read_text(encoding="utf-8"))
    settings = cfg.pop("pom_settings", None)
    if not settings:
        return
    from serve import runconfig
    try:
        cfg, changed = runconfig.apply(cfg, settings)
    except ValueError as e:
        print(f"[pom] engine setting refused: {e}", flush=True)
        raise SystemExit(2)
    print(f"[pom] engine settings: {', '.join(changed) or 'no change'}", flush=True)
    path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")


def main() -> int:
    apply_pom_settings(sys.argv)
    S.make_handler = make_handler(S.make_handler)
    return S.main()


if __name__ == "__main__":
    sys.exit(main())
