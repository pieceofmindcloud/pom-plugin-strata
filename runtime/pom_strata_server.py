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
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
for extra in (HERE / "strata", HERE / "pylib"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from serve import server as S  # noqa: E402

# The system turn the POM's llama.cpp path puts in front of every prompt
# (llama-engine `apply_chat_template`). The same here, so a model answers the
# same prompt the same way whichever engine serves it.
SYSTEM_PROMPT = "You are a helpful assistant."
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


def raw_ids(svc, prompt: str, max_tokens: int) -> tuple[list[int], int]:
    """A completion prompt as it is (no template), with Strata's context check."""
    ids = svc.tok.encode(prompt, parse_special=True)
    ctx = svc.engine.max_context or getattr(svc.engine, "known_ctx", 0)
    if ctx <= 0:
        raise S.EngineStarting("the engine is starting (a minute or two); try again shortly")
    room = ctx - S.CTX_SLACK - len(ids)
    if room < 1:
        raise ValueError(f"prompt ({len(ids)} tokens) leaves no room to answer in the context ({ctx})")
    return ids, room if max_tokens <= 0 else min(max_tokens, room)


def make_handler(original):
    def factory(svc):
        base = original(svc)

        class PomHandler(base):
            def do_GET(self):
                if self.path.split("?")[0].rstrip("/") == "/pom/status":
                    if self._authorized():
                        with svc.status_lock:
                            busy = bool(svc.status.get("busy"))
                            queued = int(svc.status.get("queued") or 0)
                        self._json(200, {"loaded": svc.loaded(), "model": svc.model,
                                         "max_context": svc.reported_ctx(), "busy": busy, "queued": queued})
                    return
                super().do_GET()

            def do_POST(self):
                if self.path.split("?")[0].rstrip("/") != "/pom/generate":
                    return super().do_POST()
                if not self._authorized():
                    return
                try:
                    req = json.loads(self._body() or b"{}")
                    if not isinstance(req, dict):
                        raise ValueError("send a JSON object")
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

            def _line(self, obj) -> None:
                self.wfile.write(json.dumps(obj, ensure_ascii=False).encode() + b"\n")
                self.wfile.flush()

            def _pom_generate(self, req: dict) -> None:
                prompt, chat, thinking, max_tokens, sampling = generation_request(req)
                self.record = svc.begin_request("/pom/generate", {"model": svc.model, "stream": True,
                                                                  "max_tokens": max_tokens})
                svc.load()
                if chat:
                    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                    ids, thinking, max_new = svc.prepare(messages, None, {"enable_thinking": thinking}, max_tokens)
                else:
                    ids, max_new = raw_ids(svc, prompt, max_tokens)
                    thinking = False
                cancel = threading.Event()
                self._watch_client(cancel)
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                think = ThinkWrapper()
                try:
                    for kind, x in svc.run(ids, thinking, None, max_new, sampling, cancel):
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
                            self._line({"done": {
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
