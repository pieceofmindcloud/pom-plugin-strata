"""The /pom/generate bridge keeps reasoning separate and returns tool calls as parseable text.

Run with the built runtime next to it: python3 tests/test_bridge.py (needs build/runtime/strata, from
scripts/build-runtime.sh)."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "build/runtime/strata"), str(ROOT / "runtime")]

import pom_strata_server as B  # noqa: E402
from serve.frontend import OutputParser  # noqa: E402

OUTPUT = ("Let me run a quick check.\n</think>\n\n<tool_call>\n<function=Bash>\n<parameter=command>\nls -la\n"
          "</parameter>\n<parameter=timeout>\n5000\n</parameter>\n</function>\n</tool_call>")


def bridge(chunks: list[str]) -> list[tuple[str, str]]:
    parser = OutputParser(thinking=True, tools=None, stream_tools=True, recover=False)
    out = []
    for chunk in chunks:
        out += [mapped for ev in parser.feed(chunk) if (mapped := B.pom_event(ev)) is not None]
    out += [mapped for ev in parser.finish("stop") if (mapped := B.pom_event(ev)) is not None]
    return out


def test_reasoning_and_tool_call_survive_whole_and_streamed():
    for chunks in ([OUTPUT], [OUTPUT[i:i + 7] for i in range(0, len(OUTPUT), 7)]):
        events = bridge(chunks)
        reasoning = "".join(text for field, text in events if field == "r")
        answer = "".join(text for field, text in events if field == "t")
        assert reasoning.rstrip("\n") == "Let me run a quick check.", events
        assert "<think>" not in answer and "</think>" not in answer, answer
        assert "<function=Bash>\n<parameter=command>\nls -la\n</parameter>" in answer, answer
        assert answer.rstrip().endswith("</function>\n</tool_call>"), answer
        assert answer.count("<tool_call>") == 1, answer


def test_non_string_arguments_are_json():
    assert "<parameter=n>\n3\n</parameter>" in B.tool_call_text("f", {"n": 3})
    assert '<parameter=o>\n{"a": [1]}\n</parameter>' in B.tool_call_text("f", {"o": {"a": [1]}})


def test_the_poms_ceiling_fits_the_room_the_templated_prompt_leaves():
    # The POM fit 249993 to the context with its own count; the template left 249926.
    assert B.fit_budget(249993, 249926) == 249926
    assert B.fit_budget(512, 249926) == 512
    assert B.fit_budget(0, 249926) == 249926



def test_the_bridge_times_its_part_of_a_turn():
    t = B.bridge_timings(10.0, 10.25, 10.3, 11.3)
    assert t == {"prepare_ms": 250.0, "restore_ms": 50.0, "first_token_ms": 1000.0}
    assert B.bridge_timings(10.0, 10.0, 10.0, None)["first_token_ms"] is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
