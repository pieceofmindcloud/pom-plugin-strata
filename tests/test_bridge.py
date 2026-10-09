"""The /pom/generate bridge returns a tool call as text the POM parses.

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


def bridge(chunks: list[str]) -> str:
    parser = OutputParser(thinking=True, tools=None, stream_tools=True, recover=False)
    think, out = B.ThinkWrapper(), []
    for chunk in chunks:
        out += [think.event(ev) for ev in parser.feed(chunk)]
    out += [think.event(ev) for ev in parser.finish("stop")]
    out.append(think.close())
    return "".join(out)


def test_tool_call_survives_whole_and_streamed():
    for chunks in ([OUTPUT], [OUTPUT[i:i + 7] for i in range(0, len(OUTPUT), 7)]):
        text = bridge(chunks)
        assert text.startswith("<think>Let me run a quick check."), text
        assert "</think>" in text and text.index("</think>") < text.index("<tool_call>"), text
        assert "<function=Bash>\n<parameter=command>\nls -la\n</parameter>" in text, text
        assert text.rstrip().endswith("</function>\n</tool_call>"), text
        assert text.count("<tool_call>") == 1, text


def test_non_string_arguments_are_json():
    assert "<parameter=n>\n3\n</parameter>" in B.tool_call_text("f", {"n": 3})
    assert '<parameter=o>\n{"a": [1]}\n</parameter>' in B.tool_call_text("f", {"o": {"a": [1]}})


def test_the_poms_ceiling_fits_the_room_the_templated_prompt_leaves():
    # The POM fit 249993 to the context with its own count; the template left 249926.
    assert B.fit_budget(249993, 249926) == 249926
    assert B.fit_budget(512, 249926) == 512
    assert B.fit_budget(0, 249926) == 249926


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
