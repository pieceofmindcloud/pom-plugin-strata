"""CPU tests of the bridge's session steps (save_turn, restore_for_prompt, apply_delta) against a fake service and a fake
engine that keeps slots and files the way the Strata engine does.  The bridge imports Strata's server, so run with
    cd runtime && PYTHONPATH=<Strata checkout> python3 -m unittest test_pom_bridge_sessions
"""

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

import pom_sessions as P

try:
    import pom_strata_server as B
    from serve import server as S
except ImportError as error:  # no Strata checkout on the path: the engine-facing steps are not testable here
    B = None
    SKIP = str(error)


class FakeEngine:
    """Slots hold tokens; a save writes the slot's tokens to the file, a restore reads them back; deltas are the tail
    after the base the slot last saved or restored, and a delta is refused when the slot holds no base.  As in the
    real StrataEngine, the batch state is here, not on the service."""

    def __init__(self, svc):
        self.svc = svc
        self.base = {}          # slot -> tokens of the base (last save or restore)
        self.calls = []

    @property
    def batch(self):
        return self.svc._slots

    @property
    def slot_cv(self):
        return self.svc.slot_cv

    @property
    def slot_busy(self):
        return self.svc.slot_busy

    @property
    def slot_held(self):
        return self.svc.slot_held

    def pick_slot(self, prompt):
        return self.svc.pick_slot(prompt)

    def session_file(self, action, path, slot=None, ids=None, in_turn=False):
        self.calls.append((action, Path(path).name, slot))
        held = self.svc.slot_held
        if action == "save":
            Path(path).write_text(json.dumps(held[slot]))
            self.base[slot] = list(held[slot])
            return {"tokens": len(held[slot]), "bytes": 1, "ms": 0.0, "ids": list(held[slot])}
        if action == "save_delta":
            if slot not in self.base:
                raise S.SessionRefused("invalid", "no base to cut a delta from")
            Path(path).write_text(json.dumps({"base": self.base[slot], "ids": held[slot]}))
            return {"tokens": len(held[slot]), "bytes": 1, "ms": 0.0}
        if action == "restore":
            if not Path(path).is_file():
                raise S.SessionRefused("io", "no file")
            held[slot] = json.loads(Path(path).read_text())
            self.base[slot] = list(held[slot])
            return {"tokens": len(held[slot]), "bytes": 1, "ms": 0.0}
        if action == "restore_delta":
            delta = json.loads(Path(path).read_text())
            if self.base.get(slot) != delta["base"]:
                raise S.SessionRefused("invalid", "the delta's base is not the slot's")
            held[slot] = list(delta["ids"])
            self.base[slot] = list(held[slot])
            return {"tokens": len(held[slot]), "bytes": 1, "ms": 0.0}
        raise AssertionError(action)


class FakeService:
    """The service has no batch attribute of its own: as in the real server, the batch lives on `engine`."""

    def __init__(self, slots=2):
        self._slots = slots
        self.slot_cv = threading.Condition()
        self.slot_busy = [False] * slots
        self.slot_held = [[] for _ in range(slots)]
        self.fifo = threading.Lock()
        self.engine = FakeEngine(self)

    def loaded(self):
        return True

    def pick_slot(self, prompt):
        free = [b for b, busy in enumerate(self.slot_busy) if not busy]
        return free[0] if free else None


@unittest.skipIf(B is None, "needs the Strata checkout on PYTHONPATH")
class BridgeSessions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("POM_STRATA_SESSIONS")
        os.environ["POM_STRATA_SESSIONS"] = self.tmp.name
        B.slot_bases.clear()
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        if self.old is None:
            os.environ.pop("POM_STRATA_SESSIONS", None)
        else:
            os.environ["POM_STRATA_SESSIONS"] = self.old
        self.tmp.cleanup()

    def test_the_first_turn_is_saved_whole_and_the_next_one_as_a_delta(self):
        svc = FakeService()
        svc.slot_held[0] = [1, 2, 3, 4]          # the turn just ran in slot 0
        first = B.save_turn(svc, "conv", [1, 2, 3])
        self.assertEqual(first, {"ids": [1, 2, 3, 4], "delta": False})
        self.assertEqual(P.read_ids(self.folder / "conv.ids.json"), [1, 2, 3, 4])
        self.assertFalse((self.folder / "conv.dlt.json").exists())
        svc.slot_held[0] = [1, 2, 3, 4, 5, 6]    # the next turn extends it
        second = B.save_turn(svc, "conv", [1, 2, 3, 4, 5])
        self.assertEqual(second["delta"], True)
        meta = P.read_delta_meta(self.folder / "conv.dlt.json")
        self.assertEqual(meta, ([1, 2, 3, 4], [1, 2, 3, 4, 5, 6]))

    def test_a_slot_taken_by_another_conversation_saves_whole_not_a_delta(self):
        svc = FakeService()
        svc.slot_held[0] = [1, 2, 3, 4]
        B.save_turn(svc, "conv", [1, 2, 3])
        svc.slot_held[0] = [9, 9, 9, 9]          # another conversation took the slot
        B.save_turn(svc, "other", [9, 9, 9])
        svc.slot_held[0] = [1, 2, 3, 4, 5]
        result = B.save_turn(svc, "conv", [1, 2, 3, 4])
        self.assertEqual(result["delta"], False)
        self.assertFalse(any(call[0] == "save_delta" and call[1] == "conv.strsdlt" for call in svc.engine.calls))

    def test_a_turn_hook_saves_in_its_own_slot_while_it_is_busy(self):
        svc = FakeService()
        svc.slot_held[0] = [1, 2, 3, 4]
        svc.slot_busy[0] = True                  # the turn's own slot, its window still open
        result = B.save_turn(svc, "conv", [1, 2, 3], held=[1, 2, 3, 4], slot=0, in_turn=True)
        self.assertEqual(result, {"ids": [1, 2, 3, 4], "delta": False})
        self.assertEqual(P.read_ids(self.folder / "conv.ids.json"), [1, 2, 3, 4])
        self.assertTrue(svc.slot_busy[0])        # the save does not free the turn's slot

    def test_a_restore_takes_the_longest_saved_start_of_the_prompt(self):
        svc = FakeService()
        svc.slot_held[0] = [1, 2, 3, 4, 5]
        B.save_turn(svc, "conv", [1, 2, 3])
        svc.slot_held = [[], []]
        B.restore_for_prompt(svc, [1, 2, 3, 4, 5, 6])
        self.assertEqual(svc.slot_held[0], [1, 2, 3, 4, 5])
        self.assertEqual(B.slot_bases[0], ("conv", [1, 2, 3, 4, 5]))
        # a slot already holding that much is left alone
        svc.engine.calls.clear()
        B.restore_for_prompt(svc, [1, 2, 3, 4, 5, 6])
        self.assertEqual(svc.engine.calls, [])

    def test_the_receiver_applies_a_delta_to_its_whole_session_once(self):
        sender = FakeService()
        sender.slot_held[0] = [1, 2, 3]
        B.save_turn(sender, "conv", [1, 2])
        sender.slot_held[0] = [1, 2, 3, 4, 5]
        B.save_turn(sender, "conv", [1, 2, 3, 4])
        # the receiver has the sender's first turn as its whole session, then gets the delta
        receiver_folder = self.folder / "receiver"
        receiver_folder.mkdir()
        (receiver_folder / "conv.strsess").write_text(json.dumps([1, 2, 3]))
        P.write_json(receiver_folder / "conv.ids.json", [1, 2, 3])
        (receiver_folder / "conv.strsdlt").write_text((self.folder / "conv.strsdlt").read_text())
        P.write_json(receiver_folder / "conv.dlt.json", {"base": [1, 2, 3], "ids": [1, 2, 3, 4, 5]})
        os.environ["POM_STRATA_SESSIONS"] = str(receiver_folder)
        receiver = FakeService()
        receiver.engine.base = {}
        result = B.apply_delta(receiver, "conv")
        self.assertEqual(result, {"ids": [1, 2, 3, 4, 5]})
        self.assertEqual(P.read_ids(receiver_folder / "conv.ids.json"), [1, 2, 3, 4, 5])
        # applied already: its whole session is no longer the base, so a second apply is refused
        with self.assertRaises(ValueError):
            B.apply_delta(receiver, "conv")

    def test_a_delta_on_another_base_is_refused_and_leaves_the_session(self):
        folder = self.folder
        (folder / "conv.strsess").write_text("[]")
        P.write_json(folder / "conv.ids.json", [7, 7])
        P.write_json(folder / "conv.dlt.json", {"base": [1, 2, 3], "ids": [1, 2, 3, 4]})
        svc = FakeService()
        with self.assertRaises(ValueError):
            B.apply_delta(svc, "conv")
        self.assertEqual(P.read_ids(folder / "conv.ids.json"), [7, 7])
        self.assertEqual(svc.engine.calls, [])


if __name__ == "__main__":
    unittest.main()
