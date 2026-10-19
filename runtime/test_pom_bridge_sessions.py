"""CPU tests of the bridge's session steps (save_turn, restore_for_prompt, apply_delta) against a fake service and a fake
engine that keeps slots and files the way the Strata engine does.  The bridge imports Strata's server, so run with
    cd runtime && PYTHONPATH=<Strata checkout> python3 -m unittest test_pom_bridge_sessions
"""

import json
import os
import tempfile
import threading
import time
import types
import unittest
from unittest import mock
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


class AsyncFakeEngine(FakeEngine):
    """A save answers SAVING at once (the slot is free) and writes its file on a thread, as the engine does for a
    layer split; await_session_file waits for that file (StrataEngine's contract)."""

    def session_file(self, action, path, slot=None, ids=None, in_turn=False):
        if action != "save":
            return super().session_file(action, path, slot=slot, ids=ids, in_turn=in_turn)
        self.calls.append((action, Path(path).name, slot))
        tokens = list(self.svc.slot_held[slot])
        ticket = {"path": path, "slot": slot, "result": None, "error": None, "done": False}

        def write():
            time.sleep(0.3)
            Path(path).write_text(json.dumps(tokens))
            ticket["result"] = {"tokens": len(tokens), "bytes": 1, "ms": 0.0}

        ticket["thread"] = threading.Thread(target=write, daemon=True)
        ticket["thread"].start()
        return {"tokens": len(tokens), "bytes": None, "ms": 0.0, "ids": tokens, "pending": ticket}

    def await_session_file(self, ticket, raise_error=True):
        ticket["thread"].join()
        return ticket["result"]


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


@unittest.skipIf(B is None, "needs the Strata checkout on PYTHONPATH")
class DeferredSaves(unittest.TestCase):
    """A turn's save written later, by save_deferred, while the engine is idle; the slot keeps the turn meanwhile."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("POM_STRATA_SESSIONS")
        os.environ["POM_STRATA_SESSIONS"] = self.tmp.name
        B.slot_bases.clear()
        B.deferred_saves.clear()
        # no background saver here: the tests run save_deferred themselves, so each step is seen as it happens
        self.saver_started = B._saver_started
        B._saver_started = True
        self.folder = Path(self.tmp.name)

    def tearDown(self):
        B.deferred_saves.clear()
        B._saver_started = self.saver_started
        if self.old is None:
            os.environ.pop("POM_STRATA_SESSIONS", None)
        else:
            os.environ["POM_STRATA_SESSIONS"] = self.old
        self.tmp.cleanup()

    def test_the_turn_writes_nothing_and_its_save_waits_for_an_idle_engine(self):
        svc = FakeService()
        svc.slot_held[0] = [1, 2, 3, 4]
        svc.slot_busy[0] = True                   # the turn's own slot, its window still open
        B.defer_save(svc, "conv", [1, 2, 3], 0, [1, 2, 3, 4])
        svc.slot_busy[0] = False                  # the engine frees the slot after the hook
        self.assertEqual(svc.engine.calls, [])
        self.assertFalse((self.folder / "conv.ids.json").exists())
        self.assertTrue(B.save_deferred(svc))
        self.assertEqual(P.read_ids(self.folder / "conv.ids.json"), [1, 2, 3, 4])
        self.assertEqual(B.deferred_saves, {})
        self.assertFalse(svc.slot_busy[0])        # the save gave the slot back

    def test_a_prefill_turn_is_kept_from_the_slot_the_engine_frees_holding_its_prompt(self):
        # max_tokens 1: the consumer stops before BDONE, the hook does not run, and the slot is freed holding the prompt
        svc = FakeService()
        svc.slot_held[1] = [1, 2, 3]
        self.assertTrue(B.keep_prefill(svc, "conv", [1, 2, 3], wait_s=0.5))
        self.assertIn(1, B.deferred_saves)
        self.assertTrue(B.save_deferred(svc))
        self.assertEqual(P.read_ids(self.folder / "conv.ids.json"), [1, 2, 3])

    def test_a_prefill_turn_no_slot_holds_is_not_kept(self):
        svc = FakeService()
        svc.slot_held[0] = [9, 9]
        self.assertFalse(B.keep_prefill(svc, "conv", [1, 2, 3], wait_s=0.2))
        self.assertEqual(B.deferred_saves, {})

    def test_a_prefill_is_skipped_when_the_decode_machine_lacks_too_little(self):
        with mock.patch.object(B, "PREFILL_MIN_TOKENS", 4096):
            self.assertIsNone(B.prefill_skip(list(range(10000)), 0))
            self.assertIsNotNone(B.prefill_skip(list(range(10000)), 8000))   # 2000 new: the decode machine reads them
            self.assertIsNotNone(B.prefill_skip(list(range(100)), None))
        with mock.patch.object(B, "PREFILL_MIN_TOKENS", 0):
            self.assertIsNone(B.prefill_skip(list(range(100)), 0))
            self.assertIsNotNone(B.prefill_skip(list(range(100)), 100))     # nothing new

    def test_a_save_waits_while_another_slot_is_busy(self):
        svc = FakeService()
        svc.slot_held[0] = [1, 2, 3, 4]
        B.defer_save(svc, "conv", [1, 2, 3], 0, [1, 2, 3, 4])
        svc.slot_busy[1] = True                   # another conversation is generating
        self.assertFalse(B.save_deferred(svc))
        self.assertIn(0, B.deferred_saves)
        svc.slot_busy[1] = False
        self.assertTrue(B.save_deferred(svc))
        self.assertEqual(P.read_ids(self.folder / "conv.ids.json"), [1, 2, 3, 4])

    def test_a_save_waits_while_a_request_is_in_flight(self):
        svc = FakeService()
        svc.slot_held[0] = [1, 2, 3, 4]
        B.defer_save(svc, "conv", [1, 2, 3], 0, [1, 2, 3, 4])
        with B.in_flight():                       # a request already admitted, its slot not marked busy yet
            self.assertFalse(B.save_deferred(svc))
            self.assertEqual(svc.engine.calls, [])
            self.assertIn(0, B.deferred_saves)
        self.assertTrue(B.save_deferred(svc))
        self.assertEqual(P.read_ids(self.folder / "conv.ids.json"), [1, 2, 3, 4])

    def test_a_save_answered_saving_frees_the_slot_and_is_finished_when_written(self):
        svc = FakeService()
        svc.engine = AsyncFakeEngine(svc)
        svc.slot_held[0] = [1, 2, 3, 4]
        B.defer_save(svc, "conv", [1, 2, 3], 0, [1, 2, 3, 4])
        self.assertTrue(B.save_deferred(svc))
        self.assertFalse(svc.slot_busy[0])                    # SAVING: the next turn may take the slot now
        self.assertTrue(B.session_pending(svc, "conv"))       # but the session is not kept until it is written
        self.assertFalse((self.folder / "conv.ids.json").exists())
        deadline = time.monotonic() + 5
        while B.session_pending(svc, "conv") and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(B.session_pending(svc, "conv"))
        self.assertEqual(P.read_ids(self.folder / "conv.ids.json"), [1, 2, 3, 4])
        self.assertEqual(B.slot_bases[0], ("conv", [1, 2, 3, 4]))

    def test_a_deferred_slot_is_not_taken_by_another_conversation(self):
        svc = FakeService(slots=2)
        svc.slot_held[0] = [1, 2, 3, 4]
        B.defer_save(svc, "conv", [1, 2, 3], 0, [1, 2, 3, 4])
        engine = B.batch_engine(svc)
        with engine.slot_cv:
            self.assertEqual(engine.pick_slot([9, 9, 9]), 1)
        self.assertIn(0, B.deferred_saves)
        self.assertFalse(svc.slot_busy[0])        # the guard gave the slot back after the pick

    def test_with_one_slot_the_other_conversation_waits_for_the_save(self):
        svc = FakeService(slots=1)
        svc.slot_held[0] = [1, 2, 3, 4]
        B.defer_save(svc, "conv", [1, 2, 3], 0, [1, 2, 3, 4])
        engine = B.batch_engine(svc)
        with engine.slot_cv:
            self.assertIsNone(engine.pick_slot([9, 9, 9]))

    def test_the_conversations_own_next_turn_takes_its_slot_and_supersedes_the_save(self):
        svc = FakeService(slots=1)
        svc.slot_held[0] = [1, 2, 3, 4]
        B.defer_save(svc, "conv", [1, 2, 3], 0, [1, 2, 3, 4])
        engine = B.batch_engine(svc)
        with engine.slot_cv:
            self.assertEqual(engine.pick_slot([1, 2, 3, 4, 5]), 0)
        self.assertEqual(B.deferred_saves, {})
        self.assertEqual(svc.engine.calls, [])

    def test_a_conversation_is_pending_until_its_save_is_written(self):
        svc = FakeService()
        svc.slot_held[0] = [1, 2, 3, 4]
        self.assertFalse(B.session_pending(svc, "conv"))
        self.assertTrue(B.defer_save(svc, "conv", [1, 2, 3], 0, [1, 2, 3, 4]))
        self.assertTrue(B.session_pending(svc, "conv"))
        self.assertFalse(B.session_pending(svc, "other"))
        seen = []
        original = B.save_turn

        def recording(*args, **kwargs):
            seen.append(B.session_pending(svc, "conv"))   # still pending while the save is written
            return original(*args, **kwargs)

        with mock.patch.object(B, "save_turn", recording):
            self.assertTrue(B.save_deferred(svc))
        self.assertEqual(seen, [True])
        self.assertFalse(B.session_pending(svc, "conv"))

    def test_without_a_batch_engine_nothing_is_deferred(self):
        svc = types.SimpleNamespace(engine=None)
        self.assertFalse(B.defer_save(svc, "conv", [1], 0, [1]))
        self.assertFalse(B.session_pending(svc, "conv"))

    def test_a_save_not_written_in_time_is_dropped(self):
        svc = FakeService()
        svc.slot_held[0] = [1, 2, 3, 4]
        B.deferred_saves[0] = ("conv", [1, 2, 3], [1, 2, 3, 4], time.monotonic() - 3600)
        B.guard_pick(svc.engine)
        self.assertFalse(B.save_deferred(svc))
        self.assertEqual(B.deferred_saves, {})
        self.assertEqual(svc.engine.calls, [])


class SessionSaveSetting(unittest.TestCase):
    """The deployment's session_save setting: taken out of the engine's settings, and off when absent."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "config.json"
        self.before = B.SESSIONS_ON if B is not None else False

    def tearDown(self):
        if B is not None:
            B.SESSIONS_ON = self.before
        self.tmp.cleanup()

    def run_with(self, cfg):
        self.path.write_text(json.dumps(cfg))
        B.apply_pom_settings(["pom_strata_server.py", "--config", str(self.path)])
        return json.loads(self.path.read_text())

    @unittest.skipIf(B is None, "needs the Strata checkout on PYTHONPATH")
    def test_session_save_true_turns_sessions_on_and_is_not_an_engine_setting(self):
        B.SESSIONS_ON = False
        written = self.run_with({"model": "m", "pom_settings": {"session_save": True}})
        self.assertTrue(B.SESSIONS_ON)
        self.assertNotIn("pom_settings", written)
        self.assertIn("--session-async-save", written["args"])     # the same setting turns the async write on

    @unittest.skipIf(B is None, "needs the Strata checkout on PYTHONPATH")
    def test_session_save_false_keeps_sessions_off(self):
        B.SESSIONS_ON = True
        written = self.run_with({"model": "m", "pom_settings": {"session_save": False}})
        self.assertFalse(B.SESSIONS_ON)
        self.assertNotIn("--session-async-save", written.get("args", []))

    @unittest.skipIf(B is None, "needs the Strata checkout on PYTHONPATH")
    def test_without_pom_settings_the_config_is_left_alone_and_sessions_stay_off(self):
        B.SESSIONS_ON = False
        cfg = {"model": "m"}
        self.assertEqual(self.run_with(cfg), cfg)
        self.assertFalse(B.SESSIONS_ON)


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(B is None, "needs the Strata checkout on PYTHONPATH")
class SessionOffer(unittest.TestCase):
    """What a machine offers a peer for a conversation (session_offer): its file on disk, its slot in RAM, its split."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = os.environ.get("POM_STRATA_SESSIONS")
        os.environ["POM_STRATA_SESSIONS"] = self.tmp.name
        B.slot_bases.clear()
        self.folder = Path(self.tmp.name)
        self.svc = FakeService()
        self.svc.engine.info = {"layers": "0-35|36-47"}

    def tearDown(self):
        if self.old is None:
            os.environ.pop("POM_STRATA_SESSIONS", None)
        else:
            os.environ["POM_STRATA_SESSIONS"] = self.old
        B.slot_bases.clear()
        self.tmp.cleanup()

    def save_on_disk(self, key, ids, body=b"x" * 100):
        P.write_json(self.folder / P.file_name(key, "full_ids"), ids)
        (self.folder / P.file_name(key, "full")).write_bytes(body)

    def test_a_session_on_disk_is_offered_with_its_tokens_bytes_and_split(self):
        self.save_on_disk("conv", [1, 2, 3, 4], b"y" * 4096)
        offer = B.session_offer(self.svc, "conv")
        self.assertEqual((offer["tokens"], offer["bytes"], offer["layers"]), (4, 4096, "0-35|36-47"))
        self.assertFalse(offer["pending"])

    def test_a_session_only_in_ram_is_not_offered_to_fetch(self):
        self.svc.slot_held[0] = [1, 2, 3, 4, 5]
        B.slot_bases[0] = ("conv", [1, 2, 3, 4, 5])
        offer = B.session_offer(self.svc, "conv")
        self.assertEqual((offer["tokens"], offer["ram_tokens"]), (0, 5))

    def test_a_whole_model_says_whole(self):
        self.svc.engine.info = {}
        self.save_on_disk("conv", [1, 2])
        self.assertEqual(B.session_offer(self.svc, "conv")["layers"], "whole")

    def test_an_unknown_conversation_offers_nothing(self):
        offer = B.session_offer(self.svc, "nobody")
        self.assertEqual((offer["tokens"], offer["bytes"], offer["ram_tokens"]), (0, 0, 0))
