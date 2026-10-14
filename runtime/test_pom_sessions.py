"""CPU tests of pom_sessions.py: run with `cd runtime && python3 -m unittest test_pom_sessions`."""

import tempfile
import unittest
from pathlib import Path

import pom_sessions as P


class KeysAndNames(unittest.TestCase):
    def test_keys(self):
        self.assertIsNone(P.key_problem("conv-1"))
        self.assertIsNone(P.key_problem("a.b_c"))
        self.assertIsNotNone(P.key_problem(""))
        self.assertIsNotNone(P.key_problem(".hidden"))
        self.assertIsNotNone(P.key_problem("a/b"))
        self.assertIsNotNone(P.key_problem("a" * (P.MAX_KEY + 1)))
        self.assertIsNotNone(P.key_problem(3))

    def test_file_names(self):
        self.assertEqual(P.file_name("c1", "full"), "c1.strsess")
        self.assertEqual(P.file_name("c1", "delta"), "c1.strsdlt")
        self.assertEqual(P.file_name("c1", "delta_meta"), "c1.dlt.json")
        with self.assertRaises(ValueError):
            P.file_name("c1", "other")


class Prefixes(unittest.TestCase):
    def test_prefix_len_needs_the_whole_session_and_a_token_left(self):
        self.assertEqual(P.prefix_len([1, 2, 3, 4], [1, 2]), 2)
        self.assertEqual(P.prefix_len([1, 2, 3], [1, 2, 3]), 0)   # nothing left to read
        self.assertEqual(P.prefix_len([1, 2, 3], [1, 9]), 0)
        self.assertEqual(P.prefix_len([1, 2, 3], []), 0)

    def test_restore_takes_the_longest_session_that_beats_the_slots(self):
        sessions = {"a": [1, 2], "b": [1, 2, 3, 4], "c": [7]}
        prompt = [1, 2, 3, 4, 5]
        self.assertEqual(P.choose_restore(prompt, sessions, [[]]), "b")
        # a slot already holds the longer start: nothing is restored
        self.assertIsNone(P.choose_restore(prompt, sessions, [[1, 2, 3, 4]]))
        self.assertEqual(P.choose_restore(prompt, sessions, [[1, 2]]), "b")

    def test_restore_ties_go_to_the_lower_key(self):
        sessions = {"z": [1, 2], "m": [1, 2]}
        self.assertEqual(P.choose_restore([1, 2, 3], sessions, []), "m")

    def test_restore_none_without_a_session_that_starts_the_prompt(self):
        self.assertIsNone(P.choose_restore([5, 6], {"a": [1, 2]}, []))
        self.assertIsNone(P.choose_restore([1, 2], {"a": [1, 2]}, []))


class Turns(unittest.TestCase):
    def test_a_turn_is_held_by_the_slot_whose_tokens_start_with_it(self):
        held = [[9, 9], [1, 2, 3, 4], [1, 2, 3, 4, 5, 6]]
        self.assertEqual(P.slot_for_turn([1, 2, 3], held), 2)
        self.assertIsNone(P.slot_for_turn([7], held))

    def test_a_base_holds_only_while_its_tokens_start_the_slot(self):
        self.assertTrue(P.base_holds([1, 2, 3, 4], [1, 2]))
        self.assertFalse(P.base_holds([1, 9, 3], [1, 2]))
        self.assertFalse(P.base_holds([1, 2], []))
        self.assertFalse(P.base_holds([1, 2], None))


class Folder(unittest.TestCase):
    def test_only_whole_sessions_with_their_tokens_are_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "good.strsess").write_bytes(b"x")
            P.write_json(folder / "good.ids.json", [1, 2, 3])
            (folder / "nofile.ids.json").write_text("[1]")               # no whole session: skipped
            (folder / "bad.strsess").write_bytes(b"x")
            (folder / "bad.ids.json").write_text('[1, "a"]')            # not token ids: skipped
            (folder / ".hidden.strsess").write_bytes(b"x")
            P.write_json(folder / ".hidden.ids.json", [1])              # a bad key: skipped
            self.assertEqual(P.read_sessions(folder), {"good": [1, 2, 3]})
            self.assertEqual(P.read_sessions(folder / "missing"), {})

    def test_delta_meta_needs_both_token_lists(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.dlt.json"
            P.write_json(path, {"base": [1, 2], "ids": [1, 2, 3]})
            self.assertEqual(P.read_delta_meta(path), ([1, 2], [1, 2, 3]))
            P.write_json(path, {"base": [1, 2]})
            self.assertIsNone(P.read_delta_meta(path))
            self.assertIsNone(P.read_delta_meta(Path(tmp) / "missing.json"))

    def test_write_json_leaves_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            P.write_json(path, {"base": [1], "ids": [1, 2]})
            self.assertEqual(P.read_ids(path), None)                    # an object is not a token list
            P.write_json(path, [4, 5])
            self.assertEqual(P.read_ids(path), [4, 5])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["x.json"])


class Chunks(unittest.TestCase):
    def test_pieces_assemble_into_the_file_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.strsess"
            data = bytes(range(256)) * 40
            total = len(data)
            P.write_chunk(path, 0, total, data[:1000])
            self.assertFalse(path.exists())            # not published until the last piece
            P.write_chunk(path, 1000, total, data[1000:5000])
            P.write_chunk(path, 5000, total, data[5000:])
            self.assertEqual(path.read_bytes(), data)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["c.strsess"])

    def test_a_piece_out_of_order_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.strsess"
            P.write_chunk(path, 0, 10, b"abc")
            with self.assertRaises(ValueError):
                P.write_chunk(path, 5, 10, b"xyz")      # a gap
            with self.assertRaises(ValueError):
                P.write_chunk(path, 0, 10, b"0123456789ab")   # longer than the file

    def test_a_file_is_published_only_when_its_last_piece_is_in(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.strsess"
            P.write_chunk(path, 0, 4, b"ab")
            self.assertFalse(path.exists())
            P.write_chunk(path, 2, 4, b"cd")
            self.assertEqual(path.read_bytes(), b"abcd")


if __name__ == "__main__":
    unittest.main()
