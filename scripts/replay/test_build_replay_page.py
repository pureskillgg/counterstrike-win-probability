#!/usr/bin/env python3
"""Tests for build_replay_page.py and make_sample_replay.py. Standard library only.

Run (project interpreter):
    G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\replay\\test_build_replay_page.py

Writes only under notes/replay/_check/build_tests and removes that folder after each test.
"""

import copy
import io
import json
import re
import shutil
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import build_replay_page as B  # noqa: E402
import make_sample_replay as G  # noqa: E402

REAL_MAPS = ROOT / "notes" / "replay" / "maps"
SAMPLE = ROOT / "notes" / "replay" / "replay_sample.json"
WORK = ROOT / "notes" / "replay" / "_check" / "build_tests"


class BuildTests(unittest.TestCase):
    def setUp(self):
        WORK.mkdir(parents=True, exist_ok=True)
        self.doc = B.load_replay(SAMPLE)

    def tearDown(self):
        shutil.rmtree(WORK.parent, ignore_errors=True)  # removes notes/replay/_check entirely

    def write(self, name, data):
        path = WORK / name
        text = data if isinstance(data, str) else json.dumps(data)
        path.write_text(text, encoding="utf-8")
        return path

    def test_sample_builds_with_real_map(self):
        out = WORK / "sample.html"
        B.build(SAMPLE, REAL_MAPS, B.DEFAULT_TEMPLATE, out)
        html = out.read_text(encoding="utf-8")
        self.assertNotIn("/*__", html)
        self.assertEqual(html.count("window.REPLAY = {"), 1)
        self.assertEqual(html.count('window.MAP_IMAGES = {"de_inferno":"data:image/png;base64,'), 1)
        # Only the template's own closing tags; the embedded JSON adds none.
        template_closes = B.read_text(B.DEFAULT_TEMPLATE, "template").count("</script>")
        self.assertEqual(html.count("</script>"), template_closes)
        # The embedded JSON parses back to the same document.
        m = re.search(r"window\.REPLAY = (.*?);\n", html)
        self.assertIsNotNone(m)
        self.assertEqual(json.loads(m.group(1)), self.doc)

    def test_missing_map_image_fails_clearly(self):
        empty = WORK / "no_maps"
        empty.mkdir()
        out = WORK / "x.html"
        with self.assertRaisesRegex(B.BuildError, r"map image missing: .*de_inferno\.png"):
            B.build(SAMPLE, empty, B.DEFAULT_TEMPLATE, out)
        self.assertFalse(out.exists())

    def test_missing_map_meta_fails_clearly(self):
        doc = copy.deepcopy(self.doc)
        del doc["map_meta"]
        path = self.write("no_meta.json", doc)
        with self.assertRaisesRegex(B.BuildError, "map_meta is missing"):
            B.build(path, REAL_MAPS, B.DEFAULT_TEMPLATE, WORK / "x.html")

    def test_map_meta_for_other_map_fails_clearly(self):
        doc = copy.deepcopy(self.doc)
        doc["map_meta"]["map"] = "de_dust2"
        path = self.write("other_meta.json", doc)
        with self.assertRaisesRegex(B.BuildError, "there is no map_meta for this map"):
            B.build(path, REAL_MAPS, B.DEFAULT_TEMPLATE, WORK / "x.html")

    def test_map_file_that_is_not_png_fails(self):
        maps = WORK / "maps"
        maps.mkdir()
        (maps / "de_inferno.png").write_bytes(b"GIF89a not a png")
        with self.assertRaisesRegex(B.BuildError, "is not a PNG"):
            B.build(SAMPLE, maps, B.DEFAULT_TEMPLATE, WORK / "x.html")

    def test_template_marker_must_appear_exactly_once(self):
        template = self.write("bad_template.html", "<script>/*__REPLAY__*/</script>")
        with self.assertRaisesRegex(B.BuildError, "exactly once"):
            B.build(SAMPLE, REAL_MAPS, template, WORK / "x.html")

    def test_script_close_tag_inside_data_is_escaped(self):
        doc = copy.deepcopy(self.doc)
        doc["rounds"][0]["players"][0]["label"] = "CT1</script><b>x"
        path = self.write("tricky.json", doc)
        out = WORK / "tricky.html"
        B.build(path, REAL_MAPS, B.DEFAULT_TEMPLATE, out)
        html = out.read_text(encoding="utf-8")
        self.assertNotIn("CT1</script>", html)
        self.assertIn("CT1<\\/script>", html)

    def test_generator_output_passes_schema_check_and_matches_sample(self):
        doc = G.build(7)
        B.check_replay(doc, "generated")  # raises BuildError on any schema problem
        self.assertEqual(doc, self.doc)

    def test_nan_or_infinity_in_json_is_a_clear_build_error(self):
        doc = copy.deepcopy(self.doc)
        doc["rounds"][0]["prob"]["p"][3] = float("nan")  # json.dumps writes NaN by default
        path = self.write("nan.json", doc)
        with self.assertRaisesRegex(B.BuildError, "not allowed in JSON"):
            B.build(path, REAL_MAPS, B.DEFAULT_TEMPLATE, WORK / "x.html")

    def test_player_slots_side_and_names_are_checked(self):
        doc = copy.deepcopy(self.doc)
        doc["rounds"][0]["players"][0]["slot"] = 1
        with self.assertRaisesRegex(B.BuildError, "must have slot 0"):
            B.check_replay(doc, "slots")
        doc = copy.deepcopy(self.doc)
        doc["rounds"][0]["players"][7]["side"] = "CT"
        with self.assertRaisesRegex(B.BuildError, "needs side T"):
            B.check_replay(doc, "side")
        doc = copy.deepcopy(self.doc)
        doc["rounds"][0]["players"][0]["name"] = "someone"
        with self.assertRaisesRegex(B.BuildError, "no real names"):
            B.check_replay(doc, "names")

    def test_weapon_and_event_slot_indexes_are_checked(self):
        doc = copy.deepcopy(self.doc)
        doc["rounds"][1]["frames"]["p"][2]["weapon"][10] = 99
        with self.assertRaisesRegex(B.BuildError, "weapon has 99"):
            B.check_replay(doc, "weapon")
        doc = copy.deepcopy(self.doc)
        kill = next(e for e in doc["rounds"][0]["events"] if e["kind"] == "kill")
        kill["victim"] = 12
        with self.assertRaisesRegex(B.BuildError, "needs slot numbers 0-9"):
            B.check_replay(doc, "event")

    def test_cli_success_and_failure_exit_codes(self):
        out = WORK / "cli.html"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(B.main([str(SAMPLE), "--maps-dir", str(REAL_MAPS), "--out", str(out)]), 0)
        self.assertTrue(out.is_file())

        empty = WORK / "no_maps_cli"
        empty.mkdir()
        err = io.StringIO()
        with redirect_stderr(err):
            code = B.main([str(SAMPLE), "--maps-dir", str(empty), "--out", str(WORK / "cli2.html")])
        self.assertEqual(code, 1)
        self.assertIn("error: map image missing", err.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
