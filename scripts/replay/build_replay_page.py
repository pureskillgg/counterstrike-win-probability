#!/usr/bin/env python3
"""Embed one replay JSON and its radar image into a single HTML page.

Reads viewer_template.html, checks the replay JSON against schema v1, and
writes one self-contained HTML file: the replay becomes window.REPLAY and the
radar PNG becomes a base64 data URI in window.MAP_IMAGES. The page needs no
network access.

Usage:
    python -I -X utf8 build_replay_page.py REPLAY_JSON [--maps-dir DIR]
                                          [--template FILE] [--out FILE]

Defaults:
    --template  scripts/replay/viewer_template.html
    --maps-dir  notes/replay/maps   (needs <map>.png, e.g. de_inferno.png)
    --out       REPLAY_JSON with a .html suffix

Exit status 1 with a message on stderr when the template is malformed, the
replay JSON breaks schema v1, map_meta is missing for the match's map, or the
map image is missing or is not a PNG.
Standard library only.
"""

import argparse
import base64
import json
import re
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DEFAULT_TEMPLATE = HERE / "viewer_template.html"
DEFAULT_MAPS_DIR = ROOT / "notes" / "replay" / "maps"

MARKER_RE = re.compile(r"/\*__(REPLAY|MAPS)__\*/")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PLAYER_KEYS = ("x", "y", "yaw", "hp", "armor", "alive", "weapon", "money")
NPLAYERS = 10


class BuildError(Exception):
    """A problem the user can fix. The message says what to do."""


def read_text(path, what):
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise BuildError("cannot read %s %s: %s" % (what, path, exc))


def _reject_constant(name):
    raise ValueError("%s is not allowed in JSON; write null or a finite number" % name)


def load_replay(path):
    text = read_text(path, "replay JSON")
    try:
        # Python's json writes NaN/Infinity by default; the page cannot embed them.
        return json.loads(text, parse_constant=_reject_constant)
    except ValueError as exc:
        raise BuildError("%s is not valid JSON: %s" % (path, exc))


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_index(v, count):
    """True when v is an integer in 0..count-1 (bools do not count)."""
    return isinstance(v, int) and not isinstance(v, bool) and 0 <= v < count


def _is_slot(v):
    return _is_index(v, NPLAYERS)


def check_replay(doc, source):
    """Check the parts of schema v1 the viewer depends on. Returns the document."""
    if not isinstance(doc, dict):
        raise BuildError("%s: top level must be a JSON object" % source)
    if doc.get("version") != 1:
        raise BuildError("%s: version must be 1 (got %r)" % (source, doc.get("version")))

    match = doc.get("match")
    if not isinstance(match, dict) or not isinstance(match.get("map"), str) or not match["map"]:
        raise BuildError("%s: match.map is missing" % source)
    for key in ("ct_score", "t_score"):
        if not _is_num(match.get(key)):
            raise BuildError("%s: match.%s must be a number" % (source, key))

    meta = doc.get("map_meta")
    if not isinstance(meta, dict):
        raise BuildError("%s: map_meta is missing; the viewer cannot place players without it" % source)
    if meta.get("map") != match["map"]:
        raise BuildError("%s: map_meta is for %r but match.map is %r; there is no map_meta for this map"
                         % (source, meta.get("map"), match["map"]))
    for key in ("pos_x", "pos_y", "scale", "size"):
        if not _is_num(meta.get(key)):
            raise BuildError("%s: map_meta.%s must be a number" % (source, key))
    if meta["scale"] <= 0 or meta["size"] <= 0:
        raise BuildError("%s: map_meta.scale and map_meta.size must be positive" % source)

    weapons = doc.get("weapons")
    if not isinstance(weapons, list) or not all(isinstance(w, str) for w in weapons):
        raise BuildError("%s: weapons must be a list of names" % source)

    rounds = doc.get("rounds")
    if not isinstance(rounds, list) or not rounds:
        raise BuildError("%s: rounds must be a non-empty list" % source)
    for i, rd in enumerate(rounds):
        where = "%s: round %s" % (source, rd.get("n", i + 1) if isinstance(rd, dict) else i + 1)
        if not isinstance(rd, dict):
            raise BuildError("%s is not an object" % where)
        if rd.get("winner") not in ("CT", "T"):
            raise BuildError("%s: winner must be CT or T" % where)
        if not _is_num(rd.get("duration")) or rd["duration"] <= 0:
            raise BuildError("%s: duration must be a positive number" % where)

        players = rd.get("players")
        if not isinstance(players, list) or len(players) != NPLAYERS:
            raise BuildError("%s: players must list exactly 10 entries" % where)
        for s, pl in enumerate(players):
            if not isinstance(pl, dict) or pl.get("slot") != s:
                raise BuildError("%s: players[%d] must have slot %d (slots run 0-9 in list order)"
                                 % (where, s, s))
            if set(pl.keys()) != {"slot", "label", "side"}:
                raise BuildError("%s: players[%d] may only hold slot, label and side (no real names)"
                                 % (where, s))
            want = "CT" if s < 5 else "T"
            if pl.get("side") != want or not isinstance(pl.get("label"), str):
                raise BuildError("%s: slot %d needs side %s (slots 0-4 are CT, 5-9 are T) and a string label"
                                 % (where, s, want))

        frames = rd.get("frames")
        if not isinstance(frames, dict) or not _is_num(frames.get("dt")) or frames["dt"] <= 0:
            raise BuildError("%s: frames.dt must be a positive number" % where)
        n = frames.get("n")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise BuildError("%s: frames.n must be a positive integer" % where)
        p = frames.get("p")
        if not isinstance(p, list) or len(p) != NPLAYERS:
            raise BuildError("%s: frames.p must hold 10 player objects" % where)
        for s, pl in enumerate(p):
            if not isinstance(pl, dict):
                raise BuildError("%s: frames.p[%d] is not an object" % (where, s))
            for key in PLAYER_KEYS:
                arr = pl.get(key)
                if not isinstance(arr, list) or len(arr) != n:
                    got = len(arr) if isinstance(arr, list) else "missing"
                    raise BuildError("%s: frames.p[%d].%s has %s entries, expected n=%d"
                                     % (where, s, key, got, n))
                if not all(_is_num(v) for v in arr):
                    raise BuildError("%s: frames.p[%d].%s must hold numbers only" % (where, s, key))
            for v in pl["weapon"]:
                if not (v == -1 or _is_index(v, len(weapons))):
                    raise BuildError("%s: frames.p[%d].weapon has %r; use -1 or an index into weapons"
                                     % (where, s, v))

        events = rd.get("events")
        if not isinstance(events, list):
            raise BuildError("%s: events must be a list" % where)
        for ev in events:
            if not isinstance(ev, dict) or not _is_num(ev.get("t")) or not isinstance(ev.get("kind"), str):
                raise BuildError("%s: every event needs a numeric t and a kind" % where)
            if ev["kind"] == "kill":
                slots = (ev.get("killer"), ev.get("victim"))
            elif ev["kind"] in ("plant", "defuse"):
                slots = (ev.get("player"),)
            else:
                slots = ()
            if not all(_is_slot(sl) for sl in slots):
                raise BuildError("%s: %s event needs slot numbers 0-9" % (where, ev["kind"]))

        prob = rd.get("prob")
        if prob is not None:
            if not isinstance(prob, dict) or not isinstance(prob.get("t"), list) \
                    or not isinstance(prob.get("p"), list) or len(prob["t"]) != len(prob["p"]):
                raise BuildError("%s: prob must be null or {t: [...], p: [...]} of equal length" % where)
            if not all(_is_num(v) for v in prob["t"]) or not all(v is None or _is_num(v) for v in prob["p"]):
                raise BuildError("%s: prob.t must be numbers and prob.p numbers or null" % where)
    return doc


def map_data_uri(maps_dir, map_name, size):
    path = Path(maps_dir) / (map_name + ".png")
    if not path.is_file():
        raise BuildError("map image missing: %s (put %s.png in the maps directory or pass --maps-dir)"
                         % (path, map_name))
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE) or data[12:16] != b"IHDR" or len(data) < 24:
        raise BuildError("%s is not a PNG file" % path)
    width, height = struct.unpack(">II", data[16:24])
    if (width, height) != (int(size), int(size)):
        print("warning: %s is %dx%d but map_meta.size is %s; the radar will be stretched to fit"
              % (path, width, height, size), file=sys.stderr)
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def js_literal(value):
    """JSON text that is safe inside a <script> element."""
    text = json.dumps(value, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def build(replay_path, maps_dir, template_path, out_path):
    template = read_text(template_path, "template")
    for name in ("REPLAY", "MAPS"):
        count = template.count("/*__%s__*/" % name)
        if count != 1:
            raise BuildError("%s must contain the /*__%s__*/ marker exactly once (found %d)"
                             % (template_path, name, count))

    doc = check_replay(load_replay(replay_path), str(replay_path))
    map_name = doc["match"]["map"]
    uri = map_data_uri(maps_dir, map_name, doc["map_meta"]["size"])

    values = {
        "REPLAY": "window.REPLAY = " + js_literal(doc) + ";",
        "MAPS": "window.MAP_IMAGES = " + js_literal({map_name: uri}) + ";",
    }
    # One pass over the template, so text inserted for one marker is never rescanned.
    html = MARKER_RE.sub(lambda m: values[m.group(1)], template)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(html)
    return out, out.stat().st_size


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build one self-contained replay HTML page.")
    ap.add_argument("replay", help="replay JSON (schema v1)")
    ap.add_argument("--maps-dir", default=str(DEFAULT_MAPS_DIR),
                    help="directory holding <map>.png (default: %(default)s)")
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE),
                    help="viewer template (default: %(default)s)")
    ap.add_argument("--out", default=None,
                    help="output HTML (default: the replay path with a .html suffix)")
    args = ap.parse_args(argv)

    replay = Path(args.replay)
    out = Path(args.out) if args.out else replay.with_suffix(".html")
    try:
        out_path, size = build(replay, args.maps_dir, args.template, out)
    except BuildError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    print("wrote %s (%.1f KB)" % (out_path, size / 1024.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
