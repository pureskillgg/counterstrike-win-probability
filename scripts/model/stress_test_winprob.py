"""Stress test of the deployable predictor (scripts/model/winprob.py). CPU only.

  python -I -X utf8 scripts\\model\\stress_test_winprob.py --model models\\winprob_v1.json [--fallback models\\winprob_v1_basic.json]
         [--grid <export sample parquet>] [--report <json>] [--write-golden] [--golden <json>] [--only 0,1,13] [--scale 1.0] [--seed 0]
  python -I -X utf8 scripts\\model\\stress_test_winprob.py --self-test      # trains throwaway models with other maps / features and runs it all

Nothing is hard-coded about the model under test: the in-scope maps and modes, the feature list, whether the model takes a player
table, the value ranges of extra inputs and the monotone constraints all come from <model>.meta.json and from the booster's own
trees (winprob constants are only defaults). The winprob rule constants (bomb timer 41 s, round clock 115 s, +-200 s unit limit,
64 ticks/s) are the DOCUMENTED CONTRACT and are written out below on purpose: they must not follow an edit of the code under test.
Exit code 1 if any check fails or crashes (a crash is recorded as a FAIL with its traceback and the report is still written).
A check that could not run is reported as NOT RUN (loudly, in the report and the RESULT line) and never counts as a pass.

Checks (each prints PASS / FAIL / WARN / NOT RUN):
  0   valid states are answered by the model: status ok, reliable, no warning, p varies
  1   fuzz: 40,000 valid / garbage mixes never raise; invariants (invalid_input -> prior, out_of_scope -> reliable False, ...)
  2   scope (from meta): out-of-scope maps / modes, blank / absent / renamed map and mode (not silently ok), case variants
  3   wingman-like 2v2 states; 4 terminal states, T-wiped-with-plant-is-not-terminal, bomb_defused
  5   invalid required inputs return the prior (incl. bomb flag / plant time absent); an unusable table falls back
  6   monotone in alive counts (alive-count model); 6c the same through the snapshot model's fallback route
  6b  remove exactly ONE player from a real-valued table: per-sample direction, wrong-sign and zero-change shares, effect size
  7   determinism, predict_many == predict, 8 threads identical; 8 latency; 8b explain() adds up
  9   real rows (--grid / auto sample): booster replay == pipeline, serving predict() == pipeline, fallback route == fallback booster
  10  input contract: falsy / truthy spellings, bool is not a number, dead rows, spectators, incomplete rows, terminals, reliability
  11  units and schema canary: ticks, ms, fractions, bool armor, equipment x100, renamed keys, text numbers, impossible clock
  12  optional player keys: the policy covers every key the feature code reads; absent keys are named; reliability follows the model
  13  scope and feature contract: scope == meta == map_is_* features, constants vs meta, booster size, meta constraints == code
      (wp_common.mono_for), constraints hold in the booster, row-builder identities
  14  feature liveness: no constant column, dead features (never split) warned, perturbing each input moves p, every table feature reachable
  15  mixed-batch differential: predict_many over a shuffled mix of ok / fallback / terminal / invalid / out_of_scope == predict
  16  fallback exactness: a 'fallback' answer equals the alive-count model's own answer
  17  golden rows: ~40 stored states and exact outputs (models/golden_<model>.json; --write-golden writes it; missing -> NOT RUN)
  18  rule boundary + impossible-state sweep against an independent oracle
  19  leakage / extra-key canary; 20 tick helper
"""
import argparse
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np

import winprob as WPM
from snapshot_features import loadout_from_players
from winprob import WinProb
from wp_items import AK, AWP, HEAVY, M4, RIFLE, SMG, SNIPER_OTHER

sys.stdout.reconfigure(encoding="utf-8")
ALLOWED = {"ok", "terminal", "out_of_scope", "invalid_input", "fallback"}
# the documented contract of the serving rules (see winprob.py docstring); deliberately NOT read from the module under test
RULE_BOMB_TIMER = 41.0     # planted and seconds_since_plant >= this: exploded, T win
RULE_ROUND_CLOCK = 115.0   # not planted and seconds_since_round_live >= this: time ran out, CT win
RULE_MAX_SECONDS = 200.0   # |clock| beyond this is not seconds (ticks, ms): invalid_input
RULE_SLACK = 1.0           # seconds_since_plant may exceed seconds_since_round_live by this much (plant before the round)
RULE_PLANT_LATEST = 125.0  # a plant later than this many seconds into the round is impossible (2026 max 115.0, 2024 max 120.8)
RULE_TICK_RATE = 64
ROOT = os.path.dirname(os.path.dirname(HERE))

GARBAGE = [None, "", "abc", "5", " 4 ", "nan", float("nan"), float("inf"), float("-inf"), -1, -1e9, 1e12, 99, 6, 5.5, 2.4999,
           True, False, [], [1, 2], {}, {"a": 1}, object(), (3,), b"3", 0, 1e-9]
MAP_UNIVERSE = ["de_ancient", "de_anubis", "de_cache", "de_dust2", "de_inferno", "de_mirage", "de_nuke", "de_vertigo", "de_overpass",
                "de_train", "de_cbble", "de_italy", "de_office", "cs_office", "cs_italy", "cs_agency", "de_basalt", "de_edin", "de_grail",
                "de_jura", "de_palais", "de_whistle", "de_golden", "de_poseidon", "de_mills", "de_rooftop"]
JUNK_MAPS = ["de_workshop_xyz", "mirage", "none", "de_ünïcode_地图", "x" * 10000, "de_mirage\x00", 123, 3.5, [], {}, True]
MODE_UNIVERSE = ["wingman", "hostage", "casual", "deathmatch", "dangerzone", "armsrace", "coop", "survival", "retakes", " HOSTAGE ", "Wingman"]
PRIMARIES = [0, 0] + sorted(set(AK + M4 + AWP + RIFLE + SNIPER_OTHER + SMG + HEAVY))
SIDES_CT = ["CT", "ct", " Ct ", 3, "3"]
SIDES_T = ["T", "t", 2, "2"]
PROTOCOL_INPUTS = {"alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_round_live", "seconds_since_plant"}


# ---------------------------------------------------------------- small helpers
def good(p):
    return isinstance(p, float) and math.isfinite(p) and 0.0 <= p <= 1.0


def internal_error(o):
    return any(str(w).startswith("internal error") for w in o.get("warnings", []))


def norm(m):
    return m.strip().lower() if isinstance(m, str) else None


def blank(v):
    return v is None or (isinstance(v, str) and not v.strip())


def same(o1, o2, tol=0.0):
    """Two predictor results are the same answer: status, reliable, warnings, version, and p within tol."""
    return (o1["status"] == o2["status"] and o1["reliable"] == o2["reliable"] and o1["warnings"] == o2["warnings"]
            and o1.get("model_version") == o2.get("model_version") and abs(o1["p_ct_win"] - o2["p_ct_win"]) <= tol)


def sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_json_atomic(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def split_stats(model):
    """feature -> [n_splits, min_threshold, max_threshold] over the trees the predictor uses, read from the booster itself."""
    trees = model.booster.get_dump(dump_format="json")[: model.best_iteration + 1]
    idx = {f: i for i, f in enumerate(model.features)}
    st = {f: [0, math.inf, -math.inf] for f in model.features}
    for t in trees:
        stack = [json.loads(t)]
        while stack:
            nd = stack.pop()
            if "split" in nd:
                key = nd["split"]
                if key in idx:
                    f = key
                elif isinstance(key, str) and key[:1] == "f" and key[1:].isdigit() and int(key[1:]) < len(model.features):
                    f = model.features[int(key[1:])]
                else:
                    f = None
                if f is not None:
                    r = st[f]
                    r[0] += 1
                    c = nd.get("split_condition")
                    if isinstance(c, (int, float)) and math.isfinite(c):
                        r[1] = min(r[1], c)
                        r[2] = max(r[2], c)
                stack.extend(nd.get("children", []))
    return st


def keys_read_by_features():
    """The per-player keys loadout_from_players actually reads (probed, not listed here)."""
    seen = set()

    class Rec(dict):
        def get(self, k, d=None):
            seen.add(k)
            return super().get(k, d)

    row = Rec(side=3, health=100, armor=100, has_helmet=True, has_defuser=True, is_defusing=False, primary=16, flashbang=1, smokegrenade=1,
              hegrenade=1, molotov=0, incgrenade=0, decoy=0, equipment_value=100)
    loadout_from_players([row])
    return seen


def garb(rng, extra=()):
    """A garbage value that is a fresh object every time (the shared [] / {} in GARBAGE must never be mutated by a later state)."""
    return copy.deepcopy(rng.choice(GARBAGE + list(extra)))


def garble_players(rng, s):
    pl_ = s.get("players")
    if not isinstance(pl_, list) or not pl_:
        return
    r = rng.random()
    if r < 0.15:
        s["players"] = pl_ * 12  # oversized table
    elif r < 0.30:
        pl_[rng.randrange(len(pl_))] = garb(rng)
    elif r < 0.50:
        p = pl_[rng.randrange(len(pl_))]
        if isinstance(p, dict) and p:
            p[rng.choice(list(p))] = garb(rng)
    elif r < 0.65:
        p = pl_[rng.randrange(len(pl_))]
        if isinstance(p, dict) and p:
            p.pop(rng.choice(list(p)), None)
    elif r < 0.80:
        p = pl_[rng.randrange(len(pl_))]
        if isinstance(p, dict):
            p["side"] = garb(rng, ["spec", "CTT", 0, 4])
    else:
        for p in pl_:
            if isinstance(p, dict):
                p["health"] = rng.choice([-50, 1e6, 250, 100.0, 0])


# ---------------------------------------------------------------- the suite
class Suite:
    def __init__(self, args):
        self.args = args
        self.seed = args.seed
        self.scale = args.scale
        self.results, self.failures, self.not_run_names, self.not_applicable = [], [], [], []
        self.durations = {}
        self.wp = WinProb.load(args.model, args.fallback)
        self.meta = self.wp.primary.meta
        self.fbm = WinProb(self.wp.fallback) if self.wp.fallback is not None else None  # the alive-count model on its own
        self.name = os.path.splitext(os.path.basename(args.model))[0]
        self.snapshot = self.wp.primary.needs_players
        self.prior = self.wp.prior
        self.features = list(self.wp.primary.features)
        # the scope, derived here independently of the serving code: meta.scope first, then the map_is_* features, module defaults last
        sc = self.meta.get("scope") if isinstance(self.meta.get("scope"), dict) else {}
        maps = sc.get("maps")
        if isinstance(maps, list) and maps and all(isinstance(m, str) for m in maps):
            self.maps, self.maps_src = tuple(m.strip().lower() for m in maps), "meta.scope.maps"
        else:
            cols = [f for f in self.features if f.startswith("map_is_")]
            self.maps, self.maps_src = (tuple("de_" + c[7:] for c in cols), "map_is_* features") if cols else (tuple(WPM.IN_SCOPE_MAPS), "default constant")
        modes = sc.get("modes")
        if isinstance(modes, list) and modes and all(isinstance(m, str) for m in modes):
            self.modes = tuple(m.strip().lower() for m in modes)
        else:
            self.modes = tuple(m for m in WPM.IN_SCOPE_MODES if m)
        self.snap_names = frozenset(loadout_from_players([]))
        fb_feats = list(self.wp.fallback.features) if self.wp.fallback is not None else []
        self.extras = sorted({f for f in self.features + fb_feats if f not in PROTOCOL_INPUTS and not f.startswith("map_is_")
                              and f not in self.snap_names and not f.startswith("st_")})
        self.stats = {"primary": split_stats(self.wp.primary)}
        if self.wp.fallback is not None:
            self.stats["fallback"] = split_stats(self.wp.fallback)
        self.ranges = {}
        for f in self.extras:
            lo, hi = 0.0, 5.0
            for st in self.stats.values():
                if f in st and st[f][0] > 0 and st[f][2] > st[f][1]:
                    lo, hi = st[f][1], st[f][2]
                    break
            self.ranges[f] = (lo, hi)

    # ---- bookkeeping
    def rng(self, name):
        return random.Random(f"{self.seed}:{name}")

    def n(self, base):
        return max(1, int(round(base * self.scale)))

    def _record(self, name, status, detail):
        self.results.append({"check": name, "passed": status in ("PASS", "WARN"), "status": status, "detail": detail})
        if status == "FAIL":
            self.failures.append(name)
        if status == "NOT RUN":
            self.not_run_names.append(name)

    def check(self, name, ok, detail="", warn=False):
        status = "PASS" if ok and not warn else "WARN" if ok else "FAIL"
        print(f"{status:4s}  {name}  {detail}", flush=True)
        self._record(name, status, detail)

    def not_run(self, name, reason):
        print(f"WARN  *** NOT RUN *** {name}: {reason}", flush=True)
        self._record(name, "NOT RUN", reason)

    def na(self, name, reason):
        print(f"N/A   {name}: {reason}", flush=True)
        self.not_applicable.append({"check": name, "reason": reason})

    def run(self, cid, fn):
        t0, before = time.time(), len(self.results)
        try:
            fn(self)
            if len(self.results) == before and not any(a["check"].startswith(cid + " ") for a in self.not_applicable):
                self.check(f"{cid} {fn.__name__} recorded no result (a check that asserts nothing is a failure)", False)
        except Exception:
            self.check(f"{cid} {fn.__name__} CRASHED", False, traceback.format_exc()[-1800:])
        self.durations[cid] = round(time.time() - t0, 1)

    # ---- state builders
    def make_players(self, rng, ac, at, planted=False):
        out = []
        for sides, n in ((SIDES_CT, ac), (SIDES_T, at)):
            ct = sides is SIDES_CT
            for _ in range(n):
                kit = rng.random() < (0.5 if ct else 0.03)  # a T with a kit is unrealistic but exercises the T wiring of st_kit_n_t
                r_def = rng.random()  # always drawn, so two tables built from the same seed differ only in what the caller changes
                out.append({"side": rng.choice(sides), "health": rng.randint(1, 100), "armor": rng.choice([0, 0, 25, 100]),
                            "has_helmet": rng.random() < 0.6, "has_defuser": kit,
                            "is_defusing": bool(planted and r_def < (0.3 if kit else 0.03)),  # a T 'defusing' is unrealistic but exercises the T wiring
                            "primary": rng.choice(PRIMARIES), "flashbang": rng.randint(0, 2), "smokegrenade": rng.randint(0, 1),
                            "hegrenade": rng.randint(0, 1), "molotov": rng.randint(0, 1) if not ct else 0,
                            "incgrenade": rng.randint(0, 1) if ct else 0, "decoy": int(rng.random() < 0.1),
                            "equipment_value": rng.randint(0, 7500)})
        return out

    def extras_dict(self, rng, mid=False):
        return {f: ((lo + hi) / 2.0 if mid else rng.uniform(lo, hi)) for f, (lo, hi) in self.ranges.items()}

    def valid_state(self, rng, with_players=None, planted=None):
        ac, at = rng.randint(1, 5), rng.randint(1, 5)
        if planted is None:
            planted = rng.random() < 0.4
        if planted:
            sp = rng.uniform(0.0, 40.0)
            secs = rng.uniform(17.0, 114.0) + sp
        else:
            sp, secs = -1.0, rng.uniform(0.0, 114.0)
        s = {"bomb_planted": int(planted), "seconds_since_round_live": secs, "seconds_since_plant": sp,
             "map_name": rng.choice(self.maps), "mode": rng.choice(self.modes)}
        s.update(self.extras_dict(rng))
        use_players = self.snapshot if with_players is None else with_players
        if use_players:
            s["players"] = self.make_players(rng, ac, at, bool(planted))
            if rng.random() < 0.5:  # a live feed usually also carries the counts; they agree with the table
                s["alive_ct"], s["alive_t"] = ac, at
        else:
            s["alive_ct"], s["alive_t"] = ac, at
        return s

    def mk(self, ac, at, planted=0, secs=60.0, sp=None, rng=None, players=None, **over):
        """A valid state with the given counts and clock: a player table for snapshot models, the counts for alive-count models."""
        rng = rng or random.Random(11)
        if sp is None:
            sp = 10.0 if planted else -1.0
        s = {"seconds_since_round_live": secs, "bomb_planted": planted, "seconds_since_plant": sp, "map_name": self.maps[0], "mode": self.modes[0]}
        s.update(self.extras_dict(rng, mid=True))
        if players is None and self.snapshot:
            players = self.make_players(rng, ac, at, bool(planted))
        if players is not None:
            s["players"] = players
        if players is None or ac + at == 0:
            s["alive_ct"], s["alive_t"] = ac, at
        s.update(over)
        return s

    def table_state(self, rng, ac=3, at=3, planted=0, secs=60.0, **over):
        """A state whose counts come from a player table, for any kind of model."""
        s = self.mk(ac, at, planted, secs, rng=rng, players=self.make_players(rng, ac, at, bool(planted)), **over)
        s.pop("alive_ct", None)
        s.pop("alive_t", None)
        return s

    def corrupt(self, rng, s):
        for k in rng.sample(list(s), rng.randint(0, 4)):
            if rng.random() < 0.25:
                del s[k]
            else:
                s[k] = garb(rng)
        if self.snapshot and rng.random() < 0.5:
            garble_players(rng, s)
        if rng.random() < 0.05:
            s = rng.choice([None, 5, "x", [], 3.2, object()])
        return s

    def x_of(self, wp, s):
        """The model input row the predictor builds for s (None when the state is answered without the model)."""
        prep = wp._prepare(s)
        return None if "early" in prep else prep["x"][0].copy()


# ---------------------------------------------------------------- checks
def c_valid(S):
    rng, n = S.rng("0"), S.n(2000)
    bad, ps, first = 0, [], None
    for _ in range(n):
        o = S.wp.predict(S.valid_state(rng))
        ps.append(o["p_ct_win"])
        if o["status"] != "ok" or internal_error(o) or not o["reliable"] or o["warnings"]:
            bad += 1
            first = first or (o["status"], o["warnings"][:2])
    spread = float(np.std(ps))
    S.check("0 valid states are answered by the model (status ok, reliable, no warning, p varies)", n > 0 and bad == 0 and spread > 0.02,
            f"bad={bad} of {n} {first or ''} std(p)={spread:.3f}")


def c_fuzz(S):
    wp, rng, n = S.wp, S.rng("1"), S.n(40000)
    t0, bad, internal, inv_bad, counts = time.time(), 0, 0, 0, {}
    for _ in range(n):
        s = S.corrupt(rng, S.valid_state(rng))
        try:
            o = wp.predict(s)
            if not (isinstance(o, dict) and good(o.get("p_ct_win")) and o.get("status") in ALLOWED and isinstance(o.get("reliable"), bool)
                    and isinstance(o.get("warnings"), list) and all(isinstance(w, str) for w in o["warnings"]) and "model_version" in o):
                bad += 1
                continue
            st = o["status"]
            counts[st] = counts.get(st, 0) + 1
            internal += int(internal_error(o))
            if (st == "invalid_input" and (o["p_ct_win"] != S.prior or o["reliable"])) or (st == "out_of_scope" and o["reliable"]) \
                    or (st == "terminal" and o["p_ct_win"] not in (0.0, 1.0)):
                inv_bad += 1
        except Exception:
            bad += 1
    floor_ok = counts.get("ok", 0) >= 0.02 * n and counts.get("invalid_input", 0) >= 0.02 * n
    S.check(f"1 fuzz: no exception, finite p, allowed status, invariants (invalid_input -> prior, out_of_scope -> unreliable, terminal 0/1), no swallowed error ({n} states)",
            n > 0 and bad == 0 and internal == 0 and inv_bad == 0 and floor_ok,
            f"bad={bad} internal_errors={internal} invariant_breaks={inv_bad} statuses={counts} floors_ok={floor_ok} ({time.time() - t0:.1f}s)")


def c_scope(S):
    wp = S.wp
    rng = S.rng("2")
    maps = set(S.maps)
    bad_maps = [m for m in MAP_UNIVERSE if m not in maps] + JUNK_MAPS
    if len([m for m in bad_maps if isinstance(m, str) and m.strip().lower() not in maps]) < 3:
        bad_maps += ["de_not_a_real_map", "cs_nothing", "zz"]
    unexpected, tested_out, tested_in = [], 0, 0

    def run(m, blank_key=False):
        s = S.valid_state(rng)
        if blank_key:
            s.pop("map_name", None)
        else:
            s["map_name"] = m
        return wp.predict(s)

    for m in bad_maps:
        o = run(m)
        n_ = norm(m)
        if n_ is not None and n_ in maps:  # junk that normalises into scope (cannot happen for the lists above, but scope is data)
            tested_in += 1
            if not (o["status"] == "ok" and o["reliable"]):
                unexpected.append((str(m)[:30], o["status"]))
        elif blank(m):
            tested_out += 1
            if not (o["status"] == "out_of_scope" and o["reliable"] is False and any("map not given" in w for w in o["warnings"])):
                unexpected.append(("blank:" + repr(m), o["status"]))
        else:
            tested_out += 1
            if not (good(o["p_ct_win"]) and o["status"] == "out_of_scope" and o["reliable"] is False and o["warnings"]):
                unexpected.append((str(m)[:30], o["status"]))
    for m in (None, "", "   "):
        for key_absent in (False, True):
            o = run(m, key_absent)
            tested_out += 1
            if not (o["status"] == "out_of_scope" and o["reliable"] is False and any("map not given" in w for w in o["warnings"])):
                unexpected.append((f"blank map {m!r} absent={key_absent}", o["status"]))
    for m in sorted(maps):
        for variant in (m, m.upper(), f"  {m.upper()} ", m.title()):
            o = run(variant)
            tested_in += 1
            if not (o["status"] == "ok" and o["reliable"] is True and not o["warnings"]):
                unexpected.append((variant[:30], o["status"], o["warnings"][:1]))
    modes = set(S.modes)
    bad_modes = [m for m in MODE_UNIVERSE if norm(m) not in modes]
    tested_modes = 0
    for mo in bad_modes + [42, [], "x" * 5000]:
        s = S.valid_state(rng)
        s["mode"] = mo
        o = wp.predict(s)
        tested_modes += 1
        if not (good(o["p_ct_win"]) and o["status"] == "out_of_scope" and o["reliable"] is False):
            unexpected.append(("mode " + repr(mo)[:20], o["status"]))
    for mo in (None, "", "  "):
        for key_absent in (False, True):
            s = S.valid_state(rng)
            if key_absent:
                s.pop("mode")
            else:
                s["mode"] = mo
            o = wp.predict(s)
            tested_modes += 1
            if not (o["status"] == "out_of_scope" and o["reliable"] is False and any("mode not given" in w for w in o["warnings"])):
                unexpected.append((f"blank mode {mo!r} absent={key_absent}", o["status"]))
    for mo in S.modes:
        for variant in (mo, mo.upper(), f" {mo} "):
            s = S.valid_state(rng)
            s["mode"] = variant
            o = wp.predict(s)
            tested_modes += 1
            if not (o["status"] == "ok" and o["reliable"] is True):
                unexpected.append(("mode " + variant, o["status"]))
    s = S.valid_state(rng)
    s["mode"] = "premier"
    o = wp.predict(s)
    want = "ok" if "competitive" in modes else "out_of_scope"
    if o["status"] != want:
        unexpected.append(("mode premier", o["status"], "wanted " + want))
    S.check("2 scope from the model's meta: out-of-scope maps / modes flagged, blank or absent map / mode not silently ok, case variants in scope",
            tested_out >= 10 and tested_in >= len(maps) and tested_modes >= 8 and not unexpected,
            f"maps in scope {sorted(maps)} ({S.maps_src}); out-of-scope tested {tested_out}, in-scope variants {tested_in}, modes {tested_modes}; unexpected={unexpected[:4]}")


def c_wingman(S):
    rng, bad, n = S.rng("3"), 0, 0
    mode = next((m for m in MODE_UNIVERSE if norm(m) not in S.modes), "wingman")
    for ac in range(3):
        for at in range(3):
            for planted in (0, 1):
                s = S.mk(ac, at, planted, 40.0, 5.0 if planted else -1.0, rng=rng, mode=mode)
                o = S.wp.predict(s)
                n += 1
                if not good(o["p_ct_win"]) or o["status"] not in ("out_of_scope", "terminal") or o["reliable"] is not False:
                    bad += 1
    S.check("3 wingman-like 2v2 states: finite p, out_of_scope / terminal, never reliable", n == 18 and bad == 0, f"bad={bad} of {n}")


def c_terminal(S):
    wp = S.wp
    P = lambda *a, **k: wp.predict(S.mk(*a, rng=random.Random(5), **k))  # the same player table every call: only the named fields differ
    bad, n_exp = [], [0]

    def expect(label, cond):
        n_exp[0] += 1
        if not cond:
            bad.append(label)

    o = P(0, 3)
    expect("CT dead -> terminal 0.0", o["status"] == "terminal" and o["p_ct_win"] == 0.0 and o["reliable"])
    o = P(3, 0)
    expect("T dead, no plant -> terminal 1.0", o["status"] == "terminal" and o["p_ct_win"] == 1.0 and o["reliable"])
    o = P(0, 3, planted=1)
    expect("CT dead with a plant -> terminal 0.0", o["status"] == "terminal" and o["p_ct_win"] == 0.0)
    for sp in (0.0, 10.0, 40.9):  # the bomb is down and inside the timer: CT must still defuse, the model answers
        o = P(3, 0, planted=1, secs=60.0 + sp, sp=sp)
        expect(f"T dead + planted {sp}s ago is NOT terminal (model number < 1)", o["status"] == "ok" and good(o["p_ct_win"]) and o["p_ct_win"] < 1.0)
    for sp in (41.0, 45.0):
        o = P(3, 0, planted=1, secs=90.0, sp=sp)
        expect(f"T dead + planted {sp}s ago: bomb exploded -> terminal 0.0", o["status"] == "terminal" and o["p_ct_win"] == 0.0)
    o = P(3, 3, secs=RULE_ROUND_CLOCK - 0.1)
    expect("114.9 s with no plant is still the model's", o["status"] == "ok")
    o = P(3, 3, secs=RULE_ROUND_CLOCK)
    expect("115 s with no plant -> terminal 1.0", o["status"] == "terminal" and o["p_ct_win"] == 1.0)
    o = P(3, 3, planted=1, secs=90.0, sp=RULE_BOMB_TIMER - 0.01)
    expect("planted 40.99 s ago is the model's", o["status"] == "ok")
    o = P(3, 3, planted=1, secs=90.0, sp=RULE_BOMB_TIMER)
    expect("planted 41 s ago -> terminal 0.0", o["status"] == "terminal" and o["p_ct_win"] == 0.0)
    ref = P(3, 3, planted=1, secs=60.0, sp=10.0)
    for v in (True, 1, "1", "true", "YES"):
        o = P(3, 3, planted=1, secs=60.0, sp=10.0, bomb_defused=v)
        expect(f"bomb_defused={v!r} -> terminal CT win", o["status"] == "terminal" and o["p_ct_win"] == 1.0)
    for v in (False, 0, "0", "no", None):
        o = P(3, 3, planted=1, secs=60.0, sp=10.0, bomb_defused=v)
        expect(f"bomb_defused={v!r} changes nothing", o["status"] == ref["status"] and o["p_ct_win"] == ref["p_ct_win"] and not o["warnings"])
    for v in ("maybe", 2, float("nan"), [], "nan"):
        o = P(3, 3, planted=1, secs=60.0, sp=10.0, bomb_defused=v)
        expect(f"bomb_defused={v!r} is invalid_input", o["status"] == "invalid_input")
    o = wp.predict(S.mk(3, 3, planted=1, secs=60.0, sp=10.0, mode=next(m for m in MODE_UNIVERSE if norm(m) not in S.modes), bomb_defused=True))
    expect("a defuse in an out-of-scope mode is terminal but reliable False", o["status"] == "terminal" and o["reliable"] is False)
    S.check("4 terminal states and their boundaries (T-wiped with a plant is not terminal before the timer; defuse flag)", n_exp[0] >= 25 and not bad,
            f"assertions={n_exp[0]} violations={len(bad)} {bad[:4]}")


def c_invalid(S):
    wp, rng, prior = S.wp, S.rng("5"), S.prior
    bad = 0
    n = 0
    for g in (None, "abc", float("nan"), [], {}, True):
        s = S.valid_state(rng)
        s["seconds_since_round_live"] = g
        o = wp.predict(s)
        n += 1
        bad += not (o["status"] == "invalid_input" and abs(o["p_ct_win"] - prior) < 1e-12 and o["reliable"] is False)
    s = S.valid_state(rng)
    del s["seconds_since_round_live"]
    bad += wp.predict(s)["status"] != "invalid_input"
    s = S.valid_state(rng, planted=False)
    del s["bomb_planted"]
    o = wp.predict(s)
    bad += not (o["status"] == "invalid_input" and any("bomb_planted" in w for w in o["warnings"]))
    s = S.valid_state(rng, planted=True)
    del s["seconds_since_plant"]
    o = wp.predict(s)
    bad += not (o["status"] == "invalid_input" and any("seconds_since_plant" in w for w in o["warnings"]))
    for g in (-1.0, None, "abc", float("nan")):
        s = S.valid_state(rng, planted=True)
        s["seconds_since_plant"] = g
        bad += wp.predict(s)["status"] != "invalid_input"
        n += 1
    if S.snapshot:
        for g in (None, "abc", [], {}, [None], ["x"], [{}], [{"side": "CT"}, 5], 7, True):  # unusable table
            s = S.valid_state(rng)
            s["players"] = g
            s.pop("alive_ct", None)
            s.pop("alive_t", None)
            o = wp.predict(s)  # no counts either: nothing to go on
            n += 1
            if not (o["status"] == "invalid_input" and abs(o["p_ct_win"] - prior) < 1e-12):
                bad += 1
                print("   unexpected (no table, no counts):", repr(g)[:40], o["status"])
            s = S.valid_state(rng)
            s["players"] = g
            s["alive_ct"], s["alive_t"] = 3, 4  # counts given: the alive-count model answers
            o = wp.predict(s)
            want = "fallback" if wp.fallback is not None else "invalid_input"
            if o["status"] != want or not good(o["p_ct_win"]):
                bad += 1
                print("   unexpected (bad table, counts given):", repr(g)[:40], o["status"], "wanted", want)
    else:
        for k in ("alive_ct", "alive_t"):
            for g in (None, "abc", float("nan"), [], {}, True):
                s = S.valid_state(rng)
                s[k] = g
                o = wp.predict(s)
                n += 1
                bad += not (o["status"] == "invalid_input" and abs(o["p_ct_win"] - prior) < 1e-12)
            s = S.valid_state(rng)
            del s[k]
            bad += wp.predict(s)["status"] != "invalid_input"
    S.check("5 invalid required inputs return the prior (incl. bomb_planted / seconds_since_plant absent); an unusable player table falls back",
            n > 10 and bad == 0, f"bad={bad} of {n}+, prior={prior:.4f}")


def _mono_sweep(S, predictor, make_state, label, want_status):
    rng, n = S.rng("6" + label), S.n(5000)
    states = []
    for _ in range(n):
        s = make_state(rng)
        for ac in range(1, 6):
            d = dict(s)
            d["alive_ct"] = ac
            states.append(d)
        for at in range(1, 6):
            d = dict(s)
            d["alive_t"] = at
            states.append(d)
    out = predictor.predict_many(states)
    viol, wrong_status, up_ct, up_t = 0, 0, [], []
    for i in range(n):
        o10 = out[10 * i: 10 * i + 10]
        wrong_status += sum(1 for o in o10 if o["status"] != want_status)
        ps_ct, ps_t = [o["p_ct_win"] for o in o10[:5]], [o["p_ct_win"] for o in o10[5:]]
        if any(b < a - 1e-7 for a, b in zip(ps_ct, ps_ct[1:])) or any(b > a + 1e-7 for a, b in zip(ps_t, ps_t[1:])):
            viol += 1
        up_ct.append(ps_ct[-1] - ps_ct[0])
        up_t.append(ps_t[0] - ps_t[-1])
    return n, viol, wrong_status, float(np.mean(up_ct)), float(np.mean(up_t))


def c_monotone(S):
    basic = S.wp if not S.snapshot else S.fbm
    if basic is None:
        S.not_run("6 monotone in alive counts: needs the alive-count model (pass --fallback)", "no alive-count model loaded")
        return
    n, viol, ws, up_ct, up_t = _mono_sweep(S, basic, lambda rng: S.valid_state(rng, with_players=False), "a", "ok")
    S.check(f"6 monotone in alive counts (alive-count model; {n} states x 10 variants): more CT never lowers p, more T never raises it",
            n > 0 and viol == 0 and ws == 0 and up_ct > 0.02 and up_t > 0.02,
            f"violations={viol} non-ok answers={ws} mean effect of 1->5 CT {up_ct:+.3f}, of 5->1 T {up_t:+.3f}")
    if S.snapshot:  # the same, through the snapshot model's own fallback route (unusable table + counts)
        def st(rng):
            s = S.valid_state(rng, with_players=False)
            s["players"] = []
            return s
        n, viol, ws, up_ct, up_t = _mono_sweep(S, S.wp, st, "c", "fallback")
        S.check(f"6c monotone in alive counts through the snapshot model's fallback route (status fallback; {n} states x 10)",
                n > 0 and viol == 0 and ws == 0 and up_ct > 0.02 and up_t > 0.02,
                f"violations={viol} non-fallback answers={ws} mean effect {up_ct:+.3f} / {up_t:+.3f}")


def c_victim(S):
    if not S.snapshot:
        S.na("6b exact single-victim removal", "alive-count model: there is no player table")
        return
    rng, n = S.rng("6b"), S.n(3000)
    dct, dt = [], []
    for _ in range(n):
        ac, at = rng.randint(2, 5), rng.randint(2, 5)
        s = S.table_state(rng, ac, at, planted=int(rng.random() < 0.4), secs=rng.uniform(30, 100))
        if s["bomb_planted"]:
            s["seconds_since_plant"] = rng.uniform(0, 25)  # plant time = secs - sp stays > 5 s: a possible clock
        pl_ = s["players"]
        p0 = S.wp.predict(s)["p_ct_win"]
        for side_set, store in ((SIDES_CT, dct), (SIDES_T, dt)):
            idx = [i for i, p in enumerate(pl_) if p["side"] in side_set]
            victim = rng.choice(idx)  # ONE victim per state and side
            s2 = dict(s)
            s2["players"] = [p for i, p in enumerate(pl_) if i != victim]
            store.append(S.wp.predict(s2)["p_ct_win"] - p0)
    dct, dt = np.array(dct), np.array(dt)

    def stats(d, sign):
        se = d.std(ddof=1) / math.sqrt(len(d))
        return {"mean": float(d.mean()), "wrong": float(np.mean(d * sign < 0)), "zero": float(np.mean(d == 0)), "z": float(abs(d.mean()) / max(se, 1e-12))}

    a, b = stats(dct, -1), stats(dt, 1)  # CT removed -> p must fall (sign -1), T removed -> p must rise
    ok = (len(dct) > 0 and a["mean"] < 0 < b["mean"] and a["z"] > 3 and b["z"] > 3
          and a["wrong"] <= 0.05 and b["wrong"] <= 0.05 and a["zero"] <= 0.01 and b["zero"] <= 0.01)
    S.check(f"6b removing exactly one CT lowers p, exactly one T raises it ({n} real-valued tables; mean z>3, wrong-sign <= 5%, zero change <= 1%)", ok,
            f"CT removed: mean {a['mean']:+.4f} wrong-sign {a['wrong']:.1%} zero {a['zero']:.1%} z={a['z']:.0f}; T removed: mean {b['mean']:+.4f} wrong-sign {b['wrong']:.1%} zero {b['zero']:.1%} z={b['z']:.0f}")


def c_determinism(S):
    wp, rng = S.wp, S.rng("7")
    states = [S.valid_state(rng) for _ in range(S.n(2000))]
    a = [wp.predict(s) for s in states]
    b = wp.predict_many(states)
    with ThreadPoolExecutor(8) as ex:
        c = list(ex.map(wp.predict, states))
    again = [wp.predict(s) for s in states[:200]]
    d_ab = max(abs(x["p_ct_win"] - y["p_ct_win"]) for x, y in zip(a, b))
    S.check("7 deterministic; predict_many == predict (full result); 8 threads identical",
            len(states) > 0 and all(same(x, y, 1e-7) for x, y in zip(a, b)) and a == c and a[:200] == again,
            f"max|single-batch|={d_ab:.2e}, threads identical={a == c}, repeat identical={a[:200] == again} ({len(states)} states)")
    return states


def c_latency(S):
    wp, rng = S.wp, S.rng("8")
    states = [S.valid_state(rng) for _ in range(S.n(1000))]
    wp.predict(states[0])
    lat = []
    for s in states:
        t = time.perf_counter()
        wp.predict(s)
        lat.append((time.perf_counter() - t) * 1e3)
    t = time.perf_counter()
    wp.predict_many(states)
    thr = len(states) / (time.perf_counter() - t)
    lat.sort()
    med, p99 = lat[len(lat) // 2], lat[min(len(lat) - 1, int(len(lat) * 0.99))]

    def timed(s):
        t = time.perf_counter()
        wp.predict(s)
        return (time.perf_counter() - t) * 1e3

    with ThreadPoolExecutor(8) as ex:
        lat8 = sorted(ex.map(timed, states[:400]))
    exp = []
    for s in states[:100]:
        t = time.perf_counter()
        wp.explain(s)
        exp.append((time.perf_counter() - t) * 1e3)
    exp.sort()
    S.check("8 latency (CPU, single call): median < 25 ms and p99 < 250 ms", len(lat) > 0 and med < 25.0 and p99 < 250.0,
            f"median {med:.3f} ms, p99 {p99:.3f} ms, {thr:,.0f} predictions/s in predict_many; 8 threads p99 {lat8[int(len(lat8) * 0.99)]:.2f} ms; explain() median {exp[len(exp) // 2]:.2f} ms p99 {exp[-1]:.2f} ms")


def c_explain(S):
    wp, rng = S.wp, S.rng("8b")
    states = [S.valid_state(rng) for _ in range(S.n(2000))]
    bad, maxerr, n_contrib = 0, 0.0, 0
    for s in states:
        e = wp.explain(s)
        p = wp.predict(s)["p_ct_win"]
        if abs(e["p_ct_win"] - p) > 1e-5:
            bad += 1
        if e["status"] in ("ok", "out_of_scope", "fallback") and e["contributions"]:
            n_contrib += 1
            tot = e["base_logit"] + sum(e["contributions"].values())
            err = abs(tot - math.log(max(p, 1e-12) / max(1 - p, 1e-12)))
            maxerr = max(maxerr, err)
            if err > 1e-3 or not all(math.isfinite(v) for v in e["contributions"].values()):
                bad += 1
    e = wp.explain(S.mk(0, 3))
    bad += not (e["status"] == "terminal" and e["contributions"] == {} and e["base_logit"] is None)
    nfz = 0
    nfuzz = S.n(5000)
    for _ in range(nfuzz):
        s = S.valid_state(rng)
        for k in rng.sample(list(s), rng.randint(0, 4)):
            if rng.random() < 0.3:
                s.pop(k, None)
            else:
                s[k] = garb(rng)
        if S.snapshot and rng.random() < 0.5:
            garble_players(rng, s)
        try:
            e = wp.explain(s)
            if not (good(e["p_ct_win"]) and e["status"] in ALLOWED and isinstance(e["contributions"], dict)):
                nfz += 1
        except Exception:
            nfz += 1
    S.check("8b explain(): contributions sum to logit(p); never raises on garbage", n_contrib > 0 and bad == 0 and nfz == 0,
            f"sum errors={bad} (max {maxerr:.2e}) over {n_contrib} explained states, fuzz failures={nfz} of {nfuzz}")


def _grid_path(S):
    a = S.args
    if a.grid:
        return a.grid if os.path.isfile(a.grid) else None
    auto = os.path.join(ROOT, "data", "combined", f"export_{S.name}_test_sample.parquet")
    return auto if os.path.isfile(auto) else None


def c_real_rows(S):
    path = _grid_path(S)
    nm = "9 real rows (booster replay, serving predict(), fallback route)"
    if path is None:
        S.not_run(nm, "no sample / grid file (pass --grid <export sample parquet>); parity_check.py covers the table path on raw tables")
        return
    import polars as pl
    G = pl.read_parquet(path)
    for m in S.maps:  # a plain grid has map_name but no one-hot columns; the export sample has both
        c = "map_is_" + (m[3:] if m[2:3] == "_" else m)
        if c not in G.columns and "map_name" in G.columns:
            G = G.with_columns((pl.col("map_name") == m).cast(pl.Int8).alias(c))
    te = G.sample(n=min(S.n(5000), G.height), seed=0)
    S.check("9 real rows: the sample is not empty", te.height >= 100, f"{te.height} rows from {os.path.basename(path)}")
    has_pipe = "p_pipeline" in te.columns

    def booster_ref(model):
        miss = [f for f in model.features if f not in te.columns]
        if miss:
            return None, miss
        X = te.select(model.features).to_numpy().astype(np.float32)
        return model.predict_rows(X), []

    # (i) the booster reproduces the training pipeline's p on the sample's own feature columns (feature order, column count, model file)
    p_b, miss = booster_ref(S.wp.primary)
    if p_b is None:
        S.not_run("9 real rows: booster replay == p_pipeline", f"sample lacks model inputs {miss[:3]}")
    elif not has_pipe:
        S.not_run("9 real rows: booster replay == p_pipeline", "sample has no p_pipeline column")
    else:
        ref = te["p_pipeline"].to_numpy()
        d = float(np.max(np.abs(p_b - ref)))
        S.check("9 real rows: booster on the sample's feature columns == p_pipeline (feature order and model file)", d < 1e-5 and bool(np.isfinite(p_b).all()),
                f"max |diff| = {d:.2e} over {len(ref)} rows")
    # (ii) serving predict() == pipeline for an alive-count primary model, from an allowlist of input keys only (no labels, no grid metadata)
    extras = [f for f in S.extras if f in te.columns]
    allow = ["seconds_since_round_live", "bomb_planted", "seconds_since_plant", "alive_ct", "alive_t", "map_name"] + extras

    def state_of(r):
        return {k: r[k] for k in allow if k in r} | {"mode": S.modes[0]}

    if not S.snapshot:
        rows = te.to_dicts()
        answers = S.wp.predict_many([state_of(r) for r in rows])
        out = np.array([a["p_ct_win"] for a in answers])
        ref = te["p_pipeline"].to_numpy() if has_pipe else booster_ref(S.wp.primary)[0]
        if ref is None:
            S.not_run("9b real rows: serving predict() == pipeline", "the sample has neither p_pipeline nor all the model's input columns")
            return
        by_rule = np.array([a["status"] == "terminal" for a in answers])
        model_rows = ~by_rule
        diff = float(np.max(np.abs(out[model_rows] - ref[model_rows]))) if model_rows.any() else float("nan")
        warns = sum(1 for a in answers if any("not given" in w for w in a["warnings"]))
        bad_status = sum(1 for a in answers if a["status"] not in ("ok", "terminal"))
        S.check("9b real rows: serving predict() == pipeline (rows answered by a terminal rule excluded; <= 1% by rule)",
                model_rows.sum() > 0.99 * len(rows) and diff < 1e-5 and bad_status == 0 and warns == 0,
                f"max |diff| = {diff:.2e} over {int(model_rows.sum())} model rows; {int(by_rule.sum())} rows by rule; non-ok/terminal statuses {bad_status}; 'not given' warnings {warns}")
        unpl = te.filter(pl.col("bomb_planted") == 0)["seconds_since_round_live"].max() if "bomb_planted" in te.columns else None
        S.check("9c rule thresholds vs the sample (report): max unplanted second and max seconds_since_plant of the real rows", True,
                f"max unplanted second {unpl}; rule at {RULE_ROUND_CLOCK}; max seconds_since_plant {te['seconds_since_plant'].max()}; rule at {RULE_BOMB_TIMER}")
    elif S.wp.fallback is not None:
        # snapshot run: the counts-only route (no player table) must be the alive-count model's own answer on real rows
        p_fb, miss = booster_ref(S.wp.fallback)
        if p_fb is None:
            S.not_run("9b real rows: fallback route == fallback booster", f"sample lacks fallback inputs {miss[:3]}")
        else:
            rows = te.to_dicts()
            answers = S.wp.predict_many([state_of(r) for r in rows])
            out = np.array([a["p_ct_win"] for a in answers])
            by_rule = np.array([a["status"] == "terminal" for a in answers])
            fb_rows = np.array([a["status"] == "fallback" for a in answers])
            diff = float(np.max(np.abs(out[fb_rows] - p_fb[fb_rows]))) if fb_rows.any() else float("nan")
            S.check("9b real rows: the snapshot model's counts-only route (fallback) == the alive-count booster on the same rows",
                    fb_rows.sum() > 0.95 * len(rows) and diff < 1e-5,
                    f"max |diff| = {diff:.2e} over {int(fb_rows.sum())} fallback rows; {int(by_rule.sum())} terminal rows")


def c_contract(S):
    wp, rng = S.wp, S.rng("10")
    bad10, n_exp = [], [0]

    def expect(label, cond):
        n_exp[0] += 1
        if not cond:
            bad10.append(label)

    pl0 = S.make_players(random.Random(5), 3, 3, False)  # ONE table for every flag spelling: only bomb_planted differs between the calls

    def P(ac=3, at=3, **k):
        return wp.predict(S.mk(ac, at, rng=random.Random(5), players=copy.deepcopy(pl0), **k))

    # bomb flag spellings: falsy ones must equal 0, truthy ones must equal 1 (a feed that sends 'false' must never be read as planted)
    # 120 s on the clock: unplanted the round clock has run out (terminal CT win), planted 30 s ago it is the model's number - so the
    # flag changes the status for ANY model, whether or not it lists bomb inputs
    ref0 = P(3, 3, planted=0, secs=120.0)
    ref1 = P(3, 3, planted=1, secs=120.0, sp=30.0)
    expect("the flag matters for the test state (unplanted: terminal, planted: the model)", ref0["status"] == "terminal" and ref1["status"] == "ok")
    for v in ("false", "False", "FALSE", " no ", "0", "n", "f", "N", 0, 0.0, False):
        o = P(3, 3, planted=v, secs=120.0)
        expect(f"bomb_planted={v!r} must behave exactly like 0", o["p_ct_win"] == ref0["p_ct_win"] and o["status"] == ref0["status"] and o["reliable"] == ref0["reliable"])
    for v in ("true", "True", "yes", " Y ", "t", "1", 1, 1.0, True):
        o = P(3, 3, planted=v, secs=120.0, sp=30.0)
        expect(f"bomb_planted={v!r} must behave exactly like 1", o["p_ct_win"] == ref1["p_ct_win"] and o["status"] == ref1["status"])
    for v in ("maybe", "", float("nan"), None, 2, -1, 0.5, [], {}, "nan", "falsey", "yes!"):
        expect(f"bomb_planted={v!r} must be invalid_input", P(3, 3, planted=v)["status"] == "invalid_input")
    # a bool where a number belongs is a bug, not a 1 (counts, clocks, plant time)
    for key in ("seconds_since_round_live", "alive_ct", "alive_t"):
        for v in (True, False):
            s = S.mk(3, 3, rng=random.Random(5), players=None)
            s.pop("players", None)
            s["alive_ct"], s["alive_t"] = 3, 3
            s[key] = v
            expect(f"{key}={v!r} (a bool) must be invalid_input", wp.predict(s)["status"] == "invalid_input")
    for v in (True, False):
        s = S.mk(3, 3, planted=1, secs=60.0, sp=10.0, rng=random.Random(5))
        s["seconds_since_plant"] = v
        expect(f"seconds_since_plant={v!r} (a bool) must be invalid_input", wp.predict(s)["status"] == "invalid_input")
    for v in (-1, -0.5, "-3", float("-inf"), 10 ** 400):
        s = S.mk(3, 3, rng=random.Random(5), players=None)
        s.pop("players", None)
        s["alive_ct"], s["alive_t"] = v, 3
        expect(f"alive_ct={v!r} must be invalid_input", wp.predict(s)["status"] == "invalid_input")
    o = wp.predict(S.mk(0, 2, mode=next(m for m in MODE_UNIVERSE if norm(m) not in S.modes), rng=random.Random(5)))
    expect("a terminal state in an out-of-scope mode is flagged reliable False", o["status"] == "terminal" and o["reliable"] is False)
    for m in [m for m in MAP_UNIVERSE if m not in S.maps][:6]:
        o = wp.predict(S.valid_state(rng) | {"map_name": m})
        expect(f"map {m}: out_of_scope must carry reliable False", o["status"] == "out_of_scope" and o["reliable"] is False)
    o = wp.predict(S.mk(0, 2, map_name="de_not_a_real_map", rng=random.Random(5)))
    expect("a terminal state on an out-of-scope map is flagged reliable False", o["status"] == "terminal" and o["reliable"] is False)
    expect("predict_many(None) / predict_many(5) return [] and never raise", wp.predict_many(None) == [] and wp.predict_many(5) == [])
    expect("predict_many([]) is []", wp.predict_many([]) == [])
    # dead rows: dropped before anything else is read, whatever they carry, for every model kind (W4: health exactly 0 is dead)
    s = S.table_state(rng, 3, 3, planted=0, secs=60.0)
    live = wp.predict(s)
    full_dead = {"side": "CT", "health": 0, "armor": 0, "primary": 0, "equipment_value": 0}
    variants = {"health 0": dict(full_dead), "health 0.0": dict(full_dead, health=0.0), "health '0'": dict(full_dead, health="0"),
                "health -5": dict(full_dead, health=-5), "null loadout": {"side": "T", "health": 0, "armor": None, "primary": None, "equipment_value": None},
                "side+health only": {"side": "CT", "health": 0}, "unknown side": {"side": "weird", "health": 0}}
    for label, dead in variants.items():
        for pos in ("end", "front"):
            pl_ = s["players"] + [dict(dead), dict(dead)] if pos == "end" else [dict(dead), dict(dead)] + s["players"]
            o2 = wp.predict(dict(s, players=pl_))
            expect(f"dead rows ({label}, at the {pos}) must not change the answer and must be reported",
                   o2["p_ct_win"] == live["p_ct_win"] and o2["status"] == live["status"] and any("health <= 0" in w for w in o2["warnings"]))
    # spectator / unassigned rows are on no team: ignored with a warning, never counted as CT or T
    for side in (1, "1", 0, "SPEC", "spectator", "unassigned", "coach"):
        spec = {"side": side, "health": 100, "armor": 0, "primary": 0, "equipment_value": 0}
        o2 = wp.predict(dict(s, players=s["players"] + [spec]))
        expect(f"spectator row (side {side!r}) is ignored, reported, and does not change the answer",
               o2["p_ct_win"] == live["p_ct_win"] and o2["status"] == live["status"] and any("spectator" in w for w in o2["warnings"]))
    o2 = wp.predict(dict(s, players=s["players"] + [{"side": "banana", "health": 100, "armor": 0, "primary": 0, "equipment_value": 0}]))
    expect("an unknown side value makes the table unusable (never a team)", o2["status"] != "ok" and not (o2["status"] == "ok" and o2["p_ct_win"] == live["p_ct_win"]))
    if S.snapshot:
        for key in WPM.PLAYER_REQUIRED:
            s3 = dict(s)
            s3["players"] = [dict(p) for p in s["players"]]
            del s3["players"][0][key]
            o3 = wp.predict(s3)
            expect(f"row missing {key} (no counts given): table unusable -> invalid_input, not zeros", o3["status"] == "invalid_input")
            s3["alive_ct"], s3["alive_t"] = 3, 3
            o4 = wp.predict(s3)
            expect(f"row missing {key} (counts given): fallback model", o4["status"] == ("fallback" if wp.fallback is not None else "invalid_input"))
    S.check("10 input contract: falsy / truthy bomb spellings, bool is not a number, counts >= 0, dead rows (health exactly 0, null loadout), spectators, incomplete rows, reliability flags",
            n_exp[0] >= 60 and not bad10, f"assertions={n_exp[0]} violations={len(bad10)} {bad10[:4]}")


def c_units(S):
    """Plausible-but-wrong inputs must never give a silent ok or terminal."""
    rng, n = S.rng("11"), max(S.n(200), 100)
    base = []
    while len(base) < n:
        s = S.valid_state(rng)
        if s["seconds_since_round_live"] >= 8.0 and (not s["bomb_planted"] or s["seconds_since_plant"] >= 4.0):
            base.append(s)
    plan = [s for s in base if s["bomb_planted"]]
    out = {}
    flagged = lambda o: o["status"] != "ok" or not o["reliable"] or bool(o["warnings"])

    def tally(name, states, pred):
        bad, tot = [], 0
        for s in states:
            o = S.wp.predict(s)
            tot += 1
            if not pred(o):
                bad.append((o["status"], o["reliable"], o["warnings"][:1]))
        out[name] = (tot, bad)

    def mut(states, fn):
        res = []
        for s in states:
            s2 = copy.deepcopy(s)
            fn(s2)
            res.append(s2)
        return res

    inv = lambda o: o["status"] == "invalid_input" and o["reliable"] is False and abs(o["p_ct_win"] - S.prior) < 1e-12
    tally("seconds x64 (ticks) -> invalid_input", mut(base, lambda s: s.__setitem__("seconds_since_round_live", s["seconds_since_round_live"] * 64)), inv)
    tally("seconds x1000 (ms) -> invalid_input", mut(base, lambda s: s.__setitem__("seconds_since_round_live", s["seconds_since_round_live"] * 1000)), inv)
    tally("seconds negative ticks -> invalid_input", mut(base, lambda s: s.__setitem__("seconds_since_round_live", -s["seconds_since_round_live"] * 64)), inv)
    tally("seconds_since_plant x64 (ticks) -> invalid_input", mut(plan, lambda s: s.__setitem__("seconds_since_plant", s["seconds_since_plant"] * 64)), inv)
    tally("planted before the round started (sp = secs + 10) -> invalid_input", mut(plan, lambda s: s.__setitem__("seconds_since_plant", s["seconds_since_round_live"] + 10.0)), inv)
    tally("planted after the clock ran out (secs - sp = 130) -> invalid_input",
          mut(plan, lambda s: s.update(seconds_since_round_live=130.0 + s["seconds_since_plant"])), inv)
    tally("seconds as text -> warned", mut(base, lambda s: s.__setitem__("seconds_since_round_live", str(s["seconds_since_round_live"]))), lambda o: any("text" in w for w in o["warnings"]))
    tally("map key renamed -> out_of_scope, unreliable", mut(base, lambda s: s.__setitem__("map", s.pop("map_name"))),
          lambda o: o["status"] == "out_of_scope" and o["reliable"] is False and any("map not given" in w for w in o["warnings"]))
    tally("mode key renamed -> out_of_scope, unreliable", mut(base, lambda s: s.__setitem__("game_mode", s.pop("mode"))),
          lambda o: o["status"] == "out_of_scope" and o["reliable"] is False and any("mode not given" in w for w in o["warnings"]))
    tally("bomb_planted key renamed -> invalid_input", mut(base, lambda s: s.__setitem__("planted", s.pop("bomb_planted"))), inv)
    tally("seconds key renamed -> invalid_input", mut(base, lambda s: s.__setitem__("secs", s.pop("seconds_since_round_live"))), inv)
    tally("seconds_since_plant key renamed (planted) -> invalid_input", mut(plan, lambda s: s.__setitem__("plant_secs", s.pop("seconds_since_plant"))), inv)
    if S.snapshot:
        def each(s, fn):
            for p in s["players"]:
                fn(p)

        no_counts = lambda s: (s.pop("alive_ct", None), s.pop("alive_t", None))
        for name, fn in (("health as a 0..1 fraction", lambda p: p.__setitem__("health", p["health"] / 100.0)),
                         ("armor as a 0..1 fraction", lambda p: p.__setitem__("armor", p["armor"] / 200.0 + 0.05)),
                         ("armor as a bool", lambda p: p.__setitem__("armor", bool(p["armor"]))),
                         ("health as a bool", lambda p: p.__setitem__("health", True)),
                         ("health key renamed", lambda p: p.__setitem__("hp", p.pop("health"))),
                         ("primary key renamed", lambda p: p.__setitem__("weapon", p.pop("primary"))),
                         ("equipment key renamed", lambda p: p.__setitem__("money", p.pop("equipment_value")))):
            def f(s, fn=fn):
                no_counts(s)
                each(s, fn)
            tally(f"table: {name} -> table unusable (never ok)", mut(base, f), lambda o: o["status"] in ("fallback", "invalid_input"))
        tally("players key renamed (no counts) -> invalid_input", mut(base, lambda s: (no_counts(s), s.__setitem__("player_list", s.pop("players")))), inv)
        tally("equipment x100 -> clamped, warned, unreliable", mut(base, lambda s: each(s, lambda p: p.__setitem__("equipment_value", p["equipment_value"] * 100 + 20000))),
              lambda o: o["status"] == "ok" and o["reliable"] is False and any("clamped" in w for w in o["warnings"]))
        tally("health x10 -> clamped, warned, unreliable", mut(base, lambda s: each(s, lambda p: p.__setitem__("health", p["health"] * 10 + 100))),
              lambda o: o["status"] == "ok" and o["reliable"] is False and any("clamped" in w for w in o["warnings"]))

        def counts_off(s):
            s["alive_ct"], s["alive_t"] = 5, 5
            s["players"] = s["players"][:-1] if len(s["players"]) > 2 else s["players"]
        tally("counts that disagree with the table -> warned", mut([s for s in base if len(s["players"]) > 2], counts_off),
              lambda o: o["status"] == "ok" and any("differ" in w for w in o["warnings"]) or o["status"] != "ok")
    minimum = 15
    short = [k for k, (tot, _) in out.items() if tot < minimum]
    fails = {k: (len(b), b[:1]) for k, (tot, b) in out.items() if b}
    S.check(f"11 units and schema canary: {len(out)} plausible-but-wrong mutations of {n} valid states never give a silent answer",
            not short and not fails and len(out) >= 12,
            f"mutations={len(out)} min population {min(t for t, _ in out.values())}; short={short[:3]} failing={ {k: v for k, v in list(fails.items())[:3]} }")


def c_optional(S):
    if not S.snapshot:
        S.na("12 optional player keys", "alive-count model: the player table is not an input")
        return
    wp, rng = S.wp, S.rng("12")
    read = keys_read_by_features()
    code_optional = read - set(WPM.PLAYER_REQUIRED)
    policy = set(WPM.PLAYER_OPTIONAL)
    S.check("12 optional-key policy covers exactly the keys the feature code reads (a new key must be added to PLAYER_OPTIONAL)",
            code_optional == policy and len(policy) > 0,
            f"read by snapshot_features: {sorted(read)}; policy lists {sorted(policy)}; missing from policy {sorted(code_optional - policy)}; unused in code {sorted(policy - code_optional)}")
    base = S.table_state(rng, 3, 3, planted=1, secs=60.0)  # planted 10 s ago: defusing and kit rows are meaningful
    bad, used_keys, n_keys = [], 0, 0
    for k in sorted(policy):
        n_keys += 1
        full = copy.deepcopy(base)
        for p in full["players"]:
            p[k] = 2 if k in WPM.PLAYER_RANGE else True  # a nonzero value in every row so that deleting the key changes the feature
        o_full, x_full = wp.predict(full), S.x_of(wp, full)
        gone = copy.deepcopy(full)
        for p in gone["players"]:
            p.pop(k)
        o_gone, x_gone = wp.predict(gone), S.x_of(wp, gone)
        affects = x_full is not None and x_gone is not None and not np.array_equal(x_full, x_gone)
        used_keys += int(affects)
        named = any(k in w for w in o_gone["warnings"])
        if o_full["warnings"] or not o_full["reliable"]:
            bad.append(f"{k}: the complete state is not silent")
        if not named:
            bad.append(f"{k}: all rows lack it but no warning names it")
        if affects and o_gone["reliable"]:
            bad.append(f"{k}: all rows lack a key the model uses but reliable stays True")
        if not affects and not o_gone["reliable"]:
            bad.append(f"{k}: reliable False although the model does not use it")
        if bool(wp._opt_used.get(k)) != bool(affects):
            bad.append(f"{k}: predictor thinks used={wp._opt_used.get(k)} but deleting it changes the input row={affects}")
        none_all = copy.deepcopy(full)
        for p in none_all["players"]:
            p[k] = None
        o_none = wp.predict(none_all)
        if not same(o_none, o_gone):
            bad.append(f"{k}: None in every row must behave like an absent key")
        one = copy.deepcopy(full)
        one["players"][0].pop(k)
        o_one = wp.predict(one)
        if not (any(k in w for w in o_one["warnings"]) and o_one["reliable"] is True and o_one["status"] == "ok"):
            bad.append(f"{k}: one row lacking it must warn (naming it) and stay reliable")
        dead_only = copy.deepcopy(full)
        dead_only["players"].append({"side": "CT", "health": 0})
        o_dead = wp.predict(dead_only)
        if any(k in w for w in o_dead["warnings"]) or o_dead["p_ct_win"] != o_full["p_ct_win"]:
            bad.append(f"{k}: a dead row's missing keys must not matter")
    S.check("12 optional player keys: absent keys are named in a warning; reliable False only when absent from every row AND the model uses the feature",
            n_keys > 0 and used_keys > 0 and not bad, f"keys={n_keys} used by the model={used_keys}; problems={bad[:4]}")


def c_scope_contract(S):
    wp = S.wp
    bad, info = [], []
    # scope derived by the predictor == scope derived here == map_is_* features
    if tuple(wp.scope_maps) != tuple(S.maps):
        bad.append(f"predictor scope maps {wp.scope_maps} != meta {S.maps}")
    if set(wp.scope_modes) - set(S.modes) - set(WPM.MODE_ALIASES):
        bad.append(f"predictor modes {wp.scope_modes} != meta {S.modes}")
    cols = [f for f in S.features if f.startswith("map_is_")]
    suffixes = {c[7:] for c in cols}
    if cols and suffixes != {m[3:] if m[2:3] == "_" else m for m in S.maps}:
        bad.append(f"map_is_ features {sorted(suffixes)} != scope maps {sorted(S.maps)}")
    if wp.scope_issues:
        bad.append(f"scope issues at load: {wp.scope_issues[:2]}")
    # the constant defaults are the production scope: they must equal the meta unless this run says another model family is expected
    const_equal = set(WPM.IN_SCOPE_MAPS) == set(S.maps) and set(m for m in WPM.IN_SCOPE_MODES if m) <= set(S.modes) | set(WPM.MODE_ALIASES)
    S.check("13 default constants (IN_SCOPE_MAPS / IN_SCOPE_MODES) equal the model's meta scope", const_equal or S.args.default_scope_may_differ,
            f"constants {tuple(WPM.IN_SCOPE_MAPS)} vs meta {S.maps}" + ("" if const_equal else " (allowed to differ: --default-scope-may-differ)"), warn=not const_equal)
    # needs_players derived independently
    inp = S.meta.get("input")
    expect_players = (inp == "players") if inp in ("players", "counts") else any(f in S.snap_names or f.startswith("st_") for f in S.features)
    if expect_players != S.snapshot:
        bad.append(f"needs_players {S.snapshot} != derived {expect_players}")
    if wp.fallback is not None and wp.fallback.needs_players:
        bad.append("the fallback model needs a player table (it must be an alive-count model)")
    # booster size and iteration
    for label, m in [("primary", wp.primary)] + ([("fallback", wp.fallback)] if wp.fallback is not None else []):
        nf, nr = m.booster.num_features(), m.booster.num_boosted_rounds()
        if nf != len(m.features):
            bad.append(f"{label}: booster has {nf} features, meta lists {len(m.features)}")
        if m.best_iteration + 1 > nr:
            bad.append(f"{label}: best_iteration {m.best_iteration} beyond the {nr} trees saved")
        if len(set(m.features)) != len(m.features):
            bad.append(f"{label}: duplicate feature names")
        pr = m.meta.get("prior_ct_win")
        if not (isinstance(pr, (int, float)) and 0.0 < pr < 1.0):
            bad.append(f"{label}: prior_ct_win {pr!r} is not in (0, 1)")
        info.append(f"{label}: {nf} features, {m.best_iteration + 1}/{nr} trees")
    # row-builder identities on real-valued states: no NaN input, ct_alive_minus_t = alive_ct - alive_t, exactly one map one-hot set (none when out of scope)
    rng = S.rng("13")
    n_rows = 0
    for _ in range(S.n(300)):
        s = S.valid_state(rng)
        prep = wp._prepare(s)
        if "early" in prep:
            bad.append("a valid state was answered without the model")
            break
        model, x = prep["model"], prep["x"][0]
        n_rows += 1
        f = {n: v for n, v in zip(model.features, x)}
        if np.isnan(x).any():
            bad.append("NaN in a model input row of a valid state")
            break
        if "ct_alive_minus_t" in f and "alive_ct" in f and "alive_t" in f and abs(f["ct_alive_minus_t"] - (f["alive_ct"] - f["alive_t"])) > 1e-6:
            bad.append("ct_alive_minus_t != alive_ct - alive_t")
            break
        if model.map_cols:
            hot = [c for c in model.map_cols if f[c] == 1.0]
            want = "map_is_" + (s["map_name"][3:] if s["map_name"][2:3] == "_" else s["map_name"])
            if hot != [want] or sum(f[c] for c in model.map_cols) != 1.0:
                bad.append(f"map one-hot {hot} for {s['map_name']}")
                break
    # meta monotone constraints == what the code derives (a refit would use mono_for(features)); then they hold in the booster
    for label, m in [("primary", wp.primary)] + ([("fallback", wp.fallback)] if wp.fallback is not None else []):
        meta_mono = m.meta.get("monotone_constraints")
        wp_common = None
        try:
            import wp_common
        except Exception as exc:
            S.not_run(f"13 meta monotone_constraints == wp_common.mono_for ({label})", f"wp_common could not be imported: {type(exc).__name__}: {str(exc)[:80]}")
            meta_mono = None
        if meta_mono is None and wp_common is not None:
            S.check(f"13 meta monotone_constraints == wp_common.mono_for(features) ({label})", True, "the meta declares no monotone_constraints (nothing to compare)", warn=True)
        if meta_mono is not None:
            got = [meta_mono.get(f, None) for f in m.features]
            old = os.environ.get("WP_EXTRA_MONO")
            variants = {}
            try:
                for flag in ("0", "1"):
                    os.environ["WP_EXTRA_MONO"] = flag
                    variants[flag] = [int(v) for v in wp_common.mono_for(m.features)]
            finally:
                if old is None:
                    os.environ.pop("WP_EXTRA_MONO", None)
                else:
                    os.environ["WP_EXTRA_MONO"] = old
            which = [f for f, v in variants.items() if v == got]
            S.check(f"13 meta monotone_constraints == wp_common.mono_for(features) ({label}; the sign of every constraint the code would refit with)",
                    bool(which), f"meta {dict((f, c) for f, c in zip(m.features, got) if c)}; code (WP_EXTRA_MONO=0) {dict((f, c) for f, c in zip(m.features, variants['0']) if c)}"
                    + (f"; matches WP_EXTRA_MONO={which[0]}" if which else ""))
        # the booster honours them: sweep each constrained feature through the serving row, other inputs fixed
        cons = {f: c for f, c in (m.meta.get("monotone_constraints") or {}).items() if c and f in m.features}
        predictor = wp if m is wp.primary else S.fbm
        rows = []
        for _ in range(S.n(150)):
            x = S.x_of(predictor, S.valid_state(rng, with_players=(m is wp.primary and S.snapshot)))
            if x is not None:
                rows.append(x)
        swept, viol = 0, []
        for f, c in cons.items():
            j = m.features.index(f)
            st = S.stats["primary" if m is wp.primary else "fallback"][f]
            vals = list(range(0, 6)) if f in ("alive_ct", "alive_t") else list(range(-5, 6)) if f == "ct_alive_minus_t" \
                else list(np.linspace(st[1] - 1, st[2] + 1, 9)) if st[0] > 0 else [0.0, 1.0, 2.0]
            for x in rows:
                M = np.tile(x, (len(vals), 1))
                M[:, j] = vals
                d = np.diff(m.predict_rows(M))
                swept += 1
                if (c > 0 and (d < -1e-6).any()) or (c < 0 and (d > 1e-6).any()):
                    viol.append(f)
        if not cons:
            S.check(f"13 booster honours the meta monotone constraints ({label})", True, "the meta declares no constraint", warn=True)
        else:
            S.check(f"13 booster honours the meta monotone constraints ({label}; each constrained feature swept through the serving row)",
                    swept > 0 and not viol, f"features {sorted(cons)}; sweeps={swept}; violations={len(viol)} {sorted(set(viol))[:3]}")
    S.check("13 scope and feature contract: scope == meta == map_is_* features, no load-time scope issue, booster/meta sizes agree, row-builder identities, derived input mode",
            not bad and n_rows > 0, f"{'; '.join(info)}; rows checked {n_rows}; problems={bad[:3]}")


def c_liveness(S):
    wp, rng = S.wp, S.rng("14")
    problems_const, dead, weak = [], [], []
    for label, m, predictor, snap in [("primary", wp.primary, wp, S.snapshot)] + ([("fallback", wp.fallback, S.fbm, False)] if wp.fallback is not None else []):
        rows = []
        for _ in range(max(S.n(1000), 400)):
            x = S.x_of(predictor, S.valid_state(rng, with_players=snap))
            if x is not None:
                rows.append(x)
        X = np.array(rows)
        if len(X) < 50:
            problems_const.append(f"{label}: only {len(X)} rows")
            continue
        const = [f for j, f in enumerate(m.features) if np.unique(X[:, j]).size < 2]
        if const:
            problems_const.append(f"{label}: constant in the test distribution {const[:6]}")
        st = S.stats["primary" if label == "primary" else "fallback"]
        for f in m.features:
            if st[f][0] == 0:
                dead.append(f"{label}:{f}")
        base = m.predict_rows(X[:300])
        for j, f in enumerate(m.features):
            if st[f][0] == 0:
                continue
            moved = np.zeros(len(base), dtype=bool)
            for v in (st[f][1] - 1.0, st[f][2] + 1.0):
                Xp = X[:300].copy()
                Xp[:, j] = v
                moved |= np.abs(m.predict_rows(Xp) - base) > 1e-9
            if moved.mean() < 0.01:
                weak.append(f"{label}:{f}({moved.mean():.1%})")
    S.check("14 feature liveness: no model input is constant over the valid-state distribution (a feature wired to nothing)", not problems_const,
            f"{problems_const[:3]}")
    S.check(f"14 booster liveness: every input has splits and moves p in >= 1% of states when pushed to its extremes ({len(dead)} never split, {len(weak)} weak)",
            True, f"never split (dead): {dead[:12]}; weak: {weak[:8]}", warn=bool(dead or weak))
    if not S.snapshot:
        S.na("14 table reachability", "alive-count model: no player table")
        return
    # reachability: every input the table can supply moves when SOME player key is perturbed (finds a feature wired to nothing)
    keys = sorted(keys_read_by_features() - {"side"})
    cand = {}
    for k in keys:
        if k in WPM.PLAYER_RANGE:
            lo, hi = WPM.PLAYER_RANGE[k]
            cand[k] = [int(lo), int(hi)] if k in ("health", "armor") else [0, 2] if k not in ("equipment_value",) else [0, 6000]
        elif k == "primary":
            cand[k] = [0] + sorted(set(PRIMARIES) - {0})
        else:
            cand[k] = [False, True]
    reached, key_moves = set(), {}
    prng = S.rng("14r")
    sample_states = [S.table_state(prng, prng.randint(2, 5), prng.randint(2, 5), planted=int(prng.random() < 0.5)) for _ in range(30)]
    for s in sample_states:
        x0 = S.x_of(wp, s)
        for k in keys:
            for side_set in (SIDES_CT, SIDES_T):
                idx = [i for i, p in enumerate(s["players"]) if p["side"] in side_set]
                for v in cand[k]:
                    s2 = copy.deepcopy(s)
                    s2["players"][idx[0]][k] = v
                    x1 = S.x_of(wp, s2)
                    if x0 is None or x1 is None:
                        continue
                    ch = np.nonzero(x0 != x1)[0]
                    if ch.size:
                        key_moves[k] = key_moves.get(k, 0) + 1
                    reached.update(wp.primary.features[j] for j in ch)
    table_feats = [f for f in wp.primary.features if f in S.snap_names or f.startswith("st_")]
    unreached = [f for f in table_feats if f not in reached]
    dead_keys = [k for k in keys if k not in key_moves]
    S.check("14 reachability: every table-derived model input moves when some player key is perturbed (no feature wired to nothing)",
            len(table_feats) > 0 and not unreached, f"{len(table_feats)} table inputs; unreached {unreached[:6]}; keys that move nothing {dead_keys}")


def c_mixed_batch(S):
    wp, rng = S.wp, S.rng("15")
    n = S.n(1500)
    states = [S.valid_state(rng) for _ in range(n)] + [S.corrupt(rng, S.valid_state(rng)) for _ in range(n)]
    states += [S.mk(0, 3), S.mk(3, 0), S.mk(3, 3, planted=1, secs=90.0, sp=45.0), S.mk(3, 3, secs=130.0), S.mk(3, 3, map_name="de_not_a_real_map"),
               S.mk(3, 3, planted="maybe"), S.mk(3, 3, mode="wingman"), S.mk(3, 3, planted=1, bomb_defused=True), S.mk(3, 3, seconds_since_round_live=3840.0)]
    if S.snapshot and wp.fallback is not None:
        states += [dict(S.valid_state(rng, with_players=False), players=g) for g in ([], None, [{"side": "CT"}], "x") for _ in range(25)]
    order = list(range(len(states)))
    rng.shuffle(order)
    shuffled = [states[i] for i in order]
    singles = [wp.predict(s) for s in states]
    batch = wp.predict_many(states)
    batch_sh = wp.predict_many(shuffled)
    inv = [None] * len(order)
    for pos, i in enumerate(order):
        inv[i] = batch_sh[pos]
    d1 = sum(not same(a, b, 1e-7) for a, b in zip(singles, batch))
    d2 = sum(not same(a, b, 1e-7) for a, b in zip(singles, inv))
    sz = [wp.predict_many(states[:k]) for k in (0, 1, 2, 3)]
    d3 = sum(len(sz[k]) != k for k in range(4)) + sum(not same(singles[i], sz[3][i], 1e-7) for i in range(3))
    seen = {o["status"] for o in singles}
    need = {"ok", "terminal", "invalid_input", "out_of_scope"} | ({"fallback"} if (S.snapshot and wp.fallback is not None) else set())
    S.check("15 mixed-batch differential: predict_many over a mixed batch (and the shuffled batch, and tiny batches) == predict, slot by slot",
            d1 == 0 and d2 == 0 and d3 == 0 and need <= seen,
            f"{len(states)} states; differences vs single {d1}, shuffled {d2}, tiny batches {d3}; statuses seen {sorted(seen)}; missing {sorted(need - seen)}")


def c_fallback(S):
    if not S.snapshot or S.fbm is None:
        S.na("16 fallback exactness", "needs a snapshot primary model and a fallback model")
        return
    wp, rng = S.wp, S.rng("16")
    n_fb, bad = 0, []
    n = max(S.n(3000), 600)
    for i in range(n):
        s = S.corrupt(rng, S.valid_state(rng))
        o = wp.predict(s)
        if o["status"] == "fallback" and isinstance(s, dict):
            n_fb += 1
            ref = S.fbm.predict({k: v for k, v in s.items() if k != "players"})
            if ref["p_ct_win"] != o["p_ct_win"] or ref["status"] != "ok" or o["reliable"] != ref["reliable"] or not set(ref["warnings"]) <= set(o["warnings"]):
                bad.append((o["p_ct_win"], ref["p_ct_win"], ref["status"]))
    for g in ([], None, "x", [{}], [{"side": "CT"}, 5]):
        s = S.valid_state(rng, with_players=False)
        s["players"] = g
        o = wp.predict(s)
        ref = S.fbm.predict({k: v for k, v in s.items() if k != "players"})
        n_fb += 1
        if not (o["status"] == "fallback" and o["p_ct_win"] == ref["p_ct_win"]):
            bad.append((g, o["status"], o["p_ct_win"], ref["p_ct_win"]))
    S.check("16 fallback exactness: every state served by 'fallback' equals the alive-count model's own answer to the same state without the table",
            n_fb >= 20 and not bad, f"fallback-served states checked {n_fb} of {n}+5; mismatches {len(bad)} {bad[:2]}")


def _golden_path(S):
    return S.args.golden or os.path.join(os.path.dirname(os.path.abspath(S.args.model)), f"golden_{S.name}.json")


def _golden_cases(S):
    rng = S.rng("17-golden")
    cases = [{"name": f"valid_unplanted_{i}", "state": S.valid_state(rng, planted=False)} for i in range(10)]
    cases += [{"name": f"valid_planted_{i}", "state": S.valid_state(rng, planted=True)} for i in range(10)]
    mode_bad = next(m for m in MODE_UNIVERSE if norm(m) not in S.modes)
    hand = {
        "ct_dead": S.mk(0, 3), "t_dead_unplanted": S.mk(3, 0), "t_dead_planted_10s": S.mk(3, 0, planted=1, secs=60.0, sp=10.0),
        "t_dead_planted_45s": S.mk(3, 0, planted=1, secs=90.0, sp=45.0), "clock_114_9": S.mk(3, 3, secs=114.9), "clock_115": S.mk(3, 3, secs=115.0),
        "plant_40_9": S.mk(3, 3, planted=1, secs=90.0, sp=40.9), "plant_41": S.mk(3, 3, planted=1, secs=90.0, sp=41.0),
        "flag_false_text": S.mk(4, 4, planted="false", secs=100.0), "flag_yes_text": S.mk(4, 4, planted="yes", secs=100.0, sp=20.0),
        "flag_maybe": S.mk(4, 4, planted="maybe"), "defused": S.mk(2, 2, planted=1, secs=70.0, sp=10.0, bomb_defused="true"),
        "oos_map": S.mk(5, 5, map_name="de_not_a_real_map"), "oos_mode": S.mk(5, 5, mode=mode_bad), "blank_map": S.mk(5, 5, map_name=""),
        "ticks": S.mk(5, 5, seconds_since_round_live=3840.0), "impossible_clock": S.mk(5, 5, planted=1, secs=5.0, sp=30.0),
        "text_counts": {k: v for k, v in S.mk(3, 2, rng=random.Random(2), players=None).items() if k != "players"} | {"alive_ct": "3", "alive_t": "2"},
        "nan_seconds": S.mk(5, 5, seconds_since_round_live=float("nan")), "bool_seconds": S.mk(5, 5, seconds_since_round_live=True),
    }
    if S.snapshot:
        pl_ = S.make_players(random.Random(3), 4, 4, True)
        dead = {"side": "CT", "health": 0, "armor": None, "primary": None, "equipment_value": None}
        spec = {"side": "SPEC", "health": 100, "armor": 0, "primary": 0, "equipment_value": 0}
        hand["dead_rows"] = S.mk(4, 4, planted=1, secs=70.0, sp=12.0, players=pl_ + [dead, dict(dead, health=0.0)])
        hand["spectator_row"] = S.mk(4, 4, planted=1, secs=70.0, sp=12.0, players=pl_ + [spec])
        hand["fallback_empty_table"] = dict(S.mk(4, 4, rng=random.Random(4), players=[]), alive_ct=4, alive_t=4)
        hand["health_fraction"] = S.mk(4, 4, players=[dict(p, health=0.5) for p in pl_])
        hand["no_helmet_key"] = S.mk(4, 4, planted=1, secs=70.0, sp=12.0, players=[{k: v for k, v in p.items() if k != "has_helmet"} for p in pl_])
        hand["one_no_kit_key"] = S.mk(4, 4, planted=1, secs=70.0, sp=12.0, players=[{k: v for k, v in p.items() if k != "has_defuser"} if i == 0 else p for i, p in enumerate(pl_)])
    cases += [{"name": k, "state": v} for k, v in hand.items()]
    for c in cases:
        o = S.wp.predict(copy.deepcopy(c["state"]))
        c["expected"] = {k: o[k] for k in ("p_ct_win", "status", "reliable", "warnings")}
    return cases


def c_golden(S):
    path = _golden_path(S)
    nm = "17 golden rows"
    prov = {"model_sha256": sha256_file(S.args.model), "meta_sha256": sha256_file(os.path.splitext(S.args.model)[0] + ".meta.json"),
            "fallback_sha256": sha256_file(S.args.fallback) if S.args.fallback else None}
    if S.args.write_golden:
        cases = _golden_cases(S)
        write_json_atomic(path, {"model": S.name, **prov, "xgboost": __import__("xgboost").__version__, "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                 "n_cases": len(cases), "cases": cases})
        S.check(f"{nm}: written {len(cases)} cases to {os.path.basename(path)}; the next runs compare", len(cases) >= 30, f"{path}")
    if not os.path.isfile(path):
        S.not_run(nm, f"WARN: no golden file {path}; run once with --write-golden after reviewing the model, then every run compares")
        return
    with open(path, encoding="utf-8") as f:
        gold = json.load(f)
    stale = [k for k in prov if gold.get(k) != prov[k]]
    if stale:
        S.check(f"{nm}: the golden file belongs to this model (sha256 of {', '.join(stale)})", False,
                f"golden {os.path.basename(path)} was written for other files ({stale}); a retrain under the same name must be reviewed, then --write-golden")
        return
    diffs = []
    for c in gold["cases"]:
        o = S.wp.predict(copy.deepcopy(c["state"]))
        e = c["expected"]
        if not (abs(o["p_ct_win"] - e["p_ct_win"]) <= 1e-6 and o["status"] == e["status"] and o["reliable"] == e["reliable"] and o["warnings"] == e["warnings"]):
            diffs.append((c["name"], e["status"], o["status"], round(e["p_ct_win"], 6), round(o["p_ct_win"], 6)))
    S.check(f"{nm}: {len(gold['cases'])} stored states reproduce their stored p (1e-6), status, reliable and warnings", len(gold["cases"]) >= 30 and not diffs,
            f"differences={len(diffs)} {diffs[:3]}")


def c_rules(S):
    wp = S.wp
    secs_grid = [0.0, 0.5, 17.0, RULE_BOMB_TIMER - 0.01, RULE_BOMB_TIMER, RULE_BOMB_TIMER + 0.01, 60.0, RULE_ROUND_CLOCK - 0.01, RULE_ROUND_CLOCK,
                 RULE_ROUND_CLOCK + 0.01, 150.0, 154.99, 155.0, RULE_MAX_SECONDS - 0.01, RULE_MAX_SECONDS, RULE_MAX_SECONDS + 0.01, 3840.0, -0.5,
                 -(RULE_MAX_SECONDS + 0.01)]
    sp_grid = [0.0, 10.0, RULE_BOMB_TIMER - 0.01, RULE_BOMB_TIMER, RULE_BOMB_TIMER + 0.01, 100.0, 3840.0, -1.0]
    counts = [(5, 5), (3, 0), (0, 3), (0, 0), (1, 1), (5, 0), (0, 5)]

    def oracle(ac, at, planted, secs, sp):
        if abs(secs) > RULE_MAX_SECONDS:
            return ("invalid",)
        sc = max(0.0, secs)
        if planted:
            if sp < 0 or sp > RULE_MAX_SECONDS or sp > sc + RULE_SLACK or sc - sp > RULE_PLANT_LATEST:
                return ("invalid",)
        if ac == 0:
            return ("terminal", 0.0)
        if at == 0 and not planted:
            return ("terminal", 1.0)
        if planted and sp >= RULE_BOMB_TIMER:
            return ("terminal", 0.0)
        if not planted and sc >= RULE_ROUND_CLOCK:
            return ("terminal", 1.0)
        return ("model",)

    bad, kinds, n = [], {"invalid": 0, "terminal": 0, "model": 0}, 0
    rng = random.Random(21)
    for ac, at in counts:
        for planted in (0, 1):
            for secs in secs_grid:
                for sp in (sp_grid if planted else [-1.0]):
                    s = S.mk(ac, at, planted, secs, sp, rng=rng)
                    exp = oracle(ac, at, planted, secs, sp)
                    o = wp.predict(s)
                    n += 1
                    kinds[exp[0]] += 1
                    ok = (o["status"] == "invalid_input" and o["p_ct_win"] == S.prior and not o["reliable"]) if exp[0] == "invalid" \
                        else (o["status"] == "terminal" and o["p_ct_win"] == exp[1]) if exp[0] == "terminal" \
                        else (o["status"] == "ok" and 0.0 < o["p_ct_win"] < 1.0)
                    if not ok:
                        bad.append((ac, at, planted, secs, sp, exp, o["status"], round(o["p_ct_win"], 4)))
    S.check("18 rule boundary + impossible-state sweep against an independent oracle (41 s bomb timer, 115 s round clock, +-200 s unit limit, clock consistency)",
            n > 1000 and all(v > 50 for v in kinds.values()) and not bad, f"{n} states {kinds}; mismatches {len(bad)} {bad[:2]}")


def c_leak(S):
    wp, rng = S.wp, S.rng("19")
    names = ["label_ct_win", "w_round", "p_pipeline", "round", "tick", "match_key", "era", "steamid", "name", "second", "p"]
    names = [k for k in names if k not in S.features and k not in (S.wp.fallback.features if S.wp.fallback is not None else [])]
    diff, n = 0, S.n(400)
    for _ in range(n):
        s = S.valid_state(rng)
        s2 = copy.deepcopy(s)
        for k in names:
            s2[k] = rng.random()
        for p in s2.get("players", []):
            p.update({"steamid": 76561190000000000 + rng.randint(0, 99), "name": "x", "label_ct_win": 1})
        diff += not (wp.predict(s) == wp.predict(s2))
    S.check("19 extra keys named like labels / grid metadata / player ids leave the answer exactly unchanged", n > 0 and names and diff == 0, f"differences={diff} of {n}; keys {names[:5]}")


def c_ticks(S):
    bad = []
    for tick, freeze, want in ((6400, 5760, 10.0), (5760, 5760, 0.0), (5744, 5760, -0.25), (5792, 5760, 0.5), (5760 + 64 * 115, 5760, 115.0)):
        got = WPM.seconds_from_ticks(tick, freeze)
        if got != want:
            bad.append((tick, freeze, got, want))
    if WPM.TICK_RATE != RULE_TICK_RATE:
        bad.append(("TICK_RATE", WPM.TICK_RATE))
    ref = S.wp.predict(S.mk(3, 3, secs=60.0))
    via = S.wp.predict(S.mk(3, 3, secs=WPM.seconds_from_ticks(5760 + 64 * 60, 5760)))
    if not same(ref, via):
        bad.append("p from converted ticks differs")
    S.check("20 tick helper: seconds_from_ticks(tick, freeze_end) at 64 ticks/s, sign and fractions", not bad, f"{bad[:3]}")


CHECKS = [("0", c_valid), ("1", c_fuzz), ("2", c_scope), ("3", c_wingman), ("4", c_terminal), ("5", c_invalid), ("6", c_monotone),
          ("6b", c_victim), ("7", c_determinism), ("8", c_latency), ("8b", c_explain), ("9", c_real_rows), ("10", c_contract),
          ("11", c_units), ("12", c_optional), ("13", c_scope_contract), ("14", c_liveness), ("15", c_mixed_batch), ("16", c_fallback),
          ("17", c_golden), ("18", c_rules), ("19", c_leak), ("20", c_ticks)]


# ---------------------------------------------------------------- running and reporting
def provenance(args):
    try:
        import xgboost
        xv = xgboost.__version__
    except Exception:
        xv = None
    meta = lambda p: (os.path.splitext(p)[0] + ".meta.json") if p else None
    return {"model_file": os.path.abspath(args.model), "model_sha256": sha256_file(args.model), "meta_sha256": sha256_file(meta(args.model)),
            "fallback_file": os.path.abspath(args.fallback) if args.fallback else None, "fallback_sha256": sha256_file(args.fallback),
            "fallback_meta_sha256": sha256_file(meta(args.fallback)),
            "winprob_py_sha256": sha256_file(os.path.join(HERE, "winprob.py")), "snapshot_features_py_sha256": sha256_file(os.path.join(HERE, "snapshot_features.py")),
            "wp_items_py_sha256": sha256_file(os.path.join(HERE, "wp_items.py")), "stress_script_sha256": sha256_file(os.path.abspath(__file__)),
            "python": sys.version.split()[0], "xgboost": xv, "numpy": np.__version__, "platform": platform.platform(), "seed": args.seed,
            "scale": args.scale, "only": args.only or None}


def run_one(args, report=None):
    name = os.path.splitext(os.path.basename(args.model))[0]
    report = report or args.report or os.path.join(ROOT, "data", "combined", f"stress_{name}.json")
    t0 = time.time()
    rep = {"model": name, "fallback": os.path.splitext(os.path.basename(args.fallback))[0] if args.fallback else None, "status": "running",
           "all_passed": False, "checks": [], "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "provenance": provenance(args)}
    write_json_atomic(report, rep)  # a crash leaves 'running', never the previous run's all_passed
    S = None
    try:
        S = Suite(args)
        only = {x.strip() for x in args.only.split(",") if x.strip()} if args.only else None
        for cid, fn in CHECKS:
            if only is None or cid in only:
                S.run(cid, fn)
        status = "done"
    except Exception:
        tb = traceback.format_exc()
        print("FAIL  stress test aborted before the checks finished:\n" + tb, flush=True)
        rep.update({"status": "error", "error": tb[-3000:]})
        write_json_atomic(report, rep)
        return 1, S
    failures = S.failures
    nr = S.not_run_names
    n_warn = sum(1 for r in S.results if r["status"] == "WARN")
    strict_fail = bool(nr) and args.strict
    rep.update({"status": status, "all_passed": not failures and not nr, "n_checks": len(S.results), "n_failed": len(failures), "n_not_run": len(nr),
                "n_warn": n_warn, "not_run": nr, "failed": failures, "checks": S.results, "not_applicable": S.not_applicable,
                "durations_s": S.durations, "duration_s": round(time.time() - t0, 1), "scope": S.wp.describe()})
    write_json_atomic(report, rep)
    tail = (f" ({len(nr)} NOT RUN: {nr})" if nr else "") + (f" ({n_warn} WARN)" if n_warn else "")
    print("\nRESULT:", ("ALL PASS" if not failures else f"FAILED: {failures}") + tail, f"[{time.time() - t0:.0f}s] -> {report}", flush=True)
    return (1 if failures or strict_fail else 0), S


# ---------------------------------------------------------------- self test: throwaway models with other maps and other features
def _train_throwaway(d):
    import xgboost as xgb

    import wp_common  # mono_for: the constraints a real refit would use
    rng = np.random.default_rng(0)
    out = {}

    def save(name, feats, X, y, scope_maps, fb=None, input_mode=None):
        mono = wp_common.mono_for(feats)
        m = xgb.XGBClassifier(n_estimators=60, max_depth=3, learning_rate=0.2, tree_method="hist", n_jobs=1, random_state=0, monotone_constraints=tuple(mono))
        m.fit(X.astype(np.float32), y)
        path = os.path.join(d, name + ".json")
        m.get_booster().save_model(path)
        meta = {"version": name, "features": feats, "best_iteration": 59, "monotone_constraints": dict(zip(feats, mono)), "prior_ct_win": float(y.mean()),
                "scope": {"maps": scope_maps, "modes": ["competitive"]}, "xgboost": xgb.__version__}
        if input_mode:
            meta["input"] = input_mode
        with open(os.path.join(d, name + ".meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f)
        out[name] = path

    n = 6000
    ac, at = rng.integers(0, 6, n), rng.integers(0, 6, n)
    secs = rng.uniform(0, 114, n)
    mp = rng.integers(0, 2, n)
    lab = lambda z: (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    # A: alive-count model, 3 numeric inputs + 2 one-hot maps (a map set the real model never saw)
    fa = ["alive_ct", "alive_t", "seconds_since_round_live", "map_is_vertigo", "map_is_overpass"]
    XA = np.stack([ac, at, secs, mp == 0, mp == 1], 1)
    save("tt_counts", fa, XA, lab(0.9 * (ac - at) + 0.01 * (secs - 57) * (ac - at) + 0.3 * (mp == 0)), ["de_vertigo", "de_overpass"])
    # C: alive-count model with extra inputs (hp_ct, hp_t) and no map input at all
    hp_c, hp_t = rng.uniform(0, 500, n), rng.uniform(0, 500, n)
    fc = ["alive_ct", "alive_t", "ct_alive_minus_t", "seconds_since_round_live", "hp_ct", "hp_t"]
    XC = np.stack([ac, at, ac - at, secs, hp_c, hp_t], 1)
    save("tt_hp", fc, XC, lab(0.6 * (ac - at) + 0.004 * (hp_c - hp_t)), ["de_inferno"])
    # B: snapshot model: a few table aggregates, other maps, the alive-count model A as its fallback
    states = []
    for i in range(n):
        a, t = max(1, ac[i]), max(1, at[i])
        pl_ = [{"side": "CT" if j < a else "T", "health": int(rng.integers(1, 101)), "armor": int(rng.choice([0, 100])), "primary": int(rng.choice([0, 7, 16, 9])),
                "equipment_value": int(rng.integers(0, 7000)), "has_helmet": bool(rng.random() < 0.5), "has_defuser": bool(j < a and rng.random() < 0.5),
                "is_defusing": bool(j < a and rng.random() < 0.15)} for j in range(a + t)]
        states.append((a, t, loadout_from_players(pl_)))
    fb = ["alive_ct", "alive_t", "bomb_planted", "seconds_since_round_live", "seconds_since_plant", "map_is_vertigo", "map_is_overpass",
          "st_hp_ct", "st_hp_t", "st_equip_diff", "st_defusing_ct", "st_helmet_n_t"]
    planted = (rng.random(n) < 0.4).astype(int)
    sp = np.where(planted == 1, rng.uniform(0, 40, n), -1.0)
    XB = np.array([[a, t, planted[i], secs[i], sp[i], mp[i] == 0, mp[i] == 1, f["st_hp_ct"], f["st_hp_t"], f["st_equip_diff"], f["st_defusing_ct"], f["st_helmet_n_t"]]
                   for i, (a, t, f) in enumerate(states)])
    yB = lab(0.5 * (XB[:, 0] - XB[:, 1]) + 0.01 * (XB[:, 7] - XB[:, 8]) + 0.0004 * XB[:, 9] - 1.0 * planted)
    save("tt_snap", fb, XB, yB, ["de_vertigo", "de_overpass"])
    return out


def self_test(args):
    d = tempfile.mkdtemp(prefix="wp_stress_selftest_")
    code, summary = 0, []
    try:
        paths = _train_throwaway(d)
        cases = [("tt_counts (3 numeric inputs + 2 maps, other map set)", paths["tt_counts"], None),
                 ("tt_hp (extra inputs hp_ct / hp_t, no map input)", paths["tt_hp"], None),
                 ("tt_snap (player table, other maps, fallback tt_counts)", paths["tt_snap"], paths["tt_counts"])]
        for label, model, fb in cases:
            for write in (True, False):
                ns = argparse.Namespace(**{**vars(args), "model": model, "fallback": fb, "grid": None, "scale": args.scale if args.scale != 1.0 else 0.05,
                                           "default_scope_may_differ": True, "write_golden": write, "golden": None, "only": None, "self_test": False,
                                           "strict": False, "report": os.path.join(d, f"report_{os.path.basename(model)}")})
                print(f"\n===== SELF TEST: {label} ({'write golden' if write else 'compare golden'}) =====", flush=True)
                rc, S = run_one(ns)
                not_run = [x for x in (S.not_run_names if S else []) if not x.startswith("9 ")]  # only the real-row check may be absent here
                if not_run:
                    print(f"FAIL  self test: checks that could not run: {not_run}", flush=True)
                    rc = 1
                summary.append((label, write, rc, (list(S.failures) + not_run) if S else ["aborted"]))
                code |= rc
        # negative controls: a stress test that cannot fail proves nothing
        meta_path = os.path.join(d, "tt_counts.meta.json")
        meta = json.load(open(meta_path, encoding="utf-8"))
        for tag, edit, expect in (("scope map without a model input", lambda m: m["scope"]["maps"].append("de_extra"), "13"),
                                  ("alive_t constraint sign flipped in the meta", lambda m: m["monotone_constraints"].__setitem__("alive_t", 1), "13")):
            m2 = copy.deepcopy(meta)
            edit(m2)
            name = "tt_neg_" + str(abs(hash(tag)) % 10000)
            shutil.copy(os.path.join(d, "tt_counts.json"), os.path.join(d, name + ".json"))
            json.dump(m2, open(os.path.join(d, name + ".meta.json"), "w", encoding="utf-8"))
            ns = argparse.Namespace(**{**vars(args), "model": os.path.join(d, name + ".json"), "fallback": None, "grid": None, "scale": 0.02,
                                       "default_scope_may_differ": True, "write_golden": False, "golden": None, "only": expect, "self_test": False,
                                       "strict": False, "report": os.path.join(d, f"report_{name}.json")})
            print(f"\n===== SELF TEST negative control: {tag} (check {expect} must FAIL) =====", flush=True)
            rc, S = run_one(ns)
            caught = rc == 1 and S is not None and any(f.startswith("13") for f in S.failures)
            print(("PASS" if caught else "FAIL") + f"  negative control '{tag}' is caught by check {expect}", flush=True)
            summary.append((f"negative control: {tag}", None, 0 if caught else 1, [] if caught else ["not caught"]))
            code |= 0 if caught else 1
    finally:
        shutil.rmtree(d, ignore_errors=True)
    print("\nSELF TEST SUMMARY")
    for label, write, rc, fails in summary:
        print(f"  {'ok  ' if rc == 0 else 'FAIL'} {label} {'' if write is None else '(write)' if write else '(compare)'} {fails if fails else ''}")
    print("SELF TEST:", "ALL PASS" if code == 0 else "FAILED", flush=True)
    return code


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--fallback", default=None)
    ap.add_argument("--grid", default=None, help="export sample parquet for the real-row checks (default: data/combined/export_<model>_test_sample.parquet if present)")
    ap.add_argument("--report", default=None, help="where to write the JSON report (default data/combined/stress_<model>.json)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=float, default=1.0, help="multiplies every sample size (default 1.0)")
    ap.add_argument("--only", default="", help="comma list of check ids to run, e.g. 0,13,17")
    ap.add_argument("--write-golden", action="store_true", help="store ~40 states and their exact outputs in the golden file, then compare")
    ap.add_argument("--golden", default=None, help="golden file (default: models/golden_<model name>.json next to the model)")
    ap.add_argument("--strict", action="store_true", help="a check that could not run (NOT RUN) makes the exit code 1")
    ap.add_argument("--default-scope-may-differ", action="store_true",
                    help="the winprob IN_SCOPE_* constants may differ from this model's meta scope (a model family other than the production one)")
    ap.add_argument("--self-test", action="store_true", help="train throwaway models with other maps / features in a temp dir and run everything on them")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test(args)
    if not args.model:
        ap.error("--model is required")
    return run_one(args)[0]


if __name__ == "__main__":
    sys.exit(main())
