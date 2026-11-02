"""Known-answer unit checks on synthetic data (seconds, no store needed). Shared by the selftests, test_snapshot_parity.py and e2e_checks.py.

Each function returns a core.Res. The expected values come from an independent description (a pinned weapon-name table, hand-computed
numbers), never from the code under test, so a fault that two paths share (a wrong constant in wp_items, a clip in ll_vec) still shows.
"""
import math
import os
import random
import tempfile

import numpy as np

from checks.core import Res, FAIL, PASS, fail, ok, verdict

# ---- pinned CS2 item definition indexes (weapon_fire.weapon_name verified on the 2026 data) and their class
CODE_OF_NAME = {"ak47": 7, "aug": 8, "awp": 9, "famas": 10, "g3sg1": 11, "galilar": 13, "m249": 14, "m4a1": 16, "mac10": 17, "p90": 19, "mp5sd": 23,
                "ump45": 24, "xm1014": 25, "bizon": 26, "mag7": 27, "negev": 28, "sawedoff": 29, "mp7": 33, "mp9": 34, "nova": 35, "scar20": 38,
                "sg556": 39, "ssg08": 40, "m4a1_silencer": 60}
CLASS_OF_NAME = {**{n: "RIFLE" for n in ("ak47", "aug", "famas", "galilar", "m4a1", "sg556", "m4a1_silencer")}, "awp": "AWP",
                 **{n: "SNIPER_OTHER" for n in ("g3sg1", "scar20", "ssg08")},
                 **{n: "SMG" for n in ("mac10", "p90", "mp5sd", "ump45", "bizon", "mp7", "mp9")},
                 **{n: "HEAVY" for n in ("m249", "xm1014", "mag7", "negev", "sawedoff", "nova")}}
SIDE_FIELDS = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle", "sniper_other", "smg", "heavy", "no_primary",
               "flash", "smoke", "he", "fire", "decoy", "equip"]
DIFF_FIELDS = ["hp", "rifle", "equip", "armor_n", "awp", "smg"]
ST_NAMES = {f"st_{f}_{s}" for f in SIDE_FIELDS for s in ("ct", "t")} | {f"st_{d}_diff" for d in DIFF_FIELDS}


def _codes(cls):
    return {CODE_OF_NAME[n] for n, c in CLASS_OF_NAME.items() if c == cls}


# ------------------------------------------------------------------ item pins (F3)
def item_pins_result():
    import wp_items as WI
    bad = []
    fam = {"RIFLE": WI.RIFLE, "AWP": WI.AWP, "SNIPER_OTHER": WI.SNIPER_OTHER, "SMG": WI.SMG, "HEAVY": WI.HEAVY}
    for name, got in fam.items():
        want = _codes(name)
        if set(got) != want:
            bad.append(f"{name}: wp_items has {sorted(got)}, pinned {sorted(want)} (missing {sorted(want - set(got))}, extra {sorted(set(got) - want)})")
    if set(WI.AK) != {7}:
        bad.append(f"AK {WI.AK} != (7,)")
    if set(WI.M4) != {16, 60}:
        bad.append(f"M4 {WI.M4} != (16, 60)")
    if (WI.SIDE_T, WI.SIDE_CT) != (2, 3):
        bad.append(f"SIDE_T/SIDE_CT {(WI.SIDE_T, WI.SIDE_CT)} != (2, 3)")
    if list(WI.SIDE_FIELDS) != SIDE_FIELDS:
        bad.append(f"SIDE_FIELDS {list(WI.SIDE_FIELDS)} != pinned order")
    if list(WI.DIFF_FIELDS) != DIFF_FIELDS:
        bad.append(f"DIFF_FIELDS {list(WI.DIFF_FIELDS)} != {DIFF_FIELDS}")
    allc = [c for s in fam.values() for c in s]
    if len(allc) != len(set(allc)):
        bad.append("an item code is in two classes")
    for n, code in CODE_OF_NAME.items():
        homes = [k for k, s in fam.items() if code in s]
        if homes != [CLASS_OF_NAME[n]]:
            bad.append(f"{n} (code {code}) is in {homes}, should be in [{CLASS_OF_NAME[n]}]")
    return verdict(not bad, f"{len(CODE_OF_NAME)} named primary weapons, {len(fam)} classes" + ("; " + "; ".join(bad) if bad else ""), n=len(CODE_OF_NAME))


# ------------------------------------------------------------------ reference implementation of the loadout aggregates
def _side_name(v):
    k = v.strip().lower() if isinstance(v, str) else v
    return {"ct": "ct", "3": "ct", 3: "ct", "t": "t", "2": "t", 2: "t"}[k]


def reference_loadout(players):
    """The 46 numbers by the definition in features_status.py's docstring, written from the pinned weapon-name table."""
    out = {}
    cls = {c: _codes(c) for c in ("RIFLE", "AWP", "SNIPER_OTHER", "SMG", "HEAVY")}
    for side in ("ct", "t"):
        ps = [p for p in players if _side_name(p["side"]) == side]
        g = lambda k: [float(p.get(k, 0) or 0) for p in ps]  # noqa: E731
        prim = [int(p.get("primary", 0) or 0) for p in ps]
        v = {"hp": sum(g("health")), "armor_n": sum(1 for a in g("armor") if a > 0), "armor_sum": sum(g("armor")), "helmet_n": sum(1 for p in ps if p.get("has_helmet")),
             "kit_n": sum(1 for p in ps if p.get("has_defuser")), "defusing": sum(1 for p in ps if p.get("is_defusing")),
             "ak": sum(c == 7 for c in prim), "m4": sum(c in (16, 60) for c in prim), "awp": sum(c == 9 for c in prim), "rifle": sum(c in cls["RIFLE"] for c in prim),
             "sniper_other": sum(c in cls["SNIPER_OTHER"] for c in prim), "smg": sum(c in cls["SMG"] for c in prim), "heavy": sum(c in cls["HEAVY"] for c in prim),
             "no_primary": sum(c == 0 for c in prim), "flash": sum(g("flashbang")), "smoke": sum(g("smokegrenade")), "he": sum(g("hegrenade")),
             "fire": sum(g("molotov")) + sum(g("incgrenade")), "decoy": sum(g("decoy")), "equip": sum(g("equipment_value"))}
        for f in SIDE_FIELDS:
            out[f"st_{f}_{side}"] = float(v[f])
    for d in DIFF_FIELDS:
        out[f"st_{d}_diff"] = out[f"st_{d}_ct"] - out[f"st_{d}_t"]
    return out


def random_players(rng, n_ct=None, n_t=None, side_style="int"):
    codes = [0] + sorted(CODE_OF_NAME.values())
    ps = []
    for side, n in ((3, rng.randint(0, 5) if n_ct is None else n_ct), (2, rng.randint(0, 5) if n_t is None else n_t)):
        for _ in range(n):
            s = side
            if side_style == "mixed":
                s = rng.choice([side, str(side), "CT" if side == 3 else "T", "ct" if side == 3 else "t"])
            ps.append({"side": s, "health": rng.randint(1, 100), "armor": rng.choice([0, 0, rng.randint(1, 100)]), "has_helmet": rng.random() < 0.5,
                       "has_defuser": side == 3 and rng.random() < 0.5, "is_defusing": rng.random() < 0.05, "primary": rng.choice(codes),
                       "flashbang": rng.randint(0, 2), "smokegrenade": rng.randint(0, 1), "hegrenade": rng.randint(0, 1), "molotov": rng.randint(0, 1),
                       "incgrenade": rng.randint(0, 1), "decoy": rng.randint(0, 1), "equipment_value": rng.randint(0, 8700)})
    return ps


def known_answer_loadout():
    """Hand-built players with hand-computed answers, then a seeded random battery against the reference implementation."""
    from snapshot_features import loadout_from_players
    hand = [{"side": 3, "health": 100, "armor": 100, "has_helmet": True, "has_defuser": True, "is_defusing": False, "primary": 16, "flashbang": 1,
             "smokegrenade": 1, "hegrenade": 0, "molotov": 0, "incgrenade": 1, "decoy": 0, "equipment_value": 5200},
            {"side": 3, "health": 50, "armor": 0, "has_helmet": False, "has_defuser": False, "is_defusing": False, "primary": 9, "flashbang": 2,
             "smokegrenade": 0, "hegrenade": 0, "molotov": 0, "incgrenade": 0, "decoy": 0, "equipment_value": 4750},
            {"side": 3, "health": 7, "armor": 0, "has_helmet": False, "has_defuser": False, "is_defusing": False, "primary": 0, "flashbang": 0,
             "smokegrenade": 0, "hegrenade": 0, "molotov": 0, "incgrenade": 0, "decoy": 0, "equipment_value": 200},
            {"side": 2, "health": 80, "armor": 50, "has_helmet": False, "has_defuser": False, "is_defusing": False, "primary": 7, "flashbang": 0,
             "smokegrenade": 1, "hegrenade": 1, "molotov": 1, "incgrenade": 0, "decoy": 0, "equipment_value": 3900},
            {"side": "T", "health": 1, "armor": 0, "has_helmet": False, "has_defuser": False, "is_defusing": False, "primary": 34, "flashbang": 0,
             "smokegrenade": 0, "hegrenade": 0, "molotov": 0, "incgrenade": 0, "decoy": 1, "equipment_value": 1250}]
    got = loadout_from_players(hand)
    want_hand = {"st_hp_ct": 157.0, "st_hp_t": 81.0, "st_armor_n_ct": 1.0, "st_armor_n_t": 1.0, "st_armor_sum_ct": 100.0, "st_armor_sum_t": 50.0,
                 "st_m4_ct": 1.0, "st_awp_ct": 1.0, "st_ak_t": 1.0, "st_smg_t": 1.0, "st_smg_ct": 0.0, "st_no_primary_ct": 1.0, "st_rifle_ct": 1.0,
                 "st_rifle_t": 1.0, "st_fire_ct": 1.0, "st_fire_t": 1.0, "st_kit_n_ct": 1.0, "st_helmet_n_ct": 1.0, "st_flash_ct": 3.0, "st_decoy_t": 1.0,
                 "st_equip_ct": 10150.0, "st_armor_n_diff": 0.0, "st_awp_diff": 1.0, "st_smg_diff": -1.0, "st_hp_diff": 76.0, "st_equip_diff": 5000.0}
    bad = [f"{k}: {got.get(k)} != {v}" for k, v in want_hand.items() if got.get(k) != v]
    miss, extra = sorted(ST_NAMES - set(got)), sorted(set(got) - ST_NAMES)
    if miss or extra:
        bad.append(f"serving keys: missing {miss}, unexpected {extra}")
    rng = random.Random(12345)
    n_rows, worst = 0, 0.0
    for style in ("int", "mixed"):
        for _ in range(300):
            ps = random_players(rng, side_style=style)
            a, b = loadout_from_players(ps), reference_loadout(ps)
            n_rows += 1
            for k, v in b.items():
                d = abs(a.get(k, float("nan")) - v)
                if not d <= 1e-9:
                    bad.append(f"random row {n_rows}: {k} serving {a.get(k)} vs reference {v}")
                    break
                worst = max(worst, d)
            if len(bad) > 5:
                break
    return verdict(not bad, f"{len(want_hand)} hand-computed values and {n_rows} random player tables; " + ("; ".join(bad[:6]) if bad else "all equal"), n=n_rows)


# ------------------------------------------------------------------ polars training path vs python serving path on a synthetic player table
STATUS_SCHEMA = None


def write_status_day(path, ticks):
    """ticks: list of (match_key, round, tick, players). Writes a status_1hz-shaped parquet (one row per living player)."""
    import polars as pl
    rows = []
    for mk, rd, tk, ps in ticks:
        for i, p in enumerate(ps):
            rows.append({"match_key": mk, "round": rd, "tick": tk, "second": 0, "player_id": i + 1, "side": int(_side_name(p["side"]) == "ct") + 2,
                         "health": p["health"], "armor": p["armor"], "has_helmet": bool(p["has_helmet"]), "has_defuser": bool(p["has_defuser"]),
                         "money": 0, "equipment_value_calc": p["equipment_value"], "is_defusing": bool(p["is_defusing"]), "inv_primary": p["primary"],
                         "inv_secondary": 0, "inv_flashbang": p["flashbang"], "inv_smokegrenade": p["smokegrenade"], "inv_hegrenade": p["hegrenade"],
                         "inv_molotov": p["molotov"], "inv_incgrenade": p["incgrenade"], "inv_decoy": p["decoy"], "inv_c4": 0})
    sch = {"match_key": pl.String, "round": pl.Int32, "tick": pl.Int64, "second": pl.Int32, "player_id": pl.Int32, "side": pl.Int8, "health": pl.Int16,
           "armor": pl.Int16, "has_helmet": pl.Boolean, "has_defuser": pl.Boolean, "money": pl.Int32, "equipment_value_calc": pl.Int32, "is_defusing": pl.Boolean,
           "inv_primary": pl.Int16, "inv_secondary": pl.Int16, "inv_flashbang": pl.Int16, "inv_smokegrenade": pl.Int16, "inv_hegrenade": pl.Int16,
           "inv_molotov": pl.Int16, "inv_incgrenade": pl.Int16, "inv_decoy": pl.Int16, "inv_c4": pl.Int16}
    pl.DataFrame(rows, schema=sch).write_parquet(path)


def polars_vs_python_synthetic():
    import polars as pl
    import features_status as FS
    from snapshot_features import loadout_from_players
    rng = random.Random(777)
    ticks = [("csds/2026/09/01/synthetic-a/csds", 1 + i // 6, 6400 + 64 * (i % 6), random_players(rng, side_style="int")) for i in range(36)]
    ticks.append(("csds/2026/09/01/synthetic-a/csds", 99, 64, random_players(rng, n_ct=0, n_t=3)))  # nobody alive on one side
    with tempfile.TemporaryDirectory(prefix="wp_status_") as d:
        write_status_day(os.path.join(d, "2026-09-01.parquet"), ticks)
        G = pl.DataFrame({"match_key": [t[0] for t in ticks], "round": [t[1] for t in ticks], "tick": [t[2] for t in ticks]}, schema_overrides={"round": pl.Int64})
        F = FS.loadout_features(G, d)
    bad = []
    py_keys = set(loadout_from_players(ticks[0][3] or []).keys())
    pol_keys = set(F.columns)
    if pol_keys != py_keys:
        bad.append(f"column sets differ: only polars {sorted(pol_keys - py_keys)}, only python {sorted(py_keys - pol_keys)}")
    n = 0
    for i, t in enumerate(ticks):
        py = loadout_from_players(t[3])
        for c, v in py.items():
            if c in F.columns:
                d_ = abs(float(F[c][i]) - v)
                if not d_ <= 1e-3:
                    bad.append(f"row {i} {c}: polars {F[c][i]} vs python {v}")
        n += 1
        if len(bad) > 6:
            break
    return verdict(not bad, f"{n} synthetic snapshots, {len(F.columns)} columns; " + ("; ".join(bad[:6]) if bad else "polars training features equal the python serving features"), n=n)


def inconsistent_rounds_synthetic():
    """features_status.inconsistent_rounds must flag a round when ANY side's table count differs from the grid's at any tick (an OR)."""
    import polars as pl
    import features_status as FS
    rng = random.Random(5)
    mk = "csds/2026/09/02/synthetic-b/csds"
    grid_rows, ticks = [], []

    def add(rd, tk, a_ct, a_t, n_ct, n_t):
        grid_rows.append((mk, rd, tk, a_ct, a_t))
        ticks.append((mk, rd, tk, random_players(rng, n_ct=n_ct, n_t=n_t)))
    for tk in (64, 128, 192):
        add(1, tk, 5, 5, 5, 5)    # consistent
        add(2, tk, 4, 5, 5, 5)    # only CT disagrees
        add(3, tk, 5, 4, 5, 5)    # only T disagrees
        add(4, tk, 5, 5, 4, 4)    # both disagree
    add(5, 64, 5, 5, 5, 5)
    add(5, 128, 5, 5, 5, 5)
    grid_rows.append((mk, 5, 192, 5, 5))   # round 5: no table rows at all at the last tick (both counts 0 vs 5)
    G = pl.DataFrame(grid_rows, schema={"match_key": pl.String, "round": pl.Int64, "tick": pl.Int64, "alive_ct": pl.Int8, "alive_t": pl.Int8}, orient="row")
    with tempfile.TemporaryDirectory(prefix="wp_incons_") as d:
        write_status_day(os.path.join(d, "2026-09-02.parquet"), ticks)
        got = set(FS.inconsistent_rounds(G, d)["round"].to_list())
    want = {2, 3, 4, 5}
    return verdict(got == want, f"flagged rounds {sorted(got)}, expected {sorted(want)} (a one-sided disagreement must flag the round)", n=len(want))


# ------------------------------------------------------------------ wp_common known answers (C1, C3, C4, split rule C2)
def ll_vec_golden(W):
    got = {"p=1e-9,y=1": float(W.ll_vec([1e-9], [1])[0]), "p=1-1e-9,y=0": float(W.ll_vec([1 - 1e-9], [0])[0]), "p=.5,y=1": float(W.ll_vec([0.5], [1])[0]),
           "p=0,y=1": float(W.ll_vec([0.0], [1])[0]), "p=.9,y=1": float(W.ll_vec([0.9], [1])[0])}
    want = {"p=1e-9,y=1": 20.723265837, "p=1-1e-9,y=0": 20.723265837, "p=.5,y=1": math.log(2), "p=0,y=1": 34.538776395, "p=.9,y=1": -math.log(0.9)}
    bad = [f"{k}: {got[k]:.6f} != {want[k]:.6f}" for k in want if abs(got[k] - want[k]) > 1e-3]
    return verdict(not bad, "ll_vec known answers: " + ("; ".join(bad) if bad else ", ".join(f"{k} -> {v:.4f}" for k, v in got.items())), n=len(want))


def ci_golden(W):
    import polars as pl
    a = np.arange(1001, dtype=float)
    ci = W._ci(a)
    bad = []
    if abs(ci[0] - 25.0) > 1e-9 or abs(ci[1] - 975.0) > 1e-9:
        bad.append(f"_ci(0..1000) = {ci}, expected [25.0, 975.0] (the 2.5th and 97.5th percentiles)")
    rng = np.random.default_rng(7)
    m, per = 80, 25
    mk = np.repeat([f"m{i}" for i in range(m)], per)
    p = rng.uniform(0.2, 0.8, m * per)
    y = (rng.random(m * per) < 0.5).astype(float)
    sub = pl.DataFrame({"match_key": mk, "w_round": np.ones(m * per)})
    out = W.evaluate(sub, y, p)
    lo, hi = out["logloss_ci95"]
    ml = W.ll_vec(p, y).reshape(m, per).mean(1)
    half = 1.96 * ml.std(ddof=1) / math.sqrt(m)
    ratio = (hi - lo) / (2 * half)
    if not (lo < out["logloss"] < hi) or not 0.8 <= ratio <= 1.25:
        bad.append(f"evaluate(): 95% interval [{lo:.4f}, {hi:.4f}] around {out['logloss']:.4f}, width {ratio:.2f}x the normal-theory width (expected 0.8-1.25)")
    return verdict(not bad, "match-cluster bootstrap interval: " + ("; ".join(bad) if bad else f"percentiles exact, width {ratio:.2f}x normal theory"), n=m * per)


def mono_golden(W):
    old = os.environ.pop("WP_EXTRA_MONO", None)
    try:
        v = W.mono_for(["alive_ct", "alive_t", "ct_alive_minus_t", "bomb_planted", "seconds_since_plant"])
        pin = dict(W.MONO_ALIVE)
    finally:
        if old is not None:
            os.environ["WP_EXTRA_MONO"] = old
    bad = []
    if pin != {"alive_ct": 1, "alive_t": -1, "ct_alive_minus_t": 1}:
        bad.append(f"MONO_ALIVE = {pin}")
    if v != [1, -1, 1, 0, 0]:
        bad.append(f"mono_for -> {v}, expected [1, -1, 1, 0, 0]")
    return verdict(not bad, "monotone constraints (more CT alive helps CT, more T alive hurts CT): " + ("; ".join(bad) if bad else "pinned"), n=5)


def split_rule_result(W, frame, test_frac=0.2, val_frac=0.1, seed=0):
    """The legacy random split has the documented sizes per era: test = round(test_frac * n), validation = round(val_frac * rest). A swap of
    validation and test (or a wrong fraction) changes them."""
    import polars as pl
    try:
        sp = W.make_split("random_legacy", frame, seed=seed)
        how = "random_legacy"
    except ValueError:
        old = getattr(W, "SPLIT_MANIFEST", None)
        W.SPLIT_MANIFEST = "none"
        try:
            sp = W.make_split("random", frame, seed=seed)
        finally:
            W.SPLIT_MANIFEST = old
        how = "random"
    bad, lines, n_tot = [], [], 0
    for era in sorted(frame["era"].unique().to_list()):
        keys = set(frame.filter(pl.col("era") == era)["match_key"].to_list())
        n = len(keys)
        n_test = round(test_frac * n)
        n_val = round(val_frac * (n - n_test))
        got = (len(sp["train"] & keys), len(sp["val"] & keys), len(sp["test"] & keys))
        want = (n - n_test - n_val, n_val, n_test)
        lines.append(f"{era}: n={n} got {got} want {want}")
        n_tot += n
        if got != want:
            bad.append(f"{era} {got} != {want}")
    disjoint = not (sp["train"] & sp["val"]) and not (sp["train"] & sp["test"]) and not (sp["val"] & sp["test"])
    return verdict(not bad and disjoint, f"legacy rule ({how}), sizes (train, val, test): " + "; ".join(lines) + (f"; WRONG: {bad}" if bad else "") + ("" if disjoint else "; NOT DISJOINT"), n=n_tot)


def split_rule_synthetic(W):
    import polars as pl
    keys = [f"csds/{era}/01/01/{i:05d}/csds" for era in ("2024", "2026") for i in range(100 if era == "2024" else 150)]
    frame = pl.DataFrame({"match_key": keys, "era": [k.split("/")[1] for k in keys], "match_date": ["2026-01-01"] * len(keys)})
    return split_rule_result(W, frame)


SUITE = {
    "item_pins": item_pins_result,
    "loadout_known_answers": known_answer_loadout,
    "polars_vs_python": polars_vs_python_synthetic,
    "inconsistent_rounds": inconsistent_rounds_synthetic,
}
SUITE_WP = {"ll_vec_golden": ll_vec_golden, "ci_golden": ci_golden, "mono_golden": mono_golden, "split_rule": split_rule_synthetic}


def run_suite(items, *args):
    """Run (name, fn) pairs; an exception is a FAIL. Returns [(name, Res)]."""
    import traceback
    out = []
    for name, fn in items.items():
        try:
            res = fn(*args)
        except Exception as exc:  # noqa: BLE001
            res = Res(FAIL, f"raised {type(exc).__name__}: {str(exc)[:300]}\n{traceback.format_exc()[-600:]}")
        out.append((name, res))
    return out
