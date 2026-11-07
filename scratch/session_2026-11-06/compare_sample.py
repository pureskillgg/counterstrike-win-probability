import glob
import hashlib
import json
import os
import re
import sys
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
import polars as pl

SCR = "C:/Users/13144/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad"
sys.path.insert(0, "G:/pureskill/csds-cs2/scripts/serving")
import win_probability as WP  # noqa: E402

MODEL = "winprob_v6_drop_heavy"
meta = json.load(open("G:/pureskill/csds-cs2/models/" + MODEL + ".meta.json", encoding="utf-8"))
FEATS = meta["features"]
COL = {f: i for i, f in enumerate(FEATS)}
assert len(FEATS) == 57


class FakeBst:
    X = None

    def predict(self, X, iteration_range=None):
        self.X = np.asarray(X, dtype=np.float64)
        return np.zeros(self.X.shape[0])


fake = FakeBst()
WP.xgb.DMatrix = lambda X, *a, **k: X
WP._cache[MODEL] = (fake, FEATS, 0)

SIDE_NAMES = ["alive", "hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle",
              "sniper_other", "smg", "heavy", "no_primary", "flash", "smoke", "he", "fire", "decoy", "equip", "money"]
NAMES = ["CT_" + x for x in SIDE_NAMES] + ["T_" + x for x in SIDE_NAMES]
assert len(NAMES) == 44
ST19 = ["hp", "armor_n", "armor_sum", "helmet_n", "kit_n", "defusing", "ak", "m4", "awp", "rifle", "sniper_other", "smg",
        "no_primary", "flash", "smoke", "he", "fire", "decoy", "equip"]
DIFF6 = ["hp", "rifle", "equip", "armor_n", "awp", "smg"]
MAPS = ["ancient", "anubis", "cache", "dust2", "inferno", "mirage", "nuke"]
MAPCOL = [COL["map_is_" + m] for m in MAPS]

exp = json.load(open(SCR + "/export_sample.json"))
KEYS = list(exp.keys())
LIMIT = int(os.environ.get("LIMIT", "0")) or len(KEYS)
TOME = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"
PM = "E:/pureskill_tomes/pertick_status_1001_1008/per_match"

t0 = time.time()
hdr = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(TOME + "/header/dataframe_*"))],
                how="vertical_relaxed").filter(pl.col("key").is_in(KEYS))
CH = ["round_state", "round_start", "round_end", "player_info", "player_death", "bomb_state"]
chan = {}
for name in CH:
    folders = sorted(glob.glob(os.path.join(TOME, f"basic_{name}*")))
    files = [f for d in folders if os.path.isdir(d) for f in sorted(glob.glob(os.path.join(d, "dataframe_*")))]
    if not files:
        print("no pages for", name)
        continue
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    if "match_key" in df.columns:
        df = df.filter(pl.col("match_key").is_in(KEYS))
    chan[name] = df
    print("channel", name, "rows (sample matches)", df.height, flush=True)
print("header rows (sample)", hdr.height, "| loaded in", round(time.time() - t0, 1), "s", flush=True)


def pm_path(key):
    return os.path.join(PM, hashlib.sha1(key.encode("utf-8")).hexdigest()[:16] + ".parquet")


stats = defaultdict(lambda: {"n": 0, "bad": 0, "maxd": 0.0, "ex": []})


def cmp(name, a, b, tol, ctx):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    d = np.abs(a - b)
    bad = d > tol
    s = stats[name]
    s["n"] += int(a.size)
    s["bad"] += int(bad.sum())
    if a.size:
        s["maxd"] = max(s["maxd"], float(d.max()))
    if bad.any() and len(s["ex"]) < 3:
        i = int(np.flatnonzero(bad)[0])
        s["ex"].append((ctx, float(a[i]), float(b[i])))


round_stats = Counter()
warn_counter = Counter()
t1 = time.time()
for mi, key in enumerate(KEYS[:LIMIT]):
    rec = exp[key]
    data = {"header": hdr.filter(pl.col("key") == key)}
    for name, df in chan.items():
        data[name] = df.filter(pl.col("match_key") == key)
    data["player_status"] = pl.read_parquet(pm_path(key))
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        probs = WP.win_probability(data, team="CT", model_name=MODEL)
    for msg in WP.last_warnings:
        warn_counter[re.sub(r"\d+", "#", msg)[:90]] += 1
    if not probs:
        round_stats["match_no_serving_output"] += 1
        continue
    X = fake.X
    ticks = sorted(probs)
    assert X.shape[0] == len(ticks)
    tidx = {t: i for i, t in enumerate(ticks)}
    exp_ticks = set()
    for rd in rec["rounds"]:
        exp_ticks.update(rd["freeze"] + 64 * kk for kk in range(len(rd["alive"])))
    srv_ticks = set(ticks)
    round_stats["serving_seconds"] += len(srv_ticks)
    round_stats["serving_seconds_outside_export_rounds"] += len(srv_ticks - exp_ticks)
    round_stats["export_seconds"] += len(exp_ticks)
    round_stats["export_seconds_not_in_serving"] += len(exp_ticks - srv_ticks)
    for rd in rec["rounds"]:
        round_stats["export_rounds"] += 1
        f = rd["freeze"]
        plant = rd["plant"]
        alive = np.asarray(rd["alive"], dtype=np.float64)
        n = alive.shape[0]
        rows = []
        k = 0
        while (f + 64 * k) in tidx:
            rows.append(tidx[f + 64 * k])
            k += 1
        m = len(rows)
        if m == 0:
            round_stats["freeze_tick_not_in_serving"] += 1
            continue
        round_stats["rounds_matched"] += 1
        if m == n:
            round_stats["same_length"] += 1
        elif m > n:
            round_stats["serving_longer"] += 1
            round_stats["extra_serving_seconds"] += m - n
        else:
            round_stats["serving_shorter"] += 1
            round_stats["missing_serving_seconds"] += n - m
        kk = min(n, m)
        idx = np.asarray(rows[:kk])
        ks = np.arange(kk, dtype=np.float64)
        tk = f + 64.0 * ks
        ctx = f"{key[-14:]} rn{rd['rn']}"
        cmp("alive_ct (per second)", X[idx, COL["alive_ct"]], alive[:kk, 0], 1e-9, ctx)
        cmp("alive_t (per second)", X[idx, COL["alive_t"]], alive[:kk, 1], 1e-9, ctx)
        cmp("ct_alive_minus_t", X[idx, COL["ct_alive_minus_t"]], alive[:kk, 0] - alive[:kk, 1], 1e-9, ctx)
        cmp("seconds_since_round_live", X[idx, COL["seconds_since_round_live"]], ks, 1e-9, ctx)
        planted = (plant >= 0) & (tk >= plant)
        cmp("bomb_planted", X[idx, COL["bomb_planted"]], planted.astype(np.float64), 1e-9, ctx)
        ssp = np.where(planted, (tk - plant) / 64.0, -1.0)
        cmp("seconds_since_plant", X[idx, COL["seconds_since_plant"]], ssp, 1e-6, ctx)
        onehot = np.zeros(len(MAPS))
        onehot[rd["map"]] = 1.0
        cmp("map_is_* (7 one-hot)", X[np.ix_(idx, MAPCOL)], np.tile(onehot, (kk, 1)), 1e-9, ctx)
        # per-second loadout is not in the export: only the freeze-second (k = 0) snapshot can be compared
        lo = np.asarray(rd["loadout"], dtype=np.float64)
        r0 = rows[0]
        for side, pref in (("ct", "CT"), ("t", "T")):
            for fld in ST19:
                cmp(f"st_{fld}_{side} (k=0 only)", X[r0, COL[f"st_{fld}_{side}"]], lo[NAMES.index(f"{pref}_{fld}")], 1e-3, ctx)
        for fld in DIFF6:
            cmp(f"st_{fld}_diff (k=0 only)", X[r0, COL[f"st_{fld}_diff"]],
                lo[NAMES.index("CT_" + fld)] - lo[NAMES.index("T_" + fld)], 1e-3, ctx)
        cmp("alive_ct at k=0 vs export loadout CT_alive", X[r0, COL["alive_ct"]], lo[NAMES.index("CT_alive")], 1e-9, ctx)
        cmp("alive_t at k=0 vs export loadout T_alive", X[r0, COL["alive_t"]], lo[NAMES.index("T_alive")], 1e-9, ctx)
        cmp("alive_ct at k=0 vs export steps_alive", X[r0, COL["alive_ct"]], alive[0, 0], 1e-9, ctx)
        cmp("alive_t at k=0 vs export steps_alive", X[r0, COL["alive_t"]], alive[0, 1], 1e-9, ctx)
        cmp("export loadout CT_alive vs steps_alive k=0", lo[NAMES.index("CT_alive")], alive[0, 0], 1e-9, ctx)
        cmp("export loadout T_alive vs steps_alive k=0", lo[NAMES.index("T_alive")], alive[0, 1], 1e-9, ctx)

print(f"== {min(LIMIT, len(KEYS))} matches processed in {round(time.time() - t1, 1)} s")
print("== round / length counts:", dict(round_stats))
print("== serving warnings (top):")
for msg, c in warn_counter.most_common(8):
    print(f"   {c:6d}  {msg}")
print("== per-feature comparison (compared values, mismatches beyond tolerance, max abs diff)")
for name, s in stats.items():
    print(f"{name:46s} n {s['n']:9d}  bad {s['bad']:7d}  maxdiff {s['maxd']:.6g}")
    for ex in s["ex"][:2]:
        print(f"      example {ex[0]} serving {ex[1]} export {ex[2]}")
