"""Identity test for the per-tick bomb-position lookup (laptop, read-only on the data folder).

Recomputes whole days with make_pos_features_v2.feats_for_day (new per-tick asof lookup) and compares every
column with the existing day file (old per-round lookup). Output goes to the scratchpad only.
Days: the first day, and the two days with the most rounds holding several bomb_begin_plant events.
"""
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
SCR = r"C:\Users\13144\AppData\Local\Temp\claude\44708acea14da7d4\047ee5fa-b35b-4b25-97b1-beaf96834ff2\scratchpad"

spec = importlib.util.spec_from_file_location("v2", os.path.join(ROOT, "scripts", "data", "make_pos_features_v2.py"))
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)

sites = {m: np.array(v, dtype="float64") for m, v in json.load(open(os.path.join(ROOT, "data", "combined", "sites_v1.json"), encoding="utf-8")).items()}
G = pl.read_parquet(os.path.join(ROOT, "data", "combined", "grid.parquet"),
                    columns=["match_key", "round", "tick", "bomb_planted", "seconds_since_plant", "map_name", "era", "second"]).filter(pl.col("era") == "2026")

plant_files = sorted(glob.glob(os.path.join(ROOT, "data", "positions_1hz", "_plants", "*.parquet")))
multi = {}
for f in plant_files:
    p = pl.read_parquet(f).group_by("match_key", "round").len()
    multi[os.path.basename(f)] = int((p["len"] > 1).sum())
top = sorted(multi, key=lambda k: -multi[k])[:2]
days = sorted(set([os.path.basename(plant_files[0])] + top))
print("days", days, "multi-begin rounds", {d: multi[d] for d in days}, flush=True)

PL = pl.concat([pl.read_parquet(f) for f in plant_files], how="vertical_relaxed")
planted_xy = (PL.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).sort("tick", maintain_order=True)
              .unique(["match_key", "round", "tick"], keep="last", maintain_order=True)
              .select("match_key", "round", "tick", pl.col("x").alias("bx"), pl.col("y").alias("by")))

KEYS = ["match_key", "round", "tick"]
summary = {}
for day in days:
    fp = os.path.join(ROOT, "data", "positions_1hz", day)
    fs = os.path.join(ROOT, "data", "status_1hz", day)
    fb = os.path.join(ROOT, "data", "blind_events", day)
    fold = os.path.join(ROOT, "data", "pos_features_1hz", "every2_sites_v1", day)
    pos = pl.read_parquet(fp).with_columns(pl.col("round").cast(pl.Int64))
    sta = pl.read_parquet(fs).with_columns(pl.col("round").cast(pl.Int64))
    mk = pos["match_key"].unique().to_list()
    Gd = G.filter(pl.col("match_key").is_in(mk)).select("match_key", "round", "tick", "bomb_planted", "seconds_since_plant", "map_name")
    ev = pl.read_parquet(fb) if os.path.exists(fb) else None
    parts = []
    for ci in range(0, len(mk), 300):
        cm = mk[ci:ci + 300]
        parts.append(v2.feats_for_day(pos.filter(pl.col("match_key").is_in(cm)), sta.filter(pl.col("match_key").is_in(cm)),
                                      Gd.filter(pl.col("match_key").is_in(cm)), planted_xy.filter(pl.col("match_key").is_in(cm)), sites, 2,
                                      ev.filter(pl.col("match_key").is_in(cm)) if ev is not None else None))
    new = pl.concat(parts, how="vertical_relaxed").with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    old = pl.read_parquet(fold).with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    print(f"{day}: new {new.height:,} rows x {new.width} | old {old.height:,} rows x {old.width}", flush=True)
    if new.height != old.height or set(new.columns) != set(old.columns):
        print("  SHAPE/COLUMN MISMATCH", set(new.columns) ^ set(old.columns), flush=True)
        summary[day] = "shape mismatch"
        continue
    a = new.sort(KEYS)
    b = old.sort(KEYS)
    same_keys = bool((a.select(KEYS).to_numpy() == b.select(KEYS).to_numpy()).all())
    diffs = {}
    for c in a.columns:
        if c in KEYS:
            continue
        x = a[c].cast(pl.Float64).to_numpy()
        y = b[c].cast(pl.Float64).to_numpy()
        ne = ~((np.isnan(x) & np.isnan(y)) | (x == y))
        diffs[c] = int(ne.sum())
    bad = {c: n for c, n in diffs.items() if n}
    print(f"  keys identical: {same_keys} | columns compared {len(diffs)} | columns with any difference: {bad if bad else 'none'}", flush=True)
    bomb = [c for c in a.columns if "bomb" in c or c == "defuse_slack"]
    print("  bomb-related null counts new/old:", {c: (int(a[c].is_null().sum()), int(b[c].is_null().sum())) for c in bomb}, flush=True)
    summary[day] = {"keys_identical": same_keys, "differing_values": bad}

# synthetic check of the lookup semantics: two plants in one round, probe ticks before, between and after them
syn_plants = pl.DataFrame({"match_key": ["m", "m"], "round": [1, 1], "tick": [1000, 2000], "bx": [10.0, 500.0], "by": [0.0, 0.0]}).with_columns(
    pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
probe = pl.DataFrame({"match_key": ["m"] * 4, "round": [1] * 4, "tick": [900, 1000, 1500, 2600]}).with_columns(
    pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).with_row_index("kid")
got = probe.sort("tick").join_asof(syn_plants.sort("tick"), on="tick", by=["match_key", "round"], strategy="backward").sort("kid")
print("synthetic bx at ticks 900, 1000, 1500, 2600 ->", got["bx"].to_list(), "(expect [None, 10.0, 10.0, 500.0])", flush=True)

with open(os.path.join(SCR, "causal_identity_test.json"), "w", encoding="utf-8") as fh:
    json.dump({k: v for k, v in summary.items()}, fh, indent=1, default=str)
print("done", flush=True)
