"""Laptop side of the c_exposed float64 fix.

Recomputes the every-2-second position features for the export's matches (export_clean_v2 match keys only) with the
float64 cone test (make_pos_features_f64.py), and compares them with the existing laptop day files
G:/pureskill/csds-cs2/data/pos_features_1hz/every2_sites_v1/<day>.parquet (read only, never written).

Every input is filtered to export match keys before any feature is computed. Days: 2026-09-01 .. 2026-10-25,
because no export key is dated after 2026-10-25 (the 2026-10-26..31 files are not opened).
Outputs (scratch only): laptop_f64/<day>.parquet, out/laptop_<day>.json, out/laptop_<day>_cone.csv
  usage: py -3.10 cexp_laptop_run.py [day ...]
"""
import glob
import json
import os
import sys
import time

SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCR)
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import make_pos_features_f64 as MF  # noqa: E402  (fixed copy; module import only, main() is not called)

ROOT = "G:/pureskill/csds-cs2"
KEYS = json.load(open(os.path.join(SCR, "match_keys.json"), encoding="utf-8"))
KEYS_SET = set(KEYS)
SITES = {m: np.array(v, dtype="float64") for m, v in json.load(open(os.path.join(SCR, "sites_v1.json"), encoding="utf-8")).items()}
K = ["match_key", "round", "tick"]
CONE = ["c_sees_enemy", "t_sees_enemy", "c_exposed", "t_exposed"]
BLOCK = 100
OUT = os.path.join(SCR, "out")
FIXED = os.path.join(SCR, "laptop_f64")
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIXED, exist_ok=True)


def scan(path):
    return pl.scan_parquet(path).filter(pl.col("match_key").is_in(KEYS)).collect()


G = (pl.scan_parquet(f"{ROOT}/data/combined/grid.parquet")
     .filter((pl.col("era") == "2026") & pl.col("match_key").is_in(KEYS))
     .select("match_key", "round", "tick", "bomb_planted", "seconds_since_plant", "map_name", "era", "second")
     .collect()
     .select("match_key", "round", "tick", "bomb_planted", "seconds_since_plant", "map_name"))
PL = pl.concat([pl.read_parquet(f) for f in sorted(glob.glob(f"{ROOT}/data/positions_1hz/_plants/*.parquet"))], how="vertical_relaxed")
PL = PL.filter(pl.col("match_key").is_in(KEYS))
planted_xy = (PL.with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)).sort("tick", maintain_order=True)
              .unique(["match_key", "round", "tick"], keep="last", maintain_order=True)
              .select("match_key", "round", "tick", pl.col("x").alias("bx"), pl.col("y").alias("by")))
print("export keys", len(KEYS), "| grid rows", G.height, "| planted rows", planted_xy.height, flush=True)

all_days = sorted(os.path.basename(f)[:-len(".parquet")] for f in glob.glob(f"{ROOT}/data/positions_1hz/2026-*.parquet"))
days = [d for d in all_days if "2026-09-01" <= d <= "2026-10-25"]
if len(sys.argv) > 1:
    days = [d for d in sys.argv[1:]]
print("days", len(days), days[0], "..", days[-1], flush=True)

for day in days:
    t0 = time.time()
    fp = f"{ROOT}/data/positions_1hz/{day}.parquet"
    fs = f"{ROOT}/data/status_1hz/{day}.parquet"
    fb = f"{ROOT}/data/blind_events/{day}.parquet"
    fe = f"{ROOT}/data/pos_features_1hz/every2_sites_v1/{day}.parquet"
    pos = scan(fp).with_columns(pl.col("round").cast(pl.Int64))
    if pos.height == 0:
        print(day, "no export matches", flush=True)
        continue
    sta = scan(fs).with_columns(pl.col("round").cast(pl.Int64))
    ev = scan(fb) if os.path.exists(fb) else None
    mk = pos["match_key"].unique().to_list()
    Gd = G.filter(pl.col("match_key").is_in(mk))
    PLd = planted_xy.filter(pl.col("match_key").is_in(mk))
    parts = []
    for ci in range(0, len(mk), BLOCK):
        cm = mk[ci:ci + BLOCK]
        parts.append(MF.feats_for_day(pos.filter(pl.col("match_key").is_in(cm)), sta.filter(pl.col("match_key").is_in(cm)),
                                      Gd.filter(pl.col("match_key").is_in(cm)), PLd.filter(pl.col("match_key").is_in(cm)), SITES, 2,
                                      ev.filter(pl.col("match_key").is_in(cm)) if ev is not None else None))
    f = pl.concat(parts, how="vertical_relaxed").with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
    f.write_parquet(os.path.join(FIXED, f"{day}.parquet"))
    feats = [c for c in f.columns if c not in K]
    rec = {"day": day, "matches": len(mk), "rows_new": f.height, "secs": round(time.time() - t0, 1)}
    if os.path.exists(fe):
        ex = pl.read_parquet(fe).filter(pl.col("match_key").is_in(KEYS)).with_columns(pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
        rec["rows_existing"] = ex.height
        j = ex.join(f, on=K, how="full", suffix="_new", coalesce=True)
        rec["only_existing"] = int(j.filter(pl.col(feats[0] + "_new").is_null() & pl.col(feats[0]).is_not_null()).height)
        rec["only_new"] = int(j.filter(pl.col(feats[0]).is_null() & pl.col(feats[0] + "_new").is_not_null()).height)
        mism = {}
        anyc = None
        for c in feats:
            neq = ~((pl.col(c) == pl.col(c + "_new")) | (pl.col(c).is_null() & pl.col(c + "_new").is_null()))
            mism[c] = int(j.select(neq.fill_null(True).sum()).item())
            if c in CONE:
                anyc = neq.fill_null(True) if anyc is None else (anyc | neq.fill_null(True))
        rec["mismatch"] = mism
        rec["cone_mismatch_total"] = sum(mism[c] for c in CONE)
        rec["non_cone_mismatch_total"] = sum(v for c, v in mism.items() if c not in CONE)
        chg = j.filter(anyc).select(K + [x for c in CONE for x in (c, c + "_new")])
        chg.write_csv(os.path.join(OUT, f"laptop_{day}_cone.csv"))
        rec["cone_changed_rows"] = chg.height
    with open(os.path.join(OUT, f"laptop_{day}.json"), "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    summary = {k: rec[k] for k in ("day", "matches", "rows_new", "secs") if k in rec}
    for k in ("rows_existing", "only_existing", "only_new", "cone_mismatch_total", "non_cone_mismatch_total", "cone_changed_rows"):
        if k in rec:
            summary[k] = rec[k]
    print(json.dumps(summary), flush=True)
print("DONE", flush=True)
