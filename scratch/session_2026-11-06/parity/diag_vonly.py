"""Diagnose the tome-only rows (V_only) against the laptop's grid and laptop rows. Reads the chunk pages already cached
under /mnt/sagemaker-nvme/parity by parity_compare.py; reads S3 nothing new. Writes nothing to S3.

Classifies every V_only row (in-admin sample matches only) by:
  A  round absent from the laptop's status rows for that match   (laptop dropped the round)
  B  round present, player absent from that round                 (laptop dropped the player: side filter?)
  C  round and player present, tick absent                        (tick-level difference)
and whether its (match, round, tick) is in the laptop grid.
"""
import glob
import json
import os

import polars as pl

HERE = os.path.expanduser("~/adam/status_scripts/parity")
WORK = "/mnt/sagemaker-nvme/parity"
GRID = os.path.expanduser("~/adam/status/data/combined/grid.parquet")
JOIN = ["match_key", "round", "tick", "player_id"]
MK_R = ["match_key", "round"]

keys = pl.read_csv(os.path.join(HERE, "sample_keys.csv"))["match_key"].to_list()
rep = json.load(open(os.path.join(HERE, "parity_report.json")))
in_admin = [m for m in keys if m not in set(rep["not_in_admin"])]

L = (pl.read_parquet(os.path.join(HERE, "laptop_status_1hz_sample.parquet"))
     .filter(pl.col("match_key").is_in(in_admin))
     .select(JOIN).with_columns([pl.col(c).cast(pl.Int64) for c in ["round", "tick", "player_id"]]))
pages = sorted(glob.glob(os.path.join(WORK, "c*", "dataframe_*")))
V = (pl.concat([pl.scan_parquet(p).filter(pl.col("match_key").is_in(in_admin)).select(JOIN).collect() for p in pages],
               how="vertical_relaxed")
     .with_columns([pl.col(c).cast(pl.Int64) for c in ["round", "tick", "player_id"]]))
print(f"in-admin sample matches {len(in_admin)}; laptop rows {L.height}; tome rows {V.height}", flush=True)

G = (pl.read_parquet(GRID, columns=["match_key", "round", "tick"])
     .filter(pl.col("match_key").is_in(in_admin))
     .with_columns([pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)]).unique())
print(f"grid rows for these matches {G.height}; grid matches {G['match_key'].n_unique()}", flush=True)

Lr = L.select(MK_R).unique().with_columns(pl.lit(1).alias("roundInL"))
Lp = L.select(MK_R + ["player_id"]).unique().with_columns(pl.lit(1).alias("playerInL"))
Vonly = V.join(L, on=JOIN, how="anti")
Vonly = (Vonly.join(Lr, on=MK_R, how="left")
         .join(Lp, on=MK_R + ["player_id"], how="left")
         .join(G.with_columns(pl.lit(1).alias("inG")), on=["match_key", "round", "tick"], how="left")
         .with_columns(pl.col("inG").is_not_null().alias("inG"),
                       pl.col("roundInL").is_not_null().alias("roundInL"),
                       pl.col("playerInL").is_not_null().alias("playerInL")))
Vonly = Vonly.with_columns(pl.when(~pl.col("roundInL")).then(pl.lit("A_round_absent_in_laptop"))
                           .when(~pl.col("playerInL")).then(pl.lit("B_player_absent_in_round_laptop"))
                           .otherwise(pl.lit("C_tick_absent_laptop")).alias("class"))
summary = (Vonly.group_by(["class", "inG"]).agg(pl.len().alias("rows")).sort(["class", "inG"]))
print("V_only classification (class x in-grid):", flush=True)
print(summary, flush=True)

# grid vs tome round sets, and grid ticks missing from the tome
Vr = V.select(MK_R).unique().with_columns(pl.lit(1).alias("inV"))
Gr = G.select(MK_R).unique().with_columns(pl.lit(1).alias("inG"))
rounds_tome_not_grid = Vr.join(Gr, on=MK_R, how="anti").height
rounds_grid_not_tome = Gr.join(Vr, on=MK_R, how="anti").height
G_not_V = G.join(V, on=["match_key", "round", "tick"], how="anti").height
print(f"rounds in tome not in grid: {rounds_tome_not_grid}; rounds in grid not in tome: {rounds_grid_not_tome}; "
      f"grid ticks not in tome: {G_not_V}", flush=True)

# per-round: tome-only ticks that fall after the laptop's last tick in that round (tail) or before its first (head)
lt = L.group_by(MK_R).agg(pl.col("tick").min().alias("L_min"), pl.col("tick").max().alias("L_max"))
C = Vonly.filter(pl.col("class") == "C_tick_absent_laptop").join(lt, on=MK_R, how="left")
tail = C.filter(pl.col("tick") > pl.col("L_max")).height
head = C.filter(pl.col("tick") < pl.col("L_min")).height
inside = C.height - tail - head
print(f"class C rows: after laptop's last tick in round {tail}; before its first {head}; inside {inside}", flush=True)
print("sample class C rows:", C.select(JOIN + ["L_min", "L_max", "inG"]).head(8).to_dicts(), flush=True)

out = {"classes": summary.to_dicts(), "rounds_tome_not_grid": rounds_tome_not_grid,
       "rounds_grid_not_tome": rounds_grid_not_tome, "grid_ticks_not_in_tome": G_not_V,
       "classC_tail": tail, "classC_head": head, "classC_inside": inside}
with open(os.path.join(HERE, "diag_vonly.json"), "w") as f:
    json.dump(out, f, indent=1, default=str)
print("wrote", os.path.join(HERE, "diag_vonly.json"), flush=True)
