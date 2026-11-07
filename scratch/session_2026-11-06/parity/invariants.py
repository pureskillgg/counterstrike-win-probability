"""Invariant check on the reduced player_status tome rows, against rounds.parquet (freeze, end_eff, use).

For the sample matches that have tome rows, in rounds the grid uses (use == True):
  - every row: health > 0, freeze <= tick < end_eff, (tick - freeze) % 64 == 0
  - per round: the tome's tick set equals the expected grid {freeze + 64 k < end_eff}
Reads cached pages and local copies on the space only. Writes nothing to S3.
"""
import glob
import json
import os

import polars as pl

HERE = os.path.expanduser("~/adam/status_scripts/parity")
COMB = os.path.expanduser("~/adam/status/data/combined")
WORK = "/mnt/sagemaker-nvme/parity"
MK_R = ["match_key", "round"]

rep = json.load(open(os.path.join(HERE, "parity_report2.json")))
in_tome = [m["match_key"] for m in rep["per_match"] if m["V_rows"] > 0]

pages = sorted(glob.glob(os.path.join(WORK, "c*", "dataframe_*")))
V = (pl.concat([pl.scan_parquet(p).filter(pl.col("match_key").is_in(in_tome))
                .select(["match_key", "round", "tick", "player_id", "health"]).collect() for p in pages],
               how="vertical_relaxed")
     .with_columns([pl.col("round").cast(pl.Int64), pl.col("tick").cast(pl.Int64)]))
R = (pl.read_parquet(os.path.join(COMB, "rounds.parquet"), columns=["match_key", "round", "freeze", "end_eff", "use"])
     .filter(pl.col("match_key").is_in(in_tome))
     .with_columns([pl.col("round").cast(pl.Int64), pl.col("freeze").cast(pl.Int64), pl.col("end_eff").cast(pl.Int64)]))
VR = V.join(R, on=MK_R, how="left")
print(f"tome rows (in-tome matches) {VR.height}; rows with no rounds.parquet entry {VR.filter(pl.col('freeze').is_null()).height}",
      flush=True)
U = VR.filter(pl.col("use") == True)  # noqa: E712
print(f"rows in use rounds {U.height}", flush=True)

bad_health = U.filter(pl.col("health") <= 0).height
bad_window = U.filter((pl.col("tick") < pl.col("freeze")) | (pl.col("tick") >= pl.col("end_eff"))).height
bad_grid = U.filter(((pl.col("tick") - pl.col("freeze")) % 64) != 0).height
print(f"health<=0: {bad_health}; outside [freeze, end_eff): {bad_window}; off the 64-tick grid: {bad_grid}", flush=True)

# expected tick set per use round: freeze + 64k < end_eff
RU = R.filter(pl.col("use") == True)  # noqa: E712
RU = RU.with_columns(
    pl.when(pl.col("end_eff") > pl.col("freeze"))
    .then((pl.col("end_eff") - pl.col("freeze") + 63) // 64).otherwise(0).alias("n_exp"))
got = U.group_by(MK_R).agg(pl.col("tick").n_unique().alias("n_got"))
cmp = RU.join(got, on=MK_R, how="left").with_columns(pl.col("n_got").fill_null(0))
short = cmp.filter(pl.col("n_got") < pl.col("n_exp")).height
over = cmp.filter(pl.col("n_got") > pl.col("n_exp")).height
exact = cmp.filter(pl.col("n_got") == pl.col("n_exp")).height
print(f"use rounds {cmp.height}: tick count exact {exact}, fewer than expected {short}, more than expected {over}",
      flush=True)
print("sample short rounds:", cmp.filter(pl.col("n_got") < pl.col("n_exp")).head(5).select(
    MK_R + ["freeze", "end_eff", "n_exp", "n_got"]).to_dicts(), flush=True)
out = {"rows_in_use_rounds": U.height, "health_le_0": bad_health, "outside_window": bad_window,
       "off_64_grid": bad_grid, "use_rounds": cmp.height, "tick_count_exact": exact,
       "tick_count_fewer": short, "tick_count_more": over}
with open(os.path.join(HERE, "invariants.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote invariants.json", flush=True)
