"""P1 check: is 'money at freeze end' remaining cash after buying, and is the first 1 Hz snapshot the freeze end?

- money + equipment_value_calc per player in pistol rounds (if ~800, money is post-buy cash
  and equipment_value_calc is what was spent or held).
- first-snapshot tick vs basic_round_state round_freeze_end tick (same match, round).
- coverage of rounds 1 and 13 for the 150 sample matches in the tome round_end.
- armor values seen for players with armor > 0.
"""
import glob
import os
from collections import Counter

import numpy as np
import polars as pl

ROOT = r"G:\pureskill\csds-cs2"
SAMPLE = os.path.join(ROOT, "data", "state_sample", "players_1hz.parquet")
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")


def chan(name):
    folder = glob.glob(os.path.join(TOME, name + ".*"))[0]
    return sorted(glob.glob(os.path.join(folder, "dataframe_*")))


cols = ["match_key", "round", "tick", "player_id", "team_code", "money",
        "equipment_value_calc", "armor", "has_helmet", "inv_secondary"]
lf = pl.scan_parquet(SAMPLE).select(cols)
t0 = lf.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("t0"))
fs = lf.join(t0, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("t0")).collect()

print("== money + equipment_value_calc per player at first snapshot ==")
for r in (1, 13, 2, 14):
    d = fs.filter(pl.col("round") == r)
    tot = (d["money"] + d["equipment_value_calc"]).to_numpy()
    print(f"round {r:>2}: n={len(tot)} quantiles p5={np.percentile(tot, 5):.0f} p25={np.percentile(tot, 25):.0f} "
          f"p50={np.percentile(tot, 50):.0f} p75={np.percentile(tot, 75):.0f} p95={np.percentile(tot, 95):.0f}; "
          f"share ==800: {np.mean(tot == 800):.3f}; share within +-50 of 800: {np.mean(np.abs(tot - 800) <= 50):.3f}")
    mon = d["money"].to_numpy()
    print(f"          money alone: p5={np.percentile(mon, 5):.0f} p50={np.percentile(mon, 50):.0f} p95={np.percentile(mon, 95):.0f}; "
          f"equipment_value_calc alone: p50={np.percentile(d['equipment_value_calc'].to_numpy(), 50):.0f}")

print("\n== armor values for players with armor > 0, pistol rounds ==")
pist = fs.filter(pl.col("round").is_in([1, 13]))
vals = Counter(pist.filter(pl.col("armor") > 0)["armor"].to_list())
print("armor values (count):", sorted(vals.items()))
print("has_helmet True count in pistol rounds:", int(pist["has_helmet"].sum()))

print("\n== first snapshot tick vs round_freeze_end tick (basic_round_state) ==")
rs = (pl.scan_parquet(chan("basic_round_state")).select(["match_key", "round", "tick", "event_type"])
      .filter(pl.col("event_type") == "round_freeze_end").collect())
print("round_freeze_end rows:", rs.height, " distinct (match, round):",
      rs.group_by(["match_key", "round"]).len().height)
fe = rs.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("freeze_tick"))
t0_df = t0.collect()
j = t0_df.join(fe, on=["match_key", "round"], how="left")
print("sample (match,round) with freeze_end tick:", j.filter(pl.col("freeze_tick").is_not_null()).height,
      " of", j.height)
jj = j.filter(pl.col("freeze_tick").is_not_null()).with_columns((pl.col("t0") - pl.col("freeze_tick")).alias("d"))
dd = jj["d"].to_numpy()
print("first-snapshot tick minus freeze_end tick: quantiles min=%d p25=%.0f p50=%.0f p75=%.0f max=%d" % (
    dd.min(), np.percentile(dd, 25), np.percentile(dd, 50), np.percentile(dd, 75), dd.max()))
print("share exactly 0: %.4f  share within +-64 ticks: %.4f" % (np.mean(dd == 0), np.mean(np.abs(dd) <= 64)))

print("\n== coverage: sample matches in tome round_end ==")
re_df = pl.scan_parquet(chan("basic_round_end")).select(["match_key", "round"]).collect()
sample_matches = set(fs["match_key"].unique().to_list())
re_s = re_df.filter(pl.col("match_key").is_in(list(sample_matches)))
print("sample matches:", len(sample_matches), " with round 1 in tome:",
      re_s.filter(pl.col("round") == 1)["match_key"].n_unique(),
      " with round 13 in tome:", re_s.filter(pl.col("round") == 13)["match_key"].n_unique(),
      " with round 14 in tome:", re_s.filter(pl.col("round") == 14)["match_key"].n_unique())
print("sample first-snapshot matches with round 13:", fs.filter(pl.col("round") == 13)["match_key"].n_unique())
print("sample rounds per match (tome round_end), quantiles:",
      np.percentile(re_s.group_by("match_key").len()["len"].to_numpy(), [10, 50, 90]))
print("\n== Done ==")
