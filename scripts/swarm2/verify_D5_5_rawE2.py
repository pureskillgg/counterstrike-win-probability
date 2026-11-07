"""verify D5-5, step 6: raw-source check for a planted-only flip and a no-bomb flip, plus a map breakdown.

  1. Picks the first late L13 round where the bomb was planted but not defused or exploded and round_end says CT
     (winner 3), and the first late L13 no-bomb round where the step and round_end disagree.
  2. Reads the raw tail-recovered round_end and bomb_state for those matches (read-only on E:) and prints rounds 12-14.
  3. Map breakdown of the late L13 step-versus-round_end disagreement (pc20, clean step), to test map dependence.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_5_rawE2.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_D5_5_check as base  # noqa: E402  (prints the check report first)

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)

RAW = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12")
J = base.J24.join(base.HD24.select(["match_key", "map_name"]), on="match_key", how="left")
L13 = J.filter((pl.col("round") == 13) & (pl.col("pc") == 20) & (pl.col("grp") == "late")
               & (pl.col("chain") != 0) & pl.col("rw").is_not_null())

print("\n\n######## RAW CHECK 2 AND MAP BREAKDOWN ########")
po = L13.filter((pl.col("status") == "planted_only") & (pl.col("rw") == 3) & (pl.col("cmp") == "disagree")) \
    .sort("match_key").head(1)["match_key"].to_list()
nb = L13.filter(pl.col("status").is_null() & (pl.col("cmp") == "disagree")).sort("match_key") \
    .head(1)["match_key"].to_list()
picks = {"planted_only_flip": po[0] if po else None, "no_bomb_flip": nb[0] if nb else None}
print("picked:", picks)

for label, key in picks.items():
    if key is None:
        continue
    mid = key.split("/")[4]
    d = RAW / mid
    re_raw = pl.read_parquet(d / "round_end")
    bs_raw = pl.read_parquet(d / "bomb_state")
    print(f"\n==== {label}: {key} ====")
    print("RAW round_end 12-14:")
    print(re_raw.filter(pl.col("round").is_between(12, 14)).sort(["round", "tick"])
          .select(["round", "tick", "winner_team_code", "win_reason_code", "player_count"]))
    print("RAW bomb_state 12-14:")
    print(bs_raw.filter(pl.col("round").is_between(12, 14)).sort(["round", "tick"])
          .select(["round", "tick", "event_type", "player_team_code"]))
    print("TOME chain/step rows 12-14 for this match:")
    print(J.filter((pl.col("match_key") == key) & pl.col("round").is_between(12, 14)).sort("round")
          .select(["round", "t", "ct", "t_n", "ct_n", "chain", "rw", "reason", "status", "end_tick", "fe_tick"]))

# map breakdown on the late L13 disagreement set
print("\nMAP BREAKDOWN, late L13 (pc20, clean step), by map_name")
mb = (L13.group_by("map_name").agg(pl.len().alias("n"), (pl.col("cmp") == "disagree").sum().alias("disagree"),
                                   pl.col("status").is_null().sum().alias("no_bomb"))
      .with_columns((pl.col("disagree") / pl.col("n")).round(3).alias("rate")).sort("n", descending=True))
print(mb)
print("\nMAP BREAKDOWN, late L25 and L28 disagreement, by map_name")
mb2 = (J.filter(pl.col("round").is_in([25, 28]) & (pl.col("pc") == 20) & (pl.col("grp") == "late")
                & (pl.col("chain") != 0) & pl.col("rw").is_not_null())
       .group_by("map_name").agg(pl.len().alias("n"), (pl.col("cmp") == "disagree").sum().alias("disagree"))
       .with_columns((pl.col("disagree") / pl.col("n")).round(3).alias("rate")).sort("n", descending=True))
print(mb2)
print("\nRAW2 DONE")
