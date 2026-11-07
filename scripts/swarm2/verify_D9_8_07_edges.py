"""verify D9-8 step 7: edge rounds (9, 25, 27, 28) and a raw-source spot check for one defect match.

Q5 round 9 (d12-25) disagreements: which matches, what messages, what score deltas.
Q6 round 27 (both day groups, 100% disagreement): score deltas and round_end messages, to decide
   whether the score oracle or the round_end record is the odd one out in overtime.
Q7 raw-source spot check (READ ONLY on E:): for one defect match (day 12, round 13 mismatch),
   list the raw round_end channel candidates and compare its round-13 winner to the tome.
Output goes to stdout (tee'd by the caller). Nothing is written under E:.
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(70)

SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
sc = pl.read_parquet(SCRATCH / "verify_D9_8_score_winner.parquet")
re_all = pl.read_parquet(SCRATCH / "verify_D9_8_round_end.parquet").filter(pl.col("player_count") == 20)
re1 = (re_all.group_by(["match_key", "round"]).agg(pl.col("winner_team_code").first(),
                                                  pl.col("win_reason_message").first()))
J = re1.join(sc.select(["match_key", "round", "score_winner", "dT", "dCT", "t_score", "ct_score"]),
             on=["match_key", "round"], how="left")
J = J.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
J = J.with_columns((pl.col("score_winner") != pl.col("winner_team_code")).alias("disagree"))

print("Q5 round 9 disagreements (d12-25), by delta pattern and message:")
r9 = J.filter((pl.col("round") == 9) & pl.col("score_winner").is_not_null())
print(r9.group_by(["disagree", "win_reason_message", "winner_team_code", "score_winner"]).agg(pl.len().alias("n")).sort(
    ["disagree", "win_reason_message"]))
print("Q5b round 9: distinct matches with >=1 disagreement:", r9.filter(pl.col("disagree"))["match_key"].n_unique())
print("Q5c round 9 disagreements per day:")
print(r9.filter(pl.col("disagree")).group_by("day").agg(pl.len().alias("n")).sort("day"))

print("\nQ6 round 27 (score vs round_end), all days:")
r27 = J.filter((pl.col("round") == 27))
print(r27.group_by(["day", "win_reason_message", "winner_team_code", "score_winner", "dT", "dCT"]).agg(pl.len().alias("n")).sort(
    ["day", "win_reason_message"]).head(40))
print("Q6b round 27 total rows:", r27.height, " with score:", int(r27["score_winner"].is_not_null().sum()))

print("\nQ5d round 25 / 28 (d12-25) disagreement counts:")
print(J.filter(pl.col("round").is_in([25, 28]) & pl.col("score_winner").is_not_null()).group_by(["round", "disagree"]).agg(
    pl.len().alias("n")).sort(["round", "disagree"]))

# ---- Q7 raw-source spot check (read-only)
key = "csds/2024/02/12/m_161472e7ab8d0a97e6250fef/csds"
mid = key.split("/")[3]
print("\nQ7 tome match:", key, "match id:", mid)
raw_roots = [
    r"E:\csds_final\csds\2024\02\12",
    r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02\12",
]
for root in raw_roots:
    p = Path(root)
    print("root exists:", p, p.exists())
    if p.exists():
        try:
            cands = [d for d in os.listdir(p) if d.startswith(mid[:8])]
            print("  dirs starting with match id prefix:", cands[:5])
            for c in cands[:1]:
                q = p / c
                print("  contents:", os.listdir(q)[:40])
        except OSError as exc:
            print("  listing failed:", exc)
