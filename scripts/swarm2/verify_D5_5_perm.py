"""verify D5-5, step 5: are the late-group disagreements single-row flips or label shifts?

For late matches with a clean step at label L and at the neighbouring label, count how often the round_end winner
at L equals the step at L-1 or L+1 (shift), versus equals the step at L (no shift, flip only).
Also characterises the 161 matches with a label-9 disagreement and no label-13 disagreement.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_5_perm.py
"""
import io
import contextlib
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_D5_5_check as base  # noqa: E402  (its report prints first; this script's section follows)

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

J = base.J24.filter((pl.col("pc") == 20) & (pl.col("grp") == "late"))
# step table per (match, label): chain and rw, keyed by label
S = J.select(["match_key", "round", "chain", "rw", "cmp"]).filter(pl.col("chain") != 0)

print("\n\n######## PERMUTATION / SHIFT TEST (late, pc20) ########")
print("A. rows where rw(L) differs from step(L), split by whether rw(L) equals step at a neighbouring label")
for L in [9, 13, 25, 28]:
    cur = S.filter((pl.col("round") == L) & (pl.col("rw").is_not_null()) & (pl.col("cmp") == "disagree"))
    prev = S.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("chain").alias("chain_prev")])
    nxt = S.select(["match_key", (pl.col("round") - 1).alias("round"), pl.col("chain").alias("chain_next")])
    k = cur.join(prev, on=["match_key", "round"], how="left").join(nxt, on=["match_key", "round"], how="left")
    n = k.height
    print(f"L{L}: disagree rows {n}; rw(L) == step(L-1): {int((k['rw'] == k['chain_prev']).sum())}; "
          f"rw(L) == step(L+1): {int((k['rw'] == k['chain_next']).sum())}; "
          f"step(L-1) and step(L+1) both absent: {int((k['chain_prev'].is_null() & k['chain_next'].is_null()).sum())}")

print("\nB. the 161 late matches with a label-9 disagreement but no label-13 disagreement")
d9 = S.filter((pl.col("round") == 9) & (pl.col("cmp") == "disagree")).select("match_key")
d13 = S.filter((pl.col("round") == 13) & (pl.col("cmp") == "disagree")).select("match_key")
only9 = d9.join(d13, on="match_key", how="anti")
print("matches:", only9.height)
W = J.select(["match_key", "round", "rw", "chain", "reason", "status"]).filter(pl.col("match_key").is_in(only9["match_key"].to_list()))
W9 = W.filter(pl.col("round") == 9).select(["match_key", pl.col("rw").alias("rw9"), pl.col("chain").alias("ch9"),
                                            pl.col("reason").alias("reason9"), pl.col("status").alias("status9")])
W13 = W.filter(pl.col("round") == 13).select(["match_key", pl.col("rw").alias("rw13"), pl.col("chain").alias("ch13"),
                                              pl.col("status").alias("status13")])
W10 = W.filter(pl.col("round") == 10).select(["match_key", pl.col("rw").alias("rw10"), pl.col("chain").alias("ch10")])
W8 = W.filter(pl.col("round") == 8).select(["match_key", pl.col("rw").alias("rw8"), pl.col("chain").alias("ch8")])
X = W9.join(W13, on="match_key", how="left").join(W10, on="match_key", how="left").join(W8, on="match_key", how="left")
print("label-9 status (bomb status of round 9 in these matches):")
print(X.group_by("status9").len().sort("status9"))
print("label-9 reason codes:")
print(X.group_by("reason9").len().sort("reason9"))
print("day distribution of these 161 matches:")
print(J.filter(pl.col("match_key").is_in(only9["match_key"].to_list())).select(["match_key", "day"]).unique()
      .group_by("day").len().sort("day"))

print("\nC. round-9 rows for these matches: rw9 vs step at 8, 9 and 10 (no shift would show rw9 == ch9 ... )")
print(X.select([(pl.col("rw9") == pl.col("ch9")).alias("rw9_eq_ch9_expected_false"),
                (pl.col("rw9") == pl.col("ch10")).alias("rw9_eq_ch10"),
                (pl.col("rw9") == pl.col("ch8")).alias("rw9_eq_ch8"),
                (pl.col("rw9") == pl.col("ch13")).alias("rw9_eq_ch13"),
                (pl.col("rw13") == pl.col("ch13")).alias("rw13_eq_ch13")]).sum())
print("\nPERM DONE")
