"""verify D1-7 (swarm2 verifier): final summary of the numbers used in the verdict.

Reads the per-round tables written by verify_D1_7_main.py (scratchpad). Same ASSUMPTIONS A1-A9 as the other verify_D1_7_* scripts.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(16)
SP = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
J24 = pl.read_parquet(SP / "verify_D1_7_mech_rounds2024.parquet")
J26 = pl.read_parquet(SP / "verify_D1_7_mech_rounds2026.parquet")
J24 = J24.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"))
J24 = J24.with_columns(pl.when(pl.col("day") >= 12).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))


def sec(t):
    print("\n" + "=" * 96 + "\n" + t + "\n" + "=" * 96)


sec("F1 claim reproduction, 2024 round 27 (player_count 20, as in D1_followup2.py)")
x = J24.filter((pl.col("player_count") == 20) & (pl.col("round") == 27))
cs = x.filter(pl.col("claim_side").is_not_null())
bb = x.filter(pl.col("bomb_side").is_not_null())
print(f"rows with claim side: {cs.height}; claim side != winner: {int(cs['claim_mm'].sum())}")
print(f"bomb rows: {bb.height}; bomb != winner: {int(bb['bomb_mm_winner'].sum())}; bomb & claim rows: "
      f"{bb.filter(pl.col('claim_side').is_not_null()).height}; bomb != claim: {int(bb.filter(pl.col('claim_side').is_not_null())['bomb_mm_claim'].sum())}")

sec("F2 claim reproduction, 2026 player_count 10 (as in D1_2026_score_check.py), rounds 27/33/39/45")
p = J26.filter(pl.col("player_count") == 10)
for r in [27, 33, 39, 45]:
    c = p.filter((pl.col("round") == r) & pl.col("claim_side").is_not_null())
    print(f"round {r}: claim side rows {c.height}, mismatches {int(c['claim_mm'].sum())}")

sec("F3 mechanism: 2026 pc10 claim mismatches vs in-window counter exchange (n_swap)")
c = p.filter(pl.col("claim_side").is_not_null())
print(c.group_by(["n_swap", "claim_mm"]).len().sort(["n_swap", "claim_mm"]))
print("claim rows at rounds 27/33/39/45 with n_swap == 0:",
      c.filter(pl.col("round").is_in([27, 33, 39, 45]) & (pl.col("n_swap") == 0)).height)

sec("F4 mechanism: 2026 pre-swap side vs winner (all pc), exchange rows")
ex = J26.filter(pl.col("n_swap") > 0)
print("exchange rows by round:", ex.group_by("round").len().sort("round").to_dicts())
print("exchange rows with pre_side == winner:", int((ex["pre_side"] == ex["winner_team_code"]).sum()),
      " pre_side != winner:", int(((ex["pre_side"] != ex["winner_team_code"]) & ex["pre_side"].is_not_null()).sum()),
      " pre_side null:", int(ex["pre_side"].is_null().sum()))

sec("F5 2026 bomb evidence (all pc): bomb rows and winner agreement at rounds 27, 33, 39, 45 and overall")
bo = J26.filter(pl.col("bomb_side").is_not_null())
for r in [27, 33, 39, 45]:
    b = bo.filter(pl.col("round") == r)
    print(f"round {r}: bomb rows {b.height}, bomb != winner {int(b['bomb_mm_winner'].sum())}")
print("overall 2026 bomb rows:", bo.height, " bomb != winner:", int(bo["bomb_mm_winner"].sum()))

sec("F6 2024 winner_team_code vs bomb by day window (all pc), rounds 2..30")
bo24 = J24.filter(pl.col("bomb_side").is_not_null() & pl.col("round").is_between(2, 30))
print(bo24.group_by("dg").agg(pl.len().alias("bomb_rows"), pl.col("bomb_mm_winner").sum().alias("bomb_ne_winner")).sort("dg").to_dicts())
print("d12-25 bomb != winner by round:", bo24.filter((pl.col("dg") == "d12-25") & pl.col("bomb_mm_winner"))
      .group_by("round").len().sort("round").to_dicts())
print("d12-25 round 13 bomb rows, pc20 only:", int(bo24.filter((pl.col("dg") == "d12-25") & (pl.col("round") == 13) & (pl.col("player_count") == 20)).height),
      " mismatches pc20:", int(bo24.filter((pl.col("dg") == "d12-25") & (pl.col("round") == 13) & (pl.col("player_count") == 20))["bomb_mm_winner"].sum()))

sec("F7 2024 claim side vs bomb (all pc), d12-25, rounds 9/13/25/27/28")
d = J24.filter((pl.col("dg") == "d12-25") & pl.col("bomb_side").is_not_null() & pl.col("round").is_in([9, 13, 25, 27, 28]) & pl.col("claim_side").is_not_null())
print(d.group_by("round").agg(pl.len().alias("bomb_and_claim_rows"), pl.col("bomb_mm_claim").sum().alias("bomb_ne_claim")).sort("round").to_dicts())

sec("F8 2024 pre-swap side vs bomb at exchange rounds (all pc)")
e = J24.filter((pl.col("n_swap") > 0) & pl.col("bomb_side").is_not_null())
print(e.group_by("round").agg(pl.len().alias("n"), (pl.col("bomb_side") == pl.col("pre_side")).sum().alias("bomb_eq_pre"),
                              (pl.col("bomb_side") == pl.col("winner_team_code")).sum().alias("bomb_eq_winner")).sort("round").to_dicts())
print("2024 max round present:", int(J24["round"].max()), " (rounds 33/39/45 cannot be checked in 2024)")

sec("F9 pre-swap side coverage and agreement with winner_team_code, exchange rows (all pc)")
for lab, J in [("2024", J24), ("2026", J26)]:
    ex9 = J.filter(pl.col("n_swap") > 0)
    print(lab, "exchange rows:", ex9.height, " pre_side null:", int(ex9["pre_side"].is_null().sum()),
          " pre_side == winner:", int((ex9["pre_side"] == ex9["winner_team_code"]).sum()),
          " pre_side != winner:", int(((ex9["pre_side"] != ex9["winner_team_code"]) & ex9["pre_side"].is_not_null()).sum()))
    print(lab, "exchange rows by round (pre_side null / n):",
          ex9.group_by("round").agg(pl.len().alias("n"), pl.col("pre_side").is_null().sum().alias("null")).sort("round").to_dicts())
print("2024 all-pc round 27 claim mismatches:", int(J24.filter((pl.col("round") == 27) & pl.col("claim_side").is_not_null())["claim_mm"].sum()),
      " of", int(J24.filter((pl.col("round") == 27) & pl.col("claim_side").is_not_null()).height))
