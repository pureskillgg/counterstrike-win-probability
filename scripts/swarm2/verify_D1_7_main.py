"""verify D1-7 (swarm2 verifier): recompute the claim's numbers from parquet, test the counter-exchange explanation,
and compare winner_team_code with bomb evidence by round.

ASSUMPTIONS (stated, not tuned):
  A1 side codes: 2 = T, 3 = CT (as in the claim and bakeoff).
  A2 bomb truth: bomb_exploded in a round => T (2) won; bomb_defused => CT (3) won (game rules; independent of team codes).
  A3 claim's score side: first and last round_state row of the round; +1 on t_score => 2, +1 on ct_score => 3, else null (as in D1_followup2.py).
  A4 exchange signature: consecutive round_state rows of one (match, round) where (t,ct) changes to (old_ct, old_t) with old_t != old_ct.
  A5 2024 population: player_count == 20 as in D1_followup2.py; also reported for all player counts.
  A6 2026 population: player_count == 10 as in D1_2026_score_check.py; also reported for all player counts.
  A7 round_end deduplicated on (match_key, round); bomb outcome = any bomb_defused/bomb_exploded in that round.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(18)
pl.Config.set_fmt_str_lengths(60)
T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
OUT = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")


def scan(base, name, suf=""):
    return pl.scan_parquet(sorted(glob.glob(str(base / f"basic_{name}{suf}*" / "dataframe_*"))))


def build(base, suf, year):
    re_ = (scan(base, "round_end", suf).select(["match_key", "round", "winner_team_code", "player_count"])
           .unique(["match_key", "round"]).collect())
    rs = (scan(base, "round_state", suf).select(["match_key", "round", "tick", "t_score", "ct_score"]).collect()
          .sort(["match_key", "round", "tick"]))
    # claim's method (A3)
    per = rs.group_by(["match_key", "round"]).agg(
        pl.col("t_score").first().alias("t0"), pl.col("t_score").last().alias("t1"),
        pl.col("ct_score").first().alias("c0"), pl.col("ct_score").last().alias("c1"))
    per = per.with_columns(
        pl.when((pl.col("t1") - pl.col("t0")) == 1).then(pl.lit(2))
          .when((pl.col("c1") - pl.col("c0")) == 1).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).alias("claim_side"))
    # A4 exchange signature (within a round, consecutive rows)
    rs2 = rs.with_columns(
        pl.col("t_score").shift(1).over(["match_key", "round"]).alias("pt"),
        pl.col("ct_score").shift(1).over(["match_key", "round"]).alias("pc"))
    swap = rs2.filter((pl.col("pt").is_not_null()) & (pl.col("pt") != pl.col("pc"))
                      & (pl.col("t_score") == pl.col("pc")) & (pl.col("ct_score") == pl.col("pt")))
    swaps = swap.group_by(["match_key", "round"]).agg(pl.len().alias("n_swap"), pl.col("tick").min().alias("swap_tick"))
    # bomb outcome (A2, A7)
    bs = (scan(base, "bomb_state", suf).filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
          .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("bomb_tick")).collect())
    bs = bs.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side")).select(["match_key", "round", "bomb_side", "bomb_tick"])
    out = (re_.join(per, on=["match_key", "round"], how="left").join(swaps, on=["match_key", "round"], how="left")
           .join(bs, on=["match_key", "round"], how="left")
           .with_columns(pl.col("n_swap").fill_null(0), pl.lit(year).alias("year")))
    out = out.with_columns(
        (pl.col("claim_side").is_not_null() & (pl.col("claim_side") != pl.col("winner_team_code"))).alias("claim_mm"),
        (pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).alias("bomb_mm_winner"),
        (pl.col("bomb_side").is_not_null() & pl.col("claim_side").is_not_null() & (pl.col("bomb_side") != pl.col("claim_side"))).alias("bomb_mm_claim"))
    return out


J24 = build(T24, SUF24, 2024)
J26 = build(T26, "", 2026)
J24.write_parquet(OUT / "verify_D1_7_rounds2024.parquet")
J26.write_parquet(OUT / "verify_D1_7_rounds2026.parquet")
print("2024 rounds:", J24.height, " matches:", J24["match_key"].n_unique(), " max round:", J24["round"].max())
print("2026 rounds:", J26.height, " matches:", J26["match_key"].n_unique(), " max round:", J26["round"].max())
print("2024 player_count:", J24.group_by("player_count").len().sort("len", descending=True).head(6).to_dicts())
print("2026 player_count:", J26.group_by("player_count").len().sort("len", descending=True).head(6).to_dicts())


def sec(t):
    print("\n" + "=" * 96 + "\n" + t + "\n" + "=" * 96)


sec("C1 claim recompute, 2024 round 26/27/28 (player_count 20 per claim; and all player counts)")
for lab, J in [("pc20", J24.filter(pl.col("player_count") == 20)), ("allpc", J24)]:
    for rr in [26, 27, 28]:
        x = J.filter(pl.col("round") == rr)
        xs = x.filter(pl.col("claim_side").is_not_null())
        xb = x.filter(pl.col("bomb_side").is_not_null())
        xbc = xb.filter(pl.col("claim_side").is_not_null())
        print(f"[{lab}] r{rr}: rows={x.height} with_claim_side={xs.height} claim_mm={int(xs['claim_mm'].sum())} | "
              f"bomb_rows={xb.height} bomb_ne_winner={int(xb['bomb_mm_winner'].sum())} | "
              f"bomb&claim rows={xbc.height} bomb_ne_claim={int(xbc['bomb_mm_claim'].sum())}")

sec("C2 bomb vs winner and bomb vs claim side by round, 2024 (all pc), rounds 9..30")
t = (J24.filter(pl.col("bomb_side").is_not_null() & pl.col("round").is_between(9, 30))
     .group_by(["round"]).agg(pl.len().alias("bomb_rows"), pl.col("bomb_mm_winner").sum().alias("bomb_ne_winner"),
                              pl.col("claim_side").is_not_null().sum().alias("bomb_rows_with_claim"),
                              pl.col("bomb_mm_claim").sum().alias("bomb_ne_claim")).sort("round"))
print(t)

sec("C3 2024 all rounds: bomb != winner count, per round index and day window (all pc)")
J24d = J24.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).cast(pl.Int64).alias("day"))
J24d = J24d.with_columns(pl.when(pl.col("day") >= 12).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
t3 = (J24d.filter(pl.col("bomb_side").is_not_null()).group_by(["dg", "round"]).agg(
    pl.len().alias("bomb_rows"), pl.col("bomb_mm_winner").sum().alias("bomb_ne_winner")).sort(["dg", "round"]))
print(t3.filter(pl.col("bomb_ne_winner") > 0))
print("total bomb rows, winner mismatch, by dg:",
      J24d.filter(pl.col("bomb_side").is_not_null()).group_by("dg").agg(pl.col("bomb_mm_winner").sum().alias("mm"), pl.len().alias("n")).to_dicts())

sec("C4 2026 bomb vs winner by round (all pc)")
J26b = J26.filter(pl.col("bomb_side").is_not_null())
t4 = J26b.group_by("round").agg(pl.len().alias("bomb_rows"), pl.col("bomb_mm_winner").sum().alias("bomb_ne_winner")).sort("round")
print("2026 bomb rows total:", J26b.height, " bomb_ne_winner total:", int(J26b["bomb_mm_winner"].sum()))
print("2026 rounds with any bomb_ne_winner:", t4.filter(pl.col("bomb_ne_winner") > 0).to_dicts())
print("2026 rounds 25..45 bomb rows:")
print(t4.filter(pl.col("round").is_between(25, 45)))

sec("C5 2026 claim recompute: claim_side vs winner by round, pc10 (claim population)")
J26p = J26.filter(pl.col("player_count") == 10)
t5 = (J26p.filter(pl.col("claim_side").is_not_null()).group_by("round").agg(
    pl.len().alias("n_claim_side"), pl.col("claim_mm").sum().alias("claim_mm")).sort("round"))
print("pc10 rounds with claim mismatches:")
print(t5.filter(pl.col("claim_mm") > 0))
print("pc10 rounds 25..45:")
print(t5.filter(pl.col("round").is_between(25, 45)))

sec("C6 exchange-in-window vs claim mismatch, 2026 pc10")
x = J26p.filter(pl.col("claim_side").is_not_null())
print("claim mismatch by n_swap (pc10, all rounds):",
      x.group_by(["n_swap", "claim_mm"]).len().sort(["n_swap", "claim_mm"]).to_dicts())
print("2026 pc10 rounds with any exchange-in-window, by round:",
      J26p.filter(pl.col("n_swap") > 0).group_by("round").len().sort("round").to_dicts())
print("2024 all-pc rounds with any exchange-in-window, by round:",
      J24.filter(pl.col("n_swap") > 0).group_by("round").len().sort("round").to_dicts())
