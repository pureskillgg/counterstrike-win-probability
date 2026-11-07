"""verify D1-6 (swarm2 verifier), part A.
Checks: (1) round_end duplicates per (match_key, round) in the 2024 tome, (2) player_count distribution,
(3) reproduction of the claimed round-9 winner-vs-score disagreement counts (305 of 3,249 in d12-25; 0 of 805 in d01-07)
with the claim's exact method, and the same count across all player_count universes, with and without de-duplication,
(4) win_reason_code / win_reason_message as a third, independent winner signal (mapping learned from the 2026 tome).
ASSUMPTIONS:
  A1 score side = side whose score rose by exactly 1 between the first and last round_state row of the round (claim method, all event types).
  A2 day group d12-25 = match_key day in 12..25 (claim method); d01-07 = days 01..07.
  A3 player_count 20 (2024 claim universe) and 10 (2026 bake-off universe) are compared as separate universes.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
SUF24 = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def f24(ch):
    return sorted(glob.glob(str(T24 / f"basic_{ch}{SUF24}" / "dataframe_*")))


def f26(ch):
    return sorted(glob.glob(str(T26 / f"basic_{ch}*" / "dataframe_*")))


def rd(files, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def side_all(RS):
    """Claim method (D1_followup2.py lines 51-59), reproduced exactly."""
    sc = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_before"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_before"),
        pl.col("t_score").sort_by("tick").last().alias("t_after"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_after")))
    sc = sc.with_columns(
        pl.when((pl.col("t_after") - pl.col("t_before")) == 1).then(pl.lit(2))
          .when((pl.col("ct_after") - pl.col("ct_before")) == 1).then(pl.lit(3))
          .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side")).select(["match_key", "round", "score_side"])
    return sc


RE_COLS = ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "win_reason_message", "player_count"]
RE24 = rd(f24("round_end"), RE_COLS).with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
RE24 = RE24.with_columns(
    pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
print("2024 round_end rows:", RE24.height, " null day:", RE24.filter(pl.col("day").is_null()).height,
      " matches:", RE24["match_key"].n_unique())

# ---------------------------------------------------------------- 1 duplicates
hdr("1. round_end duplicates per (match_key, round), 2024 tome")
g = (RE24.group_by(["match_key", "round"]).agg(
        pl.len().alias("n"),
        pl.col("winner_team_code").n_unique().alias("nw"),
        pl.col("tick").n_unique().alias("nt"),
        pl.col("player_count").n_unique().alias("npc"),
        pl.col("dg").first().alias("dg")))
print("(match, round) groups:", g.height)
print("groups with n>1:", g.filter(pl.col("n") > 1).height,
      " groups with conflicting winner_team_code:", g.filter(pl.col("nw") > 1).height,
      " groups with >1 distinct tick:", g.filter(pl.col("nt") > 1).height)
print("n distribution:", g.group_by("n").len().sort("n").to_dicts())
print("round 9 groups by dg, n, conflicting winner:")
print(g.filter(pl.col("round") == 9).group_by(["dg", "n", "nw"]).len().sort(["dg", "n", "nw"]))
print("conflicting-winner groups by round (top 12):")
print(g.filter(pl.col("nw") > 1).group_by("round").len().sort("len", descending=True).head(12))

# ---------------------------------------------------------------- 2 player_count
hdr("2. player_count distribution, 2024 round_end rows (top values)")
print(RE24.group_by(["dg", "player_count"]).len().sort("len", descending=True).head(14))

# ---------------------------------------------------------------- 3 claim reproduction
hdr("3. round_state event types (2024) and claim reproduction of round-9 disagreements")
RS24 = rd(f24("round_state"), ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
print("2024 round_state rows:", RS24.height)
print("event_type counts (2024):")
print(RS24.group_by("event_type").len().sort("len", descending=True).head(12))
SC24 = side_all(RS24)
print("score-side rows (2024, all rounds):", SC24.height, " non-null:", SC24.filter(pl.col("score_side").is_not_null()).height)


def disagree_table(RE, label):
    J = (RE.join(SC24, on=["match_key", "round"], how="left"))
    x = J.filter((pl.col("round") == 9) & pl.col("score_side").is_not_null())
    x = x.with_columns((pl.col("score_side") != pl.col("winner_team_code")).alias("mm"))
    tab = x.group_by(["dg"]).agg(pl.len().alias("rows_with_score"), pl.col("mm").sum().alias("disagree"),
                                  pl.col("match_key").n_unique().alias("matches")).sort("dg")
    print(f"\n-- {label}: round 9 rows with score side, winner vs score")
    print(tab)
    return x


# claim universe: pc==20, de-duplicated with keep='any' (claim)
claim = RE24.filter(pl.col("player_count") == 20).unique(["match_key", "round"])
xc = disagree_table(claim, "CLAIM universe pc==20, dedup keep=any (claim's method)")
d9c = xc.filter(pl.col("mm") & (pl.col("dg") == "d12-25"))
print("distinct d12-25 matches with round-9 disagreement (pc20):", d9c["match_key"].n_unique())

# same, pc==20, dedup keep=first (deterministic: keep the first row in file order)
claim_f = RE24.filter(pl.col("player_count") == 20).unique(["match_key", "round"], keep="first", maintain_order=True)
disagree_table(claim_f, "pc==20, dedup keep=first, maintain_order")

# pc==20, NO dedup (every round_end row)
xr = (RE24.filter(pl.col("player_count") == 20).join(SC24, on=["match_key", "round"], how="left")
      .filter((pl.col("round") == 9) & pl.col("score_side").is_not_null()))
print("\n-- pc==20 NO dedup: round-9 rows with score side:", xr.height, " disagreements:",
      int((xr["score_side"] != xr["winner_team_code"]).sum()))

# all player counts, dedup keep=first
allpc = RE24.unique(["match_key", "round"], keep="first", maintain_order=True)
xa = disagree_table(allpc, "ALL player_count, dedup keep=first")
print("round 9 disagreements by player_count (ALL, dedup keep=first, d12-25 and d01-07):")
print((xa.with_columns((pl.col("score_side") != pl.col("winner_team_code")).alias("mm"))
       .group_by(["dg", "player_count"]).agg(pl.len().alias("rows"), pl.col("mm").sum().alias("disagree"))
       .sort(["dg", "player_count"])))

# control rounds in the claim universe (pc==20, d12-25) : disagreement rate by round
hdr("3b. control: winner-vs-score disagreement by round, claim universe pc==20 dedup keep=first, d12-25")
xx = claim_f.join(SC24, on=["match_key", "round"], how="left").filter(
    pl.col("score_side").is_not_null() & (pl.col("dg") == "d12-25"))
ctl = (xx.with_columns((pl.col("score_side") != pl.col("winner_team_code")).alias("mm"))
       .group_by("round").agg(pl.len().alias("n"), pl.col("mm").sum().alias("disagree"))
       .with_columns((pl.col("disagree") / pl.col("n")).round(4).alias("rate")).sort("round"))
print(ctl.head(30))

# ---------------------------------------------------------------- 4 win_reason signal
hdr("4. win_reason_code / message vs winner_team_code, 2026 tome (pc==10): mapping learned")
RE26 = rd(f26("round_end"), RE_COLS)
m26 = (RE26.filter(pl.col("player_count") == 10)
       .group_by(["win_reason_code", "win_reason_message", "winner_team_code"]).len()
       .sort(["win_reason_code", "winner_team_code"]))
print(m26)

hdr("4b. 2024 round 9, pc==20 claim universe: win_reason of disagreeing vs agreeing rows (d12-25)")
r9 = (claim_f.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25"))
      .join(SC24, on=["match_key", "round"], how="left").filter(pl.col("score_side").is_not_null())
      .with_columns((pl.col("score_side") != pl.col("winner_team_code")).alias("mm")))
print(r9.group_by(["mm", "win_reason_code", "win_reason_message", "winner_team_code", "score_side"]).len()
      .sort(["mm", "win_reason_code", "winner_team_code", "score_side"]))
