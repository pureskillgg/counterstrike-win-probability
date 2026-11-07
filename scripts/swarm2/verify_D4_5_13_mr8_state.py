"""verify D4-5, step 13: MR8 mapping check using ONLY the round_state final score (no round_end labels).

For complete 2024 matches with an MR8 halftime event (label 7) and label 1 present, days 12-25 (and MR12 for contrast):
  X_SC : X (started T) final score from round_state's last row: t_score if X is on T in the last round, else
         ct_score. X on T in round r iff s(r) even, with s(r) = number of swaps before r (per-match convention).
  Compare X_SC with header hT. If MR8 mapping is right, X_SC == hT for (almost) all MR8 matches.
  Also: the number of swaps implied by the final side of X is compared with the physical field swaps
  observed in round_state (values exchanged between freeze ends), for the MR8 group.
Writes stdout only.
"""
import sys

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import polars as pl  # noqa: E402
from verify_D4_5_common import TOMES, scan, header, T, CT  # noqa: E402

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(14)
base = TOMES["2024"]
rs = scan(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"]).collect()
re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code"]).collect().sort(
    ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
H = header(base, ["key", "t_starters_score_final", "ct_starters_score_final"]).rename(
    {"key": "match_key", "t_starters_score_final": "hT", "ct_starters_score_final": "hC"})
half = rs.filter(pl.col("event_type") == "round_announce_last_round_half").group_by("match_key").agg(
    pl.col("round").min().alias("half_label"))
M = re_.group_by("match_key").agg(pl.len().alias("n_RE"), pl.col("round").min().alias("min_label"),
                                  pl.col("round").max().alias("max_label")).join(H, on="match_key").join(
    half, on="match_key", how="left")
M = M.with_columns(
    pl.when(pl.col("half_label") == 7).then(pl.lit(16)).when(pl.col("half_label") == 11).then(pl.lit(24))
    .otherwise(pl.lit(0)).alias("mr"),
    (pl.col("hT") + pl.col("hC") == pl.col("n_RE")).alias("complete"),
    (pl.col("min_label") == 1).alias("label1"),
    pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32).alias("day"))
M = M.filter(pl.col("complete") & pl.col("label1") & pl.col("day").is_between(12, 25) & (pl.col("mr") > 0))
keep = M["match_key"].to_list()

# last round index and the swaps strictly before it (per-match convention)
Ls = M.select(["match_key", "mr", "max_label"]).with_columns(
    pl.struct(["max_label", "mr"]).map_elements(
        lambda s: sum(1 for k in range(1, s["max_label"]) if (k == s["mr"] // 2) or
                      (k > s["mr"] and (k - s["mr"] - 3) % 6 == 0)), return_dtype=pl.Int64).alias("s_last"))
st = rs.filter(pl.col("match_key").is_in(keep)).sort(["match_key", "tick"]).group_by("match_key").agg(
    pl.col("t_score").last().alias("t_last"), pl.col("ct_score").last().alias("c_last"))
S = Ls.join(st, on="match_key").join(H, on="match_key").with_columns(
    pl.when(pl.col("s_last") % 2 == 0).then(pl.col("t_last")).otherwise(pl.col("c_last")).alias("X_SC"))
S = S.with_columns(
    (pl.col("X_SC") == pl.col("hT")).alias("ok_SC"),
    (pl.col("t_last") + pl.col("c_last") == pl.col("hT") + pl.col("hC")).alias("state_total_eq"))
print("state final vs header, complete d12-25 matches with label 1 present")
print(S.group_by("mr").agg(pl.len().alias("n"), pl.col("ok_SC").mean().alias("X_SC_eq_header"),
                           pl.col("state_total_eq").mean().alias("state_total_eq_header_sum")).sort("mr"))
print("\nMR8 (mr=16): X_SC - header residual distribution")
print(S.filter(pl.col("mr") == 16).with_columns((pl.col("X_SC") - pl.col("hT")).alias("res")).group_by("res").len().sort("res"))
print("\nMR8 (mr=16): s_last parity vs header, first 8 rows")
print(S.filter(pl.col("mr") == 16).select(["match_key", "max_label", "s_last", "t_last", "c_last", "hT", "hC", "X_SC"]).head(8))

# same state-based check but the other convention for MR8: no swap at all
S2 = S.filter(pl.col("mr") == 16).with_columns(
    pl.when(pl.col("s_last") >= 0).then(pl.col("t_last")).otherwise(pl.col("c_last")).alias("X_SC_noswap"))
print("\nMR8 alt: X = t_last (no-swap assumption) equals header:", int((S2["X_SC_noswap"] == S2["hT"]).sum()),
      "of", S2.height, "| X = c_last equals header:", int((S2["c_last"] == S2["hT"]).sum()))
