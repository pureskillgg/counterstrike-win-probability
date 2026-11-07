# VERIFIER for A1-1 (independent of A1_*.py). Recomputes from raw tome columns:
#  (1) header final sum S = t_starters_score_final + ct_starters_score_final vs round_end row counts per match,
#  (2) engine final per-team score from score_update (max new_score per team_code, monotonicity-checked),
#      which does NOT use the side-swap rule,
#  (3) round contiguity (1..n), duplicate (match, round) rows, key/match_id join integrity,
#  (4) the 80 shortfall matches: game-over test using engine final scores, and day/build/map/server profile.
# ASSUMPTION: win threshold = max_rounds // 2 + 1 (first-to-13 for 24-round, first-to-9 for 16-round).
# Output: stdout only (redirected into logs\swarm2\verify_A1_1.log by the caller).
import os, glob, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
HDR = sorted(glob.glob(os.path.join(T, "header", "dataframe_*")))
RE = sorted(glob.glob(os.path.join(T, "basic_round_end.2026-09-18,2026-10-08", "dataframe_*")))
SU = sorted(glob.glob(os.path.join(T, "basic_score_update.2026-09-18,2026-10-08", "dataframe_*")))
print("header pages", len(HDR), "round_end pages", len(RE), "score_update pages", len(SU))

# ---- header ----
hdr_cols = ["key", "match_id", "match_date", "map_name", "max_rounds", "build_num", "server_name",
            "t_starters_score_final", "ct_starters_score_final"]
h = pl.concat([pl.scan_parquet(p).select(hdr_cols) for p in HDR], how="vertical").collect()
print("header rows", h.height, "distinct key", h.get_column("key").n_unique(), "distinct match_id", h.get_column("match_id").n_unique())
h = h.with_columns([
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("S"),
    pl.col("key").str.split("/").list.get(3).alias("day"),
    pl.col("key").str.split("/").list.get(4).alias("mid_from_key"),
])
print("header key-path match_id == header match_id:", h.filter(pl.col("mid_from_key") == pl.col("match_id")).height, "of", h.height)
print("header max_rounds dist:", h.group_by("max_rounds").len().sort("max_rounds").rows())
print("header day dist:", h.group_by("day").len().sort("day").rows())

# ---- round_end ----
re_cols_want = ["match_key", "round", "winner_team_code", "win_reason_code"]
re_schema = pl.scan_parquet(RE[0]).collect_schema().names()
re_cols = [c for c in re_cols_want if c in re_schema]
print("round_end cols used:", re_cols)
r = pl.concat([pl.scan_parquet(p).select(re_cols) for p in RE], how="diagonal_relaxed").collect()
print("round_end total rows", r.height, "| distinct match_key", r.get_column("match_key").n_unique())
dup = r.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("duplicate (match_key, round) groups:", dup.height)
print("round_end match_keys not in header:", r.filter(~pl.col("match_key").is_in(h.get_column("key").to_list())).get_column("match_key").n_unique())
agg = r.group_by("match_key").agg([
    pl.len().alias("rows"), pl.col("round").n_unique().alias("distinct"),
    pl.col("round").min().alias("rmin"), pl.col("round").max().alias("rmax")])
m = h.join(agg, left_on="key", right_on="match_key", how="left").with_columns([
    pl.col("rows").fill_null(0), pl.col("distinct").fill_null(0), pl.col("rmax").fill_null(0)])
m = m.with_columns([
    (pl.col("S") - pl.col("rows")).alias("S_minus_rows"),
    ((pl.col("rmin") == 1) & (pl.col("rmax") == pl.col("distinct"))).alias("contiguous_1_to_n")])
print("header matches with 0 round_end rows:", m.filter(pl.col("rows") == 0).height)
print("S == rows:", m.filter(pl.col("S_minus_rows") == 0).height,
      "| S > rows:", m.filter(pl.col("S_minus_rows") > 0).height,
      "| S < rows:", m.filter(pl.col("S_minus_rows") < 0).height, "of", m.height)
print("sum S over all:", int(m.get_column("S").sum()), "| sum rows over all:", int(m.get_column("rows").sum()),
      "| gap sum (S - rows):", int(m.get_column("S_minus_rows").sum()))
gap_dist = m.filter(pl.col("S_minus_rows") != 0).group_by("S_minus_rows").len().sort("S_minus_rows").rows()
print("gap distribution (S - rows) for mismatches:", gap_dist)
print("contiguous 1..n among all matches:", m.filter(pl.col("contiguous_1_to_n")).height, "of", m.height)
print("contiguous 1..n among S>rows (shortfall):", m.filter((pl.col("S_minus_rows") > 0) & pl.col("contiguous_1_to_n")).height,
      "of", m.filter(pl.col("S_minus_rows") > 0).height)
print("by day, S>rows / S<rows / S==rows:")
print(m.with_columns([
    (pl.col("S_minus_rows") > 0).alias("gt"), (pl.col("S_minus_rows") < 0).alias("lt"), (pl.col("S_minus_rows") == 0).alias("eq")
]).group_by("day").agg(pl.col("gt").sum().alias("S>rows"), pl.col("lt").sum().alias("S<rows"),
                       pl.col("eq").sum().alias("S==rows"), pl.len().alias("n")).sort("day").rows())

# ---- engine score_update (team-level check, no swap rule) ----
su = pl.concat([pl.scan_parquet(p).select(["match_key", "round", "tick", "team_code", "new_score"]) for p in SU],
               how="diagonal_relaxed").collect()
print("score_update rows", su.height, "| team_code values:", su.get_column("team_code").unique().sort().to_list(),
      "| matches with score_update:", su.get_column("match_key").n_unique())
su = su.sort(["match_key", "team_code", "round", "tick"]).with_columns(
    (pl.col("new_score") - pl.col("new_score").shift(1).over(["match_key", "team_code"])).alias("dscore"))
neg = su.filter(pl.col("dscore") < 0)
print("per (match, team_code) score decreases:", neg.height, "rows; matches affected:", neg.get_column("match_key").n_unique())
fin = su.group_by(["match_key", "team_code"]).agg(pl.col("new_score").max().alias("fmax"), pl.len().alias("nrows"))
ncodes = fin.group_by("match_key").len().rename({"len": "ncodes"})
print("matches with exactly 2 team codes in score_update:", ncodes.filter(pl.col("ncodes") == 2).height, "of", ncodes.height)
eng = fin.group_by("match_key").agg([pl.col("fmax").sum().alias("eng_sum"), pl.col("fmax").max().alias("eng_max"),
                                     pl.col("fmax").min().alias("eng_min")])
m = m.join(eng, left_on="key", right_on="match_key", how="left")
m = m.with_columns([
    (pl.col("eng_sum") - pl.col("rows")).alias("eng_minus_rows"),
    (pl.col("eng_sum") - pl.col("S")).alias("eng_minus_S"),
    (pl.col("max_rounds") // 2 + 1).alias("thr")])
print("engine sum == round_end rows:", m.filter(pl.col("eng_minus_rows") == 0).height, "of", m.height)
print("engine sum == header S:", m.filter(pl.col("eng_minus_S") == 0).height, "of", m.height)
sh = m.filter(pl.col("S_minus_rows") > 0)
print("SHORTFALL (S>rows) n =", sh.height)
print("  engine sum == rows within shortfall:", sh.filter(pl.col("eng_minus_rows") == 0).height, "of", sh.height)
print("  engine sum == header S within shortfall:", sh.filter(pl.col("eng_minus_S") == 0).height, "of", sh.height)
print("  engine max team score == threshold (game over by engine):", sh.filter(pl.col("eng_max") >= pl.col("thr")).height, "of", sh.height)
print("  shortfall: rows (min,max):", sh.get_column("rows").min(), sh.get_column("rows").max(),
      "| S-rows min/max:", sh.get_column("S_minus_rows").min(), sh.get_column("S_minus_rows").max(),
      "| sum of (S-rows):", int(sh.get_column("S_minus_rows").sum()), "| sum S:", int(sh.get_column("S").sum()),
      "| sum rows:", int(sh.get_column("rows").sum()))
print("  shortfall by day:", sh.group_by("day").len().sort("day").rows())
print("  shortfall by max_rounds:", sh.group_by("max_rounds").len().sort("max_rounds").rows())
print("  shortfall engine_max distribution (team score at end):", sh.group_by("eng_max").len().sort("eng_max").rows())
print("  NON-game-over shortfall (eng_max < thr):")
print(sh.filter(pl.col("eng_max") < pl.col("thr")).select(["key", "max_rounds", "rows", "S", "t_starters_score_final",
                                                              "ct_starters_score_final", "eng_max", "eng_min", "eng_sum", "thr"]).rows())
print("  rows-level game-over reached on last row? (eng_max >= thr AND rows == eng_sum):",
      sh.filter((pl.col("eng_max") >= pl.col("thr")) & (pl.col("eng_minus_rows") == 0)).height, "of", sh.height)
print("  whole-tome: matches with eng_max >= thr:", m.filter(pl.col("eng_max") >= pl.col("thr")).height, "of", m.height)
print("  whole-tome: non-shortfall with eng_max < thr (threshold not reached):",
      m.filter((pl.col("S_minus_rows") == 0) & (pl.col("eng_max") < pl.col("thr"))).height)

# ---- profile: does the shortfall cluster by header-level recording config? ----
print("profile day 01-03 only: shortfall vs non-shortfall by build_num:")
d3 = m.filter(pl.col("day").is_in(["01", "02", "03"]))
print(d3.with_columns((pl.col("S_minus_rows") > 0).alias("short")).group_by(["build_num", "short"]).len().sort(["build_num", "short"]).rows())
print("profile day 01-03 by server_name (shortfall, total):")
print(d3.with_columns((pl.col("S_minus_rows") > 0).alias("short")).group_by(["server_name"]).agg(
    pl.col("short").sum().alias("short"), pl.len().alias("n")).sort("server_name").rows()[:10])
print("profile day 01-03 by map_name (shortfall, total):")
print(d3.with_columns((pl.col("S_minus_rows") > 0).alias("short")).group_by(["map_name"]).agg(
    pl.col("short").sum().alias("short"), pl.len().alias("n")).sort("map_name").rows())
print("profile day 01-03 by max_rounds (shortfall, total):")
print(d3.with_columns((pl.col("S_minus_rows") > 0).alias("short")).group_by(["max_rounds"]).agg(
    pl.col("short").sum().alias("short"), pl.len().alias("n")).sort("max_rounds").rows())

print("done")
