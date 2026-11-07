# VERIFIER for A1-1, part 4: header t/ct labels vs engine end-of-match side scores (round_state at cs_win_panel_match).
# Engine end state: fin_t, fin_ct = t_score, ct_score on the win-panel row (side-level at match end).
# For a regulation match (one halftime swap), header t_starters should equal fin_ct and header ct should equal fin_t
# if header t/ct label the round-1 starters. ASSUMPTION: one swap for regulation matches; OT adds swaps every 3 rounds.
# Counts by day and by max_rounds; also the 10 split-only matches (sum agrees, split differs under the swap rule).
import os, glob
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
def pages(ch):
    return sorted(glob.glob(os.path.join(T, ch, "dataframe_*")))
def chdir(prefix):
    return [x for x in os.listdir(T) if x.startswith(prefix + ".")][0]

h = pl.concat([pl.scan_parquet(p).select(["key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"])
               for p in pages("header")], how="vertical").collect()
h = h.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("S"),
                   pl.col("key").str.split("/").list.get(3).alias("day"))
rs = pl.concat([pl.scan_parquet(p).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
                for p in pages(chdir("basic_round_state"))], how="diagonal_relaxed").collect()
wp = (rs.filter(pl.col("event_type") == "cs_win_panel_match").sort(["match_key", "tick"])
        .group_by("match_key").last().select(["match_key", pl.col("t_score").alias("fin_t"), pl.col("ct_score").alias("fin_ct")]))
re_ = pl.concat([pl.scan_parquet(p).select(["match_key", "round"]) for p in pages(chdir("basic_round_end"))],
                how="diagonal_relaxed").collect()
rn = re_.group_by("match_key").agg(pl.len().alias("rows"))
m = h.join(wp, left_on="key", right_on="match_key", how="left").join(rn, left_on="key", right_on="match_key", how="left")
m = m.with_columns([
    (pl.col("t_starters_score_final") == pl.col("fin_ct")).alias("t_eq_fin_ct"),
    (pl.col("t_starters_score_final") == pl.col("fin_t")).alias("t_eq_fin_t"),
    ((pl.col("t_starters_score_final") == pl.col("fin_ct")) & (pl.col("ct_starters_score_final") == pl.col("fin_t"))).alias("swapped_exact"),
    ((pl.col("t_starters_score_final") == pl.col("fin_t")) & (pl.col("ct_starters_score_final") == pl.col("fin_ct"))).alias("straight_exact"),
    (pl.col("S") == pl.col("rows")).alias("sum_agree"),
    (pl.col("max_rounds") == 16).alias("is16"),
])
print("matches", m.height, "| with win panel", m.filter(pl.col("fin_t").is_not_null()).height)
print("split exact, straight (hdr t=fin_t, ct=fin_ct):", m.filter(pl.col("straight_exact")).height,
      "| swapped (hdr t=fin_ct, ct=fin_t):", m.filter(pl.col("swapped_exact")).height,
      "| neither:", m.filter(~pl.col("straight_exact") & ~pl.col("swapped_exact")).height)
print("by day: day, n, straight, swapped, neither, sum_agree")
print(m.with_columns([
    (~pl.col("straight_exact") & ~pl.col("swapped_exact")).alias("neither")
]).group_by("day").agg([pl.len().alias("n"), pl.col("straight_exact").sum().alias("straight"),
                       pl.col("swapped_exact").sum().alias("swapped"), pl.col("neither").sum().alias("neither"),
                       pl.col("sum_agree").sum().alias("sum_agree")]).sort("day").rows())
print("by day and max_rounds, among sum-agreeing matches only: day, mr, n, straight, swapped, neither")
ag = m.filter(pl.col("sum_agree"))
print(ag.with_columns((~pl.col("straight_exact") & ~pl.col("swapped_exact")).alias("neither")).group_by(["day", "max_rounds"]).agg([
    pl.len().alias("n"), pl.col("straight_exact").sum().alias("straight"), pl.col("swapped_exact").sum().alias("swapped"),
    pl.col("neither").sum().alias("neither")]).sort(["day", "max_rounds"]).rows())
print("split-only (sum agrees, split differs) matches: key, mr, S, hdr t, hdr ct, fin_t, fin_ct, straight, swapped:")
so = m.filter(pl.col("sum_agree") & ~pl.col("straight_exact"))
print(so.select(["key", "max_rounds", "S", "t_starters_score_final", "ct_starters_score_final", "fin_t", "fin_ct",
                 "straight_exact", "swapped_exact"]).rows())
print("done")
