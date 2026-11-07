# verify A1-2: which header convention explains t_starters/ct_starters across all 1752 matches?
# Candidates for header t_starters_score_final:
#   C1 roster: starter-T group (players on T in round 1, observed in player_info) total round wins
#   C2 raw_T2: total rounds won with winner_team_code 2 (per-side count over the whole match)
#   C3 swap:   swap-model starter-T total (ASSUMPTION: swap after max_rounds//2, bakeoff.py line ~129)
# Also: whether t_starters_avg_* equals ct_starters_avg_* (degenerate starter split), by max_rounds.
import glob, os, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
R = pl.read_csv(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_A1_2_roster.csv", infer_schema_length=0)
R = R.with_columns([pl.col(c).cast(pl.Int64, strict=False) for c in ["n", "S", "hdr_t", "hdr_ct", "A_roster", "B_roster", "A_swap", "B_swap", "max_rounds", "unresolved"]])


def pages(ch):
    return sorted(glob.glob(os.path.join(T, f"basic_{ch}.2026-09-18,2026-10-08", "dataframe_*")))


re_ = pl.concat([pl.scan_parquet(p).select(["match_key", "round", "winner_team_code"]) for p in pages("round_end")],
                how="diagonal_relaxed").collect()
rawc = re_.group_by("match_key").agg(
    (pl.col("winner_team_code") == 2).sum().alias("raw_T2"),
    (pl.col("winner_team_code") == 3).sum().alias("raw_T3")).rename({"match_key": "key"})
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"),
                    columns=["key", "max_rounds", "t_starters_avg_wins", "ct_starters_avg_wins",
                             "t_starters_avg_rank", "ct_starters_avg_rank", "t_starters_score_final", "ct_starters_score_final"])
h = h.with_columns((pl.col("t_starters_avg_wins") == pl.col("ct_starters_avg_wins")).alias("avg_wins_equal"),
                   (pl.col("t_starters_avg_rank") == pl.col("ct_starters_avg_rank")).alias("avg_rank_equal"))
J = R.join(h.select(["key", "avg_wins_equal", "avg_rank_equal"]), on="key", how="left").join(rawc, on="key", how="left")
print("joined", J.height)
print("avg_wins_equal by max_rounds:", J.group_by(["max_rounds", "avg_wins_equal"]).len().sort(["max_rounds", "avg_wins_equal"]).rows())
print("avg_rank_equal by max_rounds:", J.group_by(["max_rounds", "avg_rank_equal"]).len().sort(["max_rounds", "avg_rank_equal"]).rows())
both = J.filter(pl.col("n") == pl.col("S"))
print("n==S matches:", both.height)
print("n==S: header t == roster A:", both.filter(pl.col("hdr_t") == pl.col("A_roster")).height,
      "| header t == raw_T2:", both.filter(pl.col("hdr_t") == pl.col("raw_T2")).height,
      "| header t == swap A:", both.filter(pl.col("hdr_t") == pl.col("A_swap")).height)
ex = both.filter(pl.col("hdr_t") != pl.col("A_swap"))
print("n==S, header t != swap-model A (claim-style swap set):", ex.height, "keys:")
for r in ex.select(["key", "max_rounds", "n", "hdr_t", "hdr_ct", "A_roster", "B_roster", "A_swap", "B_swap", "raw_T2", "raw_T3", "unresolved", "avg_wins_equal"]).iter_rows():
    print("  ", r)
print("n==S, header t == raw_T2 among exceptions:", ex.filter(pl.col("hdr_t") == pl.col("raw_T2")).height, "of", ex.height)
print("n==S & NOT exception (header == swap A): header t == raw_T2 count:", both.filter((pl.col("hdr_t") == pl.col("A_swap")) & (pl.col("hdr_t") == pl.col("raw_T2"))).height,
      "of", both.filter(pl.col("hdr_t") == pl.col("A_swap")).height)
print("n==S exceptions by max_rounds and day:", ex.with_columns(pl.col("key").str.split("/").list.get(3).alias("day")).group_by(["max_rounds", "day"]).len().sort(["max_rounds", "day"]).rows())
print("all 16-round matches: n==S count:", J.filter((pl.col("max_rounds") == 16) & (pl.col("n") == pl.col("S"))).height,
      "| 16-round day distribution:", J.filter(pl.col("max_rounds") == 16).with_columns(pl.col("key").str.split("/").list.get(3).alias("day")).group_by("day").len().sort("day").rows())
