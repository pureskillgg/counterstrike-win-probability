"""Independent verifier for MA, step 2: match funnel, team identity, and dedup diagnostics."""
import polars as pl

PATH = r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_MA_matches.parquet"

m = pl.read_parquet(PATH)
print("rows", m.height)

# Duplicate detection: exact round_end sequence (round, tick, winner, reason) across different match_keys
dup_groups = m.group_by("sig_full").agg(pl.len().alias("n"), pl.col("match_key").alias("keys"))
dups = dup_groups.filter(pl.col("n") > 1)
print("exact-sequence duplicate groups:", dups.height, "rows in groups:", int(dups["n"].sum()) if dups.height else 0,
      "extra copies:", int((dups["n"] - 1).sum()) if dups.height else 0)

# Looser: same winner string only (sanity, not the dedup rule)
wdup = m.group_by("sig_winners").agg(pl.len().alias("n"))
print("same winner-string groups with n>1:", wdup.filter(pl.col("n") > 1).height,
      "extra copies:", int((wdup.filter(pl.col("n") > 1)["n"] - 1).sum()))

# Also duplicate on match_id / other header fields? (match_id is unique; check map+date+final score)
kdup = m.group_by(["map_name", "match_date", "t_starters_score_final", "ct_starters_score_final"]).agg(pl.len().alias("n"))
print("map+date+score groups with n>1:", kdup.filter(pl.col("n") > 1).height,
      "extra copies:", int((kdup.filter(pl.col("n") > 1)["n"] - 1).sum()))

d = m.filter(pl.col("sig_full").is_in(dups["sig_full"].to_list()) if dups.height else pl.lit(False))
print("duplicate rows by max_rounds/is_wingman:")
print(d.group_by(["max_rounds", "is_wingman"]).len().rows())

# Keep first per sig_full (sorted by match_key) as the dedup rule
md = m.sort("match_key").unique(subset=["sig_full"], keep="first", maintain_order=True)
print("after exact-sequence dedup:", md.height)

# Decided / tie / ends-before-13 counts (all formats, deduplicated)
md = md.with_columns(
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("final_sum"),
)
md = md.with_columns(
    (pl.col("final_sum") == pl.col("n_round_end")).alias("sum_matches_rounds"),
    (pl.col("t_starters_score_final") != pl.col("ct_starters_score_final")).alias("not_tied"),
)
print("deduplicated, sum==rounds:", md.filter(pl.col("sum_matches_rounds")).height,
      " sum!=rounds:", md.filter(~pl.col("sum_matches_rounds")).height)
print("deduplicated ties (final equal):", md.filter(pl.col("t_starters_score_final") == pl.col("ct_starters_score_final")).height)
dec = md.filter(pl.col("not_tied") & pl.col("sum_matches_rounds"))
print("deduplicated decided (not tied and sum==rounds):", dec.height)
print("  max_rounds x is_wingman among decided:", dec.group_by(["max_rounds", "is_wingman"]).len().sort(["max_rounds", "is_wingman"]).rows())
print("  decided and round_end count < 13:", dec.filter(pl.col("n_round_end") < 13).height)
print("  decided with n_round_end distribution (top):",
      dec.group_by("n_round_end").len().sort("n_round_end").head(40).rows())

# Not decided but sum!=rounds
bad = md.filter(~pl.col("sum_matches_rounds"))
print("sum!=rounds sample:", bad.select("match_key", "t_starters_score_final", "ct_starters_score_final", "n_round_end", "max_rounds").head(10))

# player_count variation inside decided matches
print("decided: player_count (max) distribution:", dec.group_by("player_count").len().sort("player_count").rows())
print("decided: player_count_min distribution:", dec.group_by("player_count_min").len().sort("player_count_min").rows())

# Save deduplicated decided table for the analysis step
out = r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_MA_dedup.parquet"
md.write_parquet(out)
print("wrote", out, md.height)
