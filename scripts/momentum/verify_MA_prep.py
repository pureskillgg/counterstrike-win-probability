"""Independent verifier for MA, step 1: per-match table from header + round_end.

Header join key is header.key (round_end.match_key). Writes a per-match parquet to the scratchpad
and prints the funnel. Lazy scans with column projection; well under 2 GB.
"""
import hashlib
import sys

import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
HEADER = BASE + r"\header\dataframe_00000"
ROUND_END = BASE + r"\basic_round_end.2026-09-18,2026-10-08\dataframe_00000"
OUT = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_MA_matches.parquet"

hdr = (
    pl.scan_parquet(HEADER)
    .select(
        pl.col("key").alias("match_key"),
        "map_name",
        "max_rounds",
        "is_wingman",
        "is_gotv_recording",
        "match_id",
        "sharecode",
        "demo_id",
        "t_starters_score_final",
        "ct_starters_score_final",
        "match_date",
    )
    .collect()
)
print("header rows", hdr.height, "unique match_key", hdr["match_key"].n_unique(),
      "unique match_id", hdr["match_id"].n_unique(), "unique sharecode", hdr["sharecode"].n_unique(),
      "unique demo_id", hdr["demo_id"].n_unique())
print("max_rounds counts:", hdr.group_by("max_rounds").len().sort("max_rounds").rows())
print("is_wingman counts:", hdr.group_by("is_wingman").len().rows())
print("is_gotv_recording counts:", hdr.group_by("is_gotv_recording").len().rows())

re_ = (
    pl.scan_parquet(ROUND_END)
    .select("match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count")
    .collect()
)
print("round_end rows", re_.height, "distinct match_key", re_["match_key"].n_unique())
print("winner_team_code counts:", re_.group_by("winner_team_code").len().rows())
print("win_reason_code counts:", re_.group_by("win_reason_code").len().sort("win_reason_code").rows())
print("player_count counts:", re_.group_by("player_count").len().rows())

# Round-number sanity: duplicates or gaps within a match
dup_rounds = re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("match/round pairs with >1 round_end row:", dup_rounds.height)

re_ = re_.sort(["match_key", "round"])
seq = (
    re_.group_by("match_key", maintain_order=False)
    .agg(
        pl.col("round").alias("rounds"),
        pl.col("tick").alias("ticks"),
        pl.col("winner_team_code").alias("winners"),
        pl.col("win_reason_code").alias("reasons"),
        pl.col("player_count").max().alias("player_count"),
        pl.col("player_count").min().alias("player_count_min"),
        pl.len().alias("n_round_end"),
        pl.col("round").max().alias("max_round"),
        pl.col("round").min().alias("min_round"),
    )
)
print("matches with round_end:", seq.height)
print("matches with contiguous rounds 1..n:",
      seq.filter((pl.col("max_round") == pl.col("n_round_end")) & (pl.col("min_round") == 1)).height)
print("player_count max!=min matches:", seq.filter(pl.col("player_count") != pl.col("player_count_min")).height)

# Signature for duplicate detection: exact round_end sequence (round, tick, winner, reason)
def sig(row):
    s = "|".join(f"{r},{t},{w},{c}" for r, t, w, c in zip(row["rounds"], row["ticks"], row["winners"], row["reasons"]))
    return hashlib.sha1(s.encode()).hexdigest()

seq = seq.with_columns(
    pl.struct(["rounds", "ticks", "winners", "reasons"]).map_elements(sig, return_dtype=pl.String).alias("sig_full"),
    pl.struct(["winners"]).map_elements(
        lambda r: "".join(str(w) for w in r["winners"]), return_dtype=pl.String
    ).alias("sig_winners"),
)

m = hdr.join(seq, on="match_key", how="left")
print("header rows joined to round_end:", m.filter(pl.col("n_round_end").is_not_null()).height, "of", m.height)
m.write_parquet(OUT)
print("wrote", OUT, "rows", m.height)
