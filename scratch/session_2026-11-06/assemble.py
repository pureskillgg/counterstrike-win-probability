"""Assemble the training rounds from the two builds, with one duplicate pass across both sources.

Sources
  laptop  ~/adam/status/data/combined/rounds.parquet + final/laptop_matches.parquet   used for 2026-09 and 2026-10 days 1-25
  bucket  final/bucket_rounds.parquet + final/bucket_matches.parquet                  used for every other month (incl. Feb 2024 all days)
Excluded from training: 2026-10 days 26-31 (holdout). It is only used to report duplicate leakage into training.

Duplicate key: for matches with >= 16 round_end rows, the key is built from version-independent parts only
(map, first round, round count, final scores, winner/death sequence), so it matches across the laptop and space polars builds.
Shorter matches keep the per-source dup_key (no cross-source check for them).
Writes final/training_matches_v1.parquet and final/training_rounds_v1.parquet. Writes nothing to S3.
"""
import os

import polars as pl

F = os.path.expanduser("~/adam/builder/final")
LAPTOP_R = os.path.expanduser("~/adam/status/data/combined/rounds.parquet")
LAPTOP_M = os.path.join(F, "laptop_matches.parquet")
MONTH = r"(\d{4}/\d{2})/(\d{2})"

LM = pl.read_parquet(LAPTOP_M)
BM = pl.read_parquet(os.path.join(F, "bucket_matches.parquet"))
LR = pl.read_parquet(LAPTOP_R)
BR = pl.read_parquet(os.path.join(F, "bucket_rounds.parquet"))


def with_month(df):
    return df.with_columns(
        pl.col("match_key").str.extract(MONTH, 1).alias("month"),
        pl.col("match_key").str.extract(MONTH, 2).cast(pl.Int32).alias("day"))


LM, BM = with_month(LM), with_month(BM)
print("laptop matches:", LM.height, "bucket matches:", BM.height, flush=True)

holdout = LM.filter((pl.col("month") == "2026/10") & (pl.col("day") >= 26))
L_sel = LM.filter((pl.col("month") == "2026/09") | ((pl.col("month") == "2026/10") & (pl.col("day") <= 25)))
B_sel = BM.filter(pl.col("month") != "2026/10")
bucket_oct = BM.filter(pl.col("month") == "2026/10")
print("bucket Oct matches not in laptop Oct 1-25:",
      bucket_oct.join(L_sel.select("match_key"), on="match_key", how="anti").height, flush=True)
print("holdout matches (laptop Oct 26-31):", holdout.height, flush=True)

COLS = {"match_key": pl.String, "dup_key": pl.String, "n_ok_rounds": pl.Int64, "channels_complete": pl.Boolean,
        "in_scope": pl.Boolean, "map_name": pl.String, "n_rounds_re": pl.Int64, "r0": pl.Int64, "wseq": pl.String,
        "t_starters_score_final": pl.Int64, "ct_starters_score_final": pl.Int64, "month": pl.String}


def pick(df, src):
    return df.select([pl.col(c).cast(t, strict=False).alias(c) for c, t in COLS.items()]).with_columns(pl.lit(src).alias("src"))


U = pl.concat([pick(L_sel, "laptop"), pick(B_sel, "bucket")], how="vertical_relaxed")
print("training candidate matches:", U.height, "distinct:", U["match_key"].n_unique(), flush=True)


def gkey(df):
    k16 = pl.concat_str([pl.col("map_name"), pl.col("r0").cast(pl.String), pl.col("n_rounds_re").cast(pl.String),
                         pl.col("t_starters_score_final").cast(pl.String), pl.col("ct_starters_score_final").cast(pl.String),
                         pl.col("wseq")], separator="|")
    return df.with_columns(
        pl.when(pl.col("n_rounds_re").fill_null(0) >= 16).then(pl.lit("k:") + k16)
        .otherwise(pl.lit("s:") + pl.col("src") + ":" + pl.col("dup_key")).alias("gkey"))


U = gkey(U)
U = U.sort(["gkey", "n_ok_rounds", "match_key"], descending=[False, True, False], maintain_order=True)
U = U.with_columns((pl.int_range(pl.len()).over("gkey") == 0).alias("gkeeper"), pl.len().over("gkey").alias("ggroup"))
cross = U.group_by("gkey").agg(pl.col("src").n_unique().alias("n_src"), pl.len().alias("n")).filter(pl.col("n") > 1)
print("duplicate groups:", U.filter(pl.col("ggroup") > 1)["gkey"].n_unique(),
      "| groups spanning both sources:", cross.filter(pl.col("n_src") > 1).height,
      "| matches not kept:", U.filter(~pl.col("gkeeper")).height, flush=True)

# leakage: holdout matches whose duplicate key is also a training match
H = gkey(pick(holdout, "holdout"))
leak = H.join(U.select("gkey").unique(), on="gkey", how="inner")["gkey"].unique().to_list()
print("holdout matches with a duplicate inside training:", len(leak), "groups; removing them from training", flush=True)
U = U.filter(~pl.col("gkey").is_in(leak))

keys = U["match_key"].to_list()
RL = LR.filter(pl.col("match_key").is_in(U.filter(pl.col("src") == "laptop")["match_key"].to_list()))
RB = BR.filter(pl.col("match_key").is_in(U.filter(pl.col("src") == "bucket")["match_key"].to_list()))
R = pl.concat([RL, RB], how="diagonal_relaxed")
assert R.select(pl.col("match_key").n_unique()).item() == len(keys), "a match has rounds from neither source"
R = R.drop([c for c in ["dup_keeper", "dup_group_size", "use", "channels_complete", "in_scope"] if c in R.columns])
R = R.join(U.select(pl.col("match_key"), pl.col("gkeeper").alias("dup_keeper"), pl.col("ggroup").alias("dup_group_size"),
                    pl.col("channels_complete"), pl.col("in_scope"), pl.col("src")), on="match_key", how="left")
R = R.with_columns((pl.col("round_status").eq("ok") & pl.col("in_scope") & pl.col("dup_keeper")
                    & pl.col("channels_complete")).alias("use"))
U = U.with_columns(pl.col("gkeeper").alias("dup_keeper"))
U.write_parquet(os.path.join(F, "training_matches_v1.parquet"))
R.write_parquet(os.path.join(F, "training_rounds_v1.parquet"))

mon =R.with_columns(pl.col("match_key").str.extract(MONTH, 1).alias("month"))
out = (mon.group_by(["src", "month"]).agg(pl.col("match_key").n_unique().alias("matches"),
                                          pl.col("match_key").filter(pl.col("use")).n_unique().alias("use_matches"),
                                          pl.col("use").sum().alias("use_rounds")).sort(["month", "src"]))
print(out, flush=True)
print("TOTAL use matches (training v1):", R.filter(pl.col("use"))["match_key"].n_unique(),
      "| use rounds:", int(R["use"].sum()), flush=True)
