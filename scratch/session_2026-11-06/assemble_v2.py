"""Training rounds v2: the v1 assembly, with 5vX rounds added back in.

5vX = one side has 5 known players, the other side has 4 or fewer, 10 players recorded, and player_count_norm 10.
Those rounds were marked 'roster not 5v5' in prepare.py. Here they are re-evaluated with the rules that prepare.py
applies after the roster check, in the same order. Nothing else changes: no chunk is rebuilt, and the labels, freeze
ticks and death sides are the ones prepare.py already computed. Matches get their ok-round counts recomputed, and the
duplicate pass and holdout exclusion run as in v1.

Writes final/training_matches_v2.parquet and final/training_rounds_v2.parquet. Writes nothing to S3.
"""
import os

import polars as pl

F = os.path.expanduser("~/adam/builder/final")
LAPTOP_R = os.path.expanduser("~/adam/status/data/combined/rounds.parquet")
LAPTOP_M = os.path.join(F, "laptop_matches.parquet")
MONTH = r"(\d{4}/\d{2})/(\d{2})"


def with_month(df):
    return df.with_columns(
        pl.col("match_key").str.extract(MONTH, 1).alias("month"),
        pl.col("match_key").str.extract(MONTH, 2).cast(pl.Int32).alias("day"))


def add_5vx(R, tag):
    five = ((pl.col("n_t") == 5) & (pl.col("n_ct") <= 4)) | ((pl.col("n_ct") == 5) & (pl.col("n_t") <= 4))
    target = (pl.col("round_status") == "roster not 5v5") & five & (pl.col("player_count_norm") == 10) & (pl.col("n_players") == 10)
    later = (pl.when(pl.col("disc_or_bot")).then(pl.lit("disconnect or bot takeover"))
             .when(~pl.col("has_round_start")).then(pl.lit("no round_start"))
             .when(pl.col("freeze").is_null()).then(pl.lit("no freeze_end"))
             .when(pl.col("deaths_unresolved") > 0).then(pl.lit("death side unresolved"))
             .when((pl.col("deaths_t") > 5) | (pl.col("deaths_ct") > 5)).then(pl.lit("more than 5 deaths on a side"))
             .when(pl.col("final").is_null()).then(pl.lit("label unresolved"))
             .when(pl.col("end_eff") <= pl.col("freeze")).then(pl.lit("no undecided second"))
             .otherwise(pl.lit("ok")))
    n_target = R.filter(target).height
    out = R.with_columns(pl.when(target).then(later).otherwise(pl.col("round_status")).alias("round_status"))
    became = out.filter(target & (pl.col("round_status") == "ok")).height
    print(f"{tag}: 5vX rounds {n_target}, now ok {became}", flush=True)
    return out


def recount(M, R):
    n_ok = R.filter(pl.col("round_status") == "ok").group_by("match_key").len().rename({"len": "n_ok_new"})
    return (M.join(n_ok, on="match_key", how="left")
            .with_columns(pl.col("n_ok_new").fill_null(0).alias("n_ok_rounds")).drop("n_ok_new"))


LR = add_5vx(pl.read_parquet(LAPTOP_R), "laptop")
BR = add_5vx(pl.read_parquet(os.path.join(F, "bucket_rounds.parquet")), "bucket")
LM = recount(with_month(pl.read_parquet(LAPTOP_M)), LR)
BM = recount(with_month(pl.read_parquet(os.path.join(F, "bucket_matches.parquet"))), BR)

holdout = LM.filter((pl.col("month") == "2026/10") & (pl.col("day") >= 26))
L_sel = LM.filter((pl.col("month") == "2026/09") | ((pl.col("month") == "2026/10") & (pl.col("day") <= 25)))
B_sel = BM.filter(pl.col("month") != "2026/10")

COLS = {"match_key": pl.String, "dup_key": pl.String, "n_ok_rounds": pl.Int64, "channels_complete": pl.Boolean,
        "in_scope": pl.Boolean, "map_name": pl.String, "n_rounds_re": pl.Int64, "r0": pl.Int64, "wseq": pl.String,
        "t_starters_score_final": pl.Int64, "ct_starters_score_final": pl.Int64, "month": pl.String}


def pick(df, src):
    return df.select([pl.col(c).cast(t, strict=False).alias(c) for c, t in COLS.items()]).with_columns(pl.lit(src).alias("src"))


def gkey(df):
    k16 = pl.concat_str([pl.col("map_name"), pl.col("r0").cast(pl.String), pl.col("n_rounds_re").cast(pl.String),
                         pl.col("t_starters_score_final").cast(pl.String), pl.col("ct_starters_score_final").cast(pl.String),
                         pl.col("wseq")], separator="|")
    return df.with_columns(
        pl.when(pl.col("n_rounds_re").fill_null(0) >= 16).then(pl.lit("k:") + k16)
        .otherwise(pl.lit("s:") + pl.col("src") + ":" + pl.col("dup_key")).alias("gkey"))


U = gkey(pl.concat([pick(L_sel, "laptop"), pick(B_sel, "bucket")], how="vertical_relaxed"))
U = U.sort(["gkey", "n_ok_rounds", "match_key"], descending=[False, True, False], maintain_order=True)
U = U.with_columns((pl.int_range(pl.len()).over("gkey") == 0).alias("gkeeper"), pl.len().over("gkey").alias("ggroup"))
print("candidate matches:", U.height, "| matches not kept as duplicates:", U.filter(~pl.col("gkeeper")).height, flush=True)

H = gkey(pick(holdout, "holdout"))
leak = H.join(U.select("gkey").unique(), on="gkey", how="inner")["gkey"].unique().to_list()
print("holdout copies removed from training:", len(leak), "groups", flush=True)
U = U.filter(~pl.col("gkey").is_in(leak))

RL = LR.filter(pl.col("match_key").is_in(U.filter(pl.col("src") == "laptop")["match_key"].to_list()))
RB = BR.filter(pl.col("match_key").is_in(U.filter(pl.col("src") == "bucket")["match_key"].to_list()))
R = pl.concat([RL, RB], how="diagonal_relaxed")
assert R.select(pl.col("match_key").n_unique()).item() == U.height, "a match has rounds from neither source"
R = R.drop([c for c in ["dup_keeper", "dup_group_size", "use", "channels_complete", "in_scope"] if c in R.columns])
R = R.join(U.select(pl.col("match_key"), pl.col("gkeeper").alias("dup_keeper"), pl.col("ggroup").alias("dup_group_size"),
                    pl.col("channels_complete"), pl.col("in_scope"), pl.col("src")), on="match_key", how="left")
R = R.with_columns((pl.col("round_status").eq("ok") & pl.col("in_scope") & pl.col("dup_keeper")
                    & pl.col("channels_complete")).alias("use"))
U = U.with_columns(pl.col("gkeeper").alias("dup_keeper"))
U.write_parquet(os.path.join(F, "training_matches_v2.parquet"))
R.write_parquet(os.path.join(F, "training_rounds_v2.parquet"))

mon = R.with_columns(pl.col("match_key").str.extract(MONTH, 1).alias("month"))
print(mon.group_by(["src", "month"]).agg(pl.col("match_key").n_unique().alias("matches"),
                                         pl.col("match_key").filter(pl.col("use")).n_unique().alias("use_matches"),
                                         pl.col("use").sum().alias("use_rounds")).sort(["month", "src"]), flush=True)
use_keys = R.filter(pl.col("use"))["match_key"].unique().to_list()
use_m = U.filter(pl.col("match_key").is_in(use_keys)).select("match_key", "map_name", "month")
print("TOTAL v2 use matches:", use_m.height, "| use rounds:", int(R["use"].sum()), flush=True)
print("rows: rounds", R.height, "| match rows", U.height, flush=True)
print("by map:", use_m.group_by("map_name").len().sort("len", descending=True).rows(), flush=True)
