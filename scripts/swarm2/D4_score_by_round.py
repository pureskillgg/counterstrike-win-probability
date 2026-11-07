r"""D4 check: score-side vs round_end winner disagreement by round number, 2024 (two day groups) and 2026 control.

For round r with no side swap after it (bakeoff.swap_expected(r), max_rounds 24 for 2024 per A12, header value for
2026), score side from round_state freeze_end(r) -> freeze_end(r+1) (A13). Unambiguous rounds only.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_score_by_round.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
         "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(base, ch, cols):
    return read_dir(sorted(base.glob(f"basic_{ch}.*"))[0], cols)


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


out = []
for era, base in TOMES.items():
    if era == "2026":
        H = read_dir(base / "header", ["key", "max_rounds"])
        mr_map = {k: int(v) for k, v in zip(H["key"].to_list(), H["max_rounds"].to_list())}
    else:
        mr_map = None
    rs = read(base, "round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
    fe = (rs.filter(pl.col("event_type") == "round_freeze_end").select(["match_key", "round", "t_score", "ct_score"])
          .unique(["match_key", "round"], keep="first"))
    nx = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"t_score": "t2", "ct_score": "c2"})
    D = fe.join(nx, on=["match_key", "round"], how="inner").with_columns(
        (pl.col("t2") - pl.col("t_score")).alias("dT"), (pl.col("c2") - pl.col("ct_score")).alias("dC")).with_columns(
        pl.when((pl.col("dC") == 1) & (pl.col("dT") == 0)).then(pl.lit(CT))
        .when((pl.col("dT") == 1) & (pl.col("dC") == 0)).then(pl.lit(T)).otherwise(pl.lit(0)).alias("score_side"))
    if mr_map is None:
        D = D.with_columns(pl.lit(24).alias("mr"))
    else:
        D = D.with_columns(pl.col("match_key").replace_strict(mr_map, default=24, return_dtype=pl.Int64).alias("mr"))
    D = D.filter(~pl.struct(["round", "mr"]).map_elements(lambda s: swap_expected(s["round"], s["mr"]),
                                                          return_dtype=pl.Boolean))
    re_ = read(base, "round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
    J = D.join(re_, on=["match_key", "round"], how="inner").filter(pl.col("score_side") != 0).with_columns(
        (pl.col("score_side") != pl.col("winner_team_code")).alias("dis"),
        pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).cast(pl.Int32).alias("dd"))
    J = J.with_columns(pl.when(pl.col("dd") <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d08-25")).alias("dg"))
    tab = J.group_by(["dg", "round"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis")).with_columns(
        (pl.col("dis") / pl.col("n")).alias("rate")).sort(["dg", "round"])
    print(f"\n==== {era}: score side vs round_end winner, by round (valid, unambiguous) ====")
    print(tab.filter(pl.col("round") <= 30).pivot(on="dg", index="round", values=["n", "dis", "rate"]).sort("round"))
    tot = J.group_by("dg").agg(pl.len().alias("n"), pl.col("dis").sum().alias("dis"),
                               (pl.col("round") == 13).sum().alias("n_r13"))
    print(tot)
