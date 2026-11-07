"""verify_D2_3 build: is the round-13/25/28 disagreement a game/demo build property rather than a match or day property?

Joins per-match flags from base_2024.parquet / base_2026.parquet (scratchpad) to header metadata
(build_num, map_name, match_date, starters' final scores). Also re-checks the header final-score test
with the selection that the previous run exposed (header total == N only for flagged matches).
Read-only with respect to the tome.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_3_build.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(50)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
               r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D2_3")
KEY = ["match_key", "round"]


def header(label):
    fs = sorted(glob.glob(str(TOME[label] / "header" / "dataframe_*")))
    h = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "map_name",
                                               "t_starters_score_final", "ct_starters_score_final"])
                   for f in fs], how="vertical_relaxed").rename({"key": "match_key"})
    return h.unique("match_key", keep="first")


def prep(label):
    b = pl.read_parquet(SCRATCH / f"base_{label}.parquet")
    b = b.with_columns(
        pl.when(pl.col("sx") == pl.col("sx_nx")).then(pl.col("s_score_free")).otherwise(pl.lit(None, dtype=pl.Int64))
        .alias("s_score_v"))
    b = b.with_columns(pl.coalesce(["s_score_v", "s_bomb_w", "s_elim_free"]).alias("T_v"))
    b = b.with_columns((pl.col("R") != pl.col("T_v")).alias("flip_v"))
    return b


def main():
    for label in ["2024", "2026"]:
        b = prep(label)
        h = header(label)
        print(f"\n################ {label} ################")
        print("header rows:", h.height, " header build_num values:")
        print(h.group_by("build_num").len().sort("build_num"))
        m = (b.group_by("match_key").agg(
            pl.len().alias("N"),
            pl.col("T_v").filter(pl.col("round") == 13).first().alias("t13"),
            pl.col("flip_v").filter(pl.col("round") == 13).first().alias("f13"),
            pl.col("flip_v").filter(pl.col("round") == 25).first().alias("f25"),
            pl.col("flip_v").filter(pl.col("round") == 28).first().alias("f28"),
            pl.col("flip_v").sum().alias("n_flip"),
            pl.col("round").max().alias("rmax"))
            .join(h, on="match_key", how="left"))
        m = m.with_columns(
            pl.when(pl.col("t13").is_null()).then(pl.lit("no_T13")).when(pl.col("f13")).then(pl.lit("flip13"))
            .otherwise(pl.lit("noflip13")).alias("cls"))
        print("\n-- match classes by build (cls: no_T13 / flip13 / noflip13) --")
        print(m.group_by(["build_num", "cls"]).len().sort(["build_num", "cls"]))
        print("\n-- build x match_date (first 10 characters) for flip13 and noflip13 --")
        print(m.with_columns(pl.col("match_date").str.slice(0, 10).alias("d"))
              .group_by(["d", "build_num", "cls"]).len().sort(["d", "build_num", "cls"]).head(120))

        if label == "2024":
            print("\n-- map x build x class (flip13 share within build 9961) --")
            print(m.group_by(["build_num", "map_name", "cls"]).len().sort(["build_num", "map_name", "cls"]))
            print("\n-- matches with no T13 (rule cannot fire): by build and whether a flip occurs at 25 or 28 --")
            nt = m.filter(pl.col("cls") == "no_T13")
            print(nt.group_by(["build_num"]).agg(pl.len().alias("n"),
                                                  ((pl.col("f25") == True) | (pl.col("f28") == True)).sum().alias("flip_25_or_28")))  # noqa: E712

            # header final-score test by build, with the selection made explicit
            rc = b.group_by("match_key").agg(pl.len().alias("N"), pl.col("round").min().alias("rmin"),
                                             pl.col("round").max().alias("rmax"),
                                             pl.col("sx").is_not_null().all().alias("sx_all"),
                                             pl.col("T_v").is_not_null().all().alias("T_all"),
                                             pl.col("sx").filter(pl.col("round") == 1).first().alias("sx1"))
            comp = rc.filter((pl.col("rmin") == 1) & (pl.col("rmax") == pl.col("N")) & pl.col("sx_all")
                             & pl.col("T_all") & pl.col("sx1").is_not_null())
            x = b.join(comp.select("match_key"), on="match_key", how="inner").group_by("match_key").agg(
                ((pl.col("R") == pl.col("sx")).cast(pl.Int64)).sum().alias("xw_R"),
                ((pl.col("T_v") == pl.col("sx")).cast(pl.Int64)).sum().alias("xw_T"),
                pl.col("flip_v").sum().alias("n_flip"),
                pl.len().alias("N"))
            x = x.join(h, on="match_key", how="inner").with_columns(
                (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hdr_total"))
            x = x.with_columns(
                (pl.col("hdr_total") == pl.col("N")).alias("sum_ok"),
                (pl.col("xw_R") == pl.col("t_starters_score_final")).alias("R_eq_hdr"),
                (pl.col("xw_T") == pl.col("t_starters_score_final")).alias("T_eq_hdr"))
            print("\n-- header final-score test by build (complete matches only) --")
            print(x.group_by(["build_num", "sum_ok", "n_flip"]).agg(
                pl.len().alias("n"), pl.col("R_eq_hdr").sum().alias("R_eq_hdr"),
                pl.col("T_eq_hdr").sum().alias("T_eq_hdr")).sort(["build_num", "sum_ok", "n_flip"]).head(60))
            print("\n-- header total minus N (complete matches), by build and flag status --")
            print(x.with_columns((pl.col("hdr_total") - pl.col("N")).alias("diff"))
                  .group_by(["build_num", "n_flip"]).agg(pl.len().alias("n"),
                                                         pl.col("diff").min().alias("min_diff"),
                                                         pl.col("diff").max().alias("max_diff"),
                                                         pl.col("diff").mean().round(2).alias("mean_diff"))
                  .sort(["build_num", "n_flip"]).head(40))


if __name__ == "__main__":
    main()
