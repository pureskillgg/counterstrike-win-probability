"""verify_D2_3 bybuild: sx-free channel results at rounds 13, 25, 28 split by 2024 build_num (9842 vs 9961) and 2026.
Read-only; reads scratchpad base tables and the header.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_3_bybuild.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
               r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D2_3")


def header(label):
    fs = sorted(glob.glob(str(TOME[label] / "header" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=["key", "build_num", "match_date"]) for f in fs],
                     how="vertical_relaxed").rename({"key": "match_key"}).unique("match_key", keep="first")


def main():
    for label in ["2024", "2026"]:
        b = pl.read_parquet(SCRATCH / f"base_{label}.parquet")
        h = header(label)
        b = b.join(h, on="match_key", how="left")
        print(f"\n################ {label}: sx-free channels by build at rounds 13, 25, 28 ################")
        k = b.filter(pl.col("round").is_in([13, 25, 28]))
        bomb = k.filter(pl.col("s_bomb_w").is_not_null()).group_by(["round", "build_num"]).agg(
            pl.len().alias("n_bomb"), (pl.col("s_bomb_w") != pl.col("R")).sum().alias("bomb_R_ne"))
        print("bomb (sx-free, windowed):")
        print(bomb.sort(["round", "build_num"]))
        score = k.filter(pl.col("s_score_free").is_not_null() & (pl.col("sx") == pl.col("sx_nx"))).group_by(
            ["round", "build_num"]).agg(pl.len().alias("n_score_valid"), (pl.col("s_score_free") != pl.col("R")).sum().alias("score_R_ne"))
        print("score delta, valid only where sx(r)==sx(r+1) (no team-side change across the boundary):")
        print(score.sort(["round", "build_num"]))
        elim = k.filter(pl.col("s_elim_free").is_not_null()).group_by(["round", "build_num"]).agg(
            pl.len().alias("n_elim"), (pl.col("s_elim_free") != pl.col("R")).sum().alias("elim_R_ne"))
        print("elimination (sx-free, 5-death rule):")
        print(elim.sort(["round", "build_num"]))
        other = b.filter((pl.col("round") != 13) & (pl.col("round") != 25) & (pl.col("round") != 28) & pl.col("s_bomb_w").is_not_null())
        print("bomb rows outside 13/25/28 that disagree with R, by build and round:")
        print(other.filter(pl.col("s_bomb_w") != pl.col("R")).group_by(["build_num", "round"]).len().sort(["build_num", "round"]))


if __name__ == "__main__":
    main()
