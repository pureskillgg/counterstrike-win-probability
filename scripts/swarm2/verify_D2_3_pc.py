"""verify_D2_3 pc: why do ~550 build-9961 matches have no round-13 truth? Checks round_end player_count,
player_info roster counts and round_state freeze-end presence at rounds 12-14, by build. Also tests whether
a sides-unchanged-between-13-and-14 score delta (sx-free, roster-free) is available for those matches.
Read-only; reads the 2024 tome and the scratchpad base table.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_3_pc.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
               r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D2_3")
KEY = ["match_key", "round"]


def tbl(ch, cols):
    fd = sorted(glob.glob(str(TOME / f"basic_{ch}.*")))[0]
    return pl.scan_parquet(sorted(glob.glob(fd + "/dataframe_*"))).select(cols).collect()


def main():
    fs = sorted(glob.glob(str(TOME / "header" / "dataframe_*")))
    h = pl.concat([pl.read_parquet(f, columns=["key", "build_num", "t_starters_score_final", "ct_starters_score_final"])
                   for f in fs], how="vertical_relaxed").rename({"key": "match_key"}).unique("match_key", keep="first")
    re_all = tbl("round_end", ["match_key", "round", "tick", "player_count"])
    b = pl.read_parquet(SCRATCH / "base_2024.parquet")
    b = b.with_columns(
        pl.when(pl.col("sx") == pl.col("sx_nx")).then(pl.col("s_score_free")).otherwise(pl.lit(None, dtype=pl.Int64))
        .alias("s_score_v"))
    b = b.with_columns(pl.coalesce(["s_score_v", "s_bomb_w", "s_elim_free"]).alias("T_v")).select(
        KEY + ["sx", "sx_nx", "sx_prev", "s_score_free", "s_bomb_w", "s_elim_free", "T_v", "R"])
    build_of = h.select(["match_key", "build_num"])

    print("################ 1. round_end player_count distribution by build (rounds 1-30) ################")
    pcd = (re_all.filter(pl.col("round").is_between(1, 30)).join(build_of, on="match_key", how="left")
           .with_columns((pl.col("player_count") == 20).alias("pc20")))
    print(pcd.group_by(["build_num", "pc20"]).len().sort(["build_num", "pc20"]))
    print("top player_count values by build:")
    print(pcd.group_by(["build_num", "player_count"]).len().sort(["build_num", "len"], descending=[False, True])
          .group_by("build_num").head(6))

    print("\n################ 2. header total vs round_end rows (all player_count) vs pc==20 rows, by build ################")
    cnt = re_all.group_by("match_key").agg(
        pl.len().alias("rows_all"), (pl.col("player_count") == 20).sum().alias("rows_pc20"),
        pl.col("round").n_unique().alias("rounds_all")).join(build_of, on="match_key", how="left").join(
        h.select(["match_key", "t_starters_score_final", "ct_starters_score_final"]), on="match_key", how="left")
    cnt = cnt.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hdr"))
    print(cnt.group_by("build_num").agg(
        pl.len().alias("matches"),
        (pl.col("hdr") == pl.col("rows_all")).sum().alias("hdr==rows_all"),
        (pl.col("hdr") == pl.col("rows_pc20")).sum().alias("hdr==rows_pc20"),
        (pl.col("rows_all") - pl.col("rows_pc20")).mean().round(2).alias("mean_nonpc20_rows")).sort("build_num"))

    print("\n################ 3. rounds 12-14 for matches with no round-13 truth (by build) ################")
    nt = b.filter(pl.col("round") == 13).filter(pl.col("T_v").is_null()).select("match_key").unique()
    nt = nt.join(build_of, on="match_key", how="left")
    print(nt.group_by("build_num").len().sort("build_num"))
    sel = nt.select("match_key")
    re_sel = re_all.join(sel, on="match_key", how="inner").filter(pl.col("round").is_between(12, 14))
    print("round_end rows for rounds 12-14 in these matches, by player_count:")
    print(re_sel.join(build_of, on="match_key", how="left").group_by(["build_num", "round", "player_count"]).len()
          .sort(["build_num", "round", "player_count"]).head(60))
    rs = tbl("round_state", ["match_key", "round", "tick", "event_type"])
    fe = (rs.filter(pl.col("event_type") == "round_freeze_end").group_by(KEY).agg(pl.len().alias("n_fe")))
    fe_sel = fe.join(sel, on="match_key", how="inner").filter(pl.col("round").is_between(12, 14))
    print("round_freeze_end rows for rounds 12-14 in these matches:")
    print(fe_sel.join(build_of, on="match_key", how="left").group_by(["build_num", "round"]).len().sort(["build_num", "round"]))
    pi = tbl("player_info", ["match_key", "round", "player_id", "team_code"]).unique(
        ["match_key", "round", "player_id"], keep="first", maintain_order=True)
    pc_np = pi.group_by(KEY).agg(pl.len().alias("np"), (pl.col("team_code") == 2).sum().alias("nt"),
                                 (pl.col("team_code") == 3).sum().alias("nct"))
    pn = pc_np.join(sel, on="match_key", how="inner").filter(pl.col("round").is_between(12, 14))
    print("player_info roster counts for rounds 12-14 in these matches (np, nt, nct):")
    print(pn.join(build_of, on="match_key", how="left").group_by(["build_num", "round", "np", "nt", "nct"]).len()
          .sort(["build_num", "round", "len"], descending=[False, False, True]).head(40))

    print("\n################ 4. sx-free, roster-free score delta at 13 (FE13 -> FE14) for no-T13 matches ################")
    rs2 = tbl("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
    fe2 = (rs2.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
           .unique(KEY, keep="first", maintain_order=True).select(KEY + ["t_score", "ct_score"]))
    d13 = (fe2.filter(pl.col("round") == 13).select(["match_key", pl.col("t_score").alias("t13"), pl.col("ct_score").alias("c13")])
           .join(fe2.filter(pl.col("round") == 14).select(["match_key", pl.col("t_score").alias("t14"), pl.col("ct_score").alias("c14")]),
                 on="match_key", how="full", coalesce=True))
    d13 = d13.with_columns(
        pl.when((pl.col("t14") - pl.col("t13") == 1) & (pl.col("c14") - pl.col("c13") == 0)).then(pl.lit(2))
        .when((pl.col("c14") - pl.col("c13") == 1) & (pl.col("t14") - pl.col("t13") == 0)).then(pl.lit(3))
        .otherwise(pl.lit(None, dtype=pl.Int64)).alias("side_free13"))
    x = nt.join(d13, on="match_key", how="left").join(
        b.filter(pl.col("round") == 13).select(["match_key", "R", "s_bomb_w", "s_elim_free"]), on="match_key", how="left")
    print("no-T13 matches: side_free13 available (FE13 & FE14 present, clean +1 pattern):")
    print(x.group_by("build_num").agg(pl.len().alias("n"), pl.col("side_free13").is_not_null().sum().alias("avail"),
                                      (pl.col("side_free13") != pl.col("R")).sum().alias("R_ne_side_free13")))
    print("control, matches WITH T13, same sx-free rule (no roster needed), R disagreement by build:")
    y = d13.join(b.filter(pl.col("round") == 13).select(["match_key", "R", "T_v"]), on="match_key", how="inner").join(
        build_of, on="match_key", how="left")
    print(y.filter(pl.col("side_free13").is_not_null()).group_by("build_num").agg(
        pl.len().alias("n"), (pl.col("side_free13") != pl.col("R")).sum().alias("R_ne_side_free13"),
        (pl.col("side_free13") != pl.col("R")).mean().round(3).alias("rate")).sort("build_num"))


if __name__ == "__main__":
    main()
