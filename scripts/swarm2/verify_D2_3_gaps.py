"""verify_D2_3 gaps: coverage holes, mixed matches, header mismatches, outside flips by channel, and the
match_key day versus header match_date. Reads scratchpad base tables and header (read-only).
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_3_gaps.py
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
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds"}
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
               r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D2_3")
KEY = ["match_key", "round"]


def header():
    fs = sorted(glob.glob(str(TOME["2024"] / "header" / "dataframe_*")))
    h = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "map_name",
                                               "t_starters_score_final", "ct_starters_score_final"])
                   for f in fs], how="vertical_relaxed").rename({"key": "match_key"})
    return h.unique("match_key", keep="first")


def main():
    b = pl.read_parquet(SCRATCH / "base_2024.parquet")
    b = b.with_columns(
        pl.when(pl.col("sx") == pl.col("sx_nx")).then(pl.col("s_score_free")).otherwise(pl.lit(None, dtype=pl.Int64))
        .alias("s_score_v"))
    b = b.with_columns(pl.coalesce(["s_score_v", "s_bomb_w", "s_elim_free"]).alias("T_v"),
                       pl.coalesce(["s_score_v", "s_bomb_w", "s_elim_free"]).alias("T_v_src_tmp"))
    b = b.with_columns(
        pl.when(pl.col("s_score_v").is_not_null()).then(pl.lit("score"))
        .when(pl.col("s_bomb_w").is_not_null()).then(pl.lit("bomb"))
        .when(pl.col("s_elim_free").is_not_null()).then(pl.lit("elim")).otherwise(pl.lit("none")).alias("src"),
        (pl.col("R") != pl.coalesce(["s_score_v", "s_bomb_w", "s_elim_free"])).alias("flip_v"))
    h = header()
    mk = b.group_by("match_key").agg(pl.len().alias("N"), pl.col("round").max().alias("rmax")).join(h, on="match_key", how="left")
    t13 = b.filter(pl.col("round") == 13).select(["match_key", "T_v", "flip_v", "src", "sx", "sx_nx", "sx_prev"]).rename(
        {"T_v": "T13", "flip_v": "f13", "src": "src13", "sx": "sx13", "sx_nx": "sx14", "sx_prev": "sx12"})
    mk = mk.join(t13, on="match_key", how="left")

    print("################ 1. Round-13 coverage holes, build 9961 and 9842 ################")
    nt = mk.filter(pl.col("T13").is_null())
    print(nt.group_by("build_num").len().sort("build_num"))
    r13 = b.filter(pl.col("round") == 13).select(["match_key", "sx", "sx_nx", "s_score_free", "s_bomb_w", "s_elim_free"])
    fe13 = (pl.read_parquet(SCRATCH / "base_2024.parquet").filter(pl.col("round") == 13)
            .select(["match_key", pl.col("t_fe").is_not_null().alias("fe13")]))
    nt2 = nt.join(r13, on="match_key", how="left").join(fe13, on="match_key", how="left")
    print("\n-- reasons for missing round-13 truth (by build) --")
    print(nt2.with_columns(
        pl.when(pl.col("sx13").is_null()).then(pl.lit("sx13_null"))
        .when(pl.col("sx14").is_null()).then(pl.lit("sx14_null"))
        .when(pl.col("sx13") != pl.col("sx14")).then(pl.lit("swap_13_14_score_invalid"))
        .when(pl.col("fe13") == False).then(pl.lit("no_freeze_end_13"))  # noqa: E712
        .otherwise(pl.lit("score_missing_no_bomb_no_elim")).alias("reason"))
        .group_by(["build_num", "reason"]).len().sort(["build_num", "reason"]))

    print("\n################ 2. Mixed matches (flip at 25 or 28 without flip at 13) and the 3 unflagged cases ################")
    fl25 = b.filter(pl.col("round") == 25).select(["match_key", "flip_v", "src"]).rename({"flip_v": "f25", "src": "src25"})
    fl28 = b.filter(pl.col("round") == 28).select(["match_key", "flip_v", "src"]).rename({"flip_v": "f28", "src": "src28"})
    mm = mk.join(fl25, on="match_key", how="left").join(fl28, on="match_key", how="left")
    mixed = mm.filter((pl.col("f13") == False) & ((pl.col("f25") == True) | (pl.col("f28") == True)))  # noqa: E712
    print(mixed.select(["match_key", "build_num", "map_name", "match_date", "f13", "T13", "src13", "f25", "src25",
                        "f28", "src28"]))

    print("\n################ 3. Three-flip matches (sum_ok) where header T-starters score != truth-derived X wins ################")
    sx = b.select(["match_key", "round", "sx", "R", "T_v", "src", "flip_v"])
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
        pl.len().alias("N")).join(h, on="match_key", how="inner")
    x = x.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") == pl.col("N")).alias("sum_ok"))
    xs = x.filter(pl.col("sum_ok") & (pl.col("n_flip") == 3))
    print("sum_ok three-flip matches:", xs.height)
    bad = xs.filter(pl.col("xw_T") != pl.col("t_starters_score_final"))
    print("mismatches (xw_T != header):", bad.height)
    print(bad.select(["match_key", "map_name", "xw_T", "xw_R", "t_starters_score_final", "N"]).head(25))
    # which channels supply the three flipped rounds in those matches
    bad_keys = bad.select("match_key")
    print("channel source at 13/25/28 for the mismatched matches:")
    print(b.join(bad_keys, on="match_key", how="inner").filter(pl.col("round").is_in([13, 25, 28]))
          .select(["round", "src", "flip_v"]).group_by(["round", "src", "flip_v"]).len().sort(["round", "src", "flip_v"]))
    good3 = xs.filter(pl.col("xw_T") == pl.col("t_starters_score_final"))
    print("channel source at 13/25/28 for the matches that agree with the header:")
    print(b.join(good3.select("match_key"), on="match_key", how="inner").filter(pl.col("round").is_in([13, 25, 28]))
          .select(["round", "src", "flip_v"]).group_by(["round", "src", "flip_v"]).len().sort(["round", "src", "flip_v"]))

    print("\n################ 4. Header total vs rounds for all 2024 matches, by build ################")
    print(mk.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") - pl.col("N")).alias("diff"))
          .group_by(["build_num", "diff"]).len().sort(["build_num", "diff"]).head(40))

    print("\n################ 5. Outside-13/25/28 flipped rows by source channel and build ################")
    out = b.filter(pl.col("flip_v") & ~pl.col("round").is_in([13, 25, 28]))
    out = out.join(mk.select(["match_key", "build_num"]), on="match_key", how="left")
    print(out.group_by(["build_num", "src"]).len().sort(["build_num", "src"]))
    print(out.group_by("round").len().sort("round"))

    print("\n################ 6. Lag test: is R(13) equal to the team-tracked truth of round 12 (T_sx at 12)? ################")
    t12 = b.filter(pl.col("round") == 12).select(["match_key", pl.col("T_sx").alias("T12_sx")])
    lag = b.filter(pl.col("round") == 13).join(t12, on="match_key", how="inner").join(
        mk.select(["match_key", "build_num"]), on="match_key", how="left")
    lag = lag.filter(pl.col("T_v").is_not_null() & pl.col("T12_sx").is_not_null())
    print(lag.group_by(["build_num", "flip_v"]).agg(
        pl.len().alias("n"), (pl.col("R") == pl.col("T12_sx")).sum().alias("R13==T12_sx"),
        (pl.col("T_v") == pl.col("T12_sx")).sum().alias("T13==T12_sx")).sort(["build_num", "flip_v"]))

    print("\n################ 7. match_key day folder versus header match_date (2024) ################")
    mk2 = mk.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("folder_day"),
                          pl.col("match_date").str.slice(0, 10).alias("date"))
    print("folder day -> header date (top 8 pairs by count):")
    print(mk2.group_by(["folder_day", "date"]).len().sort("len", descending=True).head(8))
    print("folder days 12 and 13: header date range and build:")
    print(mk2.filter(pl.col("folder_day").is_in(["12", "13"])).group_by(["folder_day", "build_num"]).agg(
        pl.len().alias("n"), pl.col("date").min().alias("min_date"), pl.col("date").max().alias("max_date")).sort(["folder_day", "build_num"]))

    print("\n################ 8. The 10 build-9961 matches with truth at 13 but no flip at 13 ################")
    print(mk.filter((pl.col("build_num") == 9961) & (pl.col("f13") == False)).select(  # noqa: E712
        ["match_key", "map_name", "match_date", "T13", "src13"]))


if __name__ == "__main__":
    main()
