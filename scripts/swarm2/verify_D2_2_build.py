"""Verifier D2-2: does the round-13/25/28 defect follow the game build in the match header?

Self-contained rebuild of the side-attached flags (truth T_SB = team-tracked score, else in-window bomb)
for 2024 (player_count 20) and 2026 (player_count 10, control), then cross-tabs against header build_num,
rushb_version and match day. No D2_*.parquet is read.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_2_build.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(16)
pl.Config.set_fmt_str_lengths(36)

ROOT = Path(r"G:\pureskill\csds-cs2")
PAD = 40  # ASSUMPTION: bomb events allowed up to 40 ticks past round_end (as D2_truth.py)
TOMES = {"2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", 20),
         "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", 10)}


def build(label):
    base, pc = TOMES[label]

    def files(ch):
        if ch == "header":
            return sorted(glob.glob(str(base / "header" / "dataframe_*")))
        out = []
        for fd in sorted(glob.glob(str(base / f"basic_{ch}.*"))):
            out += sorted(glob.glob(fd + "/dataframe_*"))
        return out

    def scan(ch, cols):
        return pl.scan_parquet(files(ch)).select(cols)

    re_raw = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
    RE = (re_raw.filter(pl.col("player_count") == pc).sort("tick").unique(["match_key", "round"], keep="first")
          .select(["match_key", "round", pl.col("tick").alias("t_end"), pl.col("winner_team_code").alias("R")]))
    hdr = pl.scan_parquet(files("header")).collect().unique("key", keep="first")
    HDR = hdr.select([pl.col("key").alias("match_key"), "map_name", "build_num", "rushb_version", "match_date"])

    rs = (scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
          .filter(pl.col("event_type") == "round_freeze_end").collect()
          .sort("tick").unique(["match_key", "round"], keep="first")
          .select(["match_key", "round", pl.col("tick").alias("t_fe"),
                   pl.col("t_score").alias("tS"), pl.col("ct_score").alias("cS")]))

    pi = (scan("player_info", ["match_key", "round", "player_id", "team_code"]).collect()
          .unique(["match_key", "round", "player_id"], keep="first"))
    half = 5  # player_info holds 10 players per 5v5 round in both tomes (round_end.player_count is a different count)
    cnt = pi.group_by(["match_key", "round"]).agg(
        pl.len().alias("n"), (pl.col("team_code") == 2).sum().alias("n2"), (pl.col("team_code") == 3).sum().alias("n3"))
    valid = cnt.filter((pl.col("n") == 2 * half) & (pl.col("n2") == half) & (pl.col("n3") == half))
    first_valid = valid.group_by("match_key").agg(pl.col("round").min().alias("F"))
    xmem = (pi.join(first_valid, on="match_key").filter(pl.col("round") == pl.col("F"))
            .filter(pl.col("team_code") == 2).select(["match_key", "player_id"]).with_columns(pl.lit(True).alias("isX")))
    pi2 = pi.join(xmem, on=["match_key", "player_id"], how="left").with_columns(pl.col("isX").fill_null(False))
    sxg = pi2.group_by(["match_key", "round"]).agg(
        (pl.col("isX") & (pl.col("team_code") == 2)).sum().alias("x2"),
        (pl.col("isX") & (pl.col("team_code") == 3)).sum().alias("x3"))
    sxg = sxg.with_columns(
        pl.when((pl.col("x2") >= 3) & (pl.col("x3") == 0)).then(pl.lit(2))
        .when((pl.col("x3") >= 3) & (pl.col("x2") == 0)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("sx"))
    sx_tbl = sxg.select(["match_key", "round", "sx"])
    sx_prev_tbl = sx_tbl.with_columns((pl.col("round") + 1).alias("round")).rename({"sx": "sx_prev"})
    sx_nx_tbl = sx_tbl.with_columns((pl.col("round") - 1).alias("round")).rename({"sx": "sx_nx"})

    bs = (scan("bomb_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).collect())
    bw = (bs.join(RE.select(["match_key", "round", "t_end"]), on=["match_key", "round"], how="inner")
          .join(rs.select(["match_key", "round", "t_fe"]), on=["match_key", "round"], how="inner")
          .filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + PAD)))
    bwin = bw.group_by(["match_key", "round"]).agg(
        (pl.col("event_type") == "bomb_exploded").any().alias("exp"),
        (pl.col("event_type") == "bomb_defused").any().alias("dfs"))
    bwin = bwin.with_columns(
        pl.when(pl.col("exp") & ~pl.col("dfs")).then(pl.lit(2))
        .when(pl.col("dfs") & ~pl.col("exp")).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("B_win")).select(["match_key", "round", "B_win"])

    fe_nx = rs.select(["match_key", (pl.col("round") - 1).alias("round"),
                       pl.col("tS").alias("tS_nx"), pl.col("cS").alias("cS_nx")])
    m = (RE.join(rs.select(["match_key", "round", "tS", "cS"]), on=["match_key", "round"], how="left")
         .join(fe_nx, on=["match_key", "round"], how="left")
         .join(sx_prev_tbl, on=["match_key", "round"], how="left")
         .join(sx_nx_tbl, on=["match_key", "round"], how="left")
         .join(sxg.select(["match_key", "round", "sx"]), on=["match_key", "round"], how="left")
         .join(bwin, on=["match_key", "round"], how="left")
         .join(HDR, on="match_key", how="left")
         .with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day")))
    m = m.with_columns(
        pl.when(pl.col("sx") == 2).then(pl.col("tS")).otherwise(pl.col("cS")).alias("x0"),
        pl.when(pl.col("sx") == 3).then(pl.col("tS")).otherwise(pl.col("cS")).alias("y0"),
        pl.when(pl.col("sx_nx") == 2).then(pl.col("tS_nx")).otherwise(pl.col("cS_nx")).alias("x1"),
        pl.when(pl.col("sx_nx") == 3).then(pl.col("tS_nx")).otherwise(pl.col("cS_nx")).alias("y1"))
    m = m.with_columns(
        (pl.col("x1") - pl.col("x0")).alias("dX"), (pl.col("y1") - pl.col("y0")).alias("dY"))
    m = m.with_columns(
        pl.when((pl.col("dX") == 1) & (pl.col("dY") == 0)).then(pl.col("sx"))
        .when((pl.col("dY") == 1) & (pl.col("dX") == 0)).then(5 - pl.col("sx"))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("S"))
    m = m.with_columns(pl.coalesce(["S", "B_win"]).alias("T_SB"))
    m = m.with_columns((pl.col("R") != pl.col("T_SB")).alias("fSB"))
    m = m.with_columns(pl.lit(label).alias("year"))
    return m


def report(m, label):
    print(f"\n########## {label} ##########")
    print(f"matches {m['match_key'].n_unique()}; rows {m.height}; T_SB known {m.filter(pl.col('T_SB').is_not_null()).height}")
    print("build_num x rushb_version (matches):")
    print(m.select(["match_key", "build_num", "rushb_version"]).unique("match_key")
          .group_by(["build_num", "rushb_version"]).len().sort("len", descending=True).rows())
    print("R != T_SB by round group:")
    print(m.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                         .when(pl.col("round") == 25).then(pl.lit("r25"))
                         .when(pl.col("round") == 28).then(pl.lit("r28"))
                         .otherwise(pl.lit("other")).alias("rb"))
          .filter(pl.col("T_SB").is_not_null()).group_by("rb")
          .agg(pl.len().alias("n"), pl.col("fSB").sum().alias("flips")).sort("rb").rows())
    mf = (m.filter((pl.col("round") == 13) & pl.col("T_SB").is_not_null())
          .group_by("match_key").agg(pl.col("fSB").any().alias("mflag13"), pl.col("build_num").first().alias("build"),
                                     pl.col("rushb_version").first().alias("rushb"), pl.col("day").first().alias("day"),
                                     pl.col("map_name").first().alias("map")))
    print("round-13 match flag by build_num (matches with round-13 truth):")
    print(mf.group_by("build").agg(pl.len().alias("n"), pl.col("mflag13").sum().alias("flagged")).sort("build").rows())
    print("round-13 match flag by day x build:")
    print(mf.group_by(["day", "build"]).agg(pl.len().alias("n"), pl.col("mflag13").sum().alias("flagged"))
          .sort(["day", "build"]).rows())
    print("round-13 match flag by build x map:")
    print(mf.group_by(["build", "map"]).agg(pl.len().alias("n"), pl.col("mflag13").sum().alias("flagged"))
          .sort(["build", "map"]).rows())
    return mf


m24 = build("2024")
mf24 = report(m24, "2024")
m26 = build("2026")
mf26 = report(m26, "2026")

# exceptions inside build 9961: flagged-free matches with round-13 truth
ex = mf24.filter((pl.col("build") == 9961) & (~pl.col("mflag13")))
print(f"\n2024 build 9961 matches with round-13 truth but no flip: {ex.height}")
print(ex.select(["match_key", "day", "map", "rushb"]).rows())

# ---------------------------------------------------------------- exceptions and discriminator detail (2024)
print("\n== 2024 detail: which header field separates? (match counts with round-13 truth)")
print(mf24.group_by(["rushb", "build"]).agg(pl.len().alias("n"), pl.col("mflag13").sum().alias("flagged")).sort(["build", "rushb"]).rows())
print("\n== 2024 detail: the 5 build-9961 matches with no round-13 flip, rounds 12-14 and 24-28")
ex_ids = ex["match_key"].to_list()
det = (m24.filter(pl.col("match_key").is_in(ex_ids) & pl.col("round").is_in([12, 13, 14, 24, 25, 27, 28]))
       .select(["match_key", "round", "sx_prev", "sx", "sx_nx", "R", "B_win", "S", "T_SB", "tS", "cS"])
       .sort(["match_key", "round"]))
print(det)
print("\n== 2024 build 9961 round-13 rows with T from bomb only (no score) and flip status")
bonly = m24.filter((pl.col("build_num") == 9961) & (pl.col("round") == 13) & pl.col("S").is_null() & pl.col("B_win").is_not_null())
print("bomb-only rows:", bonly.height, " flipped:", bonly.filter(pl.col("R") != pl.col("B_win")).height)

# ---------------------------------------------------------------- side-signal split (final)
print("\n== final: round-13 flag by build x side signal (S = team-tracked score available) ")
m13 = m24.filter((pl.col("round") == 13) & pl.col("T_SB").is_not_null()).with_columns(
    pl.when(pl.col("S").is_not_null()).then(pl.lit("S_known")).otherwise(pl.lit("bomb_only")).alias("src"))
print(m13.group_by(["build_num", "src"]).agg(pl.len().alias("rows"), pl.col("fSB").sum().alias("flips")).sort(["build_num", "src"]).rows())
print("\n== final: rows at rounds 14-24 in build-9961 matches with a side signal: sx and flips")
flagged_ids = mf24.filter(pl.col("mflag13"))["match_key"].to_list()
mid = m24.filter(pl.col("match_key").is_in(flagged_ids) & pl.col("round").is_between(14, 24) & pl.col("T_SB").is_not_null())
print(mid.group_by("sx").agg(pl.len().alias("rows"), pl.col("fSB").sum().alias("flips")).sort("sx").rows())
print("\n== final: 2026 control, rows at boundary rounds (flip counts)")
m26b = m26.filter(pl.col("round").is_in([13, 25, 28]) & pl.col("T_SB").is_not_null())
print(m26b.group_by("round").agg(pl.len().alias("rows"), pl.col("fSB").sum().alias("flips")).sort("round").rows())
