"""verify D7-3, part 2: (a) 2026 tome build_num distribution and round_end vs score/bomb mismatch in 2026,
(b) 2024 phase-based locator of each match's first second-half round (independent of the ledger swap flags),
and whether the score-vs-round_end flips sit on that round. Read-only; prints only.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")


def chan_dirs(base, ch):
    return sorted(p for p in base.glob(f"basic_{ch}.*") if p.is_dir())


def read_chan(base, ch, cols, where=None):
    parts = []
    for d in chan_dirs(base, ch):
        for f in sorted(glob.glob(str(d / "dataframe_*"))):
            lf = pl.scan_parquet(f).select(cols)
            if where is not None:
                lf = lf.filter(where)
            parts.append(lf.collect())
    return pl.concat(parts, how="vertical_relaxed")


def read_header(base):
    fs = sorted(glob.glob(str(base / "header" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "map_name"]) for f in fs],
                     how="vertical_relaxed")


def mdt(col):
    return col.str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False)


# ---------- (a) 2026 ----------
H26 = read_header(T26).rename({"key": "match_key"}).with_columns(mdt(pl.col("match_date")).alias("mdt"))
print("=== 2026 header ===")
print("rows:", H26.height, "| unique keys:", H26["match_key"].n_unique())
print("build_num counts:", H26.group_by("build_num").len().sort("build_num").rows())
print(H26.group_by("build_num").agg(pl.col("mdt").min().alias("mdt_min"), pl.col("mdt").max().alias("mdt_max"),
                                    pl.len().alias("n")).sort("build_num"))

RE26 = read_chan(T26, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_message"]) \
    .join(H26.select(["match_key", "build_num", "mdt"]), on="match_key", how="left")
RS26 = read_chan(T26, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
pre26 = (RS26.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"])
         .agg(pl.col("t_score").first().alias("pre_t"), pl.col("ct_score").first().alias("pre_ct")))
post26 = (RE26.select(["match_key", "round", "tick"]).rename({"tick": "end_tick"}).sort("end_tick")
          .join_asof(RS26.sort("tick").select(["match_key", "tick", "t_score", "ct_score"]),
                     left_on="end_tick", right_on="tick", by="match_key", strategy="forward", tolerance=2000)
          .unique(subset=["match_key", "round"], keep="first")
          .rename({"t_score": "post_t", "ct_score": "post_ct"}))
S26 = (RE26.join(pre26, on=["match_key", "round"], how="left")
       .join(post26.select(["match_key", "round", "post_t", "post_ct"]), on=["match_key", "round"], how="left")
       .with_columns(pl.when((pl.col("post_t") == pl.col("pre_t") + 1) & (pl.col("post_ct") == pl.col("pre_ct"))).then(pl.lit(2))
                     .when((pl.col("post_ct") == pl.col("pre_ct") + 1) & (pl.col("post_t") == pl.col("pre_t"))).then(pl.lit(3))
                     .otherwise(pl.lit(None)).cast(pl.Int64).alias("winner_score")))
S26 = S26.filter(pl.col("winner_score").is_not_null()).with_columns(
    (pl.col("winner_score") != pl.col("winner_team_code")).alias("flip"))
print("\n2026 rounds with a score-based winner:", S26.height)
print("2026 round_end vs score mismatch by build:")
print(S26.group_by("build_num").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips")).sort("build_num"))
print("2026 round_end reason vs winner check (reason text, winner, n) top:")
print(S26.group_by(["win_reason_message", "winner_team_code"]).len().sort("len", descending=True).head(10))

BS26 = read_chan(T26, "bomb_state", ["match_key", "round", "event_type"],
                 where=pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bo26 = (BS26.group_by(["match_key", "round"]).agg(pl.col("event_type").eq("bomb_defused").any().alias("defused"))
        .with_columns(pl.when(pl.col("defused")).then(pl.lit(3)).otherwise(pl.lit(2)).cast(pl.Int64).alias("bomb_side")))
Bm = S26.join(bo26.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="inner")
print("2026 rounds with bomb outcome and score:", Bm.height, "| bomb != round_end:",
      Bm.filter(pl.col("bomb_side") != pl.col("winner_team_code")).height,
      "| bomb != score:", Bm.filter(pl.col("bomb_side") != pl.col("winner_score")).height)

# ---------- (b) 2024 phase locator ----------
H24 = read_header(T24).rename({"key": "match_key"})
RS24 = read_chan(T24, "round_state", ["match_key", "round", "tick", "event_type", "phase"],
                 where=pl.col("event_type") == "round_freeze_end")
ph = RS24.group_by(["match_key", "round"]).agg(pl.col("phase").first().alias("phase"))
print("\n2024 phase values at round_freeze_end:", ph.group_by("phase").len().sort("len", descending=True).rows())
# first round per match whose phase is 'second' (ASSUMPTION: phase 'second' marks the post-switch half)
first_second = (ph.filter(pl.col("phase") == "second").group_by("match_key")
                .agg(pl.col("round").min().alias("first_second")))
Sx = pl.read_parquet(str(SCR / "verify_D7_3_score_rounds.parquet")).filter(pl.col("has_score")) \
    .with_columns((pl.col("winner_team_code") != pl.col("winner_score")).alias("flip"))
fl = (Sx.filter(pl.col("flip")).select(["match_key", "round"])
      .join(first_second, on="match_key", how="left")
      .with_columns((pl.col("round") == pl.col("first_second")).alias("at_fs")))
per = (fl.group_by("match_key").agg(pl.len().alias("n_flips"), pl.col("at_fs").sum().alias("n_at_fs")))
base = (H24.select(["match_key", "build_num", "map_name"]).join(first_second, on="match_key", how="left")
        .join(per, on="match_key", how="left")
        .with_columns(pl.col("n_flips").fill_null(0), pl.col("n_at_fs").fill_null(0)))
print("\n2024 matches with a 'second' phase round:", base.filter(pl.col("first_second").is_not_null()).height)
b9961 = base.filter(pl.col("build_num") == 9961).with_columns(
    pl.when(pl.col("n_flips") == 0).then(pl.lit("no flip"))
    .when(pl.col("n_flips") == pl.col("n_at_fs")).then(pl.lit("flips only at first-second round"))
    .otherwise(pl.lit("flip(s) elsewhere")).alias("cat"))
print("build 9961 matches by category:")
print(b9961.group_by("cat").len().sort("cat"))
print("build 9961, category x map:")
print(b9961.group_by(["map_name", "cat"]).len().sort(["map_name", "cat"]))
print("build 9961 matches where flips sit elsewhere (sample 15):")
print(b9961.filter(pl.col("cat") == "flip(s) elsewhere").head(15).rows())
print("build 9961 matches with no flip (sample 15):")
print(b9961.filter(pl.col("cat") == "no flip").head(15).rows())
b9842 = base.filter(pl.col("build_num") == 9842)
print("\nbuild 9842: matches with any flip:", b9842.filter(pl.col("n_flips") > 0).height, "of", b9842.height)
