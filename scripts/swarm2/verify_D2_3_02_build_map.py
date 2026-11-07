"""verify_D2_3_02: is the 2024 round-13/25/28 flip explained by a per-match attribute other than the
side truth itself? Tests build_num (game build in the demo header) and map_name against the bomb-only truth.

Reuses the bomb-only truth from verify_D2_3_01 (B = side of first in-window bomb outcome, A1-A4 as there).
Reads 2024 tome only. Column projection; one channel at a time. No files written.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D2_3_02_build_map.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
MK = "match_key"
TOL = 40  # ASSUMPTION A1 (same as verify_D2_3_01)


def files(ch):
    fd = sorted(glob.glob(str(BASE / f"basic_{ch}.*")))
    assert len(fd) == 1, (ch, fd)
    return sorted(glob.glob(fd[0] + "/dataframe_*"))


def scan(ch, cols):
    return pl.scan_parquet(files(ch)).select(cols)


# ---------------------------------------------------------------- header and join check
hdr = pl.read_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*"))),
                      columns=["match_id", "key", "map_name", "build_num", "match_date", "demo_id"])
print("header rows:", hdr.height, " distinct match_id:", hdr["match_id"].n_unique(),
      " distinct key:", hdr["key"].n_unique())

re_raw = scan("round_end", [MK, "round", "tick", "winner_team_code", "player_count"]).collect()
mk_tbl = pl.DataFrame({MK: re_raw[MK].unique().to_list()})
mk_tbl = mk_tbl.with_columns(pl.col(MK).str.split("/").list.last().alias("tail"))
print("match_keys in round_end:", mk_tbl.height)
print("  match_key == header.key coverage:", mk_tbl.join(hdr, left_on=MK, right_on="key", how="inner").height)
print("  key tail == header.match_id coverage:",
      mk_tbl.join(hdr, left_on="tail", right_on="match_id", how="inner").height)
print("  key tail == header.demo_id coverage:",
      mk_tbl.join(hdr, left_on="tail", right_on="demo_id", how="inner").height)
print("  sample match_key:", mk_tbl[MK][0], " sample key:", hdr["key"][0], " sample match_id:", hdr["match_id"][0])

hm = None
for lk, rk in [(MK, "key"), ("tail", "match_id"), ("tail", "demo_id")]:
    cand = mk_tbl.join(hdr, left_on=lk, right_on=rk, how="inner")
    if cand.height == mk_tbl.height:
        hm = cand.select([MK, "map_name", "build_num", "match_date"])
        print(f"using join {lk} -> {rk}: full coverage")
        break
if hm is None:
    # fall back to the best candidate and report coverage
    hm = mk_tbl.join(hdr, left_on="tail", right_on="match_id", how="inner").select([MK, "map_name", "build_num", "match_date"])
    print("WARNING: no full-coverage join; using tail->match_id with coverage", hm.height)

# ---------------------------------------------------------------- bomb-only truth (as in 01)
rs = (scan("round_state", [MK, "round", "tick", "event_type"])
      .filter(pl.col("event_type") == "round_freeze_end").collect())
fe = rs.sort("tick").unique([MK, "round"], keep="first").select([MK, "round", pl.col("tick").alias("t_fe")])
re_ = re_raw.sort("tick").unique([MK, "round"], keep="first").rename({"tick": "t_end"})
bs = scan("bomb_state", [MK, "round", "tick", "event_type"]).collect()
bs = (bs.join(fe, on=[MK, "round"], how="left").join(re_.select([MK, "round", "t_end"]), on=[MK, "round"], how="left")
      .with_columns(((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + TOL)).alias("in_win")))
ob = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]) & pl.col("in_win"))
      .sort("tick").group_by([MK, "round"]).agg(pl.col("event_type").first().alias("bo_first"))
      .with_columns(pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
                    .cast(pl.Int64).alias("B")))
k = (re_.select([MK, "round", "winner_team_code", "player_count"])
     .join(ob.select([MK, "round", "B"]), on=[MK, "round"], how="inner")
     .join(hm, on=MK, how="left")
     .with_columns(pl.col(MK).str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
                   (pl.col("winner_team_code") != pl.col("B")).alias("flip")))
print("\nrounds with bomb-only truth:", k.height, " flips:", k.filter("flip").height,
      " with map/build joined:", k.filter(pl.col("build_num").is_not_null()).height)

# ---------------------------------------------------------------- build and map at round 13
print("\n== B1. Build vs day across all 2024 matches (header)")
hd = hdr.with_columns(pl.col("match_date").str.slice(0, 10).alias("date"))
print(hd.group_by(["build_num"]).agg(pl.len().alias("matches"),
                                     pl.col("match_date").min().alias("first_date"),
                                     pl.col("match_date").max().alias("last_date")).sort("build_num"))
print(hd.with_columns(pl.col("match_date").str.slice(8, 2).alias("d"))
      .group_by(["d", "build_num"]).len().sort(["d", "build_num"]).rows())

r13 = k.filter(pl.col("round") == 13)
print("\n== B2. Round 13 flips by build_num (bomb-only truth)")
print(r13.group_by("build_num").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"),
                                    pl.col("flip").mean().round(3).alias("rate")).sort("build_num"))

print("\n== B3. Round 13 flips by build_num x day")
print(r13.group_by(["build_num", "day"]).agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"))
      .sort(["build_num", "day"]).rows())

print("\n== B4. Round 13 flips by map_name")
print(r13.group_by("map_name").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"),
                                   pl.col("flip").mean().round(3).alias("rate")).sort("map_name"))

print("\n== B5. Round 13 flips by build x map")
print(r13.group_by(["build_num", "map_name"]).agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped"))
      .sort(["build_num", "map_name"]))

# ---------------------------------------------------------------- all rounds with truth, by build
print("\n== B6. All rounds with bomb-only truth: flips by build x round bin")
rb = pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25")) \
    .when(pl.col("round") == 28).then(pl.lit("r28")).when(pl.col("round") <= 12).then(pl.lit("r01_12")) \
    .when(pl.col("round") <= 24).then(pl.lit("r14_24")).otherwise(pl.lit("other"))
print(k.with_columns(rb.alias("rbin")).group_by(["build_num", "rbin"])
      .agg(pl.len().alias("n"), pl.col("flip").sum().alias("flipped")).sort(["build_num", "rbin"]))

# ---------------------------------------------------------------- match-level: do r13 flips coincide with flips elsewhere?
print("\n== B7. Round 9 flips (bomb-only truth): which matches, which build, which map, and do they carry a round-13 flip?")
r9f = k.filter((pl.col("round") == 9) & pl.col("flip")).select([MK, "winner_team_code", "B", "build_num", "map_name", "day"])
print(r9f.sort(MK))
f13 = r13.select([MK, pl.col("flip").alias("f13")])
print("round-9 flipped matches with bomb truth at 13:",
      r9f.join(f13, on=MK, how="inner").height,
      " of which flipped at 13:", r9f.join(f13, on=MK, how="inner").filter("f13").height)
r9all = k.filter(pl.col("round") == 9).select([MK, pl.col("flip").alias("f9")])
print("round-9 rows with bomb truth:", r9all.height, " flipped:", r9all.filter("f9").height,
      " (round-9 matches with bomb truth in the same build and day ranges as the flips:",
      k.filter((pl.col("round") == 9) & pl.col("build_num").is_in([9961])).height, "rows in build 9961)")

# ---------------------------------------------------------------- match-level pattern under bomb truth
print("\n== B8. Match-level flips at 13/25/28 (bomb-only truth) by build")
piv = (k.filter(pl.col("round").is_in([13, 25, 28]))
       .select([MK, pl.col("round").cast(pl.Utf8), "flip", "build_num", "map_name"])
       .pivot(on="round", index=[MK, "build_num", "map_name"], values="flip", aggregate_function="first"))
print(piv.group_by(["build_num", "13"]).len().sort(["build_num", "13"]))
print("matches with a 13 flip value by map:")
print(piv.drop_nulls(["13"]).group_by(["map_name", "13"]).len().sort(["map_name", "13"]))

# ---------------------------------------------------------------- full-coverage build check at round 13 for matches without a bomb at 13
print("\n== B9. Matches in each build: how many have a bomb outcome at 13 at all (bomb-only coverage)")
allm = hm.with_columns(pl.col(MK).str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
has13 = k.filter(pl.col("round") == 13).select([MK]).unique().with_columns(pl.lit(True).alias("has13"))
cov = allm.join(has13, on=MK, how="left").with_columns(pl.col("has13").fill_null(False))
print(cov.group_by(["build_num", "has13"]).len().sort(["build_num", "has13"]))

print("\nend of build/map tables (no files written)")
