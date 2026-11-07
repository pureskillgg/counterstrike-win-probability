"""verify_D7_4 part C: (1) round-1 completeness of round_end by period and build; (2) which halftime-swap
convention reproduces the header final scores in post-onset regulation matches (positive control: clean matches).
Reads: header, basic_round_end, basic_bomb_state (projected). Writes nothing.
ASSUMPTION: onset = 2024-02-07 01:01:18 (first -429 outcome row, from the recompute script).
ASSUMPTION: starter-T team's wins = code 2 (T) in rounds <= s, code 3 (CT) in rounds > s, for swap point s (tested 10..14);
            'flip' reverses the codes before/after s.
"""
import glob, sys
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(230)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
ONSET = pl.datetime(2024, 2, 7, 1, 1, 18)


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(ch):
    return pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_{ch}{SUF}")])


def hr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


H = (pl.read_parquet(files(BASE / "header")[0], columns=["key", "match_date", "build_num", "t_starters_score_final", "ct_starters_score_final"])
       .rename({"key": "match_key"})
       .with_columns(pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt")))
H = H.with_columns(pl.when(pl.col("mdt") < ONSET).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"))

RE = scan("round_end").select(["match_key", "round", "winner_team_code", "player_count"]).collect()
RE = RE.unique(["match_key", "round"], keep="first", maintain_order=True)
BS = (scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .select(["match_key", "round", "event_type"]).collect())
OUT = BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
R = (RE.join(OUT, on=["match_key", "round"], how="left")
       .with_columns(
           pl.when(pl.col("evs").list.contains("bomb_defused")).then(3)
             .when(pl.col("evs").list.contains("bomb_exploded")).then(2)
             .otherwise(None).cast(pl.Int64).alias("bomb_side"))
       .with_columns(pl.coalesce([pl.col("bomb_side"), pl.col("winner_team_code")]).alias("bsrc")))

hr("1. round-1 completeness in round_end, all 4,435 matches")
has1 = R.filter(pl.col("round") == 1).select("match_key").unique().with_columns(pl.lit(True).alias("has_r1"))
A = H.join(has1, on="match_key", how="left").with_columns(pl.col("has_r1").fill_null(False))
print(A.group_by(["period", "build_num", "has_r1"]).len().sort(["period", "build_num", "has_r1"]).rows())
r1_bomb = BS.filter(pl.col("round") == 1).select("match_key").unique()
lack = A.filter(~pl.col("has_r1")).join(r1_bomb.with_columns(pl.lit(True).alias("r1_bomb_event")), on="match_key", how="left")
print("matches lacking round_end round 1:", lack.height,
      "| of which have a bomb outcome event in round 1 (bomb_state):", lack.filter(pl.col("r1_bomb_event") == True).height)
print("round-number min per match (all matches), by period:",
      R.group_by("match_key").agg(pl.col("round").min().alias("minr")).join(H.select(["match_key", "period"]), on="match_key")
       .group_by(["period", "minr"]).len().sort(["period", "minr"]).rows())

hr("2. halftime-swap convention vs header t_starters_score_final, post-onset regulation matches")
mr = R.group_by("match_key").agg(pl.col("round").max().alias("maxr"))
reg = mr.filter(pl.col("maxr") == 24).select("match_key")
P = (R.join(reg, on="match_key", how="inner")
       .join(H.select(["match_key", "period", "t_starters_score_final", "ct_starters_score_final"]), on="match_key", how="inner")
       .filter(pl.col("period") == "post"))
M = P.group_by("match_key").agg(
    pl.col("t_starters_score_final").first().alias("t"),
    pl.col("ct_starters_score_final").first().alias("ct"),
    (pl.col("winner_team_code") != pl.coalesce([pl.col("bomb_side"), pl.col("winner_team_code")])).any().alias("has_mm"),
    pl.len().alias("nrows"),
)
cond_rows = []
for s in [10, 11, 12, 13, 14]:
    for flip in [False, True]:
        a, b = (3, 2) if flip else (2, 3)
        f_w = ((pl.col("round") <= s) & (pl.col("winner_team_code") == a)) | ((pl.col("round") > s) & (pl.col("winner_team_code") == b))
        f_b = ((pl.col("round") <= s) & (pl.col("bsrc") == a)) | ((pl.col("round") > s) & (pl.col("bsrc") == b))
        X = P.group_by("match_key").agg(f_w.cast(pl.Int64).sum().alias("w"), f_b.cast(pl.Int64).sum().alias("b"))
        Y = X.join(M, on="match_key")
        clean = Y.filter(~pl.col("has_mm"))
        mmm = Y.filter(pl.col("has_mm"))
        cond_rows.append((s, "flip" if flip else "std",
                          "clean n", clean.height, "exact W", int((clean["w"] == clean["t"]).sum()),
                          "| mm n", mmm.height, "exact W", int((mmm["w"] == mmm["t"]).sum()), "exact B", int((mmm["b"] == mmm["t"]).sum())))
for row in cond_rows:
    print(row)

hr("3. post clean matches: signed diff (header t - round-derived W) under swap point 12 std (the D7 convention)")
X12 = P.group_by("match_key").agg(
    pl.col("winner_team_code").count().alias("n"),
    (((pl.col("round") <= 12) & (pl.col("winner_team_code") == 2)) | ((pl.col("round") > 12) & (pl.col("winner_team_code") == 3))).cast(pl.Int64).sum().alias("w12"))
Z = X12.join(M, on="match_key")
print("clean: (t - w12) counts:", sorted(((Z.filter(~pl.col("has_mm"))["t"] - Z.filter(~pl.col("has_mm"))["w12"]).value_counts().rows())))
print("mismatch: (t - w12) counts:", sorted(((Z.filter(pl.col("has_mm"))["t"] - Z.filter(pl.col("has_mm"))["w12"]).value_counts().rows())))
print("header t + ct == 24 (post regulation):", int((M["t"] + M["ct"] == 24).sum()), "of", M.height)

print("\nDONE (no files written by this script).")
