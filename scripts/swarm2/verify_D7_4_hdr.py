"""verify_D7_4 part B: re-check of the D7_4(b) header-score test, with a positive control.

Question: does the header final score (t_starters_score_final / ct_starters_score_final) discriminate
between label sources (winner_team_code vs bomb-derived side)?
Scope: regulation-only matches (max round 24 over all round_end rows); first row kept per (match_key, round).
Conventions for the team that started on T (ASSUMPTIONS, both tested):
  C1 (D7): T wins rounds 1-12 (code 2) + CT wins rounds 13-24 (code 3)
  C2 (flipped): CT wins rounds 1-12 (code 3) + T wins rounds 13-24 (code 2)
Label sources: W = winner_team_code; B = bomb_side where a bomb outcome exists, else winner_team_code.
Writes nothing; stdout is appended to logs\\swarm2\\verify_D7_4.log by the caller.
"""
import glob, sys
from collections import Counter
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(230)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
CUT = pl.datetime(2024, 2, 7)


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(ch, cols, pred=None):
    lf = pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_{ch}{SUF}")])
    if pred is not None:
        lf = lf.filter(pred)
    return lf.select(cols)


def hr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "t_starters_score_final",
                                           "ct_starters_score_final"]) for f in files(BASE / "header")])
H = H.rename({"key": "match_key"}).with_columns(
    pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"))

RE = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type"],
          pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).collect()
OUT = BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))

R = (RE.unique(subset=["match_key", "round"], keep="first")  # ASSUMPTION: first row kept for the one duplicate
       .join(OUT, on=["match_key", "round"], how="left")
       .with_columns(
           pl.when(pl.col("evs").list.contains("bomb_defused")).then(3)
             .when(pl.col("evs").list.contains("bomb_exploded")).then(2)
             .otherwise(None).cast(pl.Int64).alias("bomb_side"))
       .with_columns(
           pl.when(pl.col("bomb_side").is_not_null()).then(pl.col("bomb_side") != pl.col("winner_team_code")).otherwise(None).alias("mm"),
           pl.coalesce([pl.col("bomb_side"), pl.col("winner_team_code")]).alias("bsrc"),
       ))

mr = R.group_by("match_key").agg(pl.col("round").max().alias("maxr"))
reg = mr.filter(pl.col("maxr") == 24).select("match_key")
RR = R.join(reg, on="match_key", how="inner")
hr("0. scope")
print("regulation-only matches:", reg.height, "| round rows:", RR.height)


def wins(src, flip):
    a, b = (3, 2) if flip else (2, 3)
    return (((pl.col("round") <= 12) & (pl.col(src) == a)) | ((pl.col("round") >= 13) & (pl.col(src) == b))).cast(pl.Int64)


M = (RR.with_columns(
        wins("winner_team_code", False).alias("c1w"),
        wins("bsrc", False).alias("c1b"),
        wins("winner_team_code", True).alias("c2w"),
        wins("bsrc", True).alias("c2b"),
     )
     .group_by("match_key").agg(
        pl.len().alias("nrows"),
        pl.col("mm").sum().alias("n_mm"),
        pl.col("c1w").sum().alias("c1w"), pl.col("c1b").sum().alias("c1b"),
        pl.col("c2w").sum().alias("c2w"), pl.col("c2b").sum().alias("c2b"),
        pl.col("winner_team_code").count().alias("n_wc"),
     )
     .join(H.select(["match_key", "mdt", "build_num", "t_starters_score_final", "ct_starters_score_final"]), on="match_key", how="inner")
     .with_columns(
        pl.when(pl.col("mdt") < CUT).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"),
        (pl.col("n_mm") > 0).alias("has_mm"),
        (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") == 24).alias("hdr_sum24"),
        (pl.col("nrows") == 24).alias("rows24"),
     ))

hr("1. completeness and header internal consistency (regulation matches)")
print(M.group_by("period").agg(pl.len().alias("n"), pl.col("rows24").sum().alias("rows==24"),
                               pl.col("hdr_sum24").sum().alias("hdr t+ct==24"),
                               pl.col("nrows").min().alias("min_rows"), pl.col("nrows").max().alias("max_rows")).sort("period").rows())
print("nrows distribution, pre:", Counter(M.filter(pl.col("period") == "pre")["nrows"].to_list()).most_common(12))
print("nrows distribution, post:", Counter(M.filter(pl.col("period") == "post")["nrows"].to_list()).most_common(12))
print("header t+ct distribution, pre (top):", Counter(M.filter(pl.col("period") == "pre").select(
    (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))).to_series().to_list()).most_common(8))

hr("2. header-match rates by period x has_mm, for four formulas (C1/C2 x W/B)")
M = M.with_columns(
    (pl.col("c1w") == pl.col("t_starters_score_final")).alias("C1W_eq_t"),
    (pl.col("c1b") == pl.col("t_starters_score_final")).alias("C1B_eq_t"),
    (pl.col("c2w") == pl.col("t_starters_score_final")).alias("C2W_eq_t"),
    (pl.col("c2b") == pl.col("t_starters_score_final")).alias("C2B_eq_t"),
)
print(M.group_by(["period", "has_mm"]).agg(
    pl.len().alias("n"),
    pl.col("C1W_eq_t").sum().alias("C1W"), pl.col("C1B_eq_t").sum().alias("C1B"),
    pl.col("C2W_eq_t").sum().alias("C2W"), pl.col("C2B_eq_t").sum().alias("C2B"),
).sort(["period", "has_mm"]).rows())
print("note: for clean matches (has_mm False) B and W are identical by construction, so C1B == C1W there.")

hr("3. signed differences vs header t_starters_score_final")
for per, hm in [("post", False), ("post", True), ("pre", False), ("pre", True)]:
    sub = M.filter((pl.col("period") == per) & (pl.col("has_mm") == hm))
    if sub.height == 0:
        print(per, hm, "empty")
        continue
    d1w = Counter((sub["c1w"] - sub["t_starters_score_final"]).to_list())
    d1b = Counter((sub["c1b"] - sub["t_starters_score_final"]).to_list())
    d2w = Counter((sub["c2w"] - sub["t_starters_score_final"]).to_list())
    print(f"{per} has_mm={hm} n={sub.height}: C1W-t {sorted(d1w.items())} | C1B-t {sorted(d1b.items())} | C2W-t {sorted(d2w.items())}")

hr("4. one sample of post clean matches: header vs formulas")
print(M.filter((pl.col("period") == "post") & (~pl.col("has_mm"))).select(
    ["mdt", "build_num", "t_starters_score_final", "ct_starters_score_final", "c1w", "c2w", "nrows"])
    .sort("mdt").head(6).rows())

hr("5. pre-period regulation matches with fewer than 24 round rows: which rounds are present (first 3)")
pre_short = M.filter((pl.col("period") == "pre") & (~pl.col("rows24"))).select("match_key").head(3)["match_key"].to_list()
for mk in pre_short:
    rounds = RR.filter(pl.col("match_key") == mk)["round"].sort().to_list()
    print(mk[:44], "| n_rounds =", len(rounds), "| rounds =", rounds)

print("\nDONE (no files written by this script).")
