"""Verifier for D5-10 (independent recomputation from the 2024 tome, read only).

Claim under test: in regime A (round_announce_match_start present), the team-A cumulative score built from
round_end winners disagrees with the freeze-end team-A score in 2,103 of 2,223 matches without an 8->9 swap
(15,614 of 41,818 states), with differences beginning at label 14, after an assumed round-13 winner inversion.

Sources used here (all in tomes\\2024-02-g\\tome\\csds, read only, column-projected):
  basic_round_state: round_freeze_end rows (t_score, ct_score = live-start scores), round_announce_match_start
                     and cs_win_panel_round markers (regime).
  basic_round_end:   winner_team_code per round (2 = T, 3 = CT).
  header:            t_starters_score_final, ct_starters_score_final (team totals for the starters of each side).

Side model (no team-mapping assumption in the boundary test):
  At boundary r -> r+1 with winner w of round r and freeze-end fields (t0, c0) at r and (t1, c1) at r+1:
    H0 (no side swap):  t1 = t0 + [w = T],  c1 = c0 + [w = CT]
    H1 (side swap):     t1 = c0 + [w = CT], c1 = t0 + [w = T]
  The test classifies each boundary by which hypothesis fits the recorded winner, and whether the flipped winner
  (round r result inverted) fits instead.

ASSUMPTIONS (marked where used):
  A1  Score fields are side-based (t_score = score of the team on T), per the bake-off note (bakeoff.py line 863).
  A2  Winner codes 2 and 3 are T and CT.
  A3  Team A = the starters on T in round 1; header t_starters_score_final is team A's final score.
  A4  Regime A = round_announce_match_start count > 0 (D5_10 definition).
  A5  Duplicate round_freeze_end or round_end rows for the same (match, round): first by tick is kept.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_10_main.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(50)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def scan(folder):
    return pl.scan_parquet(sorted(glob.glob(str(folder / "dataframe_*"))))


def hdr_path():
    return BASE / "header"


def ch_path(ch):
    return BASE / f"basic_{ch}{SUF}"


# ---------------------------------------------------------------- 0. load
print("== 0. load (2024 tome, read only) ==")
HD = scan(hdr_path()).select(["key", "match_id", "match_date", "number_of_points", "match_type",
                              "t_starters_score_final", "ct_starters_score_final"]).collect()
print(f"header rows: {HD.height}; unique keys: {HD['key'].n_unique()}")
print("number_of_points:", HD["number_of_points"].value_counts().sort("count", descending=True).head(5).rows())
print("match_type:", HD["match_type"].value_counts().sort("count", descending=True).head(5).rows())

RS = scan(ch_path("round_state")).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect()
RS = RS.with_columns(pl.col("round").cast(pl.Int64), pl.col("t_score").cast(pl.Int64), pl.col("ct_score").cast(pl.Int64))
RE_raw = scan(ch_path("round_end")).select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
RE_raw = RE_raw.with_columns(pl.col("round").cast(pl.Int64), pl.col("winner_team_code").cast(pl.Int64))
print(f"round_state rows: {RS.height}; round_end rows: {RE_raw.height}")
print("round_end winner_team_code values:", RE_raw["winner_team_code"].value_counts().sort("winner_team_code").rows())
print("round_end player_count values (top):", RE_raw["player_count"].value_counts().sort("count", descending=True).head(5).rows())

keys_hdr = set(HD["key"].to_list())
keys_rs = set(RS["match_key"].unique().to_list())
keys_re = set(RE_raw["match_key"].unique().to_list())
print(f"match keys: header {len(keys_hdr)}; round_state {len(keys_rs)} (not in header {len(keys_rs - keys_hdr)}); "
      f"round_end {len(keys_re)} (not in header {len(keys_re - keys_hdr)}); header with no round_state {len(keys_hdr - keys_rs)}")
print("example header key:", sorted(keys_hdr)[0])
print("example round_state key:", sorted(keys_rs)[0])

# ---------------------------------------------------------------- 1. regime markers
print("\n== 1. regime markers and duplicates ==")
mk = (RS.group_by("match_key").agg(
    (pl.col("event_type") == "round_announce_match_start").sum().alias("n_mstart"),
    (pl.col("event_type") == "cs_win_panel_round").sum().alias("n_cwp"),
    (pl.col("event_type") == "round_start").sum().alias("n_rs_start")))
MR = HD.select(["key", "match_id", "match_date", "t_starters_score_final", "ct_starters_score_final"]).rename({"key": "match_key"}).join(
    mk, on="match_key", how="left").with_columns(
    regime=pl.when(pl.col("n_mstart").fill_null(0) > 0).then(pl.lit("A")).otherwise(pl.lit("B")))
print("regime counts (all header matches):", MR["regime"].value_counts().sort("regime").rows())
print("crosstab regime x (mstart>0, cwp>0, round_state round_start>0):")
print(MR.with_columns((pl.col("n_cwp").fill_null(0) > 0).alias("cwp"), (pl.col("n_rs_start").fill_null(0) > 0).alias("rs"))
      .group_by(["regime", "cwp", "rs"]).len().sort(["regime", "cwp", "rs"]))
MR = MR.with_columns(day=pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1))
print("day range:", MR["day"].min(), MR["day"].max())

fe_all = RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
fe_dup = fe_all.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print(f"round_freeze_end rows: {fe_all.height}; (match, round) pairs with >1 row: {fe_dup.height}")
FE = fe_all.unique(subset=["match_key", "round"], keep="first").select(
    ["match_key", "round", "tick", "t_score", "ct_score"]).rename({"t_score": "t0", "ct_score": "c0", "tick": "fe_tick"})

re_dup = RE_raw.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("winner_team_code").n_unique().alias("nw"))
print(f"round_end (match, round) pairs with >1 row: {(re_dup['n'] > 1).sum()}; of these with >1 distinct winner: "
      f"{(re_dup['nw'] > 1).sum()}")
RE = RE_raw.sort(["match_key", "round", "tick"]).unique(subset=["match_key", "round"], keep="first").select(
    ["match_key", "round", "winner_team_code", "player_count"])
print(f"round_end rows after dedupe: {RE.height}; round_end labels == 1 by regime:")
print(RE.join(MR.select(["match_key", "regime"]), on="match_key", how="left").filter(pl.col("round") == 1)
      .group_by("regime").len().sort("regime"))

# ---------------------------------------------------------------- 2. boundary test
print("\n== 2. boundary test: freeze-end step (r -> r+1) vs round_end winner of round r ==")
RIGHT = FE.select(["match_key", pl.col("round") - 1, pl.col("t0").alias("t1"), pl.col("c0").alias("c1")])
PAIR = (FE.join(RIGHT, on=["match_key", "round"], how="inner")
        .join(RE.select(["match_key", "round", "winner_team_code"]), on=["match_key", "round"], how="left")
        .join(MR.select(["match_key", "regime", "day", "t_starters_score_final", "ct_starters_score_final"]),
              on="match_key", how="left"))
wT = pl.col("winner_team_code") == T
wC = pl.col("winner_team_code") == CT
wv = pl.col("winner_team_code").is_in([T, CT])
PAIR = PAIR.with_columns(
    H0=(pl.col("t1") == pl.col("t0") + wT.cast(pl.Int64)) & (pl.col("c1") == pl.col("c0") + wC.cast(pl.Int64)),
    H1=(pl.col("t1") == pl.col("c0") + wC.cast(pl.Int64)) & (pl.col("c1") == pl.col("t0") + wT.cast(pl.Int64)),
    H0f=(pl.col("t1") == pl.col("t0") + wC.cast(pl.Int64)) & (pl.col("c1") == pl.col("c0") + wT.cast(pl.Int64)),
    H1f=(pl.col("t1") == pl.col("c0") + wT.cast(pl.Int64)) & (pl.col("c1") == pl.col("t0") + wC.cast(pl.Int64)))
PAIR = PAIR.with_columns(
    cls=pl.when(~wv).then(pl.lit("no_winner"))
    .when(pl.col("H0") & pl.col("H1")).then(pl.lit("rec_both"))
    .when(pl.col("H0")).then(pl.lit("rec_H0"))
    .when(pl.col("H1")).then(pl.lit("rec_H1"))
    .when(pl.col("H0f") | pl.col("H1f")).then(pl.lit("flip_fit"))
    .otherwise(pl.lit("neither")))
PAIR = PAIR.with_columns(
    cls_flip_kind=pl.when(pl.col("cls") == "flip_fit").then(
        pl.when(pl.col("H0f") & pl.col("H1f")).then(pl.lit("both"))
        .when(pl.col("H0f")).then(pl.lit("H0"))
        .otherwise(pl.lit("H1"))).otherwise(pl.lit(None)))
print(f"boundary pairs: {PAIR.height}")
print("overall class counts by regime:")
print(PAIR.group_by(["regime", "cls"]).len().sort(["regime", "cls"]))

tab = (PAIR.filter(pl.col("round") <= 30).group_by(["regime", "round", "cls"]).len()
       .pivot(on="cls", index=["regime", "round"], values="len").fill_null(0).sort(["regime", "round"]))
print("\nclass counts by boundary label r (boundary r -> r+1), labels 1..30:")
print(tab)

# swap boundary specifics
print("\nH1 (swap) share at specific boundaries, by regime:")
spec = PAIR.filter(pl.col("round").is_in([8, 12, 13, 24, 25, 27, 28, 30, 31, 33]))
print(spec.group_by(["regime", "round"]).agg(
    pl.len().alias("pairs"),
    (pl.col("cls") == "rec_H1").sum().alias("rec_H1"),
    (pl.col("cls") == "rec_H0").sum().alias("rec_H0"),
    (pl.col("cls") == "flip_fit").sum().alias("flip_fit"),
    (pl.col("cls") == "neither").sum().alias("neither")).sort(["regime", "round"]))

# ---------------------------------------------------------------- 3. per-match picture
print("\n== 3. per-match picture ==")
pm = PAIR.group_by("match_key").agg(
    pl.col("cls").is_in(["flip_fit"]).sum().alias("n_flip"),
    pl.col("cls").is_in(["neither"]).sum().alias("n_neither"),
    pl.col("cls").is_in(["rec_H1"]).sum().alias("n_H1"),
    pl.col("cls").is_in(["rec_H1"]).sum().alias("n_H1_dummy"),
    pl.col("round").filter(pl.col("cls") == "flip_fit").min().alias("first_flip_r"),
    pl.col("round").filter(pl.col("cls") == "rec_H1").min().alias("first_H1_r"),
    pl.col("round").filter(pl.col("cls") == "rec_H1").max().alias("last_H1_r"),
).join(MR.select(["match_key", "regime"]), on="match_key", how="left")
print("matches with >=1 flip_fit boundary, by regime:")
print(pm.group_by("regime").agg((pl.col("n_flip") > 0).sum().alias("matches_with_flip"), pl.len().alias("matches")).sort("regime"))
print("matches with >=1 neither boundary, by regime:")
print(pm.group_by("regime").agg((pl.col("n_neither") > 0).sum().alias("matches_with_neither"), pl.len().alias("matches")).sort("regime"))
print("first flip_fit boundary label (count of matches), by regime:")
print(pm.filter(pl.col("n_flip") > 0).group_by(["regime", "first_flip_r"]).len().sort(["regime", "first_flip_r"]).head(60))
print("H1 swap pattern by match: first and last rec_H1 boundary, regime A (top combinations):")
print(pm.filter(pl.col("regime") == "A").group_by(["first_H1_r", "last_H1_r"]).len().sort("len", descending=True).head(12))
print("H1 swap pattern by match: first and last rec_H1 boundary, regime B (top combinations):")
print(pm.filter(pl.col("regime") == "B").group_by(["first_H1_r", "last_H1_r"]).len().sort("len", descending=True).head(12))

PAIR.write_parquet(Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
                        r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_10\pairs.parquet"))
FE.write_parquet(Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
                      r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_10\fe.parquet"))
RE.write_parquet(Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
                      r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_10\re.parquet"))
MR.write_parquet(Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
                      r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_10\mr.parquet"))
print("\ncached intermediate tables in scratchpad\\verify_D5_10 (no project output written)")
