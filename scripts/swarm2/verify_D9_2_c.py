"""verify_D9_2_c: third independent source for the round winner: score increments from round_state.

Method (ASSUMPTIONS marked):
  - ASSUMPTION: round_state rows with event_type == 'round_poststart' carry the score at the start of the round.
    score_start(r) = t_score / ct_score at the earliest such row of round r in the match.
  - Score-derived winner of round r = side whose score increased from score_start(r) to score_start(r+1)
    (T increment -> winner 2, CT increment -> winner 3; otherwise ambiguous, code 0).
  - ASSUMPTION: t_score / ct_score are side-scores at that moment; checked empirically below by comparing
    with the recorded round_end winner on non-swap rounds (9842 and 9961), where no defect is claimed.
  - Recorded winner = round_end.winner_team_code (first row per (match, round)).
  - Bomb side = 3 if bomb_defused, 2 if bomb_exploded (same rule as the claim).
  - pc20 = round_end.player_count == 20.
Question answered: at 9961 round 13, does the score agree with the recorded winner (winner field right,
and bomb/death side labels wrong), or with the bomb side (winner field wrong)?
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(name, cols):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    lf = pl.scan_parquet(files_of(fd))
    have = set(lf.collect_schema().names())
    miss = [c for c in cols if c not in have]
    print(f"[load] {name}: cols={[c for c in cols if c in have]} missing={miss}")
    return lf.select([c for c in cols if c in have]).collect()


H = scan("header", ["key", "build_num"]).with_columns(pl.col("key").alias("match_key"))

RE = scan("round_end", ["match_key", "round", "winner_team_code", "player_count"]).group_by(
    ["match_key", "round"]).agg(pl.col("winner_team_code").first().alias("winner"),
                                pl.col("player_count").first().alias("pc"))

BS = scan("bomb_state", ["match_key", "round", "tick", "event_type"])
OUT = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).group_by(
    ["match_key", "round"]).agg((pl.col("event_type") == "bomb_defused").any().alias("defused"),
                                (pl.col("event_type") == "bomb_exploded").any().alias("exploded"))

RST = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                           "t_score_raw", "ct_score_raw"])
print("round_state event types (poststart, win_panel):",
      RST.filter(pl.col("event_type") == "round_poststart").height,
      RST.filter(pl.col("event_type") == "cs_win_panel_round").height)
S = (RST.filter(pl.col("event_type") == "round_poststart")
        .sort("tick")
        .group_by(["match_key", "round"])
        .agg(pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("c0"),
             pl.col("t_score_raw").first().alias("t0r"), pl.col("ct_score_raw").first().alias("c0r")))
print("score_start rows:", S.height, "| (match,round) unique:", S.select(["match_key", "round"]).unique().height)
S = S.sort(["match_key", "round"]).with_columns(
    pl.col("t0").shift(-1).over("match_key").alias("t1"),
    pl.col("c0").shift(-1).over("match_key").alias("c1"),
    pl.col("t0r").shift(-1).over("match_key").alias("t1r"),
    pl.col("c0r").shift(-1).over("match_key").alias("c1r"),
    pl.col("round").shift(-1).over("match_key").alias("round_next"),
)
S = S.filter(pl.col("round_next") == pl.col("round") + 1)
S = S.with_columns(
    pl.when((pl.col("t1") - pl.col("t0") == 1) & (pl.col("c1") - pl.col("c0") == 0)).then(2)
      .when((pl.col("c1") - pl.col("c0") == 1) & (pl.col("t1") - pl.col("t0") == 0)).then(3)
      .otherwise(0).alias("score_winner"),
    pl.when((pl.col("t1r") - pl.col("t0r") == 1) & (pl.col("c1r") - pl.col("c0r") == 0)).then(2)
      .when((pl.col("c1r") - pl.col("c0r") == 1) & (pl.col("t1r") - pl.col("t0r") == 0)).then(3)
      .otherwise(0).alias("score_winner_raw"),
)
print("score-winner distribution (0=ambiguous):", S.group_by("score_winner").len().sort("score_winner").rows())
print("raw score-winner distribution:", S.group_by("score_winner_raw").len().sort("score_winner_raw").rows())

R = (RE.join(OUT, on=["match_key", "round"], how="left")
       .join(H.select(["match_key", "build_num"]), on="match_key", how="left")
       .join(S.select(["match_key", "round", "score_winner", "score_winner_raw"]),
             on=["match_key", "round"], how="left")
       .with_columns(pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False))
       .with_columns((pl.col("defused") | pl.col("exploded")).alias("has_out"),
                     (pl.col("pc") == 20).alias("pc20"),
                     (pl.col("build_num") == 9961).alias("is9961"),
                     pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                       .when(pl.col("round").is_in([25, 28])).then(pl.lit("r25_28"))
                       .when(pl.col("round") == 9).then(pl.lit("r9"))
                       .otherwise(pl.lit("other")).alias("rclass"))
       .with_columns(
           pl.when(pl.col("defused")).then(pl.lit(3, dtype=pl.Int64))
             .when(pl.col("exploded")).then(pl.lit(2, dtype=pl.Int64))
             .otherwise(None).alias("bomb_side"))
       .with_columns(
           (pl.col("score_winner") == pl.col("winner")).alias("score_eq_recorded"),
           (pl.col("score_winner") == pl.col("bomb_side")).alias("score_eq_bomb"),
           (pl.col("winner") != pl.col("bomb_side")).alias("rec_ne_bomb"),
       ))

print("\n== A. agreement of score-winner (non-ambiguous) with recorded winner, by build x round class, pc20")
A = R.filter(pl.col("pc20") & (pl.col("score_winner") != 0))
print(A.group_by(["build_num", "rclass"]).agg(
    pl.len().alias("n_nonamb"),
    pl.col("score_eq_recorded").sum().alias("eq_recorded"),
    (~pl.col("score_eq_recorded")).sum().alias("ne_recorded")).sort(["build_num", "rclass"]))

print("\n== B. bomb-decided rounds, pc20: score-winner vs recorded winner and vs bomb side")
B = R.filter(pl.col("pc20") & pl.col("has_out") & (pl.col("score_winner") != 0))
print(B.group_by(["build_num", "rclass"]).agg(
    pl.len().alias("n"),
    pl.col("score_eq_recorded").sum().alias("score=recorded"),
    pl.col("score_eq_bomb").sum().alias("score=bomb_side"),
    pl.col("rec_ne_bomb").sum().alias("recorded!=bomb")).sort(["build_num", "rclass"]))

print("\n== C. 9961 round 13 bomb-decided pc20 rounds: crosstab of score-winner x recorded winner x bomb side")
C = R.filter(pl.col("pc20") & pl.col("has_out") & pl.col("is9961") & (pl.col("round") == 13))
print(C.group_by(["score_winner", "winner", "bomb_side"]).agg(pl.len().alias("n")).sort(
    ["score_winner", "winner", "bomb_side"]))

print("\n== D. 9842 round 13 bomb-decided pc20 rounds: same crosstab")
D = R.filter(pl.col("pc20") & pl.col("has_out") & ~pl.col("is9961") & (pl.col("round") == 13))
print(D.group_by(["score_winner", "winner", "bomb_side"]).agg(pl.len().alias("n")).sort(
    ["score_winner", "winner", "bomb_side"]))

print("\n== E. raw score field (t_score_raw) agreement, 9961 vs 9842, pc20, round 13 and other rounds")
E = R
print(E.filter(pl.col("pc20") & (pl.col("score_winner_raw") != 0)).group_by(["build_num", "rclass"]).agg(
    pl.len().alias("n_nonamb"),
    (pl.col("score_winner_raw") == pl.col("winner")).sum().alias("raw_eq_recorded")).sort(["build_num", "rclass"]))

print("\n== F. 9961 round 13 round_end rows WITHOUT outcome (no bomb event): score-winner vs recorded")
F = R.filter(pl.col("pc20") & ~pl.col("has_out") & pl.col("is9961") & (pl.col("round") == 13) & (pl.col("score_winner") != 0))
print(F.group_by(["winner", "score_winner"]).agg(pl.len().alias("n")).sort(["winner", "score_winner"]))
G = R.filter(pl.col("pc20") & ~pl.col("has_out") & ~pl.col("is9961") & (pl.col("round") == 13) & (pl.col("score_winner") != 0))
print("  9842 round 13 no-outcome:")
print(G.group_by(["winner", "score_winner"]).agg(pl.len().alias("n")).sort(["winner", "score_winner"]))
print("\nDONE part c")
