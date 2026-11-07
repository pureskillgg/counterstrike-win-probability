"""verify D1-3: match-level scoreboard test and neighbour-round (label shift) test (read-only, 2024 tome).

Idea: the engine's final scoreboard (round_state t_score / ct_score at match end) is an aggregate that
does not depend on round_end.winner_team_code. If winner_team_code is side-correct, the number of rounds
won with winner==2 must equal the final T-side score, and winner==3 the final CT-side score.
The same test with S2 (score-side from round_state deltas) is tautological in part, so it is reported
as a control only.

ASSUMPTIONS:
  A1 t_score / ct_score are side-level cumulative counters with no halftime swap (T8b in the claim found
     no swap: round 13 start equals round 12 end). Checked here by the agreement rates below.
  A2 final score = the t_score / ct_score on the last round_state row (max tick) of the match.
  A3 rounds with no winner or no S2 are counted as missing, not as a mismatch.
  A4 day groups d01-07 vs d12-25 as in the claim.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def rd(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = rd("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).sort(["match_key", "round", "tick"])

pre0 = RS.group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("tp"), pl.col("ct_score").first().alias("cp"))
post0 = RS.group_by(["match_key", "round"]).agg(pl.col("t_score").last().alias("tq"), pl.col("ct_score").last().alias("cq"))
pre_fe = (RS.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"])
          .agg(pl.col("t_score").first().alias("tpf"), pl.col("ct_score").first().alias("cpf")))
post_fe = (RS.filter(pl.col("event_type") == "round_officially_ended").group_by(["match_key", "round"])
           .agg(pl.col("t_score").last().alias("tqf"), pl.col("ct_score").last().alias("cqf")))
sc = pre0.join(post0, on=["match_key", "round"]).join(pre_fe, on=["match_key", "round"], how="left").join(post_fe, on=["match_key", "round"], how="left")
sc = sc.with_columns(pl.coalesce(["tpf", "tp"]).alias("tp"), pl.coalesce(["cpf", "cp"]).alias("cp"),
                     pl.coalesce(["tqf", "tq"]).alias("tq"), pl.coalesce(["cqf", "cq"]).alias("cq"))
sc = sc.with_columns(
    pl.when(((pl.col("tq") - pl.col("tp")) == 1) & ((pl.col("cq") - pl.col("cp")) == 0)).then(2)
      .when(((pl.col("cq") - pl.col("cp")) == 1) & ((pl.col("tq") - pl.col("tp")) == 0)).then(3)
      .otherwise(None).alias("S2")).select(["match_key", "round", "S2"])

# per-match final scores (last row by tick over the whole match)
final = (RS.sort("tick").group_by("match_key").agg(pl.col("t_score").last().alias("final_t"),
                                                    pl.col("ct_score").last().alias("final_ct"),
                                                    pl.col("round").max().alias("max_round")))
R = RE.join(sc, on=["match_key", "round"], how="left")
R = R.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
R = R.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

M = (R.group_by("match_key").agg(
        pl.col("dg").first().alias("dg"),
        (pl.col("winner_team_code") == 2).sum().alias("W2"),
        (pl.col("winner_team_code") == 3).sum().alias("W3"),
        (pl.col("S2") == 2).sum().alias("S2_2"),
        (pl.col("S2") == 3).sum().alias("S2_3"),
        pl.col("round").max().alias("n_rounds_seen"),
        pl.len().alias("n_rows"))
     .join(final, on="match_key", how="left"))
M = M.with_columns(
    ((pl.col("W2") == pl.col("final_t")) & (pl.col("W3") == pl.col("final_ct"))).alias("winner_ok_scoreboard"),
    ((pl.col("S2_2") == pl.col("final_t")) & (pl.col("S2_3") == pl.col("final_ct"))).alias("S2_ok_scoreboard"),
)
# round-13 disagreement flag per match (winner vs S2 at round 13)
r13 = R.filter(pl.col("round") == 13).select(["match_key", (pl.col("winner_team_code") != pl.col("S2")).alias("r13_dis"),
                                               pl.col("S2").is_not_null().alias("r13_s2_defined")])
M = M.join(r13, on="match_key", how="left")

hdr("1. MATCH-LEVEL scoreboard test: rounds won by side (winner_team_code, S2) vs final scoreboard")
print("matches:", M.height)
print(M.group_by("dg").agg(pl.len().alias("n"),
                           pl.col("winner_ok_scoreboard").sum().alias("winner_matches_scoreboard"),
                           pl.col("S2_ok_scoreboard").sum().alias("S2_matches_scoreboard")).sort("dg"))
print("\nd12-25 matches: winner_ok_scoreboard by r13_dis flag")
print(M.filter(pl.col("dg") == "d12-25").group_by(["r13_dis", "winner_ok_scoreboard"]).len().sort(["r13_dis", "winner_ok_scoreboard"]))
print("\nd12-25 matches: S2_ok_scoreboard by r13_dis flag")
print(M.filter(pl.col("dg") == "d12-25").group_by(["r13_dis", "S2_ok_scoreboard"]).len().sort(["r13_dis", "S2_ok_scoreboard"]))
print("\nd01-07 matches: winner_ok_scoreboard distribution")
print(M.filter(pl.col("dg") == "d01-07").group_by("winner_ok_scoreboard").len())

hdr("2. examples: first 8 d12-25 matches with r13_dis true (W2, W3, S2_2, S2_3, final_t, final_ct)")
print(M.filter((pl.col("dg") == "d12-25") & (pl.col("r13_dis") == True))
      .select(["match_key", "W2", "W3", "S2_2", "S2_3", "final_t", "final_ct", "n_rounds_seen"]).head(8))
print("\nd12-25 matches with r13_dis false, first 4:")
print(M.filter((pl.col("dg") == "d12-25") & (pl.col("r13_dis") == False))
      .select(["match_key", "W2", "W3", "S2_2", "S2_3", "final_t", "final_ct", "n_rounds_seen"]).head(4))

hdr("3. NEIGHBOUR-ROUND test (label shift): for the d12-25 round-13 disagreements, is winner(13) equal to the result of round 12 or 14?")
X = R.filter(pl.col("dg") == "d12-25").select(["match_key", "round", "winner_team_code", "S2"])
w = X.select(["match_key", "round", pl.col("winner_team_code").alias("w"), pl.col("S2").alias("s2")])
w13 = w.filter(pl.col("round") == 13).select(["match_key", pl.col("w").alias("w13"), pl.col("s2").alias("s13")])
w12 = w.filter(pl.col("round") == 12).select(["match_key", pl.col("w").alias("w12"), pl.col("s2").alias("s12")])
w14 = w.filter(pl.col("round") == 14).select(["match_key", pl.col("w").alias("w14"), pl.col("s2").alias("s14")])
N = w13.join(w12, on="match_key", how="left").join(w14, on="match_key", how="left")
N = N.with_columns((pl.col("w13") != pl.col("s13")).alias("dis13"))
dis = N.filter(pl.col("dis13") & pl.col("s13").is_not_null())
print("disagreement rows (winner13 != S2_13):", dis.height)
print("  winner13 == winner12:", dis.filter(pl.col("w13") == pl.col("w12")).height,
      "  winner13 == S2_12:", dis.filter(pl.col("w13") == pl.col("s12")).height,
      "  winner13 == S2_14:", dis.filter(pl.col("w13") == pl.col("s14")).height,
      "  winner13 == winner14:", dis.filter(pl.col("w13") == pl.col("w14")).height)
print("  S2_13 == winner12:", dis.filter(pl.col("s13") == pl.col("w12")).height,
      "  S2_13 == winner14:", dis.filter(pl.col("s13") == pl.col("w14")).height)
ok = N.filter(~pl.col("dis13") & pl.col("s13").is_not_null())
print("control (agreeing rows): winner13 == S2_12:", ok.filter(pl.col("w13") == pl.col("s12")).height, " of", ok.height)
