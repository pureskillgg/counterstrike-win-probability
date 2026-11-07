"""D1 follow-up checks (swarm2).

F1 team-code flip at every boundary 1->2 .. 29->30 (overtime switch points), mismatch-round groups
F2 mismatch rounds 9, 25 and 28: winner_team_code vs bomb outcome vs score side, and planter team
F3 spawn-based flip min 0.0 anomaly: which matches, and what player_info says for the same pairs
F4 round-13 mismatch split by bomb event (exploded vs defused) and planter team codes
F5 score side vs winner_team_code by round number in d12-25 (rounds not 13/25)
ASSUMPTIONS as in D1_flip_test_v2.py (A1-A8). Same data paths.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(70)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(60)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
OUT = Path(r"G:\pureskill\csds-cs2\scripts\swarm2")
FULL, NO, MIN_COV = 0.9, 0.1, 6
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fl(ch)], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"])
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_id_fixed", "player_team_code"])
PI_raw = rd("player_info", ["match_key", "round", "player_id_fixed", "team_code"])
SP = rd("player_spawn", ["match_key", "round", "tick", "player_id_fixed", "player_team_code"])
RS = rd("round_state", ["match_key", "round", "tick", "t_score", "ct_score"])

k = ["match_key", "round", "player_id_fixed"]
dup_matches = sorted(PI_raw.filter(pl.len().over(k) > 1)["match_key"].unique().to_list())
PI = PI_raw.filter(~pl.col("match_key").is_in(dup_matches))

outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
RO = (RE.filter(pl.col("player_count") == 20)
      .join(outcome, on=["match_key", "round"], how="inner")
      .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
      .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
                    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
                    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit("defused"))
                      .otherwise(pl.lit("exploded")).alias("bomb_event")))
mm_match = RO.group_by("match_key").agg(
    (pl.col("mismatch") & (pl.col("round") == 13)).any().alias("r13_mismatch_any"),
    (pl.col("round") == 13).sum().alias("r13_outcome_n"),
    pl.col("mismatch").sum().alias("n_mismatch"),
    pl.col("round").filter(pl.col("mismatch")).sort().alias("mismatch_rounds"))
mm_match = mm_match.with_columns(
    pl.col("mismatch_rounds").list.eval(pl.element().cast(pl.Utf8)).list.join(",").alias("mm_rounds_str"))
S = (MATCH_UNIVERSE := PI.select("match_key").unique()).join(mm_match, on="match_key", how="left").with_columns(
    pl.col("r13_mismatch_any").fill_null(False), pl.col("n_mismatch").fill_null(0),
    pl.col("r13_outcome_n").fill_null(0), pl.col("mm_rounds_str").fill_null(""))
S = S.with_columns(pl.when(pl.col("n_mismatch") > 0).then(pl.lit("affected")).otherwise(pl.lit("not_affected")).alias("aff"))

# ----------------------------------------------------------------------------- F1 flips 1..29
hdr("F1 team-code flip at every boundary b->b+1 for b=1..29 (ff = fraction of covered players whose code changed)")
parts = []
for b in range(1, 30):
    a = PI.filter(pl.col("round") == b).select(["match_key", "player_id_fixed", pl.col("team_code").alias("ta")])
    c = PI.filter(pl.col("round") == b + 1).select(["match_key", "player_id_fixed", pl.col("team_code").alias("tb")])
    g = (a.join(c, on=["match_key", "player_id_fixed"], how="inner")
          .with_columns((pl.col("ta") != pl.col("tb")).alias("flip"))
          .group_by("match_key").agg(pl.len().alias("n_both"), pl.col("flip").sum().alias("n_flip"))
          .with_columns(pl.lit(b).alias("b")))
    parts.append(g)
FL = pl.concat(parts).with_columns((pl.col("n_flip") / pl.col("n_both")).alias("ff"))
FLS = FL.join(S.select(["match_key", "mm_rounds_str", "r13_mismatch_any", "aff"]), on="match_key", how="inner")
FLS = FLS.with_columns(
    pl.when(pl.col("r13_mismatch_any")).then(pl.lit("r13_MISMATCH"))
      .when(pl.col("aff") == "affected").then(pl.lit("affected_no_r13"))
      .otherwise(pl.lit("not_affected")).alias("grp"))
prof = (FLS.filter(pl.col("n_both") >= MIN_COV)
        .group_by(["b", "grp"]).agg(pl.len().alias("n_eval"),
                                    (pl.col("ff") >= FULL).sum().alias("n_full"),
                                    (pl.col("ff") <= NO).sum().alias("n_none"),
                                    pl.col("ff").median().alias("median_ff"))
        .sort(["b", "grp"]))
print(prof.filter(pl.col("b").is_in([11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29])))
print("\nboundaries where at least one evaluable match flips partially (0.1 < ff < 0.9):")
print(FL.filter((pl.col("ff") > NO) & (pl.col("ff") < FULL)).group_by("b").len().sort("b"))

# ----------------------------------------------------------------------------- F2 rounds 9, 25, 28
hdr("F2 mismatch rounds 9, 25, 28: winner_team_code vs bomb outcome vs score side and planter team")
sc = (RS.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_before"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_before"),
        pl.col("t_score").sort_by("tick").last().alias("t_after"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_after")))
sc = sc.with_columns(
    pl.when((pl.col("t_after") - pl.col("t_before")) == 1).then(pl.lit(2))
      .when((pl.col("ct_after") - pl.col("ct_before")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side"))
PL = (BS.filter(pl.col("event_type") == "bomb_planted")
      .with_columns(pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("pid"))
      .filter(pl.col("pid").is_not_null())
      .group_by(["match_key", "round"]).agg(pl.col("pid").first(), pl.col("player_team_code").first().alias("bs_team")))
pi_same = PI.select(["match_key", "round", pl.col("player_id_fixed").alias("pid"), pl.col("team_code").alias("pi_team")])
PL = PL.join(pi_same, on=["match_key", "round", "pid"], how="left").select(["match_key", "round", "bs_team", "pi_team"])
R = (RO.join(sc.select(["match_key", "round", "score_side"]), on=["match_key", "round"], how="left")
       .join(PL, on=["match_key", "round"], how="left")
       .join(S.select(["match_key", "r13_mismatch_any"]), on="match_key", how="left"))
for rr in [9, 25, 28]:
    x = R.filter(pl.col("round") == rr)
    print(f"\n-- round {rr}: rows with bomb outcome = {x.height}, mismatches = {int(x['mismatch'].sum())}")
    print(x.group_by(["mismatch", "bomb_event", "bomb_side", "winner_team_code", "score_side"]).len()
          .sort(["mismatch", "bomb_event", "winner_team_code", "score_side"]))
    print("planter team (bs_team, pi_team) among mismatches:")
    print(x.filter(pl.col("mismatch")).group_by(["bs_team", "pi_team"]).len().sort("len", descending=True))
print("\nround 9 mismatch matches (match_key, day):")
m9 = R.filter((pl.col("round") == 9) & pl.col("mismatch")).select(["match_key", "winner_team_code", "bomb_event", "score_side", "bs_team", "pi_team"])
print(m9)

# ----------------------------------------------------------------------------- F3 spawn min-0 anomaly
hdr("F3 spawn-based flip: matches where spawn flip fraction is 0.0 (12->13), vs player_info flip for the same pairs")
spw = (SP.with_columns(pl.col("player_id_fixed").cast(pl.Int64, strict=False).alias("pid"))
       .filter(pl.col("pid").is_not_null() & ~pl.col("match_key").is_in(dup_matches))
       .sort("tick").group_by(["match_key", "round", "pid"]).agg(pl.col("player_team_code").first().alias("sp_team")))
sp12 = spw.filter(pl.col("round") == 12).select(["match_key", "pid", pl.col("sp_team").alias("s12")])
sp13 = spw.filter(pl.col("round") == 13).select(["match_key", "pid", pl.col("sp_team").alias("s13")])
sf = sp12.join(sp13, on=["match_key", "pid"], how="inner").with_columns((pl.col("s12") != pl.col("s13")).alias("flip"))
sfm = sf.group_by("match_key").agg(pl.len().alias("n_sp"), pl.col("flip").sum().alias("n_sp_flip"))
zero = sfm.filter(pl.col("n_sp_flip") == 0)
print("matches with spawn pairs at 12 and 13:", sfm.height, " matches with zero spawn flips:", zero.height)
print("pairs in zero-flip matches:", int(zero["n_sp"].sum()))
print("zero-flip matches, r13 status and day:")
zz = zero.join(S.select(["match_key", "r13_mismatch_any", "aff"]), on="match_key", how="left")
zz = zz.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
print(zz.group_by(["r13_mismatch_any", "aff"]).len())
print(zz.group_by("day").len().sort("day"))
print(zz.head(12))
for mk in zz.head(3)["match_key"].to_list():
    pi_b = FL.filter((pl.col("match_key") == mk) & (pl.col("b") == 12))
    print("\nexample", mk, "player_info boundary 12->13:", pi_b.select(["n_both", "n_flip", "ff"]).to_dicts())
    print(" spawn pairs 12->13 (pid, s12, s13):", sf.filter(pl.col("match_key") == mk).select(["pid", "s12", "s13"]).to_dicts())
    print(" player_info rounds 12/13 for these pids:")
    print(PI.filter((pl.col("match_key") == mk) & pl.col("round").is_in([11, 12, 13, 14]))
          .group_by("round").agg(pl.col("team_code").alias("codes"), pl.col("player_id_fixed").alias("pids")).sort("round"))

# ----------------------------------------------------------------------------- F4 r13 mismatch by bomb event and planter
hdr("F4 round 13 mismatch rounds: bomb event split and planter team codes")
x = R.filter((pl.col("round") == 13) & pl.col("mismatch"))
print(x.group_by(["bomb_event", "bomb_side", "winner_team_code", "score_side"]).len().sort("len", descending=True))
print(x.group_by(["bs_team", "pi_team"]).len().sort("len", descending=True))

# ----------------------------------------------------------------------------- F5 score vs winner by round number, d12-25
hdr("F5 winner_team_code vs score side by round number (player_count == 20), day group d12-25 vs d01-07")
RE20 = RE.filter(pl.col("player_count") == 20).join(sc.select(["match_key", "round", "score_side"]),
                                                     on=["match_key", "round"], how="inner")
RE20 = RE20.filter(pl.col("score_side").is_not_null()).with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"))
RE20 = RE20.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"),
                         (pl.col("score_side") != pl.col("winner_team_code")).alias("dis"))
rr = (RE20.group_by(["dg", "round"]).agg(pl.len().alias("n"), pl.col("dis").sum().alias("disagree"))
      .with_columns((pl.col("disagree") / pl.col("n")).round(4).alias("rate")).sort(["dg", "round"]))
print(rr.filter(pl.col("dg") == "d12-25").filter(pl.col("disagree") > 0))
print("\nd01-07 rounds with any disagreement:")
print(rr.filter((pl.col("dg") == "d01-07") & (pl.col("disagree") > 0)))
print("\nd12-25 disagreements by round, excluding 13 and 25 (top 15):")
print(rr.filter((pl.col("dg") == "d12-25") & (~pl.col("round").is_in([13, 25]))).sort("disagree", descending=True).head(15))

# are the d12-25 non-13/25 disagreements in matches flagged as affected?
dd = RE20.filter((pl.col("dg") == "d12-25") & (~pl.col("round").is_in([13, 25])) & pl.col("dis"))
print("\nd12-25 non-13/25 disagreement rows:", dd.height, " in matches with any bomb mismatch:",
      dd.join(S.select(["match_key", "aff"]), on="match_key", how="left").filter(pl.col("aff") == "affected").height)
print("d12-25 non-13/25 disagreement rows by match (top 10):")
print(dd.group_by("match_key").len().sort("len", descending=True).head(10))
print("\nd12-25 rounds 13 disagreement by day:")
print(RE20.filter((pl.col("dg") == "d12-25") & (pl.col("round") == 13)).group_by("day")
      .agg(pl.len().alias("n"), pl.col("dis").sum().alias("disagree"), pl.col("dis").mean().round(3).alias("rate")).sort("day"))
