"""verify D1-3: independent recomputation of the round-13 / 9 / 25 / 28 mismatch claim (2024 tome).

Read-only. Loads only round_end, bomb_state, round_state, round_start (projected columns).
Writes nothing except the printed report (captured to logs\\swarm2\\verify_D1_3.log).

Definitions (ASSUMPTIONS, marked in the report):
  A1 bomb outcome: bomb_defused -> bomb side 3 (CT); bomb_exploded -> bomb side 2 (T). Same as the claim.
  A2 headline set: round_end rows with player_count == 20 that have a bomb outcome row (claim definition).
  A3 score side S1 (claim-style): side whose score rose by exactly 1 between the first and last
     round_state row of the round (all event types, sorted by tick).
  A4 score side S2 (stricter, event-anchored): pre = score at round_freeze_end row (fallback first row);
     post = score at round_officially_ended row (fallback last row). Side = the one with +1 and the other +0.
  A5 day group: day parsed from match_key 'csds/2024/02/DD/...'; d01-07 vs d12-25 as in the claim.
  A6 score fields are treated as side-level (t = T side, ct = CT side) for the same round; this is
     checked, not assumed, by the bomb-vs-score agreement table.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def rd(ch, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


# ----------------------------------------------------------------------------- load
RE_raw = rd("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
RST = rd("round_start", ["match_key", "round", "tick"])
print("loaded: round_end", RE_raw.height, " bomb_state", BS.height, " round_state", RS.height, " round_start", RST.height)

hdr("0. integrity: duplicates and key coverage")
dupRE = RE_raw.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("round_end (match, round) keys with >1 row:", dupRE.height)
dupRE_win = (RE_raw.group_by(["match_key", "round"]).agg(pl.col("winner_team_code").n_unique().alias("nw"))
             .filter(pl.col("nw") > 1).height)
print("  of which with conflicting winner_team_code:", dupRE_win)
RE = RE_raw.unique(["match_key", "round"], keep="first")
print("unique round_end rounds:", RE.height, " matches:", RE["match_key"].n_unique())
print("winner_team_code values:", RE.group_by("winner_team_code").len().sort("winner_team_code").to_dicts())
print("player_count==20 rounds:", RE.filter(pl.col("player_count") == 20).height)

# outcome per round
oc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .group_by(["match_key", "round"]).agg(
          pl.col("event_type").unique().alias("evs"),
          pl.col("tick").min().alias("bomb_tick_min"),
          pl.col("tick").max().alias("bomb_tick_max"),
          pl.len().alias("n_out_rows")))
print("rounds with a defuse/explode row:", oc.height,
      " with BOTH defused and exploded:", oc.filter(pl.col("evs").list.len() > 1).height)
oc = oc.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))

# ----------------------------------------------------------------------------- score sides
RSs = RS.sort(["match_key", "round", "tick"])
s1 = RSs.group_by(["match_key", "round"]).agg(
    pl.col("t_score").first().alias("t_pre1"), pl.col("ct_score").first().alias("ct_pre1"),
    pl.col("t_score").last().alias("t_post1"), pl.col("ct_score").last().alias("ct_post1"),
    pl.col("tick").min().alias("rs_tick_min"), pl.col("tick").max().alias("rs_tick_max"))


def anchored(ev, which):
    d = RS.filter(pl.col("event_type") == ev)
    if which == "first":
        d = d.sort("tick").group_by(["match_key", "round"]).agg(
            pl.col("t_score").first().alias("t_" + ev), pl.col("ct_score").first().alias("ct_" + ev))
    else:
        d = d.sort("tick").group_by(["match_key", "round"]).agg(
            pl.col("t_score").last().alias("t_" + ev), pl.col("ct_score").last().alias("ct_" + ev))
    return d


pre2 = anchored("round_freeze_end", "first").rename({"t_round_freeze_end": "t_pre2", "ct_round_freeze_end": "ct_pre2"})
post2 = anchored("round_officially_ended", "last").rename({"t_round_officially_ended": "t_post2",
                                                           "ct_round_officially_ended": "ct_post2"})
s2 = s1.join(pre2, on=["match_key", "round"], how="left").join(post2, on=["match_key", "round"], how="left")
s2 = s2.with_columns(
    pl.coalesce(["t_pre2", "t_pre1"]).alias("t_pre2"), pl.coalesce(["ct_pre2", "ct_pre1"]).alias("ct_pre2"),
    pl.coalesce(["t_post2", "t_post1"]).alias("t_post2"), pl.coalesce(["ct_post2", "ct_post1"]).alias("ct_post2"))


def side_from(tpre, tpost, cpre, cpost):
    tr = tpost - tpre
    cr = cpost - cpre
    return (pl.when((tr == 1) & (cr == 0)).then(2)
              .when((cr == 1) & (tr == 0)).then(3)
              .otherwise(None))


s2 = s2.with_columns(
    side_from(pl.col("t_pre1"), pl.col("t_post1"), pl.col("ct_pre1"), pl.col("ct_post1")).alias("side_S1strict"),
    side_from(pl.col("t_pre2"), pl.col("t_post2"), pl.col("ct_pre2"), pl.col("ct_post2")).alias("side_S2"),
    pl.when((pl.col("t_post1") - pl.col("t_pre1")) == 1).then(2)
      .when((pl.col("ct_post1") - pl.col("ct_pre1")) == 1).then(3)
      .otherwise(None).alias("side_S1claim"),
)
print("\nscore side coverage over all (match, round) in round_state:", s2.height)
print("  S1 strict:", s2.group_by("side_S1strict", maintain_order=True).len().to_dicts())
print("  S2 (event-anchored):", s2.group_by("side_S2", maintain_order=True).len().to_dicts())
print("  S1 claim-style:", s2.group_by("side_S1claim", maintain_order=True).len().to_dicts())

# round window ticks
freeze = RST.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("start_tick"))  # round_start row
RE_t = RE.select(["match_key", "round", "tick", "winner_team_code", "player_count"]).rename({"tick": "end_tick"})

# ----------------------------------------------------------------------------- round table
R = (RE_t.join(oc, on=["match_key", "round"], how="left")
     .join(s2, on=["match_key", "round"], how="left")
     .join(freeze, on=["match_key", "round"], how="left")
     .with_columns(
         pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
     )
     .with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg")))
R = R.with_columns(
    (pl.col("bomb_side") != pl.col("winner_team_code")).alias("bomb_mm"),
    (pl.col("side_S2") != pl.col("winner_team_code")).alias("S2_dis"),
    (pl.col("side_S1strict") != pl.col("winner_team_code")).alias("S1_dis"),
)

# ----------------------------------------------------------------------------- 1. headline recompute
hdr("1. HEADLINE recompute (claim definition: player_count==20 and bomb outcome)")
H = R.filter((pl.col("player_count") == 20) & pl.col("bomb_side").is_not_null())
print("outcome rounds:", H.height, " mismatches:", int(H["bomb_mm"].sum()), "  (claim: 15436 / 499)")
print("by round bin:")
print(H.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
                     .otherwise(pl.lit("other")).alias("rbin"))
      .group_by("rbin").agg(pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm"),
                            pl.col("bomb_mm").mean().round(4).alias("rate")).sort("rbin"))
print("mismatch count by round number (rounds with >=1 mismatch):")
mm_by_round = H.filter(pl.col("bomb_mm")).group_by("round").len().sort("round")
print(mm_by_round.transpose(include_header=True) if mm_by_round.height <= 20 else mm_by_round)
print("bomb-round count by round number 1..30 (n, mm):")
bym = H.group_by("round").agg(pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm")).sort("round")
print(bym.filter(pl.col("round") <= 30))
print("\nround-13 mismatch by day group:")
print(H.filter(pl.col("round") == 13).group_by("dg").agg(pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm"),
                                                          pl.col("bomb_mm").mean().round(4).alias("rate")))
print("round-13 mismatch by day (d12-25):")
print(H.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25"))
      .group_by("day").agg(pl.len().alias("n"), pl.col("bomb_mm").sum().alias("mm")).sort("day"))

# sensitivity: no player_count filter
Hall = R.filter(pl.col("bomb_side").is_not_null())
print("\nSENSITIVITY no player_count filter: outcome rounds", Hall.height, " mismatches", int(Hall["bomb_mm"].sum()))
print("  rounds with player_count != 20 and a bomb outcome:", Hall.filter(pl.col("player_count") != 20).height,
      " mismatches among them:", int(Hall.filter(pl.col("player_count") != 20)["bomb_mm"].sum()))

# ----------------------------------------------------------------------------- 2. score side vs winner, all rounds
hdr("2. WINNER vs SCORE SIDE on ALL rounds (not only bomb rounds), by round number, day group")
A = R.filter(pl.col("side_S2").is_not_null())
print("rounds with an unambiguous S2 score side:", A.height, " of", R.height)
prof = (A.group_by(["dg", "round"]).agg(pl.len().alias("n"), pl.col("S2_dis").sum().alias("disagree"),
                                        pl.col("S2_dis").mean().round(4).alias("rate"))
        .sort(["dg", "round"]))
print("\nd12-25, per round number (n, disagree with S2, rate):")
print(prof.filter((pl.col("dg") == "d12-25") & (pl.col("round") <= 30)).select(["round", "n", "disagree", "rate"]))
print("\nd01-07, per round number (n, disagree with S2, rate):")
print(prof.filter((pl.col("dg") == "d01-07") & (pl.col("round") <= 30)).select(["round", "n", "disagree", "rate"]))

# ----------------------------------------------------------------------------- 3. S1 vs S2 vs claim agreement
hdr("3. score-side definition sensitivity")
both = R.filter(pl.col("side_S2").is_not_null() & pl.col("side_S1strict").is_not_null())
print("rounds where S1 strict and S2 both defined:", both.height,
      " agree:", both.filter(pl.col("side_S1strict") == pl.col("side_S2")).height)
cl = R.filter(pl.col("side_S2").is_not_null() & pl.col("side_S1claim").is_not_null())
print("claim-style S1 vs S2 agree:", cl.filter(pl.col("side_S1claim") == pl.col("side_S2")).height, "of", cl.height)
print("S2 defined but S1 claim undefined:", R.filter(pl.col("side_S2").is_not_null() & pl.col("side_S1claim").is_null()).height)
print("S1 claim defined but S2 undefined:", R.filter(pl.col("side_S2").is_null() & pl.col("side_S1claim").is_not_null()).height)

# ----------------------------------------------------------------------------- 4. yardstick validation: bomb vs S2
hdr("4. VALIDATION of S2 as a yardstick: bomb side vs S2 on bomb rounds (player_count==20)")
V = R.filter((pl.col("player_count") == 20) & pl.col("bomb_side").is_not_null() & pl.col("side_S2").is_not_null())
V = V.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
                   .otherwise(pl.lit("other")).alias("rbin"))
print(V.group_by(["dg", "rbin"]).agg(pl.len().alias("n"),
                                     (pl.col("bomb_side") == pl.col("side_S2")).sum().alias("bomb_eq_S2"),
                                     (pl.col("bomb_side") != pl.col("side_S2")).sum().alias("bomb_ne_S2"))
      .sort(["dg", "rbin"]))
print("round-number profile of bomb_side != S2 on bomb rounds (all days):")
print(V.filter(pl.col("bomb_side") != pl.col("side_S2")).group_by("round").len().sort("round"))

# ----------------------------------------------------------------------------- 5. round-13 disagreement per match
hdr("5. ROUND 13 winner vs S2, per match (d12-25 and d01-07), all round-13 rows (bomb or not)")
r13 = R.filter((pl.col("round") == 13) & pl.col("side_S2").is_not_null())
print("round-13 rows with S2:", r13.height)
print(r13.group_by(["dg", "S2_dis"]).len().sort(["dg", "S2_dis"]))
print("\nround-13 rows by (dg, bomb status, S2 disagree):")
print(r13.with_columns(pl.when(pl.col("bomb_side").is_null()).then(pl.lit("no_bomb"))
                       .when(pl.col("bomb_mm")).then(pl.lit("bomb_MM")).otherwise(pl.lit("bomb_ok")).alias("bstat"))
      .group_by(["dg", "bstat", "S2_dis"]).len().sort(["dg", "bstat", "S2_dis"]))

pm = r13.group_by(["match_key", "dg"]).agg(pl.len().alias("n13"), pl.col("S2_dis").sum().alias("dis13"))
print("\nper-match round-13 disagreement fraction (d12-25 matches with a round-13 score side):")
print(pm.filter(pl.col("dg") == "d12-25").with_columns((pl.col("dis13") / pl.col("n13")).round(2).alias("frac"))
      .group_by("frac").len().sort("frac"))
print("d01-07 matches, round-13 disagreement count:", pm.filter((pl.col("dg") == "d01-07") & (pl.col("dis13") > 0)).height,
      "of", pm.filter(pl.col("dg") == "d01-07").height)

# ----------------------------------------------------------------------------- 6. do rounds 14-24 agree with S2 in matches where r13 disagrees?
hdr("6. Matches with a round-13 S2 disagreement: disagreement elsewhere (14-24, 25, 28, 9, others)")
bad13 = pm.filter((pl.col("dis13") > 0) & (pl.col("dg") == "d12-25")).select("match_key")
print("d12-25 matches with >=1 round-13 S2 disagreement:", bad13.height)
rest = R.filter(pl.col("side_S2").is_not_null()).join(bad13, on="match_key", how="inner")
rest = rest.with_columns(pl.when(pl.col("round") == 13).then(pl.lit("13"))
                         .when(pl.col("round").is_between(14, 24)).then(pl.lit("14-24"))
                         .when(pl.col("round") == 25).then(pl.lit("25"))
                         .when(pl.col("round") == 28).then(pl.lit("28"))
                         .when(pl.col("round") == 9).then(pl.lit("9"))
                         .when(pl.col("round") <= 12).then(pl.lit("1-12 (not 9)"))
                         .otherwise(pl.lit("26+ (not 28)")).alias("bin"))
print(rest.group_by("bin").agg(pl.len().alias("n_rounds"), pl.col("S2_dis").sum().alias("S2_disagree"),
                               pl.col("S2_dis").mean().round(4).alias("rate")).sort("bin"))

# ----------------------------------------------------------------------------- 7. timing alignment of bomb rows
hdr("7. timing: bomb outcome tick vs labelled round window (round_start tick .. round_end tick)")
T = R.filter(pl.col("bomb_side").is_not_null()).with_columns(
    (pl.col("bomb_tick_min") >= pl.col("start_tick")).alias("ge_start"),
    (pl.col("bomb_tick_max") <= pl.col("end_tick")).alias("le_end"),
    (pl.col("end_tick") - pl.col("bomb_tick_max")).alias("end_minus_bomb"))
print(T.group_by(["bomb_mm", "round"]).agg(pl.len().alias("n"), pl.col("ge_start").mean().round(3).alias("ge_start"),
                                          pl.col("le_end").mean().round(3).alias("le_end"),
                                          pl.col("end_minus_bomb").median().alias("median_end_minus_bomb"))
      .filter(pl.col("round").is_in([9, 12, 13, 14, 25, 28])).sort(["round", "bomb_mm"]))
