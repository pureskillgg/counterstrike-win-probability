"""Independent verifier for D1-3 (2024 tome). Recomputes the claim's numbers and tests alternatives.

Claim under test:
  (a) bomb-based mismatch (player_count == 20, bomb outcome present) = 499 of 15,436 rounds;
      by round: 9 -> 10, 13 -> 434, 25 -> 24, 28 -> 31; rounds 14-24 -> 0; 480 affected matches.
  (b) mismatch is confined to rounds 9, 13, 25, 28 (not 'from round 13 on').
  (c) fix: correct only rounds 9, 13, 25, 28 of affected matches; rounds 14-24 agree with
      the bomb outcome and with the score side.

ASSUMPTIONS (marked in the log):
  A1 bomb outcome: bomb_exploded -> side 2 (T), bomb_defused -> side 3 (CT); if both events are
     present in a round, defused wins (same convention as scripts/swarm2/D1_flip_test_v2.py).
  A2 round window: round_start.tick <= outcome tick <= round_end.tick (no tolerance).
  A3 score side: side whose score increments between the first round_state row of round r and
     the first round_state row of round r+1 (same match). Last rounds without r+1 are skipped.
  A4 day group: folder day from match_key (claim convention) AND header match_date (alternative).
  A5 round_end duplicates: keep first (only one duplicated pair; values identical).
  A6 map: header.map_name, one value per match.
  A7 the claim's round bins: '9', '13', '25', '28', '14-24', 'other'.
Read-only on the tome. Writes nothing except what the caller redirects to the log.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)
pl.Config.set_tbl_width_chars(220)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
HI_DAYS = [f"{d:02d}" for d in range(12, 26)]
CLAIM_ROUNDS = [9, 13, 25, 28]


def files(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")


def hdr(t):
    print("\n" + "=" * 110 + f"\n{t}\n" + "=" * 110)


def rbin_expr():
    return (pl.when(pl.col("round").is_in(CLAIM_ROUNDS)).then(pl.col("round").cast(pl.Utf8))
            .when(pl.col("round").is_between(14, 24)).then(pl.lit("14-24"))
            .otherwise(pl.lit("other")).alias("rbin"))


# ------------------------------------------------------------------ load
hdr("A. LOAD AND SETUP")
RE_all = rd("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                          "win_reason_message", "player_count"])
n_dup = RE_all.height - RE_all.select(["match_key", "round"]).unique().height
print(f"round_end rows={RE_all.height} duplicate (match,round) rows={n_dup}")
RE = RE_all.unique(["match_key", "round"], keep="first", maintain_order=True)
RSTA = rd("round_start", ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
BS = rd("bomb_state", ["match_key", "round", "tick", "event_type"])
RST = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
HD = (pl.concat([pl.read_parquet(f, columns=["key", "map_name", "match_date"]) for f in hf],
                how="vertical_relaxed")
      .rename({"key": "match_key"}))
print(f"round_end={RE.height} round_start={RSTA.height} bomb_state={BS.height} round_state={RST.height} header={HD.height}")
print("bomb_state event types:", BS.group_by("event_type").len().sort("event_type").to_dicts())

MATCH = HD.with_columns(
    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("fday"),
    pl.col("match_date").str.slice(0, 10).alias("mdate"),
    pl.col("match_date").str.slice(8, 2).alias("mday_str"),
)
mon_check = MATCH.group_by(pl.col("match_date").str.slice(0, 7).alias("ym")).len().sort("ym")
print("header match_date months:", mon_check.to_dicts())
fd_vs_md = MATCH.group_by(["fday", "mdate"]).len().height
print("distinct (folder day, match_date) pairs:", fd_vs_md)
print("matches where folder day != match_date day:",
      MATCH.filter(pl.col("fday") != pl.col("mday_str")).height, "of", MATCH.height)
# folder day groups as used by claim
MATCH = MATCH.with_columns(pl.when(pl.col("fday").is_in(HI_DAYS)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg_f"))
MATCH = MATCH.with_columns(pl.when(pl.col("mday_str").is_in(HI_DAYS)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg_m"))
print("claim's d12-25 matches by folder day:", MATCH.filter(pl.col("dg_f") == "d12-25").height,
      " by match_date day:", MATCH.filter(pl.col("dg_m") == "d12-25").height)

# ------------------------------------------------------------------ bomb outcome (A1)
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"])
        .agg(pl.col("event_type").unique().alias("evs"),
             pl.col("tick").min().alias("out_tick"),
             pl.len().alias("n_out_events")))
planted = (BS.filter(pl.col("event_type") == "bomb_planted")
           .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("plant_tick")))

RO_all = (RE.join(outc, on=["match_key", "round"], how="inner")
          .join(planted, on=["match_key", "round"], how="left")
          .join(RSTA.rename({"tick": "start_tick"}), on=["match_key", "round"], how="left")
          .with_columns(
              pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3)).otherwise(pl.lit(2)).alias("bomb_side"),
              pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit("defused")).otherwise(pl.lit("exploded")).alias("otype"),
              pl.col("evs").list.len().alias("n_types"))
          .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))
          .join(MATCH.select(["match_key", "fday", "mdate", "dg_f", "dg_m", "map_name"]), on="match_key", how="left")
          .with_columns(rbin_expr()))
RO = RO_all.filter(pl.col("player_count") == 20)
print(f"\nbomb-outcome rounds (all player counts)={RO_all.height}; player_count==20 -> {RO.height}; "
      f"mismatches={int(RO['mismatch'].sum())} (claim 15436 / 499)")
print("rounds with both defused and exploded events:", RO_all.filter(pl.col("n_types") > 1).height,
      " of which mismatch:", RO_all.filter((pl.col("n_types") > 1) & pl.col("mismatch")).height)

# ------------------------------------------------------------------ 1. counts by round
hdr("1. BOMB-BASED MISMATCH BY ROUND NUMBER (player_count == 20)")
byr = (RO.group_by("round").agg(pl.len().alias("n_bomb_rounds"), pl.col("mismatch").sum().alias("mm"))
       .sort("round"))
print(byr.filter((pl.col("mm") > 0) | pl.col("round").is_in([13, 25, 28])))
print("rounds 1-8,10-12 mismatches:", RO.filter(pl.col("mismatch") & pl.col("round").is_in(list(range(1, 9)) + [10, 11, 12])).height)
print("rounds 14-24 mismatches:", RO.filter(pl.col("mismatch") & pl.col("round").is_between(14, 24)).height)
print("rounds 26+ mismatches:", RO.filter(pl.col("mismatch") & (pl.col("round") >= 26)).height)
print("by claim bin:")
print(RO.group_by("rbin").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                              pl.col("mismatch").mean().round(4).alias("rate")).sort("rbin"))
aff_m = RO.filter(pl.col("mismatch")).select("match_key").unique().height
aff_13 = RO.filter(pl.col("mismatch") & (pl.col("round") == 13)).select("match_key").unique().height
print(f"affected matches (any mismatch)={aff_m} (claim 480); with r13 mismatch={aff_13} (claim 434)")

# ------------------------------------------------------------------ 2. window and timing (A2)
hdr("2. WINDOW AND TIMING CHECK FOR BOMB OUTCOME EVENTS (A2)")
w = RO.with_columns(
    (pl.col("out_tick") >= pl.col("start_tick")).alias("ge_start"),
    (pl.col("out_tick") <= pl.col("tick")).alias("le_end"),
    (pl.col("tick") - pl.col("out_tick")).alias("gap_end_minus_out"),
    (pl.col("out_tick") - pl.col("plant_tick")).alias("plant_to_out"))
w = w.with_columns((pl.col("ge_start") & pl.col("le_end")).alias("in_window"))
print("in_window by mismatch and rbin:")
print(w.group_by(["rbin", "mismatch", "in_window"]).len().sort(["rbin", "mismatch", "in_window"]))
print("gap (round_end.tick - outcome tick) quantiles by mismatch, rbin (ticks):")
print(w.group_by(["rbin", "mismatch"]).agg(
    pl.col("gap_end_minus_out").quantile(0.05).alias("q05"),
    pl.col("gap_end_minus_out").median().alias("med"),
    pl.col("gap_end_minus_out").quantile(0.95).alias("q95"),
    pl.col("plant_to_out").median().alias("med_plant_to_out"),
    pl.len().alias("n")).sort(["rbin", "mismatch"]))

# ------------------------------------------------------------------ reason codes
hdr("3. ENGINE WIN REASON VS BOMB OUTCOME (round_end reason vs bomb event)")
print(RO.group_by(["rbin", "mismatch", "otype", "win_reason_message"]).len()
      .sort(["rbin", "mismatch", "otype", "len"], descending=[False, False, False, True]).head(60))

# ------------------------------------------------------------------ 4. day alignment
hdr("4. DAY ALIGNMENT: FOLDER DAY (claim) VS HEADER match_date, round 13 bomb rounds")
r13 = RO.filter(pl.col("round") == 13)
print("round 13 bomb rounds by folder day:")
print(r13.group_by("fday").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                               pl.col("mismatch").mean().round(3).alias("rate")).sort("fday"))
print("round 13 bomb rounds by match_date (YYYY-MM-DD), mismatch and count:")
print(r13.group_by("mdate").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort("mdate"))
print("claim groups, folder day:", r13.group_by("dg_f").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort("dg_f").to_dicts())
print("claim groups, match_date day:", r13.group_by("dg_m").agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm")).sort("dg_m").to_dicts())

# ------------------------------------------------------------------ 5. map
hdr("5. MAP: round 13 bomb rounds, by map (folder-day groups)")
print(r13.group_by(["map_name", "dg_f"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"),
                                            pl.col("mismatch").mean().round(3).alias("rate")).sort(["dg_f", "map_name"]))
print("\nall bomb rounds (player_count==20) by map and folder-day group, mismatch count / n:")
print(RO.group_by(["map_name", "dg_f"]).agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
      .sort(["dg_f", "map_name"]))
print("\nmap of affected matches (any mismatch):")
print(RO.filter(pl.col("mismatch")).group_by("map_name").agg(pl.col("match_key").n_unique().alias("affected_matches"))
      .sort("affected_matches", descending=True))
print("matches per map in d12-25 (folder day):",
      MATCH.filter(pl.col("dg_f") == "d12-25").group_by("map_name").len().sort("len", descending=True).to_dicts())

# ------------------------------------------------------------------ 6. score side (A3)
hdr("6. SCORE-BASED SIDE WINNER (A3): score increment between round r start and round r+1 start")
S = (RST.sort(["match_key", "round", "tick"])
     .group_by(["match_key", "round"]).agg(
         pl.col("t_score").first().alias("t0"), pl.col("ct_score").first().alias("ct0"),
         pl.col("phase").first().alias("phase0")))
Snext = S.with_columns((pl.col("round") - 1).alias("round")).select(
    ["match_key", "round", pl.col("t0").alias("t_next"), pl.col("ct0").alias("ct_next")])
SC = (S.join(Snext, on=["match_key", "round"], how="inner")
      .with_columns(((pl.col("t_next") - pl.col("t0"))).alias("dT"),
                    ((pl.col("ct_next") - pl.col("ct0"))).alias("dCT"))
      .with_columns(
          pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
            .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
            .otherwise(pl.lit(None, dtype=pl.Int64)).alias("score_side")))
print("score-side coverage (round r with round r+1 present):", SC.height,
      " score_side non-null:", SC.filter(pl.col("score_side").is_not_null()).height)
print("increment pattern (dT, dCT) counts:")
print(SC.group_by(["dT", "dCT"]).len().sort("len", descending=True).head(10))

X = (RO_all.join(SC.select(["match_key", "round", "score_side", "phase0"]), on=["match_key", "round"], how="left")
     .join(MATCH.select(["match_key", "dg_f", "map_name"]), on="match_key", how="left", suffix="_m")
     .with_columns(
         (pl.col("score_side") == pl.col("winner_team_code")).alias("score_eq_winner"),
         (pl.col("score_side") == pl.col("bomb_side")).alias("score_eq_bomb")))
print("\nscore side vs bomb side on bomb rounds (player_count==20), by claim bin and folder day:")
Xc = X.filter((pl.col("player_count") == 20) & pl.col("score_side").is_not_null())
print(Xc.group_by(["dg_f", "rbin"]).agg(pl.len().alias("n"),
                                        (~pl.col("score_eq_bomb")).sum().alias("score_ne_bomb"),
                                        (~pl.col("score_eq_winner")).sum().alias("score_ne_winner"))
      .sort(["dg_f", "rbin"]))

# all rounds (bomb or not) : winner_team_code vs score side, by round number, d12-25 vs d01-07
hdr("7. WINNER_TEAM_CODE VS SCORE SIDE ON ALL ROUNDS (bomb or not), player_count == 20")
A_ = (RE.filter(pl.col("player_count") == 20)
      .join(SC.select(["match_key", "round", "score_side"]), on=["match_key", "round"], how="inner")
      .filter(pl.col("score_side").is_not_null())
      .join(MATCH.select(["match_key", "dg_f", "map_name"]), on="match_key", how="left")
      .with_columns((pl.col("score_side") != pl.col("winner_team_code")).alias("disagree")))
print("disagreement by folder day group and round bin:")
print(A_.with_columns(rbin_expr()).group_by(["dg_f", "rbin"]).agg(
    pl.len().alias("n"), pl.col("disagree").sum().alias("dis"),
    pl.col("disagree").mean().round(4).alias("rate")).sort(["dg_f", "rbin"]))
print("\ndisagreement by round number, d12-25 (all rounds, bomb or not):")
hi = A_.filter(pl.col("dg_f") == "d12-25").group_by("round").agg(
    pl.len().alias("n"), pl.col("disagree").sum().alias("dis"), pl.col("disagree").mean().round(3).alias("rate")).sort("round")
print(hi.filter(pl.col("round") <= 30))
print("\ndisagreement by round number, d01-07:")
lo = A_.filter(pl.col("dg_f") == "d01-07").group_by("round").agg(
    pl.len().alias("n"), pl.col("disagree").sum().alias("dis")).sort("round")
print(lo.filter(pl.col("round") <= 30))

# whether non-bomb round-13 disagreements are concentrated in some maps
print("\nround 13 disagreement by map, d12-25 (all rounds):")
print(A_.filter((pl.col("dg_f") == "d12-25") & (pl.col("round") == 13)).group_by("map_name").agg(
    pl.len().alias("n"), pl.col("disagree").sum().alias("dis"), pl.col("disagree").mean().round(3).alias("rate")).sort("map_name"))

# ------------------------------------------------------------------ 8. shift test (A8)
hdr("8. INDEX-SHIFT TEST: does round_end winner at r match score side at r-1 or r+1 in mismatch matches?")
SS = SC.select(["match_key", "round", "score_side"])
RE_w = RE.filter(pl.col("player_count") == 20).select(["match_key", "round", "winner_team_code"])
shift = (RE_w.join(SS.with_columns((pl.col("round") + 1).alias("round")).rename({"score_side": "ss_prev"}),
                   on=["match_key", "round"], how="inner")  # ss_prev = score side of round r-1
         .join(SS.with_columns((pl.col("round") - 1).alias("round")).rename({"score_side": "ss_next"}),
               on=["match_key", "round"], how="inner")       # ss_next = score side of round r+1
         .join(SS.rename({"score_side": "ss_same"}), on=["match_key", "round"], how="inner")
         .join(RO_all.select(["match_key", "round", "bomb_side", "mismatch"]), on=["match_key", "round"], how="left"))
print("round 13 rows with winner, score(r-1), score(r), score(r+1) and bomb outcome (d12-25 folder day):")
s13 = shift.filter((pl.col("round") == 13)).join(MATCH.select(["match_key", "dg_f"]), on="match_key")
s13 = s13.filter(pl.col("dg_f") == "d12-25")
print(s13.group_by(["ss_same", "ss_prev", "ss_next", "winner_team_code"]).len().sort("len", descending=True).head(12))
print("rate winner==score(r):", round(float((s13["winner_team_code"] == s13["ss_same"]).mean()), 4),
      " winner==score(r-1):", round(float((s13["winner_team_code"] == s13["ss_prev"]).mean()), 4),
      " winner==score(r+1):", round(float((s13["winner_team_code"] == s13["ss_next"]).mean()), 4), " n=", s13.height)

# ------------------------------------------------------------------ 9. spot check of score fields at halftime
hdr("9. SPOT CHECK: score fields and phase around halftime (round 11-14) in one affected and one clean match")
aff_key = (RO.filter(pl.col("mismatch") & (pl.col("round") == 13)).select("match_key").unique().sort("match_key").head(1)["match_key"].to_list())
clean_key = (RO.filter(~pl.col("mismatch") & (pl.col("round") == 13) & (pl.col("dg_f") == "d12-25"))
             .select("match_key").unique().sort("match_key").head(1)["match_key"].to_list())
for k in aff_key + clean_key:
    tag = "AFFECTED" if k in aff_key else "CLEAN"
    print(f"\n[{tag}] {k}")
    rows = (RE.filter((pl.col("match_key") == k) & pl.col("round").is_between(11, 14))
            .select(["round", "winner_team_code", "win_reason_message", "player_count"]).sort("round"))
    print(rows)
    ss = S.filter((pl.col("match_key") == k) & pl.col("round").is_between(11, 14)).sort("round")
    print(ss)
    bo = RO_all.filter((pl.col("match_key") == k) & pl.col("round").is_between(11, 14)).select(
        ["round", "otype", "bomb_side", "mismatch", "out_tick", "tick"]).sort("round")
    print(bo)

print("\nDONE")
