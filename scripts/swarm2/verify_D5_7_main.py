"""D5-7 independent verifier: recompute the regime marker, round-1 round_end presence, the
round-13 bomb-side mismatch, and the header build_num from the raw 2024 tome on G: (read only).

Claim under test (D5-7): two recording regimes keyed by a per-match marker
  (round_announce_match_start count >= 1 = regime A, 2,478 matches incl. 434 affected),
  round_end label 1 absent in regime B (1,957 matches), round-13 side mismatch only in regime A,
  proposed fix: derive round-1 winner from freeze-end scores at round 2 in regime B.

Checks:
  1. Marker counts, regime x build_num (header), regime x day, label-1 round_end presence.
  2. Round-13 bomb-side mismatch (same definition as scripts/swarm/e2_side_check_r13.py) by regime, day.
  3. Alternative: is the mismatch a label shift (bomb tick in neighbour round window) or a side error?
  4. Restart example csds/2024/02/12/m_partial... (two round_announce_match_start, two round-1 ends).
  5. Freeze-end score sanity for the proposed round-1 fix (fe(1) total 0, fe(2) total 1).
ASSUMPTION: a bomb outcome (defused -> CT wins, exploded -> T wins) is ground truth for the round winner.
ASSUMPTION: round window = [first round_freeze_end tick, first basic round_end tick] of the same label.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_7_main.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(70)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
EX = "csds/2024/02/12/m_56c0f3b373d8adecd20c6222/csds"


def ch(name, cols):
    fs = sorted(glob.glob(str(BASE / f"basic_{name}{SUF}" / "dataframe_*")))
    if not fs:
        raise FileNotFoundError(name)
    lf = pl.concat([pl.scan_parquet(f).select(cols) for f in fs], how="vertical_relaxed")
    if "round" in cols:
        lf = lf.with_columns(pl.col("round").cast(pl.Int64))
    return lf


def cnt(et):
    return (pl.col("event_type") == et).sum()


def day_of(col):
    return pl.col(col).str.extract(r"csds/2024/02/(\d\d)/", 1)


# ---------------- header ----------------
hf = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
H = pl.concat([pl.scan_parquet(f).select(["key", "match_id", "map_name", "match_date", "build_num",
                                          "rushb_version", "ppp_version"]) for f in hf],
              how="vertical_relaxed").collect()
print(f"header rows {H.height}, unique keys {H['key'].n_unique()}")
H = H.rename({"key": "match_key"}).with_columns(day=day_of("match_key"))

# ---------------- round_state markers ----------------
RS_TYPES = ["round_announce_match_start", "cs_win_panel_round", "round_start", "round_end",
            "round_freeze_end", "round_announce_last_round_half", "round_announce_match_point",
            "cs_win_panel_match"]
RS = (ch("round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"])
      .filter(pl.col("event_type").is_in(RS_TYPES)).collect())
MK = RS.group_by("match_key").agg(
    cnt("round_announce_match_start").alias("n_ann"),
    cnt("cs_win_panel_round").alias("n_cwp"),
    cnt("round_start").alias("n_rs_start"),
    cnt("round_end").alias("n_rs_end"),
    cnt("round_freeze_end").alias("n_fe"),
    cnt("cs_win_panel_match").alias("n_cwp_match"),
    cnt("round_announce_last_round_half").alias("n_lrh"),
)
RS1 = RS.filter(pl.col("round") == 1).group_by("match_key").agg(
    cnt("round_end").alias("rs_end_r1"),
    cnt("cs_win_panel_round").alias("cwp_r1"),
    cnt("round_freeze_end").alias("fe_r1"),
    cnt("round_announce_match_start").alias("ann_r1"),
)
print(f"\nround_state matches with any marker row: {MK.height}")
print("sum of round_announce_match_start rows:", int(MK["n_ann"].sum()),
      "| matches with >=1:", int((MK["n_ann"] >= 1).sum()),
      "| matches with >=2:", int((MK["n_ann"] >= 2).sum()))

# ---------------- basic round_end ----------------
RE = ch("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
print(f"\nbasic round_end rows {RE.height}; player_count values:")
print(RE.group_by("player_count").len().sort("player_count"))
dup = RE.group_by(["match_key", "round", "player_count"]).len().filter(pl.col("len") > 1)
print("dup (match,round,player_count) groups in round_end:", dup.height)
dup2 = RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print("dup (match,round) groups in round_end (any player_count):", dup2.height)
RE_PM = RE.group_by("match_key").agg(
    pl.len().alias("be_rows"),
    (pl.col("round") == 1).sum().alias("be_r1_rows"),
    pl.col("round").min().alias("be_min_label"),
    pl.col("round").max().alias("be_max_label"),
)

# ---------------- match table ----------------
M = (H.join(MK, on="match_key", how="left").join(RS1, on="match_key", how="left")
     .join(RE_PM, on="match_key", how="left"))
print("\nheader matches without any round_state row:", M.filter(pl.col("n_fe").is_null()).height)
M = M.with_columns([pl.col(c).fill_null(0) for c in
                    ["n_ann", "n_cwp", "n_rs_start", "n_rs_end", "n_fe", "n_cwp_match", "n_lrh",
                     "rs_end_r1", "cwp_r1", "fe_r1", "ann_r1", "be_rows", "be_r1_rows"]])
M = M.with_columns(
    regime=pl.when(pl.col("n_ann") > 0).then(pl.lit("A")).otherwise(pl.lit("B")),
    grp=pl.when(pl.col("day").is_in(LATE)).then(pl.lit("late"))
    .when(pl.col("day").is_in(EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")),
    be_r1=pl.col("be_r1_rows") > 0,
)
print("\n== 1a. marker crosstab (n_ann>0 x n_cwp>0 x n_rs_end>0) ==")
print(M.group_by(["regime", pl.col("n_cwp") > 0, pl.col("n_rs_end") > 0, pl.col("n_rs_start") > 0])
      .agg(pl.len().alias("matches")).sort("regime"))
print("\n== 1b. regime x header build_num ==")
print(M.group_by(["regime", "build_num"]).agg(pl.len().alias("matches")).sort(["regime", "build_num"]))
print("\n== 1c. regime x label-1 round_end presence (basic round_end channel) ==")
print(M.group_by(["regime", "be_r1"]).agg(pl.len().alias("matches")).sort(["regime", "be_r1"]))
print("\n== 1d. regime x round_state round_end label 1 and cs_win_panel_round label 1 ==")
print(M.group_by(["regime", pl.col("rs_end_r1") > 0, pl.col("cwp_r1") > 0, pl.col("fe_r1") > 0])
      .agg(pl.len().alias("matches")).sort(["regime"]))
print("\n== 1e. min basic round_end label by regime (is label 1 missing or is numbering shifted?) ==")
print(M.filter(pl.col("be_rows") > 0).group_by(["regime", "be_min_label"]).agg(pl.len().alias("matches"))
      .sort(["regime", "be_min_label"]))
print("\n== 1f. label-1 round_end presence and regime by day ==")
print(M.group_by(["day", "regime"]).agg(pl.len().alias("matches"), pl.col("be_r1").sum().alias("be_r1_present"))
      .sort(["day", "regime"]))
print("\n== 1g. regime x build_num x day group ==")
print(M.group_by(["grp", "regime", "build_num"]).agg(pl.len().alias("matches")).sort(["grp", "regime", "build_num"]))

# ---------------- bomb outcomes ----------------
BS = (ch("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
      .filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).collect())
OUT = BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
OUTR = OUT.group_by(["match_key", "round"]).agg(
    pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("out_tick"),
    pl.len().alias("n_out_rows"))
print("\nbomb outcome (match,round) groups with both defused and exploded:",
      OUTR.filter(pl.col("evs").list.len() > 1).height)
OUTR = OUTR.with_columns(bomb_side=pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2))

RE20 = RE.filter(pl.col("player_count") == 20)
J20 = RE20.join(OUTR, on=["match_key", "round"], how="inner").with_columns(
    mismatch=(pl.col("bomb_side") != pl.col("winner_team_code")))
JALL = RE.join(OUTR, on=["match_key", "round"], how="inner").with_columns(
    mismatch=(pl.col("bomb_side") != pl.col("winner_team_code")))
print(f"control (e2 definition, player_count==20): rounds with outcome {J20.height}, mismatches {int(J20['mismatch'].sum())}"
      f" (e2 log: 15436 / 499)")
print(f"same with all player_count rows: rows {JALL.height}, mismatches {int(JALL['mismatch'].sum())}")
J = J20.join(M.select(["match_key", "regime", "grp", "day", "build_num", "n_ann"]), on="match_key", how="left")
print("\nJ rows with no match-table row:", J.filter(pl.col("regime").is_null()).height)

print("\n== 2a. mismatches by round label bin x regime (player_count==20) ==")
J = J.with_columns(rbin=pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                   .when(pl.col("round") == 25).then(pl.lit("r25")).otherwise(pl.lit("other")))
print(J.group_by(["rbin", "regime"]).agg(pl.len().alias("rounds"), pl.col("mismatch").sum().alias("mm"),
                                         pl.col("match_key").n_unique().alias("matches"))
      .sort(["rbin", "regime"]))
print("\n== 2b. round 13 by regime x grp (rows with bomb outcome) ==")
r13 = J.filter(pl.col("round") == 13)
print(r13.group_by(["grp", "regime"]).agg(pl.len().alias("rounds"), pl.col("mismatch").sum().alias("mm"),
                                          pl.col("match_key").n_unique().alias("matches")).sort(["grp", "regime"]))
print("distinct matches with an r13 mismatch:", r13.filter(pl.col("mismatch"))["match_key"].n_unique())
print("\n== 2c. round 13 by day (late window) x regime ==")
print(r13.group_by(["day", "regime"]).agg(pl.len().alias("rounds"), pl.col("mismatch").sum().alias("mm"))
      .sort(["day", "regime"]))
print("\n== 2d. all non-13 mismatches: label, regime, grp, build ==")
oth = J.filter(pl.col("mismatch") & (pl.col("round") != 13))
print(oth.group_by(["round", "regime", "grp", "build_num"]).agg(pl.len().alias("n")).sort(["round", "regime"]))
print("non-13 mismatch matches by regime:", oth.group_by("regime").agg(pl.col("match_key").n_unique().alias("matches")))
print("\n== 2e. round 25 mismatches by regime ==")
print(J.filter(pl.col("round") == 25).group_by(["regime", "mismatch"]).len().sort(["regime", "mismatch"]))
print("\n== 2f. build_num x mismatch (r13 rows) ==")
print(r13.group_by(["build_num", "mismatch"]).len().sort(["build_num", "mismatch"]))
print("\n== 2g. late r13 bomb-outcome rows, regime A non-mismatch rows (the ones that do not flip) ==")
print("regime A late r13 non-mismatch rows:", r13.filter((pl.col("regime") == "A") & (pl.col("grp") == "late")
                                                       & (~pl.col("mismatch"))).height)
print("regime B late r13 rows with outcome:", r13.filter((pl.col("regime") == "B") & (pl.col("grp") == "late")).height)

# ---------------- tick-window label attribution (alternative: label shift) ----------------
FE = RS.filter(pl.col("event_type") == "round_freeze_end").group_by(["match_key", "round"]).agg(
    pl.col("tick").min().alias("fe_tick"))
EN = RE.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("end_tick"))
W = FE.join(EN, on=["match_key", "round"], how="full", coalesce=True)
W_own = W
W_prev = W.with_columns(round=pl.col("round") + 1, fe_p=pl.col("fe_tick"), end_p=pl.col("end_tick")).select(
    ["match_key", "round", "fe_p", "end_p"])
W_next = W.with_columns(round=pl.col("round") - 1, fe_n=pl.col("fe_tick"), end_n=pl.col("end_tick")).select(
    ["match_key", "round", "fe_n", "end_n"])
OUT2 = (OUT.select(["match_key", "round", "tick", "event_type"])
        .join(W_own, on=["match_key", "round"], how="left")
        .join(W_prev, on=["match_key", "round"], how="left")
        .join(W_next, on=["match_key", "round"], how="left")
        .join(J20.select(["match_key", "round", "mismatch"]), on=["match_key", "round"], how="left")
        .join(M.select(["match_key", "regime"]), on="match_key", how="left"))
OUT2 = OUT2.with_columns(
    in_own=(pl.col("tick") >= pl.col("fe_tick")) & (pl.col("tick") <= pl.col("end_tick")),
    in_prev=(pl.col("tick") >= pl.col("fe_p")) & (pl.col("tick") <= pl.col("end_p")),
    in_next=(pl.col("tick") >= pl.col("fe_n")) & (pl.col("tick") <= pl.col("end_n")),
).with_columns(pl.col(["in_own", "in_prev", "in_next"]).fill_null(False))
print("\n== 3. outcome-event tick vs own/prev/next round window, rounds with a J20 row ==")
print(OUT2.filter(pl.col("mismatch").is_not_null()).group_by(["regime", "mismatch", "in_own", "in_prev", "in_next"])
      .len().sort(["regime", "mismatch", "in_own", "in_prev", "in_next"]))

# ---------------- restart example ----------------
print(f"\n== 4. restart example {EX} ==")
print(RS.filter((pl.col("match_key") == EX) & (pl.col("round") <= 2))
      .sort(["tick"]).select(["round", "event_type", "tick", "t_score", "ct_score"]))
print(RE.filter((pl.col("match_key") == EX) & (pl.col("round") <= 2)).sort(["tick"]))
print("round_announce_match_start count in this match:", M.filter(pl.col("match_key") == EX).select("n_ann").item())

# ---------------- fix sanity: freeze-end totals at rounds 1 and 2 ----------------
FE_TOT = (RS.filter((pl.col("event_type") == "round_freeze_end") & pl.col("round").is_in([1, 2]))
          .sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
              (pl.col("t_score").first() + pl.col("ct_score").first()).alias("tot"),
              pl.col("t_score").first().alias("t"), pl.col("ct_score").first().alias("ct")))
FT = FE_TOT.pivot(on="round", index="match_key", values="tot", aggregate_function="first").rename(
    {"1": "tot_fe1", "2": "tot_fe2"})
MF = M.select(["match_key", "regime"]).join(FT, on="match_key", how="left")
print("\n== 5. freeze-end total at round 1 (should be 0) and round 2 (should be 1) by regime ==")
print(MF.group_by(["regime", "tot_fe1", "tot_fe2"], maintain_order=False).agg(pl.len().alias("matches"))
      .sort(["regime", "tot_fe1", "tot_fe2"]))

# ---------------- 6. score-based round winner from freeze-end totals ----------------
# ASSUMPTION: no team-side swap between label L and L+1 (boundaries 8, 12, 24, 27 excluded).
# Winner code 2 (T) if only t_score rose by 1 between fe(L) and fe(L+1); 3 (CT) if only ct_score rose.
FE_ts = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
         .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("t"),
                                               pl.col("ct_score").first().alias("ct")))
NXT = FE_ts.with_columns(round=pl.col("round") - 1).select(
    ["match_key", "round", pl.col("t").alias("t_next"), pl.col("ct").alias("ct_next")])
SC = FE_ts.join(NXT, on=["match_key", "round"], how="inner").with_columns(
    dt=pl.col("t_next") - pl.col("t"), dct=pl.col("ct_next") - pl.col("ct"))
SC = SC.with_columns(sw_winner=pl.when((pl.col("dt") == 1) & (pl.col("dct") == 0)).then(2)
                     .when((pl.col("dct") == 1) & (pl.col("dt") == 0)).then(3).otherwise(None))
SC = SC.filter(~pl.col("round").is_in([8, 12, 24, 27]))
WN = (RE.filter(pl.col("player_count") == 20).sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(pl.col("winner_team_code").first().alias("wc")))
BSIDE = OUTR.select(["match_key", "round", "bomb_side"])
SCJ = (SC.join(WN, on=["match_key", "round"], how="left").join(BSIDE, on=["match_key", "round"], how="left")
       .join(M.select(["match_key", "regime", "grp", "build_num"]), on="match_key", how="left"))
print("\n== 6a. score-based winner vs winner_team_code vs bomb side, by regime (labels excl. 8,12,24,27) ==")
print(SCJ.group_by("regime").agg(
    pl.len().alias("rounds_score_known"),
    (pl.col("wc").is_not_null() & pl.col("sw_winner").is_not_null()).sum().alias("with_wc"),
    (pl.col("sw_winner") == pl.col("wc")).sum().alias("score_eq_wc"),
    (pl.col("sw_winner") != pl.col("wc")).sum().alias("score_ne_wc"),
    (pl.col("bomb_side").is_not_null()).sum().alias("with_bomb"),
    (pl.col("sw_winner") == pl.col("bomb_side")).sum().alias("score_eq_bomb"),
    (pl.col("sw_winner") != pl.col("bomb_side")).sum().alias("score_ne_bomb"),
    (pl.col("wc") != pl.col("bomb_side")).sum().alias("wc_ne_bomb")).sort("regime"))
print("\n== 6b. score winner != winner_team_code, by regime and label (all labels with a known score winner) ==")
print(SCJ.filter(pl.col("wc").is_not_null()).group_by(["regime", "round"]).agg(
    pl.len().alias("n"), (pl.col("sw_winner") != pl.col("wc")).sum().alias("score_ne_wc"))
      .sort(["regime", "round"]).filter(pl.col("score_ne_wc") > 0))
print("\n== 6c. round 13 and 25: score winner vs bomb side for bomb rounds, regime x mismatch flag ==")
print(SCJ.filter(pl.col("round").is_in([13, 25]) & pl.col("bomb_side").is_not_null()).group_by(
    ["round", "regime", "sw_winner", "bomb_side", "wc"]).len().sort(["round", "regime", "sw_winner", "bomb_side", "wc"]))

# ---------------- 7. full round-1 footprint and label-0 check by regime ----------------
ALL_R = ch("round_state", ["match_key", "round", "event_type"]).filter(pl.col("round") <= 1).collect()
fp = (ALL_R.group_by(["match_key", "event_type", "round"]).agg(pl.len().alias("n"))
      .join(M.select(["match_key", "regime"]), on="match_key", how="left"))
print("\n== 7a. round_state rows with round<=1, by regime, event_type and label (matches with >=1 row) ==")
print(fp.group_by(["regime", "round", "event_type"]).agg(pl.col("match_key").n_unique().alias("matches"))
      .sort(["regime", "round", "event_type"]))
print("\nround_state min label per regime:")
print(ALL_R.join(M.select(["match_key", "regime"]), on="match_key", how="left").group_by(["regime"]).agg(
    pl.col("round").min().alias("min_label"), pl.col("round").max().alias("max_label_le1")))
RSM = ch("round_state", ["match_key", "round"]).group_by("match_key").agg(pl.col("round").min().alias("rs_min"),
                                                                        pl.col("round").max().alias("rs_max")).collect()
print(RSM.join(M.select(["match_key", "regime"]), on="match_key", how="left").group_by("regime").agg(
    pl.col("rs_min").value_counts().alias("rs_min_dist")).sort("regime"))
BS1 = (BS.filter(pl.col("round") <= 1).join(M.select(["match_key", "regime"]), on="match_key", how="left")
       .group_by(["regime", "round", "event_type"]).agg(pl.len().alias("rows")).sort(["regime", "round", "event_type"]))
print("\nbomb_state rows with round<=1 by regime:")
print(BS1)

# ---------------- 8. does round-1 derived from fe(2) (proposed fix) agree with recorded round-1 winners? ----------------
W1 = WN.filter(pl.col("round") == 1).select(["match_key", pl.col("wc").alias("wc1")])
F2 = FE_ts.filter(pl.col("round") == 2).select(["match_key", pl.col("t").alias("t2"), pl.col("ct").alias("ct2")])
F2 = F2.with_columns(fix_w1=pl.when((pl.col("t2") == 1) & (pl.col("ct2") == 0)).then(2)
                     .when((pl.col("ct2") == 1) & (pl.col("t2") == 0)).then(3).otherwise(None))
V = M.select(["match_key", "regime"]).join(W1, on="match_key", how="left").join(F2, on="match_key", how="left")
print("\n== 8. proposed round-1 fix (fe(2) totals) vs recorded round-1 winner (basic round_end label 1) ==")
print(V.group_by(["regime"]).agg(
    pl.len().alias("matches"),
    pl.col("wc1").is_not_null().sum().alias("recorded_r1"),
    pl.col("fix_w1").is_not_null().sum().alias("fix_derivable"),
    (pl.col("fix_w1") == pl.col("wc1")).sum().alias("fix_eq_recorded"),
    (pl.col("fix_w1") != pl.col("wc1")).sum().alias("fix_ne_recorded")).sort("regime"))
print("fix_w1 distribution, regime B (where no recorded round 1 exists):")
print(V.filter(pl.col("regime") == "B").group_by("fix_w1").len().sort("fix_w1"))

# ---------------- 9. do the post-swap disagreements line up with detected side swaps? ----------------
# Swap at boundary b detected from fe(b), fe(b+1) and winner_team_code at label b (labels 8,12,24,27 are not flipped in
# regime A, see 6b). Team X on T before the boundary; a swap means X is on CT afterwards.
#   no-swap consistent: t(b+1) = t(b) + [wc==2], ct(b+1) = ct(b) + [wc==3]
#   swap consistent:    t(b+1) = ct(b) + [wc==3], ct(b+1) = t(b) + [wc==2]
flags = []
for b in [8, 12, 24, 27]:
    fb = FE_ts.filter(pl.col("round") == b).select(["match_key", pl.col("t").alias("tb"), pl.col("ct").alias("cb")])
    fb1 = FE_ts.filter(pl.col("round") == b + 1).select(["match_key", pl.col("t").alias("tb1"), pl.col("ct").alias("cb1")])
    wb = WN.filter(pl.col("round") == b).select(["match_key", pl.col("wc").alias("wcb")])
    X = fb.join(fb1, on="match_key", how="inner").join(wb, on="match_key", how="inner")
    a = pl.when(pl.col("wcb") == 2).then(1).otherwise(0)
    c = pl.when(pl.col("wcb") == 3).then(1).otherwise(0)
    X = X.with_columns(
        ns_ok=(pl.col("tb1") == pl.col("tb") + a) & (pl.col("cb1") == pl.col("cb") + c),
        sw_ok=(pl.col("tb1") == pl.col("cb") + c) & (pl.col("cb1") == pl.col("tb") + a))
    X = X.with_columns(swapped=pl.col("sw_ok") & ~pl.col("ns_ok"), ambiguous=pl.col("sw_ok") & pl.col("ns_ok"))
    flags.append(X.select(["match_key", pl.col("swapped").alias(f"sw{b}"), pl.col("ambiguous").alias(f"amb{b}")]))
SWF = M.select(["match_key", "regime"])
for f in flags:
    SWF = SWF.join(f, on="match_key", how="left")
SWF = SWF.with_columns([pl.col(c).fill_null(False) for c in SWF.columns if c.startswith(("sw", "amb"))])
print("\n== 9a. swap flags by regime (boundaries 8,12,24,27; swapped = swap-consistent and not no-swap-consistent) ==")
print(SWF.group_by("regime").agg(
    pl.len().alias("matches"), pl.col("sw8").sum().alias("swap8"), pl.col("sw12").sum().alias("swap12"),
    pl.col("sw24").sum().alias("swap24"), pl.col("sw27").sum().alias("swap27"),
    pl.col("amb8").sum().alias("amb8"), pl.col("amb12").sum().alias("amb12"),
    pl.col("amb24").sum().alias("amb24"), pl.col("amb27").sum().alias("amb27")).sort("regime"))
SCS = SCJ.join(SWF.select(["match_key", "sw8", "sw12", "sw24", "sw27"]), on="match_key", how="left")
print("\n== 9b. label L vs winner_team_code/score disagreement, split by the swap flag at the boundary just before L ==")
for L, col in [(9, "sw8"), (13, "sw12"), (25, "sw24"), (28, "sw27")]:
    print(f"-- label {L} (boundary flag {col}):")
    print(SCS.filter((pl.col("round") == L) & pl.col("wc").is_not_null()).group_by(["regime", col]).agg(
        pl.len().alias("rounds"), (pl.col("sw_winner") != pl.col("wc")).sum().alias("score_ne_wc"),
        (pl.col("sw_winner") == pl.col("bomb_side")).sum().alias("score_eq_bomb"),
        pl.col("bomb_side").is_not_null().sum().alias("with_bomb")).sort(["regime", col]))

# ---------------- 9c. split by ambiguity: a boundary is ambiguous when swap and no-swap are both consistent ----------------
SCS2 = SCJ.join(SWF.select(["match_key"] + [c for c in SWF.columns if c.startswith(("sw", "amb"))]),
                on="match_key", how="left")
print("\n== 9c. label L disagreements by swap flag and ambiguity at its boundary ==")
for L, b in [(9, 8), (13, 12), (25, 24), (28, 27)]:
    keys = ["regime", f"sw{b}", f"amb{b}"] if b == 8 else ["regime", "sw8", f"sw{b}", f"amb{b}"]
    print(f"-- label {L}, boundary {b}:")
    print(SCS2.filter((pl.col("round") == L) & pl.col("wc").is_not_null()).group_by(keys).agg(
        pl.len().alias("rounds"), (pl.col("sw_winner") != pl.col("wc")).sum().alias("score_ne_wc"),
        (pl.col("sw_winner") == pl.col("bomb_side")).sum().alias("score_eq_bomb"),
        pl.col("bomb_side").is_not_null().sum().alias("with_bomb")).sort(keys))
