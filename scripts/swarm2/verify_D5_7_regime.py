"""D5-7 independent verifier (2024 tome on G: only; no scratchpad inputs).

Claim under test (D5-7): two recording regimes keyed by a per-match marker
(round_announce_match_start count >= 1 = regime A; absent = regime B). Round-1 end
missing in regime B. Proposed fix: derive the round-1 winner in regime B from the
freeze-end score at round 2.

Reads only tomes/2024-02-g/tome/csds via lazy scans with column projection.
Writes only logs/swarm2/verify_D5_7.log (plus stdout).
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_7_regime.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
LOG = ROOT / "logs" / "swarm2" / "verify_D5_7.log"
LOG.parent.mkdir(parents=True, exist_ok=True)
FH = open(LOG, "a", encoding="utf-8", newline="\n")  # append: the log is shared with another run

pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(60)
pl.Config.set_tbl_width_chars(220)

LATE = [f"{d:02d}" for d in range(12, 26)]
EARLY = [f"{d:02d}" for d in range(1, 8)]
RM = "csds/2024/02/12/2f5bcb89-9ef8-4788-9f3a-2fa3d633ea4d/csds"


def out(*parts):
    s = " ".join(str(p) for p in parts)
    print(s, flush=True)
    FH.write(s + "\n")
    FH.flush()


def show(df):
    out(str(df))


def chfiles(ch):
    fs = sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))
    if not fs:
        raise SystemExit(f"no dataframe files for channel {ch}")
    return fs


def scan(ch):
    return pl.concat([pl.scan_parquet(f) for f in chfiles(ch)], how="vertical_relaxed")


out("# verify_D5_7 log (independent recomputation from tomes/2024-02-g)")
out("ASSUMPTION A1: regime A = round_announce_match_start count (round_state, all labels) >= 1; else regime B.")
out("ASSUMPTION A2: round winner = first round_end row per (match, round) by tick (as scripts/swarm2/D5_02_build.py).")
out("ASSUMPTION A3: bomb outcome = bomb_state channel, defused -> CT (3), exploded -> T (2) (as scripts/swarm/e2_side_check_r13.py).")
out("ASSUMPTION A4: score before a round = t_score/ct_score on the first round_freeze_end row of that round (round_state).")
out("ASSUMPTION A5: side-based score fields: after a T win, t_score increases by 1 (side semantics); team semantics would flip in H2.")
out("ASSUMPTION A6: halves H1 = rounds 1-12, H2 = 13-24, OT = 25+ (MR12). Transition rounds 12 and 24 are reported separately.")

# ---------------- 0. inventory ----------------
out("\n== 0. inventory ==")
hfiles = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
out("header dataframe files:", len(hfiles))
H = (pl.concat([pl.read_parquet(f, columns=["key", "match_id", "map_name", "match_date"]) for f in hfiles],
               how="vertical_relaxed")
     .rename({"key": "match_key"})
     .with_columns(day=pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1)))
out("header rows:", H.height, " unique match_key:", H["match_key"].n_unique(),
    " unique match_id:", H["match_id"].n_unique(), " null day:", H.filter(pl.col("day").is_null()).height)
show(H.group_by("day").agg(pl.len().alias("matches")).sort("day"))
for ch in ["round_state", "round_end", "round_start", "bomb_state"]:
    fs = chfiles(ch)
    n = scan(ch).select(pl.len()).collect().item()
    out(f"{ch}: files={len(fs)} rows={n} cols={list(pl.read_parquet_schema(fs[0]).keys())}")

# ---------------- 1. per-match markers ----------------
out("\n== 1. per-match markers from round_state and round_end ==")
MARK = ["round_announce_match_start", "cs_win_panel_round", "round_start", "round_end",
        "round_freeze_end", "round_officially_ended", "round_prestart", "round_poststart",
        "bomb_planted", "bomb_defused", "bomb_exploded", "begin_new_match", "cs_win_panel_match"]
RS = (scan("round_state").select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type").is_in(MARK)).collect())
out("round_state marker rows:", RS.height, " matches with any marker:", RS["match_key"].n_unique())
out("null round label among marker rows:")
show(RS.filter(pl.col("round").is_null()).group_by("event_type").agg(pl.len().alias("n")))

cnt = (RS.group_by(["match_key", "event_type"]).agg(pl.len().alias("n"))
       .pivot(on="event_type", index="match_key", values="n", aggregate_function="sum")
       .fill_null(0))
cnt = cnt.rename({c: f"n_{c}" for c in cnt.columns if c != "match_key"})

RE_raw = scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "player_count"]).collect()
out("round_end rows:", RE_raw.height, " player_count top values:")
show(RE_raw.group_by("player_count").agg(pl.len().alias("n")).sort("n", descending=True).head(8))
RE1 = (RE_raw.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
    pl.col("winner_team_code").first().alias("winner"),
    pl.col("player_count").first().alias("pc"),
    pl.len().alias("n_rows")))
re_m = RE_raw.group_by("match_key").agg(
    (pl.col("round") == 1).sum().alias("r1_rows"),
    ((pl.col("round") == 1) & (pl.col("player_count") == 20)).sum().alias("r1_p20"),
    pl.col("round").min().alias("min_end_round"),
    pl.col("round").max().alias("max_end_round"),
    pl.len().alias("end_rows_total"))
re_pairs = RE1.group_by("match_key").agg(
    pl.len().alias("end_pairs"),
    (pl.col("n_rows") > 1).sum().alias("dup_pairs"))

MK = (H.join(cnt, on="match_key", how="left").join(re_m, on="match_key", how="left")
      .join(re_pairs, on="match_key", how="left"))
NUMC = [c for c in MK.columns if c.startswith("n_") or c in (
    "r1_rows", "r1_p20", "min_end_round", "max_end_round", "end_rows_total", "end_pairs", "dup_pairs")]
MK = MK.with_columns([pl.col(c).fill_null(0) for c in NUMC])
for need in ["n_round_announce_match_start", "n_cs_win_panel_round", "n_round_start", "n_round_freeze_end"]:
    if need not in MK.columns:
        MK = MK.with_columns(pl.lit(0).alias(need))
MK = MK.with_columns(
    regime=pl.when(pl.col("n_round_announce_match_start") > 0).then(pl.lit("A")).otherwise(pl.lit("B")),
    ras_pos=pl.col("n_round_announce_match_start") > 0,
    cwp_pos=pl.col("n_cs_win_panel_round") > 0,
    rs_start_pos=pl.col("n_round_start") > 0,
    r1_pos=pl.col("r1_rows") > 0)

# ---------------- 2. crosstab and claim numbers ----------------
out("\n== 2. marker crosstab over all header matches ==")
show(MK.group_by(["regime", "ras_pos", "cwp_pos", "rs_start_pos", "r1_pos"]).agg(pl.len().alias("matches"))
     .sort(["regime", "ras_pos", "cwp_pos", "rs_start_pos", "r1_pos"]))
nA = MK.filter(pl.col("regime") == "A").height
nB = MK.filter(pl.col("regime") == "B").height
out("regime A (ras>0) matches:", nA, " regime B matches:", nB, " total:", MK.height)
out("disagreements regime A vs r1_pos:", MK.filter((pl.col("regime") == "A") != pl.col("r1_pos")).height)
out("disagreements regime A vs cs_win_panel_round>0 (A should never have it):",
    MK.filter((pl.col("regime") == "A") == pl.col("cwp_pos")).height)
out("disagreements regime B vs cs_win_panel_round>0 (B should always have it):",
    MK.filter((pl.col("regime") == "B") != pl.col("cwp_pos")).height)
out("round_announce_match_start count per match (value -> matches):")
show(MK.group_by("n_round_announce_match_start").agg(pl.len().alias("matches")).sort("n_round_announce_match_start"))
out("round_end label-1 rows per match (value -> matches):")
show(MK.group_by("r1_rows").agg(pl.len().alias("matches")).sort("r1_rows"))

out("\n== 2b. per-day table ==")
show(MK.group_by("day").agg(
    pl.len().alias("matches"),
    (pl.col("regime") == "A").sum().alias("regimeA"),
    pl.col("r1_pos").sum().alias("r1_end_present"),
    pl.col("cwp_pos").sum().alias("cs_win_panel_present"),
    pl.col("rs_start_pos").sum().alias("rs_round_start_present")).sort("day"))

out("\n== 2c. claim numbers ==")
ea = MK.filter(pl.col("day").is_in(EARLY))
out("days 01-07: matches", ea.height, " r1_end present", ea.filter(pl.col("r1_pos")).height,
    " regime A", ea.filter(pl.col("regime") == "A").height)
d12 = MK.filter(pl.col("day") == "12")
out("day 12: matches", d12.height, " r1_end present", d12.filter(pl.col("r1_pos")).height)
d25 = MK.filter(pl.col("day") == "25")
out("day 25: matches", d25.height, " r1_end present", d25.filter(pl.col("r1_pos")).height)
late = MK.filter(pl.col("day").is_in(LATE))
out("late days 12-25: matches", late.height, " regime A", late.filter(pl.col("regime") == "A").height,
    " regime B", late.filter(pl.col("regime") == "B").height)

# ---------------- 3. restart match ----------------
out("\n== 3. restart-type match ==")
show(MK.filter(pl.col("match_key") == RM).select(
    ["match_key", "regime", "n_round_announce_match_start", "n_cs_win_panel_round", "r1_rows", "r1_p20",
     "end_pairs", "dup_pairs", "min_end_round", "max_end_round"]))
show(RS.filter((pl.col("match_key") == RM) & (pl.col("round") <= 2)).sort("tick")
     .select(["round", "tick", "event_type", "t_score", "ct_score"]))
show(RE_raw.filter((pl.col("match_key") == RM) & (pl.col("round") <= 2)).sort(["round", "tick"]))

# ---------------- 4. round-13 bomb mismatch by regime ----------------
out("\n== 4. bomb-outcome vs round_end winner, by regime and round label ==")
BSc = (scan("bomb_state").select(["match_key", "round", "tick", "event_type"])
       .filter(pl.col("event_type").is_in(["bomb_planted", "bomb_defused", "bomb_exploded"])).collect())
OUT_BS = (BSc.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
          .group_by(["match_key", "round"]).agg(
              pl.col("event_type").unique().alias("evs"),
              pl.len().alias("n_outcome_rows"))
          .with_columns(bomb_side=pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2)))
out("bomb outcome (match, round) pairs:", OUT_BS.height, " with >1 outcome row:",
    OUT_BS.filter(pl.col("n_outcome_rows") > 1).height)
RW = RE1.select(["match_key", "round", "winner", "pc"])
MMall = RW.join(OUT_BS, on=["match_key", "round"], how="inner").with_columns(
    mismatch=(pl.col("bomb_side") != pl.col("winner")))
MM = MMall.filter(pl.col("pc") == 20)
out("outcome rows with round_end pc==20:", MM.height, " mismatches:", MM.filter(pl.col("mismatch")).height)
out("outcome rows any pc:", MMall.height, " mismatches:", MMall.filter(pl.col("mismatch")).height)
MMr = MM.join(MK.select(["match_key", "regime", "day", "map_name"]), on="match_key", how="left").with_columns(
    grp=pl.when(pl.col("day").is_in(LATE)).then(pl.lit("late")).when(pl.col("day").is_in(EARLY))
    .then(pl.lit("early")).otherwise(pl.lit("other")))
out("mismatch rows by round label and regime (pc==20):")
show(MMr.group_by(["regime", "round"]).agg(pl.len().alias("rows"), pl.col("mismatch").sum().alias("mm"))
     .sort(["regime", "round"]))
out("round 13 by regime and day group:")
R13 = MMr.filter(pl.col("round") == 13)
show(R13.group_by(["grp", "regime"]).agg(pl.len().alias("rows"), pl.col("mismatch").sum().alias("mm"),
                                         pl.col("mismatch").mean().alias("rate")).sort(["grp", "regime"]))
AFF = R13.filter(pl.col("mismatch"))["match_key"].unique().to_list()
out("matches with a round-13 mismatch (pc==20):", len(AFF))
MK = MK.with_columns(aff13=pl.col("match_key").is_in(AFF))
show(MK.group_by(["regime", "aff13"]).agg(pl.len().alias("matches")).sort(["regime", "aff13"]))
out("round-13 outcome rows (pc==20) late/early/other by regime, for denominators:")
show(R13.group_by(["grp", "regime"]).agg(pl.len().alias("rows")).sort(["grp", "regime"]))

# ---------------- 5. within regime A late: what separates affected matches ----------------
out("\n== 5. within regime A late matches: affected (r13 mismatch) vs control ==")
AL = MK.filter((pl.col("regime") == "A") & pl.col("day").is_in(LATE))
show(AL.group_by("aff13").agg(pl.len().alias("matches")))
show(AL.group_by(["aff13", "n_round_announce_match_start"]).agg(pl.len().alias("matches")).sort(["aff13", "n_round_announce_match_start"]))
show(AL.group_by(["map_name", "aff13"]).agg(pl.len().alias("matches")).sort(["map_name", "aff13"]))
show(AL.group_by(["aff13", "n_round_freeze_end"]).agg(pl.len().alias("matches")).sort(["aff13", "n_round_freeze_end"]).head(20))

RS_PAIRS = scan("round_state").select(["match_key", "event_type"]).unique().collect()
out("round_state distinct (match, event_type) pairs:", RS_PAIRS.height, " distinct event types:",
    RS_PAIRS["event_type"].n_unique())
PRES = (RS_PAIRS.with_columns(p=pl.lit(1)).pivot(on="event_type", index="match_key", values="p",
                                                  aggregate_function="max").fill_null(0))
types = [c for c in PRES.columns if c != "match_key"]
A2 = AL.select(["match_key", "aff13"]).join(PRES, on="match_key", how="left").fill_null(0)
g = A2.group_by("aff13").agg([pl.col(t).mean().alias(t) for t in types]).to_dicts()
rate = {row["aff13"]: row for row in g}
if True in rate and False in rate:
    diffs = sorted(((rate[True][t] - rate[False][t], t, rate[True][t], rate[False][t]) for t in types),
                   key=lambda x: -abs(x[0]))
    out("event types with the largest presence-rate difference (affected minus control), regime A late:")
    out("diff | event_type | rate_affected | rate_control")
    for d, t, ra, rc in diffs[:20]:
        out(f"{d:+.3f} | {t} | {ra:.3f} | {rc:.3f}")

# ---------------- 6. freeze-end score vs round_end winner ----------------
out("\n== 6. freeze-end score delta vs round_end winner (side semantics) ==")
FE = (RS.filter(pl.col("event_type") == "round_freeze_end").sort(["match_key", "round", "tick"])
      .group_by(["match_key", "round"]).agg(
          pl.col("t_score").first().alias("t_fe"),
          pl.col("ct_score").first().alias("ct_fe"),
          pl.len().alias("n_fe")))
out("freeze-end (match, round) pairs:", FE.height, " with >1 fe row:", FE.filter(pl.col("n_fe") > 1).height)
FEN = FE.select(["match_key", (pl.col("round") - 1).alias("round"),
                 pl.col("t_fe").alias("t_next"), pl.col("ct_fe").alias("ct_next")])
PAIR = (FE.join(FEN, on=["match_key", "round"], how="inner")
        .join(RW.select(["match_key", "round", "winner"]), on=["match_key", "round"], how="left")
        .join(MK.select(["match_key", "regime", "day", "aff13"]), on="match_key", how="left")
        .with_columns(dT=pl.col("t_next") - pl.col("t_fe"), dCT=pl.col("ct_next") - pl.col("ct_fe")))
PAIR = PAIR.with_columns(fe_side=pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(2)
                         .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(3)
                         .otherwise(pl.lit(None)))
PAIR = PAIR.with_columns(
    cat=pl.when(pl.col("winner").is_null()).then(pl.lit("winner_null"))
    .when(pl.col("fe_side").is_null()).then(pl.lit("fe_delta_not_1_0"))
    .when(pl.col("winner") == pl.col("fe_side")).then(pl.lit("agree"))
    .otherwise(pl.lit("disagree")))
show(PAIR.group_by(["regime", "cat"]).agg(pl.len().alias("pairs")).sort(["regime", "cat"]))
out("per round label, regime A and B (pairs / agree / disagree / fe_delta_not_1_0 / winner_null):")
show(PAIR.group_by(["regime", "round", "cat"]).agg(pl.len().alias("n")).sort(["regime", "round", "cat"])
     .pivot(on="cat", index=["regime", "round"], values="n", aggregate_function="sum").fill_null(0)
     .sort(["regime", "round"]))

# ---------------- 7. round-1 proxy used by the proposed fix ----------------
out("\n== 7. round-1 winner vs freeze-end scores at round 2 (proposed regime-B proxy) ==")
F1 = FE.filter(pl.col("round") == 1).select(["match_key", pl.col("t_fe").alias("t1"), pl.col("ct_fe").alias("ct1")])
F2 = FE.filter(pl.col("round") == 2).select(["match_key", pl.col("t_fe").alias("t2"), pl.col("ct_fe").alias("ct2")])
R1 = (MK.select(["match_key", "regime"]).join(F1, on="match_key", how="left").join(F2, on="match_key", how="left")
      .join(RW.filter(pl.col("round") == 1).select(["match_key", pl.col("winner").alias("w1")]), on="match_key", how="left"))
R1 = R1.with_columns(
    proxy=pl.when((pl.col("t2") == 1) & (pl.col("ct2") == 0)).then(2)
    .when((pl.col("ct2") == 1) & (pl.col("t2") == 0)).then(3).otherwise(pl.lit(None)),
    fe1_zero=(pl.col("t1") == 0) & (pl.col("ct1") == 0),
    fe2_sum1=(pl.col("t2") + pl.col("ct2")) == 1)
show(R1.group_by("regime").agg(
    pl.len().alias("matches"),
    pl.col("fe1_zero").fill_null(False).sum().alias("fe1_is_0_0"),
    pl.col("fe2_sum1").fill_null(False).sum().alias("fe2_total_is_1"),
    pl.col("proxy").is_not_null().sum().alias("proxy_defined"),
    pl.col("w1").is_not_null().sum().alias("round1_end_winner_present"),
    ((pl.col("proxy") == pl.col("w1")) & pl.col("w1").is_not_null()).sum().alias("proxy_eq_round_end_w1")))
R1A = R1.filter((pl.col("regime") == "A") & pl.col("w1").is_not_null() & pl.col("proxy").is_not_null())
out("regime A: round-1 proxy vs round_end w1 disagreements:", R1A.filter(pl.col("proxy") != pl.col("w1")).height,
    "of", R1A.height)

# ---------------- 8. round-13 three-way: bomb vs round_end vs freeze-end delta ----------------
out("\n== 8. round 13: bomb side vs round_end winner vs freeze-end delta (pc==20 outcome rows) ==")
R13_3 = (MM.filter(pl.col("round") == 13)
         .join(PAIR.filter(pl.col("round") == 13).select(["match_key", "fe_side"]), on="match_key", how="left")
         .join(MK.select(["match_key", "regime", "day"]), on="match_key", how="left")
         .with_columns(grp=pl.when(pl.col("day").is_in(LATE)).then(pl.lit("late")).when(pl.col("day").is_in(EARLY))
                       .then(pl.lit("early")).otherwise(pl.lit("other"))))
show(R13_3.group_by(["grp", "regime", "mismatch"]).agg(
    pl.len().alias("rows"),
    (pl.col("fe_side") == pl.col("bomb_side")).sum().alias("fe_side_eq_bomb"),
    (pl.col("fe_side") == pl.col("winner")).sum().alias("fe_side_eq_roundend"),
    pl.col("fe_side").is_null().sum().alias("fe_side_null")).sort(["grp", "regime", "mismatch"]))

# bomb source consistency: round_state bomb events vs bomb_state channel (round 13)
RSB = (RS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
       .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs2"))
       .with_columns(bomb_side_rs=pl.when(pl.col("evs2").list.contains("bomb_defused")).then(3).otherwise(2)))
XB = (OUT_BS.join(RSB.select(["match_key", "round", "bomb_side_rs"]), on=["match_key", "round"], how="left")
      .join(MK.select(["match_key", "regime"]), on="match_key", how="left"))
out("bomb outcome pairs in bomb_state vs round_state, by regime (rs_missing / agree / disagree), all rounds:")
show(XB.group_by("regime").agg(
    pl.len().alias("pairs"),
    pl.col("bomb_side_rs").is_null().sum().alias("rs_missing"),
    (pl.col("bomb_side_rs") == pl.col("bomb_side")).sum().alias("agree"),
    ((pl.col("bomb_side_rs") != pl.col("bomb_side")) & pl.col("bomb_side_rs").is_not_null()).sum().alias("disagree")))

out("\n== end ==")
FH.close()
