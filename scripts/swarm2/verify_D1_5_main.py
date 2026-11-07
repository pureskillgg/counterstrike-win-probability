"""verify_D1_5_main: independent recomputation of the D1-5 claim (swarm2 verifier).

Claim under test: round_end winner_team_code is the inverted label at 2024 round 13
(score side and bomb outcome agree; winner disagrees). Proposed fix: relabel 2024 round-13
winners from the round_state score side (fallback bomb outcome). Validation claimed on 2026.

ASSUMPTIONS (marked where used)
  A1 side codes: 2 = T, 3 = CT. Bomb outcome: bomb_exploded -> 2, bomb_defused -> 3.
  A2 population: 2024-02-g tome round_end rows with player_count == 20 (the claim's filter).
  A3 day group d12-25 = match_key day folder 12..25; d01-07 = 01..07 (the claim's grouping).
  A4 score side (claim T8): side whose score rose by exactly 1 between first and last round_state
     tick of the round (t_score / ct_score), else null.
  A5 header: t_starters_score_final = team that started on T (team A). Side of team A in round r
     follows bakeoff.swap_expected (halftime after round max_rounds//2, then after round
     max_rounds+3 and every 6 rounds). Header final totals are treated as engine truth.
  A6 complete match = number of round_end rounds equals t_starters + ct_starters final.
  A7 header is a reference only; it is not used to pick a fix, it is used to test which source
     reproduces the engine final totals.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
DAYS_HI = [f"{d:02d}" for d in range(12, 26)]


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def fl(ch):
    return sorted(glob.glob(str(BASE / f"basic_{ch}{SUF}" / "dataframe_*")))


def rd(ch, cols=None):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fl(ch)], how="vertical_relaxed")


def swap_expected(r, mr):  # copy of bakeoff.swap_expected (A5), reference only
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


def side_a(r, mr):
    """Side (2 = T, 3 = CT) of team A (started on T) during round r."""
    n = sum(1 for s in range(1, r) if swap_expected(s, mr))
    return 2 if n % 2 == 0 else 3


# ------------------------------------------------------------------ load
RE_all = rd("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                          "win_reason_message", "player_count"])
dup_re = RE_all.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
RE = RE_all.unique(["match_key", "round"], keep="first")
print("round_end rows", RE_all.height, "| unique (match,round)", RE.height, "| duplicated keys", dup_re)

BS = rd("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score",
                        "t_score_raw", "ct_score_raw"])

hd_files = sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
HD0 = pl.concat([pl.read_parquet(f) for f in hd_files], how="vertical_relaxed")
print("header cols:", HD0.columns, "rows", HD0.height)
KEYCOL = "key" if "key" in HD0.columns else "match_key"
# A8: the 2024 header has no max_rounds column; MR12 regulation (24) is assumed for every match.
# Consistent with the claimed F1 flip at 27->28 (overtime swap rule) in D1_followup.log.
HD = (HD0.select([pl.col(KEYCOL).alias("match_key"), pl.lit(24).alias("max_rounds"), "t_starters_score_final",
                  "ct_starters_score_final"]).unique("match_key"))

# ------------------------------------------------------------------ bomb outcome per round
outc = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"),
                                             pl.col("tick").min().alias("bomb_tick")))
RO = (RE.filter(pl.col("player_count") == 20)
      .join(outc, on=["match_key", "round"], how="inner")
      .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
      .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"),
                    pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
RO = RO.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))

# ------------------------------------------------------------------ score side (claim T8 definition, A4)
RSs = RS.sort(["match_key", "round", "tick"])
sc = RSs.group_by(["match_key", "round"]).agg(
    pl.col("t_score").sort_by("tick").first().alias("t0"), pl.col("ct_score").sort_by("tick").first().alias("c0"),
    pl.col("t_score").sort_by("tick").last().alias("t1"), pl.col("ct_score").sort_by("tick").last().alias("c1"),
    pl.col("t_score_raw").sort_by("tick").first().alias("t0r"), pl.col("ct_score_raw").sort_by("tick").first().alias("c0r"),
    pl.col("t_score_raw").sort_by("tick").last().alias("t1r"), pl.col("ct_score_raw").sort_by("tick").last().alias("c1r"),
    pl.col("tick").min().alias("w_first"), pl.col("tick").max().alias("w_last"))
sc = sc.with_columns(
    pl.when((pl.col("t1") - pl.col("t0")) == 1).then(2)
      .when((pl.col("c1") - pl.col("c0")) == 1).then(3).otherwise(None).cast(pl.Int64).alias("score_side"),
    pl.when((pl.col("t1r") - pl.col("t0r")) == 1).then(2)
      .when((pl.col("c1r") - pl.col("c0r")) == 1).then(3).otherwise(None).cast(pl.Int64).alias("score_side_raw"))

fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first().alias("fe_t"),
                                           pl.col("ct_score").first().alias("fe_c"),
                                           pl.col("tick").first().alias("fe_tick")))
oe = (RS.filter(pl.col("event_type") == "round_officially_ended").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").last().alias("oe_t"),
                                           pl.col("ct_score").last().alias("oe_c")))
sc = sc.join(fe, on=["match_key", "round"], how="left").join(oe, on=["match_key", "round"], how="left")
sc = sc.with_columns(
    pl.when((pl.col("oe_t") - pl.col("fe_t")) == 1).then(2)
      .when((pl.col("oe_c") - pl.col("fe_c")) == 1).then(3).otherwise(None).cast(pl.Int64).alias("score_side_B"))

RO2 = RO.join(sc.select(["match_key", "round", "score_side", "score_side_raw", "score_side_B", "fe_tick"]),
              on=["match_key", "round"], how="left")
RO2 = RO2.with_columns(
    (pl.col("bomb_side") == pl.col("score_side")).alias("bomb_eq_score"),
    (pl.col("winner_team_code") == pl.col("score_side")).alias("win_eq_score"))

# ------------------------------------------------------------------ T-A headline recompute
hdr("T-A headline recompute: round-13 rows with a bomb outcome (player_count == 20)")
r13 = RO2.filter(pl.col("round") == 13)
print(r13.group_by("dg").agg(pl.len().alias("n_bomb_rows"), pl.col("mismatch").sum().alias("mismatch"),
                             pl.col("mismatch").mean().round(4).alias("rate")).sort("dg"))
print("claim: d01-07 n=180 mm=0 | d12-25 n=605 mm=434 | total 785 (mm 434)")
r13s = r13.filter(pl.col("score_side").is_not_null())
print("\nround-13 bomb rows with non-null score side (A4):")
print(r13s.group_by(["dg", "mismatch"]).agg(pl.len().alias("n"), pl.col("bomb_eq_score").sum().alias("bomb_eq_score"),
                                            pl.col("win_eq_score").sum().alias("winner_eq_score")).sort(["dg", "mismatch"]))
print("claim: bomb==score 785/785 ; winner==score 351/351 clean, 0/434 mismatch")
nul13 = r13.filter(pl.col("score_side").is_null())
print("round-13 bomb rows with NULL score side:", nul13.height, nul13.group_by(["dg", "mismatch"]).len().sort(["dg", "mismatch"]).to_dicts())
print("all round-13 bomb rows, bomb_eq_score by mismatch (non-null only) and raw-score agreement:",
      r13.filter(pl.col("score_side_raw").is_not_null()).group_by("mismatch").agg(
          pl.len().alias("n"), (pl.col("bomb_side") == pl.col("score_side_raw")).sum().alias("bomb_eq_raw_score")).to_dicts())

# ------------------------------------------------------------------ T-B round 13 winner vs score side, all rows
hdr("T-B round 13 winner_team_code vs score side, ALL player_count == 20 rounds (no bomb filter)")
bomb13 = set(RO.filter(pl.col("round") == 13).select(["match_key"]).to_series().to_list())
mm13 = RO.filter((pl.col("round") == 13) & pl.col("mismatch")).select("match_key").to_series().to_list()
R13A = (RE.filter((pl.col("player_count") == 20) & (pl.col("round") == 13))
        .join(sc.select(["match_key", "round", "score_side", "score_side_raw", "score_side_B"]), on=["match_key", "round"], how="left"))
R13A = R13A.with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day"),
                         pl.col("match_key").is_in(list(bomb13)).alias("has_r13_bomb"),
                         pl.col("match_key").is_in(mm13).alias("r13_bomb_mismatch"))
R13A = R13A.with_columns(pl.when(pl.col("day").is_in(DAYS_HI)).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
print(R13A.group_by(["dg", "has_r13_bomb"]).agg(
    pl.len().alias("n_rows"), pl.col("score_side").is_null().sum().alias("null_score"),
    ((pl.col("winner_team_code") != pl.col("score_side")) & pl.col("score_side").is_not_null()).sum().alias("win_ne_score"),
    ((pl.col("winner_team_code") == pl.col("score_side")) & pl.col("score_side").is_not_null()).sum().alias("win_eq_score"),
).sort(["dg", "has_r13_bomb"]))
print("\nd12-25 round 13 rows, winner vs score side, split by whether a bomb outcome exists and mismatch status:")
x = R13A.filter(pl.col("dg") == "d12-25")
print(x.group_by(["has_r13_bomb", "r13_bomb_mismatch"]).agg(
    pl.len().alias("n"), pl.col("score_side").is_null().sum().alias("null_score"),
    ((pl.col("winner_team_code") != pl.col("score_side")) & pl.col("score_side").is_not_null()).sum().alias("win_ne_score")
).sort(["has_r13_bomb", "r13_bomb_mismatch"]))
print("alt score def B (freeze_end->officially_ended), d12-25 round 13 win_ne_scoreB:",
      x.filter((pl.col("winner_team_code") != pl.col("score_side_B")) & pl.col("score_side_B").is_not_null()).height,
      " nulls:", x.filter(pl.col("score_side_B").is_null()).height)
print("raw score def, d12-25 round 13 win_ne_score_raw:",
      x.filter((pl.col("winner_team_code") != pl.col("score_side_raw")) & pl.col("score_side_raw").is_not_null()).height,
      " nulls:", x.filter(pl.col("score_side_raw").is_null()).height)
print("claim: d12-25 round 13 disagreements 2080 of 3062 (67.9%); d01-07 0 of 772")

# ------------------------------------------------------------------ T-C null score side by round
hdr("T-C null score side (A4) by round number, all player_count == 20 rounds")
RE20 = RE.filter(pl.col("player_count") == 20).join(sc.select(["match_key", "round", "score_side"]),
                                                     on=["match_key", "round"], how="left")
print("total rows", RE20.height, "| null score side", RE20.filter(pl.col("score_side").is_null()).height)
print("null score side by round (count):")
print(RE20.filter(pl.col("score_side").is_null()).group_by("round").len().sort("round").to_dicts())
print("null score side among d12-25 round 13:", RE20.filter((pl.col("round") == 13) & pl.col("score_side").is_null()).height)

# ------------------------------------------------------------------ T-D header test (A5, A6)
hdr("T-D header final-score test: which source reproduces the engine starter totals?")
winners = {}
for mk, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    winners.setdefault(mk, {})[int(r)] = int(w)
rs_last = (RS.sort(["match_key", "tick"]).group_by("match_key").agg(
    pl.col("t_score").last().alias("t_fin"), pl.col("ct_score").last().alias("c_fin"))).to_dicts()
rs_last = {d["match_key"]: d for d in rs_last}
bomb_side_by_round = {}
for mk, r, bs in RO.select(["match_key", "round", "bomb_side"]).iter_rows():
    bomb_side_by_round.setdefault(mk, {})[int(r)] = int(bs)
mm_rounds = {}
for mk, r in RO.filter(pl.col("mismatch")).select(["match_key", "round"]).iter_rows():
    mm_rounds.setdefault(mk, set()).add(int(r))
score_side_by_round = {}
for mk, r, s in sc.select(["match_key", "round", "score_side"]).iter_rows():
    if s is not None:
        score_side_by_round.setdefault(mk, {})[int(r)] = int(s)

stats = {}
def bump(k):
    stats[k] = stats.get(k, 0) + 1

complete = []
for mk, mr, tf, cf in HD.select(["match_key", "max_rounds", "t_starters_score_final", "ct_starters_score_final"]).iter_rows():
    if mk not in winners:
        bump("no_round_end"); continue
    w = winners[mk]
    n = len(w)
    if sorted(w) != list(range(1, n + 1)):
        bump("round_gap"); continue
    if n != tf + cf:
        bump("round_count_not_header_total"); continue
    complete.append((mk, mr, tf, cf, n))
print("complete matches (A6):", len(complete))

def classify(v_a, v_b, tf, cf):
    if (v_a, v_b) == (tf, cf):
        return "ok"
    if (v_a, v_b) == (cf, tf):
        return "swapped"
    return "neither"

per_match = []
for mk, mr, tf, cf, n in complete:
    w = winners[mk]
    # round_end totals with the swap rule (A5)
    rd_a = sum(1 for r in range(1, n + 1) if w[r] == side_a(r, mr))
    rd_b = n - rd_a
    # round_end totals with the score-side fix applied to round 13 (claim's proposed relabel)
    w_fix = dict(w)
    if 13 in w and mk in mm13:
        s13 = score_side_by_round.get(mk, {}).get(13)
        if s13 is not None:
            w_fix[13] = s13
        else:
            w_fix[13] = bomb_side_by_round[mk][13]
    fa = sum(1 for r in range(1, n + 1) if w_fix[r] == side_a(r, mr))
    fb = n - fa
    # round_end totals with the bomb outcome substituted for every bomb-mismatch round
    w_bomb = dict(w)
    for r in mm_rounds.get(mk, set()):
        w_bomb[r] = bomb_side_by_round[mk][r]
    ba = sum(1 for r in range(1, n + 1) if w_bomb[r] == side_a(r, mr))
    bb = n - ba
    # round_state final totals mapped to team A through the swap rule
    rsf = rs_last.get(mk)
    if rsf is None:
        bump("no_round_state"); continue
    end_side_a = side_a(n + 1, mr)
    rs_a = rsf["t_fin"] if end_side_a == 2 else rsf["c_fin"]
    rs_b = rsf["c_fin"] if end_side_a == 2 else rsf["t_fin"]
    per_match.append({
        "mk": mk, "mm13": mk in mm13, "mm_any": mk in mm_rounds, "n_mm": len(mm_rounds.get(mk, set())),
        "rd": classify(rd_a, rd_b, tf, cf), "rd_fix13": classify(fa, fb, tf, cf),
        "rd_fixbomb": classify(ba, bb, tf, cf), "rs": classify(rs_a, rs_b, tf, cf),
        "rd_a": rd_a, "rd_b": rd_b, "rs_a": rs_a, "rs_b": rs_b, "tf": tf, "cf": cf,
    })

import collections
tab = collections.Counter((p["mm13"], p["rd"], p["rs"]) for p in per_match)
print("\n(mismatch-at-r13, round_end-raw status, round_state status) -> matches")
for k, v in sorted(tab.items()):
    print(k, v)
print("\nmismatch-at-r13 matches: header status under each candidate source")
for lbl, key in [("round_end raw", "rd"), ("round_end, R13 := score side (claim fix)", "rd_fix13"),
                 ("round_end, every bomb-mismatch round := bomb side", "rd_fixbomb"), ("round_state final totals", "rs")]:
    c = collections.Counter(p[key] for p in per_match if p["mm13"])
    print(f"  {lbl:50s} {dict(c)}")
print("\nclean matches (no bomb mismatch) header status under round_end raw / round_state:")
c_clean = collections.Counter((p["rd"], p["rs"]) for p in per_match if not p["mm_any"])
print("  ", dict(c_clean))
print("\nmatches with any bomb mismatch but NOT r13 (other rounds only):")
oth = [p for p in per_match if p["mm_any"] and not p["mm13"]]
print("  count", len(oth), collections.Counter((p["rd"], p["rd_fixbomb"], p["rs"]) for p in oth))
print("\nmatches with mismatches in r13 AND other rounds:")
both = [p for p in per_match if p["mm13"] and p["n_mm"] > 1]
print("  count", len(both), "rd_fixbomb", collections.Counter(p["rd_fixbomb"] for p in both),
      "rd_fix13", collections.Counter(p["rd_fix13"] for p in both))
print("\nstats", stats)
