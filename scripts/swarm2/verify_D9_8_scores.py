"""verify D9-8, part 2: score-based and header-based truth for round 13 (2024 tome, read-only).

Why: round_state score_winner (from round_end rows) exists only for build 9842 matches (checked in
verify_D9_8_recompute.py S4), so it cannot adjudicate build 9961. This script uses two sources that
do not use round_end winners or bomb_state events:

  T1  next-round pre-score delta: for round r, the change in team scores between round_freeze_end of
      r and round_freeze_end of r+1 is the result of round r. Valid when no halftime/OT side swap lies
      between the two rows. Swaps are assumed at the end of round 12 (halftime), end of round 24
      (OT start) and at OT blocks ending after rounds 27, 30, ... (ASSUMPTION A5), so rounds 12, 24
      and 27, 30, ... are excluded from T1.
  T2  header final score: header t_starters_score_final / ct_starters_score_final. Team A (starting T)
      wins rounds where A is T and winner_team_code == 2, or A is CT and winner_team_code == 3.
      Side of A: T for rounds 1-12, CT for 13-24, OT blocks of 3 rounds alternate starting with T at 25
      (ASSUMPTION A6). A is taken as t_starters (ASSUMPTION A7; the alternate mapping is also printed).

ASSUMPTIONS:
  A5  swaps at end of round 12, end of round 24, and after OT rounds 27, 30, ... (excluded from T1)
  A6  OT block rule above; checked on build 9842 (zero bomb mismatches) before use on 9961
  A7  header t_starters_score_final is the final score of the team that started on T
  A8  bomb side rule as in verify_D9_8_recompute.py (defused -> 3, exploded -> 2)
Output is printed; no files are written under E:.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(120)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)

BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"


def files_of(name):
    fd = BASE / "header" if name == "header" else BASE / f"basic_{name}{SUF}"
    return sorted(glob.glob(str(fd / "dataframe_*")))


def lz(name, cols):
    lf = pl.scan_parquet(files_of(name))
    have = lf.collect_schema().names()
    miss = [c for c in cols if c not in have]
    if miss:
        raise SystemExit(f"{name} missing columns {miss}")
    return lf.select(cols)


def sec(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


H = lz("header", ["key", "build_num", "t_starters_score_final", "ct_starters_score_final"]).collect()
H = H.rename({"key": "match_key"})
RE = lz("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_message",
                      "player_count"]).collect()
RE = RE.unique(subset=["match_key", "round"], keep="first")
BO = lz("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).collect()
OUT = BO.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_defused").any().alias("defused"),
    (pl.col("event_type") == "bomb_exploded").any().alias("exploded"),
).with_columns(
    pl.when(pl.col("defused")).then(pl.lit(3)).when(pl.col("exploded")).then(pl.lit(2))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("bomb_side"))
FE = lz("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(
    pl.col("event_type") == "round_freeze_end").collect()
FE = FE.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("t_score").first().alias("t_pre"), pl.col("ct_score").first().alias("ct_pre"))

# event-type presence in round_state by build (explains where round_end-based score winner exists)
RSE = lz("round_state", ["match_key", "event_type"]).collect()
RSE = RSE.join(H.select(["match_key", "build_num"]), on="match_key", how="left")
sec("S7 round_state event presence by build (matches having >=1 row of the event)")
pres = RSE.group_by(["build_num", "event_type", "match_key"]).len().group_by(["build_num", "event_type"]).agg(
    pl.len().alias("matches_with_event")).sort(["event_type", "build_num"])
print(pres.filter(pl.col("event_type").is_in(["round_freeze_end", "round_end", "cs_win_panel_round",
                                                "bomb_defused", "bomb_exploded"])))
print("matches per build:", H.group_by("build_num").len().sort("build_num").rows())

# ---------------------------------------------------------------- T1 next-round pre-score delta
F = FE.sort(["match_key", "round"]).with_columns(
    pl.col("round").shift(-1).over("match_key").alias("next_round"),
    pl.col("t_pre").shift(-1).over("match_key").alias("t_next"),
    pl.col("ct_pre").shift(-1).over("match_key").alias("ct_next"),
).filter(pl.col("next_round") == pl.col("round") + 1)
F = F.with_columns(
    (pl.col("t_next") - pl.col("t_pre")).alias("dT_next"),
    (pl.col("ct_next") - pl.col("ct_pre")).alias("dCT_next"),
)
F = F.with_columns(
    pl.when((pl.col("dCT_next") == 1) & (pl.col("dT_next") == 0)).then(pl.lit(3))
      .when((pl.col("dT_next") == 1) & (pl.col("dCT_next") == 0)).then(pl.lit(2))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("next_winner"),
    (~pl.col("round").is_in([12, 24, 27, 30, 33, 36, 39, 42, 45, 48])).alias("no_swap_between"),
)
J = (RE.join(OUT, on=["match_key", "round"], how="left")
       .join(F.select(["match_key", "round", "next_winner", "no_swap_between", "dT_next", "dCT_next"]),
             on=["match_key", "round"], how="left")
       .join(H.select(["match_key", "build_num"]), on="match_key", how="left")
       .with_columns(
           pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
             .otherwise(pl.lit("other")).alias("rbin"),
           pl.col("defused").fill_null(False), pl.col("exploded").fill_null(False)))
J = J.with_columns(
    (pl.col("defused") | pl.col("exploded")).alias("has_out"),
    (pl.col("winner_team_code") != pl.col("next_winner")).alias("re_vs_next_mismatch"),
    (pl.col("bomb_side") != pl.col("next_winner")).alias("bomb_vs_next_mismatch"),
)
sec("S7b T1 next-round pre-score winner vs round_end winner and vs bomb side (rows with next_winner)")
A = J.filter(pl.col("next_winner").is_not_null() & pl.col("no_swap_between"))
print("rows used:", A.height)
print(A.group_by(["build_num", "rbin"]).agg(
    pl.len().alias("n"),
    pl.col("re_vs_next_mismatch").sum().alias("round_end_vs_next_mismatches"),
    pl.col("bomb_vs_next_mismatch").filter(pl.col("has_out")).sum().alias("bomb_vs_next_mismatches_with_outcome"),
    pl.col("has_out").sum().alias("n_with_outcome"),
).sort(["build_num", "rbin"]))
print("\nround 13, build 9961, has_out rows: round_end winner vs next-round winner vs bomb side")
r13 = A.filter((pl.col("round") == 13) & pl.col("has_out"))
print(r13.group_by(["build_num", "winner_team_code", "next_winner", "bomb_side", "win_reason_message"]).len()
      .sort(["build_num", "winner_team_code", "next_winner", "bomb_side", "win_reason_message"]))
print("\nround 13 dT_next/dCT_next classes by build (all round-13 rows, incl. no outcome):")
print(J.filter(pl.col("round") == 13).group_by(["build_num", "dT_next", "dCT_next"]).len()
      .sort(["build_num", "dT_next", "dCT_next"]))

# ---------------------------------------------------------------- T2 header totals
sec("S8 header final score vs round_end totals (T2). A = team starting T (ASSUMPTION A6, A7)")
win = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
bomb = {(m, r): b for m, r, b in OUT.filter(pl.col("bomb_side").is_not_null())
        .select(["match_key", "round", "bomb_side"]).iter_rows()}
hdr = {m: (b, t, c) for m, b, t, c in H.select(["match_key", "build_num", "t_starters_score_final",
                                               "ct_starters_score_final"]).iter_rows()}
rounds_of = collections.defaultdict(list)
for (m, r) in win:
    rounds_of[m].append(r)


def a_is_t(r):
    if r <= 12:
        return True
    if r <= 24:
        return False
    return ((r - 25) // 3) % 2 == 0


def a_wins(m, variant_bomb):
    n = 0
    for r in rounds_of[m]:
        w = bomb.get((m, r), win[(m, r)]) if variant_bomb else win[(m, r)]
        if (a_is_t(r) and w == 2) or ((not a_is_t(r)) and w == 3):
            n += 1
    return n


res = collections.Counter()
per_match = {}
for m, (b, t_fin, c_fin) in hdr.items():
    if m not in rounds_of:
        res[(b, "no_round_end")] += 1
        continue
    nr = len(rounds_of[m])
    a_re = a_wins(m, False)
    a_bo = a_wins(m, True)
    ok_re = (a_re == t_fin) and (nr - a_re == c_fin)
    ok_bo = (a_bo == t_fin) and (nr - a_bo == c_fin)
    alt_re = (a_re == c_fin) and (nr - a_re == t_fin)
    res[(b, "roundend_matches_header" if ok_re else "roundend_mismatch")] += 1
    res[(b, "bomb_variant_matches_header" if ok_bo else "bomb_variant_mismatch")] += 1
    res[(b, "sum_final_eq_rounds" if (t_fin + c_fin) == nr else "sum_final_ne_rounds")] += 1
    res[(b, "alt_mapping_matches" if alt_re and not ok_re else "alt_mapping_no_extra")] += 1
    has_r13m1 = (m, 13) in bomb and bomb[(m, 13)] != win[(m, 13)]
    if has_r13m1:
        res[(b, "r13_m1_match_roundend_ok" if ok_re else "r13_m1_match_roundend_bad")] += 1
        res[(b, "r13_m1_match_bomb_ok" if ok_bo else "r13_m1_match_bomb_bad")] += 1
    per_match[m] = (b, t_fin, c_fin, nr, a_re, a_bo, has_r13m1)

for k in sorted(res):
    print(k, res[k])

sec("S8b matches with an r13 bomb mismatch: header agreement by build")
rows = [(b, nr, a_re, a_bo, t_fin, c_fin) for m, (b, t_fin, c_fin, nr, a_re, a_bo, flag) in per_match.items()
        if flag]
print("n r13 mismatch matches:", len(rows))
print(collections.Counter((b, a_re == t_fin, a_bo == t_fin) for b, _, a_re, a_bo, t_fin, _ in rows))

sec("S8c build 9842 OT/rule check: matches with >24 rounds, header agreement under round_end")
ot = [(m, v) for m, v in per_match.items() if v[3] > 24]
print("matches with >24 rounds:", len(ot), "| builds:", collections.Counter(v[0] for _, v in ot))
print("round_end agreement:", collections.Counter((v[0], v[4] == v[1]) for _, v in ot))

print("\nDONE")
