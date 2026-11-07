r"""Verifier for D5-5, stage 8: team-based header-final check (replaces the side-based check in stage 4 E,
which compared side-coded tallies with team-based header totals and is not valid).

Team A starts on T; side swap after a completed round r when swap_expected(r, 24) (copied from
scripts/model/bakeoff.py: r == 12 or (r > 24 and (r - 27) % 6 == 0)). Team tallies from
  (a) round_end winners (rw),
  (b) score-step winners where clean, else round_end winner (fallback),
compared with header t_starters_score_final / ct_starters_score_final (team A = T start, team B = CT start;
orientation either way accepted). ASSUMPTION: header finals are the team scores at match end.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D5_5_09_header_team.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)

SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D5_5")
T, CT = 2, 3
MR = 24


def swap_expected(r):
    return r == MR // 2 or (r > MR and (r - MR - 3) % 6 == 0)


W = pl.read_parquet(SCR / "wide.parquet").filter(pl.col("rw").is_not_null())
R1 = W.filter(pl.col("k") == 1).select(["match_key", pl.col("rw").is_not_null().alias("r1")])
H = pl.read_parquet(SCR / "header.parquet").rename({"key": "match_key"}).select(
    ["match_key", "t_starters_score_final", "ct_starters_score_final"])
rows = W.select(["match_key", "k", "rw", "step"]).sort(["match_key", "k"]).to_dicts()

out = []
by = {}
for r in rows:
    by.setdefault(r["match_key"], []).append((r["k"], r["rw"], r["step"]))
for m, lst in by.items():
    res = {}
    for mode in ["rw", "step_else_rw"]:
        side_a = T
        a = b = 0
        ok_contig = True
        for idx, (k, rw, st) in enumerate(lst):
            if k != idx + 1:
                ok_contig = False
            w = rw if mode == "rw" else (st if (st in (2, 3) and not swap_expected(k)) else rw)
            if w == side_a:
                a += 1
            else:
                b += 1
            if swap_expected(k):
                side_a = CT if side_a == T else T
        res[mode] = (a, b)
        res[f"{mode}_contig"] = ok_contig
    out.append({"match_key": m, "n": len(lst), "a_rw": res["rw"][0], "b_rw": res["rw"][1],
                "a_st": res["step_else_rw"][0], "b_st": res["step_else_rw"][1], "contig": res["rw_contig"]})
D = pl.DataFrame(out).join(H, on="match_key", how="left").join(R1, on="match_key", how="left").with_columns(
    pl.col("r1").fill_null(False))
D = D.with_columns(
    rw_match=((pl.col("a_rw") == pl.col("t_starters_score_final")) & (pl.col("b_rw") == pl.col("ct_starters_score_final")))
    | ((pl.col("a_rw") == pl.col("ct_starters_score_final")) & (pl.col("b_rw") == pl.col("t_starters_score_final"))),
    st_match=((pl.col("a_st") == pl.col("t_starters_score_final")) & (pl.col("b_st") == pl.col("ct_starters_score_final")))
    | ((pl.col("a_st") == pl.col("ct_starters_score_final")) & (pl.col("b_st") == pl.col("t_starters_score_final"))),
)
D = D.with_columns(
    rw_ne_st=(pl.col("a_rw") != pl.col("a_st")) | (pl.col("b_rw") != pl.col("b_st")))
print("== team-based header check (2024 tome) by r1 ==")
print(D.group_by("r1").agg(pl.len().alias("matches"),
                           pl.col("rw_match").sum().alias("rw_eq_header"),
                           pl.col("st_match").sum().alias("step_eq_header"),
                           pl.col("rw_ne_st").sum().alias("rw_and_step_tallies_differ")).sort("r1"))
print("\nr1-true matches where rw and step tallies differ: header equals step only:",
      int((pl.col("st_match") & ~pl.col("rw_match")).sum()) if False else
      D.filter(pl.col("r1") & pl.col("rw_ne_st") & pl.col("st_match") & ~pl.col("rw_match")).height,
      " header equals rw only:",
      D.filter(pl.col("r1") & pl.col("rw_ne_st") & pl.col("rw_match") & ~pl.col("st_match")).height,
      " neither:",
      D.filter(pl.col("r1") & pl.col("rw_ne_st") & ~pl.col("rw_match") & ~pl.col("st_match")).height)
print("\nexample of header totals vs tallies (first 8 r1-true rw!=step):")
print(D.filter(pl.col("r1") & pl.col("rw_ne_st")).select(
    ["match_key", "n", "a_rw", "b_rw", "a_st", "b_st", "t_starters_score_final", "ct_starters_score_final"]).head(8))
print("\nmatch-level round count vs header sum (all matches), by r1:")
print(D.with_columns((pl.col("n") - (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))).alias("d"))
      .group_by(["r1", (pl.col("d") == 0)]).len().sort(["r1"]))
D.write_parquet(SCR / "header_team.parquet")
