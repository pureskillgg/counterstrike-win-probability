"""verify_D9_9 step 5b (exploration): compact round sequences for a few matches per pattern, build 9961 and 9842.

Columns: round, rec (round_end winner), bomb (bomb side or -), tA,cA (end scores), tP,cP (prev end scores),
straight (dT,dC), swapped (dT',dC'), score winner, swap_sig flag, n_T, n_CT, T_dead, CT_dead.
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(20)
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D9_9")
R = pl.read_parquet(SCR / "verify_D9_9_round_oracle.parquet")
pat = (R.filter(pl.col("swap_sig")).sort(["match_key", "round"]).group_by("match_key")
       .agg(pl.col("round").cast(pl.Utf8).str.join(",").alias("swap_pat")))
R = R.join(pat, on="match_key", how="left").with_columns(pl.col("swap_pat").fill_null("none"))


def show(build, pattern, n, rmin, rmax):
    keys = (R.filter((pl.col("build_num") == build) & (pl.col("swap_pat") == pattern))
            .select("match_key").unique().head(n)["match_key"].to_list())
    for k in keys:
        d = R.filter((pl.col("match_key") == k) & pl.col("round").is_between(rmin, rmax)).sort("round")
        print(f"\n## build {build} pattern {pattern} match ...{k[-20:]}")
        rows = d.select(["round", "winner_team_code", "bomb_side", "t_after", "ct_after", "t_prev", "ct_prev",
                         "dT_straight", "dCT_straight", "dT_swapped", "dCT_swapped", "w_score", "swap_sig",
                         "n_T", "n_CT", "T_dead", "CT_dead"]).rows()
        print("  r  rec bomb tA cA tP cP | dT dC | dT' dC' | wS sig | nT nCT Td CTd")
        for r in rows:
            rd, rec, bomb, tA, cA, tP, cP, dT, dC, dTs, dCs, wS, sig, nT, nC, Td, CTd = r
            print(f" {rd:>2} {rec!s:>3} {str(bomb):>4} {tA!s:>2} {cA!s:>2} {tP!s:>2} {cP!s:>2} | {dT!s:>2} {dC!s:>2} | {dTs!s:>2} {dCs!s:>2} | {wS!s:>2} {str(sig)[0]:>3} | {nT!s:>2} {nC!s:>3} {Td!s:>2} {CTd!s:>3}")


show(9961, "12", 2, 9, 15)
show(9961, "none", 3, 9, 15)
show(9842, "none", 2, 9, 15)
show(9961, "8", 1, 6, 12)
show(9961, "12,27", 1, 24, 30)
show(9842, "12,27", 1, 24, 30)
