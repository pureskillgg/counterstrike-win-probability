"""verify_D9_2 kscan: which round-label position is the halftime, per build, judged by header final scores.

For a match with round_end labels L (winner codes w), and header final scores T_hdr (team that started T)
and CT_hdr (team that started CT):
  T_k  = #(label <= k and w == 2) + #(label > k and w == 3)   (rows labelled <= k are the first half)
  CT_k = #(label <= k and w == 3) + #(label > k and w == 2)
A match agrees at k if T_k == T_hdr and CT_k == CT_hdr. The standard MR12 halftime is k = 12.
Two code versions: recorded winner codes, and bomb-corrected codes (bomb side replaces winner on bomb rounds).
Reads only the scratch parquet from verify_D9_2_core.py and the header (projection).
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")

R = pl.read_parquet(SCR / "verify_D9_2_round.parquet").select(
    ["match_key", "round", "winner_team_code", "w_bomb", "build_num", "player_count", "has_out", "mm"])
H = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "t_starters_score_final", "ct_starters_score_final", "match_date"]).collect()
H = H.with_columns(pl.col("key").alias("match_key"), pl.col("match_date").str.slice(0, 10).alias("date_utc"))

pm = (R.group_by("match_key").agg(pl.len().alias("n_lab"), pl.col("round").min().alias("lo"),
                                  pl.col("round").max().alias("hi"), pl.col("build_num").first())
        .join(H.select(["match_key", "t_starters_score_final", "ct_starters_score_final", "date_utc"]),
              on="match_key", how="left"))
pm = pm.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hdr_sum"),
                     (pl.col("hi") - pl.col("lo") + 1 - pl.col("n_lab")).alias("internal_gaps"))
print("== label structure per build")
print("(build, lo, count):", pm.group_by(["build_num", "lo"]).len().sort(["build_num", "lo"]).rows())
print("(build, internal gaps, count):", pm.group_by(["build_num", "internal_gaps"]).len()
      .sort(["build_num", "internal_gaps"]).rows())
print("(build, hdr_sum - n_lab, count):", pm.group_by(["build_num", pl.col("hdr_sum") - pl.col("n_lab")]).len()
      .sort(["build_num"]).rows()[:0])
diff = pm.with_columns((pl.col("hdr_sum") - pl.col("n_lab")).alias("d")).group_by(["build_num", "d"]).len().sort(
    ["build_num", "d"])
print("(build, hdr_sum - n_labels, count) =", diff.rows())
print("(build, hdr_sum - hi, count) =", pm.with_columns((pl.col("hdr_sum") - pl.col("hi")).alias("d2"))
      .group_by(["build_num", "d2"]).len().sort(["build_num", "d2"]).rows())

# scan halftime position k on matches that have no internal gaps (labels contiguous)
cont = pm.filter(pl.col("internal_gaps") == 0)
Rc = R.join(cont.select(["match_key"]), on="match_key", how="inner")


def agree_table(code_col, label):
    rows = []
    for k in [10, 11, 12, 13, 14]:
        T = (pl.when((pl.col("round") <= k) & (pl.col(code_col) == 2)).then(1).otherwise(0)
             + pl.when((pl.col("round") > k) & (pl.col(code_col) == 3)).then(1).otherwise(0))
        C = (pl.when((pl.col("round") <= k) & (pl.col(code_col) == 3)).then(1).otherwise(0)
             + pl.when((pl.col("round") > k) & (pl.col(code_col) == 2)).then(1).otherwise(0))
        g = (Rc.with_columns(T.alias("T"), C.alias("C"))
               .group_by("match_key").agg(pl.col("T").sum(), pl.col("C").sum(), pl.col("build_num").first()))
        g = g.join(H.select(["match_key", "t_starters_score_final", "ct_starters_score_final"]),
                   on="match_key", how="left")
        g = g.with_columns(((pl.col("T") == pl.col("t_starters_score_final")) &
                            (pl.col("C") == pl.col("ct_starters_score_final"))).alias("ok"))
        s = g.group_by("build_num").agg(pl.len().alias("n"), pl.col("ok").sum().alias("agree")).sort("build_num")
        for b, n, a in s.rows():
            rows.append((label, k, b, n, a, round(a / n, 4)))
    print(f"-- {label}: halftime label k scan (continuous-label matches only)")
    for r in rows:
        print("   ", r)


agree_table("winner_team_code", "recorded")
agree_table("w_bomb", "bomb-corrected")

# per-build detail at k=12 with recorded codes: which matches disagree and by how much
print("\n== k=12 recorded: distribution of (T_k - T_hdr, C_k - C_hdr) by build (continuous matches)")
for code_col in ["winner_team_code"]:
    T = (pl.when((pl.col("round") <= 12) & (pl.col(code_col) == 2)).then(1).otherwise(0)
         + pl.when((pl.col("round") > 12) & (pl.col(code_col) == 3)).then(1).otherwise(0))
    g = (Rc.with_columns(T.alias("T"))
           .group_by("match_key").agg(pl.col("T").sum(), pl.col("build_num").first())
           .join(H.select(["match_key", "t_starters_score_final", "ct_starters_score_final", "date_utc"]),
                 on="match_key", how="left")
           .join(pm.select(["match_key", "n_lab", "hdr_sum"]), on="match_key", how="left"))
    g = g.with_columns((pl.col("T") - pl.col("t_starters_score_final")).alias("dT"))
    print(g.group_by(["build_num", "dT"]).len().sort(["build_num", "dT"]).head(40))
    print("per build, matches with hdr_sum == n_lab (complete by header):",
          g.filter(pl.col("hdr_sum") == pl.col("n_lab")).group_by("build_num").len().rows())
    print("per build, matches with hdr_sum == n_lab + 1:",
          g.join(pm.select(["match_key", "hdr_sum", "n_lab"]), on="match_key", how="left", suffix="_x")
           .filter(pl.col("hdr_sum") - pl.col("n_lab") == 1).group_by("build_num").len().rows())
