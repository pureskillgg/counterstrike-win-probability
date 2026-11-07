"""verify_D9_2 scoretest: header final scores vs round_end codes, build 9961, complete-coverage matches.

Complete coverage = labels 1..hi contiguous AND header T+CT final == number of labels (checked for both builds).
Score rule (MR12, halftime after label 12; roster flip confirmed at label 13 in both builds):
  T_k  = #(label<=12, w=2) + #(label>=13, w=3)     (team that started on T)
  CT_k = #(label<=12, w=3) + #(label>=13, w=2)
Hypotheses for the winner used in the rule:
  H0  recorded winner codes
  H0s recorded codes, header orientation swapped (T_k vs CT_hdr)
  H1  flip recorded round 13 only (all matches)
  H2  bomb-corrected codes (bomb side replaces winner on bomb rounds)
  H3  bomb-corrected, and round 13 flipped where there is no bomb outcome (the claim's round-13 rule)
  H4  bomb-corrected, and rounds 13, 25, 28 flipped where there is no bomb outcome
ASSUMPTION: regulation rule MR12 (24 rounds, halftime after round 12); matches with labels above 24 are excluded.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(40)
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill"
           r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")

R = pl.read_parquet(SCR / "verify_D9_2_round.parquet").select(
    ["match_key", "round", "winner_team_code", "w_bomb", "build_num", "has_out", "mm"])
H = pl.scan_parquet(sorted(glob.glob(str(BASE / "header" / "dataframe_*")))).select(
    ["key", "t_starters_score_final", "ct_starters_score_final", "match_date"]).collect()
H = H.with_columns(pl.col("key").alias("match_key"), pl.col("match_date").str.slice(0, 10).alias("date_utc"))

pm = R.group_by("match_key").agg(pl.len().alias("n_lab"), pl.col("round").min().alias("lo"),
                                 pl.col("round").max().alias("hi"), pl.col("build_num").first()).join(
    H.select(["match_key", "t_starters_score_final", "ct_starters_score_final", "date_utc"]), on="match_key")
pm = pm.with_columns((pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("hdr_sum"))
complete = pm.filter((pl.col("hi") <= 24) & (pl.col("lo") == 1) & (pl.col("n_lab") == pl.col("hdr_sum")))
print("complete-coverage matches by build:", complete.group_by("build_num").len().sort("build_num").rows())
print("note: 9842 has lo=2 for every match (round 1 absent from round_end), so it cannot pass this filter")

S = R.join(complete.select("match_key"), on="match_key", how="inner").join(
    H.select(["match_key", "t_starters_score_final", "ct_starters_score_final", "date_utc"]), on="match_key")


def tc(winner_expr_col, flip_rounds=(), flip_nonbomb_rounds=()):
    """return (T_k, C_k) expressions per match after optional flips."""
    w = pl.col(winner_expr_col)
    if flip_rounds:
        w = pl.when(pl.col("round").is_in(list(flip_rounds))).then(
            pl.when(w == 2).then(3).otherwise(2)).otherwise(w)
    if flip_nonbomb_rounds:
        w = pl.when(pl.col("round").is_in(list(flip_nonbomb_rounds)) & (~pl.col("has_out"))).then(
            pl.when(w == 2).then(3).otherwise(2)).otherwise(w)
    return w


def agree(label, wexpr, swap=False):
    d = S.with_columns(wexpr.alias("w")).with_columns(
        pl.when((pl.col("round") <= 12) & (pl.col("w") == 2)).then(1).otherwise(0).alias("a"),
        pl.when((pl.col("round") >= 13) & (pl.col("w") == 3)).then(1).otherwise(0).alias("b"),
        pl.when((pl.col("round") <= 12) & (pl.col("w") == 3)).then(1).otherwise(0).alias("c"),
        pl.when((pl.col("round") >= 13) & (pl.col("w") == 2)).then(1).otherwise(0).alias("d"),
    ).group_by("match_key").agg(
        (pl.col("a") + pl.col("b")).sum().alias("T"), (pl.col("c") + pl.col("d")).sum().alias("C"),
        pl.col("t_starters_score_final").first().alias("Th"), pl.col("ct_starters_score_final").first().alias("CTh"),
        pl.col("date_utc").first())
    if swap:
        ok = (pl.col("T") == pl.col("CTh")) & (pl.col("C") == pl.col("Th"))
    else:
        ok = (pl.col("T") == pl.col("Th")) & (pl.col("C") == pl.col("CTh"))
    d = d.with_columns(ok.alias("ok"))
    tot = d.group_by("build_num" if "build_num" in d.columns else "date_utc").agg(pl.len().alias("n"),
                                                                                   pl.col("ok").sum().alias("ok"))
    n = d.height
    k = int(d["ok"].sum())
    print(f"{label:55s} agree {k}/{n} = {k/n:.4f}")
    return d


print("\n== build 9961 complete matches: header-score agreement under each hypothesis")
agree("H0 recorded codes", pl.col("winner_team_code"))
agree("H0s recorded codes, orientation swapped", pl.col("winner_team_code"), swap=True)
agree("H1 flip recorded round 13 only (all matches)", tc("winner_team_code", flip_rounds=(13,)))
agree("H2 bomb-corrected codes", pl.col("w_bomb"))
agree("H3 bomb-corrected + flip r13 where no bomb", tc("w_bomb", flip_nonbomb_rounds=(13,)))
agree("H4 bomb-corrected + flip r13,25,28 where no bomb", tc("w_bomb", flip_nonbomb_rounds=(13, 25, 28)))

# split by whether the match has a round-13 bomb outcome
S13 = S.filter(pl.col("round") == 13).select(["match_key", "has_out", "mm"]).unique()
print("\nround 13 of complete 9961 matches: has bomb outcome count:",
      S13.group_by("has_out").len().sort("has_out").rows())
hasout13 = S13.filter(pl.col("has_out")).select("match_key")
nob13 = S13.filter(~pl.col("has_out")).select("match_key")
print("matches with a round-13 bomb outcome:", hasout13.height, "| without:", nob13.height)
for name, keys in [("with r13 bomb", hasout13), ("without r13 bomb", nob13)]:
    print(f"-- subset {name}")
    Ssub = S.join(keys, on="match_key", how="inner")
    Sbak = S
    S = Ssub
    agree("   H0 recorded", pl.col("winner_team_code"))
    agree("   H1 flip recorded r13", tc("winner_team_code", flip_rounds=(13,)))
    agree("   H2 bomb-corrected", pl.col("w_bomb"))
    agree("   H3 bomb-corrected + flip r13 where no bomb", tc("w_bomb", flip_nonbomb_rounds=(13,)))
    S = Sbak

# orientation/size of residuals under H0 for the complete sample
d0 = agree("H0 recorded (again, for residual table)", pl.col("winner_team_code"))
print("H0 residual table (T_k - Th, C_k - CTh):")
print(d0.with_columns((pl.col("T") - pl.col("Th")).alias("dT"), (pl.col("C") - pl.col("CTh")).alias("dC"))
      .group_by(["dT", "dC"]).len().sort("len", descending=True).head(12))
print("\nDONE scoretest")
