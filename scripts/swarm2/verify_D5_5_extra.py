"""verify D5-5, step 3: controls and diagnostics that the first check did not settle.

  A. 2026 control, with the player-count and join problem diagnosed first.
  B. 2024 completeness (round_end rows vs header final score sum) by day group.
  C. Header-final-score test restricted to regulation matches (max round <= 24), so the OT side-swap
     predicate is not used. ASSUMPTION: t_starters_score_final = final score of the team that started on T.
  D. Does a round-9 chain disagreement (late, 9.1%) coincide with the round-13 disagreement in the same match?
  E. Round_end tick minus bomb outcome tick, by day group (timing signature of the late batch).

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D5_5_extra.py
"""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verify_D5_5_check as base  # noqa: E402  (re-runs the check module's prints; kept for shared builders)

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_fmt_str_lengths(70)

J24, J26 = base.J24, base.J26
RE24, RE26 = base.RE24, base.RE26
HD24, HD26 = base.HD24, base.HD26
aff_keys = base.aff_keys

print("\n\n######## EXTRA CHECKS ########")

# ---------- A. 2026 control diagnosis ----------
print("\n== A. 2026 player_count distribution (round_end rows) ==")
print(RE26.group_by("pc").len().sort("len", descending=True).head(10))
print("2026 outcome rows (any pc):", J26.filter(pl.col("bside").is_not_null()).height)
print("2026 outcome rows by pc:")
print(J26.filter(pl.col("bside").is_not_null()).group_by("pc").len().sort("pc"))
print("2026 round_end rounds with pc != 20 share:",
      round(RE26.filter(pl.col("pc") != 20).height / RE26.height, 4))
print("2024 round_end pc distribution:")
print(RE24.group_by("pc").len().sort("len", descending=True).head(6))

# re-run the mismatch metric on 2026 with whatever pc value the 2026 tome actually uses
top_pc = RE26.group_by("pc").len().sort("len", descending=True)["pc"][0]
x26 = J26.filter((pl.col("pc") == top_pc) & pl.col("bside").is_not_null() & pl.col("rw").is_not_null())
print(f"\n2026 outcome rounds with pc == {top_pc}: {x26.height}, mismatches (bomb side != rw): {int(x26['bmm'].sum())}")
x26_any = J26.filter(pl.col("bside").is_not_null() & pl.col("rw").is_not_null())
print(f"2026 outcome rounds, any pc: {x26_any.height}, mismatches: {int(x26_any['bmm'].sum())}")

# 2026 chain vs round_end at L13 and other labels (any pc, since pc is not 20 for most rows)
print("\n2026 chain vs round_end disagreement by label (all pc, nonzero chain, boundary labels excluded):")
c26 = (J26.filter(~pl.col("round").is_in(base.BOUNDARY) & (pl.col("chain") != 0) & pl.col("rw").is_not_null())
       .group_by("round").agg(pl.len().alias("n"), (pl.col("cmp") == "disagree").sum().alias("disagree"))
       .with_columns((pl.col("disagree") / pl.col("n")).round(4).alias("rate")).sort("round"))
print(c26.filter(pl.col("round").is_in([9, 13, 14, 16, 20, 25, 26, 28])))
print("2026 total disagreements over all labels:", int(c26["disagree"].sum()), " of", int(c26["n"].sum()))

# ---------- B. completeness by day group ----------
print("\n== B. 2024 completeness (round_end rows vs header final sum) by day ==")
Hd = HD24.with_columns(total=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"))
nr = RE24.group_by("match_key").agg(pl.len().alias("n_rounds"), pl.col("round").max().alias("max_round"))
cb = Hd.join(nr, on="match_key", how="left").with_columns(
    diff=pl.col("n_rounds") - pl.col("total"),
    complete=pl.col("n_rounds") == pl.col("total"))
cb = cb.with_columns(grp=pl.when(pl.col("day").is_in(base.LATE)).then(pl.lit("late"))
                     .when(pl.col("day").is_in(base.EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
print(cb.group_by("grp").agg(pl.len().alias("matches"), pl.col("complete").sum().alias("complete"),
                             (pl.col("diff") < 0).sum().alias("rounds_short"),
                             (pl.col("diff") > 0).sum().alias("rounds_over"),
                             pl.col("total").median().alias("median_total")).sort("grp"))
print("complete rate by day (2024):")
print(cb.group_by("day").agg(pl.len().alias("n"), pl.col("complete").mean().round(3).alias("complete_rate"),
                             pl.col("diff").median().alias("median_diff")).sort("day"))
print("the claim's worked example (05d51edb): n_rounds, total, max_round:")
print(cb.filter(pl.col("match_key") == base.EXAMPLE).select(["match_key", "n_rounds", "total", "max_round", "diff"]))

# ---------- C. header test on regulation-only matches ----------
print("\n== C. header final-score test, regulation-only matches (max round <= 24, complete) ==")


def reg_test(year, RE1, J, hd):
    c13 = J.filter(pl.col("round") == 13).select(["match_key", pl.col("chain").alias("chain13")])
    R = RE1.join(c13, on="match_key", how="left").with_columns(
        w_fix=pl.when((pl.col("round") == 13) & (pl.col("chain13") != 0)).then(pl.col("chain13"))
        .otherwise(pl.col("rw")))
    R = R.with_columns(a_on_t=pl.col("round") <= 12)
    R = R.with_columns(
        a_rw=((pl.col("a_on_t") & (pl.col("rw") == 2)) | (~pl.col("a_on_t") & (pl.col("rw") == 3))).cast(pl.Int32),
        a_fix=((pl.col("a_on_t") & (pl.col("w_fix") == 2)) | (~pl.col("a_on_t") & (pl.col("w_fix") == 3))).cast(pl.Int32))
    M = R.group_by("match_key").agg(pl.len().alias("n"), pl.col("round").max().alias("max_round"),
                                    pl.col("a_rw").sum().alias("A_rw"), pl.col("a_fix").sum().alias("A_fix"))
    M = M.join(hd.select(["match_key", "day", "t_starters_score_final", "ct_starters_score_final"]),
               on="match_key", how="left")
    M = M.with_columns(total=pl.col("t_starters_score_final") + pl.col("ct_starters_score_final"),
                       aff=pl.col("match_key").is_in(aff_keys))
    M = M.filter((pl.col("max_round") <= 24) & (pl.col("n") == pl.col("total")))
    M = M.with_columns(eq_t_rw=pl.col("A_rw") == pl.col("t_starters_score_final"),
                       eq_t_fix=pl.col("A_fix") == pl.col("t_starters_score_final"),
                       eq_c_rw=pl.col("A_rw") == pl.col("ct_starters_score_final"),
                       eq_c_fix=pl.col("A_fix") == pl.col("ct_starters_score_final"))
    M = M.with_columns(grp=pl.when(pl.col("day").is_in(base.LATE)).then(pl.lit("late"))
                       .when(pl.col("day").is_in(base.EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
    print(f"\n{year} regulation complete matches: {M.height}")
    print(M.group_by(["grp", "aff"]).agg(pl.len().alias("n"),
                                         pl.col("eq_t_rw").sum().alias("A_rw==t"),
                                         pl.col("eq_c_rw").sum().alias("A_rw==c"),
                                         pl.col("eq_t_fix").sum().alias("A_fix==t"),
                                         pl.col("eq_c_fix").sum().alias("A_fix==c")).sort(["grp", "aff"]))
    return M


R24 = reg_test("2024", RE24, J24, HD24)
R26 = reg_test("2026", RE26, J26, HD26)

# ---------- D. label-9 disagreement vs label-13 disagreement ----------
print("\n== D. late-group label 9 disagreements: do they co-occur with label-13 disagreements in the same match? ==")
d9 = J24.filter((pl.col("round") == 9) & (pl.col("pc") == 20) & (pl.col("chain") != 0) & pl.col("rw").is_not_null()
                & (pl.col("grp") == "late")).select(["match_key", "day", (pl.col("cmp") == "disagree").alias("dis9")])
d13 = J24.filter((pl.col("round") == 13) & (pl.col("pc") == 20) & (pl.col("chain") != 0) & pl.col("rw").is_not_null()
                 & (pl.col("grp") == "late")).select(["match_key", (pl.col("cmp") == "disagree").alias("dis13")])
co = d9.join(d13, on="match_key", how="inner")
print(co.group_by(["dis9", "dis13"]).len().sort(["dis9", "dis13"]))
print("label-9 disagreements by day:")
print(d9.group_by("day").agg(pl.len().alias("n"), pl.col("dis9").sum().alias("dis9")).sort("day"))

# ---------- E. timing offset by group (round_end - bomb outcome) ----------
print("\n== E. round_end tick minus bomb outcome tick (outcome rounds, pc20): share at exactly 429 / exactly 0 ==")
O = J24.filter((pl.col("pc") == 20) & pl.col("bside").is_not_null() & pl.col("rw").is_not_null()).with_columns(
    off=pl.col("end_tick") - pl.col("out_tick"))
print(O.group_by("grp").agg(pl.len().alias("n"), (pl.col("off") == 429).mean().round(3).alias("share_429"),
                            (pl.col("off") == 0).mean().round(3).alias("share_0"),
                            pl.col("off").median().alias("median")).sort("grp"))
Oi = O.filter(pl.col("grp") == "late")
print("late: share at 429 by label class:")
print(Oi.with_columns(lc2=pl.when(pl.col("round") == 13).then(pl.lit("L13")).otherwise(pl.lit("other")))
      .group_by("lc2").agg(pl.len().alias("n"), (pl.col("off") == 429).mean().round(3).alias("share_429")))
print("tick rate by day group (header):")
print(HD24.with_columns(grp=pl.when(pl.col("day").is_in(base.LATE)).then(pl.lit("late"))
                        .when(pl.col("day").is_in(base.EARLY)).then(pl.lit("early")).otherwise(pl.lit("other")))
      .group_by(["grp", "tick_rate"]).len().sort(["grp", "tick_rate"]) if "tick_rate" in HD24.columns else
      "tick_rate not extracted")

print("\nEXTRA DONE")
