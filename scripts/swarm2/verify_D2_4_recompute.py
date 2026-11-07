"""Verifier D2-4: recompute the claimed counts from raw channels and split circular from non-circular checks.

Claim under test (D2-4): side-attached truth T agrees with in-window bomb outcomes and eliminations on the rows
where winner_team_code (R) does not; 2024 bomb==T 14,525 of 14,525; bomb==R 14,032; 493 flipped bomb rows all
agree with T; elim==T 1,891 of 1,914 flipped rows; score delta == bomb 12,920 of 12,920; 2026 bomb==T 6,478 of 6,478.

Circularity: T = coalesce(s_score, s_bomb, s_elim). Where T came from s_bomb, "bomb == T" is true by construction,
and where T came from s_elim, "elim == T" is true by construction. Those rows are reported separately below.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_4_recompute.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import polars as pl  # noqa: E402

from verify_D2_4_build import bin_expr, build  # noqa: E402

pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_fmt_str_lengths(30)


def hdr(s):
    print("\n" + "=" * 10 + " " + s + " " + "=" * 10)


def n(df):
    return df.height


def eq(df, a, b):
    return df.filter(pl.col(a) == pl.col(b)).height


b24 = build("2024").with_columns(bin_expr())
b26 = build("2026").with_columns(bin_expr())

# ---------------------------------------------------------------- 0. build coverage
for name, b in [("2024", b24), ("2026", b26)]:
    hdr(f"{name}: build coverage")
    tot = b.height
    print(f"(match, round) rows after claim filter: {tot}")
    print("with freeze-end fields (t0):", n(b.filter(pl.col("t0").is_not_null())))
    print("with sx:", n(b.filter(pl.col("sx").is_not_null())))
    print("with T:", n(b.filter(pl.col("T").is_not_null())), f"({n(b.filter(pl.col('T').is_not_null()))/tot:.4f})")
    print("T from s_score:", n(b.filter(pl.col("s_score").is_not_null())))
    print("T from s_bomb only (no s_score):", n(b.filter(pl.col("s_score").is_null() & pl.col("s_bomb").is_not_null())))
    print("T from s_elim only (no s_score, no s_bomb):",
          n(b.filter(pl.col("s_score").is_null() & pl.col("s_bomb").is_null() & pl.col("s_elim").is_not_null())))
    print("flipped (R != T):", n(b.filter(pl.col("flip"))))
    print("rounds with unknown death membership (>0 deaths not in round-F roster):",
          n(b.filter(pl.col("n_unknown_member") > 0)))

# ---------------------------------------------------------------- 1. 2024 claimed counts
k = b24.filter(pl.col("T").is_not_null())
bb = k.filter(pl.col("s_bomb").is_not_null())
hdr("1. 2024 bomb outcome vs T and R (claim: 14,525 / 14,525 / 14,032 / 493)")
print("bomb rows with T:", n(bb), "  bomb==T:", eq(bb, "s_bomb", "T"), "  bomb==R:", eq(bb, "s_bomb", "R"))
fb = bb.filter(pl.col("flip"))
print("flipped bomb rows:", n(fb), "  of which bomb==T:", eq(fb, "s_bomb", "T"))
print("bomb rows with T from s_bomb only (agreement is by construction):", n(bb.filter(pl.col("s_score").is_null())))
bs_ = bb.filter(pl.col("s_score").is_not_null())
print("bomb rows with T from s_score (non-circular):", n(bs_), "  s_score==s_bomb:", eq(bs_, "s_score", "s_bomb"))
fbs = bs_.filter(pl.col("flip"))
print("  flipped among non-circular bomb rows:", n(fbs), "  s_score==s_bomb:", eq(fbs, "s_score", "s_bomb"),
      "  s_score==R:", eq(fbs, "s_score", "R"))
print("flipped bomb rows whose T comes from s_bomb only:", n(fb.filter(pl.col("s_score").is_null())))

hdr("1b. bomb vs elimination (2024)")
be = bb.filter(pl.col("s_elim").is_not_null())
print("bomb rows with elim:", n(be), "  s_elim==s_bomb:", eq(be, "s_elim", "s_bomb"),
      "  disagree:", n(be) - eq(be, "s_elim", "s_bomb"))
bes = bb.filter(pl.col("s_elim").is_not_null() & pl.col("s_score").is_not_null())
print("  of which s_score known (non-circular):", n(bes), "  s_elim==s_score:", eq(bes, "s_elim", "s_score"))
print("  elim disagreement by round bin (bomb rows, elim known):")
print(be.with_columns((pl.col("s_elim") != pl.col("s_bomb")).alias("dis")).group_by("bin").agg(
    pl.len().alias("n"), pl.col("dis").sum().alias("disagree")).sort("bin"))

hdr("2. 2024 score delta vs bomb (claim: 12,920 / 12,920)")
sb = b24.filter(pl.col("s_score").is_not_null() & pl.col("s_bomb").is_not_null())
print("rows with s_score and s_bomb:", n(sb), "  equal:", eq(sb, "s_score", "s_bomb"))
print("  by bin, n / equal:")
print(sb.group_by("bin").agg(pl.len().alias("n"), (pl.col("s_score") == pl.col("s_bomb")).sum().alias("equal")).sort("bin"))
print("  swap_now rows (player side changed at round r):",
      n(sb.filter(pl.col("swap_now"))), "  equal:", eq(sb.filter(pl.col("swap_now")), "s_score", "s_bomb"))

hdr("3. 2024 elimination vs T on flipped rows (claim: 1,891 of 1,914; elim==R 23)")
fl = k.filter(pl.col("flip"))
fe_ = fl.filter(pl.col("s_elim").is_not_null())
print("flipped rows with elim:", n(fe_), "  elim==T:", eq(fe_, "s_elim", "T"), "  elim==R:", eq(fe_, "s_elim", "R"))
fse = fl.filter(pl.col("s_score").is_not_null() & pl.col("s_elim").is_not_null())
print("flipped rows with T from s_score and elim (non-circular):", n(fse),
      "  elim==T:", eq(fse, "s_elim", "T"), "  elim==R:", eq(fse, "s_elim", "R"))
un = k.filter((~pl.col("flip")) & pl.col("s_score").is_not_null() & pl.col("s_elim").is_not_null())
print("baseline: unflipped rows with T from s_score and elim:", n(un), "  elim==T:", eq(un, "s_elim", "T"),
      f"  rate {eq(un, 's_elim', 'T') / max(n(un), 1):.4f}")
print("flipped non-circular rate:", f"{eq(fse, 's_elim', 'T') / max(n(fse), 1):.4f}")

# ---------------------------------------------------------------- 4. R-independent truth (no player sides beyond gate)
hdr("4. sx-free side-field truth (round_state delta, only where player sides do not change r -> r+1)")
sf = b24.filter(pl.col("s_side").is_not_null() & pl.col("R").is_not_null())
print("2024 rows with s_side:", n(sf))
print(sf.group_by("bin").agg(
    pl.len().alias("n"),
    (pl.col("R") != pl.col("s_side")).sum().alias("R!=side"),
    (pl.col("R") != pl.col("s_side")).mean().round(4).alias("rate")).sort("bin"))
ss = sf.filter(pl.col("s_score").is_not_null())
print("rows with both s_side and claim s_score:", n(ss), "  equal:", eq(ss, "s_side", "s_score"))
sbb = sf.filter(pl.col("s_bomb").is_not_null())
print("rows with s_side and bomb:", n(sbb), "  s_side==s_bomb:", eq(sbb, "s_side", "s_bomb"),
      "  R==s_bomb:", eq(sbb, "R", "s_bomb"))
print("  bomb rows with s_side by bin, n / s_side==bomb / R==bomb:")
print(sbb.group_by("bin").agg(pl.len().alias("n"), (pl.col("s_side") == pl.col("s_bomb")).sum().alias("side==bomb"),
                              (pl.col("R") == pl.col("s_bomb")).sum().alias("R==bomb")).sort("bin"))
r13 = sf.filter(pl.col("round") == 13)
print("round 13 with s_side:", n(r13), " R!=s_side:", n(r13.filter(pl.col("R") != pl.col("s_side"))))
print("round 13 by day, n / R!=s_side:")
print(r13.group_by("day").agg(pl.len().alias("n"), (pl.col("R") != pl.col("s_side")).sum().alias("R!=side")).sort("day"))

# ---------------------------------------------------------------- 5. planter side (independent of scores)
hdr("5. planter side (bomb_state.player_team_code at bomb_planted) by flip status")
pl_all = b24.filter(pl.col("planter").is_not_null())
print("planted rounds (2024):", n(pl_all), " planter side counts:")
print(pl_all.group_by("planter").len().sort("planter"))
print("planted round 13 by planter side:")
print(pl_all.filter(pl.col("round") == 13).group_by("planter").len().sort("planter"))
pb = fb.filter(pl.col("planter").is_not_null())
print("flipped bomb rows with planter known:", n(pb), " planter side counts:")
print(pb.group_by(["planter", "s_bomb"]).len().sort(["planter", "s_bomb"]))

# ---------------------------------------------------------------- 6. membership-null deaths and the claim's when(isX) fallthrough
hdr("6. death membership")
un_rounds = b24.filter(pl.col("n_unknown_member") > 0)
print("2024 rounds with >=1 death whose player is not in round-F roster:", n(un_rounds),
      " deaths:", int(un_rounds["n_unknown_member"].sum()), " flipped among them:", n(un_rounds.filter(pl.col("flip"))))

# ---------------------------------------------------------------- 7. 2026 control
hdr("7. 2026 control (claim: bomb==T 6,478 of 6,478; R==T everywhere with T)")
k6 = b26.filter(pl.col("T").is_not_null())
print("2026 rows with T:", n(k6), "  R==T:", eq(k6, "R", "T"), "  flipped:", n(k6.filter(pl.col("flip"))))
bb6 = k6.filter(pl.col("s_bomb").is_not_null())
print("2026 bomb rows with T:", n(bb6), "  bomb==T:", eq(bb6, "s_bomb", "T"), "  bomb==R:", eq(bb6, "s_bomb", "R"))
bs6 = bb6.filter(pl.col("s_score").is_not_null())
print("2026 bomb rows with T from s_score (non-circular):", n(bs6), "  s_score==s_bomb:", eq(bs6, "s_score", "s_bomb"))
sf6 = b26.filter(pl.col("s_side").is_not_null() & pl.col("R").is_not_null())
print("2026 rows with sx-free side truth:", n(sf6), "  R!=side:", n(sf6.filter(pl.col("R") != pl.col("s_side"))))
