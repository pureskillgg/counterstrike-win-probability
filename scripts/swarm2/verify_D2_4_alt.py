"""Verifier D2-4: strongest alternative explanations for the flipped rows.

A1  team-attached score fields (t_score follows team X, not the T side): would make the side-field truth wrong on
    rounds where team X stood on CT. Test: side-field truth vs bomb on rounds with sx == 3 (X on CT).
A2  lag-by-value: R(r) equals the true winner of round r-1. Test: R(13) vs T(12) on flipped round-13 rows.
A3  membership key: rebuild eliminations keyed on player_id_fixed instead of player_id.
A4  bomb-window coverage: late bomb events (after round_end) and the claimed 12,920 count.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_4_alt.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import polars as pl  # noqa: E402

from verify_D2_4_build import bin_expr, build, channel  # noqa: E402

pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)


def hdr(s):
    print("\n" + "=" * 10 + " " + s + " " + "=" * 10)


def eq(df, a, b):
    return df.filter(pl.col(a) == pl.col(b)).height


b24 = build("2024").with_columns(bin_expr())
b26 = build("2026").with_columns(bin_expr())

# ------------------------------------------------------------------ A1 team-attached alternative
hdr("A1 side-field truth vs in-window bomb, split by team X side (sx)")
for name, b in [("2024", b24), ("2026", b26)]:
    sbb = b.filter(pl.col("s_side").is_not_null() & pl.col("s_bomb").is_not_null())
    print(name, "rows with s_side and bomb:", sbb.height)
    print(sbb.group_by(["bin", "sx"]).agg(
        pl.len().alias("n"),
        (pl.col("s_side") == pl.col("s_bomb")).sum().alias("side==bomb"),
        (pl.col("R") == pl.col("s_bomb")).sum().alias("R==bomb")).sort(["bin", "sx"]))
x_on_ct = b24.filter(pl.col("s_side").is_not_null() & pl.col("s_bomb").is_not_null() & (pl.col("sx") == 3))
print("2024 rounds with X on CT and s_side and bomb:", x_on_ct.height,
      "  side==bomb:", eq(x_on_ct, "s_side", "s_bomb"),
      "  team-attached alt (side = sx if X won, field = team X) would give side != bomb on these rows if wrong")
# team-attached alternative: X's field is always the T field; winner side = sx if X won, else 5-sx
alt = b24.filter(pl.col("sx").is_not_null() & pl.col("s_bomb").is_not_null() & pl.col("t0").is_not_null() & pl.col("t1").is_not_null())
alt = alt.with_columns(
    (pl.col("t1") - pl.col("t0")).alias("dT_team"), (pl.col("ct1") - pl.col("ct0")).alias("dCT_team"))
alt = alt.with_columns(
    pl.when((pl.col("dT_team") == 1) & (pl.col("dCT_team") == 0)).then(pl.col("sx"))
    .when((pl.col("dCT_team") == 1) & (pl.col("dT_team") == 0)).then(5 - pl.col("sx"))
    .otherwise(None).cast(pl.Int64).alias("s_team_alt"))
alt = alt.filter(pl.col("s_team_alt").is_not_null())
print("2024 bomb rows with team-attached alt value:", alt.height,
      "  alt==bomb:", eq(alt, "s_team_alt", "s_bomb"),
      "  alt==bomb on X-on-CT rows:", eq(alt.filter(pl.col("sx") == 3), "s_team_alt", "s_bomb"),
      "of", alt.filter(pl.col("sx") == 3).height)
print("  side-field (claim) on the same rows, X-on-CT:", eq(alt.filter(pl.col("sx") == 3), "s_side", "s_bomb"),
      "(s_side may be null where a player swap occurs)")

# ------------------------------------------------------------------ A2 lag-by-value
hdr("A2 lag-by-value: R(13) vs winner of round 12")
prev = b24.select(["match_key", (pl.col("round") + 1).alias("round"),
                   pl.col("T").alias("T_prev"), pl.col("R").alias("R_prev")])
r13 = b24.filter(pl.col("round") == 13).join(prev, on=["match_key", "round"], how="left")
fl13 = r13.filter(pl.col("flip") & pl.col("T_prev").is_not_null())
print("flipped round-13 rows with a round-12 truth:", fl13.height)
print("  R13 == T12:", eq(fl13, "R", "T_prev"), "  R13 == R12:", eq(fl13, "R", "R_prev"),
      "  R12 == T12 (round-12 record correct):", eq(fl13, "R_prev", "T_prev"))
ok13 = r13.filter((~pl.col("flip")) & pl.col("T_prev").is_not_null())
print("unflipped round-13 rows with a round-12 truth:", ok13.height, "  R13 == T12:", eq(ok13, "R", "T_prev"))

# ------------------------------------------------------------------ A3 membership key
hdr("A3 eliminations keyed on player_id_fixed (2024)")
bf = build("2024", membership_key="player_id_fixed").with_columns(bin_expr())
for lab, b in [("player_id", b24), ("player_id_fixed", bf)]:
    k = b.filter(pl.col("T").is_not_null())
    fl = k.filter(pl.col("flip") & pl.col("s_score").is_not_null() & pl.col("s_elim").is_not_null())
    un = k.filter((~pl.col("flip")) & pl.col("s_score").is_not_null() & pl.col("s_elim").is_not_null())
    print(f"{lab:16s} flipped non-circular: {fl.height}, elim==T {eq(fl, 's_elim', 'T')}, "
          f"elim==R {eq(fl, 's_elim', 'R')}; unflipped baseline: {un.height}, elim==T {eq(un, 's_elim', 'T')} "
          f"({eq(un, 's_elim', 'T') / max(un.height, 1):.4f})")
    print(f"{'':16s} rounds with unknown membership: {b.filter(pl.col('n_unknown_member') > 0).height}, "
          f"elim known: {b.filter(pl.col('s_elim').is_not_null()).height}")

# ------------------------------------------------------------------ A4 bomb-window coverage
hdr("A4 bomb-window coverage (2024, pc20)")
b_all = b24
print("rounds with T:", b_all.filter(pl.col("T").is_not_null()).height)
print("rounds with s_score:", b_all.filter(pl.col("s_score").is_not_null()).height)
print("rounds with s_score and in-window s_bomb:", b_all.filter(pl.col("s_score").is_not_null() & pl.col("s_bomb").is_not_null()).height,
      "  equal:", eq(b_all.filter(pl.col("s_score").is_not_null() & pl.col("s_bomb").is_not_null()), "s_score", "s_bomb"))
print("T source counts: s_score", b_all.filter(pl.col("s_score").is_not_null()).height,
      "| bomb-only", b_all.filter(pl.col("s_score").is_null() & pl.col("s_bomb").is_not_null()).height,
      "| elim-only", b_all.filter(pl.col("s_score").is_null() & pl.col("s_bomb").is_null() & pl.col("s_elim").is_not_null()).height)
print("elim-only rounds flipped:", b_all.filter(pl.col("s_score").is_null() & pl.col("s_bomb").is_null()
                                                & pl.col("s_elim").is_not_null() & pl.col("flip")).height)
print("bomb rows with s_elim: s_elim==s_bomb:", eq(b_all.filter(pl.col("s_elim").is_not_null() & pl.col("s_bomb").is_not_null()), "s_elim", "s_bomb"),
      "of", b_all.filter(pl.col("s_elim").is_not_null() & pl.col("s_bomb").is_not_null()).height)

# ------------------------------------------------------------------ fix-relevant coverage by round bin
hdr("coverage of the proposed 2024 label set (score-only vs score or bomb vs with elim-only)")
tot = b24.height
for name, m in [("score only", pl.col("s_score").is_not_null()),
                ("score or in-window bomb", pl.col("s_score").is_not_null() | pl.col("s_bomb").is_not_null()),
                ("score or bomb or elim (claim)", pl.col("T").is_not_null())]:
    c = b24.filter(m)
    print(f"{name:32s} {c.height:6d} of {tot} = {c.height / tot:.4f}; flipped R!=T among them: {c.filter(pl.col('flip')).height}")
