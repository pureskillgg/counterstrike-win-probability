"""verify_D2_6 probe 2: independent check of the side-free truth S and of R, using the side-attached score change.

S_delta(r) = winner side from round_state score fields alone: CT score rose by 1 and T score did not (3),
or T rose by 1 and CT did not (2), between freeze_end(r) and freeze_end(r+1). Valid only when there is no swap
between r and r+1 (sx(r) == sx(r+1)), because a swap moves the team's score to the other field.
This uses neither bomb, elimination nor the recorded winner, so it is an independent third source.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_6_probe2.py
Prints only.
"""
import sys
from collections import Counter

import polars as pl

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import verify_D2_6_recompute as m  # noqa: E402  (import is safe: main() is guarded)

sys.stdout.reconfigure(encoding="utf-8")


def run(label):
    tab, info = m.build(label)
    nxt = tab.select([
        "match_key", (pl.col("round") - 1).alias("round"),
        pl.col("t0").alias("t1"), pl.col("ct0").alias("ct1"), pl.col("sx").alias("sx_next")])
    t = tab.join(nxt, on=["match_key", "round"], how="left")
    t = t.with_columns(
        (pl.col("ct1") - pl.col("ct0")).alias("dCT"), (pl.col("t1") - pl.col("t0")).alias("dT"))
    t = t.with_columns(
        pl.when((pl.col("sx") == pl.col("sx_next")) & (pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
        .when((pl.col("sx") == pl.col("sx_next")) & (pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("S_delta"))
    print("=" * 90)
    print(f"{label}: rounds {t.height}; S_delta defined {t.filter(pl.col('S_delta').is_not_null()).height}")
    both = t.filter(pl.col("S_delta").is_not_null() & pl.col("S").is_not_null())
    print(f"  S_delta vs S (bomb/elim), both known: n={both.height} equal={both.filter(pl.col('S_delta') == pl.col('S')).height}")
    bb = t.filter(pl.col("S_delta").is_not_null() & pl.col("S_bomb").is_not_null())
    print(f"  S_delta vs bomb outcome, both known: n={bb.height} equal={bb.filter(pl.col('S_delta') == pl.col('S_bomb')).height}")
    ee = t.filter(pl.col("S_delta").is_not_null() & pl.col("S_elim").is_not_null() & pl.col("S_bomb").is_null())
    print(f"  S_delta vs elim-only outcome: n={ee.height} equal={ee.filter(pl.col('S_delta') == pl.col('S_elim')).height}")
    rr = t.filter(pl.col("S_delta").is_not_null() & pl.col("R").is_not_null())
    print(f"  S_delta vs recorded R: n={rr.height} equal={rr.filter(pl.col('S_delta') == pl.col('R')).height} "
          f"differ={rr.filter(pl.col('S_delta') != pl.col('R')).height}")
    grp = (rr.with_columns(
        pl.when(pl.col("round") <= 12).then(pl.lit("01-12")).when(pl.col("round") == 13).then(pl.lit("13"))
        .when(pl.col("round") <= 24).then(pl.lit("14-24")).when(pl.col("round") == 25).then(pl.lit("25"))
        .when(pl.col("round") == 28).then(pl.lit("28")).otherwise(pl.lit("other")).alias("grp"),
        (pl.col("S_delta") != pl.col("R")).alias("flip"))
        .group_by("grp").agg(pl.len().alias("n"), pl.col("flip").sum().alias("R_ne_Sdelta")).sort("grp"))
    print("  R vs S_delta by round group:")
    for row in grp.iter_rows():
        print(f"     {row[0]:>6}: n={row[1]:>6} R!=S_delta={row[2]:>5} ({row[2] / row[1] * 100 if row[1] else 0:.2f}%)")
    r13 = t.filter(pl.col("round") == 13)
    f13 = r13.filter(pl.col("S_delta").is_not_null() & (pl.col("S_delta") != pl.col("R")))
    print(f"  round 13: S_delta defined {r13.filter(pl.col('S_delta').is_not_null()).height}; "
          f"R != S_delta {f13.height}; of these S (bomb/elim) == S_delta {f13.filter(pl.col('S') == pl.col('S_delta')).height}")
    # S disagreements with S_delta, where both known (where is the truth noisy?)
    dis = both.filter(pl.col("S_delta") != pl.col("S"))
    print(f"  disagreements S vs S_delta (both known): {dis.height}; by source: bomb "
          f"{dis.filter(pl.col('S_bomb').is_not_null()).height}, elim-only {dis.filter(pl.col('S_bomb').is_null()).height}")
    print(f"  disagreements sample (match, round, R, S, S_delta, S_bomb, S_elim, t_dead, ct_dead):")
    for row in dis.head(8).select(["match_key", "round", "R", "S", "S_delta", "S_bomb", "S_elim", "t_dead", "ct_dead"]).iter_rows():
        print("    ", row[0][-40:], row[1:])
    return t


def main():
    t24 = run("2024")
    t26 = run("2026")
    # 2024 round 13 flips: confirm via S_delta alone
    fl = t24.filter((pl.col("round") == 13) & pl.col("S_delta").is_not_null() & pl.col("R").is_not_null())
    print("=" * 90)
    print(f"2024 round 13 rows with S_delta and R: {fl.height}; R != S_delta: {fl.filter(pl.col('R') != pl.col('S_delta')).height}; "
          f"R == S_delta: {fl.filter(pl.col('R') == pl.col('S_delta')).height}")
    print(f"2026 round 13 rows with S_delta and R: {t26.filter((pl.col('round') == 13) & pl.col('S_delta').is_not_null()).height}; "
          f"R != S_delta: {t26.filter((pl.col('round') == 13) & pl.col('S_delta').is_not_null() & (pl.col('R') != pl.col('S_delta'))).height}")


if __name__ == "__main__":
    main()
