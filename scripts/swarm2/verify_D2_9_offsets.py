"""verify D2-9, alternative explanation test: is the 2024 round_start channel row for round r the start of
round r (as in 2026) or something else? Builds one per-round timeline table per tome from round channels
and round_state event rows (earliest tick per key), then reports offsets in seconds (tick / 64, measured by
scripts/swarm2/verify_D2_9_check.py S3). Lazy scans with projection; one tome at a time.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_offsets.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
TICK_RATE = 64  # ASSUMPTION confirmed by data: tick/second == 64 within every match (S3 of verify_D2_9_check.py)
QS = [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]


def parts(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))
    assert len(fd) == 1, (label, ch, fd)
    p = sorted(glob.glob(fd[0] + "/dataframe_*"))
    assert p, (label, ch)
    return p


def read(label, ch, cols):
    return pl.scan_parquet(parts(label, ch)).select(cols).collect()


def first_tick(df, name):
    d = df.sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True)
    return d.select(["match_key", "round", pl.col("tick").alias(name)])


def qsec(s):
    s = s.drop_nulls()
    if s.len() == 0:
        return "(empty)"
    return "  ".join(f"q{int(q * 100):03d}={s.quantile(q, interpolation='nearest') / TICK_RATE:,.2f}" for q in QS)


def build(label):
    rst = read(label, "round_state", ["match_key", "round", "tick", "event_type"])
    ev_names = {"round_freeze_end": "F", "round_officially_ended": "O", "round_prestart": "P",
                "round_start": "RSS", "round_end": "REV"}
    parts_ = [first_tick(read(label, "round_start", ["match_key", "round", "tick"]), "S"),
              first_tick(read(label, "round_end", ["match_key", "round", "tick"]), "E")]
    for ev, nm in ev_names.items():
        sub = rst.filter(pl.col("event_type") == ev).select(["match_key", "round", "tick"])
        if sub.height:
            parts_.append(first_tick(sub, nm))
    t = parts_[0]
    for p in parts_[1:]:
        t = t.join(p, on=["match_key", "round"], how="full", coalesce=True)
    t = t.sort(["match_key", "round"])
    t = t.with_columns(pl.col("round").shift(-1).over("match_key").alias("nr"),
                       pl.col("F").shift(-1).over("match_key").alias("Fn"),
                       pl.col("S").shift(-1).over("match_key").alias("Sn"),
                       pl.col("E").shift(-1).over("match_key").alias("En"))
    return t


def report(label, t):
    print(f"\n== [{label}] per-round timeline (seconds = tick / {TICK_RATE})")
    keys = {c: t.filter(pl.col(c).is_not_null()).height for c in ["S", "E", "F", "O", "P", "RSS", "REV"] if c in t.columns}
    print("   keys with each field:", keys, f"; total (match,round) keys: {t.height:,}")
    S_only = t.filter(pl.col("S").is_not_null() & pl.col("E").is_null())
    E_only = t.filter(pl.col("E").is_not_null() & pl.col("S").is_null())
    print(f"   round_start without round_end: {S_only.height:,}; round_end without round_start: {E_only.height:,}")
    print("   round_start-only rounds (by round, first 10):", S_only.group_by("round").agg(pl.len().alias("n"))
          .sort("round").head(10).rows())
    print("   round_end-only rounds (by round, first 10):", E_only.group_by("round").agg(pl.len().alias("n"))
          .sort("round").head(10).rows())
    nb = t.filter(pl.col("nr") == pl.col("round") + 1)
    print(f"   consecutive-round pairs: {nb.height:,}")
    # (name, expression, needs_next_round)
    comps = [
        ("S(r) - E(r)", pl.col("S") - pl.col("E"), False),
        ("S(r) - F(r)  [2026 expect ~ -20 s]", pl.col("S") - pl.col("F"), False),
        ("S(r) - F(r+1)", pl.col("S") - pl.col("Fn"), True),
        ("F(r) - E(r)  [window; must be < 0]", pl.col("F") - pl.col("E"), False),
        ("F(r+1) - E(r)  [gap across round boundary]", pl.col("Fn") - pl.col("E"), True),
        ("S(r+1) - E(r)  [if S were next round start]", pl.col("Sn") - pl.col("E"), True),
        ("E(r+1) - S(r) [should be > 0]", pl.col("En") - pl.col("S"), True),
    ]
    if "P" in t.columns:
        comps.append(("P(r) - S(r)", pl.col("P") - pl.col("S"), False))
    if "O" in t.columns:
        comps.append(("O(r) - E(r)  [officially ended after end]", pl.col("O") - pl.col("E"), False))
    for nm, expr, needs_next in comps:
        src = nb if needs_next else t
        d = src.select(expr.alias("d")).get_column("d")
        d_nonnull = d.drop_nulls()
        pos = int((d_nonnull > 0).sum())
        neg = int((d_nonnull < 0).sum())
        zer = int((d_nonnull == 0).sum())
        print(f"   {nm}: n={d_nonnull.len():,} >0:{pos:,} <0:{neg:,} =0:{zer:,}")
        print(f"      {qsec(d)}")


def main():
    print("polars", pl.__version__)
    for label in ["2024", "2026"]:
        t = build(label)
        report(label, t)
        if label == "2024":
            # exact-equality checks between channels and state events (are they the same clock event?)
            for a, b in [("S", "P"), ("E", "REV"), ("E", "O")]:
                if a in t.columns and b in t.columns:
                    x = t.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
                    dd = x.select((pl.col(a) - pl.col(b)).alias("d")).get_column("d")
                    print(f"   [{label}] {a} - {b} (ticks): n={dd.len():,} zero={int((dd == 0).sum()):,} "
                          f"nonzero={int((dd != 0).sum()):,}")


if __name__ == "__main__":
    main()
