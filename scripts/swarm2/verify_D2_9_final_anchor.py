"""verify D2-9, final anchor counts. For each tome and day group, the share of rounds where
  round_end (E) minus last player_death, or E minus bomb_exploded, equals 428 or 429 ticks (the signature seen
  in 2024 days 13-25), and where round_officially_ended (O) minus last player_death equals 448 ticks (7.0 s).
Day groups use the match_key day number. NOTE: for 2026 the day numbers are 2026-09-18..10-08 (not 2024).
Lazy scans, projection only.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_9_final_anchor.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(30)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}


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


def grp_expr(label):
    d = pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).cast(pl.Int64)
    if label == "2024":
        return (pl.when(d <= 7).then(pl.lit("2024 d01-07"))
                .when(d <= 12).then(pl.lit("2024 d08-12"))
                .when(d <= 25).then(pl.lit("2024 d13-25"))
                .otherwise(pl.lit("2024 d26+")).alias("grp"))
    return pl.lit("2026 all").alias("grp")


def main():
    print("polars", pl.__version__)
    for label in ["2024", "2026"]:
        rst = read(label, "round_state", ["match_key", "round", "tick", "event_type"])
        E = first_tick(read(label, "round_end", ["match_key", "round", "tick"]), "E")
        O = first_tick(rst.filter(pl.col("event_type") == "round_officially_ended").select(["match_key", "round", "tick"]), "O")
        X = first_tick(rst.filter(pl.col("event_type") == "bomb_exploded").select(["match_key", "round", "tick"]), "X")
        D = read(label, "player_death", ["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
            pl.col("tick").max().alias("LD"))
        base = E.join(D, on=["match_key", "round"], how="inner").join(O, on=["match_key", "round"], how="left")
        base = base.with_columns(grp_expr(label))
        base = base.with_columns(((pl.col("E") - pl.col("LD")).is_in([428, 429])).alias("E_LD_428_429"),
                                 ((pl.col("E") - pl.col("LD")) == 0).alias("E_LD_0"),
                                 ((pl.col("O") - pl.col("LD")) == 448).alias("O_LD_448"),
                                 ((pl.col("O") - pl.col("LD")).is_between(440, 456)).alias("O_LD_440_456"))
        print(f"\n== [{label}] rounds with round_end and last death: {base.height:,}")
        summ = base.group_by("grp").agg(
            pl.len().alias("n"),
            pl.col("E_LD_428_429").mean().round(4).alias("share_E_minus_LD_428_429"),
            pl.col("E_LD_0").mean().round(4).alias("share_E_minus_LD_0"),
            pl.col("O").is_not_null().sum().alias("n_with_O"),
            pl.col("O_LD_448").mean().round(4).alias("share_O_minus_LD_448"),
            pl.col("O_LD_440_456").mean().round(4).alias("share_O_minus_LD_440_456")).sort("grp")
        print(summ)
        bx = E.join(X, on=["match_key", "round"], how="inner").with_columns(grp_expr(label))
        bx = bx.with_columns(((pl.col("E") - pl.col("X")).is_in([428, 429])).alias("E_X_428_429"),
                             ((pl.col("E") - pl.col("X")) == 0).alias("E_X_0"))
        bsum = bx.group_by("grp").agg(
            pl.len().alias("n_bomb_rounds"),
            pl.col("E_X_428_429").mean().round(4).alias("share_E_minus_X_428_429"),
            pl.col("E_X_0").mean().round(4).alias("share_E_minus_X_0")).sort("grp")
        print(f"-- [{label}] bomb_exploded rounds: E - X")
        print(bsum)


if __name__ == "__main__":
    main()
