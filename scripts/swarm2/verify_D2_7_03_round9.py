"""verify_D2_7 step 3: check the 16-round (MR8) round-9 flags found in step 1, and whether 16-round
matches are in the cross_period kept set at all. Prints only; nothing written to disk.
"""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(220)

ROOT = Path(r"G:\pureskill\csds-cs2")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cp = load("cross_period", ROOT / "scripts" / "model" / "sufficiency" / "cross_period.py")
t1 = load("verify_D2_7_01_truth", ROOT / "scripts" / "swarm2" / "verify_D2_7_01_truth.py")


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        tab, _ = t1.build("2024", t1.TOME24, t1.SUF24)
        B = cp.run_2024()

    rs_ev = t1.read(t1.TOME24, t1.SUF24, "round_state", ["match_key", "round", "event_type"])
    half = (rs_ev.filter(pl.col("event_type") == "round_announce_last_round_half")
            .group_by("match_key").agg(pl.col("round").min().alias("half_round")))
    tab = (tab.join(half, on="match_key", how="left")
           .with_columns(pl.when(pl.col("half_round").is_not_null())
                         .then(pl.col("half_round") * 2 + 2).otherwise(24).alias("max_rounds")))

    kept = B["S"].select(["match_key", "round"]).unique()
    kept_mr = (kept.join(tab.select(["match_key", "round", "max_rounds"]), on=["match_key", "round"], how="left")
               .group_by("max_rounds").len().sort("max_rounds"))
    print("kept rounds (cross_period kept set) by max_rounds:")
    print(kept_mr)
    print("header matches by max_rounds (all 2024 round_end matches):")
    print(tab.select(["match_key", "max_rounds"]).unique().group_by("max_rounds").len().sort("max_rounds"))

    full = tab.filter(pl.col("player_count") == 20)
    kset = kept.with_columns(pl.lit(True).alias("in_kept"))
    full = full.join(kset, on=["match_key", "round"], how="left").with_columns(pl.col("in_kept").fill_null(False))

    mr16_r9 = full.filter((pl.col("max_rounds") == 16) & (pl.col("round") == 9))
    print(f"\nMR16 round-9 rows (player_count 20): {mr16_r9.height}; with score truth: "
          f"{mr16_r9.filter(pl.col('truth_ct').is_not_null()).height}; in kept set: "
          f"{mr16_r9.filter(pl.col('in_kept')).height}")
    t9 = mr16_r9.filter(pl.col("truth_ct").is_not_null())
    print(f"  flagged (score truth != R): {int(t9['flag'].sum())} of {t9.height} "
          f"({100 * t9['flag'].mean():.1f}%)")
    print(f"  flagged with bomb outcome: {t9.filter(pl.col('flag') == 1).filter(pl.col('bomb_ct').is_not_null()).height}; "
          f"bomb == score truth among those: "
          f"{t9.filter(pl.col('flag') == 1).filter(pl.col('bomb_ct').is_not_null()).filter(pl.col('bomb_ct') == pl.col('truth_ct')).height}")
    print(f"  flagged with elimination outcome: {t9.filter(pl.col('flag') == 1).filter(pl.col('elim_ct').is_not_null()).height}; "
          f"elim == score truth among those: "
          f"{t9.filter(pl.col('flag') == 1).filter(pl.col('elim_ct').is_not_null()).filter(pl.col('elim_ct') == pl.col('truth_ct')).height}")
    print(f"  flagged with neither bomb nor elimination: "
          f"{t9.filter(pl.col('flag') == 1).filter(pl.col('bomb_ct').is_null() & pl.col('elim_ct').is_null()).height}")
    print(f"  unflagged with bomb outcome: {t9.filter(pl.col('flag') == 0).filter(pl.col('bomb_ct').is_not_null()).height}; "
          f"bomb == R among those: "
          f"{t9.filter(pl.col('flag') == 0).filter(pl.col('bomb_ct').is_not_null()).filter(pl.col('bomb_ct') == pl.col('R_ct')).height}")

    print("\nflagged rounds in full 2024 set (player_count 20), by round, max_rounds and in-kept status:")
    fl = full.filter(pl.col("flag") == 1)
    print(fl.group_by(["max_rounds", "round", "in_kept"]).len().sort(["max_rounds", "round", "in_kept"]).head(40))

    print("\nround-9 MR16 flag rate by match day:")
    print(t9.group_by("day").agg(pl.len().alias("n"), pl.col("flag").sum().alias("flagged"))
          .sort("day"))


if __name__ == "__main__":
    main()
