"""verify_D2_7 step 4: reconcile my flagged set with D2's intermediate table (read-only).

Reads logs/swarm2/D2_truth_2024.parquet (D2's output, read only) and compares its flip flag with the
flag from verify_D2_7_01_truth.build (my score-only side truth). Prints only; nothing written.
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


t1 = load("verify_D2_7_01_truth", ROOT / "scripts" / "swarm2" / "verify_D2_7_01_truth.py")


def main():
    with contextlib.redirect_stdout(io.StringIO()):
        tab, _ = t1.build("2024", t1.TOME24, t1.SUF24)
    mine = (tab.filter(pl.col("player_count") == 20)
            .select(["match_key", "round", "truth_ct", "R_ct", "flag", "grp", "bomb_ct", "elim_ct"]))

    d2 = pl.read_parquet(ROOT / "logs" / "swarm2" / "D2_truth_2024.parquet")
    print(f"D2 table rows: {d2.height}; T null: {d2.filter(pl.col('T').is_null()).height}; "
          f"flip True: {d2.filter(pl.col('flip') == True).height}")  # noqa: E712
    d2s = d2.select(["match_key", "round", pl.col("T").alias("T_d2"), pl.col("R").alias("R_d2"),
                     pl.col("flip").alias("flip_d2")])
    j = mine.join(d2s, on=["match_key", "round"], how="full", coalesce=True)
    print(f"joined rows: {j.height}")

    both_t = j.filter(pl.col("truth_ct").is_not_null() & pl.col("T_d2").is_not_null())
    print(f"rounds with truth in both: {both_t.height}")
    print(f"  truth agreement (mine CT-win == D2 T==3): "
          f"{int((both_t['truth_ct'] == (both_t['T_d2'] == 3).cast(pl.Int64)).sum())} of {both_t.height}")
    print(f"  R agreement (my R_ct == D2 R==3): "
          f"{int((both_t['R_ct'] == (both_t['R_d2'] == 3).cast(pl.Int64)).sum())} of {both_t.height}")

    print("\nflag by round, mine vs D2 (only rounds with truth in each):")
    print(j.filter(pl.col("round").is_in([9, 13, 25, 28]))
          .group_by("round").agg(
              (pl.col("flag") == 1).sum().alias("flag_mine"),
              pl.col("truth_ct").is_not_null().sum().alias("truth_mine"),
              pl.col("flip_d2").sum().alias("flip_d2"),
              pl.col("T_d2").is_not_null().sum().alias("truth_d2"))
          .sort("round"))

    mine_fl = j.filter((pl.col("flag") == 1)).select(["match_key", "round"])
    d2_fl = j.filter(pl.col("flip_d2") == True).select(["match_key", "round"])  # noqa: E712
    inter = mine_fl.join(d2_fl, on=["match_key", "round"], how="inner")
    only_mine = mine_fl.join(d2_fl, on=["match_key", "round"], how="anti")
    only_d2 = d2_fl.join(mine_fl, on=["match_key", "round"], how="anti")
    print(f"\nflagged: mine {mine_fl.height}, D2 {d2_fl.height}, both {inter.height}, "
          f"only mine {only_mine.height}, only D2 {only_d2.height}")
    om = (only_mine.join(j.select(["match_key", "round", "T_d2", "R_d2", "flip_d2", "bomb_ct", "elim_ct", "grp"]),
                         on=["match_key", "round"], how="left"))
    print("only-mine rows by round and whether D2 had a truth value:")
    print(om.group_by(["round"]).agg(pl.len().alias("n"),
                                     pl.col("T_d2").is_null().sum().alias("d2_truth_null"),
                                     pl.col("bomb_ct").is_not_null().sum().alias("has_bomb"),
                                     pl.col("elim_ct").is_not_null().sum().alias("has_elim")).sort("round"))
    od = only_d2.join(j.select(["match_key", "round", "truth_ct", "R_ct", "grp"]), on=["match_key", "round"], how="left")
    print("only-D2 rows by round (my truth value shown):")
    print(od.group_by(["round"]).agg(pl.len().alias("n"), pl.col("truth_ct").is_null().sum().alias("mine_truth_null"))
          .sort("round"))


if __name__ == "__main__":
    main()
