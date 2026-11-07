"""verify_D2_3 channels: validity of the sx-free score delta, round-tag offset tests, bomb counts by round.

Reads the base tables written by verify_D2_3_recompute.py (scratchpad, not the project).
Read-only with respect to the tome and the claim's files.
Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_3_channels.py
"""
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(30)
SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill"
               r"\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad\verify_D2_3")
KEY = ["match_key", "round"]


def main():
    for label in ["2024", "2026"]:
        b = pl.read_parquet(SCRATCH / f"base_{label}.parquet")
        # ASSUMPTION: a score delta FE(r)->FE(r+1) is side-valid only when the team-side map does not change
        # across the boundary, i.e. sx(r) == sx(r+1). sx_nx is sx(r+1).
        b = b.with_columns(
            (pl.col("sx") == pl.col("sx_nx")).alias("no_swap_next"),
            (pl.col("sx") != pl.col("sx_prev")).alias("swap_at_r"))
        print(f"\n################ {label} ################")

        print("\n-- rows per round with s_score_free, split by boundary validity (sx(r) == sx(r+1)); mismatch vs R --")
        k = b.filter(pl.col("s_score_free").is_not_null() & pl.col("sx_nx").is_not_null() & pl.col("sx").is_not_null())
        t = (k.group_by(["round", "no_swap_next"]).agg(
            pl.len().alias("n"), (pl.col("s_score_free") != pl.col("R")).sum().alias("mismatch"))
             .sort(["round", "no_swap_next"]))
        print(t.filter(pl.col("round").is_between(1, 30)))

        print("\n-- s_score_free (no-swap boundary only) vs R, by round group --")
        kv = k.filter(pl.col("no_swap_next"))
        print(kv.group_by("grp").agg(pl.len().alias("n"), (pl.col("s_score_free") != pl.col("R")).sum().alias("mismatch"),
                                     (pl.col("s_score_free") != pl.col("R")).mean().round(4).alias("rate")).sort("grp"))

        print("\n-- sx change at round r (swap_at_r) counts by round, rounds 1-30 --")
        print(b.filter(pl.col("sx").is_not_null() & pl.col("sx_prev").is_not_null() & pl.col("round").is_between(1, 30))
              .group_by(["round", "swap_at_r"]).len().sort(["round", "swap_at_r"]))

        # bomb rows per round (sanity: is n=752 at both rounds 9 and 13 a bug?)
        kb = b.filter(pl.col("s_bomb_w").is_not_null())
        print("\n-- s_bomb_w rows and mismatches by round (1-30) --")
        print(kb.group_by("round").agg(pl.len().alias("n"), (pl.col("s_bomb_w") != pl.col("R")).sum().alias("mismatch"))
              .filter(pl.col("round").is_between(1, 30)).sort("round"))

        # round-tag offset test: does the bomb outcome (or the sx-free score winner) of round r match R(r-1) or R(r+1)?
        R_by = b.select(KEY + ["R"])
        nb = (b.select(KEY + ["s_bomb_w", "s_score_free", "no_swap_next", "R", "grp"])
              .join(R_by.with_columns((pl.col("round") + 1).alias("round")).rename({"R": "R_prev"}), on=KEY, how="left")
              .join(R_by.with_columns((pl.col("round") - 1).alias("round")).rename({"R": "R_next"}), on=KEY, how="left"))
        print("\n-- offset test, bomb outcome vs recorded winner of round r, r-1, r+1 (rows where bomb exists and R disagrees with bomb at r) --")
        bm = nb.filter(pl.col("s_bomb_w").is_not_null() & (pl.col("s_bomb_w") != pl.col("R")))
        print(bm.group_by("grp").agg(
            pl.len().alias("n_disagree_at_r"),
            (pl.col("s_bomb_w") == pl.col("R_prev")).sum().alias("bomb==R(r-1)"),
            (pl.col("s_bomb_w") == pl.col("R_next")).sum().alias("bomb==R(r+1)")).sort("grp"))
        print("-- same test where bomb agrees with R at r (control) --")
        bg = nb.filter(pl.col("s_bomb_w").is_not_null() & (pl.col("s_bomb_w") == pl.col("R")))
        print(bg.group_by("grp").agg(
            pl.len().alias("n_agree_at_r"),
            (pl.col("s_bomb_w") == pl.col("R_prev")).sum().alias("bomb==R(r-1)"),
            (pl.col("s_bomb_w") == pl.col("R_next")).sum().alias("bomb==R(r+1)")).sort("grp"))

        print("-- offset test, valid sx-free score winner vs R at r, r-1, r+1 (no-swap boundary rows) --")
        sm = nb.filter(pl.col("s_score_free").is_not_null() & pl.col("no_swap_next") & (pl.col("s_score_free") != pl.col("R")))
        print(sm.group_by("grp").agg(
            pl.len().alias("n_disagree_at_r"),
            (pl.col("s_score_free") == pl.col("R_prev")).sum().alias("score==R(r-1)"),
            (pl.col("s_score_free") == pl.col("R_next")).sum().alias("score==R(r+1)")).sort("grp"))


if __name__ == "__main__":
    main()
