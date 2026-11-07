"""Verifier D2-4, third follow-up: (a) D2's lag-1 reading at round 28, (b) elimination check on the claim's
population (R != T with T from any source), (c) bomb planter team code on flipped rounds (side-attached check).
"""
import sys

import polars as pl

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import verify_D2_4_check as V  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(25)


def main():
    full = V.build("2024")
    pop = full.filter(pl.col("player_count") == V.PC["2024"])

    # (a) D2 lag-1 reading: R = side, in round r-1, of the team that won round r
    x_won = pl.col("s_score") == pl.col("sx")
    lag = (pop.filter(pl.col("flip_score") & pl.col("s_score").is_not_null() & pl.col("sx_p").is_not_null())
           .with_columns(
               pl.when(x_won).then(pl.col("sx_p")).otherwise(5 - pl.col("sx_p")).alias("pred_lag"),
               pl.when(x_won).then(pl.lit(2)).otherwise(pl.lit(3)).alias("pred_static")))
    for rr in [13, 25, 28]:
        g = lag.filter(pl.col("round") == rr)
        print(f"[A] round {rr} flipped rows {g.height}: R==pred_lag {g.filter(pl.col('R') == pl.col('pred_lag')).height}; "
              f"R==pred_static(identity) {g.filter(pl.col('R') == pl.col('pred_static')).height}")

    # (b) claim population: flipped = R != T, T from any source, elimination known
    fl = pop.filter(pl.col("flip") & pl.col("s_elim").is_not_null())
    print(f"[B] claim-style population (R != T, T any source, elim known): {fl.height}; "
          f"elim==T {fl.filter(pl.col('s_elim') == pl.col('T')).height}; elim==R {fl.filter(pl.col('s_elim') == pl.col('R')).height}")
    print("    by T_src:", fl.group_by("T_src").len().sort("T_src").rows())
    print(f"    all rows with T known, elim known: {pop.filter(pl.col('T').is_not_null() & pl.col('s_elim').is_not_null()).height}; "
          f"T==elim {pop.filter(pl.col('T').is_not_null() & pl.col('s_elim').is_not_null() & (pl.col('T') == pl.col('s_elim'))).height}")

    # (c) planter side (bomb_state.player_team_code on bomb_planted rows) for flipped rounds
    plant = (V.rd("2024", "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"],
                  pl.col("event_type") == "bomb_planted")
             .sort("tick").unique(["match_key", "round"], keep="first")
             .select(["match_key", "round", pl.col("player_team_code").alias("planter_side")]))
    pp = pop.join(plant, on=["match_key", "round"], how="inner")
    print(f"[C] rounds with a planted bomb: {pp.height}; planter side==2 (T): {pp.filter(pl.col('planter_side') == 2).height}; "
          f"planter side==3: {pp.filter(pl.col('planter_side') == 3).height}")
    pf = pp.filter(pl.col("flip_score") & pl.col("s_score").is_not_null())
    print(f"    flipped-by-score planted rounds: {pf.height}; planter side==2: {pf.filter(pl.col('planter_side') == 2).height}; "
          f"planter side==s_score: {pf.filter(pl.col('planter_side') == pl.col('s_score')).height}")


if __name__ == "__main__":
    main()
