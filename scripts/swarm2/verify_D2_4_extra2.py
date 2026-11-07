"""Verifier D2-4, second follow-up: why elimination-only and bomb-only truth rows lack a score truth, and the
round-13 / round-25 / round-28 mechanisms on flipped rows. Uses verify_D2_4_check.build (no D2 outputs read).
"""
import sys

import polars as pl

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import verify_D2_4_check as V  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(25)
pl.Config.set_fmt_str_lengths(40)


def enrich(label):
    full = V.build(label)
    plant = (V.rd(label, "bomb_state", ["match_key", "round", "tick", "event_type"],
                  pl.col("event_type") == "bomb_planted")
             .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("t_plant")))
    pi = (V.rd(label, "player_info", ["match_key", "round", "player_id", "team_code"])
          .filter(pl.col("team_code").is_in([2, 3]))
          .unique(["match_key", "round", "player_id"], keep="first"))
    npl = pi.group_by(["match_key", "round"]).agg(pl.len().alias("np_round"))
    full = (full.join(plant, on=["match_key", "round"], how="left")
            .join(npl, on=["match_key", "round"], how="left")
            .with_columns(
                pl.col("t_plant").is_not_null().alias("planted"),
                (pl.col("sx").is_not_null() & pl.col("sx_p").is_not_null()
                 & (pl.col("sx") != pl.col("sx_p"))).alias("start_swap"),
                pl.when(pl.col("s_score").is_not_null()).then(pl.lit("score_ok"))
                .when(pl.col("t_fe").is_null()).then(pl.lit("no_freeze_end_this_round"))
                .when(pl.col("tn").is_null()).then(pl.lit("no_next_freeze_end"))
                .when(pl.col("sx").is_null()).then(pl.lit("no_sx_this_round"))
                .when(pl.col("sx_n").is_null()).then(pl.lit("no_sx_next_round"))
                .when((pl.col("dX") == 0) & (pl.col("dY") == 0)).then(pl.lit("no_delta"))
                .otherwise(pl.lit("unclean_delta")).alias("score_status")))
    return full


def main():
    full = enrich("2024")
    pop = full.filter(pl.col("player_count") == V.PC["2024"])

    print("[A] score-truth status by T_src (2024 population):")
    print(pop.group_by(["score_status", "T_src"]).len().sort(["score_status", "T_src"]).rows())

    eo = pop.filter((pl.col("T_src") == "elim") & (pl.col("R") != pl.col("T")))
    other = eo.filter(~pl.col("round").is_in([13, 25, 28]))
    print(f"[B] elim-only rows with R != elim: {eo.height}; other rounds (not 13/25/28): {other.height}")
    print("    status of other-round rows:", other.group_by("score_status").len().sort("score_status").rows())
    print("    planted among other-round rows:", other.filter(pl.col("planted")).height,
          "; round (round, n):", other.group_by("round").len().sort("round").rows())
    for r in other.select(["match_key", "round", "R", "T", "t_dead", "ct_dead", "planted", "score_status", "np_round"]).head(12).iter_rows():
        print("     ", r[0][-24:], r[1:])

    # round 25 rows: start-of-round side change? score / bomb / identity mechanism
    f25 = pop.filter(pl.col("round") == 25)
    print(f"[C] round 25: rows {f25.height}; with score truth {f25.filter(pl.col('s_score').is_not_null()).height}; "
          f"start-of-round side change {f25.filter(pl.col('start_swap')).height}; flipped {f25.filter(pl.col('flip_score')).height}")
    f25f = f25.filter(pl.col("flip_score") & pl.col("s_score").is_not_null())
    print("    flipped r25 by (sx, s_score, R) [sx=3 means team X on CT]:",
          f25f.group_by(["sx", "s_score", "R"]).len().sort(["sx", "s_score", "R"]).rows())
    f25b = f25.filter(pl.col("s_bomb").is_not_null() & pl.col("s_score").is_not_null())
    print(f"    r25 bomb rows with score: {f25b.height}; score==bomb {f25b.filter(pl.col('s_score') == pl.col('s_bomb')).height}; "
          f"R==bomb {f25b.filter(pl.col('R') == pl.col('s_bomb')).height}")

    # round 13 and 28 flipped rows: R vs team-identity reading and vs previous-round side truth
    prev = full.select(["match_key", (pl.col("round") + 1).alias("round"), pl.col("s_score").alias("s_prev")])
    for rr in [13, 25, 28]:
        g = (pop.filter((pl.col("round") == rr) & pl.col("flip_score") & pl.col("s_score").is_not_null())
             .join(prev, on=["match_key", "round"], how="left")
             .with_columns(
                 pl.when(pl.col("s_score") == pl.col("sx")).then(pl.lit(2)).otherwise(pl.lit(3)).alias("R_identity")))
        print(f"[D] round {rr} flipped rows {g.height}: R==team-identity code {g.filter(pl.col('R') == pl.col('R_identity')).height}; "
              f"R==side of previous round's winner {g.filter(pl.col('R') == pl.col('s_prev')).height} "
              f"(known {g.filter(pl.col('s_prev').is_not_null()).height})")

    # round-9 bomb-only rows with R != bomb: inspect raw side-read freeze-end deltas
    b9 = pop.filter((pl.col("round") == 9) & (pl.col("T_src") == "bomb") & (pl.col("R") != pl.col("T")))
    print(f"[E] round-9 bomb-only rows with R != bomb: {b9.height}")
    for r in b9.select(["match_key", "round", "R", "T", "t_score", "ct_score", "tn", "ctn", "sx", "sx_n"]).iter_rows():
        raw_t = (r[6] - r[4]) if (r[6] is not None and r[4] is not None) else None
        raw_c = (r[7] - r[5]) if (r[7] is not None and r[5] is not None) else None
        print("     ", r[0][-24:], "R", r[2], "bomb", r[3], "raw dT", raw_t, "raw dCT", raw_c, "sx", r[8], "sx_n", r[9])


if __name__ == "__main__":
    main()
