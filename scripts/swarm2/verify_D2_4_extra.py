"""Verifier D2-4, follow-up checks on top of verify_D2_4_check.build (independent of D2 scripts).

Adds: corrected day (match_key ends in /csds), start-of-round swap flag, bomb-planted flag, roster completeness,
elimination-only and bomb-only truth audits, team-identity reading of R on flipped rows, flagged-match rate by day.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, r"G:\pureskill\csds-cs2\scripts\swarm2")
import verify_D2_4_check as V  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(25)
pl.Config.set_fmt_str_lengths(40)


def main():
    tabs = {}
    for label in ["2024", "2026"]:
        full = V.build(label)
        full = full.with_columns(
            pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).alias("day2"))
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
                     & (pl.col("sx") != pl.col("sx_p"))).alias("start_swap")))
        tabs[label] = full
        pop = full.filter(pl.col("player_count") == V.PC[label])

        print(f"\n################ {label}")
        # 1. start-of-round swap rounds with score truth, by round number
        ss = pop.filter(pl.col("start_swap") & pl.col("s_score").is_not_null())
        print("[1] start-of-round swap rows with score truth, by round: [round, n, flipped]:",
              ss.group_by("round").agg(pl.len().alias("n"), pl.col("flip_score").sum().alias("fl"))
              .sort("round").rows())
        nons = pop.filter(~pl.col("start_swap") & pl.col("flip_score") & pl.col("s_score").is_not_null())
        print(f"    flipped rows that are NOT start-of-round swaps: {nons.height}")
        print(f"    start-swap rows NOT flipped, rounds >= 25: "
              f"{ss.filter(~pl.col('flip_score') & (pl.col('round') >= 25)).height} of {ss.filter(pl.col('round') >= 25).height}")

        # 2. flagged-match rate by day (match flagged = any start-swap row flipped against score truth)
        mm = (ss.group_by("match_key").agg(pl.col("flip_score").any().alias("flagged"), pl.col("day2").first().alias("day"))
              .group_by("day").agg(pl.len().alias("matches_with_swap_truth"), pl.col("flagged").sum().alias("flagged"),
                                   pl.col("flagged").mean().round(3).alias("rate")).sort("day"))
        print("[2] flagged matches by day (swap truth available):")
        print(mm.rows())

        # 3. elimination disagreements with score truth: are they post-plant?
        el = pop.filter(pl.col("s_elim").is_not_null() & pl.col("s_score").is_not_null())
        dis = el.filter(pl.col("s_elim") != pl.col("s_score"))
        print(f"[3] elim vs score: known {el.height}; disagree {dis.height} ({dis.height / max(el.height, 1):.2%}); "
              f"disagreements with bomb planted: {dis.filter(pl.col('planted')).height}; "
              f"disagreements with bomb outcome present: {dis.filter(pl.col('s_bomb').is_not_null()).height}")
        dis_f = dis.filter(pl.col("flip_score"))
        print(f"    flipped rows where elim disagrees with score: {dis_f.height}, all planted: {dis_f.filter(pl.col('planted')).height == dis_f.height}; "
              f"elim==R among them: {dis_f.filter(pl.col('s_elim') == pl.col('R')).height}")
        print(f"    rows with elim AND bomb known: {el.filter(pl.col('s_bomb').is_not_null()).height}; "
              f"elim==bomb: {el.filter(pl.col('s_bomb').is_not_null() & (pl.col('s_elim') == pl.col('s_bomb'))).height}; "
              f"non-planted elim==bomb: {el.filter(~pl.col('planted') & pl.col('s_bomb').is_not_null() & (pl.col('s_elim') == pl.col('s_bomb'))).height} "
              f"of non-planted {el.filter(~pl.col('planted') & pl.col('s_bomb').is_not_null()).height}")

        # 4. elimination-only truth (no score, no bomb): audit against R
        eo = pop.filter(pl.col("T_src") == "elim")
        eo_r = eo.filter(pl.col("R").is_not_null())
        print(f"[4] elim-only truth rows: {eo.height}; R==elim {eo_r.filter(pl.col('R') == pl.col('T')).height}; "
              f"R!=elim {eo_r.filter(pl.col('R') != pl.col('T')).height}; of R!=elim, planted: "
              f"{eo_r.filter((pl.col('R') != pl.col('T')) & pl.col('planted')).height}")
        print("    elim-only R!=T by round (round, n):",
              eo_r.filter(pl.col("R") != pl.col("T")).group_by("round").len().sort("round").rows())

        # 5. bomb-only truth (no score): audit against R
        bo = pop.filter(pl.col("T_src") == "bomb")
        print(f"[5] bomb-only truth rows: {bo.height}; R==bomb {bo.filter(pl.col('R') == pl.col('T')).height}; "
              f"R!=bomb {bo.filter(pl.col('R') != pl.col('T')).height}; "
              f"R!=bomb by round (round, n): {bo.filter(pl.col('R') != pl.col('T')).group_by('round').len().sort('round').rows()}")

        # 6. R on flipped rows: team-identity reading (2 if X won, else 3) vs side reading
        fl = pop.filter(pl.col("flip_score") & pl.col("s_score").is_not_null() & pl.col("sx").is_not_null())
        xw = fl.with_columns(
            pl.when(pl.col("s_score") == pl.col("sx")).then(pl.lit(2)).otherwise(pl.lit(3)).alias("R_identity"))
        print(f"[6] flipped rows with sx: {fl.height}; R == team-identity code: "
              f"{xw.filter(pl.col('R') == pl.col('R_identity')).height}; R == side-of-winner (5-s_score): "
              f"{fl.filter(pl.col('R') == 5 - pl.col('s_score')).height}")

        # 7. roster completeness vs score-bomb agreement (sx coverage difference vs D2)
        cc = pop.filter(pl.col("s_score").is_not_null() & pl.col("s_bomb").is_not_null())
        for nm, cond in [("full roster np_round>=10", pl.col("np_round") >= 10), ("partial np_round<10", pl.col("np_round") < 10)]:
            sub = cc.filter(cond)
            print(f"[7] {nm}: score&bomb rows {sub.height}; score==bomb {sub.filter(pl.col('s_score') == pl.col('s_bomb')).height}; "
                  f"flipped-by-score {sub.filter(pl.col('flip_score')).height}")
        print(f"    coverage: sx known {pop.filter(pl.col('sx').is_not_null()).height}; T known {pop.filter(pl.col('T').is_not_null()).height}; "
              f"T_src {pop.group_by('T_src').len().sort('T_src').rows()}; "
              f"round_end rows with np_round null {pop.filter(pl.col('np_round').is_null()).height}")

        # 8. falls of t_score with no side change (should be impossible under side-read scores)
        mono = pop.filter(pl.col("tn").is_not_null() & pl.col("t_score").is_not_null())
        bad = mono.filter(((pl.col("tn") < pl.col("t_score")) | (pl.col("ctn") < pl.col("ct_score")))
                          & ~(pl.col("sx").is_not_null() & pl.col("sx_n").is_not_null() & (pl.col("sx") != pl.col("sx_n"))))
        print(f"[8] freeze-end falls with no sx change between r and r+1 (or sx unknown): {bad.height}; "
              f"sx unknown among them: {bad.filter(pl.col('sx').is_null() | pl.col('sx_n').is_null()).height}")

    # 9. 2026 control, start-swap rows
    t26 = tabs["2026"].filter(pl.col("player_count") == V.PC["2026"])
    print("\n[9] 2026 start-swap rows with score truth: ",
          t26.filter(pl.col("start_swap") & pl.col("s_score").is_not_null()).height,
          " flipped:", t26.filter(pl.col("start_swap") & pl.col("flip_score")).height)


if __name__ == "__main__":
    main()
