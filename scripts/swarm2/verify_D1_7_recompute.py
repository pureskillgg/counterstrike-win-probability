"""verify_D1_7 recompute (swarm2 verifier, independent of the claimant's D1_followup2 and D1_2026_score_check).

Recomputes the claimed round-27 / 33 / 39 / 45 numbers with the claimant's rule (first vs last
round_state score value in the round, first increment of exactly 1), then tests the mechanism:
round_officially_ended is a side swap that sits inside the round_state window at overtime block ends.

Independent truth sources used here:
  - bomb_state events bomb_defused (CT=3) / bomb_exploded (T=2)
  - round_end win_reason_message (game's own notice; Terrorists_Win/Target_Bombed=2, CTs_Win/Bomb_Defused/Target_Saved=3)
  - round_state scoreboard values read before the swap (pre-swap orientation) across the round

ASSUMPTIONS
  A1 team side codes: 2 = T, 3 = CT (as in the claimant's scripts and in the 2026 bomb check).
  A2 round window for the corrected score rule = first tick of the round to the last tick before round_officially_ended.
  A3 a swap at round_officially_ended is detected when the post-event (t,c) equals the swapped pre-event (c,t) and t != c.
  A4 player_count filters as in the claimant: 2024 == 20, 2026 == 10 (also reported: all pc).
  A5 Memory: lazy scan with column projection; 2 tomes processed one after the other; no E: access.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(40)

T24 = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T26 = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
ROUNDS_KEY = [27, 33, 39, 45]
CONTROL = [9, 12, 13, 24, 25, 26, 28, 29, 30]
WIN_T = {"#SFUI_Notice_Terrorists_Win": 2, "#SFUI_Notice_Target_Bombed": 2, "#SFUI_Notice_CTs_Surrender": 2,
         "#SFUI_Notice_CTs_Win": 3, "#SFUI_Notice_Bomb_Defused": 3, "#SFUI_Notice_Target_Saved": 3,
         "#SFUI_Notice_Terrorists_Surrender": 3}


def hdr(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100, flush=True)


def files(root, ch):
    return sorted(glob.glob(str(root / f"basic_{ch}*" / "dataframe_*")))


def scan(root, ch, cols):
    return pl.concat([pl.scan_parquet(f).select(cols) for f in files(root, ch)], how="vertical_relaxed")


def build(root, label, pc):
    hdr(f"{label}: build tables (player_count == {pc})")
    RE = (scan(root, "round_end", ["match_key", "round", "winner_team_code", "win_reason_message", "player_count"])
          .collect().unique(["match_key", "round"], keep="first"))
    print("round_end unique (match,round):", RE.height, flush=True)
    RE = RE.with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d\d)/", 1).alias("day"))
    RE = RE.with_columns(pl.col("win_reason_message").replace_strict(WIN_T, default=None, return_dtype=pl.Int64).alias("msg_side"))

    BS = scan(root, "bomb_state", ["match_key", "round", "event_type"]).filter(
        pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
    BO = (BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).collect()
          .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
          .select(["match_key", "round", "bomb_side"]))
    print("bomb outcome rounds:", BO.height, flush=True)

    RS = scan(root, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).collect()
    RS = RS.with_row_index("ord").sort(["match_key", "round", "tick", "ord"])
    RS = RS.with_columns(
        pl.col("t_score").shift(1).over(["match_key", "round"]).alias("prev_t"),
        pl.col("ct_score").shift(1).over(["match_key", "round"]).alias("prev_c"),
        (pl.col("event_type") == "round_officially_ended").alias("is_off"))

    base = RS.group_by(["match_key", "round"]).agg(
        pl.col("t_score").first().alias("first_t"), pl.col("ct_score").first().alias("first_c"),
        pl.col("t_score").last().alias("last_t"), pl.col("ct_score").last().alias("last_c"),
        pl.len().alias("rs_rows"))
    off = (RS.filter(pl.col("is_off")).group_by(["match_key", "round"]).agg(
        pl.col("prev_t").first().alias("pre_t"), pl.col("prev_c").first().alias("pre_c"),
        pl.col("t_score").first().alias("off_t"), pl.col("ct_score").first().alias("off_c"),
        pl.col("tick").first().alias("off_tick")))
    S = base.join(off, on=["match_key", "round"], how="left")
    # pre-end values: values just before the swap event if it exists, else last values
    S = S.with_columns(
        pl.coalesce(["pre_t", "last_t"]).alias("pre_end_t"),
        pl.coalesce(["pre_c", "last_c"]).alias("pre_end_c"),
        (pl.col("pre_t").is_not_null()).alias("has_off"),
        ((pl.col("off_t") == pl.col("pre_c")) & (pl.col("off_c") == pl.col("pre_t")) & (pl.col("pre_t") != pl.col("pre_c"))).alias("swap_sig"),
        ((pl.col("off_t") == pl.col("pre_c")) & (pl.col("off_c") == pl.col("pre_t")) & (pl.col("pre_t") == pl.col("pre_c"))).alias("swap_tie"))

    # claimant rule: first vs last value, first increment of exactly 1
    S = S.with_columns(
        pl.when((pl.col("last_t") - pl.col("first_t")) == 1).then(2)
          .when((pl.col("last_c") - pl.col("first_c")) == 1).then(3)
          .otherwise(None).alias("claim_side"))
    # corrected rule: first vs pre-swap end value
    dt = pl.col("pre_end_t") - pl.col("first_t")
    dc = pl.col("pre_end_c") - pl.col("first_c")
    S = S.with_columns(
        pl.when((dt == 1) & (dc == 0)).then(2)
          .when((dc == 1) & (dt == 0)).then(3)
          .otherwise(None).alias("corr_side"),
        (pl.col("first_t") == pl.col("first_c")).alias("pre_tie"))

    A = (RE.filter(pl.col("player_count") == pc) if pc is not None else RE)
    A = (A.join(BO, on=["match_key", "round"], how="left")
         .join(S, on=["match_key", "round"], how="left"))
    A = A.with_columns(
        (pl.col("claim_side").is_not_null() & (pl.col("claim_side") != pl.col("winner_team_code"))).alias("claim_dis"),
        (pl.col("corr_side").is_not_null() & (pl.col("corr_side") != pl.col("winner_team_code"))).alias("corr_dis"),
        (pl.col("bomb_side").is_not_null() & (pl.col("bomb_side") != pl.col("winner_team_code"))).alias("bomb_dis"),
        (pl.col("msg_side").is_not_null() & (pl.col("msg_side") != pl.col("winner_team_code"))).alias("msg_dis"),
        pl.col("rs_rows").is_not_null().alias("has_rs"))
    return A


def summarize(A, label):
    hdr(f"{label}: per-round table for claimed rounds and controls")
    cols = [pl.len().alias("rows"), pl.col("has_rs").sum().alias("with_rs"),
            pl.col("claim_side").is_not_null().sum().alias("claim_n"),
            pl.col("claim_dis").sum().alias("claim_dis"),
            pl.col("corr_side").is_not_null().sum().alias("corr_n"),
            pl.col("corr_dis").sum().alias("corr_dis"),
            pl.col("has_off").sum().alias("off_events"),
            pl.col("swap_sig").sum().alias("swap_sig"),
            pl.col("swap_tie").sum().alias("swap_tie"),
            pl.col("bomb_side").is_not_null().sum().alias("bomb_n"),
            pl.col("bomb_dis").sum().alias("bomb_dis"),
            pl.col("msg_side").is_not_null().sum().alias("msg_n"),
            pl.col("msg_dis").sum().alias("msg_dis")]
    t = A.group_by("round").agg(cols).sort("round")
    print(t.filter(pl.col("round").is_in(ROUNDS_KEY + CONTROL)), flush=True)
    print("\nrounds 1..30 with claim_dis > 0:", t.filter((pl.col("round") <= 30) & (pl.col("claim_dis") > 0))["round"].to_list())
    print("rounds 1..30 with corr_dis > 0:", t.filter((pl.col("round") <= 30) & (pl.col("corr_dis") > 0))["round"].to_list())
    print("rounds 1..30 with swap_sig > 0:", t.filter((pl.col("round") <= 30) & (pl.col("swap_sig") > 0))["round"].to_list())
    print("rounds 1..30 with msg_dis > 0:", t.filter((pl.col("round") <= 30) & (pl.col("msg_dis") > 0))["round"].to_list())
    print("rounds 1..30 with bomb_dis > 0:", t.filter((pl.col("round") <= 30) & (pl.col("bomb_dis") > 0))["round"].to_list())


def key_rounds(A, label):
    hdr(f"{label}: claimed rounds {ROUNDS_KEY} in detail")
    for r in ROUNDS_KEY:
        x = A.filter(pl.col("round") == r)
        n = x.height
        cn = x.filter(pl.col("claim_side").is_not_null())
        print(f"\n-- round {r}: rows={n}, with round_state={x.filter(pl.col('has_rs')).height}, "
              f"claim score side non-null={cn.height}, claim disagreements={cn.filter(pl.col('claim_dis')).height}")
        print("   game win-message side disagrees with winner:", x.filter(pl.col("msg_side").is_not_null() & pl.col("msg_dis")).height,
              "of", x.filter(pl.col("msg_side").is_not_null()).height, "rows with a message side")
        xb = x.filter(pl.col("bomb_side").is_not_null())
        print("   bomb-verified rows:", xb.height, " winner disagrees with bomb:", xb.filter(pl.col("bomb_dis")).height,
              " claim score disagrees with bomb (rows with claim side):",
              xb.filter(pl.col("claim_side").is_not_null() & (pl.col("claim_side") != pl.col("bomb_side"))).height,
              "of", xb.filter(pl.col("claim_side").is_not_null()).height)
        print("   bomb-verified rows with claim score side:",
              xb.filter(pl.col("claim_side").is_not_null()).height,
              " claim score side == winner there:",
              xb.filter(pl.col("claim_side").is_not_null() & (pl.col("claim_side") == pl.col("winner_team_code"))).height)
        print("   corrected score side non-null:", x.filter(pl.col("corr_side").is_not_null()).height,
              " corrected disagreements with winner:", x.filter(pl.col("corr_dis")).height,
              " swap signature rows:", x.filter(pl.col("swap_sig")).height,
              " swap-tie rows:", x.filter(pl.col("swap_tie")).height)
        print("   claim disagreement by pre-round tie (first_t == first_c):")
        print(cn.group_by(["pre_tie", "claim_dis"]).len().sort(["pre_tie", "claim_dis"]).to_dicts())
        print("   claim disagreement by day group (d01-07 vs d12-25):")
        print(cn.with_columns(pl.when(pl.col("day").is_in([f"{d:02d}" for d in range(12, 26)])).then(pl.lit("d12-25")).otherwise(pl.lit("d01-07")).alias("dg"))
              .group_by(["dg", "claim_dis"]).len().sort(["dg", "claim_dis"]).to_dicts())


def run():
    A24 = build(T24, "2024 tome", 20)
    summarize(A24, "2024 (pc==20)")
    key_rounds(A24, "2024 (pc==20)")
    A24all = build(T24, "2024 tome, all player counts", None)
    hdr("2024 all player counts: claimed rounds")
    print(A24all.group_by("round").agg(pl.col("claim_dis").sum().alias("claim_dis"), pl.col("claim_side").is_not_null().sum().alias("claim_n"),
                                    pl.col("corr_dis").sum().alias("corr_dis"), pl.col("corr_side").is_not_null().sum().alias("corr_n"),
                                    pl.col("swap_sig").sum().alias("swap_sig"))
          .filter(pl.col("round").is_in(ROUNDS_KEY + CONTROL)).sort("round"))
    del A24all

    A26 = build(T26, "2026 tome", 10)
    summarize(A26, "2026 (pc==10)")
    key_rounds(A26, "2026 (pc==10)")
    A26all = build(T26, "2026 tome, all player counts", None)
    hdr("2026 all player counts: claimed rounds")
    print(A26all.group_by("round").agg(pl.col("claim_dis").sum().alias("claim_dis"), pl.col("claim_side").is_not_null().sum().alias("claim_n"),
                                    pl.col("corr_dis").sum().alias("corr_dis"), pl.col("corr_side").is_not_null().sum().alias("corr_n"),
                                    pl.col("swap_sig").sum().alias("swap_sig"))
          .filter(pl.col("round").is_in(ROUNDS_KEY + CONTROL)).sort("round"))


if __name__ == "__main__":
    run()
