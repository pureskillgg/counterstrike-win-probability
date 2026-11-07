"""D9-12 independent verifier, label-free ordering test. Read-only on the tomes.

Question: is the round label on round_start rows one round early?

Label-free reference: ord(t) = (number of round_end ticks in the same match strictly before t) + 1.
  * an event inside round k's freeze or live window has ord = k;
  * an event in the post-round gap after round k (between round_end k and the freeze of round k+1) has ord = k+1.
  ord uses only tick ORDER of round_end rows, never a round label.
  Each event family's label is compared with ord: diff = label - ord.
    diff = 0  -> the family is labelled with the round it sits in.
    diff = -1 -> the family is labelled one round early (the claim).
    diff = +1 -> the family is labelled one round late.

Checks that the ordinal itself is physical:
  * at each round_state round_freeze_end, the score sum (t_score + ct_score) should equal ord - 1
    (completed rounds before the freeze), as the score is set by round outcomes, not by labels.
  * per match, the number of round_end ticks should equal the header final-score sum (ASSUMPTION:
    the tome holds every round; the bake-off says some 2024/2026 matches are truncated).

ASSUMPTIONS (marked): ASSUMPTION O: round_end ticks are the true round ends and are complete.
ASSUMPTION P: ticks are integers; strict inequality (tick < t) is used for the count.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(12)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2024-02-g": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
    "2026-10-01_to_08": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
}


def scan(base, suf, ch, cols):
    files = sorted(glob.glob(str(base / f"basic_{ch}{suf}" / "dataframe_*")))
    return pl.scan_parquet(files).select(cols).collect()


def diff_table(j, title, build_map_only=False):
    """Print diff = label - ord counts overall and by build."""
    tot = j.group_by("diff").len().sort("diff").rows()
    n = j.height
    print(f"   {title:34s} n={n:7d} | overall diff counts: "
          + ", ".join(f"{d}:{k}" for d, k in tot))
    by_b = j.group_by(["build_num", "diff"]).len().sort(["build_num", "diff"]).rows()
    builds = sorted({b for b, _, _ in by_b})
    for b in builds:
        parts = [f"{d}:{k}" for bb, d, k in by_b if bb == b]
        s = sum(k for bb, d, k in by_b if bb == b)
        print(f"        build {b} (n={s}): " + ", ".join(parts))


for tome, (base, suf) in TOMES.items():
    print(f"\n########## tome {tome}")
    H = pl.concat([pl.read_parquet(f, columns=["key", "build_num", "t_starters_score_final",
                                               "ct_starters_score_final"])
                   for f in sorted(glob.glob(str(base / "header" / "dataframe_*")))]).rename({"key": "match_key"})
    print("   header builds:", H["build_num"].value_counts().sort("build_num").rows())

    RE_raw = scan(base, suf, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    E = RE_raw.select(["match_key", "tick"]).unique().sort(["match_key", "tick"])
    print(f"   round_end rows={RE_raw.height}; unique (match, tick)={E.height}")
    E = E.with_columns(pl.col("tick").rank("ordinal").over("match_key").cast(pl.Int64).alias("rank"))
    E_lt = E.with_columns((pl.col("tick") + 1).alias("tick_lt")).select(["match_key", "tick_lt", "rank"]) \
        .sort("tick_lt")

    # completeness: round_end count per match vs header final-score sum
    cnt = E.group_by("match_key").agg(pl.len().alias("n_end"))
    chk = H.join(cnt, on="match_key", how="left").with_columns(
        (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final")).alias("final_sum"))
    chk = chk.with_columns((pl.col("n_end") - pl.col("final_sum")).alias("n_minus_final"))
    print("   header matches:", H.height, "| matches with round_end:", chk.filter(pl.col("n_end").is_not_null()).height)
    print("   n_end minus header final-sum (count by build):",
          chk.filter(pl.col("n_end").is_not_null()).group_by(["build_num", "n_minus_final"]).len()
          .sort(["build_num", "n_minus_final"]).rows())

    RS = scan(base, suf, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])

    def ordinal(tbl, tcol, extra=()):
        cols = ["match_key", "round", tcol] + list(extra)
        d = tbl.select(cols).rename({tcol: "t"}).sort("t")
        j = d.join_asof(E_lt, left_on="t", right_on="tick_lt", by="match_key", strategy="backward")
        j = j.with_columns((pl.col("rank").fill_null(0) + 1).alias("ord"))
        j = j.with_columns((pl.col("round") - pl.col("ord")).alias("diff"))
        return j.join(H.select(["match_key", "build_num"]), on="match_key", how="left")

    print("\n  -- A. label minus label-free ordinal, by event family (diff 0 = labelled with its own round)")
    diff_table(ordinal(RE_raw, "tick"), "round_end table (self check)")
    fe_rows = RS.filter(pl.col("event_type") == "round_freeze_end")
    fe_j = ordinal(fe_rows, "tick", extra=["t_score", "ct_score"])
    diff_table(fe_j, "round_freeze_end (round_state)")
    diff_table(ordinal(RS.filter(pl.col("event_type") == "round_prestart"), "tick"), "round_prestart (round_state)")
    diff_table(ordinal(RS.filter(pl.col("event_type") == "round_poststart"), "tick"), "round_poststart (round_state)")
    diff_table(ordinal(RS.filter(pl.col("event_type") == "round_start"), "tick"), "round_start (round_state)")
    diff_table(ordinal(RS.filter(pl.col("event_type") == "round_officially_ended"), "tick"),
               "round_officially_ended (round_state)")
    diff_table(ordinal(RS.filter(pl.col("event_type") == "round_end"), "tick"), "round_end (round_state)")
    RST_raw = scan(base, suf, "round_start", ["match_key", "round", "tick"])
    diff_table(ordinal(RST_raw, "tick"), "round_start TABLE (basic)")
    BS = scan(base, suf, "bomb_state", ["match_key", "round", "tick", "event_type"])
    diff_table(ordinal(BS.filter(pl.col("event_type") == "bomb_planted"), "tick"), "bomb_planted")
    diff_table(ordinal(BS.filter(pl.col("event_type") == "bomb_defused"), "tick"), "bomb_defused")
    diff_table(ordinal(BS.filter(pl.col("event_type") == "bomb_exploded"), "tick"), "bomb_exploded")
    DTH = scan(base, suf, "player_death", ["match_key", "round", "tick"])
    diff_table(ordinal(DTH, "tick"), "player_death rows")

    print("\n  -- B. physical check of the ordinal: freeze_end score sum minus (ord - 1), by build")
    fe_j = fe_j.with_columns((pl.col("t_score") + pl.col("ct_score") - (pl.col("ord") - 1)).alias("score_gap"))
    print("   score_gap counts overall:", fe_j.group_by("score_gap").len().sort("score_gap").rows())
    print("   score_gap by build:", fe_j.group_by(["build_num", "score_gap"]).len().sort(["build_num", "score_gap"]).rows())
