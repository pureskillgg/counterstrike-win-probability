"""D4 extra check (outside the off-by-one test): is the round_end winner side wrong beyond bomb rounds?

Rule. A round that ends by elimination ends at the death of the last player of one side. So the victim side of
the last death inside the live window [round_freeze_end, round_end] is the LOSING side, and the winner is the
other side: predicted winner = 5 - victim side (T=2, CT=3).

Scope. Full rounds only (player_count 20 in 2024, 10 in 2026). Rounds with a bomb outcome (defused or exploded)
are excluded, because their winner is fixed by the bomb. A round is kept only if its last death is within
3 s (192 ticks, ASSUMPTION A6) of round_end, so the rule is not applied to timer rounds.

ASSUMPTIONS
  A6. 3 s (192 ticks) window between the last death and round_end.
  A7. player_death.player_team_code is the victim side at the time of the death.
  A8. Ties on the last death tick with different victim sides are dropped.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D4_elim_check.py
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
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", 20),
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", 10),
}
OUTCOMES = ["bomb_defused", "bomb_exploded"]
MAX_GAP = 192
T, CT = 2, 3


def read(base, ch, cols):
    fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def bin_expr():
    return (pl.when(pl.col("round") == 13).then(pl.lit("r13"))
            .when(pl.col("round") == 25).then(pl.lit("r25"))
            .when(pl.col("round") == 28).then(pl.lit("r28"))
            .otherwise(pl.lit("other")))


for era, (base, pc) in TOMES.items():
    print(f"\n==================== ERA {era} (full round = player_count {pc}) ====================")
    re_ = read(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    re_ = re_.unique(["match_key", "round"], keep="first").filter(pl.col("player_count") == pc)
    live = (read(base, "round_state", ["match_key", "round", "tick", "event_type"])
            .filter(pl.col("event_type") == "round_freeze_end")
            .select(["match_key", "round", pl.col("tick").alias("live")])
            .unique(["match_key", "round"], keep="first"))
    W = re_.rename({"tick": "end"}).join(live, on=["match_key", "round"], how="inner")
    print(f"full rounds with live start: {W.height}")

    bs = read(base, "bomb_state", ["match_key", "round", "tick", "event_type"])
    out_keys = (bs.filter(pl.col("event_type").is_in(OUTCOMES))
                .select(["match_key", "round"]).unique())
    de = read(base, "player_death", ["match_key", "round", "tick", "player_team_code"])
    de = de.filter(pl.col("player_team_code").is_in([T, CT]))

    D = de.join(W.select(["match_key", "round", "live", "end"]), on=["match_key", "round"], how="inner")
    D = D.filter((pl.col("tick") >= pl.col("live")) & (pl.col("tick") <= pl.col("end")))
    last_tick = D.group_by(["match_key", "round"]).agg(pl.col("tick").max().alias("last_tick"))
    lastD = D.join(last_tick, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("last_tick"))
    sides = lastD.group_by(["match_key", "round"]).agg(
        pl.col("player_team_code").n_unique().alias("n_sides"),
        pl.col("player_team_code").first().alias("victim_side"),
        pl.col("last_tick").first(),
    )
    n_tie = sides.filter(pl.col("n_sides") > 1).height
    sides = sides.filter(pl.col("n_sides") == 1)

    J = (W.join(sides, on=["match_key", "round"], how="inner")
         .join(out_keys.with_columns(pl.lit(True).alias("has_bomb_outcome")), on=["match_key", "round"], how="left")
         .with_columns(pl.col("has_bomb_outcome").fill_null(False)))
    n_before = J.height
    J = J.filter(~pl.col("has_bomb_outcome"))
    print(f"rounds with any death in window: {sides.height + n_tie}; last-death ties dropped (A8): {n_tie}")
    print(f"rounds with a last death and no bomb outcome: {n_before - int(J.filter(pl.col('has_bomb_outcome')).height)} "
          f"(bomb-outcome rounds excluded: {int(J['has_bomb_outcome'].sum()) if 'has_bomb_outcome' in J.columns else 0})")

    J = J.with_columns(
        (pl.col("end") - pl.col("last_tick")).alias("gap"),
        (5 - pl.col("victim_side")).alias("pred_winner"),
    )
    K = J.filter(pl.col("gap") <= MAX_GAP).with_columns(
        (pl.col("pred_winner") == pl.col("winner_team_code")).alias("agree"),
        bin_expr().alias("rbin"),
        pl.col("match_key").str.extract(r"csds/\d{4}/\d\d/(\d\d)/", 1).alias("day"),
    )
    print(f"kept (last death within {MAX_GAP} ticks of round_end): {K.height}; "
          f"dropped as timer rounds: {J.height - K.height}")
    print("agreement of round_end winner with the opposite of the last victim side, by round bin:")
    print(K.group_by("rbin").agg(pl.len().alias("n"), (~pl.col("agree")).sum().alias("disagree"),
                                 pl.col("agree").mean().alias("agree_rate")).sort("rbin"))
    if era == "2024":
        print("2024 by day (disagree counts, rounds 13 and other):")
        print(K.group_by(["day", "rbin"]).agg(pl.len().alias("n"), (~pl.col("agree")).sum().alias("disagree"))
              .filter(pl.col("rbin").is_in(["r13", "other"]))
              .pivot(on="rbin", index="day", values=["n", "disagree"]).sort("day"))
        print("2024 round 13 kept rounds by winner code and predicted winner:")
        print(K.filter(pl.col("round") == 13).group_by(["winner_team_code", "pred_winner"]).len()
              .sort(["winner_team_code", "pred_winner"]))
    print("disagreeing rounds by round number (rounds with >=1 disagreement):")
    print(K.filter(~pl.col("agree")).group_by("round").len().sort("round").head(40))
