"""verify D2-5: recompute the win_reason_code claim from the raw tome, independent of D2 parquet outputs.

Claim under test (D2-5, notes\\swarm2\\D2.md):
  2024: win-reason side == R (winner_team_code) in 68,391 of 68,392 rounds with truth;
        in flipped rows 2,136 of 2,137 reason == R, and 1 == T.
  2026: reason == R in 31,141 of 31,141.
  Conclusion: win_reason_code carries the same defect and cannot repair the labels.

Method (all from the tome, column-projected lazy scans, one channel at a time):
  reason_side  from win_reason_message NAME (engine string, not a code-number assumption)
               T names: Target_Bombed, Terrorists_Win, Hostages_Not_Rescued, CTs_Surrender -> code 2
               CT names: Bomb_Defused, CTs_Win, All_Hostages_Rescued, Target_Saved, Terrorists_Surrender -> code 3
               (Terrorists_Surrender = the T side gave up, so CT wins; CTs_Surrender = T wins.)
  R            winner_team_code (round_end)
  s_bomb       in-window bomb outcome (bomb_exploded -> 2 = T, bomb_defused -> 3 = CT), physical,
               uses no team codes. Window = [round_freeze_end tick, round_end tick + 40]. ASSUMPTION (+40).
  elim_code    code of the side with all five victims dead, from player_death.player_team_code of
               victims in [freeze_end, round_end]. ASSUMPTION: victim codes are per-round side codes.
  planter / actor codes from bomb_state.player_team_code (event-level codes, separate channel).

Full 5v5 filter: 2024 player_count == 20, 2026 player_count == 10 (ASSUMPTION, as in D2 and bakeoff).
Dedup: first round_end per (match_key, round) by tick.

Run: G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\verify_D2_5_c_recompute.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(200)
pl.Config.set_tbl_cols(40)
pl.Config.set_fmt_str_lengths(60)

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
PC = {"2024": 20, "2026": 10}
T_NAMES = ["Target_Bombed", "Terrorists_Win", "Hostages_Not_Rescued", "CTs_Surrender"]
CT_NAMES = ["Bomb_Defused", "CTs_Win", "All_Hostages_Rescued", "Target_Saved", "Terrorists_Surrender"]
BOMB_TAIL = 40  # ASSUMPTION


def files(label, ch):
    fd = sorted(glob.glob(str(TOME[label] / f"basic_{ch}.*")))[0]
    return sorted(glob.glob(fd + "/dataframe_*"))


def load(label, ch, want):
    fs = files(label, ch)
    have = set(pl.read_parquet_schema(fs[0]).keys())
    cols = [c for c in want if c in have]
    missing = [c for c in want if c not in have]
    if missing:
        print(f"  [{label} {ch}] columns absent (ok if optional): {missing}")
    return pl.scan_parquet(fs).select(cols).collect()


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def build(label):
    section(f"{label}: round_end and dedup")
    re_raw = load(label, "round_end", ["match_key", "round", "tick", "winner_team_code",
                                        "win_reason_code", "win_reason_message", "legacy_code", "player_count"])
    print("round_end raw rows:", re_raw.height)
    print("player_count distribution (top 8):")
    print(re_raw.group_by("player_count").len().sort("len", descending=True).head(8))
    re_ = (re_raw.filter(pl.col("player_count") == PC[label])
           .sort("tick").unique(["match_key", "round"], keep="first"))
    dup_total = re_raw.filter(pl.col("player_count") == PC[label]).height - re_.height
    print(f"full-5v5 rows (pc=={PC[label]}) after dedup: {re_.height} (dropped duplicates: {dup_total})")

    re_ = re_.with_columns(
        pl.col("win_reason_message").str.replace("#SFUI_Notice_", "").alias("msg"))
    re_ = re_.with_columns(
        pl.when(pl.col("msg").is_in(T_NAMES)).then(pl.lit(2))
        .when(pl.col("msg").is_in(CT_NAMES)).then(pl.lit(3))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("reason_side"))
    print("message -> reason_side check (all full-5v5 rows):")
    print(re_.group_by(["win_reason_code", "msg", "reason_side"]).len().sort("win_reason_code"))
    print("unmapped messages:", re_.filter(pl.col("reason_side").is_null()).height)
    if "legacy_code" in re_.columns:
        print("legacy_code x reason_side:")
        print(re_.group_by(["legacy_code", "reason_side"]).len().sort(["legacy_code", "reason_side"]))

    # round window
    fe = (load(label, "round_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("event_type") == "round_freeze_end").sort("tick")
          .unique(["match_key", "round"], keep="first").select(["match_key", "round", pl.col("tick").alias("t_fe")]))
    re_ = re_.join(fe, on=["match_key", "round"], how="left").rename({"tick": "t_end"})
    print("rows with freeze_end:", re_.filter(pl.col("t_fe").is_not_null()).height, "of", re_.height)

    # bomb outcome and planter, from bomb_state
    bs = load(label, "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
    print("bomb_state event types:", sorted(bs["event_type"].unique().to_list()))
    bs = bs.join(re_.select(["match_key", "round", "t_fe", "t_end"]), on=["match_key", "round"], how="inner")
    outc = (bs.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
            .filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end") + BOMB_TAIL))
            .sort("tick"))
    n_outc = outc.group_by(["match_key", "round"]).agg(pl.len().alias("n_outcome_events"),
                                                        pl.col("event_type").n_unique().alias("n_types"))
    outc = (outc.group_by(["match_key", "round"]).agg(
        pl.col("event_type").first().alias("bo_first"),
        pl.col("player_team_code").first().alias("actor_code"))
        .with_columns(
            pl.when(pl.col("bo_first") == "bomb_exploded").then(pl.lit(2)).otherwise(pl.lit(3))
            .cast(pl.Int64).alias("s_bomb")))
    outc = outc.join(n_outc, on=["match_key", "round"], how="left")
    print("in-window bomb outcome rows (rounds):", outc.height,
          " with >1 outcome type:", outc.filter(pl.col("n_types") > 1).height)

    planter = (bs.filter((pl.col("event_type") == "bomb_planted") & (pl.col("tick") >= pl.col("t_fe"))
                         & (pl.col("tick") <= pl.col("t_end") + BOMB_TAIL))
               .sort("tick").group_by(["match_key", "round"]).agg(
                   pl.col("player_team_code").first().alias("planter_code")))
    print("rounds with a planter event in window:", planter.height)

    # victim side counts from player_death, codes as recorded on the victim
    dt = load(label, "player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
    print("player_death victim code distribution:")
    print(dt.group_by("player_team_code").len().sort("player_team_code"))
    dt = (dt.join(re_.select(["match_key", "round", "t_fe", "t_end"]), on=["match_key", "round"], how="inner")
          .filter((pl.col("tick") >= pl.col("t_fe")) & (pl.col("tick") <= pl.col("t_end")))
          .unique(["match_key", "round", "player_id"], keep="first"))
    dsum = dt.group_by(["match_key", "round"]).agg(
        (pl.col("player_team_code") == 2).sum().alias("v2"),
        (pl.col("player_team_code") == 3).sum().alias("v3"))
    dsum = dsum.with_columns(
        pl.when((pl.col("v2") == 5) & (pl.col("v3") < 5)).then(pl.lit(3))
        .when((pl.col("v3") == 5) & (pl.col("v2") < 5)).then(pl.lit(2))
        .otherwise(pl.lit(None)).cast(pl.Int64).alias("elim_code"))

    tab = (re_.join(outc.select(["match_key", "round", "s_bomb", "bo_first", "actor_code", "n_outcome_events"]),
                    on=["match_key", "round"], how="left")
           .join(planter, on=["match_key", "round"], how="left")
           .join(dsum.select(["match_key", "round", "v2", "v3", "elim_code"]), on=["match_key", "round"], how="left")
           .with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).alias("day"),
                         pl.when(pl.col("round") == 13).then(pl.lit("r13"))
                         .when(pl.col("round") == 25).then(pl.lit("r25"))
                         .when(pl.col("round") == 28).then(pl.lit("r28"))
                         .otherwise(pl.lit("other")).alias("bin")))
    tab = tab.with_columns(
        (pl.col("winner_team_code") != pl.col("s_bomb")).alias("R_ne_bomb"))
    tab = tab.with_columns(pl.lit(label).alias("year"))
    return tab


def analyse(label, tab):
    section(f"{label}: reason vs R vs bomb (full 5v5 rows, dedup)")
    n = tab.height
    eq_rr = tab.filter(pl.col("reason_side") == pl.col("winner_team_code")).height
    print(f"rows: {n}")
    print(f"reason_side == R : {eq_rr} of {n}  (not equal: {n - eq_rr})")
    ne = tab.filter(pl.col("reason_side") != pl.col("winner_team_code"))
    print("rows where reason_side != R (up to 20):")
    print(ne.select(["match_key", "round", "winner_team_code", "win_reason_code", "msg", "s_bomb", "elim_code"]).head(20))

    sb = tab.filter(pl.col("s_bomb").is_not_null())
    print(f"\nrows with in-window bomb outcome: {sb.height}")
    print("by bin: n | R==s_bomb | reason==s_bomb | reason==R")
    print(sb.group_by("bin").agg(
        pl.len().alias("n"),
        (pl.col("winner_team_code") == pl.col("s_bomb")).sum().alias("R==bomb"),
        (pl.col("reason_side") == pl.col("s_bomb")).sum().alias("reason==bomb"),
        (pl.col("reason_side") == pl.col("winner_team_code")).sum().alias("reason==R")).sort("bin"))

    flip = sb.filter(pl.col("winner_team_code") != pl.col("s_bomb"))
    ctrl = sb.filter(pl.col("winner_team_code") == pl.col("s_bomb"))
    print(f"\nDEFECT rows (R != s_bomb): {flip.height}")
    print(f"  reason==R        : {flip.filter(pl.col('reason_side') == pl.col('winner_team_code')).height}")
    print(f"  reason==s_bomb   : {flip.filter(pl.col('reason_side') == pl.col('s_bomb')).height}")
    print(f"  reason null      : {flip.filter(pl.col('reason_side').is_null()).height}")
    print("  by round bin (n, reason==R, reason==bomb):")
    print(flip.group_by("bin").agg(pl.len().alias("n"),
                                   (pl.col("reason_side") == pl.col("winner_team_code")).sum().alias("reason==R"),
                                   (pl.col("reason_side") == pl.col("s_bomb")).sum().alias("reason==bomb")).sort("bin"))
    print("  reason message counts in defect rows:")
    print(flip.group_by("msg").len().sort("len", descending=True))
    print("  defect rows: s_bomb x R:")
    print(flip.group_by(["s_bomb", "winner_team_code"]).len().sort(["s_bomb", "winner_team_code"]))
    print("  defect rows: actor code on the bomb outcome event:")
    print(flip.group_by("actor_code").len().sort("actor_code"))
    print("  defect rows: planter code:")
    print(flip.group_by("planter_code").len().sort("planter_code"))
    print("  defect rows: elim_code (all five of one side dead):")
    print(flip.group_by("elim_code").len().sort("elim_code"))
    print("  defect rows: elim_code == s_bomb (where elim known):")
    fe_known = flip.filter(pl.col("elim_code").is_not_null())
    print("   elim known:", fe_known.height,
          " elim==s_bomb:", fe_known.filter(pl.col("elim_code") == pl.col("s_bomb")).height,
          " elim==R:", fe_known.filter(pl.col("elim_code") == pl.col("winner_team_code")).height)

    print("\nCONTROL rows (R == s_bomb): reason==R =",
          ctrl.filter(pl.col("reason_side") == pl.col("winner_team_code")).height, "of", ctrl.height)
    cf = ctrl.filter(pl.col("elim_code").is_not_null())
    print("  control elim known:", cf.height, " elim==s_bomb:", cf.filter(pl.col("elim_code") == pl.col("s_bomb")).height)

    # round-label shift test: does R(r) match the bomb outcome of r-1 or r+1 (same match)?
    shift_src = sb.select(["match_key", "round", "s_bomb"])
    prev = shift_src.with_columns((pl.col("round") + 1).alias("round")).rename({"s_bomb": "sb_prev"})
    nxt = shift_src.with_columns((pl.col("round") - 1).alias("round")).rename({"s_bomb": "sb_next"})
    j = sb.join(prev, on=["match_key", "round"], how="left").join(nxt, on=["match_key", "round"], how="left")
    print("\nLabel-shift test (R(r) vs s_bomb(r-1), s_bomb(r+1)), defect rows:")
    fj = j.filter(pl.col("winner_team_code") != pl.col("s_bomb"))
    print("  defect rows with neighbour outcomes:", fj.filter(pl.col("sb_prev").is_not_null() & pl.col("sb_next").is_not_null()).height)
    print("  R==sb_prev:", fj.filter(pl.col("winner_team_code") == pl.col("sb_prev")).height,
          " R==sb_next:", fj.filter(pl.col("winner_team_code") == pl.col("sb_next")).height)
    print("  control (all rounds with bomb outcome): R(r)==sb(r-1) rate and R(r)==sb(r+1) rate")
    jj = j.filter(pl.col("sb_prev").is_not_null() & pl.col("sb_next").is_not_null())
    print("   n:", jj.height,
          " R==sb_prev:", jj.filter(pl.col("winner_team_code") == pl.col("sb_prev")).height,
          " R==sb_next:", jj.filter(pl.col("winner_team_code") == pl.col("sb_next")).height,
          " R==s_bomb:", jj.filter(pl.col("winner_team_code") == pl.col("s_bomb")).height)

    if label == "2024":
        r13 = sb.filter(pl.col("round") == 13)
        print("\nRound 13 by day (n, defect, reason==R in defect):")
        print(r13.group_by("day").agg(
            pl.len().alias("n"),
            (pl.col("winner_team_code") != pl.col("s_bomb")).sum().alias("defect"),
            ((pl.col("winner_team_code") != pl.col("s_bomb")) & (pl.col("reason_side") == pl.col("winner_team_code")))
            .sum().alias("defect_reason==R")).sort("day"))
        print("\nLegacy code in defect rows:")
        print(flip.group_by("legacy_code").len().sort("legacy_code"))
        print("\nDefect rows with reason == s_bomb (the single unequal case, if any):")
        print(flip.filter(pl.col("reason_side") == pl.col("s_bomb")).select(
            ["match_key", "round", "winner_team_code", "msg", "s_bomb", "elim_code", "planter_code", "actor_code"]))


def main():
    for label in ["2024", "2026"]:
        tab = build(label)
        analyse(label, tab)


if __name__ == "__main__":
    main()
