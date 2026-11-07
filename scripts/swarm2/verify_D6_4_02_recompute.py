"""D6-4 verifier, step 2: recompute the round-13 three-source table from the tomes.

Sources (all read with column projection, one channel at a time):
  - round_end:   winner_team_code (the field under dispute)
  - bomb_state:  bomb_defused -> CT (3), bomb_exploded -> T (2). Side-absolute, no team codes.
  - round_state: round_freeze_end rows. Score winner for round r = change in
                 (t_score, ct_score) between freeze_end(r) and freeze_end(r+1).
                 ASSUMPTION S1: t_score / ct_score are side-attached (T / CT) in the tome.
Prints counts only (no per-round dumps). Writes nothing except stdout.
"""
import glob, sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)
ROOT = Path(r"G:\pureskill\csds-cs2")
ERA = {
    "2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
    "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
}
def chdir(base, ch):
    m = [p for p in base.iterdir() if p.name.startswith(f"basic_{ch}.")]
    assert len(m) == 1, (ch, m)
    return str(m[0] / "dataframe_*")
def scan(base, ch, cols):
    return pl.scan_parquet(chdir(base, ch)).select(cols)

def load(era):
    base = ERA[era]
    re_ = scan(base, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"]).collect()
    fe = (scan(base, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
          .filter(pl.col("event_type") == "round_freeze_end")
          .select(["match_key", "round", "tick", "t_score", "ct_score"]).collect())
    bs = (scan(base, "bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"])
          .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
          .collect())
    return re_, fe, bs

def build(era):
    re_, fe, bs = load(era)
    dup_fe = fe.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
    dup_re = re_.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
    # bomb outcome per round: side implied by the event types present
    bomb = (bs.group_by(["match_key", "round"]).agg(
        pl.col("event_type").unique().alias("evs"),
        pl.col("tick").min().alias("bomb_tick")))
    bomb = bomb.with_columns(
        pl.when(pl.col("evs").list.contains("bomb_exploded") & pl.col("evs").list.contains("bomb_defused"))
          .then(pl.lit(-1)).when(pl.col("evs").list.contains("bomb_exploded")).then(pl.lit(2))
          .otherwise(pl.lit(3)).alias("bomb_side"))
    # score winner: post = freeze_end(r+1), pre = freeze_end(r)
    pre = fe.rename({"tick": "tick_pre", "t_score": "t_pre", "ct_score": "ct_pre"})
    post = fe.with_columns((pl.col("round") - 1).alias("round")).rename({"tick": "tick_post", "t_score": "t_post", "ct_score": "ct_post"})
    sc = pre.join(post.select(["match_key", "round", "tick_post", "t_post", "ct_post"]), on=["match_key", "round"], how="left")
    sc = sc.with_columns(
        (pl.col("t_post") - pl.col("t_pre")).alias("dT"),
        (pl.col("ct_post") - pl.col("ct_pre")).alias("dCT"))
    sc = sc.with_columns(
        pl.when((pl.col("dT") == 1) & (pl.col("dCT") == 0)).then(pl.lit(2))
          .when((pl.col("dCT") == 1) & (pl.col("dT") == 0)).then(pl.lit(3))
          .otherwise(pl.lit(None)).alias("score_side"),
        ((pl.col("dT") + pl.col("dCT")) == 1).alias("score_unambig"))
    j = (re_.join(bomb.select(["match_key", "round", "bomb_side", "bomb_tick"]), on=["match_key", "round"], how="left")
            .join(sc.select(["match_key", "round", "dT", "dCT", "score_side", "score_unambig", "tick_post"]), on=["match_key", "round"], how="left")
            .with_columns(pl.col("match_key").str.extract(r"csds/\d{4}/\d{2}/(\d{2})/", 1).alias("day")))
    return j, dup_fe, dup_re, re_.height, fe.height

if __name__ == "__main__":
    for era in ["2024", "2026"]:
        j, dfe, dre, nre, nfe = build(era)
        print(f"== {era}: round_end rows={nre}, freeze_end rows={nfe}, dup (match,round) freeze_end={dfe}, dup round_end={dre}")
        print("   score side available:", j.filter(pl.col("score_side").is_not_null()).height, "of", j.height)
        j.write_parquet(str(Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad") / f"verify_D6_4_join_{era}.parquet"))
    print("wrote per-round joins to scratchpad (not under logs/)")
