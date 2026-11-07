"""verify D9-8 step 8: source provenance of day groups, raw spot check, and bomb cross-check on edge rounds.

Q8 provenance: which E: tree holds each 2024-02 day (canonical E:\\csds_final\\csds\\2024\\02\\DD vs
   tail-recovered E:\\csds_final\\csds\\csds_tail_recovered_20261011\\csds\\2024\\02\\DD). READ ONLY.
Q9 raw spot check: for match csds/2024/02/12/05d51edb-... (defect, round 13 mismatch), read the raw
   round_end and round_start channels from the tail-recovered tree and compare round 13 winner and
   tick with the tome copy.
Q10 edge-round bomb cross-check: for round 9 (d12-25), round 25/28 (d12-25) and round 27 (both groups),
   compare bomb-outcome side with round_end winner and with score_winner on the rounds that have a bomb.
Output goes to stdout (tee'd by the caller). Nothing is written under E:.
"""
import glob
import os
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(30)
pl.Config.set_fmt_str_lengths(70)

SCRATCH = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
CANON = Path(r"E:\csds_final\csds\2024\02")
TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02")

print("Q8 provenance per day folder")
for dd in ["01", "02", "06", "07", "11", "12", "13", "14", "20", "25"]:
    c = CANON / dd
    t = TAIL / dd
    nc = len(os.listdir(c)) if c.exists() else -1
    nt = len(os.listdir(t)) if t.exists() else -1
    print(f"  day {dd}: canonical entries={nc}  tail-recovered entries={nt}")

sc = pl.read_parquet(SCRATCH / "verify_D9_8_score_winner.parquet")
re_all = pl.read_parquet(SCRATCH / "verify_D9_8_round_end.parquet").filter(pl.col("player_count") == 20)
re1 = re_all.group_by(["match_key", "round"]).agg(pl.col("winner_team_code").first(), pl.col("win_reason_message").first(),
                                                  pl.col("tick").min().alias("end_tick"))
rr = pl.read_parquet(SCRATCH / "verify_D9_8_rounds.parquet")
bomb_side = rr.select(["match_key", "round", "bomb_side", "mismatch", "dgrp"])
K = (re1.join(sc.select(["match_key", "round", "score_winner"]), on=["match_key", "round"], how="left")
     .join(bomb_side, on=["match_key", "round"], how="left")
     .with_columns(pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
K = K.with_columns(pl.when(pl.col("day").cast(pl.Int32) <= 7).then(pl.lit("d01-07")).otherwise(pl.lit("d12-25")).alias("dg"))

def block(label, sub):
    s = sub.filter(pl.col("score_winner").is_not_null())
    b = sub.filter(pl.col("bomb_side").is_not_null())
    print(f"\n{label}: rounds={sub.height} scored={s.height} bomb_rounds={b.height}")
    if b.height:
        print(f"  bomb rounds: bomb==round_end: {int((b['bomb_side'] == b['winner_team_code']).sum())}  "
              f"bomb!=round_end: {int((b['bomb_side'] != b['winner_team_code']).sum())}")
    if b.height and s.height:
        bs = b.filter(pl.col("score_winner").is_not_null())
        if bs.height:
            print(f"  bomb rounds with score: score==bomb: {int((bs['score_winner'] == bs['bomb_side']).sum())}  "
                  f"score==round_end: {int((bs['score_winner'] == bs['winner_team_code']).sum())}  n={bs.height}")
    print(f"  scored: score==round_end: {int((s['score_winner'] == s['winner_team_code']).sum())}  "
          f"score!=round_end: {int((s['score_winner'] != s['winner_team_code']).sum())}")

block("Q10a round 9, d12-25", K.filter((pl.col("round") == 9) & (pl.col("dg") == "d12-25")))
block("Q10b round 25, d12-25", K.filter((pl.col("round") == 25) & (pl.col("dg") == "d12-25")))
block("Q10c round 28, d12-25", K.filter((pl.col("round") == 28) & (pl.col("dg") == "d12-25")))
block("Q10d round 27, d01-07", K.filter((pl.col("round") == 27) & (pl.col("dg") == "d01-07")))
block("Q10e round 27, d12-25", K.filter((pl.col("round") == 27) & (pl.col("dg") == "d12-25")))
block("Q10f round 13, d12-25 (control)", K.filter((pl.col("round") == 13) & (pl.col("dg") == "d12-25")))
block("Q10g round 13, d01-07 (control)", K.filter((pl.col("round") == 13) & (pl.col("dg") == "d01-07")))
block("Q10h round 12, d12-25 (control)", K.filter((pl.col("round") == 12) & (pl.col("dg") == "d12-25")))

# Q9 raw spot check
key = "csds/2024/02/12/05d51edb-04bc-420c-91b9-4bb95317ca19/csds"
mid = key.split("/")[4]
raw_match = TAIL / "12" / mid
print(f"\nQ9 raw match dir exists: {raw_match.exists()}  ({raw_match})")
if raw_match.exists():
    print("  channels:", sorted(os.listdir(raw_match)))
    for ch in ["round_end", "round_start"]:
        node = raw_match / ch
        if not node.exists():
            print(f"  {ch}: missing")
            continue
        if node.is_file():
            print(f"  {ch}: file, size={node.stat().st_size}")
            df = pl.read_parquet(node)
        else:
            files = sorted(glob.glob(str(node / "dataframe_*")))
            print(f"  {ch}: dir with {len(files)} dataframe files")
            df = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed") if files else pl.DataFrame()
        cols = [c for c in df.columns if c in ("round", "tick", "winner_team_code", "win_reason_code",
                                               "win_reason_message", "player_count")]
        print(f"  raw {ch} columns: {df.columns}")
        if df.height:
            print(df.select(cols).sort("round") if "round" in cols else df.head(5))

    # tome copy for the same match, same rounds
    tome_re = re_all.filter(pl.col("match_key") == key).select(["round", "tick", "winner_team_code", "win_reason_message"]).sort("round")
    print("\n  tome round_end (pc20) for this match:")
    print(tome_re)
    raw_re_path = raw_match / "round_end"
    if raw_re_path.is_file():
        raw_re = pl.read_parquet(raw_re_path)
        if "player_count" in raw_re.columns:
            raw_re = raw_re.filter(pl.col("player_count") == 20)
        raw_cmp = raw_re.select(["round", pl.col("tick").alias("raw_tick"), pl.col("winner_team_code").alias("raw_winner"),
                                 pl.col("win_reason_message").alias("raw_msg")]).sort("round")
        cmp = tome_re.join(raw_cmp, on="round", how="full", coalesce=True)
        print("  tome vs raw round_end comparison (round, tome winner, raw winner, tick equal):")
        print(cmp.with_columns((pl.col("tick") == pl.col("raw_tick")).alias("tick_eq"),
                               (pl.col("winner_team_code") == pl.col("raw_winner")).alias("winner_eq")).sort("round"))
