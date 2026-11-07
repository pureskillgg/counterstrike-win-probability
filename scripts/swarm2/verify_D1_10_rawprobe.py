"""verify_D1_10_rawprobe.py -- read-only raw E: vs tome check for claim-flagged 2026 matches (stdout only).

For one claim-flagged match at each of rounds 27, 33, 39 and 45 (player_count 10, chosen from the tome by
the claim's own rule), read the raw E: channel files score_update and round_end and compare them with the
tome copy for the same match: full-match equality on the shared columns, plus rounds 26-28 detail.

Reads only. E:\\csds_final\\csds\\2026 is read-only here; the tome is read via lazy scans.
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)
pl.Config.set_fmt_str_lengths(30)

ROOT = Path(r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds")
RAW_BASE = Path(r"E:\csds_final")


def tome(ch):
    p = sorted(glob.glob(str(ROOT / f"basic_{ch}*" / "dataframe_*")))
    return pl.scan_parquet(p)


# ---- pick one claim-flagged match per target round ----
TARGETS = [27, 33, 39, 45]
re_ = (tome("round_end").select(["match_key", "round", "winner_team_code", "player_count"])
       .filter(pl.col("round").is_in(TARGETS) & (pl.col("player_count") == 10)).collect())
rs_ = (tome("round_state").select(["match_key", "round", "tick", "t_score", "ct_score"])
       .filter(pl.col("round").is_in(TARGETS)).collect())
g = (rs_.sort(["match_key", "round", "tick"]).group_by(["match_key", "round"]).agg(
        pl.col("t_score").sort_by("tick").first().alias("t_b"),
        pl.col("ct_score").sort_by("tick").first().alias("ct_b"),
        pl.col("t_score").sort_by("tick").last().alias("t_a"),
        pl.col("ct_score").sort_by("tick").last().alias("ct_a")))
fl = re_.join(g, on=["match_key", "round"], how="inner").with_columns(
    pl.when((pl.col("t_a") - pl.col("t_b")) == 1).then(pl.lit(2))
      .when((pl.col("ct_a") - pl.col("ct_b")) == 1).then(pl.lit(3))
      .otherwise(pl.lit(None, dtype=pl.Int64)).alias("claim_side")).filter(
    pl.col("claim_side").is_not_null() & (pl.col("claim_side") != pl.col("winner_team_code")))
print("claim-flagged rows at target rounds (pc10):", fl.height)

picks = []
for r in TARGETS:
    sub = fl.filter(pl.col("round") == r).sort("match_key")
    if sub.height:
        picks.append((r, sub["match_key"][0]))
print("picked matches:", [(r, mk.split("/")[-2][:8]) for r, mk in picks])

for r, mk in picks:
    print(f"\n==================== flagged round {r}: {mk} ====================")
    parts = mk.split("/")  # csds / YYYY / MM / DD / guid / csds
    raw_dir = RAW_BASE.joinpath(*parts[:5])
    print("raw dir exists:", raw_dir.exists(), raw_dir)
    for ch, cols in [("score_update", ["round", "tick", "team_code", "old_score", "new_score"]),
                     ("round_end", ["round", "tick", "winner_team_code", "win_reason_code"])]:
        raw = pl.read_parquet(raw_dir / ch).select(cols).with_columns(pl.all().cast(pl.Int64))
        tm = (tome(ch).filter(pl.col("match_key") == mk).select(cols).collect()
              .with_columns(pl.all().cast(pl.Int64)))
        same = raw.sort(["round", "tick"]).equals(tm.sort(["round", "tick"]))
        print(f"  {ch}: raw rows={raw.height} tome rows={tm.height} identical_on_{len(cols)}_cols={same}")
        if ch == "round_end":
            rr = raw.filter(pl.col("round").is_in([r]))
            print(f"    raw round_end round {r}: winner_team_code={rr['winner_team_code'].to_list()} "
                  f"win_reason_code={rr['win_reason_code'].to_list()}")
    raw_su = pl.read_parquet(raw_dir / "score_update").filter(pl.col("round").is_in([r - 1, r, r + 1])).sort(["round", "tick"])
    print(f"  raw score_update rounds {r-1}-{r+1}:")
    print(raw_su.select(["round", "tick", "team_code", "old_score", "new_score"]))
    # earliest +1 rule on the raw rows for round r
    inc = raw_su.filter((pl.col("round") == r) & ((pl.col("new_score") - pl.col("old_score")) == 1)).sort("tick")
    print(f"  raw earliest +1 side in round {r}: {inc['team_code'][0] if inc.height else None}")
print("\nDONE")
