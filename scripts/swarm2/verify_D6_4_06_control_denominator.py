"""D6-4 verifier, step 6: control denominator if round-13 controls from all days are pooled, deduplicated.
Only prints counts. Reads the scratchpad join from step 2 and the 2024 round_end tome (projection only).
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
base = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
fd = [p for p in base.iterdir() if p.name.startswith("basic_round_end.")][0]
re_ = pl.scan_parquet(str(fd / "dataframe_*")).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()
fp = (re_.sort(["match_key", "round"]).group_by("match_key", maintain_order=True)
      .agg(pl.concat_str([pl.col("round").cast(pl.Utf8), pl.col("tick").cast(pl.Utf8),
                          pl.col("winner_team_code").cast(pl.Utf8), pl.col("win_reason_code").cast(pl.Utf8)], separator=":").alias("seq"))
      .with_columns(pl.col("seq").list.join("|").hash().alias("fp")))
keep = fp.group_by("fp").agg(pl.col("match_key").sort().first().alias("keep"))
dk = set(keep["keep"].to_list())
j = pl.read_parquet(str(SCR / "verify_D6_4_join_2024.parquet")).filter(pl.col("player_count") == 20)
r13 = j.filter((pl.col("round") == 13) & pl.col("bomb_side").is_not_null())
for label, df in [("all days, raw", r13), ("all days, dedup", r13.filter(pl.col("match_key").is_in(list(dk))))]:
    ctl = df.filter(pl.col("bomb_side") == pl.col("winner_team_code"))
    aff = df.filter(pl.col("bomb_side") != pl.col("winner_team_code"))
    cs = ctl.filter(pl.col("score_side").is_not_null())
    print(f"{label}: affected={aff.height}, control={ctl.height}, control score known={cs.height}, "
          f"control score==bomb={(cs['score_side'] == cs['bomb_side']).sum()}")
