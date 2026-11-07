"""D6-4 verifier, step 4: does duplicate-match removal explain the claim's 370 / 274 counts?
Fingerprint = full round_end sequence (round, winner, reason, tick) per match; same rule as A6 (ASSUMPTION D1).
Reads the scratchpad joins from step 2 and the raw round_end tome (projection only).
"""
import sys
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
SCR = Path(r"C:\Users\13144\AppData\Local\Temp\claude\C--Users-13144-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
base = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
fd = [p for p in base.iterdir() if p.name.startswith("basic_round_end.")][0]
re_ = pl.scan_parquet(str(fd / "dataframe_*")).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()
fp = (re_.sort(["match_key", "round"]).group_by("match_key", maintain_order=True)
      .agg(pl.concat_str([pl.col("round").cast(pl.Utf8), pl.col("tick").cast(pl.Utf8), pl.col("winner_team_code").cast(pl.Utf8), pl.col("win_reason_code").cast(pl.Utf8)], separator=":").alias("seq"))
      .with_columns(pl.col("seq").list.join("|").hash().alias("fp")))
print("matches:", fp.height, " distinct fingerprints:", fp["fp"].n_unique())
keep_first = fp.group_by("fp").agg(pl.col("match_key").sort().first().alias("keep"), pl.len().alias("copies"))
print("fingerprint groups with >1 copy:", keep_first.filter(pl.col("copies") > 1).height,
      " non-first copies:", int((keep_first["copies"] - 1).sum()))
dedup_keys = set(keep_first["keep"].to_list())
j = pl.read_parquet(str(SCR / "verify_D6_4_join_2024.parquet"))
j = j.join(fp.select(["match_key", "fp"]), on="match_key", how="left")
jd = j.filter(pl.col("match_key").is_in(list(dedup_keys)))
for label, df in [("raw", j), ("dedup", jd)]:
    r13 = df.filter((pl.col("player_count") == 20) & (pl.col("round") == 13) & pl.col("bomb_side").is_not_null() & pl.col("day").is_in(["12","13","14","15","16","17","18","19","20","21","22","23","24","25"]))
    aff = r13.filter(pl.col("bomb_side") != pl.col("winner_team_code"))
    ctl = r13.filter(pl.col("bomb_side") == pl.col("winner_team_code"))
    print(f"[{label}] d12-25 round13 bomb rows={r13.height} affected={aff.height} control={ctl.height}")
    print(f"[{label}] affected: score known={aff.filter(pl.col('score_side').is_not_null()).height}, score==bomb={(aff['score_side']==aff['bomb_side']).sum()}, score!=bomb={(aff.filter(pl.col('score_side').is_not_null())['score_side']!=aff.filter(pl.col('score_side').is_not_null())['bomb_side']).sum()}")
    cu = ctl.filter(pl.col("score_side").is_not_null())
    print(f"[{label}] control: score known={cu.height}, score==bomb==round_end={(cu['score_side']==cu['bomb_side']).sum()}")
