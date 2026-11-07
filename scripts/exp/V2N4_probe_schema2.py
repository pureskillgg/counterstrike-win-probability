import glob, os
import polars as pl

BASE = r"G:\pureskill\csds-cs2\tomes\combined_test\tome\csds"
SAMPLE = r"G:\pureskill\csds-cs2\data\state_sample"

def pages(channel):
    d = os.path.join(BASE, f"basic_{channel}.COMBINED")
    return sorted(glob.glob(os.path.join(d, "dataframe_*")))

for ch in ["round_end", "player_death"]:
    lf = pl.concat([pl.scan_parquet(p) for p in pages(ch)], how="diagonal_relaxed")
    sc = lf.collect_schema()
    print("==", ch, "cols:", sc.names())

re_ = pl.concat([pl.read_parquet(p) for p in pages("round_end")], how="diagonal_relaxed")
print("round_end rows", re_.height, "cols", re_.columns)
print("round_end sample:")
print(re_.head(2).to_dicts())
print("player_count distribution (2026):")
re26 = re_.filter(pl.col("match_key").str.starts_with("csds/2026/"))
print(re26.group_by("player_count").len().sort("player_count").to_dicts())
print("2026 round_end rows", re26.height, "matches", re26["match_key"].n_unique())
print("team/winner columns sample values:")
for c in ["winner_side", "winner", "team_code", "win_side", "reason", "winner_team_code"]:
    if c in re26.columns:
        print(c, re26.group_by(c).len().to_dicts())

for f in ["players_1hz.parquet", "rounds.parquet"]:
    p = os.path.join(SAMPLE, f)
    lf = pl.scan_parquet(p)
    sc = lf.collect_schema()
    print("==", f, "cols:", sc.names())
    print({k: str(v) for k, v in sc.items()})
    print(lf.head(2).collect().to_dicts())
