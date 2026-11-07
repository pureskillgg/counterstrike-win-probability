import json, os
import polars as pl

RAW = "/home/sagemaker-user/adam/builder/raw_cache_bucket/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
inv = json.load(open(FX + "inv_matches.json"))
ov = pl.read_csv(FX + "override_rounds.csv", infer_schema_length=0).with_columns(pl.col("round").cast(pl.Int64))
print("status counts:", ov.group_by("status").len().to_dicts())
print("unique (match,round) rows:", ov.select(["match_key", "round"]).unique().height, "rows:", ov.height)
print("matches in override:", ov["match_key"].n_unique(), "of", len(inv))

fixed = ov.filter(pl.col("status") == "fixed")
sizes = []
for row in fixed.iter_rows(named=True):
    o = json.loads(row["override"])
    ct = sum(1 for v in o.values() if int(v) == 3)
    t = sum(1 for v in o.values() if int(v) == 2)
    sizes.append((ct, t))
from collections import Counter
print("fixed rounds CT/T counts from override:", Counter(sizes).most_common(6))

# side flips per fixed round vs player_info roster (first code)
flip_rounds = 0
flip_rows = 0
for key in inv:
    d = RAW + os.path.dirname(key) + "/"
    pinfo = pl.read_parquet(d + "player_info", columns=["round", "player_id", "team_code"])
    roster = pinfo.unique(subset=["round", "player_id"], keep="first")
    for row in fixed.filter(pl.col("match_key") == key).iter_rows(named=True):
        r = int(row["round"])
        o = {int(k): int(v) for k, v in json.loads(row["override"]).items()}
        rr = {int(p): int(t) for p, t in roster.filter(pl.col("round") == r).select(["player_id", "team_code"]).iter_rows()}
        n = sum(1 for p, t in o.items() if rr.get(p) not in (None, t))
        flip_rows += n
        flip_rounds += 1 if n else 0
print("fixed rounds with >=1 flipped player:", flip_rounds, "flipped player-rounds:", flip_rows)
