import collections
import json
import os

import polars as pl

RAW = "/home/sagemaker-user/adam/builder/raw_cache_bucket/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
inv = json.load(open(FX + "inv_matches.json"))
ov = pl.read_csv(FX + "override_rounds.csv", infer_schema_length=0).filter(pl.col("status") == "fixed")

agg = collections.Counter()
samples = []
for key in inv:
    d = RAW + os.path.dirname(key) + "/"
    rows = ov.filter(pl.col("match_key") == key).iter_rows(named=True)
    rows = list(rows)
    rl = sorted(set(int(r["round"]) for r in rows))
    pv = pl.read_parquet(d + "player_vector", columns=["round", "player_id", "player_id_fixed", "team_code"]).filter(pl.col("round").is_in(rl))
    sp = pl.read_parquet(d + "player_spawn", columns=["round", "player_id", "player_id_fixed", "player_team_code"]).filter(pl.col("round").is_in(rl))
    for r in rows:
        rd = int(r["round"])
        u = int(json.loads(r["unrostered"])[0])
        p0 = int(json.loads(r["dups"])[0])
        pvu = pv.filter((pl.col("round") == rd) & (pl.col("player_id") == u))
        agg["u_pv_rows_total"] += pvu.height
        agg["u_pv_rows_team_null"] += pvu.filter(pl.col("team_code").is_null()).height
        for tc, n in pvu.group_by("team_code").len().rows():
            agg[f"u_pv_team_{tc}"] += n
        pvf = pv.filter((pl.col("round") == rd) & (pl.col("player_id_fixed") == u))
        agg["u_pv_rows_by_fixed_id"] += pvf.height
        agg["rounds_u_pv_present"] += int(pvu.height > 0)
        spu = sp.filter((pl.col("round") == rd) & (pl.col("player_id") == u))
        agg["u_spawn_rows"] += spu.height
        agg["u_spawn_team_null"] += spu.filter(pl.col("player_team_code").is_null()).height
        agg["rounds_u_spawn_present"] += int(spu.height > 0)
        spf = sp.filter((pl.col("round") == rd) & (pl.col("player_id_fixed") == u))
        agg["u_spawn_rows_by_fixed_id"] += spf.height
        # duplicate id p0: code mix in vector and spawn
        pvp = pv.filter((pl.col("round") == rd) & (pl.col("player_id") == p0))
        codes = dict(pvp.group_by("team_code").len().rows())
        agg["p0_pv_rounds_with_both_2_and_3"] += int({2, 3}.issubset(set(k for k in codes if k is not None)))
        spp = sp.filter((pl.col("round") == rd) & (pl.col("player_id") == p0))
        agg["p0_spawn_rows"] += spp.height
        agg["p0_spawn_rounds_with_both_2_and_3"] += int({2, 3}.issubset(set(v for v in spp["player_team_code"].to_list() if v is not None)))
        if len(samples) < 3 and pvu.height:
            samples.append((key[-40:], rd, u, pvu.group_by("team_code").len().rows(), spu.select(["player_team_code"]).to_series().to_list()))

for k in sorted(agg):
    print(k, agg[k])
for s in samples:
    print("sample", s)
print("done")
