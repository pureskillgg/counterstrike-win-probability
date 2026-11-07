import json
import os

import polars as pl

RAW = "/home/sagemaker-user/adam/builder/raw_cache_bucket/"
FX = "/home/sagemaker-user/adam/builder/fixes/freeze_fix/"
OUT = "/home/sagemaker-user/adam/builder/fixes/fix332/"
inv = json.load(open(FX + "inv_matches.json"))
ovall = pl.read_csv(FX + "override_rounds.csv", infer_schema_length=0)
nf = ovall.filter(pl.col("status") != "fixed")
print("not_fixed rows:", nf.select(["match_key", "round", "freeze", "alive", "dups", "unrostered", "reason"]).rows())
fixed = ovall.filter(pl.col("status") == "fixed")

found = []
for key in inv:
    d = RAW + os.path.dirname(key) + "/"
    rows = list(fixed.filter(pl.col("match_key") == key).iter_rows(named=True))
    if not rows:
        continue
    rl = sorted(set(int(r["round"]) for r in rows))
    dt = pl.read_parquet(d + "player_death", columns=["round", "tick", "player_id", "attacker_id"]).filter(pl.col("round").is_in(rl))
    for r in rows:
        rd = int(r["round"])
        ovmap = {int(k): int(v) for k, v in json.loads(r["override"]).items()}
        u = int(json.loads(r["unrostered"])[0])
        p0 = int(json.loads(r["dups"])[0])
        for rdd, tick, vic, att in dt.filter(pl.col("round") == rd).select(["round", "tick", "player_id", "attacker_id"]).rows():
            if vic is None or att is None or int(att) == int(vic):
                continue
            vic, att = int(vic), int(att)
            if u not in (vic, att) and p0 not in (vic, att):
                continue
            if vic in ovmap and att in ovmap and ovmap[vic] == ovmap[att]:
                found.append((key.split("/")[3][:8], rd, tick, "att", att, ovmap[att], "vic", vic, ovmap[vic],
                              "u" if u in (att, vic) else "", "p0" if p0 in (att, vic) else ""))
print("same-side kills under override reference:", len(found))
for f in found:
    print(f)

# where does team_code come from: grep the build scripts for the join (read-only)
os.system("grep -n 'team_code' /home/sagemaker-user/adam/builder/prepare.py | head -20")
os.system("grep -rln 'player_info' /home/sagemaker-user/adam/builder/*.py | head -10")
print("done")
