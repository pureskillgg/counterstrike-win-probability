# Probe: verify inv_primary item codes that the snapshot treats as rifles or AWP.
# Uses only the G-side basic_weapon_fire tome (player_weapon_code vs weapon_name),
# filtered to the 40 sampled matches. No E: reads.
import csv, glob, os
import pyarrow.parquet as pq
import pandas as pd

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
keys = [r["match_key"] for r in csv.DictReader(open(os.path.join(ROOT, "logs", "eda", "economy_snapshot_matches.csv"), encoding="utf-8")) if r["match_key"]]
keys = sorted(set(keys))
codes = [7, 8, 9, 10, 11, 13, 16, 17, 19, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 38, 39, 40, 60]
d = glob.glob(os.path.join(TOME, "basic_weapon_fire.*"))[0]
parts = []
for f in sorted(glob.glob(os.path.join(d, "dataframe_*"))):
    t = pq.read_table(f, columns=["match_key", "player_weapon_code", "weapon_name"],
                      filters=[("match_key", "in", keys), ("player_weapon_code", "in", codes)]).to_pandas()
    parts.append(t)
wf = pd.concat(parts, ignore_index=True)
print("matches in sample:", len(keys), "weapon_fire rows with listed codes:", len(wf))
ct = wf.groupby(["player_weapon_code", "weapon_name"]).size().rename("rows").reset_index()
print(ct.to_string(index=False))
print("codes with no weapon_fire rows in sample:", sorted(set(codes) - set(wf["player_weapon_code"].unique().tolist())))
