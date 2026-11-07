import glob, os, collections, pyarrow.parquet as pq
root = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
def pages(ch):
    d = [x for x in os.listdir(root) if x.startswith("basic_" + ch + ".")][0]
    return sorted(glob.glob(os.path.join(root, d, "dataframe_*")))
def col(ch, c):
    out = []
    for p in pages(ch):
        out += pq.read_table(p, columns=[c]).column(c).to_pylist()
    return out
print("## kills by weapon (player_death.weapon_name, top 15)")
for w, n in collections.Counter(col("player_death", "weapon_name")).most_common(15): print(f"  {w}: {n}")
print("## weapon_fire shots by weapon (top 12)")
for w, n in collections.Counter(col("weapon_fire", "weapon_name")).most_common(12): print(f"  {w}: {n}")
for ch in ["grenade_state", "molotov_fire", "player_blind", "grenade_bounce"]:
    s = pq.read_schema(pages(ch)[0]).names
    print(f"## {ch}: columns={s[:14]}")
ev = col("grenade_state", "event_type") if "event_type" in pq.read_schema(pages("grenade_state")[0]).names else []
print("grenade_state events:", collections.Counter(ev).most_common(8))
print("player_blind rows:", sum(pq.ParquetFile(p).metadata.num_rows for p in pages("player_blind")))
print("molotov_fire rows:", sum(pq.ParquetFile(p).metadata.num_rows for p in pages("molotov_fire")))
print("grenade_bounce rows:", sum(pq.ParquetFile(p).metadata.num_rows for p in pages("grenade_bounce")))
