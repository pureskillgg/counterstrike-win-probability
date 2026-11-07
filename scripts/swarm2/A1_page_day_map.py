# A1: which key-days (2026/10/DD) fall in each page of the drifting channels, from the page keysets.
import os, glob, collections
import pyarrow.parquet as pq
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
for c in ["basic_item_equip", "basic_item_pickup", "basic_player_hurt"]:
    folder = os.path.join(T, c + ".2026-09-18,2026-10-08")
    for ks in sorted(glob.glob(os.path.join(folder, "keyset_*"))):
        keys = pq.read_table(ks).column(0).to_pylist()
        days = collections.Counter(k.split("/")[3] for k in keys)
        s = pq.read_schema(os.path.join(folder, "dataframe_" + os.path.basename(ks).split("_")[1]))
        types = dict(zip(s.names, [str(t) for t in s.types]))
        dk = [k for k in ("player_id_fixed", "attacker_id_fixed") if k in types]
        print(c.replace("basic_", ""), os.path.basename(ks), "matches", len(keys), "days", dict(sorted(days.items())), "| ncols", len(s.names), "| id_fixed", {k: types[k] for k in dk}, "| def_index" if "def_index" in types else "")
