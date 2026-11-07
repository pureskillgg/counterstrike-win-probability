# Verifier for A1-3: recompute per-page schema, row counts, key-day composition, null counts,
# ID magnitude / float exactness, and tome page timestamps for the three drifting channels.
# Reads only metadata and projected ID columns (low memory). Writes nothing; prints to stdout.
import os, json, collections
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CHANNELS = {
    "basic_item_equip": ["player_id_fixed", "def_index"],
    "basic_item_pickup": ["player_id_fixed", "def_index"],
    "basic_player_hurt": ["player_id_fixed", "attacker_id_fixed"],
}
TWO53 = 2 ** 53

grand_rows = {}
for ch, idcols in CHANNELS.items():
    folder = os.path.join(T, ch + ".2026-09-18,2026-10-08")
    tome = json.load(open(os.path.join(folder, "tome"), encoding="utf8"))
    created = {p["number"]: p["createdAt"] for p in tome["pages"]}
    pages = sorted(f for f in os.listdir(folder) if f.startswith("dataframe_"))
    print("=" * 100)
    print(ch, "| dataframe pages:", len(pages), "| tome pages:", len(tome["pages"]), "| tome isComplete:", tome.get("isComplete"))
    total_rows = 0
    for pf_name in pages:
        num = int(pf_name.split("_")[1])
        dpath = os.path.join(folder, pf_name)
        kpath = os.path.join(folder, "keyset_" + pf_name.split("_")[1])
        md = pq.ParquetFile(dpath).metadata
        sch = pq.ParquetFile(dpath).schema_arrow
        names = sch.names
        types = {n: str(t) for n, t in zip(names, sch.types)}
        total_rows += md.num_rows
        # keyset: one row per match; day = path component 3 ("csds/2026/10/DD/<uuid>/csds")
        keys = pq.read_table(kpath).column(0).to_pylist()
        days = collections.Counter(k.split("/")[3] for k in keys)
        # id columns: projected read, null counts + magnitude + float exactness
        proj = [c for c in idcols if c in names]
        tbl = pq.read_table(dpath, columns=proj)
        idinfo = []
        for c in proj:
            arr = tbl.column(c)
            nulls = arr.null_count
            vals = arr.drop_null()
            if len(vals) == 0:
                idinfo.append(f"{c}: all-null")
                continue
            mn = pc.min(vals).as_py(); mx = pc.max(vals).as_py()
            if types[c] == "double":
                np_v = vals.to_numpy()
                frac = int(np.sum(np_v != np.floor(np_v)))
                big = int(np.sum(np.abs(np_v) > TWO53))
                idinfo.append(f"{c}: double nulls={nulls} min={mn:.6g} max={mx:.6g} nonint={frac} |v|>2^53={big}")
            else:
                np_v = vals.to_numpy()
                big = int(np.sum(np.abs(np_v.astype(np.int64)) > TWO53))
                idinfo.append(f"{c}: {types[c]} nulls={nulls} min={mn} max={mx} |v|>2^53={big}")
        defi = ""
        if "def_index" in names:
            defi = f" def_index_nulls={pq.read_table(dpath, columns=['def_index']).column(0).null_count}"
        print(f"  page {num}: rows={md.num_rows} ncols={len(names)} keyset_matches={len(keys)} "
              f"days={dict(sorted(days.items()))} created={created.get(num)}")
        print(f"     def_index_present={'def_index' in names} pos={names.index('def_index') if 'def_index' in names else None}{defi}")
        print("     types:", {c: types[c] for c in proj})
        for line in idinfo:
            print("     ", line)
        print("     col_order_tail:", names[-4:])
    grand_rows[ch] = total_rows
    print(ch, "TOTAL rows from parquet metadata:", total_rows)
print("=" * 100)
print("GRAND ROWS:", grand_rows)
