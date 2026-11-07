# Verifier for A1-3, alternative mechanism: does the storage class of each page track where its nulls sit
# (e.g. nulls in the first rows, which a writer's dtype inference might read)? Projected single column reads.
import os, glob
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CH = {"basic_item_equip": "player_id_fixed", "basic_item_pickup": "player_id_fixed",
      "basic_player_hurt": "attacker_id_fixed"}

for ch, idc in CH.items():
    folder = os.path.join(T, ch + ".2026-09-18,2026-10-08")
    print("=" * 100)
    print(ch, idc)
    for dp in sorted(glob.glob(os.path.join(folder, "dataframe_*"))):
        n = os.path.basename(dp)
        typ = str(pq.ParquetFile(dp).schema_arrow.field(idc).type)
        arr = pq.read_table(dp, columns=[idc]).column(0).combine_chunks()
        isnull = np.asarray(pc.is_null(arr).to_numpy(zero_copy_only=False))
        nrows = len(isnull)
        nn = int(isnull.sum())
        first_null = int(np.argmax(isnull)) if nn else None
        first_val = int(np.argmax(~isnull)) if nn < nrows else None
        print(f"{n} class={typ} rows={nrows} nulls={nn} first_null_idx={first_null} "
              f"nulls_in_first100={int(isnull[:100].sum())} nulls_in_first1000={int(isnull[:1000].sum())} "
              f"first_nonnull_idx={first_val} first_value={arr[first_val].as_py() if first_val is not None else None} "
              f"first5_null_flags={isnull[:5].astype(int).tolist()}")
print("done")
