import glob, os, collections
import pyarrow.parquet as pq
ROOT = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"
CH = ["item_equip", "item_pickup", "item_dropped", "item_refund", "rank_update"]
for ch in CH:
    pages = sorted(glob.glob(os.path.join(ROOT, f"basic_{ch}.2026-09-18,2026-10-08", "dataframe_*")))
    sigs = collections.OrderedDict()
    for p in pages:
        s = pq.ParquetFile(p).schema_arrow
        sig = tuple((f.name, str(f.type)) for f in s if not f.name.startswith("__"))
        extra = [f.name for f in s if f.name.startswith("__")]
        sigs.setdefault(sig, []).append((os.path.basename(p), pq.ParquetFile(p).metadata.num_rows, extra))
    print(f"== {ch}: {len(pages)} pages, {len(sigs)} distinct schema(s)")
    base = list(sigs.keys())[0]
    for i, (sig, pl) in enumerate(sigs.items()):
        names = [n for n, _ in sig]
        print(f"  schema {i}: {len(sig)} cols; pages={[x[0][-5:] for x in pl]}; extra_fields={pl[0][2]}")
        if sig != base:
            bset, sset = set(base), set(sig)
            print("    missing vs schema0:", sorted(bset - sset))
            print("    added vs schema0:", sorted(sset - bset))
