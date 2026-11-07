# Verifier for A1-3 control: do OTHER 2026 channels with *_id_fixed columns show mixed raw dtypes (double vs int64)
# across their per-match raw files? If they do but their tome pages stay uniform, the page-level promotion
# mechanism is not universal. Footer-only reads on every 6th match (deterministic sample) to bound time.
import os, glob, collections
import pyarrow.parquet as pq

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
RAW = r"E:\csds_final\csds\2026\10"
CONTROLS = {"basic_weapon_fire": "weapon_fire", "basic_molotov_fire": "molotov_fire", "basic_player_death": "player_death"}

for tome_ch, raw_ch in CONTROLS.items():
    folder = os.path.join(T, tome_ch + ".2026-09-18,2026-10-08")
    pages = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    tome_types = collections.Counter()
    for dp in pages:
        s = pq.ParquetFile(dp).schema_arrow
        for n, t in zip(s.names, s.types):
            if n.endswith("_id_fixed"):
                tome_types[(n, str(t))] += 1
    raw_types = collections.defaultdict(collections.Counter)
    missing = 0
    n_sampled = 0
    for i, dp in enumerate(pages):
        pn = os.path.basename(dp).split("_")[1]
        keys = pq.read_table(os.path.join(folder, "keyset_" + pn)).column(0).to_pylist()
        for j, k in enumerate(keys):
            if j % 6:
                continue
            parts = k.split("/")
            rp = os.path.join(RAW, parts[3], parts[4], raw_ch)
            if not os.path.exists(rp):
                missing += 1
                continue
            s = pq.ParquetFile(rp).schema_arrow
            n_sampled += 1
            for n, t in zip(s.names, s.types):
                if n.endswith("_id_fixed"):
                    raw_types[n][str(t)] += 1
    print("=" * 100)
    print(tome_ch, "tome id_fixed (name,type) page-count:", dict(tome_types))
    print(tome_ch, "raw sampled matches:", n_sampled, "missing:", missing)
    for n, c in raw_types.items():
        print(f"   raw {n}: {dict(c)}")
print("done")
