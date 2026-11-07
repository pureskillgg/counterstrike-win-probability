# Probe: schema of G-side tomes used for team membership and weapon names (footer only).
import glob, os
import pyarrow.parquet as pq
TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
for ch in ["player_info", "weapon_fire", "round_freeze"]:
    ds = glob.glob(os.path.join(TOME, "basic_%s*" % ch))
    for d in ds:
        pages = sorted(glob.glob(os.path.join(d, "dataframe_*")))
        print(os.path.basename(d), "pages", len(pages))
        if pages:
            g = pq.ParquetFile(pages[0])
            print(g.schema_arrow.names, "rows(page0)", g.metadata.num_rows)
