import glob
import polars as pl

ROOT = "G:/pureskill/csds-cs2/tomes/"
checks = [
    "2024-02-g/tome/csds/basic_bot_takeover.2023-12-28,2024-02-25",
    "2024-02-g/tome/csds/basic_player_disconnect.2023-12-28,2024-02-25",
    "2026-10-01_to_08/tome/csds/basic_bot_takeover.2026-09-18,2026-10-08",
]
for rel in checks:
    folder = ROOT + rel
    files = sorted(glob.glob(folder + "/*"))
    print(rel)
    for f in files:
        name = f.replace("\\", "/").split("/")[-1]
        if name.startswith("dataframe") or name.startswith("keyset"):
            sch = pl.read_parquet_schema(f)
            df = pl.read_parquet(f)
            print("   ", name, "ncols=", len(sch), "cols=", list(sch.keys())[:6], "height=", df.height)
