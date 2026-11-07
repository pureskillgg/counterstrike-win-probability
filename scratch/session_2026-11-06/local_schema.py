import sys, glob, os
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
D = r"G:/pureskill/csds-cs2/evac_2026-10-13/tomes/2026-10-01_to_08/tome/csds"
for ch in ["basic_player_name", "basic_player_personal", "basic_player_info", "basic_player_status"]:
    files = sorted(glob.glob(os.path.join(D, ch + ".*", "dataframe_*")))
    if not files:
        print(ch, "no dataframe files found")
        continue
    sch = pq.read_schema(files[0])
    md = pq.ParquetFile(files[0]).metadata
    print(f"{ch}: files={len(files)} first_file_rows={md.num_rows}")
    for f in sch:
        print(f"    {f.name}: {f.type}")
