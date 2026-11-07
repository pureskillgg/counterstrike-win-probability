"""Schema-only (column names and types) for personal-risk channels in two raw windows."""
import glob, sys
import pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
BASES = [r"G:/pureskill/csds-cs2/evac_2026-10-13/tomes/2024-02-g/tome/csds",
         r"G:/pureskill/csds-cs2/evac_2026-10-13/tomes/2026-10-01_to_08/tome/csds"]
CH = ["basic_player_chat", "basic_player_connect", "basic_player_disconnect", "basic_player_name", "basic_player_personal", "basic_player_info", "basic_player_status"]
for b in BASES:
    print("== window", b.split("/tomes/")[1].split("/tome")[0])
    for ch in CH:
        fs = sorted(glob.glob(f"{b}/{ch}.*/dataframe_*"))
        if not fs:
            print(f"  {ch}: no dataframe files"); continue
        s = pq.read_schema(fs[0])
        print(f"  {ch}: cols=" + ", ".join(f"{f.name}:{f.type}" for f in s) if len(s) else f"  {ch}: empty schema")
