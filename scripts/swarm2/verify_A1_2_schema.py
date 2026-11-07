# verify A1-2: schema probe for the channels needed (projection only, no full loads)
import glob, os
import polars as pl
T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
for ch in ["score_update", "round_end", "player_info", "team_change", "round_start", "round_state", "round_freeze_end" , "player_death"]:
    folder = os.path.join(T, f"basic_{ch}.2026-09-18,2026-10-08")
    if not os.path.isdir(folder):
        print("MISSING", ch); continue
    pages = sorted(glob.glob(os.path.join(folder, "dataframe_*")))
    sch = pl.scan_parquet(pages[0]).collect_schema()
    print(ch, "pages", len(pages), "cols", list(sch.items()))
    print()
h = pl.read_parquet_schema(os.path.join(T, "header", "dataframe_00000"))
print("header cols", list(h.items()))
