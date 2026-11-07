import glob, polars as pl
BASE = "G:/pureskill/csds-cs2/tomes/combined_test/tome/csds"
def scan(ch):
    return sorted(glob.glob(BASE + "/basic_" + ch + ".COMBINED/dataframe_*"))
for ch in ["item_dropped", "item_pickup", "player_death", "round_end", "item_equip"]:
    files = scan(ch)
    print("==", ch, "pages:", len(files))
    lf = pl.scan_parquet(files)
    sch = lf.collect_schema()
    print({k: str(v) for k, v in sch.items()})
    df = lf.head(5).collect()
    keep = [c for c in df.columns if not c.startswith("player_") or c in ("player_id","player_id_fixed","player_team_code","player_weapon_code")]
    print(df.select(keep))
    n = lf.select(pl.len()).collect().item()
    print("rows:", n)
