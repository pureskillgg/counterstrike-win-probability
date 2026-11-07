# verify_A1_3_dupcheck.py
# Side check for A1-3: the day-08 item_equip row inflation (about 1.7x rows per round versus days 06-07, same raw schema)
# could be duplicated events rather than real equips. Samples raw per-match item_equip files (read-only, full read of
# small files) for days 06 and 08 and counts exact duplicate rows and duplicate event keys.
# ASSUMPTION: a sample of 40 matches per day (first 40 in sorted uuid order) is representative of the day.
import os, glob, random
import polars as pl
import pyarrow.parquet as pq

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
RAW = "E:/csds_final/csds/2026/10"
folder = os.path.join(TOME, "basic_item_equip.2026-09-18,2026-10-08")
keys = []
for k in sorted(glob.glob(os.path.join(folder, "keyset_*"))):
    keys += pq.read_table(k).column(0).to_pylist()
by_day = {}
for k in keys:
    parts = k.split("/")
    by_day.setdefault(parts[3], []).append((parts[4], parts[3]))
for day in ["06", "08"]:
    sample = sorted(by_day[day])[:40]
    rows = dup_full = dup_key = 0
    per_match_dup = []
    for uuid, d in sample:
        p = f"{RAW}/{d}/{uuid}/item_equip"
        if not os.path.exists(p):
            continue
        df = pl.read_parquet(p)
        n = df.height
        if n == 0:
            continue
        nf = df.is_duplicated().sum()
        keycols = [c for c in ["round", "tick", "player_id", "item", "player_id_fixed", "item_type"] if c in df.columns]
        nk = df.select(keycols).is_duplicated().sum() if keycols else 0
        rows += n
        dup_full += int(nf)
        dup_key += int(nk)
        per_match_dup.append(int(nf) / n)
    print(f"day {day}: sampled_matches={len(sample)} rows={rows} exact_duplicate_rows={dup_full} "
          f"({100.0 * dup_full / max(rows, 1):.2f}%) duplicate_event_keys={dup_key} ({100.0 * dup_key / max(rows, 1):.2f}%) "
          f"key_cols={keycols} median_match_dup_share={sorted(per_match_dup)[len(per_match_dup)//2] if per_match_dup else None}")
print("DONE dupcheck")
