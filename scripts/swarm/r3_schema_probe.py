"""R3 probe: read schema and sample rows of CSDS tome pages (G: drive only).

Purpose: check which per-player identifiers exist in the event data and whether
they are unique within a match or across matches. Reads a few rows only.
"""
import os
import sys

BASE = r"G:\pureskill\csds-cs2\tomes"
TOMES = {
    "2026_header": os.path.join(BASE, r"2026-10-01_to_08\tome\csds\header"),
    "2026_death": os.path.join(BASE, r"2026-10-01_to_08\tome\csds\basic_player_death.2026-09-18,2026-10-08"),
    "2026_info": os.path.join(BASE, r"2026-10-01_to_08\tome\csds\basic_player_info.2026-09-18,2026-10-08"),
    "2024_death": os.path.join(BASE, r"2024-02-g\tome\csds\basic_player_death.2023-12-28,2024-02-25"),
}

try:
    import pyarrow.parquet as pq
    print("pyarrow", pq.__name__, "ok")
except Exception as exc:  # noqa: BLE001
    print("pyarrow import failed:", exc)
    sys.exit(1)

for name, folder in TOMES.items():
    print("==", name, folder)
    if not os.path.isdir(folder):
        print("  missing folder")
        continue
    entries = sorted(os.listdir(folder))
    print("  entries:", entries[:12])
    for fname in entries:
        if not fname.startswith("dataframe_"):
            continue
        path = os.path.join(folder, fname)
        try:
            table = pq.read_table(path)
        except Exception as exc:  # noqa: BLE001
            print("  read failed", fname, exc)
            continue
        print("  file", fname, "rows", table.num_rows)
        for field in table.schema:
            print("    col", field.name, str(field.type))
        head = table.slice(0, 3).to_pylist()
        for row in head:
            short = {k: (str(v)[:40] if v is not None else None) for k, v in row.items()}
            print("    row", short)
        break
