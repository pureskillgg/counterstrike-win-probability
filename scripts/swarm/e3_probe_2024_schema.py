"""E3 probe: column names and row counts for the channels the kill analysis reads, on the
2024 and 2026 G: tomes. Reads parquet schemas and row counts only (no E: access, no writes)."""
import glob
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}
CH = ["header", "round_end", "round_start", "round_state", "player_death", "player_info",
      "bomb_state", "player_disconnect", "bot_takeover", "team_change"]

for era, (base, suffix) in TOMES.items():
    print(f"===== {era}")
    for ch in CH:
        fd = base / "header" if ch == "header" else base / f"basic_{ch}{suffix}"
        if not fd.exists():
            print(f"  {ch}: FOLDER MISSING ({fd.name})")
            continue
        files = sorted(glob.glob(str(fd / "dataframe_*")))
        if not files:
            print(f"  {ch}: no dataframe files")
            continue
        rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        names = pq.read_schema(files[0]).names
        keys = sorted(glob.glob(str(fd / "keyset_*")))
        kn = pq.read_schema(keys[0]).names if keys else None
        print(f"  {ch}: files={len(files)} rows={rows} cols={names} keyset_cols={kn}")
