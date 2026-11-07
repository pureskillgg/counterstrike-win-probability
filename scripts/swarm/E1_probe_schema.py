"""E1 probe: schema, row counts, and sample rows for the G: tomes used by the economy proxy.

Reads G: tome pages only (2026 canonical tome, days 01 to 08). No E: reads.
Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I -X utf8 scripts\\swarm\\E1_probe_schema.py
"""

import glob
import os

import pyarrow.parquet as pq

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
CHANNELS = ["item_equip", "item_pickup", "item_dropped", "item_refund", "player_info",
            "round_state", "round_start", "round_end"]


def tome_pages(channel):
    dirs = [d for d in glob.glob(os.path.join(TOME, "basic_%s.*" % channel)) if os.path.isdir(d)]
    assert len(dirs) == 1, (channel, dirs)
    return sorted(glob.glob(os.path.join(dirs[0], "dataframe_*")))


def main():
    hdr_pages = sorted(glob.glob(os.path.join(TOME, "header", "dataframe_*")))
    print("header pages:", [os.path.basename(p) for p in hdr_pages])
    for ch in CHANNELS:
        pages = tome_pages(ch)
        n_rows = 0
        for p in pages:
            n_rows += pq.ParquetFile(p).metadata.num_rows
        schema = pq.ParquetFile(pages[0]).schema_arrow
        print("=== %s pages=%d rows=%d" % (ch, len(pages), n_rows))
        print("  cols:", [(f.name, str(f.type)) for f in schema])
        tbl = pq.read_table(pages[0]).slice(0, 3).to_pandas()
        with_pd = tbl.to_string(max_colwidth=40)
        print(with_pd[:1500])


if __name__ == "__main__":
    main()
