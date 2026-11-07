"""E4 probe: schemas, key formats, day counts for both tomes (read-only, G: only).

Run: cd G:\\pureskill\\csds-cs2 ; pixi run python -I -X utf8 scripts\\swarm\\e4_probe.py
"""
import collections
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOMES = {
    "2026": (ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds", ".2026-09-18,2026-10-08"),
    "2024": (ROOT / "tomes" / "2024-02-g" / "tome" / "csds", ".2023-12-28,2024-02-25"),
}
CHANNELS = ["round_end", "round_state", "bomb_state", "player_death", "player_info"]


def folder(tome_key, ch):
    base, suffix = TOMES[tome_key]
    return base / ("header" if ch == "header" else f"basic_{ch}{suffix}")


def pages(tome_key, ch):
    return sorted(glob.glob(str(folder(tome_key, ch) / "dataframe_*")))


def main():
    for tk in TOMES:
        print(f"\n===== tome {tk} =====")
        for ch in ["header"] + CHANNELS:
            ps = pages(tk, ch)
            if not ps:
                print(f"- {ch}: NO PAGES in {folder(tk, ch)}")
                continue
            schema = pl.read_parquet_schema(ps[0])
            print(f"- {ch}: {len(ps)} pages; first page columns: " + ", ".join(f"{k}:{v}" for k, v in schema.items()))
    # key format and day counts from header
    for tk in TOMES:
        ps = pages(tk, "header")
        hdr = pl.concat([pl.read_parquet(p) for p in ps], how="vertical_relaxed")
        keys = hdr["key"].to_list()
        print(f"\n## {tk} header rows: {hdr.height}; distinct keys {len(set(keys))}")
        print("sample keys:")
        for k in keys[:3]:
            print("  ", k)
        day = collections.Counter("/".join(k.split("/")[1:4]) for k in keys)
        print(f"key-path day counts ({len(day)} days): " + "; ".join(f"{d}={n}" for d, n in sorted(day.items())))
        if "match_date" in hdr.columns:
            md = collections.Counter(str(x)[:10] for x in hdr["match_date"].to_list())
            print("match_date day counts: " + "; ".join(f"{d}={n}" for d, n in sorted(md.items())))
        print("header columns: " + ", ".join(hdr.columns))


if __name__ == "__main__":
    main()
