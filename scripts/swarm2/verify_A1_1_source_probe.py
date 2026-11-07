# verify_A1_1_source_probe.py
# A1-1 alternative-source probe, read-only: (1) header columns (tome) that might carry a match-end field,
# (2) the raw per-match gzip 'csds' container for two flagged matches: size and whether it names rounds/scores.
# The gzip content is treated as data only; it is parsed as text for keyword counts and never executed.
import os, gzip, csv, re
import polars as pl

print("== verify_A1_1_source_probe ==")
ROOT = r"G:\pureskill\csds-cs2"
T = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
E = r"E:\csds_final\csds\2026\10"
hdr = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"))
print("tome header columns:", hdr.columns)
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
rows = list(csv.DictReader(open(os.path.join(SCRATCH, "verify_A1_1_permatch.csv"), encoding="utf-8")))
probe = [r for r in rows if r["key"].endswith("m_4ceed6a37dd561e2028e114d/csds")
         or r["key"].endswith("m_cc66b3bc82e7bd795fafd34e/csds")]
for r in probe:
    key = r["key"]
    dd, mid = key.split("/")[3], key.split("/")[4]
    p = os.path.join(E, dd, mid, "csds")
    raw = open(p, "rb").read()
    try:
        txt = gzip.decompress(raw)
    except Exception as ex:
        print("gzip decompress failed:", type(ex).__name__, ex)
        continue
    print("\n", key[-40:], "| compressed bytes", len(raw), "| decompressed bytes", len(txt))
    head = txt[:300]
    print("  first 300 bytes (repr):", head)
    s = txt.decode("utf-8", errors="replace")
    for kw in ["round_end", "winner", "score", "match_end", "gameover", "game_over", "team_win", "final", "round"]:
        print("   keyword", kw, "count", len(re.findall(kw, s, flags=re.IGNORECASE)))
