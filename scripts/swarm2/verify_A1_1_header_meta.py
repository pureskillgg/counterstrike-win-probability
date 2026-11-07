# verify_A1_1_header_meta.py
# Alternative-explanation test: do header metadata fields (build, protocol, server, round-time settings,
# platform, match_type) separate the flagged matches (B+C) from agreeing matches (A), or key-days 01-03 from 04-08?
import os, csv, collections, glob
import polars as pl
ROOT = r"G:\pureskill\csds-cs2"
T = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
cat = {r["key"]: (r["cat"], r["day"]) for r in csv.DictReader(open(os.path.join(SCRATCH, "verify_A1_1_permatch.csv"), encoding="utf-8"))}
cols = ["build_num", "demo_version_name", "network_protocol", "rushb_version", "ppp_version", "server_name",
        "freeze_time", "round_time_defuse", "bomb_time", "max_unique_players", "platform", "match_type",
        "is_gotv_recording", "game", "tick_rate", "max_rounds", "map_name"]
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"))
cols = [c for c in cols if c in h.columns]
print("== verify_A1_1_header_meta ==")
print("columns checked:", cols)
groups = collections.defaultdict(list)
for row in h.select(["key"] + cols).iter_rows(named=True):
    c, d = cat[row["key"]]
    gname = "flag" if c in ("B_sum_eq_split_diff", "C_sum_short") else "agree"
    dayg = "day01-03" if d in ("01", "02", "03") else "day04-08"
    groups[(gname, dayg)].append(row)
for col in cols:
    print("\n-- field:", col)
    for g in sorted(groups):
        cnt = collections.Counter(str(r[col]) for r in groups[g])
        print("   ", g, "n=", len(groups[g]), "top values:", cnt.most_common(4))
