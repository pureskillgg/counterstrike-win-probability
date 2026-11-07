"""R3 probe 4: shape-only check of steam_id in 2026 and 2024 tomes.
Prints lengths, character classes and counts only. No identifier values are printed.
"""
import collections
import os
import re

import pyarrow.parquet as pq

ROOTS = {
    "2026": r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds\basic_player_personal.2026-09-18,2026-10-08",
    "2024": r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds\basic_player_personal.2023-12-28,2024-02-25",
}

for label, folder in ROOTS.items():
    print("==", label, folder)
    if not os.path.isdir(folder):
        print("  missing folder")
        continue
    pages = sorted(f for f in os.listdir(folder) if f.startswith("dataframe_"))
    print("  pages", len(pages))
    if not pages:
        continue
    table = pq.read_table(os.path.join(folder, pages[0]), columns=["match_key", "steam_id", "is_bot"])
    sid = [str(s) if s is not None else None for s in table.column("steam_id").to_pylist()]
    mk = table.column("match_key").to_pylist()
    print("  rows", table.num_rows)
    lengths = collections.Counter(len(s) for s in sid if s is not None)
    print("  length_hist", sorted(lengths.items())[:10])
    classes = collections.Counter()
    for s in sid:
        if s is None:
            classes["null"] += 1
        elif re.fullmatch(r"\d{17}", s):
            classes["steam64_numeric"] += 1
        elif re.fullmatch(r"[0-9a-fA-F]{32,64}", s):
            classes["hex_digest_like"] += 1
        elif re.fullmatch(r"[A-Za-z]+", s):
            classes["alpha_only"] += 1
        else:
            classes["other"] += 1
    print("  classes", dict(classes))
    counts = collections.Counter(sid)
    print("  distinct_values", len(counts), "top_share", round(counts.most_common(1)[0][1] / len(sid), 4) if sid else None)
    per_match = collections.defaultdict(set)
    for m, s in zip(mk, sid):
        per_match[m].add(s)
    print("  matches_in_page", len(per_match), "distinct_per_match_hist", sorted(collections.Counter(len(v) for v in per_match.values()).items())[:8])
