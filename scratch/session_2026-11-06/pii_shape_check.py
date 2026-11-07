import re
import sys
from collections import Counter

import pyarrow.parquet as pq

# Read-only shape check on train and val matches. Prints counts and shape
# classes only (letters -> a/A, digits -> 9). No raw cell value is printed.
paths_file = sys.argv[1]
with open(paths_file, encoding="utf-8") as fh:
    paths = [line.strip() for line in fh if line.strip()]


def shape(s):
    out = []
    for ch in s:
        if ch.isdigit():
            out.append("9")
        elif ch.isalpha() and ch.islower():
            out.append("a")
        elif ch.isalpha():
            out.append("A")
        else:
            out.append(ch)
    # collapse runs so the shape stays generic
    res = []
    for ch in out:
        if not res or res[-1] != ch or ch not in ("9", "a", "A"):
            res.append(ch)
    return "".join(res)


steam_vals = Counter()
name_vals = Counter()
clan_vals = Counter()
pair_vals = Counter()
steam_len = Counter()
name_len = Counter()
steam_shape = Counter()
name_shape = Counter()
clan_shape = Counter()
placeholder_hits = Counter()
hex_like = 0
pid_per_file = []
steam_matches_pid_slot = 0
rows = 0
bot_rows = 0
files = 0

PLACEHOLDER = re.compile(r"(?i)(player|anon|redact|unknown|bot|slot|pseud|hash|null|none|n/?a|tbd|team|ct|t)[ _\-]?\d*$")
HEX = re.compile(r"^[0-9a-fA-F]{32,128}$")

for path in paths:
    try:
        tbl = pq.read_table(path)
    except Exception as exc:
        placeholder_hits["read_error_" + type(exc).__name__] += 1
        continue
    files += 1
    try:
        pn = pq.read_table(path.rsplit("/", 1)[0] + "/player_name")
        placeholder_hits["player_name_rows_total"] += pn.num_rows
        if pn.num_rows > 0:
            placeholder_hits["player_name_files_with_rows"] += 1
    except FileNotFoundError:
        placeholder_hits["player_name_missing"] += 1
    except Exception as exc:
        placeholder_hits["player_name_read_error_" + type(exc).__name__] += 1
    sid = tbl.column("steam_id").to_pylist()
    nm = tbl.column("name").to_pylist()
    ct = tbl.column("clan_tag").to_pylist()
    bots = tbl.column("is_bot").to_pylist()
    pids = tbl.column("player_id").to_pylist()
    pid_per_file.append(len(set(pids)))
    for s, n, c, b, p in zip(sid, nm, ct, bots, pids):
        rows += 1
        if b:
            bot_rows += 1
        steam_vals[s] += 1
        name_vals[n] += 1
        clan_vals[c] += 1
        pair_vals[(s, n)] += 1
        if s is not None:
            steam_len[len(s)] += 1
            steam_shape[shape(s)] += 1
            if HEX.match(s):
                hex_like += 1
        if n is not None:
            name_len[len(n)] += 1
            name_shape[shape(n)] += 1
            if PLACEHOLDER.search(n.strip()):
                placeholder_hits["name_placeholder_like"] += 1
            if " " in n.strip():
                placeholder_hits["name_contains_space"] += 1
            if any(ord(ch) > 127 for ch in n):
                placeholder_hits["name_non_ascii"] += 1
        if c is not None:
            clan_shape[shape(c)] += 1

print("files:", files, "rows:", rows, "bot_rows:", bot_rows)
print("distinct steam_id values:", len(steam_vals))
print("distinct name values:", len(name_vals))
print("distinct clan_tag values:", len(clan_vals))
print("distinct (steam_id, name) pairs:", len(pair_vals))
print("distinct player_id per file (min, max):", min(pid_per_file) if pid_per_file else None, max(pid_per_file) if pid_per_file else None)
print("steam_id length counts:", dict(sorted(steam_len.items())))
print("steam_id shape classes (top 10, shape: count):")
for k, v in steam_shape.most_common(10):
    print("   ", repr(k), v)
print("name length counts (top 10 lengths):", steam_len and dict(name_len.most_common(10)))
print("name shape classes (top 10):")
for k, v in name_shape.most_common(10):
    print("   ", repr(k), v)
print("clan_tag shape classes (top 10):")
for k, v in clan_shape.most_common(10):
    print("   ", repr(k), v)
print("steam_id hex-like rows:", hex_like)
print("rows per distinct steam_id value (sorted desc, counts only):", sorted(steam_vals.values(), reverse=True)[:12])
print("rows per distinct name value (sorted desc, counts only, top 12):", sorted(name_vals.values(), reverse=True)[:12])
print("rows per distinct clan_tag value (top 12):", sorted(clan_vals.values(), reverse=True)[:12])
for k, v in sorted(placeholder_hits.items()):
    print(k + ":", v)
