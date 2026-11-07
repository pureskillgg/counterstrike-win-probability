"""D3-6 independent verification, part 2 (read-only): round numbering and the round-1 question.

1. Recomputes round_end round-numbering classes for every 2024-02-g match from the G: tome (own code).
2. For every header match, compares the G: round_end round set with the E: source round_end:
   tail copy C if its round_end file is non-zero, else copy A (E:\\csds_extracted) if non-zero.
3. For matches whose G: rounds start at 2, checks whether round 1 appears in the E: source round_end,
   round_start and round_state, and in the G: round_start, round_state, bomb_state and bomb_action tables.
4. Byte-compares copy A round_end with the merged_2024 round_end that the -g tome was built from (day 01).
5. Control: round numbering in the 2026-10-01_to_08 tome.
6. Compares RAR salvage index uuids with the G: header uuids.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_6_rounds.py
"""
import collections
import glob
import os
import sys
import time
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time()
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME_G = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
TOME_26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
E_TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds")
E_A = Path(r"E:\csds_extracted\csds")
MERGED = ROOT / "merged_2024" / "csds"
TSV = Path(r"E:\csds_rar5_tail_salvage_20261011.tsv")


def files_of(folder):
    return sorted(glob.glob(str(folder / "dataframe_*")))


def read_cols(folder, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files_of(folder)], how="vertical_relaxed")


def round_sets(df, key_col="match_key"):
    out = collections.defaultdict(set)
    dup = collections.Counter()
    for mk, r in df.select([key_col, "round"]).iter_rows():
        if r is None:
            continue
        if (mk, r) in dup:
            dup[(mk, r)] += 1
        else:
            dup[(mk, r)] = 1
        out[mk].add(r)
    n_dup_pairs = sum(1 for v in dup.values() if v > 1)
    return out, n_dup_pairs


def cls(rs):
    rs = sorted(rs)
    n = len(rs)
    if rs == list(range(1, n + 1)):
        return "contig_from_1"
    if n and rs == list(range(rs[0], rs[0] + n)) and rs[0] > 1:
        return f"contig_from_{rs[0]}"
    return "other"


def e_base(key, root):
    p = key.split("/")
    return root / p[1] / p[2] / p[3] / p[4]


def nonzero(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def read_rounds_e(path):
    df = pl.read_parquet(path, columns=["round"])
    return set(r for r in df.get_column("round").to_list() if r is not None)


# ---------------------------------------------------------------- header and G: round_end
H = read_cols(TOME_G / "header", ["key", "match_date"])
header_keys = H["key"].to_list()
mdate = dict(H.select(["key", "match_date"]).iter_rows())
header_set = set(header_keys)

RE_G = read_cols(TOME_G / f"basic_round_end{SUF}", ["match_key", "round"])
G_sets, n_dup = round_sets(RE_G)
print(f"G: round_end rows {RE_G.height}; matches with round_end {len(G_sets)}; "
      f"(match, round) pairs with duplicate rows {n_dup}")
print(f"header keys without round_end: {len(header_set - set(G_sets))}; round_end keys not in header: "
      f"{len(set(G_sets) - header_set)}")

classes = {k: cls(v) for k, v in G_sets.items()}
print("G: round numbering classes (all matches with round_end):", dict(collections.Counter(classes.values())))
print("G: classes by key-day:")
by_kd = collections.defaultdict(collections.Counter)
for k, c in classes.items():
    by_kd[k.split("/")[3]][c] += 1
for d in sorted(by_kd):
    print(f"  {d}: " + ", ".join(f"{c}={n}" for c, n in sorted(by_kd[d].items())))

start2 = sorted(k for k, c in classes.items() if c == "contig_from_2")
c1 = sorted(k for k, c in classes.items() if c == "contig_from_1")
print(f"\nstart-at-2 matches: {len(start2)}; contiguous-from-1 matches: {len(c1)}")
print("max round / n rounds distribution, start-at-2 (top 8 by max):",
      collections.Counter(max(G_sets[k]) for k in start2).most_common(8))
print("max round / n rounds distribution, contiguous-from-1 (top 8 by max):",
      collections.Counter(max(G_sets[k]) for k in c1).most_common(8))
print("n rounds per match, start-at-2 (top 8):", collections.Counter(len(G_sets[k]) for k in start2).most_common(8))
print("n rounds per match, contiguous-from-1 (top 8):", collections.Counter(len(G_sets[k]) for k in c1).most_common(8))

# start-at-2 rate by UTC match_date (not the folder day)
md_rate = collections.defaultdict(lambda: [0, 0])
for k in header_keys:
    md = mdate.get(k)
    dkey = md[:10] if md else "NULL"
    md_rate[dkey][0] += 1
    if classes.get(k) == "contig_from_2":
        md_rate[dkey][1] += 1
print("\nstart-at-2 by UTC match_date (date: matches, start-at-2):")
print("  " + "; ".join(f"{d}: {n}/{s}" for d, (n, s) in sorted(md_rate.items())))

# ---------------------------------------------------------------- G: other channels for start-at-2 matches
s2set = set(start2)
g_other = {}
for ch in ["round_start", "round_state", "bomb_state", "bomb_action"]:
    folder = TOME_G / f"basic_{ch}{SUF}"
    df = read_cols(folder, ["match_key", "round"]).filter(pl.col("match_key").is_in(start2))
    sets, _ = round_sets(df)
    g_other[ch] = sets
    n_has1 = sum(1 for k in start2 if 1 in sets.get(k, set()))
    n_any = sum(1 for k in start2 if k in sets)
    print(f"G: {ch:<12} start-at-2 matches with any rows: {n_any}; with round 1 present: {n_has1}")
print(f"  ({time.time() - t0:.0f}s)")

# ---------------------------------------------------------------- E: source round_end for all matches
src = {}
for k in header_keys:
    tb = e_base(k, E_TAIL)
    ab = e_base(k, E_A)
    if nonzero(str(tb / "round_end")):
        src[k] = ("tail", tb)
    elif nonzero(str(ab / "round_end")):
        src[k] = ("A", ab)
    else:
        src[k] = (None, None)
print("\nE: round_end source per header match:", dict(collections.Counter(s for s, _ in src.values())))
print("E: round_end source by key-day:")
sd = collections.defaultdict(collections.Counter)
for k, (s, _) in src.items():
    sd[k.split("/")[3]][s] += 1
print("  " + "; ".join(f"{d}: " + ",".join(f"{s}={n}" for s, n in sorted(v.items(), key=lambda x: str(x[0])))
                       for d, v in sorted(sd.items())))

eq = collections.Counter()
neq_ex = []
e_has1_re = []
e_round_sets = {}
for k in header_keys:
    s, base = src[k]
    if s is None:
        eq["no_E_round_end"] += 1
        continue
    try:
        es = read_rounds_e(str(base / "round_end"))
    except Exception as ex:  # noqa: BLE001
        eq["E_read_error"] += 1
        if len(neq_ex) < 3:
            neq_ex.append((k, repr(ex)[:120]))
        continue
    e_round_sets[k] = es
    gs = G_sets.get(k, set())
    if es == gs:
        eq["G_equals_E"] += 1
    else:
        eq["G_differs_E"] += 1
        if len(neq_ex) < 6:
            neq_ex.append((k, sorted(gs)[:5], sorted(es)[:5], s))
    if k in s2set and 1 in es:
        e_has1_re.append(k)
print(f"G: round set vs E: round_end round set, all header matches: {dict(eq)}")
print("  examples:", neq_ex)
print(f"start-at-2 matches whose E: round_end contains round 1: {len(e_has1_re)} of {len(start2)}")
print(f"  ({time.time() - t0:.0f}s)")

# round_start / round_state in E: for start-at-2 matches
e_chan_has1 = {}
for ch in ["round_start", "round_state"]:
    n_has1, n_read, n_err = 0, 0, 0
    ex = []
    for k in start2:
        s, base = src[k]
        if s is None:
            continue
        p = base / ch
        if not nonzero(str(p)):
            continue
        try:
            rs = read_rounds_e(str(p))
            n_read += 1
            if 1 in rs:
                n_has1 += 1
                if len(ex) < 3:
                    ex.append((k, sorted(rs)[:4]))
        except Exception as e:  # noqa: BLE001
            n_err += 1
    e_chan_has1[ch] = n_has1
    print(f"E: {ch:<12} start-at-2 matches read {n_read}; with round 1 present {n_has1}; read errors {n_err}; ex {ex}")
print(f"  ({time.time() - t0:.0f}s)")

# ---------------------------------------------------------------- start-at-2 matches with no E: round_end at all
no_e = [k for k in start2 if src[k][0] is None]
print(f"\nstart-at-2 matches with no E: round_end in either copy: {len(no_e)}")

# ---------------------------------------------------------------- day-01 copy A vs merged_2024 (feeds the -g tome)
d01_tail_empty = [k for k in header_keys if k.split("/")[3] == "01" and src[k][0] == "A"]
same_bytes = 0
diff_bytes = []
for k in d01_tail_empty:
    u = k.split("/")[4]
    a = E_A / "2024" / "02" / "01" / u / "round_end"
    m = MERGED / "2024" / "02" / "01" / u / "round_end"
    if not (os.path.isfile(a) and os.path.isfile(m)):
        diff_bytes.append((u, "missing"))
        continue
    with open(a, "rb") as fa, open(m, "rb") as fm:
        if fa.read() == fm.read():
            same_bytes += 1
        else:
            diff_bytes.append((u, "bytes_differ"))
print(f"day-01 round_end: copy A vs merged_2024 byte-identical for {same_bytes} of {len(d01_tail_empty)} "
      f"copy-A-sourced matches; differences: {diff_bytes[:3]}")

# ---------------------------------------------------------------- control: 2026 tome round numbering
folders_26 = sorted(glob.glob(str(TOME_26 / "basic_round_end*")))
print(f"\n2026 control folders: {[Path(f).name for f in folders_26]}")
if folders_26:
    RE26 = pl.concat([pl.read_parquet(f, columns=["match_key", "round"]) for f in sorted(glob.glob(os.path.join(folders_26[0], "dataframe_*")))], how="vertical_relaxed")
    s26, _ = round_sets(RE26)
    c26 = collections.Counter(cls(v) for v in s26.values())
    print(f"2026 tome round_end matches {len(s26)}; numbering classes {dict(c26)}")

# ---------------------------------------------------------------- RAR index vs G: header uuids
idx_uuids = collections.defaultdict(set)
with open(TSV, encoding="utf-8", errors="replace", newline="") as f:
    f.readline()
    for line in f:
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) < 4:
            continue
        segs = parts[3].split("/")
        if len(segs) == 6 and segs[0] == "csds" and segs[1] == "2024" and segs[2] == "02":
            idx_uuids[segs[4]].add(segs[5])
h_uuid = {k.split("/")[4] for k in header_keys}
idx_set = set(idx_uuids)
only_idx = idx_set - h_uuid
only_h = h_uuid - idx_set
print(f"\nRAR index 2024-02 uuids: {len(idx_set)}; header uuids: {len(h_uuid)}; "
      f"index-only: {len(only_idx)}; header-only (not in index): {len(only_h)}")
# index-only uuids: do they exist as tail folders?
ex_rows = []
for u in sorted(only_idx)[:20]:
    dirs = [p for p in (E_TAIL / "2024" / "02").glob(f"*/{u}")]
    ex_rows.append((u, [str(p.parent.name) for p in dirs], sorted(idx_uuids[u])[:4]))
print("index-only uuids (first 20): ", ex_rows)
print(f"done part 2 ({time.time() - t0:.0f}s)")
