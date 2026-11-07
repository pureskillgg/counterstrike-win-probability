"""D3-6 cross-check (read-only): alternative explanations for the round-2 start and the 224 day-01 gap.

1. Recounts the claimed 4,350 G: round_end rows for the 224 claim-set day-01 matches (own code).
2. Round-1 presence per G: channel, for start-at-2 matches vs contiguous-from-1 matches.
3. Round-1 presence in E: bomb_state (raw) for start-at-2 matches and a seeded sample of contiguous matches.
   Sample seed 0 is an ASSUMPTION used only for the contiguous comparison group.
4. Start-at-2 status against the UTC match_date instead of the folder day (the date cutover).
5. Round-13 side-vs-bomb mismatch (same definition as the claimed e2 check) split by folder day, by
   match_date cutover and by start-at-2 status. Cross-check only; it is outside the D3-6 claim itself.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_6_crosscheck.py
"""
import collections
import glob
import os
import random
import sys
import time
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
t0 = time.time()
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME_G = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
E_TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds")
E_A = Path(r"E:\csds_extracted\csds")
CH4 = ["round_end", "round_state", "bomb_state", "bomb_action"]
CUTOVER_LAST_PRE = "2024-02-06"  # last UTC match_date of the pre block; section 4 shows the cutover from data
SEED = 0  # ASSUMPTION: seed for the contiguous-from-1 comparison sample


def files_of(folder):
    return sorted(glob.glob(str(folder / "dataframe_*")))


def read_cols(folder, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files_of(folder)], how="vertical_relaxed")


def nonzero(path):
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def base_for(key, ch):
    p = key.split("/")
    tb = E_TAIL / p[1] / p[2] / p[3] / p[4]
    ab = E_A / p[1] / p[2] / p[3] / p[4]
    if nonzero(str(tb / ch)):
        return tb
    if nonzero(str(ab / ch)):
        return ab
    return None


def round_sets(df):
    out = collections.defaultdict(set)
    for mk, r in df.select(["match_key", "round"]).iter_rows():
        if r is not None:
            out[mk].add(r)
    return out


# ---------------------------------------------------------------- header, G: round_end
H = read_cols(TOME_G / "header", ["key", "match_date"])
keys = H["key"].to_list()
day_of = {k: k.split("/")[3] for k in keys}
md_of = {k: (m[:10] if m else None) for k, m in H.select(["key", "match_date"]).iter_rows()}

RE = read_cols(TOME_G / f"basic_round_end{SUF}", ["match_key", "round", "player_count", "winner_team_code"])
G_sets = round_sets(RE)
rows_per_key = collections.Counter(RE["match_key"].to_list())
cls_of = {}
for k, rs in G_sets.items():
    s = sorted(rs)
    n = len(s)
    if s == list(range(1, n + 1)):
        cls_of[k] = "c1"
    elif s == list(range(s[0], s[0] + n)) and s[0] > 1:
        cls_of[k] = "c2"
    else:
        cls_of[k] = "other"
start2 = sorted(k for k, c in cls_of.items() if c == "c2")
contig1 = sorted(k for k, c in cls_of.items() if c == "c1")
print(f"G: round_end rows {RE.height}; matches {len(G_sets)}; classes {dict(collections.Counter(cls_of.values()))}")

# ---------------------------------------------------------------- 1. the claimed 224 set and its 4,350 rows
claim224 = []
for k in keys:
    if day_of[k] != "01":
        continue
    if not any(nonzero(str(E_TAIL / k.split("/")[1] / k.split("/")[2] / k.split("/")[3] / k.split("/")[4] / c))
               for c in CH4):
        claim224.append(k)
raw_rows = sum(rows_per_key[k] for k in claim224)
uniq_rounds = sum(len(G_sets.get(k, set())) for k in claim224)
print(f"\n1. claim-set day-01 keys with no non-zero compare channel in the tail copy: {len(claim224)}")
print(f"   G: round_end rows for them: raw rows {raw_rows}; unique (match, round) {uniq_rounds}")
ca_ok = sum(1 for k in claim224 if base_for(k, "round_end") is not None and str(base_for(k, "round_end")).startswith(str(E_A)))
print(f"   of them with a non-zero round_end in copy A (E:\\csds_extracted): {ca_ok} of {len(claim224)}")

# ---------------------------------------------------------------- 2. round-1 presence per G: channel
print("\n2. G: rounds 1 presence by channel (start-at-2 vs contiguous-from-1):")
g_ch = {}
for ch in ["round_start", "round_state", "round_mvp", "bomb_state", "bomb_action"]:
    df = read_cols(TOME_G / f"basic_{ch}{SUF}", ["match_key", "round"])
    sets = round_sets(df)
    g_ch[ch] = sets
    for name, grp in (("start2", start2), ("contig1", contig1)):
        has_rows = sum(1 for k in grp if k in sets)
        has1 = sum(1 for k in grp if 1 in sets.get(k, set()))
        has2 = sum(1 for k in grp if 2 in sets.get(k, set()))
        print(f"   {ch:<12} {name:<8} n={len(grp)}: any rows {has_rows}; round 1 {has1}; round 2 {has2}")
print(f"   ({time.time() - t0:.0f}s)")

# ---------------------------------------------------------------- 3. E: bomb_state round 1
rng = random.Random(SEED)
sample_c1 = sorted(rng.sample(contig1, 300))
print("\n3. E: bomb_state round 1 presence (raw, tail or copy A):")
for name, grp in (("start2", start2), ("contig1 sample(300, seed0)", sample_c1)):
    n_read = n_has1 = n_none = 0
    n_has_rows = 0
    for k in grp:
        b = base_for(k, "bomb_state")
        if b is None:
            n_none += 1
            continue
        df = pl.read_parquet(str(b / "bomb_state"), columns=["round"])
        rs = set(r for r in df.get_column("round").to_list() if r is not None)
        n_read += 1
        if rs:
            n_has_rows += 1
        if 1 in rs:
            n_has1 += 1
    print(f"   {name:<28} read {n_read}; no E: bomb_state file {n_none}; with any rows {n_has_rows}; with round 1 {n_has1}")
print(f"   ({time.time() - t0:.0f}s)")

# ---------------------------------------------------------------- 4. start-at-2 vs UTC match_date
pre = [k for k in keys if md_of[k] and md_of[k] <= CUTOVER_LAST_PRE]
post = [k for k in keys if md_of[k] and md_of[k] > CUTOVER_LAST_PRE]
nulls = [k for k in keys if not md_of[k]]
s2 = set(start2)
print(f"\n4. UTC match_date cutover (pre = <= {CUTOVER_LAST_PRE}): pre matches {len(pre)}, start-at-2 {sum(k in s2 for k in pre)}; "
      f"post matches {len(post)}, start-at-2 {sum(k in s2 for k in post)}; null match_date {len(nulls)}")
exc_post = [k for k in post if k in s2]
exc_pre_c1 = [k for k in pre if k not in s2]
print("   start-at-2 with match_date after cutover:", [(k, md_of[k], sorted(G_sets[k])[:3], len(G_sets[k])) for k in exc_post])
print("   pre-cutover matches that start at round 1:", [(k, md_of[k]) for k in exc_pre_c1][:10], len(exc_pre_c1))
for k in exc_post:
    b = base_for(k, "round_start")
    rs = set(pl.read_parquet(str(b / "round_start"), columns=["round"]).get_column("round").to_list()) if b else set()
    rend = set(pl.read_parquet(str(b / "round_end"), columns=["round"]).get_column("round").to_list()) if b else set()
    print(f"   exception {k}: E: round_start has round 1: {1 in rs}; E: round_end has round 1: {1 in rend}; "
          f"base {b}")

# ---------------------------------------------------------------- 5. round-13 mismatch cross-tab
BS = read_cols(TOME_G / f"basic_bomb_state{SUF}", ["match_key", "round", "event_type"])
outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
           .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
flag = (RE.filter((pl.col("player_count") == 20) & (pl.col("round") == 13))
        .join(outcome, on=["match_key", "round"], how="inner")
        .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("side"))
        .with_columns((pl.col("side") != pl.col("winner_team_code")).alias("mm")))
rows = flag.select(["match_key", "winner_team_code", "side", "mm"]).iter_rows()
cells = collections.defaultdict(lambda: [0, 0])
for mk, _, _, mm in rows:
    d = int(day_of.get(mk, "0"))
    grp = "day01_07" if d <= 7 else "day12_25"
    md = md_of.get(mk)
    cut = "pre" if md and md <= CUTOVER_LAST_PRE else ("post" if md else "null")
    c = "start2" if mk in s2 else "contig1"
    for label in (f"folder:{grp}", f"md:{cut}", f"start:{c}", f"folder:{grp} x md:{cut}", f"folder:{grp} x start:{c}"):
        cells[label][0] += 1
        cells[label][1] += int(mm)
print("\n5. Round-13 side-vs-bomb mismatch (claimed e2 definition: player_count 20, bomb outcome, side vs winner):")
for label in sorted(cells):
    n, m = cells[label]
    print(f"   {label:<40} rows {n:>4}; mismatches {m:>4}")
print(f"   total round-13 rows with outcome: {flag.height}; mismatches: {int(flag['mm'].sum())}")
print(f"\ndone crosscheck ({time.time() - t0:.0f}s)")
