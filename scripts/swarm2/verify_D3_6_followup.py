"""Follow-up for D3-6 verifier. Read-only. Appends to logs\\swarm2\\verify_D3_6.log via stdout redirect.

  J. Row-level provenance: G: round_end rows vs E:\\csds_extracted and E:\\csds_recovered_20261011
     round_end rows, for the 224 claim-set day-01 keys (and all 230 day-01 keys).
  K. Per day-01 key, which E: copies hold each compare channel (union coverage).
  L. Tail-archive uuids (2024/02) that are absent from the G: header (5 expected): raw header
     match_id/demo_id vs G: header, and E: tail folder content.
  M. Raw round_state round-1 presence for start-at-2 matches (tail copy, sample), to test the
     claim's "round-1 states absent".
  N. Start-at-2 vs start-at-1 matches in days 12-25: key day vs match_date day, tick_rate, map,
     build_num, match_type (context for the cause of the round-1 absence; ASSUMPTION: header
     fields are the demo's own metadata).
  O. Day-label check: key day vs match_date day distribution for days 12-25.
"""
import collections
import glob
import os
import sys
import time
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(30)
pl.Config.set_tbl_width_chars(200)

ROOT = Path(r"G:\pureskill\csds-cs2")
T24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF24 = ".2023-12-28,2024-02-25"
E_TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds")
E_EXT = Path(r"E:\csds_extracted\csds")
E_REC = Path(r"E:\csds_recovered_20261011\csds")
TSV = Path(r"E:\csds_rar5_tail_salvage_20261011.tsv")
COMP = ["round_end", "round_state", "bomb_state", "bomb_action"]
T0 = time.time()


def hdr(t):
    print(f"\n==== {t} (t={time.time() - T0:.0f}s)", flush=True)


def g_read(folder, cols=None):
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def g24(ch, cols=None):
    return g_read(T24 / ("header" if ch == "header" else f"basic_{ch}{SUF24}"), cols)


def kday(k):
    return int(k.split("/")[3])


def ep(base, key, ch):
    p = key.split("/")
    return base / p[1] / p[2] / p[3] / p[4] / ch


# ----------------------------------------------------------- headers
H = g24("header", ["key", "match_date", "match_id", "map_name", "tick_rate", "match_type", "build_num",
                   "demo_version_name", "demo_id", "sharecode"])
keys = H["key"].to_list()
Hd = {r["key"]: r for r in H.to_dicts()}
RE = g24("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
rounds_set = collections.defaultdict(set)
for m, r in RE.select(["match_key", "round"]).iter_rows():
    rounds_set[m].add(r)
start2 = {k for k in keys if k in rounds_set and min(rounds_set[k]) == 2}
day01 = [k for k in keys if kday(k) == 1]
tail_status = {}
for k in keys:
    p = ep(E_TAIL, k, "")  # uuid folder
    if not p.is_dir():
        tail_status[k] = "missing"
        continue
    names = set(os.listdir(p))
    tail_status[k] = "empty" if not names else ("all4" if all(c in names for c in COMP) else
                                                 ("none" if not any(c in names for c in COMP) else "partial"))
claim_set = [k for k in day01 if tail_status[k] in ("empty", "none")]
print("claim-set size", len(claim_set), "| day-01 keys", len(day01), "| start2 matches", len(start2))

# ----------------------------------------------------------- J. row-level provenance
hdr("J. Row-level provenance: G: round_end vs E: extracted / recovered (day-01 keys)")
cols = ["round", "tick", "winner_team_code", "win_reason_code", "player_count"]


def rows_of(path):
    df = pl.read_parquet(str(path), columns=cols)
    return df.sort(["round", "tick"]).select(cols)


def g_rows(k):
    d = RE.filter(pl.col("match_key") == k).select(cols).sort(["round", "tick"])
    return d


res = collections.Counter()
examples = []
for k in day01:
    g = RE.filter(pl.col("match_key") == k).select(cols).sort(["round", "tick"])
    for name, base in (("extracted", E_EXT), ("recovered", E_REC)):
        p = ep(base, k, "round_end")
        if not p.is_file():
            res[f"{name}:no_file"] += 1
            continue
        raw = rows_of(p)
        same = raw.equals(g)
        grp = "claim" if k in claim_set else "other"
        res[f"{name}:{grp}:{'row_equal' if same else 'DIFF'}"] += 1
        if not same and len(examples) < 4:
            examples.append((name, k, raw.height, g.height, raw.head(3).rows(), g.head(3).rows()))
for c, n in sorted(res.items()):
    print(f"  {c:<35} {n}")
print("  diff examples:", examples)
# compare extracted vs recovered directly
same_ext_rec = collections.Counter()
for k in day01:
    a, b = ep(E_EXT, k, "round_end"), ep(E_REC, k, "round_end")
    if a.is_file() and b.is_file():
        same_ext_rec["equal" if rows_of(a).equals(rows_of(b)) else "DIFF"] += 1
    else:
        same_ext_rec["one_missing"] += 1
print("  extracted vs recovered round_end (day-01):", dict(same_ext_rec))

# ----------------------------------------------------------- K. union coverage
hdr("K. Union coverage of compare channels, day-01 keys")
union = collections.Counter()
per_key = {}
for k in day01:
    got = {}
    for name, base in (("tail", E_TAIL), ("extracted", E_EXT), ("recovered", E_REC)):
        p = ep(base, k, "")  # uuid folder
        names = set(os.listdir(p)) if p.is_dir() else set()
        got[name] = sum(1 for c in COMP if c in names) if names else 0
    per_key[k] = got
    best = max(got.values())
    union["any_source_all4" if best == 4 else f"best_source_{best}_of_4"] += 1
    union[f"tail={tail_status[k]}|ext={got['extracted']}|rec={got['recovered']}"] += 1
for c, n in sorted(union.items(), key=lambda x: -x[1]):
    print(f"  {c:<60} {n}")
for k, got in per_key.items():
    if max(got.values()) < 4:
        chans = {}
        for name, base in (("tail", E_TAIL), ("extracted", E_EXT), ("recovered", E_REC)):
            p = ep(base, k, "")
            chans[name] = sorted(c for c in COMP if p.is_dir() and c in set(os.listdir(p)))
        print("  best<4 key:", k, chans, "| G round_end rows:", len(rounds_set.get(k, [])))
        tp = ep(E_TAIL, k, "round_end")
        if tp.is_file():
            t_rows = rows_of(tp)
            g_r = RE.filter(pl.col("match_key") == k).select(cols).sort(["round", "tick"])
            print("    tail round_end equals G: rows:", t_rows.equals(g_r), "| tail round set:",
                  sorted(set(t_rows["round"].to_list())))

# ----------------------------------------------------------- L. archive-only uuids
hdr("L. Tail-archive 2024/02 uuids absent from G: header")
tsv = pl.read_csv(str(TSV), separator="\t")
idx = collections.defaultdict(set)
for name in tsv["name"].to_list():
    p = name.split("/")
    if len(p) == 6 and p[0] == "csds" and p[1] == "2024" and p[2] == "02":
        idx[(p[3], p[4])].add(p[5])
hdr_pairs = {(k.split("/")[3], k.split("/")[4]): k for k in keys}
orphans = [(dd, u) for (dd, u) in idx if (dd, u) not in hdr_pairs]
print("orphan (day, uuid) in archive:", len(orphans))
for dd, u in sorted(orphans):
    chs = sorted(idx[(dd, u)])
    comp_present = [c for c in COMP if c in chs]
    print(f"  day {dd} uuid {u} archive channels={len(chs)} compare={comp_present}")
    tp = E_TAIL / "2024" / "02" / dd / u
    print(f"    E: tail folder exists={tp.is_dir()} entries={len(os.listdir(tp)) if tp.is_dir() else 0}")
    hp = tp / "header"
    if hp.is_file():
        hr = pl.read_parquet(str(hp), columns=["demo_id", "sharecode", "map_name", "match_date",
                                              "tick_rate", "build_num"])
        r = hr.to_dicts()[0]
        PLACEHOLDER = ("", "redacted", None)
        dup_g = [kk for kk, v in Hd.items() if (v["demo_id"] not in PLACEHOLDER and v["demo_id"] == r["demo_id"])
                 or (v["sharecode"] not in PLACEHOLDER and v["sharecode"] == r["sharecode"])]
        dup_g = dup_g[:5] + ([f"...{len(dup_g)} total"] if len(dup_g) > 5 else [])
        print(f"    raw header: demo_id={r['demo_id']} map={r['map_name']} date={r['match_date']} "
              f"tick={r['tick_rate']} build={r['build_num']} | G: duplicates by demo_id/sharecode: {dup_g}")
    else:
        print("    no raw header file")

# ----------------------------------------------------------- M. raw round_state round 1
hdr("M. Raw round_state round-1 presence, start-at-2 tail matches (sample of 300)")
sample = sorted(k for k in start2 if tail_status[k] == "all4")[:300]
c = collections.Counter()
for k in sample:
    p = ep(E_TAIL, k, "round_state")
    if not p.is_file():
        c["no_round_state_file"] += 1
        continue
    rs = pl.read_parquet(str(p), columns=["round"])
    c["has_round1" if (rs["round"] == 1).any() else "no_round1"] += 1
print("  sample", len(sample), dict(c))
# G: round_state round-1 rows: count of rows and tick range for round 1 in start-2 matches
RS = g24("round_state", ["match_key", "round", "tick"])
r1 = RS.filter((pl.col("round") == 1) & pl.col("match_key").is_in(list(start2)))
print("  G: round_state round-1 rows for start-2 matches:", r1.height, "| matches:", r1["match_key"].n_unique())
if r1.height:
    print("  G: round-1 state tick range:", r1["tick"].min(), r1["tick"].max())
rs_round_counts = (RS.filter(pl.col("match_key").is_in(list(start2)))
                   .group_by("match_key").agg(pl.col("round").min().alias("rmin")))
print("  G: round_state min round for start-2 matches:", dict(collections.Counter(rs_round_counts["rmin"].to_list())))

# ----------------------------------------------------------- N/O. day label and round-1 split in days 12-25
hdr("N/O. Day-label vs match_date, and start-2 vs start-1 split in days 12-25")
late = [k for k in keys if kday(k) >= 12]
lab = collections.Counter()
for k in keys:
    md = Hd[k]["match_date"]
    lab["key_day==match_date_day" if int(md[8:10]) == kday(k) else "key_day!=match_date_day"] += 1
print("  all keys:", dict(lab))
lab_late = collections.Counter()
for k in late:
    md = Hd[k]["match_date"]
    cls = "start2" if k in start2 else "start1"
    lab_late[(cls, "same_day" if int(md[8:10]) == kday(k) else "diff_day")] += 1
print("  days 12-25 by class x (key day vs match_date day):", dict(lab_late))
for field in ("tick_rate", "map_name", "match_type", "build_num", "demo_version_name"):
    tab = collections.defaultdict(collections.Counter)
    for k in late:
        cls = "start2" if k in start2 else "start1"
        tab[Hd[k][field]][cls] += 1
    print(f"  {field} by class (days 12-25):")
    for v, cc in sorted(tab.items(), key=lambda x: -(x[1]["start2"] + x[1]["start1"]))[:10]:
        tot = cc["start2"] + cc["start1"]
        print(f"    {str(v):<28} start2={cc['start2']:<5} start1={cc['start1']:<5} start2_share={cc['start2'] / tot:.2f}")
mdm = collections.Counter()
for k in keys:
    mdm[Hd[k]["match_date"][:7]] += 1
print("  match_date year-month across all keys:", dict(mdm))

# ----------------------------------------------------------- bakeoff round-1 use (grep only)
hdr("P. How the bake-off consumes round_end (grep, read only)")
bk = Path(r"G:\pureskill\csds-cs2\scripts\model\bakeoff.py")
if bk.is_file():
    lines = bk.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [(i + 1, ln.strip()) for i, ln in enumerate(lines) if "round_end" in ln or "freeze" in ln]
    for i, ln in hits[:20]:
        print(f"  {i}: {ln[:160]}")
    print(f"  total matching lines: {len(hits)}")

hdr("DONE")
