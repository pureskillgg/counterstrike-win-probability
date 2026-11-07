"""Independent verifier for finding D3-6 (coverage gaps in the 2024 G: tome vs E: sources).

Recomputes, without importing the claimed scripts:
  A. G: 2024 header keys, key day vs match_date day.
  B. G: 2024 round_end per match: row counts, duplicates, round-set class (contiguous from 1,
     starts at 2, other), day groups.
  C. E: tail copy census for every header key (all4 / partial / empty / no compare channels).
  D. Alternative E: copies of day-01 folders: E:\\csds_extracted and E:\\csds_recovered_20261011.
  E. Salvage index E:\\csds_rar5_tail_salvage_20261011.tsv (archive listing, read only).
  F. Raw round_end round sets (E:) compared with G: round_end round sets, per match:
     tail copy for all4 keys, and the day-01 alternative copies.
  G. 2026 control tome: round_end starting round per match.
  H. G: round_state round 1 presence among start-at-2 matches (claim says round-1 states absent).
  I. map_name distribution of the day-01 group (context for the map question).

Read-only. Writes nothing; stdout is redirected to logs\\swarm2\\verify_D3_6.log by the caller.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_6_main.py
ASSUMPTIONS are marked in the printed output where they apply.
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
T26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUF24 = ".2023-12-28,2024-02-25"
E_TAIL = Path(r"E:\csds_final\csds\csds_tail_recovered_20261011\csds")
E_EXT = Path(r"E:\csds_extracted\csds")
E_REC = Path(r"E:\csds_recovered_20261011\csds")
TSV = Path(r"E:\csds_rar5_tail_salvage_20261011.tsv")
COMP = ["round_end", "round_state", "bomb_state", "bomb_action"]

T0 = time.time()


def hdr(title):
    print(f"\n==== {title} (t={time.time() - T0:.0f}s)", flush=True)


def g_read(folder, cols=None):
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    if not files:
        raise SystemExit(f"no pages in {folder}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def g24(ch, cols=None):
    return g_read(T24 / ("header" if ch == "header" else f"basic_{ch}{SUF24}"), cols)


def key_day(key):
    return int(key.split("/")[3])


def key_group(key):
    d = key_day(key)
    if d == 1:
        return "day01"
    if d <= 7:
        return "day02_07"
    return "day12_25"


def e_status(base, key):
    parts = key.split("/")
    p = base / parts[1] / parts[2] / parts[3] / parts[4]
    if not p.is_dir():
        return "missing", []
    names = set(os.listdir(p))
    if not names:
        return "empty", []
    present = [c for c in COMP if c in names]
    if len(present) == len(COMP):
        return "all4", present
    if present:
        return f"partial{len(present)}", present
    return "no_comp", present


def raw_rounds(base, key):
    """Distinct round values in the raw round_end file for key under base, or None if absent."""
    parts = key.split("/")
    p = base / parts[1] / parts[2] / parts[3] / parts[4] / "round_end"
    if not p.is_file():
        return None
    df = pl.read_parquet(str(p), columns=["round"])
    return sorted(set(df["round"].to_list())), df.height


# ---------------------------------------------------------------- A. header
hdr("A. G: 2024 header")
H = g24("header", ["key", "match_date", "match_id", "map_name"])
keys = H["key"].to_list()
print("header rows", H.height, "| unique key", H["key"].n_unique(), "| unique match_id", H["match_id"].n_unique())
bad_fmt = [k for k in keys if not k.startswith("csds/2024/02/")]
print("keys not under csds/2024/02/:", len(bad_fmt))
kd = collections.Counter(key_day(k) for k in keys)
print("header matches by key day:", dict(sorted(kd.items())))
mismatch = 0
mdate_by_kday = collections.defaultdict(collections.Counter)
for k, md in zip(H["key"].to_list(), H["match_date"].to_list()):
    d_key = key_day(k)
    d_md = int(md[8:10])
    mdate_by_kday[d_key][md[:7]] += 1
    if d_key != d_md:
        mismatch += 1
print(f"key day != match_date UTC day: {mismatch} of {H.height}")
for d in (1, 2, 7):
    print(f"  key day {d:02d}: match_date year-month counts {dict(mdate_by_kday[d])}")
print("(ASSUMPTION: match_date is UTC ISO string; key day taken from csds/2024/02/DD path)")

# ---------------------------------------------------------------- B. G: round_end
hdr("B. G: 2024 round_end per match")
RE = g24("round_end", ["match_key", "round", "player_count"])
print("round_end rows", RE.height, "| matches with round_end", RE["match_key"].n_unique())
dup_rows = RE.height - RE.unique(["match_key", "round"]).height
print("duplicate (match_key, round) rows:", dup_rows)
rounds_raw = collections.defaultdict(list)
for m, r in RE.select(["match_key", "round"]).iter_rows():
    rounds_raw[m].append(r)
rounds_set = {m: sorted(set(v)) for m, v in rounds_raw.items()}
no_re = [k for k in keys if k not in rounds_set]
print("header keys with no round_end rows in G:", len(no_re))

cls_all = collections.Counter()
cls_grp = collections.defaultdict(collections.Counter)
for k in keys:
    rs = rounds_set.get(k)
    if rs is None:
        cls_all["no_round_end"] += 1
        cls_grp[key_group(k)]["no_round_end"] += 1
        continue
    n = len(rs)
    if rs == list(range(1, n + 1)):
        c = "contiguous_from_1"
    elif rs == list(range(rs[0], rs[0] + n)):
        c = f"contiguous_from_{rs[0]}"
    else:
        c = "interior_gap_or_other"
    cls_all[c] += 1
    cls_grp[key_group(k)][c] += 1
print("round-set classes (all header keys):", dict(cls_all))
for g in ("day01", "day02_07", "day12_25"):
    print(f"  {g}: {dict(cls_grp[g])}")
start_hist = collections.Counter(rs[0] for rs in rounds_set.values())
print("first round value counts:", sorted(start_hist.items())[:6])
start2_day0107 = sum(1 for k in keys if key_group(k) in ("day01", "day02_07") and rounds_set.get(k, [None])[0] == 2)
n_0107 = sum(1 for k in keys if key_group(k) in ("day01", "day02_07"))
print(f"days 01-07 keys starting at round 2: {start2_day0107} of {n_0107}")

# ---------------------------------------------------------------- C. E: tail census (own code)
hdr("C. E: tail copy census (all header keys)")
st_tail = {}
cnt = collections.Counter()
cnt_day01 = collections.Counter()
no_comp_names = []
for k in keys:
    s, present = e_status(E_TAIL, k)
    st_tail[k] = s
    cnt[s] += 1
    if key_day(k) == 1:
        cnt_day01[s] += 1
    if s == "no_comp":
        no_comp_names.append(k)
print("all keys:", dict(cnt))
print("day 01 keys:", dict(cnt_day01))
print("no_comp keys (folder has files, none of the 4 compare channels):", no_comp_names)
day01_keys = [k for k in keys if key_day(k) == 1]
claim_set = [k for k in day01_keys if st_tail[k] in ("empty", "no_comp")]
print(f"day-01 keys with no compare-channel files in tail copy (empty + no_comp): {len(claim_set)}")
g_rows_claim = sum(len(rounds_raw.get(k, [])) for k in claim_set)
g_uniq_claim = sum(len(rounds_set.get(k, [])) for k in claim_set)
print(f"  G: round_end rows for them: raw rows {g_rows_claim}, unique (match, round) {g_uniq_claim}")
print(f"  of them with any G: round_end rows: {sum(1 for k in claim_set if k in rounds_set)}")

# ---------------------------------------------------------------- D. alternative E: copies, day 01
hdr("D. Alternative E: copies of day-01 folders")
alt = {}
for name, base in (("extracted", E_EXT), ("recovered", E_REC)):
    d1 = base / "2024" / "02" / "01"
    print(f"{name}: {d1} exists={d1.is_dir()} folders={len(os.listdir(d1)) if d1.is_dir() else 0}")
    c = collections.Counter()
    for k in day01_keys:
        s, present = e_status(base, k)
        alt[(name, k)] = (s, present)
        c[s] += 1
    print(f"  status for the 230 day-01 header keys: {dict(c)}")
    c2 = collections.Counter(alt[(name, k)][0] for k in claim_set)
    print(f"  status for the {len(claim_set)} claim-set keys: {dict(c2)}")
extra_ext = set(os.listdir(E_EXT / "2024" / "02" / "01")) if (E_EXT / "2024" / "02" / "01").is_dir() else set()
extra_rec = set(os.listdir(E_REC / "2024" / "02" / "01")) if (E_REC / "2024" / "02" / "01").is_dir() else set()
hk = set(k.split("/")[4] for k in day01_keys)
print(f"uuid sets: header day-01 {len(hk)}; extracted folders {len(extra_ext)} (in header {len(extra_ext & hk)}); "
      f"recovered folders {len(extra_rec)} (in header {len(extra_rec & hk)})")
print("extracted and recovered folder sets equal:", extra_ext == extra_rec)
print("header day-01 uuids absent from extracted:", len(hk - extra_ext), "| absent from recovered:", len(hk - extra_rec))

# ---------------------------------------------------------------- E. salvage index
hdr("E. Salvage index (tail archive listing)")
tsv = pl.read_csv(str(TSV), separator="\t")
print("tsv rows", tsv.height, "columns", tsv.columns)
idx = collections.defaultdict(set)
idx_day = collections.Counter()
for name in tsv["name"].to_list():
    parts = name.split("/")
    if len(parts) == 6 and parts[0] == "csds" and parts[1] == "2024" and parts[2] == "02":
        idx[(parts[3], parts[4])].add(parts[5])
uuids_day = collections.defaultdict(set)
for (dd, u), chs in idx.items():
    uuids_day[dd].add(u)
    idx_day[dd] += 1
print("tail archive 2024/02 uuid-match folders by day:", {d: len(s) for d, s in sorted(uuids_day.items())})
print("tail archive 2024/02 entries (files) by day:", dict(sorted(idx_day.items())))
hdr_uuids = set(k.split("/")[4] for k in keys)
tsv_uuids = set(u for (_, u) in idx.keys())
print(f"tsv 2024/02 uuids not in G header: {len(tsv_uuids - hdr_uuids)}; header uuids absent from tsv 2024/02: {len(hdr_uuids - tsv_uuids)}")


def tsv_status(key):
    parts = key.split("/")
    chs = idx.get((parts[3], parts[4]), set())
    present = [c for c in COMP if c in chs]
    if not chs:
        return "absent", []
    if len(present) == len(COMP):
        return "all4", present
    if present:
        return f"partial{len(present)}", present
    return "no_comp", present


cross = collections.Counter()
for k in keys:
    cross[(st_tail[k], tsv_status(k)[0])] += 1
print("E: tail status x tsv status (all header keys):")
for (a, b), n in sorted(cross.items(), key=lambda x: -x[1]):
    print(f"  tail={a:<10} tsv={b:<10} n={n}")
c_claim_tsv = collections.Counter(tsv_status(k)[0] for k in claim_set)
print(f"claim-set ({len(claim_set)}) tsv status:", dict(c_claim_tsv))
c_empty_tsv = collections.Counter(tsv_status(k)[0] for k in day01_keys if st_tail[k] == "empty")
print("day-01 empty-folder keys tsv status:", dict(c_empty_tsv))

# ---------------------------------------------------------------- F. raw round sets vs G:
hdr("F. Raw round_end round sets vs G: (tail copy, all4 keys)")
t = time.time()
cmp_tail = collections.Counter()
examples = collections.defaultdict(list)
for k in keys:
    if st_tail[k] != "all4":
        continue
    rr = raw_rounds(E_TAIL, k)
    g = rounds_set.get(k, [])
    if rr is None:
        cmp_tail["raw_missing"] += 1
        continue
    raw_set, raw_rows = rr
    if raw_set == g:
        c = "equal"
    elif 1 in raw_set and 1 not in g:
        c = "raw_has_round1_G_lacks"
    elif 1 not in raw_set and 1 not in g:
        c = "neither_has_round1"
    else:
        c = "other_diff"
    cmp_tail[c] += 1
    cmp_tail[f"{key_group(k)}:{c}"] += 1
    if len(examples[c]) < 3:
        examples[c].append((k, raw_set[:5], g[:5]))
print("tail all4 compare (raw round set vs G round set):")
for c, n in sorted(cmp_tail.items()):
    print(f"  {c:<40} {n}")
for c, ex in examples.items():
    print(f"  examples {c}: {ex}")
print(f"(read time {time.time() - t:.0f}s)")

hdr("F2. Raw round_end round sets vs G: (day-01 alternative copies)")
cmp_alt = collections.Counter()
ex_alt = []
for k in claim_set + [k for k in day01_keys if k not in claim_set]:
    for name, base in (("extracted", E_EXT), ("recovered", E_REC)):
        s, present = alt[(name, k)]
        if "round_end" not in present:
            continue
        rr = raw_rounds(base, k)
        raw_set, raw_rows = rr
        g = rounds_set.get(k, [])
        if raw_set == g:
            c = "equal"
        elif 1 in raw_set and 1 not in g:
            c = "raw_has_round1_G_lacks"
        elif 1 not in raw_set and 1 not in g:
            c = "neither_has_round1"
        else:
            c = "other_diff"
        grp = "claim_set" if k in claim_set else "other_day01"
        cmp_alt[f"{name}:{grp}:{c}"] += 1
        if len(ex_alt) < 4 and c != "equal":
            ex_alt.append((name, k, raw_set[:5], g[:5]))
for c, n in sorted(cmp_alt.items()):
    print(f"  {c:<55} {n}")
print("  examples non-equal:", ex_alt)

# ---------------------------------------------------------------- G. 2026 control
hdr("G. 2026 control tome: round_end start round per match")
RE26 = g_read(T26 / "basic_round_end.2026-09-18,2026-10-08", ["match_key", "round"])
r26 = RE26.group_by("match_key").agg(pl.col("round").min().alias("rmin"), pl.len().alias("n"))
print("2026 matches with round_end:", r26.height)
print("2026 start round counts:", dict(collections.Counter(r26["rmin"].to_list())))
rs26 = RE26.unique(["match_key", "round"]).group_by("match_key").agg(pl.col("round").sort()).to_dicts()
c26 = collections.Counter()
for row in rs26:
    rs = row["round"]
    n = len(rs)
    c26["contiguous_from_1" if rs == list(range(1, n + 1)) else "other"] += 1
print("2026 round-set classes:", dict(c26))

# ---------------------------------------------------------------- H. round_state round 1 in G:
hdr("H. G: round_state round-1 presence (start-at-2 matches vs others)")
RS = g24("round_state", ["match_key", "round"]).unique()
rs_round1 = set(RS.filter(pl.col("round") == 1)["match_key"].to_list())
rs_any = set(RS["match_key"].to_list())
start2 = [k for k in keys if rounds_set.get(k, [None])[0] == 2]
other = [k for k in keys if k in rounds_set and rounds_set[k][0] != 2]
print(f"round_state has any rows: start2 {sum(1 for k in start2 if k in rs_any)} of {len(start2)}; "
      f"other {sum(1 for k in other if k in rs_any)} of {len(other)}")
print(f"round_state has round 1: start2 {sum(1 for k in start2 if k in rs_round1)} of {len(start2)}; "
      f"other {sum(1 for k in other if k in rs_round1)} of {len(other)}")

# ---------------------------------------------------------------- I. map_name context
hdr("I. map_name context (day-01 claim set vs rest)")
mp = dict(zip(H["key"].to_list(), H["map_name"].to_list()))
mc_claim = collections.Counter(mp.get(k) for k in claim_set)
mc_rest = collections.Counter(mp.get(k) for k in keys if k not in set(claim_set))
print("claim set maps:", mc_claim.most_common(8))
print("rest maps:", mc_rest.most_common(8))

hdr("DONE")
