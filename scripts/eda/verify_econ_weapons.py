"""Verify economy and weapon channel claims against the 2026-10-01_to_08 csds tome.

Read-only. Streams one page and one column at a time and writes nothing.
Covers the item_* channels that economy.json omits, rank_update, the six weapon
channels, and bullet_damage (for the batch coverage check).

Run:  cd G:\\pureskill\\csds-cs2 ; pixi run python -I scripts\\eda\\verify_econ_weapons.py small
      cd G:\\pureskill\\csds-cs2 ; pixi run python -I scripts\\eda\\verify_econ_weapons.py weapons
"""
import collections
import glob
import os
import sys
import time

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"
ITEM_CH = ["item_equip", "item_pickup", "item_dropped", "item_refund"]
WEAPON_CH = ["weapon_fire", "weapon_action", "grenade_state", "grenade_bounce",
             "molotov_state", "molotov_fire"]
SENT = [-1, 65535, 32767, 9999, 999, 4294967295, 2147483647]
NO_VC = ("tick", "second", "_pos", "_vel", "_ang", "match_key")

hdr = pq.read_table(os.path.join(ROOT, "header", "dataframe_00000"))
print("header columns:", hdr.column_names, flush=True)
KEYS = hdr.column("key").to_pylist()
if "match_date" in hdr.column_names:
    DAYS = [str(x)[:10] for x in hdr.column("match_date").to_pylist()]
    print("day source: match_date", flush=True)
else:
    DAYS = ["-".join(k.split("/")[1:4]) for k in KEYS]
    print("day source: key path", flush=True)


def batch(day):
    if day == "2026-10-08":
        return "10-08"
    if "2026-10-01" <= day <= "2026-10-07":
        return "10-01..07"
    return "other"


KEY_BATCH = {k: batch(d) for k, d in zip(KEYS, DAYS)}
BATCH_N = collections.Counter(KEY_BATCH.values())
UNIV = len(KEYS)
print("universe", UNIV, "batches", dict(BATCH_N), flush=True)
print("other-batch days:", dict(collections.Counter(d for d in DAYS if batch(d) == "other")), flush=True)


def folder(ch):
    c = glob.glob(os.path.join(ROOT, "basic_%s.*" % ch))
    assert len(c) == 1, (ch, c)
    return c[0]


def pages(ch):
    return sorted(glob.glob(os.path.join(folder(ch), "dataframe_*")))


def is_num(t):
    return pa.types.is_integer(t) or pa.types.is_floating(t)


def profile(ch, full=False):
    t0 = time.time()
    pl = pages(ch)
    rows = 0
    page_rows = []
    schema0 = None
    drift = []
    matches = set()
    nulls = collections.Counter()
    nans = collections.Counter()
    sent = collections.defaultdict(collections.Counter)
    vc = collections.defaultdict(collections.Counter)
    mm = {}
    for p in pl:
        pf = pq.ParquetFile(p)
        sch = pf.schema_arrow
        names = [(f.name, str(f.type)) for f in sch]
        if schema0 is None:
            schema0 = names
        elif names != schema0:
            drift.append((os.path.basename(p), [x for x in names if x not in schema0],
                          [x for x in schema0 if x not in names]))
        n = pf.metadata.num_rows
        page_rows.append(n)
        rows += n
        matches.update(pc.unique(pf.read(columns=["match_key"]).column(0)).to_pylist())
        for f in sch:
            name, t = f.name, f.type
            if name == "match_key":
                continue
            arr = pf.read(columns=[name]).column(0)
            nulls[name] += arr.null_count
            if pa.types.is_floating(t):
                nans[name] += pc.sum(pc.cast(pc.is_nan(arr), pa.int64())).as_py() or 0
            if is_num(t):
                for s in SENT:
                    c = pc.sum(pc.cast(pc.equal(arr, s), pa.int64())).as_py() or 0
                    if c:
                        sent[name][s] += c
                r = pc.min_max(arr).as_py() or {}
                lo, hi = mm.get(name, (None, None))
                if r.get("min") is not None:
                    lo = r["min"] if lo is None else min(lo, r["min"])
                if r.get("max") is not None:
                    hi = r["max"] if hi is None else max(hi, r["max"])
                mm[name] = (lo, hi)
            if (not pa.types.is_floating(t)) and not any(s in name for s in NO_VC):
                for item in pc.value_counts(arr).to_pylist():
                    if item["values"] is not None:
                        vc[name][item["values"]] += item["counts"]
    have = collections.Counter(KEY_BATCH.get(m, "unknown") for m in matches)
    print("== %s | pages %d %s | rows %d | %.0fs" % (ch, len(pl), page_rows, rows, time.time() - t0), flush=True)
    print("   matches with rows %d of %d; not in header %d; by batch: %s" % (
        len(matches), UNIV, len(matches - set(KEYS)),
        ", ".join("%s %d/%d" % (b, have[b], BATCH_N[b]) for b in ["10-08", "10-01..07", "other"])), flush=True)
    print("   schema drift:", drift if drift else "none", flush=True)
    for name, typ in schema0:
        if name == "match_key":
            continue
        interesting = nulls[name] > 0 or nans[name] > 0 or bool(sent[name]) or (name in vc and len(vc[name]) <= 12)
        if not (full or interesting):
            continue
        distinct = len(vc[name]) if name in vc else None
        top = vc[name].most_common(4) if name in vc else None
        sent_s = {str(k): v for k, v in sorted(sent[name].items())} if sent[name] else None
        print("   %-26s %-8s null=%d nan=%d distinct=%s min/max=%s sent=%s top=%s" % (
            name, typ, nulls[name], nans[name], distinct, mm.get(name), sent_s, top), flush=True)


def read_all(ch, cols):
    return pa.concat_tables([pq.read_table(p, columns=cols) for p in pages(ch)])


def rank_checks():
    df = read_all("rank_update", ["match_key", "round", "tick", "player_id",
                                  "rank_old", "rank_new", "rank_change", "win_count"]).to_pandas()
    print("== rank_update checks: rows", len(df), "matches", df.match_key.nunique(), flush=True)
    print("   rank_change != rank_new - rank_old:", int((df.rank_change != (df.rank_new - df.rank_old)).sum()), flush=True)
    per_mp = df.groupby(["match_key", "player_id"]).size()
    print("   rows per (match, player) -> matches:", per_mp.value_counts().sort_index().to_dict(), flush=True)
    per_m = df.groupby("match_key")["round"].nunique()
    print("   distinct rounds per match -> matches:", per_m.value_counts().sort_index().to_dict(), flush=True)
    re_ = read_all("round_end", ["match_key", "round", "tick"]).to_pandas()
    last = re_.groupby("match_key").agg(last_round=("round", "max"), last_tick=("tick", "max"))
    d = df.join(last, on="match_key")
    in_last = (d["round"] == d["last_round"])
    after = (d["tick"] > d["last_tick"])
    print("   rank rows in the match's last round: %d (%.3f)" % (in_last.sum(), in_last.mean()), flush=True)
    print("   rank rows with tick after last round_end tick: %d" % after.sum(), flush=True)
    print("   matches with rank rows but no round_end:", len(set(df.match_key) - set(re_.match_key)), flush=True)
    print("   rank row round range:", int(df["round"].min()), int(df["round"].max()), flush=True)


def weapon_code_check():
    cnt = collections.Counter()
    for p in pages("weapon_fire"):
        t = pq.read_table(p, columns=["player_weapon_code", "weapon_name"]).to_pandas()
        g = t.groupby(["player_weapon_code", "weapon_name"], dropna=False).size()
        for (c, n), k in g.items():
            cnt[(c, n)] += int(k)
    by_code = collections.defaultdict(dict)
    for (c, n), k in cnt.items():
        by_code[c][n] = k
    total = sum(cnt.values())
    mism = 0
    rows_by = []
    for c, d in by_code.items():
        tot = sum(d.values())
        modal_n, modal_k = max(d.items(), key=lambda x: x[1])
        mism += tot - modal_k
        rows_by.append((tot - modal_k, str(c), str(modal_n), modal_k, tot, len(d)))
    print("== weapon_fire code vs name: rows", total, "not modal name for code", mism,
          "share %.4f" % (mism / total), "codes", len(by_code), flush=True)
    for r in sorted(rows_by, reverse=True)[:8]:
        print("   code %s: off-modal %d | modal %s (%d of %d) | names for code %d" % (
            r[1], r[0], r[2], r[3], r[4], r[5]), flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "small"
    if which == "small":
        for ch in ITEM_CH:
            profile(ch, full=True)
        profile("rank_update", full=True)
        profile("bullet_damage")
        rank_checks()
    elif which == "weapons":
        for ch in WEAPON_CH:
            profile(ch)
        weapon_code_check()
    else:
        raise SystemExit("unknown group: " + which)
