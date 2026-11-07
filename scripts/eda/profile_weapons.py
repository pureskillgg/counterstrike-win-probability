"""Profile the weapons channel group of the 2026 CSDS tome (pyarrow only).

Streams one column at a time across pages to stay inside the RAM budget.
Read-only on the tome. Writes notes/eda/weapons.json.
"""
import glob
import json
import os
import time
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TOME = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
OUT_JSON = r"G:\pureskill\csds-cs2\notes\eda\weapons.json"
CHANNELS = ["weapon_fire", "weapon_action", "grenade_state", "grenade_bounce",
            "molotov_state", "molotov_fire"]
GROUP_KEYS = {
    "weapon_fire": ["match_key", "round", "tick", "player_id", "weapon_name"],
    "weapon_action": ["match_key", "round", "tick", "player_id", "event_type"],
    "grenade_state": ["match_key", "round", "tick", "entity_id", "event_type"],
    "grenade_bounce": ["match_key", "round", "tick", "entity_id", "bounce_nr"],
    "molotov_state": ["match_key", "round", "tick", "entity_id", "event_type"],
    "molotov_fire": ["match_key", "round", "tick", "entity_id", "fire_index"],
}
SENTINELS = [4294967295, 65535, 32767, 2147483647, -1, 999, 9999]
SPIKE_SHARE = 0.20
NULL_SHARE = 0.50
TICK_RATE = 64
SECOND_TOL = 0.05
HASH_PRIME = np.uint64(1000003)
HASH_NULL = np.uint64(0x9E3779B97F4A7C15)
IMPOSSIBLE_BY_NAME = [
    ("health", 0, 100, "health outside 0..100"),
    ("armor", 0, 100, "armor outside 0..100"),
    ("money", 0, 16000, "money outside 0..16000"),
    ("ammo", 0, None, "ammo below 0"),
    ("ping", 0, None, "ping below 0"),
]


def log(msg):
    print(msg, flush=True)


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        if f != f:
            return "NaN"
        if f in (float("inf"), float("-inf")):
            return str(f)
        return f
    return o


def is_num(t):
    return pa.types.is_integer(t) or pa.types.is_floating(t)


def tgroup(t):
    if is_num(t):
        return "num"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "str"
    if pa.types.is_boolean(t):
        return "bool"
    return str(t)


def channel_pages(ch):
    folders = [f for f in glob.glob(os.path.join(TOME, f"basic_{ch}.*")) if os.path.isdir(f)]
    if len(folders) != 1:
        raise RuntimeError(f"{ch}: folders={folders}")
    pages = sorted(p for p in glob.glob(os.path.join(folders[0], "dataframe_*")) if os.path.isfile(p))
    return folders[0], pages


def header_universe():
    p = os.path.join(TOME, "header", "dataframe_00000")
    names = pq.ParquetFile(p).schema_arrow.names
    col = "key" if "key" in names else None
    if col is None:
        raise RuntimeError(f"header has no key column: {names}")
    vals = pq.read_table(p, columns=[col]).column(0).drop_null().to_pylist()
    return set(vals), col, names


def range_rule(nm):
    for key, lo, hi, label in IMPOSSIBLE_BY_NAME:
        if key in nm:
            return lo, hi, label
    if nm in ("tick", "tick_throw") or nm.endswith("_tick"):
        return 0, None, "tick below 0"
    if nm == "round":
        return 0, 60, "round outside 0..60"
    if nm == "second":
        return 0, None, "second below 0"
    if nm.endswith("phi_ang"):
        return -180, 180, "phi outside -180..180"
    if nm.endswith("theta_ang"):
        return -90, 90, "theta outside -90..90"
    if "_vel" in nm:
        return -2000, 2000, "velocity beyond +-2000"
    if nm.endswith("_pos"):
        return -20000, 20000, "position beyond +-20000"
    if nm.startswith("fraction"):
        return 0, 1, "fraction outside 0..1"
    if nm == "burn_duration":
        return 0, 60, "burn_duration outside 0..60"
    if nm == "bounce_nr":
        return 1, None, "bounce_nr below 1"
    if nm == "fire_index":
        return 0, None, "fire_index below 0"
    if nm.endswith(("_id", "_id_fixed", "_code", "_pawn")):
        return 0, None, "id/code below 0"
    return None


def hash_array(arr):
    t = arr.type
    nulls = arr.is_null().to_numpy(zero_copy_only=False).astype(np.uint64)
    if pa.types.is_integer(t):
        vals = pc.fill_null(arr, 0).to_numpy(zero_copy_only=False).astype(np.int64).view(np.uint64)
    elif pa.types.is_floating(t):
        vals = pc.fill_null(arr, 0.0).to_numpy(zero_copy_only=False).astype(np.float64).view(np.uint64)
    elif pa.types.is_boolean(t):
        vals = pc.fill_null(arr, False).to_numpy(zero_copy_only=False).astype(np.uint64)
    else:
        return pd.util.hash_pandas_object(arr.to_pandas(), index=False).to_numpy(dtype=np.uint64)
    return vals ^ (nulls * HASH_NULL)


def examples_at(p, idx, extra=None, k=2):
    idx = np.asarray(idx, dtype=np.int64)[:k]
    if idx.size == 0:
        return []
    ia = pa.array(idx)
    t = pq.read_table(p, columns=["match_key", "round", "tick"])
    cols = {c: t.column(c).take(ia).to_pylist() for c in ("match_key", "round", "tick")}
    if extra:
        for name, arr in extra.items():
            cols[name] = arr.take(ia).to_pylist()
    return [{c: v[j] for c, v in cols.items()} for j in range(len(idx))]


def diff_mask(x, y):
    if x.type != y.type:
        if is_num(x.type) and is_num(y.type):
            x = x.cast(pa.float64())
            y = y.cast(pa.float64())
        else:
            x = x.cast(pa.string())
            y = y.cast(pa.string())
    same = pc.fill_null(pc.equal(x, y), False)
    same = pc.or_(same, pc.and_(x.is_null(), y.is_null()))
    if pa.types.is_floating(x.type):
        both_nan = pc.and_(pc.fill_null(pc.is_nan(x), False), pc.fill_null(pc.is_nan(y), False))
        same = pc.or_(same, both_nan)
    return ~np.asarray(same.to_numpy(zero_copy_only=False), dtype=bool)


def pair_diff(pages, a, b):
    total = 0
    ex = []
    for p in pages:
        t = pq.read_table(p, columns=[a, b])
        x = t.column(0).combine_chunks()
        y = t.column(1).combine_chunks()
        m = diff_mask(x, y)
        k = int(m.sum())
        if k:
            total += k
            if len(ex) < 2:
                ex += examples_at(p, np.flatnonzero(m), {a: x, b: y}, k=2 - len(ex))
    return total, ex


def value_examples(pages, nm, ft, value, k=2):
    out = []
    for p in pages:
        arr = pq.read_table(p, columns=[nm]).column(0).combine_chunks()
        m = pc.fill_null(pc.equal(arr, pa.scalar(value, type=ft)), False)
        m = np.asarray(m.to_numpy(zero_copy_only=False), dtype=bool)
        if m.any():
            out += examples_at(p, np.flatnonzero(m), {nm: arr}, k=k - len(out))
            if len(out) >= k:
                break
    return out


def dup_analysis(pages, H):
    allh = np.concatenate(H)
    u, c = np.unique(allh, return_counts=True)
    dm = c > 1
    if not dm.any():
        return {"groups": 0, "extra_rows": 0, "rows_in_dup_groups": 0, "matches_affected": 0,
                "top_matches": [], "examples": []}
    dv = u[dm]
    per_match = Counter()
    ex = []
    for pi, p in enumerate(pages):
        m = np.isin(H[pi], dv)
        if not m.any():
            continue
        idx = np.flatnonzero(m)
        sub = pq.read_table(p, columns=["match_key"]).column(0).take(pa.array(idx))
        vc = pc.value_counts(sub)
        for kk, vv in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
            per_match[kk] += vv
        if len(ex) < 3:
            ex += examples_at(p, idx, k=3 - len(ex))
    return {"groups": int(dm.sum()), "extra_rows": int((c[dm] - 1).sum()),
            "rows_in_dup_groups": int(c[dm].sum()), "matches_affected": len(per_match),
            "top_matches": per_match.most_common(3), "examples": ex}


def schema_drift(pages, pfs, schema0):
    names = schema0.names
    drift = []
    for p, pf in zip(pages, pfs):
        s = pf.schema_arrow
        if s.equals(schema0):
            continue
        d = []
        if s.names != names:
            d.append(f"column list differs: added {sorted(set(s.names) - set(names))}, "
                     f"missing {sorted(set(names) - set(s.names))}, same_order={s.names == names}")
        for nm in names:
            if nm in s.names and s.field(nm).type != schema0.field(nm).type:
                d.append(f"{nm}: {schema0.field(nm).type} -> {s.field(nm).type}")
        drift.append({"page": os.path.basename(p), "diffs": d})
    return drift


def weapon_code_pairs(pages):
    pairs = Counter()
    for p in pages:
        t = pq.read_table(p, columns=["weapon_name", "player_weapon_code"])
        k = pc.binary_join_element_wise(t.column(0).combine_chunks(),
                                        pc.cast(t.column(1).combine_chunks(), pa.string()), "|")
        vc = pc.value_counts(k)
        for kk, vv in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
            if kk is not None:
                pairs[kk] += vv
    by_code, by_name = {}, {}
    for kk, vv in pairs.items():
        name, code = kk.split("|", 1)
        by_code.setdefault(code, {})[name] = vv
        by_name.setdefault(name, {})[code] = vv
    return {
        "codes_with_more_than_one_name": {c: v for c, v in by_code.items() if len(v) > 1},
        "names_with_more_than_one_code": {n: v for n, v in by_name.items() if len(v) > 1},
    }


def profile_channel(ch, universe):
    t0 = time.time()
    folder, pages = channel_pages(ch)
    pfs = [pq.ParquetFile(p) for p in pages]
    page_rows = [int(pf.metadata.num_rows) for pf in pfs]
    n = sum(page_rows)
    schema0 = pfs[0].schema_arrow
    names = list(schema0.names)
    drift = schema_drift(pages, pfs, schema0)
    key_cols = [k for k in GROUP_KEYS.get(ch, []) if k in names]
    log(f"== {ch}: pages={len(pages)} rows={n} cols={len(names)} drift_pages={len(drift)}")

    H_full = [np.zeros(r, dtype=np.uint64) for r in page_rows]
    H_key = [np.zeros(r, dtype=np.uint64) for r in page_rows]
    has_second_tick = "second" in names and "tick" in names
    sec_pages = None
    col_out, sig, flags = {}, {}, []
    match_agg = None
    mk_page_unique = []

    for nm in names:
        ft = schema0.field(nm).type
        grp = tgroup(ft)
        nulls = 0
        vc_vals, vc_cnts = [], []
        mins, maxs, parts = [], [], []
        nan_cnt = inf_cnt = 0
        sent_counts = {s: 0 for s in SENTINELS} if grp == "num" else {}
        sent_ex = {s: [] for s in SENTINELS} if grp == "num" else {}
        rng = range_rule(nm) if grp == "num" else None
        rng_out, rng_ex = 0, []
        sec_this = []

        for pi, p in enumerate(pages):
            arr = pq.read_table(p, columns=[nm]).column(0).combine_chunks()
            if arr.type != ft:
                arr = arr.cast(ft)
            nulls += arr.null_count
            valid = pc.drop_null(arr)
            if len(valid):
                vcp = pc.value_counts(valid)
                vc_vals.append(vcp.field("values"))
                vc_cnts.append(vcp.field("counts"))
            if nm == "match_key" and len(valid):
                mk_page_unique.append(pc.unique(valid))
            if grp == "num":
                if pa.types.is_floating(ft):
                    isn = pc.fill_null(pc.is_nan(valid), False)
                    isi = pc.fill_null(pc.is_inf(valid), False)
                    nan_cnt += int(pc.sum(pc.cast(isn, pa.int64())).as_py() or 0)
                    inf_cnt += int(pc.sum(pc.cast(isi, pa.int64())).as_py() or 0)
                    fin = pc.filter(valid, pc.invert(pc.or_(isn, isi)))
                else:
                    fin = valid
                if len(fin):
                    mins.append(pc.min(fin).as_py())
                    maxs.append(pc.max(fin).as_py())
                    parts.append(fin)
                for s in SENTINELS:
                    m = np.asarray(pc.fill_null(pc.equal(arr, pa.scalar(s, type=ft)), False)
                                   .to_numpy(zero_copy_only=False), dtype=bool)
                    k = int(m.sum())
                    if k:
                        sent_counts[s] += k
                        if len(sent_ex[s]) < 2:
                            sent_ex[s] += examples_at(p, np.flatnonzero(m), {"value": arr},
                                                      k=2 - len(sent_ex[s]))
                if rng is not None:
                    lo, hi, _ = rng
                    bad = None
                    if lo is not None:
                        bad = pc.less(arr, pa.scalar(lo, type=ft))
                    if hi is not None:
                        b2 = pc.greater(arr, pa.scalar(hi, type=ft))
                        bad = b2 if bad is None else pc.or_(bad, b2)
                    m = np.asarray(pc.fill_null(bad, False).to_numpy(zero_copy_only=False), dtype=bool)
                    k = int(m.sum())
                    if k:
                        rng_out += k
                        if len(rng_ex) < 2:
                            rng_ex += examples_at(p, np.flatnonzero(m), {"value": arr},
                                                  k=2 - len(rng_ex))
                if nm == "second" and has_second_tick:
                    tk = pq.read_table(p, columns=["tick"]).column(0).combine_chunks()
                    sec_np = arr.cast(pa.float64()).to_numpy(zero_copy_only=False)
                    tk_np = tk.cast(pa.float64()).to_numpy(zero_copy_only=False)
                    sec_this.append(sec_np - tk_np / TICK_RATE)

            h = hash_array(arr)
            H_full[pi] = H_full[pi] * HASH_PRIME + h
            if nm in key_cols:
                H_key[pi] = H_key[pi] * HASH_PRIME + h

        if nm == "second" and sec_this:
            sec_pages = sec_this

        # aggregate
        st = {"dtype": str(ft), "group": grp, "rows": n, "nulls": nulls,
              "null_rate": round(nulls / n, 6) if n else None}
        if vc_vals:
            agg = (pa.table({"v": pa.chunked_array(vc_vals), "c": pa.chunked_array(vc_cnts)})
                   .group_by("v").aggregate([("c", "sum")])
                   .sort_by([("c_sum", "descending")]))
        else:
            agg = pa.table({"v": pa.array([], type=ft), "c_sum": pa.array([], type=pa.int64())})
        nd = agg.num_rows
        topn = 5 if nd <= 50 else 3
        top = agg.slice(0, topn).to_pylist()
        top_list = [{"value": r["v"], "count": int(r["c_sum"]), "share": round(r["c_sum"] / n, 6)}
                    for r in top]
        top_share = (top[0]["c_sum"] / n) if top and n else 0.0
        st.update({"n_distinct": nd, "top": top_list, "top_share": round(top_share, 6)})
        if nm == "match_key":
            match_agg = agg

        mn = mx = med = None
        if grp == "num":
            mn = min(mins) if mins else None
            mx = max(maxs) if maxs else None
            if parts:
                med = pc.quantile(pa.concat_arrays(parts), q=0.5).to_pylist()[0]
            st.update({"min": mn, "max": mx, "median": med,
                       "nan_count": nan_cnt, "inf_count": inf_cnt})
        sig[nm] = repr((grp, nulls, nd, mn, mx, med, top[0]["c_sum"] if top else 0))

        cflags = []
        if nulls == n and n:
            cflags.append({"flag": "all_null", "count": n})
        elif nd == 1 and nulls == 0:
            cflags.append({"flag": "constant", "value": top_list[0]["value"], "count": n})
        elif nd == 1:
            cflags.append({"flag": "constant_with_nulls", "value": top_list[0]["value"], "count": nulls})
        if n and nulls / n > NULL_SHARE:
            cflags.append({"flag": "null_over_50pct", "count": nulls, "share": round(nulls / n, 4)})
        if grp == "num":
            for s in SENTINELS:
                if sent_counts[s]:
                    cflags.append({"flag": "sentinel_value", "value": s, "count": sent_counts[s],
                                   "share": round(sent_counts[s] / n, 6), "examples": sent_ex[s]})
            if nan_cnt or inf_cnt:
                cflags.append({"flag": "nan_or_inf", "nan": nan_cnt, "inf": inf_cnt})
        if nd > 1 and top_share > SPIKE_SHARE and top_list:
            spike_ex = value_examples(pages, nm, ft, top_list[0]["value"])
            cflags.append({"flag": "spike_over_20pct", "value": top_list[0]["value"],
                           "count": top_list[0]["count"], "share": round(top_share, 4),
                           "examples": spike_ex})
        if rng is not None and rng_out:
            cflags.append({"flag": rng[2], "count": rng_out, "share": round(rng_out / n, 6),
                           "examples": rng_ex})
        if rng is not None and rng[2].startswith(("health", "armor", "money", "ammo", "ping")):
            st["impossible_by_name_rule"] = rng[2]
            st["impossible_count"] = rng_out
        if rng is not None:
            st["range_rule"] = {"lo": rng[0], "hi": rng[1], "out_count": rng_out, "examples": rng_ex}
        st["sentinels"] = {str(s): {"count": c, "examples": sent_ex[s]} for s, c in sent_counts.items() if c}
        st["flags"] = cflags
        col_out[nm] = st
        for f in cflags:
            flags.append({"column": nm, **f})
        log(f"   {nm}: {ft} nulls={nulls} distinct={nd} flags={[f['flag'] for f in cflags]} "
            f"t={time.time() - t0:.0f}s")

    # match-level
    mk_info = {}
    if match_agg is not None:
        mk_vals = match_agg.column("v").to_pylist()
        mk_counts = [int(x) for x in match_agg.column("c_sum").to_pylist()]
        have = set(mk_vals)
        zero_row = sorted(universe - have)
        not_in_header = sorted(have - universe)
        med = float(np.median(mk_counts)) if mk_counts else 0.0
        dist = []
        if med > 0:
            dist = sorted(((abs(c - med) / med, k, c) for k, c in zip(mk_vals, mk_counts)), reverse=True)[:3]
        spanning = 0
        if mk_page_unique:
            cnt = (pa.table({"k": pa.chunked_array(mk_page_unique)})
                   .group_by("k").aggregate([("k", "count")]))
            spanning = int((np.asarray(cnt.column("k_count").to_numpy()) > 1).sum())
        mk_info = {
            "matches_with_rows": len(have), "universe_matches": len(universe),
            "zero_row_matches": len(zero_row), "zero_row_examples": zero_row[:2],
            "keys_not_in_header": len(not_in_header), "not_in_header_examples": not_in_header[:2],
            "rows_per_match_median": med,
            "rows_per_match_min": min(mk_counts) if mk_counts else None,
            "rows_per_match_max": max(mk_counts) if mk_counts else None,
            "far_from_median_top3": [{"match_key": k, "rows": c, "ratio_to_median": round(c / med, 3) if med else None,
                                      "distance_ratio": round(d, 3)} for d, k, c in dist],
            "matches_spanning_pages": spanning,
        }
    # duplicates
    exact = dup_analysis(pages, H_full)
    keyd = dup_analysis(pages, H_key) if key_cols else None
    if exact["groups"]:
        flags.append({"column": "(row)", "flag": "exact_duplicate_rows", "count": exact["extra_rows"],
                      "examples": exact["examples"]})
    if keyd and keyd["groups"]:
        flags.append({"column": "(key " + ",".join(key_cols) + ")", "flag": "duplicate_key_rows",
                      "count": keyd["extra_rows"], "examples": keyd["examples"]})
    if drift:
        flags.append({"column": "(schema)", "flag": "schema_drift_across_pages", "count": len(drift),
                      "examples": drift[:2]})
    if mk_info.get("keys_not_in_header"):
        flags.append({"column": "match_key", "flag": "match_key_not_in_header",
                      "count": mk_info["keys_not_in_header"], "examples": mk_info["not_in_header_examples"]})

    # identical / near-identical columns (same signature first, then _fixed pairs)
    groups = {}
    for nm in names:
        groups.setdefault(sig[nm], []).append(nm)
    cand = []
    for g in groups.values():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                cand.append((g[i], g[j]))
    for nm in names:
        if nm.endswith("_fixed") and nm[:-6] in names:
            pr = (nm[:-6], nm)
            if pr not in cand and (pr[1], pr[0]) not in cand:
                cand.append(pr)
    pair_out = []
    for a, b in cand:
        dc, ex = pair_diff(pages, a, b)
        pair_out.append({"a": a, "b": b, "rows_differ": dc, "share_differ": round(dc / n, 6) if n else None,
                         "examples": ex})
        if dc == 0:
            flags.append({"column": f"{a} vs {b}", "flag": "identical_columns", "count": n})
        elif n and dc / n <= 0.001:
            flags.append({"column": f"{a} vs {b}", "flag": "near_identical_columns", "count": dc,
                          "share": round(dc / n, 6), "examples": ex})
        if a.endswith("_fixed") or b.endswith("_fixed"):
            if dc:
                flags.append({"column": f"{a} vs {b}", "flag": "fixed_column_differs_from_raw",
                              "count": dc, "share": round(dc / n, 6), "examples": ex})

    # second vs tick/64 consistency
    st_out = None
    if sec_pages:
        allv = np.concatenate([d[~np.isnan(d)] for d in sec_pages])
        med = float(np.median(allv))
        outside, sex = 0, []
        for pi, (p, d) in enumerate(zip(pages, sec_pages)):
            m = np.where(np.isnan(d), False, np.abs(d - med) > SECOND_TOL)
            k = int(m.sum())
            outside += k
            if k and len(sex) < 2:
                sex += examples_at(p, np.flatnonzero(m), {"second_minus_tick_over_64": pa.array(d)},
                                   k=2 - len(sex))
        st_out = {"median_second_minus_tick_over_64": med,
                  "p01": float(np.percentile(allv, 1)), "p99": float(np.percentile(allv, 99)),
                  "min": float(allv.min()), "max": float(allv.max()),
                  "tolerance": SECOND_TOL, "rows_outside_tolerance": outside, "examples": sex}
        if outside:
            flags.append({"column": "second vs tick", "flag": "second_not_tick_over_64",
                          "count": outside, "share": round(outside / n, 6), "examples": sex})

    wc = weapon_code_pairs(pages) if ch == "weapon_fire" else None
    if wc and (wc["codes_with_more_than_one_name"] or wc["names_with_more_than_one_code"]):
        flags.append({"column": "weapon_name vs player_weapon_code", "flag": "name_code_mapping_not_1to1",
                      "count": len(wc["codes_with_more_than_one_name"]),
                      "examples": [wc]})

    result = {
        "channel": ch, "folder": folder, "pages": len(pages), "page_rows": page_rows, "rows": n,
        "schema_drift": drift, "key_cols_for_dup_check": key_cols,
        "impossible_by_name_columns_present": {lbl: [nm for nm in names if key in nm]
                                               for key, _, _, lbl in IMPOSSIBLE_BY_NAME},
        "matches": mk_info,
        "dups_exact_rows": exact,
        "dups_key": keyd,
        "second_vs_tick": st_out,
        "pair_checks": pair_out,
        "weapon_code_name_pairs": wc,
        "columns": col_out,
        "flags": flags,
        "runtime_s": round(time.time() - t0, 1),
    }
    log(f"   done {ch} in {time.time() - t0:.0f}s; flags={len(flags)}")
    return result


def main():
    universe, ucol, ucols = header_universe()
    log(f"header universe: {len(universe)} match keys from header column '{ucol}'")
    out = {"universe": {"source": f"header/dataframe_00000 column '{ucol}'", "matches": len(universe)},
           "tick_rate_assumed": TICK_RATE, "channels": {}}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    for ch in CHANNELS:
        out["channels"][ch] = profile_channel(ch, universe)
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(clean(out), f, indent=1, default=str)
    log(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
