"""Profile the economy channel group of the 2026-10-01_to_08 csds tome.

Streams each channel page by page and column by column (low free RAM).
Writes G:/pureskill/csds-cs2/notes/eda/economy.json.

Run:  cd G:\\pureskill\\csds-cs2 ; pixi run python -I scripts\\eda\\profile_economy.py
"""
import collections
import glob
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"
OUT_JSON = "G:/pureskill/csds-cs2/notes/eda/economy.json"
CHANNELS = ["item_equip", "item_pickup", "item_dropped", "item_refund", "rank_update"]
SENTINELS = [4294967295, 65535, 32767, 2147483647, -1, 999, 9999]
STR_SENTINELS = ["", "None", "none", "null", "NULL", "nan", "NaN", "-1", "unknown"]
BAD_NAME_TOKENS = ["health", "armor", "money", "ammo", "ping"]
SPIKE_SHARE = 0.20
MAX_EX = 2

HDR = pq.read_table(ROOT + "/header/dataframe_00000", columns=["key", "tick_rate"])
UNIVERSE = set(HDR.column("key").to_pylist())
TICK_RATE = dict(zip(HDR.column("key").to_pylist(), HDR.column("tick_rate").to_pylist()))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def is_str(t):
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def is_float(t):
    return pa.types.is_floating(t)


def fmt(v):
    if isinstance(v, float):
        return float(v)
    return v


NONNEG_NAMES = {"round", "tick", "second", "player_id", "entity_id", "def_index", "rank_old",
                "rank_new", "win_count", "weapon_type_code", "player_weapon_code"}


def to_f64(a):
    return pc.cast(a, pa.float64())


def null_total_rate_of(nulls, nan_only, N):
    return (nulls + nan_only) / N


def combo_mask(masks):
    def fn(arr):
        out = masks[0](arr)
        for m in masks[1:]:
            out = pc.or_(out, m(arr))
        return out
    return fn


def rule_checks(lname, uv, uc):
    """Name-based value rules. Returns (hits, masks) where masks are pyarrow mask builders."""
    hits, masks = [], []

    def add(desc, npm, pam):
        if len(uv) == 0:
            return
        cnt = int(uc[npm].sum())
        if cnt:
            hits.append({"rule": desc, "count": cnt})
            masks.append(pam)

    if lname in NONNEG_NAMES:
        add("negative values", uv < 0, lambda a: pc.less(to_f64(a), 0.0))
    if lname == "player_phi_ang":
        add("outside [-180,180]", (uv < -180) | (uv > 180),
            lambda a: pc.or_(pc.less(to_f64(a), -180.0), pc.greater(to_f64(a), 180.0)))
    if lname == "player_theta_ang":
        add("outside [-90,90]", (uv < -90) | (uv > 90),
            lambda a: pc.or_(pc.less(to_f64(a), -90.0), pc.greater(to_f64(a), 90.0)))
    if lname == "player_team_code":
        add("team code outside {0,1,2,3}", ~np.isin(uv, [0, 1, 2, 3]),
            lambda a: pc.and_(pc.is_valid(a), pc.invert(pc.is_in(to_f64(a), value_set=pa.array([0.0, 1.0, 2.0, 3.0])))))
    if lname == "round":
        add("round > 60", uv > 60, lambda a: pc.greater(to_f64(a), 60.0))
    return hits, masks


def read_col(p, c, n, ctype):
    """Column c of page p; if the page lacks c (schema drift) return an all-null array of ctype."""
    names = pq.ParquetFile(p).schema_arrow.names
    if c in names:
        return pq.read_table(p, columns=[c]).column(c).combine_chunks()
    if is_str(ctype):
        return pa.nulls(n, pa.string())
    return pa.nulls(n, pa.float64())


def example_rows(pages, c, maskfn, ctype=pa.float64(), k=MAX_EX):
    """Return up to k rows (with match context) where maskfn(column array) is True."""
    out = []
    for p in pages:
        ctx = pq.read_table(p, columns=["match_key", "round", "tick", "player_id"])
        arr = read_col(p, c, ctx.num_rows, ctype)
        m = pc.fill_null(maskfn(arr), False).to_numpy(zero_copy_only=False)
        ids = np.flatnonzero(m)[: k - len(out)]
        if len(ids):
            sub = ctx.take(pa.array(ids)).to_pylist()
            vals = arr.take(pa.array(ids)).to_pylist()
            for rec, v in zip(sub, vals):
                out.append({
                    "file": os.path.basename(os.path.dirname(p)) + "/" + os.path.basename(p),
                    "match_key": rec["match_key"], "round": rec["round"], "tick": rec["tick"],
                    "player_id": rec["player_id"], "value": fmt(v),
                })
        if len(out) >= k:
            break
    return out[:k]


def eq_mask(t, v):
    if pa.types.is_boolean(t):
        return lambda arr: pc.equal(arr, pa.scalar(bool(v)))
    if is_str(t):
        return lambda arr: pc.equal(arr, v)
    return lambda arr: pc.equal(pc.cast(arr, pa.float64()), float(v))


def null_mask(t):
    if is_float(t):
        return lambda arr: pc.or_(pc.is_null(arr), pc.is_nan(arr))
    return lambda arr: pc.is_null(arr)


def profile_channel(ch):
    d = os.path.join(ROOT, f"basic_{ch}.2026-09-18,2026-10-08")
    pages = sorted(glob.glob(os.path.join(d, "dataframe_*")))
    page_rows = [pq.ParquetFile(p).metadata.num_rows for p in pages]
    N = int(sum(page_rows))
    page_maps = [{f.name: f.type for f in pq.ParquetFile(p).schema_arrow} for p in pages]
    cols = []
    for m in page_maps:
        for k in m:
            if k not in cols:
                cols.append(k)
    types = {}
    for c in cols:
        for m in page_maps:
            if c in m:
                types[c] = m[c]
                break
    drift = {}
    for c in cols:
        per = {os.path.basename(p): (str(m[c]) if c in m else "MISSING") for p, m in zip(pages, page_maps)}
        if len(set(per.values())) > 1:
            drift[c] = per
    col_order_variants = len(set(tuple(m.keys()) for m in page_maps))
    log(f"[{ch}] pages={len(pages)} rows={N} cols(union)={len(cols)} drifted_cols={list(drift)}")

    res = {"channel": ch, "tome_folder": os.path.basename(d), "pages": len(pages), "rows": N,
           "column_count_union": len(cols), "column_count_by_page": [len(m) for m in page_maps],
           "schema_drift": drift, "column_order_variants": col_order_variants,
           "rows_by_page": page_rows}

    # ---- match level --------------------------------------------------------
    mk_counts = collections.Counter()
    page_keysets = []
    page_summ = []
    for p, n in zip(pages, page_rows):
        arr = pq.read_table(p, columns=["match_key"]).column("match_key").combine_chunks()
        vc = pc.value_counts(arr)
        ks = vc.field("values").to_pylist()
        cs = vc.field("counts").to_pylist()
        keys = set()
        null_mk = 0
        for k, cnt in zip(ks, cs):
            if k is None:
                null_mk += cnt
            else:
                mk_counts[k] += cnt
                keys.add(k)
        page_keysets.append(keys)
        page_summ.append({"file": os.path.basename(p), "rows": n, "distinct_match_key": len(keys),
                          "null_match_key": null_mk})
    overlap = {}
    for i in range(len(pages)):
        for j in range(i + 1, len(pages)):
            s = len(page_keysets[i] & page_keysets[j])
            if s:
                overlap[f"{os.path.basename(pages[i])}&{os.path.basename(pages[j])}"] = s

    keys_in = set(mk_counts)
    zero_rows = UNIVERSE - keys_in
    counts_arr = np.array(list(mk_counts.values()), dtype=float)
    med = float(np.median(counts_arr))
    per_match = {
        "universe_matches_header": len(UNIVERSE),
        "distinct_match_key": len(keys_in),
        "zero_row_matches": len(zero_rows),
        "match_key_not_in_header": len(keys_in - UNIVERSE),
        "rows_per_match_min": int(counts_arr.min()), "rows_per_match_p25": float(np.percentile(counts_arr, 25)),
        "rows_per_match_median": med, "rows_per_match_p75": float(np.percentile(counts_arr, 75)),
        "rows_per_match_max": int(counts_arr.max()),
    }
    dev_rows = sorted(mk_counts.items(), key=lambda kv: -abs(kv[1] - med))[:3]
    per_match["top3_far_from_median_with_rows"] = [
        {"match_key": k, "rows": int(v), "abs_dev_from_median": abs(v - med)} for k, v in dev_rows]
    zero_list = sorted(zero_rows)[:3]
    per_match["zero_row_match_examples"] = zero_list
    res["match_level"] = {"pages": page_summ, "page_key_overlap": overlap, "per_match": per_match}
    log(f"[{ch}] distinct match_key={len(keys_in)} zero-row={len(zero_rows)} median/match={med}")

    # ---- column level ---------------------------------------------------------
    row_hash = [np.zeros(n, dtype=np.uint64) for n in page_rows]
    P = np.uint64(1000003)
    digests = {}
    cols_out = {}
    for c in cols:
        t = types[c]
        nulls = 0
        nan_only = 0
        hashes = []
        vals = []
        str_counts = collections.Counter()
        missing_pages = [os.path.basename(p) for p, m in zip(pages, page_maps) if c not in m]
        for pi, p in enumerate(pages):
            arr = read_col(p, c, page_rows[pi], t)
            nulls += arr.null_count
            if is_str(t):
                filled = pc.fill_null(arr, "\x00<NULL>")
                h = pd.util.hash_array(filled.to_numpy(zero_copy_only=False))
                vc = pc.value_counts(arr)
                for v, k in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
                    str_counts[v] += k
            else:
                f = pc.cast(arr, pa.float64()).to_numpy(zero_copy_only=False)
                nanm = np.isnan(f)
                nan_only += int(nanm.sum()) - arr.null_count
                h = pd.util.hash_array(f)
                vals.append(f[~nanm])
            hashes.append(h)
            row_hash[pi] = row_hash[pi] * P + h
        digests[c] = hashlib.sha1(np.concatenate(hashes).tobytes()).hexdigest()
        del hashes

        st = {"dtype": str(t), "null_count": int(nulls), "null_rate": nulls / N,
              "nan_count": int(nan_only), "nan_rate": nan_only / N}
        flags = []
        if is_str(t):
            items = [(k, v) for k, v in str_counts.items() if k is not None]
            items.sort(key=lambda kv: -kv[1])
            distinct = len(items)
            st["distinct_nonnull"] = distinct
            st["top_values"] = [{"value": k, "count": int(v), "share": v / N} for k, v in items[:5]]
            top_val, top_cnt = (items[0] if items else (None, 0))
            top_share = top_cnt / N
            # sentinel-like strings
            sent_hits = [{"value": s, "count": int(str_counts.get(s, 0))} for s in STR_SENTINELS
                         if str_counts.get(s, 0)]
            st["sentinel_hits"] = sent_hits
            if sent_hits:
                ex = []
                for s in sent_hits[:1]:
                    ex += example_rows(pages, c, eq_mask(t, s["value"]), ctype=t)
                flags.append({"flag": "sentinel_like_string", "detail": sent_hits, "examples": ex})
        else:
            v_all = np.concatenate(vals) if vals else np.array([], dtype=float)
            del vals
            uv, uc = np.unique(v_all, return_counts=True)
            distinct = len(uv)
            st["distinct_nonnull"] = int(distinct)
            if len(v_all):
                st["min"] = float(v_all.min())
                st["max"] = float(v_all.max())
                st["median"] = float(np.median(v_all))
            order = np.argsort(-uc, kind="stable")[:5]
            st["top_values"] = [{"value": float(uv[i]), "count": int(uc[i]), "share": float(uc[i]) / N}
                                for i in order]
            if distinct:
                top_val = float(uv[np.argmax(uc)])
                top_cnt = int(uc.max())
            else:
                top_val, top_cnt = None, 0
            top_share = top_cnt / N
            sent_hits = [{"value": s, "count": int(uc[uv == s].sum())} for s in SENTINELS if (uv == s).any()]
            st["sentinel_hits"] = sent_hits
            if sent_hits:
                ex = []
                for s in sent_hits[:2]:
                    ex += example_rows(pages, c, eq_mask(t, s["value"]), ctype=t, k=1)
                flags.append({"flag": "sentinel_like_value",
                              "detail": sent_hits, "examples": ex[:MAX_EX]})
            neg = int(uc[uv < 0].sum())
            st["negative_count"] = neg
            if c in ("player_team_code", "weapon_type_code", "player_weapon_code"):
                st["distinct_values_all"] = [float(x) for x in uv[:60]]
            lname = c.lower()
            rule_hits, rule_masks = rule_checks(lname, uv, uc)
            if lname.endswith("_vel"):
                st["abs_gt_3000_count"] = int(uc[np.abs(uv) > 3000].sum())
            if lname in ("player_x_pos", "player_y_pos", "player_z_pos"):
                st["abs_gt_8192_count"] = int(uc[np.abs(uv) > 8192].sum())
            if rule_hits:
                flags.append({"flag": "impossible_value_for_name", "detail": rule_hits,
                              "examples": example_rows(pages, c, combo_mask(rule_masks), ctype=t)})

        # ---- generic flags -----------------------------------------------------
        st["missing_in_pages"] = missing_pages
        st["dtype_by_page"] = drift.get(c, {})
        if c in drift:
            flags.insert(0, {"flag": "schema_drift", "detail": {"dtype_or_presence_by_page": drift[c],
                                                                "rows_missing_column": int(sum(page_rows[i] for i, p in enumerate(pages) if os.path.basename(p) in missing_pages))},
                             "examples": [{"page": n, "value": v} for n, v in list(drift[c].items())[:2]]})
        if distinct == 1:
            flags.append({"flag": "constant", "detail": {"value": st["top_values"][0]["value"],
                                                         "null_rate": null_total_rate_of(nulls, nan_only, N)},
                          "examples": []})
        if distinct == 0:
            flags.append({"flag": "all_null", "detail": {}, "examples": []})
        null_total_rate = (nulls + nan_only) / N
        if null_total_rate > 0.5:
            flags.append({"flag": "over_50pct_null", "detail": {"null_rate": null_total_rate},
                          "examples": example_rows(pages, c, null_mask(t), ctype=t)})
        st["null_rate_total"] = null_total_rate
        if top_val is not None and top_share > SPIKE_SHARE:
            flags.append({"flag": "spike_top_value_gt_20pct",
                          "detail": {"value": top_val, "share": top_share, "count": int(top_cnt)},
                          "examples": example_rows(pages, c, eq_mask(t, top_val), ctype=t)})
        st["top_value"] = fmt(top_val)
        st["top_share"] = top_share
        st["flags"] = flags
        cols_out[c] = st
        log(f"[{ch}] col {c}: distinct={distinct} null_rate={null_total_rate:.4f} "
            f"top_share={top_share:.4f} flags={[f['flag'] for f in flags]}")

    res["columns"] = cols_out

    # ---- identical columns -----------------------------------------------------
    by_digest = collections.defaultdict(list)
    for c, dg in digests.items():
        by_digest[dg].append(c)
    res["identical_column_groups"] = [v for v in by_digest.values() if len(v) > 1]
    log(f"[{ch}] identical groups: {res['identical_column_groups']}")

    # ---- duplicate rows ----------------------------------------------------------
    extra_full = 0
    groups_full = 0
    full_ex = []
    key_cols = [c for c in ["match_key", "round", "tick", "player_id", "item", "entity_id"] if c in cols]
    extra_key = 0
    key_match_round = set()
    key_ex = []
    for pi, p in enumerate(pages):
        rh = row_hash[pi]
        uh, uc = np.unique(rh, return_counts=True)
        extra_full += int((uc - 1).sum())
        groups_full += int((uc > 1).sum())
        if (uc > 1).any() and len(full_ex) < MAX_EX:
            dup_hash = uh[np.argmax(uc > 1)]
            idx = np.flatnonzero(rh == dup_hash)[:2]
            t = pq.read_table(p, columns=["match_key", "round", "tick", "player_id"]).take(pa.array(idx))
            for rec in t.to_pylist():
                full_ex.append({"file": os.path.basename(p), **rec})
        kdf = pq.read_table(p, columns=[c for c in key_cols if c in page_maps[pi]]).to_pandas()
        dupmask = kdf.duplicated(keep="first")
        extra_key += int(dupmask.sum())
        if dupmask.any() and len(key_ex) < MAX_EX:
            rec = kdf[dupmask].head(MAX_EX - len(key_ex))
            for r in rec.to_dict("records"):
                key_ex.append({"file": os.path.basename(p), **{k: (v if not isinstance(v, float) else float(v)) for k, v in r.items()}})
            mm = kdf[kdf.duplicated(keep=False)][["match_key", "round"]].drop_duplicates()
            key_match_round.update(map(tuple, mm.values.tolist()))
        del kdf
    res["duplicates"] = {
        "exact_full_row_extra_copies": extra_full,
        "exact_full_row_groups": groups_full,
        "exact_full_row_examples": full_ex,
        "key_dup_extra_copies": extra_key,
        "key_dup_key": key_cols,
        "key_dup_match_round_pairs": len(key_match_round),
        "key_dup_examples": key_ex,
    }
    log(f"[{ch}] dup full={extra_full} key={extra_key}")

    # ---- cross-column consistency extras ---------------------------------------
    extras = []
    if "player_id" in cols and "player_id_fixed" in cols:
        mism = 0
        nn = 0
        ex = []
        for p in pages:
            t = pq.read_table(p, columns=["player_id", "player_id_fixed", "match_key", "round", "tick"])
            a = t.column("player_id").to_numpy(zero_copy_only=False).astype(float)
            b = pc.cast(t.column("player_id_fixed"), pa.float64()).to_numpy(zero_copy_only=False)
            ok = ~np.isnan(b) & ~np.isnan(a)
            nn += int(ok.sum())
            bad = ok & (a != b)
            mism += int(bad.sum())
            if bad.any() and len(ex) < MAX_EX:
                for i in np.flatnonzero(bad)[: MAX_EX - len(ex)]:
                    rec = t.slice(int(i), 1).to_pylist()[0]
                    ex.append({"file": os.path.basename(p), **rec, "player_id_fixed": float(b[i])})
        extras.append({"check": "player_id != player_id_fixed (both non-null)", "count": mism,
                       "compared": nn, "examples": ex})

    if "rank_old" in cols and "rank_new" in cols and "rank_change" in cols:
        mism = 0
        nn = 0
        ex = []
        for p in pages:
            t = pq.read_table(p, columns=["rank_old", "rank_new", "rank_change", "match_key", "round", "tick"])
            o = t.column("rank_old").to_numpy(zero_copy_only=False).astype(float)
            n_ = t.column("rank_new").to_numpy(zero_copy_only=False).astype(float)
            rc = pc.cast(t.column("rank_change"), pa.float64()).to_numpy(zero_copy_only=False)
            ok = ~np.isnan(o) & ~np.isnan(n_) & ~np.isnan(rc)
            nn += int(ok.sum())
            bad = ok & (n_ - o != rc)
            mism += int(bad.sum())
            if bad.any() and len(ex) < MAX_EX:
                for i in np.flatnonzero(bad)[: MAX_EX - len(ex)]:
                    rec = t.slice(int(i), 1).to_pylist()[0]
                    ex.append({"file": os.path.basename(p), **rec, "expected_change": float(n_[i] - o[i])})
        extras.append({"check": "rank_change != rank_new - rank_old", "count": mism,
                       "compared": nn, "examples": ex})
        # distribution of rank_change
        rc_all = []
        for p in pages:
            rc_all.append(pq.read_table(p, columns=["rank_change"]).column("rank_change").to_numpy(zero_copy_only=False))
        rc_all = np.concatenate(rc_all)
        vals, cnts = np.unique(rc_all[~np.isnan(rc_all)], return_counts=True)
        extras.append({"check": "rank_change distribution (top 8 by count)",
                       "values": [{"value": float(v), "count": int(c_)} for v, c_ in
                                  sorted(zip(vals, cnts), key=lambda x: -x[1])[:8]],
                       "distinct": int(len(vals))})

    if "tick" in cols and "second" in cols:
        resid_all = []
        ex = []
        for p in pages:
            t = pq.read_table(p, columns=["tick", "second", "match_key", "round"])
            tk = t.column("tick").to_numpy(zero_copy_only=False).astype(float)
            sec = pc.cast(t.column("second"), pa.float64()).to_numpy(zero_copy_only=False)
            mk = t.column("match_key").to_pylist()
            rate = np.array([TICK_RATE.get(k, np.nan) for k in mk], dtype=float)
            ok = ~np.isnan(sec) & ~np.isnan(rate)
            r = (tk - sec * rate)[ok]
            resid_all.append(r)
        r = np.concatenate(resid_all)
        extras.append({"check": "tick - second*tick_rate(header)",
                       "compared": int(r.size),
                       "median": float(np.median(r)),
                       "abs_gt_2_ticks": int((np.abs(r) > 2).sum()),
                       "abs_gt_2_ticks_pct": float((np.abs(r) > 2).mean() * 100),
                       "p01": float(np.percentile(r, 1)), "p99": float(np.percentile(r, 99))})

    if "item" in cols and "def_index" in cols:
        pairs = set()
        pages_with_def = [p for p, m in zip(pages, page_maps) if "def_index" in m]
        def_null_rows = 0
        for p in pages_with_def:
            raw = pq.read_table(p, columns=["item", "def_index"]).to_pandas()
            def_null_rows += int(raw["def_index"].isna().sum())
            t = raw.dropna().drop_duplicates()
            pairs.update((str(a), int(b)) for a, b in t.values.tolist())
        extras_def_cov = {"pages_with_def_index": [os.path.basename(p) for p in pages_with_def],
                          "rows_with_def_index": int(sum(page_rows[i] for i, p in enumerate(pages) if "def_index" in page_maps[i])),
                          "null_def_index_rows_on_pages_with_column": def_null_rows}
        by_item = collections.defaultdict(set)
        by_def = collections.defaultdict(set)
        for it, dfi in pairs:
            by_item[it].add(dfi)
            by_def[dfi].add(it)
        multi_def = {k: sorted(v) for k, v in by_item.items() if len(v) > 1}
        multi_item = {k: sorted(v) for k, v in by_def.items() if len(v) > 1}
        extras.append({"check": "item <-> def_index one-to-one",
                       **extras_def_cov,
                       "distinct_pairs": len(pairs),
                       "items_with_multiple_def_index": len(multi_def),
                       "def_index_with_multiple_items": len(multi_item),
                       "examples_items": dict(list(multi_def.items())[:MAX_EX]),
                       "examples_def_index": dict(list(multi_item.items())[:MAX_EX])})

    if "reason" in cols:
        vc_all = collections.Counter()
        for p in pages:
            arr = pq.read_table(p, columns=["reason"]).column("reason").combine_chunks()
            vc = pc.value_counts(arr)
            for v, k in zip(vc.field("values").to_pylist(), vc.field("counts").to_pylist()):
                vc_all[v] += k
        extras.append({"check": "reason value counts", "values": [{"value": k, "count": int(v)} for k, v in vc_all.most_common(10)]})

    res["extras"] = extras
    return res


PARTIAL_DIR = "C:/Users/<user>/AppData/Local/Temp/claude/C--Users-<user>-Documents-pureskill/7308a8e7-a47e-4034-b140-5c511acb0aa4/scratchpad/economy_partials"


def partial_path(ch):
    return os.path.join(PARTIAL_DIR, f"{ch}.json")


def run_channels(channels):
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    for ch in channels:
        t0 = time.time()
        res = profile_channel(ch)
        with open(partial_path(ch), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1, default=str)
        log(f"[{ch}] done in {time.time()-t0:.1f}s -> {partial_path(ch)}")


def merge():
    out = {"group": "economy", "tome": "tomes/2026-10-01_to_08/tome/csds",
           "universe_note": "header tome key column (1752 matches); keyset per page lists that page's match keys",
           "channels": {}}
    for ch in CHANNELS:
        if os.path.exists(partial_path(ch)):
            with open(partial_path(ch), encoding="utf-8") as fh:
                out["channels"][ch] = json.load(fh)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    log("merged", list(out["channels"]), "->", OUT_JSON)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--merge":
        merge()
    else:
        run_channels(args or CHANNELS)
        merge()
