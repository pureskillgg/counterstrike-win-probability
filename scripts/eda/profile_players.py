"""Profile the 'players' channel group of the 2026-10-01_to_08 CSDS tome.

Read-only. Reads every dataframe_* page of each channel folder with pyarrow,
writes notes/eda/players.md and notes/eda/players.json.

Run from the project root:  pixi run python -I scripts/eda/profile_players.py
"""
import collections
import glob
import hashlib
import json
import math
import os

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = "G:/pureskill/csds-cs2/tomes/2026-10-01_to_08/tome/csds"
OUT_DIR = "G:/pureskill/csds-cs2/notes/eda"
OUT_MD = os.path.join(OUT_DIR, "players.md")
OUT_JSON = os.path.join(OUT_DIR, "players.json")

CHANNELS = [
    "player_info", "player_personal", "player_name", "player_connect", "player_disconnect",
    "player_sound", "player_spawn", "player_chat", "team_change", "bot_takeover",
]
SENTINELS = [4294967295, 65535, 32767, 2147483647, -1, 999, 9999]
NULLISH_STR = {"", "none", "null", "nan", "n/a", "na"}
EXAMPLE_COLS = ["match_key", "round", "tick", "player_id", "player_index"]

# Ranked goofy list (most misleading first), written after reviewing the first run.
GOOFY_TOP = [
    "Duplicate matches leak across splits: player_info has 47 groups (99 matches) and team_change has 46 groups (97 matches) with identical content under different match keys, e.g. copies stamped 2026-10-08 18:08:58 and 18:09:22 UTC. The bakeoff reports the same 46 groups (97 matches) from its round_end and death events, with copies on both sides of the train/test split. Any match-level split or grouped CV will overstate scores.",
    "Player identity is not recoverable: name, clan_tag, name_new, name_old and chat text are the literal 'redacted' in every row; steam_id is a single letter (19 values A-S), not a SteamID; player_id is a per-match slot 0-19 whose numbering disagrees with player_id_fixed on about 91% of player_info rows. Cross-match player keys are invalid.",
    "Player count is not fixed at 10: 149 header matches are Wingman (is_wingman True), and 148 of them show only 4 player IDs. A further 271 matches show 11 player IDs in player_info, 263 with a header max_unique_players of 10 (likely bot slots). The bakeoff drops 3,663 rounds for player_count != 10, so this anomaly is already costing data.",
    "player_sound covers 20 of 1,751 matches with rows (1.1%); all 20 are dated 2026-10-08. Rows per match run 6.6K-21.6K versus a median of about 200 elsewhere, and a zero-fill join reads 'no sound' as signal for 98.9% of matches. Within a tick, 23,967 excess rows repeat the same (match, round, player_id, tick) with different radius and duration; is_step is exactly duration == 0.5.",
    "Spawn and team_change velocities include impossible magnitudes: 16,801 player_spawn rows (5.1%) exceed 10,000 units/s (max about 344,000), and 889 team_change rows do (max about 280,000). 6.5% of spawn rows exceed 250 units/s, while player_chat tops out near 258. player_phi_ang is exactly 90.0 in 67% of spawn rows and stays within 1-179.",
]


def short(key):
    if key is None:
        return None
    parts = str(key).split("/")
    return parts[-2] if len(parts) >= 2 else str(key)


def is_num(typ):
    return pa.types.is_integer(typ) or pa.types.is_floating(typ)


def jsonable(x):
    if isinstance(x, dict):
        return {k: jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        f = float(x)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def load_header():
    path = os.path.join(ROOT, "header", "dataframe_00000")
    cols = ["key", "max_rounds", "tick_rate", "map_name", "is_wingman", "max_unique_players", "unique_steamids"]
    t = pq.read_table(path, columns=cols)
    df = t.to_pandas()
    df["max_rounds"] = pd.to_numeric(df["max_rounds"], errors="coerce")
    df["tick_rate"] = pd.to_numeric(df["tick_rate"], errors="coerce")
    df["is_wingman"] = df["is_wingman"].astype(bool)
    return df.drop_duplicates("key").set_index("key")


def load_channel(ch):
    d = os.path.join(ROOT, f"basic_{ch}.2026-09-18,2026-10-08")
    pages = sorted(glob.glob(os.path.join(d, "dataframe_*")))
    tabs = [pq.read_table(p) for p in pages]
    if len(tabs) == 1:
        t = tabs[0]
    else:
        t = pa.concat_tables(tabs, promote_options="permissive")
    return t, [os.path.basename(p) for p in pages], [pq.ParquetFile(p).metadata.num_rows for p in pages]


def ex_rows(t, mask, n=2, cols=EXAMPLE_COLS):
    m = pc.fill_null(mask, False)
    idx = pc.indices_nonzero(m).to_pylist()[:n]
    out = []
    for i in idx:
        r = {}
        for c in cols:
            if c in t.column_names:
                v = t.column(c)[i].as_py()
                if c == "match_key":
                    v = short(v)
                r[c] = v
        out.append(r)
    return out


def flag(column, detail, count, examples):
    return {"column": column, "detail": detail, "count": int(count), "examples": examples}


def profile_column(t, name, n):
    col = t.column(name)
    typ = col.type
    nulls = col.null_count
    valid = pc.drop_null(col)
    nd = pc.count_distinct(valid, mode="only_valid").as_py()
    rec = {
        "name": name,
        "dtype": str(typ),
        "null_count": int(nulls),
        "null_rate": round(nulls / n, 6) if n else None,
        "n_distinct": int(nd),
    }
    nan_n = inf_n = 0
    if is_num(typ) and len(valid):
        arr = valid.to_numpy(zero_copy_only=False)
        if pa.types.is_floating(typ):
            arr = arr.astype(float)
            nan_n = int(np.isnan(arr).sum())
            inf_n = int(np.isinf(arr).sum())
            fin = arr[np.isfinite(arr)]
        else:
            fin = arr.astype(np.int64)
        if fin.size:
            rec["min"] = float(fin.min()) if pa.types.is_floating(typ) else int(fin.min())
            rec["max"] = float(fin.max()) if pa.types.is_floating(typ) else int(fin.max())
            rec["median"] = float(np.median(fin))
        else:
            rec["min"] = rec["max"] = rec["median"] = None
    rec["nan_count"] = nan_n
    rec["inf_count"] = inf_n
    top1 = 0
    top1_val = None
    if len(valid):
        vc = pc.value_counts(valid)
        vals = vc.field("values").to_pylist()
        cnts = vc.field("counts").to_pylist()
        order = sorted(range(len(vals)), key=lambda i: -cnts[i])
        if order:
            top1 = cnts[order[0]]
            top1_val = vals[order[0]]
        if nd <= 50:
            rec["top5"] = [
                {"value": vals[i], "count": int(cnts[i]), "share": round(cnts[i] / n, 4)}
                for i in order[:5]
            ]
    rec["top1_value"] = top1_val
    rec["top1_share_of_rows"] = round(top1 / n, 4) if n else None
    if pa.types.is_string(typ) or pa.types.is_large_string(typ):
        lowered = pc.utf8_lower(pc.utf8_trim_whitespace(col))
        empty_n = int(pc.sum(pc.fill_null(pc.equal(lowered, ""), False)).as_py() or 0)
        nullish_n = int(pc.sum(pc.fill_null(pc.is_in(lowered, value_set=pa.array(sorted(NULLISH_STR))), False)).as_py() or 0)
        rec["empty_or_blank_string_count"] = empty_n
        rec["nullish_string_count"] = nullish_n
    return rec


def column_flags(t, rec, n):
    out = {"constant": [], "over50_null": [], "sentinel": [], "spike": []}
    name = rec["name"]
    col = t.column(name)
    if rec["n_distinct"] <= 1:
        val = rec["top1_value"]
        out["constant"].append(flag(name, f"one distinct non-null value ({'all null' if rec['n_distinct']==0 else 'value=' + repr(val)}), null rate {rec['null_rate']}", n, [{"value": val, "null_rate": rec["null_rate"]}]))
    if rec["null_rate"] is not None and rec["null_rate"] > 0.5:
        out["over50_null"].append(flag(name, f"null rate {rec['null_rate']:.4f}", rec["null_count"], []))
    if is_num(col.type):
        hits = {}
        for s in SENTINELS:
            c = int(pc.sum(pc.fill_null(pc.equal(col, s), False)).as_py() or 0)
            if c:
                hits[s] = c
        if hits:
            first_s = sorted(hits)[0]
            mask = pc.equal(col, first_s)
            out["sentinel"].append(flag(name, "sentinel-like values: " + ", ".join(f"{k}x{v}" for k, v in hits.items()), sum(hits.values()), ex_rows(t, mask, cols=EXAMPLE_COLS + [name])))
    if rec["top1_share_of_rows"] is not None and rec["top1_share_of_rows"] > 0.2 and rec["n_distinct"] > 1:
        v = rec["top1_value"]
        out["spike"].append(flag(name, f"value {v!r} holds {rec['top1_share_of_rows']:.1%} of rows", rec["top1_share_of_rows"] * n, []))
    return out


def impossible_checks(t, n):
    """Name-driven impossible-value rules. Reports which columns were checked."""
    rules = {"health": (True, 100), "armor": (True, 100), "money": (False, 16000), "cash": (False, 16000),
             "ammo": (False, None), "ping": (False, None)}
    checked = []
    flags = []
    for name in t.column_names:
        low = name.lower()
        if not is_num(t.column(name).type):
            continue
        hit = [k for k in rules if k in low]
        if not hit:
            continue
        checked.append(name)
        col = t.column(name)
        neg = int(pc.sum(pc.fill_null(pc.less(col, 0), False)).as_py() or 0)
        if neg:
            flags.append(flag(name, "negative value", neg, ex_rows(t, pc.less(col, 0), cols=EXAMPLE_COLS + [name])))
        for k in hit:
            cap = rules[k][1]
            if cap is not None and rules[k][0] is True:
                over = int(pc.sum(pc.fill_null(pc.greater(col, cap), False)).as_py() or 0)
                if over:
                    flags.append(flag(name, f"above {cap}", over, ex_rows(t, pc.greater(col, cap), cols=EXAMPLE_COLS + [name])))
            if cap is not None and rules[k][0] is False and k in ("money", "cash"):
                over = int(pc.sum(pc.fill_null(pc.greater(col, cap), False)).as_py() or 0)
                if over:
                    flags.append(flag(name, f"above {cap}", over, ex_rows(t, pc.greater(col, cap), cols=EXAMPLE_COLS + [name])))
    return flags, checked


def identical_columns(t, recs):
    cands = [r["name"] for r in recs if r["null_count"] < t.num_rows]
    pairs = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = t.column(cands[i]), t.column(cands[j])
            try:
                eq = pc.equal(a, b)
            except Exception:
                continue
            ok = pc.or_(pc.fill_null(eq, False), pc.and_(pc.is_null(a), pc.is_null(b)))
            if pc.all(ok).as_py():
                pairs.append([cands[i], cands[j]])
    return pairs, cands


def duplicate_checks(t, df):
    out = {}
    full = df.duplicated(keep="first")
    dup_full = int(full.sum())
    ex = []
    if dup_full:
        sub = df[df.duplicated(keep=False)]
        for _, row in sub.head(2).iterrows():
            ex.append({k: (short(row[k]) if k == "match_key" else (None if pd.isna(row[k]) else row[k])) for k in EXAMPLE_COLS if k in df.columns})
    out["full_row_excess"] = {"count": dup_full, "examples": ex}
    key_cols = [c for c in ["match_key", "round", "player_id", "tick"] if c in df.columns]
    if "player_id" in key_cols:
        kd = df.duplicated(subset=key_cols, keep="first")
        ex2 = []
        varying = {}
        patterns = []
        if int(kd.sum()):
            sub2 = df[df.duplicated(subset=key_cols, keep=False)].sort_values(key_cols)
            for _, row in sub2.head(2).iterrows():
                ex2.append({k: (short(row[k]) if k == "match_key" else (None if pd.isna(row[k]) else row[k])) for k in key_cols})
            grp = sub2.groupby(key_cols)
            for c in df.columns:
                if c in key_cols:
                    continue
                k = int((grp[c].nunique(dropna=False) > 1).sum())
                if k:
                    varying[c] = k
            vcols = list(varying)[:4]
            if vcols and len(vcols) <= 4:
                pats = sub2[vcols].astype(str).agg("|".join, axis=1).value_counts().head(3)
                patterns = [{"pattern": f"{'|'.join(vcols)} = {p}", "rows": int(n_)} for p, n_ in pats.items()]
        out["key_excess_" + "_".join(key_cols)] = {"count": int(kd.sum()), "key": key_cols, "examples": ex2,
                                                    "groups_with_varying_column_counts": varying, "top_patterns_in_dup_rows": patterns}
    return out


def one_to_one_pairs(t, recs, identical):
    """Column pairs that are functionally one-to-one (each value of A maps to one value of B and back), excluding identical pairs."""
    cands = [r["name"] for r in recs if 2 <= r["n_distinct"] <= 20]
    if len(cands) < 2:
        return []
    df = t.select(cands).to_pandas()
    skip = {tuple(p) for p in identical} | {tuple(reversed(p)) for p in identical}
    pairs = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            if (a, b) in skip:
                continue
            sub = df[[a, b]].dropna()
            if len(sub) == 0:
                continue
            if sub.groupby(a)[b].nunique().max() == 1 and sub.groupby(b)[a].nunique().max() == 1:
                pairs.append([a, b])
    return pairs


def match_outliers(t, universe):
    vc = pc.value_counts(t.column("match_key"))
    keys = vc.field("values").to_pylist()
    cnts = vc.field("counts").to_pylist()
    per = dict(zip(keys, cnts))
    rows_matches = list(per.values())
    zero = len(universe - set(per))
    med = float(np.median(rows_matches)) if rows_matches else 0.0
    ranked = sorted(per.items(), key=lambda kv: -abs(kv[1] - med))[:3]
    top3 = [{"match": short(k), "rows": int(c), "median_nonzero_match": med,
             "ratio_to_median": round(c / med, 3) if med else None} for k, c in ranked]
    return {"zero_row_matches": zero, "median_rows_per_nonzero_match": med,
            "max_rows_per_match": int(max(rows_matches)) if rows_matches else 0,
            "min_rows_per_match": int(min(rows_matches)) if rows_matches else 0,
            "top3_far_from_median": top3}, per


def extra_checks(t, df, hdr):
    ex = {}
    d = df.copy()
    d["max_rounds"] = d["match_key"].map(hdr["max_rounds"])
    d["tick_rate"] = d["match_key"].map(hdr["tick_rate"])
    d["map_name"] = d["match_key"].map(hdr["map_name"])
    d["is_wingman"] = d["match_key"].map(hdr["is_wingman"]).astype(bool)
    keys_u = pd.Series(d["match_key"].unique())
    date_of = keys_u.str.split("/").str[1:4].str.join("-")
    ex["match_key_dates"] = {"matches": int(keys_u.size), "by_date_top5": {str(k): int(v) for k, v in date_of.value_counts().head(5).items()},
                             "first_date": str(date_of.min()), "last_date": str(date_of.max())}
    if "player_id" in d.columns:
        ppm = d.groupby("match_key").player_id.nunique()
        ex["distinct_player_id_per_match"] = {str(int(k)): int(v) for k, v in ppm.value_counts().sort_index().items()}
        ex["wingman_matches_in_channel"] = int(d.loc[d["is_wingman"], "match_key"].nunique())
    if "steam_id" in d.columns:
        sv = sorted(d["steam_id"].dropna().unique().tolist())
        ex["steam_id_values"] = {"distinct": len(sv), "all_values": sv[:30],
                                 "length_counts": {str(k): int(v) for k, v in d["steam_id"].str.len().value_counts().items()}}
    if "round" in d.columns:
        m = d["round"] <= 0
        ex["round_le_0"] = {"count": int(m.sum()), "examples": [dict(match=short(r.match_key), round=int(r.round)) for r in d[m].head(2).itertuples()]}
        mx = d["round"] > d["max_rounds"] + 0
        ex["round_gt_header_max_rounds"] = {"count": int(mx.sum()), "examples": [dict(match=short(r.match_key), round=int(r.round), max_rounds=float(r.max_rounds)) for r in d[mx].head(2).itertuples()]}
        ex["round_min_max"] = {"min": int(d["round"].min()), "max": int(d["round"].max())}
    if "tick" in d.columns:
        m = d["tick"] < 0
        ex["tick_negative"] = {"count": int(m.sum())}
    if "second" in d.columns and "tick" in d.columns and d["tick_rate"].notna().any():
        exp_sec = d["tick"] / d["tick_rate"]
        diff = (d["second"] - exp_sec).abs()
        bad = diff > 2.0
        ex["second_vs_tick_div_rate"] = {
            "rows_gt_2s_off": int(bad.sum()), "rows_checked": int(diff.notna().sum()),
            "median_abs_diff_s": float(np.nanmedian(diff)),
            "examples": [dict(match=short(r.match_key), round=int(r.round), tick=int(r.tick), second=float(r.second), tick_over_rate=round(float(r.tick / r.tick_rate), 3)) for r in d[bad].head(2).itertuples()] if bad.any() else [],
        }
    if "player_x_vel" in d.columns and "player_y_vel" in d.columns:
        spd = np.hypot(d["player_x_vel"], d["player_y_vel"])
        ex["horizontal_speed"] = {"rows_gt_250": int((spd > 250).sum()), "rows_gt_1000": int((spd > 1000).sum()),
                                  "rows_gt_10000": int((spd > 10000).sum()), "max": float(np.nanmax(spd)),
                                  "matches_with_gt_1000": int(d.loc[spd > 1000, "match_key"].nunique()),
                                  "rows_checked": int(spd.notna().sum())}
    if "player_z_pos" in d.columns:
        m = d["player_z_pos"] > 10000
        if m.any():
            ex["z_pos_gt_10000"] = {"count": int(m.sum()), "maps": {str(k): int(v) for k, v in d.loc[m, "map_name"].value_counts().head(5).items()}}
    if "player_phi_ang" in d.columns:
        m = (d["player_phi_ang"] < -180) | (d["player_phi_ang"] > 360)
        ex["phi_outside_-180_360"] = {"count": int(m.sum()), "range": [float(d["player_phi_ang"].min()), float(d["player_phi_ang"].max())],
                                      "share_exactly_90": round(float((d["player_phi_ang"] == 90).mean()), 4)}
    if "player_theta_ang" in d.columns:
        m = (d["player_theta_ang"] < -90) | (d["player_theta_ang"] > 90)
        ex["theta_beyond_pm90_informational"] = {"count": int(m.sum()), "range": [float(d["player_theta_ang"].min()), float(d["player_theta_ang"].max())],
                                                 "note": "theta spans +/-180 so it looks like yaw, not pitch"}
    if "player_id" in d.columns and "player_id_fixed" in d.columns:
        both = d["player_id"].notna() & d["player_id_fixed"].notna()
        mism = both & (d["player_id"].astype("float64") != d["player_id_fixed"].astype("float64"))
        ex["player_id_vs_player_id_fixed"] = {"rows_both_present": int(both.sum()), "mismatch": int(mism.sum()),
                                              "examples": [dict(match=short(r.match_key), round=int(r.round), player_id=float(r.player_id), player_id_fixed=float(r.player_id_fixed)) for r in d[mism].head(2).itertuples()] if mism.any() else []}
        pid_neg = d["player_id"] < 0
        ex["player_id_negative"] = {"count": int(pid_neg.sum())}
    if "player_id_fixed" in d.columns:
        ex["player_id_fixed_null_rate"] = round(float(d["player_id_fixed"].isna().mean()), 6)
    if "is_bot" in d.columns:
        ex["is_bot_true_rows"] = {"count": int((d["is_bot"] == True).sum()),  # noqa: E712
                                  "share": round(float((d["is_bot"] == True).mean()), 4)}  # noqa: E712
    if "old_team_code" in d.columns and "silent" in d.columns:
        z = d["old_team_code"] == 0
        s_ = d["silent"] == False  # noqa: E712
        ex["old_team_code0_vs_silent_false"] = {"both": int((z & s_).sum()), "old0_only": int((z & ~s_).sum()), "silent_false_only": int((~z & s_).sum())}
    if "old_team_code" in d.columns and "new_team_code" in d.columns and "round" in d.columns:
        sw = d[d["old_team_code"].isin([2, 3])]
        swapped = sw["old_team_code"] != sw["new_team_code"]
        r12 = sw["round"] == 12
        r27 = sw["round"] == 27
        ex["team_swap_by_round"] = {"round12_rows": int(r12.sum()), "round12_swapped_share": round(float(swapped[r12].mean()), 4) if r12.any() else None,
                                    "round27_rows": int(r27.sum()), "round27_swapped_share": round(float(swapped[r27].mean()), 4) if r27.any() else None,
                                    "note": "round 12 and 27 are half-time style swaps: all but a few rows change team"}
    for c in ["radius", "duration", "wins", "elo_platform", "rank_platform", "rank_raw", "rank", "player_index"]:
        if c in d.columns:
            neg = d[c] < 0
            if neg.any():
                ex[f"{c}_negative"] = {"count": int(neg.sum()), "examples": [dict(match=short(r.match_key), round=int(r.round), value=float(getattr(r, c))) for r in d[neg].head(2).itertuples()]}
    return ex


def profile_channel(ch, universe, hdr):
    t, page_names, page_rows = load_channel(ch)
    n = t.num_rows
    recs = [profile_column(t, name, n) for name in t.column_names]
    df = t.to_pandas()
    mo, per = match_outliers(t, universe)
    keys_in = set(per)
    flags = {"constant": [], "over50_null": [], "sentinel": [], "spike": []}
    for rec in recs:
        cf = column_flags(t, rec, n)
        for k in flags:
            flags[k].extend(cf[k])
    imp, imp_checked = impossible_checks(t, n)
    ident, cands = identical_columns(t, recs)
    o2o = one_to_one_pairs(t, recs, ident)
    dups = duplicate_checks(t, df)
    extra = extra_checks(t, df, hdr)
    not_in_header = len(keys_in - universe)
    return {
        "channel": ch,
        "tome_folder": f"basic_{ch}.2026-09-18,2026-10-08",
        "pages": page_names,
        "page_rows": page_rows,
        "rows": int(n),
        "ncols": len(t.column_names),
        "distinct_match_keys": len(keys_in),
        "universe_matches": len(universe),
        "coverage_share": round(len(keys_in) / len(universe), 4),
        "zero_row_matches": mo["zero_row_matches"],
        "match_keys_not_in_header": not_in_header,
        "columns": recs,
        "flags": {**flags, "impossible_value": imp, "identical_columns": ident, "one_to_one_pairs": o2o,
                  "impossible_checked_columns": imp_checked,
                  "identical_checked_columns": cands,
                  "duplicates": dups,
                  "match_row_count_outliers": mo},
        "extra": extra,
    }


def fmt_val(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        if abs(v) >= 1e6 or (abs(v) < 1e-3 and v != 0):
            return f"{v:.3g}"
        return f"{v:,.3f}".rstrip("0").rstrip(".") if v != int(v) else f"{int(v):,}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def md_channel(r):
    L = []
    L.append(f"## {r['channel']}")
    L.append("")
    L.append(f"- Tome folder: `tomes/2026-10-01_to_08/tome/csds/{r['tome_folder']}` ({len(r['pages'])} page(s): {', '.join(r['pages'])}, rows per page {r['page_rows']})")
    L.append(f"- Rows: **{r['rows']:,}**, columns: {r['ncols']}")
    L.append(f"- Distinct match_key: **{r['distinct_match_keys']:,}** of {r['universe_matches']:,} header matches (coverage {r['coverage_share']:.1%})")
    L.append(f"- Matches with zero rows in this channel: **{r['zero_row_matches']:,}**")
    if r["match_keys_not_in_header"]:
        L.append(f"- match_keys not in header: {r['match_keys_not_in_header']}")
    mo = r["flags"]["match_row_count_outliers"]
    L.append(f"- Rows per non-zero match: min {mo['min_rows_per_match']:,}, median {mo['median_rows_per_nonzero_match']:,.1f}, max {mo['max_rows_per_match']:,}")
    L.append("")
    L.append("### Columns")
    L.append("")
    L.append("| column | dtype | null % | distinct | min | max | median | top 5 (if <=50 distinct) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in r["columns"]:
        top = ""
        if "top5" in c:
            top = "; ".join(f"`{x['value']}`={x['count']:,} ({x['share']:.1%})" for x in c["top5"])
        nullp = f"{(c['null_rate'] or 0)*100:.2f}"
        L.append(f"| {c['name']} | {c['dtype']} | {nullp} | {c['n_distinct']:,} | {fmt_val(c.get('min'))} | {fmt_val(c.get('max'))} | {fmt_val(c.get('median'))} | {top} |")
    L.append("")
    L.append("### Flags")
    L.append("")
    f = r["flags"]

    def block(title, items, empty_text):
        L.append(f"**{title}**")
        if not items:
            L.append(f"- none. {empty_text}")
        for it in items:
            ex = "; ".join(json.dumps(e, default=str) for e in it["examples"]) if it.get("examples") else ""
            L.append(f"- `{it['column']}`: {it['detail']}; count {it['count']:,}" + (f". e.g. {ex}" if ex else ""))
        L.append("")

    block("Constant columns", f["constant"], "")
    block("Columns more than 50% null", f["over50_null"], "")
    block("Sentinel-like values", f["sentinel"], "")
    block("Spike (one value >20% of rows)", f["spike"], "")
    impc = f["impossible_checked_columns"]
    block("Impossible values (name rules)", f["impossible_value"], "")
    L.append(f"Impossible-value rule columns checked: {', '.join(impc) if impc else 'none (no column name matches health/armor/money/cash/ammo/ping)'}.")
    L.append("")
    ident = f["identical_columns"]
    L.append("**Identical columns (equal in every row)**")
    if not ident:
        L.append(f"- none among {len(f['identical_checked_columns'])} non-all-null columns.")
    for a, b in ident:
        L.append(f"- `{a}` == `{b}` in every row")
    L.append("")
    dd = f["duplicates"]
    L.append("**Duplicate rows**")
    fr = dd["full_row_excess"]
    L.append(f"- Full-row duplicates (every column equal): excess rows {fr['count']:,}" + (f". e.g. {json.dumps(fr['examples'], default=str)}" if fr["examples"] else ""))
    for k, v in dd.items():
        if k.startswith("key_excess_"):
            L.append(f"- Same key {v['key']} repeated, other columns free to differ: excess rows {v['count']:,}" + (f". e.g. {json.dumps(v['examples'], default=str)}" if v["examples"] else ""))
            if v.get("groups_with_varying_column_counts"):
                L.append(f"  - columns that differ inside those key groups (count of groups): {json.dumps(v['groups_with_varying_column_counts'])}")
            if v.get("top_patterns_in_dup_rows"):
                L.append(f"  - most common patterns in duplicate rows: {json.dumps(v['top_patterns_in_dup_rows'])}")
    L.append("")
    L.append("**One-to-one column pairs (functionally equivalent, not identical)**")
    if not f.get("one_to_one_pairs"):
        L.append("- none among columns with 2-20 distinct values.")
    for a, b in f.get("one_to_one_pairs", []):
        L.append(f"- `{a}` <-> `{b}` map one-to-one")
    L.append("")
    L.append("**Matches far from median row count (top 3 by |rows - median|, median over non-zero matches)**")
    for t3 in mo["top3_far_from_median"]:
        L.append(f"- match `{t3['match']}`: {t3['rows']:,} rows, ratio to median {t3['ratio_to_median']}")
    L.append("")
    L.append("### Other checks and oddities")
    L.append("")
    if not r["extra"]:
        L.append("- none recorded.")
    for k, v in r["extra"].items():
        L.append(f"- `{k}`: {json.dumps(v, default=str)}")
    L.append("")
    return "\n".join(L)


def build_md(results, xc):
    L = []
    L.append("# players channel group: EDA profile")
    L.append("")
    L.append("Tome: `tomes/2026-10-01_to_08/tome/csds`, channels `basic_<name>.2026-09-18,2026-10-08`. Match universe = 1,752 keys in `header/dataframe_00000` (`key` column). Read-only; generated by `scripts/eda/profile_players.py`.")
    L.append("")
    L.append("## Goofy list (ranked by how much each could mislead a model)")
    L.append("")
    if GOOFY_TOP:
        for i, g in enumerate(GOOFY_TOP, 1):
            L.append(f"{i}. {g}")
    else:
        L.append("TBD")
    L.append("")
    L.append("## Definitions and checks run")
    L.append("")
    L.append("- Null rate = nulls / rows. Distinct = distinct non-null values. Median over non-null, non-NaN values.")
    L.append("- Sentinel list: 4294967295, 65535, 32767, 2147483647, -1, 999, 9999 (numeric columns).")
    L.append("- Spike = one non-null value holds more than 20% of all rows in the channel (boolean and zero-heavy flags are expected here; read with care).")
    L.append("- Impossible-value rules are name-driven: health/armor negative or >100, money/cash negative or >16000, ammo/ping negative. Also negative radius/duration/wins/rank/elo/player_index and round or tick <= 0 are reported in 'Other checks'.")
    L.append("- Identical columns: pairwise null-aware equality across all non-all-null columns.")
    L.append("- Duplicates: full-row excess duplicates, and excess rows on (match_key, round, player_id[, tick]).")
    L.append("- Match row counts: per match, counted over matches with at least one row. The top-3 list ranks by |rows - median|. Zero-row matches are reported separately.")
    L.append("- One-to-one pairs: two columns with 2-20 distinct values where each value of one determines exactly one value of the other, in both directions (e.g. is_step and duration).")
    L.append("- Also checked: NaN/inf counts, empty or null-like strings, second vs tick/tick_rate, coordinate and speed ranges (horizontal speed > 250, 1000, 10000 units/s), z-position by map, phi/theta ranges, player_id vs player_id_fixed, distinct player IDs per match vs header max_unique_players and is_wingman, steam_id values, round vs header max_rounds, team swap rounds, match-key dates, schema types across channels.")
    L.append("- Interpretation notes: round values above header max_rounds (24) are overtime (contiguous, up to 48) and are expected. z_pos above 10,000 is only on de_vertigo, which is real map geometry. theta beyond +/-90 is expected for a yaw-like angle.")
    L.append("")
    for r in results:
        L.append(md_channel(r))
    L.append("## Cross-channel checks")
    L.append("")
    pr = xc["personal_rows_without_player_info_row"]
    L.append(f"- player_personal rows with no matching (match_key, round, player_id) in player_info: {pr['count']} rows across {pr['matches']} matches; rounds {pr['rounds']}; is_bot True in {pr['is_bot_true']}.")
    for ch, v in xc["duplicate_match_content"].items():
        L.append(f"- Duplicate match content in `{ch}` (identical rows under different match_keys, sorted-row md5): {v['groups']} groups, {v['matches']} matches. First groups: {json.dumps(v['examples_first_2_groups'])}")
    L.append("")
    return "\n".join(L)


def dup_match_groups(ch):
    """Matches whose full channel content is identical under different match_keys."""
    t, _, _ = load_channel(ch)
    df = t.to_pandas()
    cols = [c for c in df.columns if c != "match_key"]
    hashes = {}
    for mk, g in df.groupby("match_key"):
        g2 = g[cols].sort_values(cols).astype(str)
        hashes[mk] = hashlib.md5(g2.to_csv(index=False).encode()).hexdigest()
    s = pd.Series(hashes)
    d = s[s.duplicated(keep=False)]
    groups = [list(v.index) for _, v in d.groupby(d.values)]
    return {"groups": len(groups), "matches": int(d.size),
            "examples_first_2_groups": [[short(k) for k in g] for g in groups[:2]]}


def cross_channel():
    """Row-key checks that join two channels in the group, plus duplicate-match content checks."""
    out = {"duplicate_match_content": {ch: dup_match_groups(ch) for ch in ["player_info", "player_personal", "team_change"]}}
    pi_t, _, _ = load_channel("player_info")
    pp_t, _, _ = load_channel("player_personal")
    pi = pi_t.select(["match_key", "round", "player_id"]).to_pandas()
    pp = pp_t.select(["match_key", "round", "player_id", "is_bot"]).to_pandas()
    m = pp.merge(pi, how="left", indicator=True)
    ro = m[m["_merge"] == "left_only"]
    out["personal_rows_without_player_info_row"] = {
        "count": int(len(ro)), "matches": int(ro["match_key"].nunique()),
        "rounds": {str(int(k)): int(v) for k, v in ro["round"].value_counts().items()},
        "is_bot_true": int((ro["is_bot"] == True).sum()),  # noqa: E712
    }
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    hdr = load_header()
    universe = set(hdr.index)
    results = []
    for ch in CHANNELS:
        print("profiling", ch, flush=True)
        results.append(profile_channel(ch, universe, hdr))
    xc = cross_channel()
    md = build_md(results, xc)
    with open(OUT_MD, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(md)
    payload = {"generated_from": "scripts/eda/profile_players.py", "universe_matches": len(universe),
               "goofy_top": GOOFY_TOP, "cross_channel": xc, "channels": results}
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(jsonable(payload), fh, indent=2, default=str)
    print("wrote", OUT_MD, OUT_JSON)


if __name__ == "__main__":
    main()
