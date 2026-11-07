# -*- coding: utf-8 -*-
"""Profile the combat channel group of the 2026-10-01_to_08 CS2 tome.

Run from G:\\pureskill\\csds-cs2 with:
    pixi run python -I scripts\\eda\\combat_profile.py

Reads every page of each combat channel with pyarrow and writes
notes\\eda\\combat.md and notes\\eda\\combat.json. Read-only on the tome.
"""
import datetime
import json
import math
import os
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

REPO = r"G:\pureskill\csds-cs2"
TOME = os.path.join(REPO, "tomes", "2026-10-01_to_08", "tome", "csds")
OUT_DIR = os.path.join(REPO, "notes", "eda")
CHANNELS = ["player_death", "player_hurt", "other_death", "bullet_damage",
            "player_blind", "player_footstep"]
SENTINELS = [4294967295, 65535, 32767, 2147483647, -1, 999, 9999]
SPIKE_SHARE = 0.20
NULL_SHARE = 0.50
TOP_N = 5
TOP_VALUES_MAX_DISTINCT = 50
EXAMPLES = 2
DEFAULT_TICK_RATE = 64
NONNEG_BY_NAME = ["health_removed", "armor_removed", "effective_health_removed",
                  "distance", "blind_duration", "num_penetrations",
                  "penetration_amount", "second", "tick", "round"]
KEY_COLS = ["match_key", "round", "tick", "player_id", "attacker_id", "other_type"]

GOOFY_TOP5 = [
    {"title": "bullet_damage is missing for 61% of matches",
     "detail": "1,075 of 1,752 header matches have zero bullet_damage rows. 1,050 of them are dated 2026-10-08, the batch where only 83 of 1,133 matches have rows, and 1,060 have match_type 'unknown'. This looks like a batch extraction gap. A model reads these rounds as no shots fired unless they are filtered out."},
    {"title": "player_hurt.health_removed is raw damage with overkill, not HP lost",
     "detail": "9.8% of rows are above 100, every one of them on a kill (health = 0), with a maximum of 4,999. health_removed equals previous health minus current health on only 38% of rows (72% within 1). effective_health_removed is above 100 in 174 rows (max 446) on a 0-100 HP scale. Derive HP lost from health before using it as a damage target."},
    {"title": "player_id and player_id_fixed are per-match relabels, not player identities",
     "detail": "Raw player_id is a 0-9 slot and player_id_fixed is a 1-10 slot, reshuffled per match. Within a match the mapping is one-to-one (99.99% of pairs), but raw 0 maps to 15 different fixed values across matches. The two disagree on 91% of player_death rows. 391 matches also carry raw ids 10-19. Any cross-match player feature built on either column is wrong."},
    {"title": "other_death has zero rows for every de_dust2 match",
     "detail": "316 of the 370 zero-row other_death matches are de_dust2, and all 316 Dust2 matches are empty. The other large maps are 97%+ covered. This looks like a map-level extraction gap, so a model sees Dust2 rounds with no prop or chicken deaths."},
    {"title": "Assister fields have three encodings, and positions are null for 60% of real assists",
     "detail": "No assister is encoded as null (139,891 rows) or 65535 (33,399 rows), while raw 0 is a real player slot (7,865 rows). Of 80,190 rows with a real assister, 47,827 (60%) have null assister position, velocity and angle. A raw 65535 reads as a huge player id."},
]

FOLLOWUPS = [
    "Sentinel hits on tick and second (65535, 32767, 9999, 999) are real times, not sentinels. second equals tick/64 on those rows (the residual check passes in every channel), so do not drop them.",
    "entity_id = 999 in player_blind (80 rows) is not special: 998 has 120 rows and 1000 has 146.",
    "player_death z above 10,000 (3,782 rows, 32 matches) is all de_vertigo, a tall map, so it is real.",
    "Velocities above 3,000 units/s occur in a handful of rows per channel (about 3 to 6), with values near 200,000. These look like teleport or parse glitches. Exact -1.0 velocity values (up to 145 rows in one column) look like placeholders; unverified.",
    "player_hurt attacker_id_fixed is double on page 0 and int64 on page 1; concatenating the pages coerces it to float.",
    "player_blind player_id == attacker_id on 50,507 rows (20.7%). These could be own flashes; unverified.",
    "Angle rule corrected: phi_ang runs 1 to 179 (polar, inside [0,180]) and theta_ang spans -180 to 180 (azimuth). The earlier theta [-90,90] rule was wrong, so its 100k+ hits per channel are not defects.",
    "Zero-row matches outside other_death: player_blind 21 (19 on 2026-10-08), player_footstep 18 (17 on 2026-10-08), player_hurt 1 (de_dust2, 2026-10-07, competitive).",
]


def py(v):
    """Convert numpy / pandas scalars to plain JSON-safe Python values."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        f = float(v)
        if math.isnan(f):
            return None
        if math.isinf(f):
            return "inf" if f > 0 else "-inf"
        return f
    if isinstance(v, pd.Timestamp):
        return str(v)
    return str(v)


def short(key):
    """csds/2026/10/01/<uuid>/csds -> <uuid>."""
    if key is None or (isinstance(key, float) and math.isnan(key)):
        return None
    parts = str(key).split("/")
    return parts[-2] if len(parts) >= 2 else str(key)


def load_header():
    p = os.path.join(TOME, "header", "dataframe_00000")
    h = pq.read_table(p, columns=["key", "match_id", "tick_rate"]).to_pandas()
    rate_map = dict(zip(h["key"], h["tick_rate"].astype(float)))
    rate_counts = {str(int(k)): int(v) for k, v in h["tick_rate"].value_counts().items()}
    return sorted(h["key"].unique().tolist()), rate_map, {
        "matches": int(len(h)),
        "distinct_key": int(h["key"].nunique()),
        "tick_rate_counts": rate_counts,
    }


def read_channel(ch):
    d = os.path.join(TOME, f"basic_{ch}.2026-09-18,2026-10-08")
    with open(os.path.join(d, "tome"), encoding="utf-8") as fh:
        man = json.load(fh)
    page_names = [p["dataframe"]["dataframe"] for p in man["pages"]]
    on_disk = sorted(f for f in os.listdir(d) if f.startswith("dataframe_"))
    frames, schemas, page_rows = [], [], []
    for i, nm in enumerate(page_names):
        t = pq.read_table(os.path.join(d, nm))  # pyarrow reads every page
        schemas.append(t.schema)
        page_rows.append(int(t.num_rows))
        df = t.to_pandas()
        df["_page"] = i
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    del frames
    return {
        "dir": d,
        "pages": page_names,
        "on_disk": on_disk,
        "page_rows": page_rows,
        "schemas": schemas,
        "df": df,
        "manifest_complete": man.get("isComplete"),
    }


def schema_drift(schemas):
    out = []
    if len(schemas) < 2:
        return out
    base = {f.name: str(f.type) for f in schemas[0]}
    for i, s in enumerate(schemas[1:], start=1):
        cur = {f.name: str(f.type) for f in s}
        out.append({
            "page": i,
            "missing_vs_page0": sorted(set(base) - set(cur)),
            "extra_vs_page0": sorted(set(cur) - set(base)),
            "type_changes": [{"col": k, "page0": base[k], "page": cur[k]}
                             for k in base if k in cur and base[k] != cur[k]],
            "same_column_order": list(cur) == list(base),
        })
    return out


def arrow_type(schemas, c):
    for s in schemas:
        i = s.get_field_index(c)
        if i >= 0:
            return str(s.field(i).type)
    return None


def kind_of(s):
    if pd.api.types.is_bool_dtype(s):
        return "bool"
    nn = s.dropna()
    if len(nn) and isinstance(nn.iloc[0], (bool, np.bool_)):
        return "bool"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    return "string"


def example_rows(df, idx, value_col=None):
    out = []
    for i in list(idx)[:EXAMPLES]:
        r = {"page": py(df.at[i, "_page"]),
             "match_id": short(df.at[i, "match_key"]),
             "round": py(df.at[i, "round"]) if "round" in df.columns else None,
             "tick": py(df.at[i, "tick"]) if "tick" in df.columns else None}
        if value_col is not None:
            r["value"] = py(df.at[i, value_col])
        out.append(r)
    return out


def bmask(x):
    """Null-safe boolean numpy mask (nullable dtypes produce NA in comparisons)."""
    return np.asarray(pd.Series(x).fillna(False), dtype=bool)


def ex_mask(df, mask, value_col=None):
    return example_rows(df, df.index[bmask(mask)], value_col)


def profile_columns(df, kinds, schemas):
    n = len(df)
    recs = []
    for c, k in kinds.items():
        s = df[c]
        nulls = int(s.isna().sum())
        nn = s.dropna()
        nun = int(nn.nunique())
        rec = {"name": c, "arrow_type": arrow_type(schemas, c),
               "pandas_dtype": str(s.dtype), "kind": k,
               "null_count": nulls,
               "null_rate": round(nulls / n, 6) if n else None,
               "distinct_non_null": nun,
               "min": None, "max": None, "median": None, "top5": None}
        if k == "numeric" and len(nn):
            rec["min"] = py(nn.min())
            rec["max"] = py(nn.max())
            rec["median"] = py(nn.median())
        if 0 < nun <= TOP_VALUES_MAX_DISTINCT:
            vc = nn.value_counts()
            rec["top5"] = [{"value": py(v), "count": int(cnt), "share": round(cnt / n, 6)}
                           for v, cnt in vc.head(TOP_N).items()]
        recs.append(rec)
    return recs


def impossible_checks(df, kinds):
    out = []
    for c, k in kinds.items():
        if k != "numeric":
            continue
        if re.search(r"(^|_)health$", c):
            group = "health"
        elif re.search(r"(^|_)armor$", c):
            group = "armor"
        elif "money" in c:
            group = "money"
        elif "ammo" in c:
            group = "ammo"
        elif re.search(r"(^|_)ping$", c):
            group = "ping"
        else:
            continue
        nn = df[c].dropna()
        rules = []
        if group in ("health", "armor", "money", "ammo", "ping"):
            rules.append(("negative " + group, nn < 0))
        if group in ("health", "armor"):
            rules.append((group + " > 100", nn > 100))
        if group == "money":
            rules.append(("money > 16000", nn > 16000))
        for rule, m in rules:
            cnt = int(m.sum())
            if cnt:
                out.append({"col": c, "rule": rule, "count": cnt,
                            "examples": example_rows(df, nn.index[bmask(m)], c)})
    return out


def identical_pairs(df, kinds):
    cols = list(kinds)
    out = []
    for i, a in enumerate(cols):
        sa = df[a]
        na = sa.isna()
        for b in cols[i + 1:]:
            if kinds[a] != kinds[b]:
                continue
            sb = df[b]
            nb = sb.isna()
            if not np.array_equal(bmask(na), bmask(nb)):
                continue
            if bool(na.all()):
                continue
            if bool(bmask((sa == sb) | (na & nb)).all()):
                out.append({"a": a, "b": b, "kind": kinds[a],
                            "both_constant": bool(sa.nunique() == 1)})
    return out


def duplicates(df, data_cols):
    dmask = df[data_cols].duplicated(keep="first").values
    n_dup = int(dmask.sum())
    res = {"full_row_duplicate_rows": n_dup,
           "full_row_duplicate_share": round(n_dup / len(df), 6) if len(df) else None,
           "match_round_groups_affected": 0,
           "matches_affected": 0,
           "examples": []}
    if n_dup:
        sub = df.loc[dmask, ["match_key", "round"]]
        res["match_round_groups_affected"] = int(len(sub.drop_duplicates()))
        res["matches_affected"] = int(sub["match_key"].nunique())
        res["examples"] = ex_mask(df, dmask)
    keyc = [c for c in KEY_COLS if c in df.columns]
    kmask = df.duplicated(subset=keyc, keep=False).values
    res["keyed_cols"] = keyc
    res["keyed_rows_in_dup_groups_all_copies"] = int(kmask.sum())
    res["keyed_examples"] = ex_mask(df, kmask)
    return res


def match_outliers(df, universe_keys):
    cnt = df.groupby("match_key").size()
    not_in_header = sorted(set(cnt.index) - set(universe_keys))
    full = cnt.reindex(universe_keys, fill_value=0).astype("int64")
    med = float(full.median())
    dev = (full - med).abs().sort_values(ascending=False)
    top_all = [{"match_id": short(k), "rows": int(full[k]), "median_rows": med,
                "ratio_to_median": round(float(full[k]) / med, 4) if med else None}
               for k in dev.index[:3]]
    nz = full[full > 0]
    med_nz = float(nz.median())
    dev_nz = (nz - med_nz).abs().sort_values(ascending=False)
    top_nz = [{"match_id": short(k), "rows": int(nz[k]), "median_rows": med_nz,
               "ratio_to_median": round(float(nz[k]) / med_nz, 4) if med_nz else None}
              for k in dev_nz.index[:3]]
    zero = full.index[full == 0]
    return {
        "universe_matches": int(len(full)),
        "median_rows_all_matches_incl_zero": med,
        "top3_abs_deviation_all_matches": top_all,
        "median_rows_nonzero_matches": med_nz,
        "top3_abs_deviation_nonzero_matches": top_nz,
        "zero_row_matches": int(len(zero)),
        "zero_row_examples": [short(k) for k in zero[:3]],
        "match_keys_not_in_header": len(not_in_header),
        "match_keys_not_in_header_examples": [short(k) for k in not_in_header[:3]],
    }


def extra_checks(df, rate_map):
    out = {}
    cols = set(df.columns)
    out["null_counts"] = {c: int(df[c].isna().sum()) for c in ["match_key", "round", "tick", "player_id", "attacker_id"] if c in cols}
    if "round" in cols:
        out["round_min"] = py(df["round"].min())
        out["round_max"] = py(df["round"].max())
        out["round_distinct"] = int(df["round"].nunique())
    if "tick" in cols:
        out["tick_min"] = py(df["tick"].min())
        out["tick_max"] = py(df["tick"].max())
        dt = df.groupby("match_key", sort=False)["tick"].diff()
        dec = bmask(dt < 0)
        out["tick_decreases_in_stored_order"] = {
            "rows": int(dec.sum()),
            "matches": int(df.loc[dec, "match_key"].nunique()),
            "examples": ex_mask(df, dec),
        }
    if {"second", "tick"} <= cols:
        rate = df["match_key"].map(rate_map).fillna(DEFAULT_TICK_RATE).astype(float)
        resid = df["second"] - df["tick"] / rate
        med = float(resid.median())
        bad = bmask((resid - med).abs() > 1.0)
        per = resid.groupby(df["match_key"]).median()
        off = per[(per - med).abs() > 1.0]
        out["second_vs_tick_over_tick_rate"] = {
            "median_residual_s": med,
            "rows_off_gt_1s": int(bad.sum()),
            "matches_median_off_gt_1s": int(len(off)),
            "examples": ex_mask(df, bad),
        }
    neg = {}
    for c in NONNEG_BY_NAME:
        if c in cols:
            m = bmask(df[c] < 0)
            if m.sum():
                neg[c] = {"count": int(m.sum()), "examples": ex_mask(df, m, c)}
    out["negative_where_nonnegative_by_name"] = neg
    ranges = {}
    for c in df.columns:
        if c.startswith("_"):
            continue
        if re.search(r"_phi_ang$", c):
            # polar angle, observed range 1..179
            m = bmask((df[c] < 0) | (df[c] > 180))
            ranges[c] = {"rule": "outside [0,180]", "count": int(m.sum())}
        elif re.search(r"_theta_ang$", c):
            # azimuth, observed range -180..180
            m = bmask((df[c] < -180) | (df[c] > 180))
            ranges[c] = {"rule": "outside [-180,180]", "count": int(m.sum())}
        elif re.search(r"_(x|y|z)_pos$", c):
            m = bmask(df[c].abs() > 10000)
            ranges[c] = {"rule": "abs > 10000", "count": int(m.sum())}
        elif re.search(r"_(x|y|z)_vel$", c):
            m = bmask(df[c].abs() > 3000)
            ranges[c] = {"rule": "abs > 3000", "count": int(m.sum())}
    out["range_checks"] = {k: v for k, v in ranges.items() if v["count"]}
    out["range_checks_run"] = len(ranges)
    empties = {}
    for c in df.columns:
        if c.startswith("_") or kind_of(df[c]) != "string":
            continue
        s = df[c]
        ne = int((s.fillna("") == "").sum())
        if ne:
            empties[c] = ne
    out["empty_string_counts"] = empties
    fixed = []
    for c in df.columns:
        if c.endswith("_fixed") and c[:-6] in cols:
            a, b = df[c[:-6]], df[c]
            both_null = bmask(a.isna() & b.isna())
            diff = bmask((a != b) & ~both_null)
            fixed.append({"base": c[:-6], "fixed": c, "rows_differ": int(diff.sum()),
                          "base_null": int(a.isna().sum()), "fixed_null": int(b.isna().sum()),
                          "examples": ex_mask(df, diff)})
    out["fixed_vs_base_columns"] = fixed
    if {"player_id", "attacker_id"} <= cols:
        same = bmask((df["player_id"] == df["attacker_id"]) & df["player_id"].notna())
        out["player_equals_attacker_rows"] = int(same.sum())
    id_cols = [c for c in df.columns if (c.endswith("_id") or c.endswith("_id_fixed")) and not c.startswith("_")]
    out["id_zero_counts"] = {c: int((df[c] == 0).sum()) for c in id_cols if (df[c] == 0).sum()}
    if {"damage_dir_x", "damage_dir_y", "damage_dir_z"} <= cols:
        v = df[["damage_dir_x", "damage_dir_y", "damage_dir_z"]].values.astype(float)
        norm = np.sqrt((v ** 2).sum(axis=1))
        ok = ~np.isnan(norm)
        not_unit = ok & (np.abs(norm - 1.0) > 0.01)
        out["damage_dir_norm"] = {
            "median_norm": float(np.nanmedian(norm)),
            "zero_vectors": int(np.sum(norm == 0)),
            "not_unit_gt_0_01": int(not_unit.sum()),
            "examples": ex_mask(df, not_unit),
        }
    if "blind_duration" in cols:
        m = bmask(df["blind_duration"] > 10)
        out["blind_duration_gt_10s"] = {"count": int(m.sum()), "examples": ex_mask(df, m, "blind_duration")}
    if {"effective_health_removed", "health_removed"} <= cols:
        m = bmask(df["effective_health_removed"] > df["health_removed"])
        out["effective_health_removed_gt_health_removed"] = {"count": int(m.sum()), "examples": ex_mask(df, m, "effective_health_removed")}
    out["pages_row_sum_ok"] = None
    return out


def channel_flags(df, kinds, n):
    constant, null50, sentinel, spike = [], [], [], []
    for c, k in kinds.items():
        s = df[c]
        nn = s.dropna()
        nu = int(nn.nunique())
        nr = float(s.isna().mean())
        if nu == 1:
            constant.append({"col": c, "value": py(nn.iloc[0]), "null_rate": round(nr, 6)})
        if nr > NULL_SHARE:
            null50.append({"col": c, "null_rate": round(nr, 6), "all_null": nu == 0})
        if k == "numeric" and nu:
            for sv in SENTINELS:
                m = bmask(nn == sv)
                if m.any():
                    sentinel.append({"col": c, "value": sv, "count": int(m.sum()),
                                     "share_of_rows": round(float(m.sum()) / n, 6),
                                     "examples": example_rows(df, nn.index[m], c)})
        if nu >= 2:
            vc = nn.value_counts()
            share = float(vc.iloc[0]) / n
            if share > SPIKE_SHARE:
                top = vc.index[0]
                m = bmask(nn == top)
                spike.append({"col": c, "value": py(top), "count": int(vc.iloc[0]),
                              "share_of_rows": round(share, 6),
                              "examples": example_rows(df, nn.index[m], c)})
    return constant, null50, sentinel, spike


def load_header_full():
    p = os.path.join(TOME, "header", "dataframe_00000")
    h = pq.read_table(p, columns=["key", "map_name", "match_type", "match_date"]).to_pandas()
    h["day"] = h["match_date"].astype(str).str[:10]
    return h.set_index("key")


def coverage_breakdown(present, hdr_full):
    zero = hdr_full[~hdr_full.index.isin(list(present))]
    out = {}
    for col in ["match_type", "day", "map_name"]:
        tot = hdr_full[col].value_counts()
        zc = zero[col].value_counts()
        out[col] = [{"group": str(g), "matches": int(tot.get(g, 0)), "zero_row_matches": int(zc[g])}
                    for g in zc.index[:5]]
    return out


def hurt_checks(df):
    """player_hurt: is health_removed the HP actually lost? Overkill and cap checks."""
    d = df.sort_values(["match_key", "round", "player_id", "tick"], kind="stable")
    prev = d.groupby(["match_key", "round", "player_id"])["health"].shift(1)
    ok = prev.notna() & d["health"].notna() & d["health_removed"].notna()
    delta = prev - d["health"]
    diff = (d["health_removed"] - delta).abs()
    n_ok = max(int(ok.sum()), 1)
    over = d["health_removed"] > 100
    eff = d["effective_health_removed"]
    return {
        "health_removed_gt_100_rows": int(over.sum()),
        "health_removed_gt_100_share": round(float(over.mean()), 6),
        "health_removed_gt_100_with_health_zero": int((over & (d["health"] == 0)).sum()),
        "health_removed_max": py(d["health_removed"].max()),
        "rows_with_prev_health": int(ok.sum()),
        "share_health_removed_equals_prev_minus_health": round(float(((diff < 0.5) & ok).sum()) / n_ok, 6),
        "share_health_removed_within_1_of_prev_minus_health": round(float(((diff <= 1) & ok).sum()) / n_ok, 6),
        "effective_health_removed_gt_100_rows": int((eff > 100).sum()),
        "effective_health_removed_max": py(eff.max()),
        "examples_health_removed_gt_100": ex_mask(d, over.values, "health_removed"),
    }


def assister_checks(df):
    if not {"assister_id", "assister_x_pos"} <= set(df.columns):
        return None
    a = df["assister_id"]
    valid = pd.Series(bmask(a.notna() & (a != 65535)), index=df.index)
    pos_null = df["assister_x_pos"].isna()
    n_valid = max(int(valid.sum()), 1)
    return {
        "null": int(a.isna().sum()),
        "value_65535": int(bmask(a == 65535).sum()),
        "value_0_real_slot": int(bmask(a == 0).sum()),
        "valid_assister_rows": int(valid.sum()),
        "valid_rows_with_null_position": int((valid & pos_null).sum()),
        "share_valid_rows_with_null_position": round(float((valid & pos_null).sum()) / n_valid, 6),
        "invalid_rows_with_position": int((~valid & ~pos_null).sum()),
    }


def id_checks(df):
    if "player_id" not in df.columns:
        return None
    per_match = df.groupby("match_key")["player_id"].nunique()
    out = {
        "matches_with_more_than_10_raw_ids": int((per_match > 10).sum()),
        "rows_with_raw_id_outside_0_9": int((~df["player_id"].between(0, 9)).sum()),
    }
    if "player_id_fixed" in df.columns:
        sub = df.dropna(subset=["player_id_fixed"])
        g = sub.groupby(["match_key", "player_id"])["player_id_fixed"].nunique()
        out["raw_to_fixed_single_valued_share_within_match"] = round(float((g == 1).mean()), 6)
        out["fixed_disagrees_with_raw_rows"] = int(((df["player_id"] != df["player_id_fixed"]) & df["player_id_fixed"].notna()).sum())
        out["raw0_distinct_fixed_values_across_matches"] = int(sub.loc[sub["player_id"] == 0, "player_id_fixed"].nunique())
    return out


def render_md(res):
    L = []
    add = L.append

    def cell(v):
        if v is None:
            return ""
        s = str(v)
        if len(s) > 30:
            s = s[:27] + "..."
        return s.replace("|", "\\|")

    def ex_text(exs):
        parts = []
        for e in exs:
            bits = [f"{k}={v}" for k, v in e.items() if v is not None]
            parts.append("{" + ", ".join(bits) + "}")
        return "; ".join(parts) if parts else "none"

    add("# Combat channel group EDA")
    add("")
    add(f"Generated {res['generated']} from `{TOME}`. Channels: " + ", ".join(CHANNELS) + ".")
    add("")
    add(f"Universe: {res['universe']['matches']} matches in the header tome (`tome\\csds\\header`); tick_rate counts {res['universe']['tick_rate_counts']}. "
        "A match counts as zero-row for a channel when its header key has no row in that channel.")
    add("")
    add("## Goofy list (5 most important, ranked by how much they could mislead a model)")
    if GOOFY_TOP5:
        for i, g in enumerate(GOOFY_TOP5, 1):
            add(f"{i}. **{g['title']}.** {g['detail']}")
    else:
        add("(not written yet)")
    add("")
    add("## Follow-up checks (scripts/eda/combat_investigate.py and combat_investigate2.py)")
    for line in FOLLOWUPS:
        add(f"- {line}")
    add("")
    add("## Method")
    add("- Pages: each channel folder's `tome` manifest lists its pages. Page files are extensionless parquet named `dataframe_NNNNN`. Every page is read with pyarrow.parquet and concatenated, with a `_page` index added.")
    add("- Null rate, distinct count (non-null), min/max/median for numeric columns, and top-5 values for columns with 50 or fewer distinct values. Share is over all rows in the channel.")
    add("- Constant: exactly one non-null value. >50% null: null rate over 0.5. Sentinels: exact match to 4294967295, 65535, 32767, 2147483647, -1, 999, 9999 in numeric columns. Spike: one non-null value holds more than 20% of all rows (constant columns excluded).")
    add("- Impossible: `*health`/`*armor` negative or above 100; `*money` negative or above 16000; `*ammo`/`*ping` negative (matched on column name). Extra non-negative-by-name check on removed/distance/duration/tick/round/second columns.")
    add("- Identical: two columns of the same kind, null-aligned, equal in every row (all-null pairs skipped).")
    add("- Duplicates: full-row duplicates over all data columns (the within-match-and-round case is included because match_key and round are in each row). Keyed duplicates on match_key, round, tick, player_id, attacker_id, other_type, reported with all copies, since multi-hit ticks can be legitimate.")
    add("- Match outliers: row count per header match (zero rows included). Median is over all 1752 header matches, and separately over non-zero matches. Top 3 by absolute deviation from the median.")
    add("- Extra: tick order inside a match in stored order; second vs tick/tick_rate residual (tick_rate from header); angle and position/velocity ranges; empty strings; `*_fixed` vs base column disagreement; id zeros; bullet damage_dir unit norm; blind duration over 10 s; effective vs raw health removed; page-spanning matches; schema drift across pages.")
    add("")
    for ch in CHANNELS:
        r = res["channels"][ch]
        add(f"## {ch}")
        add("")
        add(f"- Table: `{r['table']}`. Pages (manifest): {len(r['pages'])} {r['pages']}. On disk: {r['on_disk']}. Page rows: {r['page_rows']}. Sum matches total: {r['rows_sum_check']}.")
        add(f"- Rows: {r['rows_total']:,}. Distinct match_key: {r['distinct_match_key']} of {r['universe_matches']} header matches. Zero-row matches: {r['zero_row_matches']}. Match keys not in header: {r['match_keys_not_in_header']}.")
        add(f"- Matches with rows on more than one page: {r['matches_spanning_pages']}.")
        if r["zero_row_matches"]:
            zb = r["zero_row_breakdown"]
            parts = []
            for col in ["match_type", "day", "map_name"]:
                items = ", ".join(f"{g['group']} {g['zero_row_matches']}/{g['matches']}" for g in zb[col])
                parts.append(f"{col}: {items}")
            add("- Zero-row matches (zero/total) by group, top 5 each: " + " | ".join(parts) + ".")
        if r["schema_drift"]:
            add(f"- Schema drift across pages: {json.dumps(r['schema_drift'], default=str)}")
        else:
            add("- Schema drift across pages: none (single page or identical schemas).")
        add("")
        add("### Columns")
        add("")
        add("| column | arrow type | null % | distinct | min | max | median | top 5 (value x count, share) |")
        add("|---|---|---|---|---|---|---|---|")
        for c in r["columns"]:
            top = "" if not c["top5"] else "; ".join(
                f"{cell(t['value'])} x{t['count']} ({t['share']:.1%})" for t in c["top5"])
            add(f"| {c['name']} | {cell(c['arrow_type'])} | {c['null_rate'] * 100:.2f}% | {c['distinct_non_null']} | {cell(c['min'])} | {cell(c['max'])} | {cell(c['median'])} | {top} |")
        add("")
        f = r["flags"]
        add("### Flags")
        add("")
        add(f"- **Constant column (one value):** {len(f['constant'])}. " + (
            "e.g. " + "; ".join(f"`{x['col']}` = {x['value']}" for x in f["constant"][:2]) if f["constant"] else "none."))
        for x in f["constant"]:
            add(f"  - `{x['col']}` = {x['value']} (null rate {x['null_rate']:.2%})")
        add(f"- **More than 50% null:** {len(f['null_gt_50'])}. " + (
            "e.g. " + "; ".join(f"`{x['col']}` ({x['null_rate']:.1%})" for x in f["null_gt_50"][:2]) if f["null_gt_50"] else "none."))
        for x in f["null_gt_50"]:
            add(f"  - `{x['col']}` null {x['null_rate']:.2%}{' (all null)' if x['all_null'] else ''}")
        add(f"- **Sentinel-like values:** {len(f['sentinel'])} (col, value) hits. " + (
            "e.g. " + "; ".join(f"`{x['col']}`={x['value']} x{x['count']}" for x in f["sentinel"][:2]) if f["sentinel"] else "none."))
        for x in f["sentinel"]:
            add(f"  - `{x['col']}` = {x['value']}: {x['count']:,} rows ({x['share_of_rows']:.2%}); examples {ex_text(x['examples'])}")
        add(f"- **Spike (one value above 20% of rows):** {len(f['spike'])}. " + (
            "e.g. " + "; ".join(f"`{x['col']}`={x['value']} ({x['share_of_rows']:.1%})" for x in f["spike"][:2]) if f["spike"] else "none."))
        for x in f["spike"]:
            add(f"  - `{x['col']}` = {x['value']}: {x['count']:,} rows ({x['share_of_rows']:.2%}); examples {ex_text(x['examples'])}")
        add(f"- **Impossible for name:** {len(f['impossible'])} rule hits. " + (
            "; ".join(f"`{x['col']}` {x['rule']} {x['count']:,} rows" for x in f["impossible"]) if f["impossible"] else "none."))
        for x in f["impossible"]:
            add(f"  - `{x['col']}` {x['rule']}: {x['count']:,}; examples {ex_text(x['examples'])}")
        ident = r["identical_column_pairs"]
        add(f"- **Identical columns:** {len(ident)} pairs. " + (
            "e.g. " + "; ".join(f"`{p['a']}` = `{p['b']}`" for p in ident[:2]) if ident else "none."))
        for p in ident:
            add(f"  - `{p['a']}` == `{p['b']}` ({p['kind']}{', both constant' if p['both_constant'] else ''})")
        d = r["duplicates"]
        add(f"- **Duplicate rows within match and round (full-row):** {d['full_row_duplicate_rows']:,} rows ({(d['full_row_duplicate_share'] or 0):.4%}) in {d['match_round_groups_affected']:,} match-round groups across {d['matches_affected']} matches. Examples {ex_text(d['examples'])}")
        add(f"  - Keyed (match_key, round, tick, player_id, attacker_id, other_type where present): {d['keyed_rows_in_dup_groups_all_copies']:,} rows sit in duplicate key groups. Examples {ex_text(d['keyed_examples'])}")
        mo = r["match_row_counts"]
        add(f"- **Matches far from median row count:** median over all {mo['universe_matches']} matches = {mo['median_rows_all_matches_incl_zero']:,.0f}, zero-row matches = {mo['zero_row_matches']}.")
        for t in mo["top3_abs_deviation_all_matches"]:
            add(f"  - all-matches top: `{t['match_id']}` rows {t['rows']:,} (ratio {t['ratio_to_median']})")
        add(f"  - non-zero median = {mo['median_rows_nonzero_matches']:,.0f}")
        for t in mo["top3_abs_deviation_nonzero_matches"]:
            add(f"  - non-zero top: `{t['match_id']}` rows {t['rows']:,} (ratio {t['ratio_to_median']})")
        add("")
        ex = r["extra_checks"]
        add("### Other checks")
        add("")
        add(f"- Null counts: {ex['null_counts']}. Round range {ex.get('round_min')}..{ex.get('round_max')} ({ex.get('round_distinct')} distinct). Tick range {ex.get('tick_min')}..{ex.get('tick_max')}.")
        if "tick_decreases_in_stored_order" in ex:
            t = ex["tick_decreases_in_stored_order"]
            add(f"- Tick decreases inside a match (stored order): {t['rows']:,} rows in {t['matches']} matches. Examples {ex_text(t['examples'])}")
        if "second_vs_tick_over_tick_rate" in ex:
            s = ex["second_vs_tick_over_tick_rate"]
            add(f"- second vs tick/tick_rate: median residual {s['median_residual_s']:.3f}s; rows off by >1s: {s['rows_off_gt_1s']:,}; matches with median off by >1s: {s['matches_median_off_gt_1s']}. Examples {ex_text(s['examples'])}")
        neg = ex["negative_where_nonnegative_by_name"]
        add("- Negative in name-nonnegative columns: " + (
            "; ".join(f"`{k}` {v['count']:,}" for k, v in neg.items()) if neg else "none") + ".")
        add(f"- Range checks run: {ex['range_checks_run']}. Failing: " + (
            "; ".join(f"`{k}` {v['rule']} {v['count']:,}" for k, v in ex["range_checks"].items()) if ex["range_checks"] else "none") + ".")
        add(f"- Empty strings: " + (
            ", ".join(f"`{k}` {v:,}" for k, v in ex["empty_string_counts"].items()) if ex["empty_string_counts"] else "none") + ".")
        fx = [x for x in ex["fixed_vs_base_columns"]]
        add("- `_fixed` vs base: " + ("; ".join(f"`{x['fixed']}` differs from `{x['base']}` in {x['rows_differ']:,} rows (base null {x['base_null']:,}, fixed null {x['fixed_null']:,})" for x in fx) if fx else "no pairs") + ".")
        if "player_equals_attacker_rows" in ex:
            add(f"- player_id == attacker_id rows: {ex['player_equals_attacker_rows']:,}.")
        add(f"- Id zero counts: " + (", ".join(f"`{k}` {v:,}" for k, v in ex["id_zero_counts"].items()) if ex["id_zero_counts"] else "none") + ".")
        if "damage_dir_norm" in ex:
            dn = ex["damage_dir_norm"]
            add(f"- damage_dir norm: median {dn['median_norm']:.4f}; zero vectors {dn['zero_vectors']:,}; not unit (>0.01 off) {dn['not_unit_gt_0_01']:,}. Examples {ex_text(dn['examples'])}")
        if "blind_duration_gt_10s" in ex:
            b = ex["blind_duration_gt_10s"]
            add(f"- blind_duration > 10s: {b['count']:,}. Examples {ex_text(b['examples'])}")
        if "effective_health_removed_gt_health_removed" in ex:
            e = ex["effective_health_removed_gt_health_removed"]
            add(f"- effective_health_removed > health_removed: {e['count']:,}. Examples {ex_text(e['examples'])}")
        hh = ex.get("player_hurt_health")
        if hh:
            add(f"- health_removed vs HP lost: {hh['health_removed_gt_100_rows']:,} rows above 100 (share {hh['health_removed_gt_100_share']:.2%}), {hh['health_removed_gt_100_with_health_zero']:,} of them with health 0; max {hh['health_removed_max']}. Equals previous health minus current health on {hh['share_health_removed_equals_prev_minus_health']:.2%} of {hh['rows_with_prev_health']:,} rows with a previous row ({hh['share_health_removed_within_1_of_prev_minus_health']:.2%} within 1). effective_health_removed above 100 in {hh['effective_health_removed_gt_100_rows']:,} rows (max {hh['effective_health_removed_max']}). Examples {ex_text(hh['examples_health_removed_gt_100'])}")
        asx = ex.get("assister")
        if asx:
            add(f"- assister_id encodings: null {asx['null']:,}; 65535 {asx['value_65535']:,}; 0 (real slot) {asx['value_0_real_slot']:,}. Valid assister rows {asx['valid_assister_rows']:,}; of these {asx['valid_rows_with_null_position']:,} ({asx['share_valid_rows_with_null_position']:.1%}) have null assister position. Invalid rows with a position: {asx['invalid_rows_with_position']:,}.")
        idc = ex.get("ids")
        if idc:
            txt = f"- Raw player_id: matches with more than 10 distinct ids {idc['matches_with_more_than_10_raw_ids']}; rows with id outside 0-9 {idc['rows_with_raw_id_outside_0_9']:,}."
            if "raw_to_fixed_single_valued_share_within_match" in idc:
                txt += (f" Raw to fixed is one-to-one within match on {idc['raw_to_fixed_single_valued_share_within_match']:.2%} of pairs; "
                        f"raw 0 maps to {idc['raw0_distinct_fixed_values_across_matches']} distinct fixed values across matches; "
                        f"fixed disagrees with raw on {idc['fixed_disagrees_with_raw_rows']:,} rows.")
            add(txt)
        add("")
        add("### Checks run")
        add("")
        for line in r["checks_run"]:
            add(f"- {line}")
        add("")
    return "\n".join(L) + "\n"


def main():
    print("start", datetime.datetime.now().isoformat(timespec="seconds"), flush=True)
    universe_keys, rate_map, header_info = load_header()
    hdr_full = load_header_full()
    res = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "tome": TOME,
        "universe": {"matches": header_info["matches"],
                     "distinct_key": header_info["distinct_key"],
                     "tick_rate_counts": header_info["tick_rate_counts"]},
        "goofy_top5": GOOFY_TOP5,
        "followups": FOLLOWUPS,
        "channels": {},
    }
    for ch in CHANNELS:
        print("profiling", ch, flush=True)
        ld = read_channel(ch)
        df = ld["df"]
        n = len(df)
        cols = [c for c in df.columns if c != "_page"]
        kinds = {c: kind_of(df[c]) for c in cols}
        pages_rows_ok = sum(ld["page_rows"]) == n
        constant, null50, sentinel, spike = channel_flags(df, kinds, n)
        impossible = impossible_checks(df, kinds)
        pairs = identical_pairs(df, kinds)
        dups = duplicates(df, cols)
        outl = match_outliers(df, universe_keys)
        extra = extra_checks(df, rate_map)
        extra["pages_row_sum_ok"] = pages_rows_ok
        extra["player_hurt_health"] = hurt_checks(df) if "health_removed" in df.columns else None
        extra["assister"] = assister_checks(df)
        extra["ids"] = id_checks(df)
        zrb = coverage_breakdown(set(df["match_key"].unique()), hdr_full)
        per_match_pages = df.groupby("match_key")["_page"].nunique()
        multi_page = int((per_match_pages > 1).sum())
        cols_recs = profile_columns(df, kinds, ld["schemas"])
        distinct_mk = int(df["match_key"].nunique())
        r = {
            "channel": ch,
            "table": f"basic_{ch}.2026-09-18,2026-10-08",
            "pages": ld["pages"],
            "on_disk": ld["on_disk"],
            "page_rows": ld["page_rows"],
            "rows_sum_check": pages_rows_ok,
            "manifest_complete": ld["manifest_complete"],
            "rows_total": n,
            "ncols": len(cols),
            "distinct_match_key": distinct_mk,
            "universe_matches": len(universe_keys),
            "zero_row_matches": len(universe_keys) - distinct_mk,
            "zero_row_breakdown": zrb,
            "match_keys_not_in_header": outl["match_keys_not_in_header"],
            "matches_spanning_pages": multi_page,
            "schema_drift": schema_drift(ld["schemas"]),
            "columns": cols_recs,
            "flags": {"constant": constant, "null_gt_50": null50, "sentinel": sentinel,
                      "spike": spike, "impossible": impossible},
            "identical_column_pairs": pairs,
            "duplicates": dups,
            "match_row_counts": outl,
            "extra_checks": extra,
            "checks_run": [
                f"Read all {len(ld['pages'])} page(s) listed in the channel manifest with pyarrow; on-disk dataframe files: {ld['on_disk']}.",
                f"Matched match_key to header tome key (header has {len(universe_keys)} keys); zero-row matches = header keys with no channel row.",
                "Per-column dtype (arrow and pandas), null rate, distinct, min/max/median (numeric), top 5 (<=50 distinct).",
                "Constant, >50% null, sentinel hits (7 values), spike >20% of rows, impossible-for-name, identical columns, full-row and keyed duplicates, match row-count outliers vs median.",
                "Extra: tick order in stored order, second vs tick/tick_rate, angle/position/velocity ranges, empty strings, _fixed vs base disagreement, id zeros, damage_dir unit norm, blind duration, effective vs raw health removed, page-spanning matches, schema drift.",
            ],
        }
        res["channels"][ch] = r
        del ld, df
        print("done", ch, "rows", n, flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "combat.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False, default=str)
    with open(os.path.join(OUT_DIR, "combat.md"), "w", encoding="utf-8") as fh:
        fh.write(render_md(res))
    print("wrote", os.path.join(OUT_DIR, "combat.md"), flush=True)


if __name__ == "__main__":
    main()
