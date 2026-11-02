"""Build a small, deterministic "mini" copy of the combined tome store, for CI and mutation testing of the data steps.

The mini store has the SAME layout as tomes/combined, only for the 9 channels prepare.py touches, one page per channel holding only
the selected matches:

    tomes/mini/tome/csds/header/dataframe_00000, keyset_00000
    tomes/mini/tome/csds/basic_<channel>.COMBINED/dataframe_00000, keyset_00000
    tomes/mini/MINI_BUILD.json                       (selection hash, source ledger fingerprint, rows per channel)
    data/mini/MINI_MANIFEST.json                     (what the fixture must contain; facts after prepare + make_grid)

Selection (SELECTION FILE scripts/data/mini_selection.json, made once from the catalog data/<src>/{matches,rounds}.parquet and then
PINNED, so the fixture does not drift when the combined store grows; --reselect redoes it):
  n-in-scope   in-scope, non-duplicate matches: era balanced, builds covered, plus a greedy cover of rare features (every round_status
               and label_source, overtime per build, side-swap defects, first round / tail rounds missing, 2026 regime without
               round_announce_match_start, roster rule for a missing player, wingman flag disagreeing with the player count, ...)
  n-dup-groups BOTH members of size-2 duplicate groups, in-scope; first the groups whose exact event signature differs (only the
               tolerant key catches them), then groups with identical signatures
  n-out-of-pool / n-hostage / n-wingman   matches of those scope reasons, spread over eras and builds
Everything is a deterministic function of (catalog, --seed, counts): ranking uses sha256(seed|salt|match_key), never an RNG state.

  pixi run python -I -X utf8 scripts\\data\\make_mini_store.py                       build tomes/mini + manifest (skips if up to date)
  pixi run python -I -X utf8 scripts\\data\\make_mini_store.py --run-pipeline        ... then prepare.py + make_grid.py + --verify
  pixi run python -I -X utf8 scripts\\data\\make_mini_store.py --verify              check data/mini/* against the manifest (after prepare/grid)
  options: --force (rebuild the tome), --reselect (make a new selection), --update-golden (accept new prepare/grid output digests)
"""
import argparse
import collections
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("POLARS_MAX_THREADS", "2")
import polars as pl
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FORMAT = 1
# every channel prepare.py reads (check_prepare_channels() fails if prepare.py starts to read another one)
CHANNELS = ["header", "round_end", "round_start", "round_state", "player_info", "player_death", "player_disconnect",
            "bot_takeover", "bomb_state"]
DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state"]  # prepare.py: has_<ch> from keyset
SCOPES = ["in_scope", "map_out_of_pool", "hostage", "wingman"]
PAGE_RE = re.compile(r"^dataframe_(\d{5})$")
# targets of the greedy feature cover that need more than one match
COVER_NEED = {("short_tail", "era:2026"): 3}


# ----------------------------------------------------------------------------------------------------------------- helpers
def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def h64(seed, salt, key):
    return int(hashlib.sha256(f"{seed}|{salt}|{key}".encode("utf-8")).hexdigest()[:16], 16)


def rel(path):
    return os.path.relpath(path, ROOT).replace("\\", "/")


def folder_of(ch):
    return "header" if ch == "header" else f"basic_{ch}.COMBINED"


def key_col(ch):
    return "key" if ch == "header" else "match_key"


def collect(lf):
    try:
        return lf.collect(engine="streaming")
    except TypeError:  # older polars
        return lf.collect(streaming=True)


def apportion(total, weights, minimum=0):
    """Integers summing to `total`, proportional to `weights` (largest remainder), each at least `minimum` when possible."""
    keys = sorted(weights)
    if not keys:
        return {}
    if total < minimum * len(keys):
        minimum = 0
    out = {k: minimum for k in keys}
    rem = total - minimum * len(keys)
    wsum = float(sum(weights.values())) or 1.0
    raw = {k: rem * weights[k] / wsum for k in keys}
    for k in keys:
        out[k] += int(raw[k])
    left = total - sum(out.values())
    for k in sorted(keys, key=lambda k: (-(raw[k] - int(raw[k])), k))[:left]:
        out[k] += 1
    return out


def check_prepare_channels():
    """The mini store must hold every channel prepare.py touches; fail loudly if prepare.py was changed to read another one."""
    p = os.path.join(HERE, "prepare.py")
    txt = open(p, encoding="utf-8").read()
    used = set(re.findall(r'scan\("([a-z_]+)"', txt))
    m = re.search(r"^DENSE\s*=\s*\[(.*?)\]", txt, flags=re.M | re.S)
    if m:
        used |= set(re.findall(r'"([a-z_]+)"', m.group(1)))
    missing = used - set(CHANNELS)
    if missing:
        raise SystemExit(f"prepare.py reads channels {sorted(missing)} that make_mini_store.py does not copy: add them to CHANNELS")
    unused = set(CHANNELS) - used
    if unused:
        print(f"note: prepare.py no longer reads {sorted(unused)} (still copied)")


# ------------------------------------------------------------------------------------------------------------ selection
def load_catalog(cat_dir):
    mp, rp = os.path.join(cat_dir, "matches.parquet"), os.path.join(cat_dir, "rounds.parquet")
    for p in (mp, rp):
        if not os.path.isfile(p):
            raise SystemExit(f"{p} not found: run prepare.py on the source store first (or point --catalog at a copy)")
    M = pl.read_parquet(mp)
    R = pl.read_parquet(rp, columns=["match_key", "round", "round_status", "label_source", "missing_side", "rec_suspect",
                                     "final_differs_from_recorded", "max_rounds_final"])
    A = R.group_by("match_key").agg(
        pl.len().alias("nr"), pl.col("round").min().alias("rmin"),
        (pl.col("round") > pl.col("max_rounds_final")).any().alias("f_overtime"),
        pl.col("missing_side").is_not_null().any().alias("f_missing_side"),
        pl.col("rec_suspect").any().alias("f_rec_suspect"),
        pl.col("final_differs_from_recorded").fill_null(False).any().alias("f_final_differs"),
        pl.col("round_status").unique().alias("statuses"), pl.col("label_source").unique().alias("lsources"))
    M = M.join(A, on="match_key", how="left")
    info = {"matches_sha256": sha256_file(mp), "rounds_sha256": sha256_file(rp), "n_matches": M.height,
            "catalog_dir": rel(cat_dir) if os.path.abspath(cat_dir).startswith(ROOT) else "(outside project)"}
    return M, info


def tags_of(r):
    """Match-level features that exercise a branch of prepare.py / make_grid.py."""
    t = set()
    for s in (r["statuses"] or []):
        if s != "ok":
            t.add(f"status:{s}")
    for s in (r["lsources"] or []):
        t.add(f"label:{s}")
    for flag, name in (("f_overtime", "overtime"), ("f_missing_side", "missing_player_rule"), ("f_rec_suspect", "rec_suspect_round"),
                       ("f_final_differs", "final_differs_from_recorded")):
        if r[flag]:
            t.add(name)
    exp = (r["t_starters_score_final"] or 0) + (r["ct_starters_score_final"] or 0)
    nr = r["nr"] or 0
    if exp - nr >= 2:
        t.add("short_tail")            # the recording stops before the last rounds: round numbers missing at the end
    if (r["rmin"] or 1) > 1:
        t.add("first_round_missing")   # round numbering starts at 2 (all 2024 build 9842 matches)
    if (r["n_rounds_re"] or 0) < 16:
        t.add("fragment")              # < 16 round_end rows: duplicate key is the exact signature only
    if r["wingman_flag_disagrees"]:
        t.add("wingman_flag_disagrees")
    if not r["regime_a"]:
        t.add("no_round_announce_match_start")
    if (r["n_ok_rounds"] or 0) == 0:
        t.add("no_ok_rounds")
    if r["max_rounds"] is None:
        t.add("max_rounds_null")
    if r["is_wingman"] is None:
        t.add("is_wingman_null")
    return t


def make_selection(args, cat_dir):
    M, cat_info = load_catalog(cat_dir)
    seed = args.seed
    rows = {r["match_key"]: r for r in M.iter_rows(named=True)}
    for r in rows.values():
        r["tags"] = tags_of(r)
    H = lambda salt, k: h64(seed, salt, k)  # noqa: E731
    base = [r for r in rows.values() if r["channels_complete"] and (r["n_rounds_re"] or 0) >= 1]
    taken = set()
    cats = {"in_scope": [], "dup_groups": [], "out_of_pool": [], "hostage": [], "wingman": []}
    cover_of = {}

    # ---- duplicate groups: size 2 in the full catalog, both members in scope. Exact signature differs -> only the tolerant key
    groups = collections.defaultdict(list)
    for r in base:
        if r["dup_group_size"] and r["dup_group_size"] == 2 and r["scope_reason"] == "in_scope":
            groups[r["dup_key"]].append(r)
    groups = {k: sorted(v, key=lambda r: r["match_key"]) for k, v in groups.items() if len(v) == 2}
    gkeys = [k for k, v in groups.items() if max(r["n_ok_rounds"] or 0 for r in v) >= 8]
    differ = sorted([k for k in gkeys if groups[k][0]["sig_hash"] != groups[k][1]["sig_hash"]], key=lambda k: H("dup", groups[k][0]["match_key"]))
    same = [k for k in gkeys if k not in set(differ)]
    slot_cycle = [9961, 10896, 10924, 9842]
    chosen_groups = []
    for i in range(args.n_dup_groups):
        pick = None
        if i < 2 or not same:
            pick = next((k for k in differ if k not in chosen_groups), None)
        if pick is None:
            b = slot_cycle[(i - 2) % len(slot_cycle)] if i >= 2 else slot_cycle[0]
            cand = [k for k in same if k not in chosen_groups and groups[k][0]["build_num"] == b]
            if i == 3:  # one group whose members differ in their number of eligible rounds (keeper is not just the lowest key)
                uneven = [k for k in cand if groups[k][0]["n_ok_rounds"] != groups[k][1]["n_ok_rounds"]]
                cand = uneven or cand
            cand = cand or [k for k in same if k not in chosen_groups] or [k for k in differ if k not in chosen_groups]
            pick = min(cand, key=lambda k: H("dup", groups[k][0]["match_key"])) if cand else None
        if pick is None:
            raise SystemExit("not enough duplicate groups of size 2 in the catalog")
        chosen_groups.append(pick)
        for r in groups[pick]:
            taken.add(r["match_key"])
    cats["dup_groups"] = [{"keys": [r["match_key"] for r in groups[k]],
                           "exact_signature_differs": groups[k][0]["sig_hash"] != groups[k][1]["sig_hash"]} for k in chosen_groups]

    # ---- out-of-pool / hostage / wingman: one per (build) slot, distinct maps where possible
    def pick_scope(scope, n, slots, prefer=None):
        pool = [r for r in base if r["scope_reason"] == scope and r["dup_group_size"] == 1]
        good = [r for r in pool if (r["n_rounds_re"] or 0) >= 10]
        pool = good or pool
        out, maps = [], set()
        for i in range(n):
            b = slots[i % len(slots)]
            cand = [r for r in pool if r["match_key"] not in taken and r["build_num"] == b] or \
                   [r for r in pool if r["match_key"] not in taken]
            if prefer:
                cand = [r for r in cand if prefer(r)] or cand
            fresh = [r for r in cand if r["map_name"] not in maps]
            cand = fresh or cand
            if not cand:
                raise SystemExit(f"catalog has no more {scope} matches")
            r = min(cand, key=lambda r: H(scope, r["match_key"]))
            out.append(r["match_key"])
            taken.add(r["match_key"])
            maps.add(r["map_name"])
        return out

    cats["out_of_pool"] = pick_scope("map_out_of_pool", args.n_out_of_pool, [9842, 9961, 10896, 10924])
    cats["hostage"] = pick_scope("hostage", args.n_hostage, [9961, 10896, 9842, 10924])
    cats["wingman"] = pick_scope("wingman", args.n_wingman, [9842, 10896, 10924], prefer=lambda r: r["is_wingman"] is not False)

    # ---- in-scope, non-duplicate: (1) greedy cover of rare features, (2) era-balanced, build-quota fill spread over upload days
    pool = [r for r in base if r["scope_reason"] == "in_scope" and r["dup_group_size"] == 1 and r["match_key"] not in taken]
    pool_by_key = {r["match_key"]: r for r in pool}

    def targets(r):
        ks = [(t, f"era:{r['era']}") for t in r["tags"]]
        if "overtime" in r["tags"]:
            ks.append(("overtime", f"build:{r['build_num']}"))
        return ks

    cands = collections.defaultdict(list)
    for r in pool:
        for t in targets(r):
            cands[t].append(r["match_key"])
    need = {t: min(COVER_NEED.get(t, 1), len(v)) for t, v in cands.items()}
    have = collections.Counter()
    picks = []
    for t in sorted(cands, key=lambda t: (len(cands[t]), t)):
        while have[t] < need[t]:
            free = [k for k in cands[t] if k not in picks]
            if not free:
                break
            def gain(k):
                return sum(1 for u in targets(pool_by_key[k]) if have[u] < need.get(u, 0))
            best = min(free, key=lambda k: (-gain(k), H("cover", k)))
            picks.append(best)
            for u in targets(pool_by_key[best]):
                have[u] += 1
    if len(picks) > args.n_in_scope * 0.6:
        raise SystemExit(f"feature cover needs {len(picks)} matches, too many for --n-in-scope {args.n_in_scope}")
    n_cover = len(picks)

    eras = sorted({r["era"] for r in pool})
    builds_by_era = {e: sorted({r["build_num"] for r in pool if r["era"] == e}) for e in eras}
    size = collections.Counter(r["build_num"] for r in pool)
    era_q = apportion(args.n_in_scope, {e: 1.0 for e in eras})
    quota = {}
    for e in eras:
        quota.update(apportion(era_q[e], {b: math.sqrt(size[b]) for b in builds_by_era[e]}, minimum=4))
    got = collections.Counter(pool_by_key[k]["build_num"] for k in picks)
    deficit = {b: max(0, quota[b] - got[b]) for b in quota}
    remaining = args.n_in_scope - len(picks)
    if sum(deficit.values()) >= remaining:
        fill_q = apportion(remaining, {b: float(d) for b, d in deficit.items()})
    else:
        extra = apportion(remaining - sum(deficit.values()), {b: float(quota[b]) for b in quota})
        fill_q = {b: deficit[b] + extra[b] for b in quota}
    for b in sorted(fill_q):
        c = [r for r in pool if r["build_num"] == b and r["match_key"] not in picks]
        typical = [r for r in c if (r["n_ok_rounds"] or 0) >= 8]
        c = typical if len(typical) >= fill_q[b] else c
        c.sort(key=lambda r: H("fill", r["match_key"]))
        seen = collections.Counter()
        ranked = []
        for r in c:  # spread over upload days: the n-th match of every day before the (n+1)-th of any day
            ranked.append((seen[r["upload_day"]], H("fill", r["match_key"]), r["match_key"]))
            seen[r["upload_day"]] += 1
        ranked.sort()
        picks.extend(k for _, _, k in ranked[:fill_q[b]])
    if len(picks) != args.n_in_scope:
        raise SystemExit(f"selected {len(picks)} in-scope matches, wanted {args.n_in_scope}")
    cats["in_scope"] = sorted(picks)
    cover_of = {k: sorted(t for t in rows[k]["tags"]) for k in picks[:n_cover]}

    keys = sorted(set(cats["in_scope"]) | set(cats["out_of_pool"]) | set(cats["hostage"]) | set(cats["wingman"])
                  | {k for g in cats["dup_groups"] for k in g["keys"]})
    cat_of = {}
    for c in ("in_scope", "out_of_pool", "hostage", "wingman"):
        for k in cats[c]:
            cat_of[k] = c
    for gi, g in enumerate(cats["dup_groups"]):
        for k in g["keys"]:
            cat_of[k] = f"dup_group_{gi}"
    detail = {}
    for k in keys:
        r = rows[k]
        detail[k] = {"category": cat_of[k], "scope_reason": r["scope_reason"], "era": r["era"], "build_num": r["build_num"],
                     "map_name": r["map_name"], "upload_day": r["upload_day"], "n_rounds_re": r["n_rounds_re"],
                     "n_ok_rounds_in_catalog": r["n_ok_rounds"], "tags": sorted(r["tags"]), "from_feature_cover": k in cover_of}
    cats["in_scope"] = sorted(cats["in_scope"])
    return {"format": FORMAT,
            "params": params_of(args),
            "selected_from": cat_info,
            "selection_sha256": sha256_bytes("\n".join(keys).encode("utf-8")),
            "n_selected": len(keys),
            "n_feature_cover_matches": n_cover,
            "build_quota": {str(b): quota[b] for b in sorted(quota)},
            "categories": cats,
            "match_keys": keys,
            "detail": detail}


def params_of(a):
    return {"src": a.src, "seed": a.seed, "n_in_scope": a.n_in_scope, "n_dup_groups": a.n_dup_groups,
            "n_out_of_pool": a.n_out_of_pool, "n_hostage": a.n_hostage, "n_wingman": a.n_wingman}


def load_or_make_selection(args):
    path = os.path.abspath(args.selection)
    if os.path.isfile(path) and not args.reselect:
        sel = json.load(open(path, encoding="utf-8"))
        if sel.get("params") != params_of(args):
            raise SystemExit(f"{rel(path)} was made with {sel.get('params')} but the arguments are {params_of(args)}; "
                             f"pass --reselect to make a new selection (it overwrites the file)")
        keys = sel["match_keys"]
        if sha256_bytes("\n".join(sorted(keys)).encode("utf-8")) != sel["selection_sha256"]:
            raise SystemExit(f"{rel(path)}: selection_sha256 does not match its match_keys (file edited by hand?)")
        print(f"selection: pinned file {rel(path)}  {len(keys)} matches  sha256 {sel['selection_sha256'][:16]}")
        return sel
    cat_dir = os.path.abspath(args.catalog or os.path.join(ROOT, "data", args.src))
    t0 = time.time()
    sel = make_selection(args, cat_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sel, f, indent=1)
    c = sel["categories"]
    print(f"selection: made from {sel['selected_from']['catalog_dir']} ({sel['selected_from']['n_matches']} matches, "
          f"{time.time() - t0:.0f}s) -> {rel(path)}  {sel['n_selected']} matches: in_scope {len(c['in_scope'])} "
          f"(feature cover {sel['n_feature_cover_matches']}), dup groups {len(c['dup_groups'])}, out_of_pool {len(c['out_of_pool'])}, "
          f"hostage {len(c['hostage'])}, wingman {len(c['wingman'])}")
    return sel


# ---------------------------------------------------------------------------------------------------------------- tome
def read_keyset(path):
    for attempt in range(4):  # an append may be rewriting it
        try:
            return set(pl.read_parquet(path)["_"].to_list())
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


def build_tome(args, sel, src_root, ledger_raw, ledger):
    keys = sel["match_keys"]
    keyset_sel = set(keys)
    out_root = os.path.join(ROOT, "tomes", args.out)
    tmp = out_root + ".building"
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    base = os.path.join(tmp, "tome", "csds")
    os.makedirs(base)
    stats, contributing, page_rows, files_sha = {}, {}, {}, {}
    t0 = time.time()
    print(f"{'channel':<18} {'pages':>5} {'src rows scanned':>17} {'rows kept':>10} {'matches':>8} {'cols':>5}  s", flush=True)
    for ch in CHANNELS:
        t1 = time.time()
        folder = os.path.join(src_root, folder_of(ch))
        kc = key_col(ch)
        led = ledger["channels"].get(ch)
        if led is None:
            raise SystemExit(f"source ledger has no channel {ch}")
        n_pages = led["pages"]
        schema0, pieces, scanned = None, [], 0
        contributing[ch], page_rows[ch] = [], {}
        for i in range(n_pages):  # only the pages the ledger knows: a concurrent append adds higher numbers and *.tmp files
            f = os.path.join(folder, f"dataframe_{i:05d}")
            if not os.path.isfile(f):
                raise SystemExit(f"{f} listed by the source ledger ({n_pages} pages) but missing")
            sch = pq.read_schema(f)
            if schema0 is None:
                schema0 = sch
            elif not sch.equals(schema0, check_metadata=False):
                raise SystemExit(f"{ch}: page {i} has a different schema than page 0 (schema drift in the source store)")
            n_rows = pq.ParquetFile(f).metadata.num_rows
            scanned += n_rows
            df = collect(pl.scan_parquet(f).filter(pl.col(kc).is_in(keys)))
            if df.height:
                pieces.append(df)
                contributing[ch].append(i)
                page_rows[ch][str(i)] = n_rows
        out = pl.concat(pieces, how="vertical") if pieces else pl.DataFrame(schema=pl.scan_parquet(
            os.path.join(folder, "dataframe_00000")).collect_schema())
        os.makedirs(os.path.join(base, folder_of(ch)))
        page = os.path.join(base, folder_of(ch), "dataframe_00000")
        out.write_parquet(page, compression="zstd")
        sch_out = pq.read_schema(page)
        if sch_out.names != schema0.names or [str(t) for t in sch_out.types] != [str(t) for t in schema0.types]:
            raise SystemExit(f"{ch}: the written page does not have the source schema")
        have = set(out[kc].unique().to_list())
        ks = read_keyset(os.path.join(folder, "keyset_00000")) & keyset_sel
        if ch in DENSE and ks != keyset_sel:
            raise SystemExit(f"{ch}: {len(keyset_sel - ks)} selected matches are not in the source keyset (channels incomplete)")
        if ch in DENSE and have != keyset_sel:
            print(f"  note: {ch}: {len(keyset_sel - have)} selected matches have no rows (key present, empty)")
        if ch == "header" and out.height != len(keys):
            raise SystemExit(f"header: {out.height} rows for {len(keys)} selected matches")
        pl.DataFrame({"_": sorted(ks)}, schema={"_": pl.String}).write_parquet(
            os.path.join(base, folder_of(ch), "keyset_00000"), compression="zstd")
        stats[ch] = {"pages_scanned": n_pages, "source_rows_scanned": scanned, "rows": out.height, "matches_with_rows": len(have),
                     "keys_in_keyset": len(ks), "columns": len(schema0.names)}
        files_sha[ch] = sha256_file(page)
        print(f"{ch:<18} {n_pages:>5} {scanned:>17} {out.height:>10} {len(have):>8} {len(schema0.names):>5}  {time.time() - t1:.0f}", flush=True)
    pages_fp = sha256_bytes(canon_json({"contributing_pages_rows": page_rows}).encode("utf-8"))
    build = {"format": FORMAT, "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "seconds": round(time.time() - t0, 1),
             "src": args.src, "selection_sha256": sel["selection_sha256"], "params": sel["params"], "channels": stats,
             "source_ledger_sha256": sha256_bytes(ledger_raw), "source_ledger_built": ledger.get("built"),
             "source_ledger_n_sources": len(ledger.get("sources", [])),
             "source_channel_pages": {ch: ledger["channels"][ch]["pages"] for ch in CHANNELS},
             "source_header_keys": ledger["channels"]["header"]["keys"],
             "contributing_pages": contributing, "source_pages_fingerprint": pages_fp, "tome_files_sha256": files_sha}
    with open(os.path.join(tmp, "MINI_BUILD.json"), "w", encoding="utf-8") as f:
        json.dump(build, f, indent=1)
    old = out_root + ".replaced"
    if os.path.exists(out_root):
        if not os.path.exists(os.path.join(out_root, "MINI_BUILD.json")):
            raise SystemExit(f"{out_root} exists and is not a mini store; refusing to replace it")
        if os.path.exists(old):
            shutil.rmtree(old)
        os.replace(out_root, old)
    os.replace(tmp, out_root)
    shutil.rmtree(old, ignore_errors=True)
    print(f"tome built in {time.time() - t0:.0f}s -> {out_root}")
    return build


# -------------------------------------------------------------------------------------------------------------- manifest
def write_manifest(args, sel, build, manifest_path):
    detail = sel["detail"]
    c = sel["categories"]
    by_build, by_era, features = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for k, d in detail.items():
        by_build[str(d["build_num"])].append(k)
        by_era[d["era"]].append(k)
        for t in d["tags"]:
            features[t].append(k)
    dup_members = [k for g in c["dup_groups"] for k in g["keys"]]
    expect_scope = collections.Counter(d["scope_reason"] for d in detail.values())
    prev = {}
    if os.path.isfile(manifest_path):
        try:
            prev = json.load(open(manifest_path, encoding="utf-8"))
        except Exception:
            prev = {}
    m = {
        "format": FORMAT,
        "what": "expected content of the mini fixture (tomes/mini -> data/mini). Rebuild: scripts/data/README_MINI.txt",
        "params": sel["params"],
        "selection_sha256": sel["selection_sha256"],
        "selection_file": "scripts/data/mini_selection.json",
        "selected_from_catalog": sel["selected_from"],
        "counts": {"total": sel["n_selected"], "in_scope_nondup": len(c["in_scope"]), "dup_groups": len(c["dup_groups"]),
                   "dup_members": len(dup_members), "dup_groups_exact_signature_differs": sum(g["exact_signature_differs"] for g in c["dup_groups"]),
                   "out_of_pool": len(c["out_of_pool"]), "hostage": len(c["hostage"]), "wingman": len(c["wingman"]),
                   "expected_scope_reason": dict(sorted(expect_scope.items())),
                   "by_era": {e: len(v) for e, v in sorted(by_era.items())}, "by_build": {b: len(v) for b, v in sorted(by_build.items())}},
        "categories": {"in_scope": c["in_scope"], "dup_groups": c["dup_groups"], "out_of_pool": c["out_of_pool"],
                       "hostage": c["hostage"], "wingman": c["wingman"]},
        "by_build": {b: sorted(v) for b, v in sorted(by_build.items())},
        "by_era": {e: sorted(v) for e, v in sorted(by_era.items())},
        "features": {t: sorted(v) for t, v in sorted(features.items())},
        "must_have": {
            "build_9961": sorted(by_build.get("9961", [])), "build_9842": sorted(by_build.get("9842", [])),
            "era_2026": sorted(by_era.get("2026", [])),
            "missing_rounds_2026_tail": sorted(k for k in features.get("short_tail", []) if detail[k]["era"] == "2026"),
            "missing_first_round_2024": sorted(k for k in features.get("first_round_missing", []) if detail[k]["era"] == "2024"),
            "dup_groups_exact_signature_differs": [g["keys"] for g in c["dup_groups"] if g["exact_signature_differs"]]},
        "source_store": {"name": args.src, "ledger_sha256_at_build": build["source_ledger_sha256"],
                         "ledger_built": build["source_ledger_built"], "ledger_n_sources_at_build": build["source_ledger_n_sources"],
                         "header_keys_at_build": build["source_header_keys"], "channel_pages_at_build": build["source_channel_pages"],
                         "contributing_pages": build["contributing_pages"],
                         "contributing_pages_fingerprint_sha256": build["source_pages_fingerprint"],
                         "note": "pages are append-only: contributing_pages_fingerprint_sha256 (row counts of the pages the selected matches "
                                 "were read from) stays valid when the source store grows; the full ledger hash does not"},
        "tome": {"path": "tomes/" + args.out + "/tome/csds", "channels": build["channels"],
                 "files_sha256": build["tome_files_sha256"],
                 "files_sha256_note": "parquet bytes depend on the polars version; informational"},
    }
    if prev.get("selection_sha256") == m["selection_sha256"] and "after_prepare" in prev:
        m["after_prepare"] = prev["after_prepare"]
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=1)
    return m


# ---------------------------------------------------------------------------------------------------------------- verify
def digest(df, sort_cols, cols=None):
    d = df.select(cols) if cols else df
    d = d.sort(sort_cols)
    return sha256_bytes(d.write_csv().encode("utf-8"))


def verify(args, sel, manifest_path):
    """Check data/<out>/* against the selection and record facts + golden digests of the prepare/make_grid output in the manifest."""
    out_dir = os.path.join(ROOT, "data", args.out)
    fail, warn = [], []
    need = ["matches.parquet", "rounds.parquet", "deaths.parquet", "grid.parquet", "prep_report.json"]
    for n in need:
        if not os.path.isfile(os.path.join(out_dir, n)):
            fail.append(f"{rel(os.path.join(out_dir, n))} is missing (run prepare.py / make_grid.py --store {args.out})")
    if fail:
        raise SystemExit("verify FAILED:\n  " + "\n  ".join(fail))
    M = pl.read_parquet(os.path.join(out_dir, "matches.parquet"))
    R = pl.read_parquet(os.path.join(out_dir, "rounds.parquet"))
    DE = pl.read_parquet(os.path.join(out_dir, "deaths.parquet"))
    G = pl.read_parquet(os.path.join(out_dir, "grid.parquet"))
    rep = json.load(open(os.path.join(out_dir, "prep_report.json"), encoding="utf-8"))
    keys, c, detail = sel["match_keys"], sel["categories"], sel["detail"]
    mk = {r["match_key"]: r for r in M.iter_rows(named=True)}

    if set(mk) != set(keys) or M.height != len(keys):
        fail.append(f"matches.parquet has {M.height} matches, selection has {len(keys)}; symmetric difference {len(set(mk) ^ set(keys))}")
    for k in keys:
        r = mk.get(k)
        if r is None:
            continue
        d = detail[k]
        if r["scope_reason"] != d["scope_reason"]:
            fail.append(f"{k}: scope_reason {r['scope_reason']} != expected {d['scope_reason']}")
        if not r["channels_complete"]:
            fail.append(f"{k}: channels_complete is False")
        if r["era"] != d["era"] or r["build_num"] != d["build_num"]:
            fail.append(f"{k}: era/build {r['era']}/{r['build_num']} != {d['era']}/{d['build_num']}")
    for k in c["out_of_pool"]:
        if k in mk and (mk[k]["scope_reason"] != "map_out_of_pool" or mk[k]["in_scope"]):
            fail.append(f"{k}: out-of-pool match is not flagged map_out_of_pool / out of scope")
    for k in c["hostage"]:
        if k in mk and (not mk[k]["is_hostage_map"] or mk[k]["scope_reason"] != "hostage"):
            fail.append(f"{k}: hostage match is not flagged")
    for k in c["wingman"]:
        if k in mk and (not mk[k]["is_wingman_final"] or mk[k]["scope_reason"] != "wingman"):
            fail.append(f"{k}: wingman match is not flagged")
    dup_facts = []
    for gi, g in enumerate(c["dup_groups"]):
        rs = [mk[k] for k in g["keys"] if k in mk]
        if len(rs) != 2:
            continue
        if rs[0]["dup_key"] != rs[1]["dup_key"] or any(r["dup_group_size"] != 2 for r in rs):
            fail.append(f"dup group {gi} {g['keys']}: not detected as a group of 2 (dup_group_size {[r['dup_group_size'] for r in rs]})")
        if sum(bool(r["dup_keeper"]) for r in rs) != 1:
            fail.append(f"dup group {gi}: expected exactly one keeper, got {[r['dup_keeper'] for r in rs]}")
        differs = rs[0]["sig_hash"] != rs[1]["sig_hash"]
        if differs != g["exact_signature_differs"]:
            fail.append(f"dup group {gi}: exact signature differs={differs}, selection says {g['exact_signature_differs']}")
        dup_facts.append({"keys": g["keys"], "exact_signature_differs": differs,
                          "keeper": next((r["match_key"] for r in rs if r["dup_keeper"]), None),
                          "n_ok_rounds": [mk[k]["n_ok_rounds"] for k in g["keys"]]})
    nondup = set(keys) - {k for g in c["dup_groups"] for k in g["keys"]}
    for k in nondup:
        if k in mk and (mk[k]["dup_group_size"] != 1 or not mk[k]["dup_keeper"]):
            fail.append(f"{k}: expected a unique match (dup_group_size 1, keeper), got size {mk[k]['dup_group_size']}")
    if rep.get("dup_groups") != len(c["dup_groups"]):
        fail.append(f"prep_report dup_groups {rep.get('dup_groups')} != {len(c['dup_groups'])}")
    if rep.get("dup_groups_with_different_exact_signature") != sum(g["exact_signature_differs"] for g in c["dup_groups"]):
        fail.append(f"prep_report dup_groups_with_different_exact_signature {rep.get('dup_groups_with_different_exact_signature')} "
                    f"!= {sum(g['exact_signature_differs'] for g in c['dup_groups'])}")
    if rep.get("dup_matches_removed") != len(c["dup_groups"]):
        fail.append(f"prep_report dup_matches_removed {rep.get('dup_matches_removed')} != {len(c['dup_groups'])}")
    builds = set(M["build_num"].to_list())
    for b in (9961, 9842):
        if b not in builds:
            fail.append(f"no build {b} match")
    if not (M["era"] == "2026").any():
        fail.append("no 2026 match")
    # a match with missing rounds: final scores promise more rounds than round_end has
    short = M.join(R.group_by("match_key").agg(pl.len().alias("nr"), pl.col("round").min().alias("rmin")), on="match_key", how="left").filter(
        (pl.col("t_starters_score_final") + pl.col("ct_starters_score_final") - pl.col("nr") >= 2) & (pl.col("era") == "2026"))
    if short.height < 1:
        fail.append("no 2026 match with missing rounds")
    # same categories as the big store
    big = os.path.join(ROOT, "data", args.src, "prep_report.json")
    if os.path.isfile(big):
        brep = json.load(open(big, encoding="utf-8"))
        big_scopes = {k.split("|")[1] for k in brep.get("scope_by_era", {})}
        mini_scopes = {k.split("|")[1] for k in rep.get("scope_by_era", {})}
        if big_scopes != mini_scopes:
            fail.append(f"scope_reason categories differ from {args.src}: only in {args.src} {sorted(big_scopes - mini_scopes)}, "
                        f"only in mini {sorted(mini_scopes - big_scopes)}")
        big_status = {k.split("|")[1] for k in brep.get("round_status", {})}
        mini_status = {k.split("|")[1] for k in rep.get("round_status", {})}
        if big_status - mini_status:
            warn.append(f"round_status values of {args.src} missing in mini: {sorted(big_status - mini_status)} (consider --reselect)")
        if mini_status - big_status:
            warn.append(f"round_status values only in mini: {sorted(mini_status - big_status)}")
        big_ls = {k.split("|")[1] for k in brep.get("label_source_ok_rounds", {})}
        mini_ls = {k.split("|")[1] for k in rep.get("label_source_ok_rounds", {})}
        if big_ls - mini_ls:
            warn.append(f"label_source values of {args.src} (ok rounds) missing in mini: {sorted(big_ls - mini_ls)}")
    use = R.filter(pl.col("use"))
    for era in ("2024", "2026"):
        if use.filter(pl.col("era") == era).height == 0:
            fail.append(f"no used rounds in era {era}")
    if G.height == 0:
        fail.append("grid.parquet is empty")
    if set(G["match_key"].unique().to_list()) != set(use["match_key"].unique().to_list()):
        fail.append("grid matches != matches with used rounds")

    per_match = {}
    use_by = {k: n for k, n in use.group_by("match_key").len().iter_rows()}
    for k in sorted(keys):
        r = mk.get(k)
        if r:
            per_match[k] = {"scope_reason": r["scope_reason"], "dup_group_size": int(r["dup_group_size"]), "dup_keeper": bool(r["dup_keeper"]),
                            "n_round_end_rows": int(r["n_rounds_re"] or 0), "n_ok_rounds": int(r["n_ok_rounds"]), "use_rounds": int(use_by.get(k, 0))}
    facts = {
        "matches": M.height, "rounds": R.height, "deaths": DE.height, "grid_rows": G.height, "use_rounds": use.height,
        "use_matches": use["match_key"].n_unique(),
        "scope_reason": {s: int(n) for s, n in M.group_by("scope_reason").len().sort("scope_reason").iter_rows()},
        "scope_by_era": rep["scope_by_era"], "round_status": rep["round_status"], "label_source_ok_rounds": rep["label_source_ok_rounds"],
        "dup_groups": dup_facts, "prep_report_dup_groups": rep["dup_groups"],
        "prep_report_dup_groups_with_different_exact_signature": rep["dup_groups_with_different_exact_signature"],
        "prep_report_dup_matches_removed": rep["dup_matches_removed"], "wingman_flag_disagrees": rep["wingman_flag_disagrees"],
        "final_differs_from_recorded_ok_rounds": rep["final_differs_from_recorded_ok_rounds"], "use_by_era": rep["use_by_era"],
        "grid_matches": G["match_key"].n_unique(), "grid_by_era": {e: int(n) for e, n in G.group_by("era").len().sort("era").iter_rows()},
        "per_match": per_match}
    golden = {
        "matches": digest(M, ["match_key"], ["match_key", "scope_reason", "dup_group_size", "dup_keeper", "n_ok_rounds", "n_rounds_re", "era",
                                              "build_num", "map_name", "is_wingman_final", "is_hostage_map", "channels_complete",
                                              "max_rounds_final", "modal_pc", "regime_a", "in_scope"]),
        "rounds": digest(R, ["match_key", "round"], ["match_key", "round", "end", "rec", "freeze", "t_sc", "ct_sc", "n_players", "n_t", "n_ct",
                                                     "n_deaths", "deaths_t", "deaths_ct", "deaths_unresolved", "plant_tick", "defuse_tick",
                                                     "explode_tick", "bomb", "score", "elim", "final", "label_source", "end_eff",
                                                     "round_status", "use", "dup_keeper", "rec_suspect", "swap_boundary", "after_swap"]),
        "deaths": digest(DE, ["match_key", "round", "tick", "player_id"]),
        "grid": digest(G, ["match_key", "round", "second"])}
    mpath = manifest_path
    man = json.load(open(mpath, encoding="utf-8"))
    old_golden = (man.get("after_prepare") or {}).get("golden_sha256")
    if old_golden and not args.update_golden:
        for name, dg in golden.items():
            if old_golden.get(name) != dg:
                fail.append(f"golden digest of {name} changed (prepare.py/make_grid.py output differs from the recorded fixture "
                            f"output); if the change is intended rerun with --update-golden")
    if not fail or args.update_golden:
        man["after_prepare"] = {"facts": facts, "golden_sha256": golden,
                                "golden_note": "sha256 of the sorted CSV of the listed columns of data/mini/*.parquet; a mutation of "
                                               "prepare.py/make_grid.py that changes the output changes a digest"}
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(man, f, indent=1)
    for w in warn:
        print("WARN:", w)
    if fail:
        raise SystemExit("verify FAILED:\n  " + "\n  ".join(fail[:40]) + (f"\n  ... {len(fail) - 40} more" if len(fail) > 40 else ""))
    print(f"verify OK: {M.height} matches, {R.height} rounds, {use.height} used rounds in {use['match_key'].n_unique()} matches, "
          f"{G.height} grid rows; scope {facts['scope_reason']}; dup groups {len(dup_facts)} "
          f"({sum(d['exact_signature_differs'] for d in dup_facts)} with different exact signature)")
    return facts


def run_pipeline(args):
    env = dict(os.environ, MKL_THREADING_LAYER="SEQUENTIAL", POLARS_MAX_THREADS="2")
    for script in ("prepare.py", "make_grid.py"):  # one Python process at a time
        t0 = time.time()
        cmd = [sys.executable, "-I", "-X", "utf8", os.path.join(HERE, script), "--store", args.out]
        print("run:", " ".join(cmd), flush=True)
        r = subprocess.run(cmd, env=env, cwd=ROOT)
        print(f"{script}: exit {r.returncode}  {time.time() - t0:.0f}s", flush=True)
        if r.returncode:
            raise SystemExit(f"{script} failed")


# ------------------------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description="Build the mini tome store (see the module docstring).")
    ap.add_argument("--src", default="combined")
    ap.add_argument("--out", default="mini")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-in-scope", type=int, default=70)
    ap.add_argument("--n-dup-groups", type=int, default=4)
    ap.add_argument("--n-out-of-pool", type=int, default=4)
    ap.add_argument("--n-hostage", type=int, default=2)
    ap.add_argument("--n-wingman", type=int, default=2)
    ap.add_argument("--selection", default=os.path.join(HERE, "mini_selection.json"), help="pinned selection file")
    ap.add_argument("--catalog", default=None, help="folder with matches.parquet + rounds.parquet used to (re)select; default data/<src>")
    ap.add_argument("--reselect", action="store_true", help="make a new selection from the catalog and overwrite the pinned file")
    ap.add_argument("--force", action="store_true", help="rebuild the tome even if it is up to date")
    ap.add_argument("--verify", action="store_true", help="check data/<out>/* (after prepare.py + make_grid.py) against the selection")
    ap.add_argument("--run-pipeline", action="store_true", help="after the build run prepare.py and make_grid.py on the mini store, then --verify")
    ap.add_argument("--update-golden", action="store_true", help="accept the current prepare/make_grid output as the golden digests")
    args = ap.parse_args()
    t0 = time.time()
    check_prepare_channels()
    sel = load_or_make_selection(args)
    src_root = os.path.join(ROOT, "tomes", args.src, "tome", "csds")
    ledger_path = os.path.join(ROOT, "tomes", args.src, "COMBINED_BUILD.json")
    ledger_raw = open(ledger_path, "rb").read()  # read ONCE: its page counts bound which pages are scanned
    ledger = json.loads(ledger_raw)
    out_root = os.path.join(ROOT, "tomes", args.out)
    mb = os.path.join(out_root, "MINI_BUILD.json")
    build = None
    if os.path.isfile(mb) and not args.force:
        cur = json.load(open(mb, encoding="utf-8"))
        if cur.get("selection_sha256") == sel["selection_sha256"] and cur.get("format") == FORMAT and \
                all(os.path.isfile(os.path.join(out_root, "tome", "csds", folder_of(ch), "dataframe_00000")) for ch in CHANNELS):
            build = cur
            print(f"tome {rel(out_root)} is up to date (selection {sel['selection_sha256'][:16]}): skipped (use --force to rebuild)")
        else:
            raise SystemExit(f"{rel(out_root)} was built from a different selection or is incomplete: rerun with --force")
    if build is None:
        build = build_tome(args, sel, src_root, ledger_raw, ledger)
    manifest_path = os.path.join(ROOT, "data", args.out, "MINI_MANIFEST.json")
    write_manifest(args, sel, build, manifest_path)
    print(f"manifest -> {rel(manifest_path)}")
    if args.run_pipeline:
        run_pipeline(args)
    if args.verify or args.run_pipeline:
        verify(args, sel, manifest_path)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
