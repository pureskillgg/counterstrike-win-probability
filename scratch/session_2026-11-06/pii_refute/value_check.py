import sys, os, re, json, collections
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

ROOT = "G:/pureskill/csds-cs2"
man = json.load(open(ROOT + "/deliverables/email_package_2026-10-19_review/results/split_manifest.json", encoding="utf-8"))
assign = man["assign"]
hold = set(l.strip() for l in open(ROOT + "/data/holdout_b/holdout_keys.txt", encoding="utf-8") if l.strip())
OCT_DAYS = set("%02d" % i for i in range(22, 32))


def key_date(k):
    p = k.split("/")
    if len(p) >= 4:
        return p[1], p[2], p[3]
    return None


def is_oct22_31(k):
    d = key_date(k)
    return d is not None and d[0] == "2026" and d[1] == "10" and d[2] in OCT_DAYS


key_forms = collections.Counter("csds-prefix" if k.startswith("csds/") else "other" for k in assign)
print("KEY_FORMS", dict(key_forms))
print("ASSIGN_COUNTS", dict(collections.Counter(assign.values())))
print("HOLDOUT_KEYS", len(hold), "in_manifest", sum(1 for k in hold if k in assign),
      "holdout_assign_values", dict(collections.Counter(assign[k] for k in hold if k in assign)))
print("OCT22_31_KEYS_IN_MANIFEST", sum(1 for k in assign if is_oct22_31(k)),
      "by_set", dict(collections.Counter(assign[k] for k in assign if is_oct22_31(k))))

allowed = [k for k, v in assign.items() if v in ("train", "val") and k not in hold and not is_oct22_31(k)]
print("ALLOWED_TRAIN_VAL_KEYS", len(allowed))
allowed_arr = pa.array(allowed, type=pa.string())

PATS = {
    "steamid64_digits": r"7656119[0-9]{10}",
    "steamid2": r"STEAM_[0-5]:[01]:[0-9]+",
    "steamid3": r"\[U:1:[0-9]+\]",
    "ipv4": r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}",
    "email": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
    "url_or_steam_domain": r"https?://|www\.|steamcommunity|steampowered",
    "windows_user_path": r"[A-Za-z]:[\\/]+Users[\\/]+",
    "digit_run_9plus": r"[0-9]{9,}",
}
CASE_INSENSITIVE = ("steamid2", "url_or_steam_domain", "windows_user_path")
UUID_KEY = r"^csds/[0-9]{4}/[0-9]{2}/[0-9]{2}/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}/csds$"
ID_RE = re.compile(r"(player_id|attacker_id|assister|victim|steam|userid|user_id|account|_sid$|_uid$|^id$)", re.I)
SID64_LO, SID64_HI = 76561197960265728, 76561202255233023
CAT_CAP = 40
DIST_CAP = 200000


def scan(path, label):
    dset = ds.dataset(path, format="parquet")
    names = dset.schema.names
    keycol = None
    for c in ("match_key", "match", "key"):
        if c in names:
            keycol = c
            break
    if keycol is None:
        print("FILE", label, "SKIP no string match key; ncols", len(names))
        return
    kt = dset.schema.field(keycol).type
    if not (pa.types.is_string(kt) or pa.types.is_large_string(kt)):
        print("FILE", label, "SKIP key col not string:", keycol, str(kt))
        return
    flt = ds.field(keycol).isin(allowed_arr)
    rows = 0
    stats = {}
    for c in names:
        t = dset.schema.field(c).type
        if pa.types.is_integer(t) or pa.types.is_floating(t):
            kind = "num"
        elif pa.types.is_string(t) or pa.types.is_large_string(t):
            kind = "str"
        else:
            kind = "other"
        stats[c] = {"type": str(t), "kind": kind, "min": None, "max": None, "dist": set(),
                    "dist_capped": False, "sid64_hits": 0, "ge2e32": 0,
                    "rx": collections.Counter(), "keyfmt_ok": 0}
    scanner = dset.scanner(filter=flt, batch_size=262144)
    for b in scanner.to_batches():
        n = b.num_rows
        if n == 0:
            continue
        rows += n
        for c in names:
            st = stats[c]
            col = b.column(c)
            if st["kind"] == "num" and (ID_RE.search(c) or c == "player_id") and c != keycol:
                mm = pc.min_max(col).as_py()
                if mm["min"] is not None:
                    st["min"] = mm["min"] if st["min"] is None else min(st["min"], mm["min"])
                    st["max"] = mm["max"] if st["max"] is None else max(st["max"], mm["max"])
                if not st["dist_capped"]:
                    st["dist"].update(x for x in pc.unique(col).to_pylist() if x is not None)
                    if len(st["dist"]) > DIST_CAP:
                        st["dist_capped"] = True
                if pa.types.is_int64(col.type):
                    lo = pc.greater_equal(col, pa.scalar(SID64_LO, pa.int64()))
                    hi = pc.less_equal(col, pa.scalar(SID64_HI, pa.int64()))
                    st["sid64_hits"] += pc.sum(pc.cast(pc.and_(lo, hi), pa.int64())).as_py() or 0
                    st["ge2e32"] += pc.sum(pc.cast(pc.greater_equal(col, pa.scalar(2 ** 32, pa.int64())), pa.int64())).as_py() or 0
                elif pa.types.is_integer(col.type):
                    st["ge2e32"] += 0  # int8/16/32 cannot hold SteamID64-sized values; range check not applicable
            if st["kind"] == "str":
                for pname, rx in PATS.items():
                    m = pc.match_substring_regex(col, rx, ignore_case=(pname in CASE_INSENSITIVE))
                    st["rx"][pname] += pc.sum(pc.cast(m, pa.int64())).as_py() or 0
                if c == keycol:
                    st["keyfmt_ok"] += pc.sum(pc.cast(pc.match_substring_regex(col, UUID_KEY), pa.int64())).as_py() or 0
                if not st["dist_capped"]:
                    st["dist"].update(x for x in pc.unique(col).to_pylist() if x is not None)
                    if len(st["dist"]) > DIST_CAP:
                        st["dist_capped"] = True
    print("FILE", label, "rows_in_train_val_scope", rows, "key_col", keycol)
    for c in names:
        st = stats[c]
        if st["kind"] == "str":
            nd = "capped" if st["dist_capped"] else len(st["dist"])
            hits = {k: v for k, v in st["rx"].items() if v}
            line = "  STR %s: distinct=%s hits=%s" % (c, nd, hits if hits else "none")
            if c == keycol:
                line += " key_uuid_form_ok=%d/%d" % (st["keyfmt_ok"], rows)
            print(line)
            if (not st["dist_capped"]) and 0 < len(st["dist"]) <= CAT_CAP and c != keycol:
                print("     CATEGORIES", sorted(str(x) for x in st["dist"]))
        elif st["kind"] == "num" and (ID_RE.search(c) or c == "player_id") and c != keycol:
            nd = "capped" if st["dist_capped"] else len(st["dist"])
            print("  ID %s (%s): min=%s max=%s distinct=%s steamid64_range_hits=%d ge_2^32_count=%d" % (
                c, st["type"], st["min"], st["max"], nd, st["sid64_hits"], st["ge2e32"]))
    sys.stdout.flush()


targets = [
    ("csds-cs2/data/positions_1hz/2026-09-01.parquet", "positions_1hz 2026-09-01"),
    ("csds-cs2/data/status_1hz/2026-09-01.parquet", "status_1hz 2026-09-01"),
    ("csds-cs2/data/blind_events/2026-09-01.parquet", "blind_events 2026-09-01"),
    ("csds-cs2/data/combined/deaths.parquet", "combined deaths (scoped)"),
    ("csds-cs2/data/state_sample/players_1hz.parquet", "state_sample players_1hz (scoped)"),
    ("csds-cs2/data/state_sample/place_names.parquet", "state_sample place_names (scoped)"),
    ("csds-cs2/logs/swarm/E1/cache/header.parquet", "swarm E1 header"),
    ("csds-cs2/logs/swarm/E1/cache/item_dropped.parquet", "swarm E1 item_dropped"),
    ("csds-cs2/logs/swarm/E1/cache/round_state.parquet", "swarm E1 round_state"),
    ("csds-cs2/scratchpad_val_failure_join.parquet", "validation failure join"),
    ("csds-cs2/data/combined/grid.parquet", "combined grid (scoped)"),
    ("csds-cs2/data/diagnostics/w2v_tokens_de_mirage.parquet", "w2v tokens de_mirage (scoped)"),
]
only = sys.argv[1:]
for rel, label in targets:
    if only and not any(o in rel or o in label for o in only):
        continue
    p = "G:/pureskill/" + rel
    if not os.path.exists(p):
        print("MISSING", rel)
        continue
    try:
        scan(p, label)
    except Exception as e:
        print("ERROR", label, type(e).__name__, str(e)[:300])
