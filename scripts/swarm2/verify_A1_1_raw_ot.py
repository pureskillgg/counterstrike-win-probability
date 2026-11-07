# verify_A1_1_raw_ot.py
# A1-1 follow-up: (1) overtime end-state rule from agreeing OT matches; (2) raw E: counts and header values
# for all 90 split-flagged matches (B + C) and a seeded sample of agreeing matches (A), read-only.
# Raw layout: E:\csds_final\csds\2026\10\<DD>\<match_id>\<channel>  (extensionless parquet).
import os, csv, random, collections, time
import polars as pl

print("== verify_A1_1_raw_ot ==")
t0 = time.time()
ROOT = r"G:\pureskill\csds-cs2"
T = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
E = r"E:\csds_final\csds\2026\10"
SCRATCH = r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad"
rows = list(csv.DictReader(open(os.path.join(SCRATCH, "verify_A1_1_permatch.csv"), encoding="utf-8")))
for r in rows:
    for c in ["mr", "thr", "S", "hdr_t", "hdr_c", "n", "rnd_t", "rnd_c", "first_reach"]:
        r[c] = int(r[c])
    r["ot"] = r["n"] > r["mr"]

print("\n== 1. Overtime end-state rule (agreeing OT matches, round-derived = header) ==")
A_ot = [r for r in rows if r["cat"] == "A_agree" and r["ot"]]
lead = collections.Counter()
for r in A_ot:
    w, l = max(r["hdr_t"], r["hdr_c"]), min(r["hdr_t"], r["hdr_c"])
    lead[w - l] += 1
print("agreeing OT matches:", len(A_ot), "| lead (winner - loser) distribution at end:", sorted(lead.items()))
print("agreeing OT matches with lead 1 at end:", lead.get(1, 0))
print("agreeing OT: winner total distribution:", sorted(collections.Counter(max(r['hdr_t'], r['hdr_c']) for r in A_ot).items()))
C_ot = [r for r in rows if r["cat"] == "C_sum_short" and r["ot"]]
print("\nC overtime matches (7): round-derived end state vs header end state")
for r in C_ot:
    rw, rl = max(r["rnd_t"], r["rnd_c"]), min(r["rnd_t"], r["rnd_c"])
    hw, hl = max(r["hdr_t"], r["hdr_c"]), min(r["hdr_t"], r["hdr_c"])
    print("   ", r["key"][-40:], "day", r["day"], "mr", r["mr"], "n", r["n"], "rounds-end", (rw, rl), "lead", rw - rl,
          "| header-end", (hw, hl), "lead", hw - hl, "| header S", r["S"])

print("\n== 2. Raw E: vs tome for flagged (B+C) and sampled agreeing (A) matches ==")
flag = [r for r in rows if r["cat"] in ("B_sum_eq_split_diff", "C_sum_short")]
A_all = [r for r in rows if r["cat"] == "A_agree"]
rng = random.Random(20261014)
A_samp = rng.sample(A_all, 40)
targets = flag + A_samp
print("targets:", len(flag), "flagged +", len(A_samp), "sampled agreeing")

def raw_info(key):
    dd, mid = key.split("/")[3], key.split("/")[4]
    base = os.path.join(E, dd, mid)
    out = {}
    for ch in ["round_end", "header", "score_update"]:
        p = os.path.join(base, ch)
        if not os.path.exists(p):
            out[ch] = None
            continue
        out[ch] = pl.read_parquet(p)
    return out

def tome_rows(key):
    res = {}
    for ch in ["basic_round_end.2026-09-18,2026-10-08", "basic_score_update.2026-09-18,2026-10-08"]:
        pages = sorted(os.path.join(T, ch, f) for f in os.listdir(os.path.join(T, ch)) if f.startswith("dataframe_"))
        res[ch] = pl.concat([pl.read_parquet(p) for p in pages], how="diagonal_relaxed").filter(pl.col("match_key") == key)
    return res

hdr_tome = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"),
                           columns=["key", "t_starters_score_final", "ct_starters_score_final", "max_rounds"])
hdr_map = {k: (t, c, m) for k, t, c, m in hdr_tome.iter_rows()}

res = collections.Counter()
detail = []
for i, r in enumerate(targets):
    k = r["key"]
    try:
        raw = raw_info(k)
    except Exception as ex:
        res["raw_error"] += 1
        detail.append((k[-40:], r["cat"], "raw_error", type(ex).__name__))
        continue
    if raw["round_end"] is None or raw["header"] is None:
        res["raw_missing_channel"] += 1
        detail.append((k[-40:], r["cat"], "missing", raw["round_end"] is None, raw["header"] is None))
        continue
    re_raw = raw["round_end"]
    n_raw = re_raw.height
    n_raw_distinct = re_raw.get_column("round").n_unique() if "round" in re_raw.columns else None
    max_round_raw = int(re_raw.get_column("round").max()) if "round" in re_raw.columns else None
    h_raw = raw["header"]
    tS = h_raw.get_column("t_starters_score_final").to_list()[0] if "t_starters_score_final" in h_raw.columns else None
    cS = h_raw.get_column("ct_starters_score_final").to_list()[0] if "ct_starters_score_final" in h_raw.columns else None
    mr_raw = h_raw.get_column("max_rounds").to_list()[0] if "max_rounds" in h_raw.columns else None
    tome_t, tome_c, tome_m = hdr_map[k]
    tm = tome_rows(k)
    n_tome = tm["basic_round_end.2026-09-18,2026-10-08"].height
    n_tome_su = tm["basic_score_update.2026-09-18,2026-10-08"].height
    su_raw = raw["score_update"]
    res["checked"] += 1
    res["raw_round_end_rows == tome_round_end_rows"] += int(n_raw == n_tome)
    res["raw_header_t,c == tome_header_t,c"] += int(tS == tome_t and cS == tome_c)
    res["raw_header_t,c == tome_header_t,c"] += 0
    res["raw_header_sum == tome_header_sum"] += int((tS or 0) + (cS or 0) == tome_t + tome_c)
    res["raw_round_end_rows == header_sum"] += int(n_raw == (tS or 0) + (cS or 0))
    res["raw_round_end_rows == tome_n_from_CSV"] += int(n_raw == r["n"])
    res["raw_maxround == rows"] += int(max_round_raw == n_raw)
    res["raw_max_rounds == tome_max_rounds"] += int(mr_raw == tome_m)
    if r["cat"] != "A_agree":
        res[f"{r['cat']}: raw header t,c == tome header t,c"] += int(tS == tome_t and cS == tome_c)
        res[f"{r['cat']}: raw rows == tome rows"] += int(n_raw == n_tome)
        res[f"{r['cat']}: raw_round_end_rows == csv_n"] += int(n_raw == r["n"])
    if r["cat"] == "C_sum_short" and i < 4:
        detail.append((k[-40:], r["cat"], "raw rows", n_raw, "tome rows", n_tome, "raw hdr", (tS, cS), "tome hdr", (tome_t, tome_c), "raw score_update rows", su_raw.height if su_raw is not None else None, "tome score_update rows", n_tome_su))
print("elapsed s:", round(time.time() - t0, 1))
for kk in sorted(res):
    print("  ", kk, res[kk])
for d in detail:
    print("  detail:", d)
print("Note: 'raw header t,c == tome header t,c' counts appear in the per-category lines above.")
