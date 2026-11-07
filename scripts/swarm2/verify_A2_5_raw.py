# verify_A2_5_raw.py
# A2-5 verification, part 3: raw copies under E: (tail-recovered 2024 export and csds_extracted for day 01).
# Questions: (1) do the duplicate UUIDs exist as separate raw match folders WITH content? (day-01 folders in the
# tail export are empty stubs; the content is in csds_extracted); (2) are the copies byte-identical in the small
# channel files? (3) what separates the 5 round_end-only groups (7 pairs) that the header key misses?
# Reads only: 2024 tome header and round_end parquet, E: raw match folders. Writes nothing.
import os, hashlib, collections, json, gc
import pyarrow.parquet as pq

BASE = r"G:\pureskill\csds-cs2"
TOME = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
HDR = os.path.join(TOME, "header", "dataframe_00000")
TAIL = r"E:\csds_final\csds\csds_tail_recovered_20261011\csds\2024\02"
EXT = r"E:\csds_extracted\csds\2024\02"
HCOLS = ["key", "match_id", "match_date", "map_name", "t_starters_score_final",
         "ct_starters_score_final", "number_of_points", "build_num", "ppp_version", "rushb_version",
         "providence"]
SMALL = ["header", "round_end", "csds"]             # small per-match files, hashed in full
BIG_CHECK = ["tick", "player_death", "player_info"]  # hashed only for the round_end-only pairs
SIZES = ["header", "round_end", "csds", "tick", "player_death", "player_info", "player_vector", "player_status"]


def fhash(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def round_end_digest():
    dirs = [d for d in os.listdir(TOME) if d.split(".", 1)[0] == "basic_round_end"]
    dp = os.path.join(TOME, dirs[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf-8"))
    rows_by = collections.defaultdict(list)
    for p in meta["pages"]:
        pf = pq.ParquetFile(os.path.join(dp, p["dataframe"]["dataframe"]))
        for batch in pf.iter_batches(batch_size=65536, columns=["match_key", "round", "tick", "winner_team_code"]):
            d = batch.to_pydict()
            for i in range(len(d["match_key"])):
                rows_by[d["match_key"][i]].append(f"{d['round'][i]}|{d['tick'][i]}|{d['winner_team_code'][i]}")
    out = {}
    for mk, lst in rows_by.items():
        lst.sort()
        out[mk] = hashlib.sha1("\n".join(lst).encode("utf-8")).hexdigest()
    return out


def groups_by(d):
    g = collections.defaultdict(list)
    for k, v in d.items():
        g[v].append(k)
    return {v: sorted(ks) for v, ks in g.items() if len(ks) > 1}


def main():
    hrows = pq.read_table(HDR, columns=HCOLS).to_pylist()
    H = {r["key"]: r for r in hrows}
    keys = sorted(H)
    day = {k: k.split("/")[3] for k in keys}
    uuid = {k: k.split("/")[4] for k in keys}
    hsig = {k: "|".join(str(H[k][c]) for c in ["match_date", "map_name", "t_starters_score_final",
                                               "ct_starters_score_final", "number_of_points"]) for k in keys}
    fp = round_end_digest()
    gc.collect()
    hdr_dg = groups_by(hsig)
    fp_dg = groups_by({k: v for k, v in fp.items() if k in H})
    hdr_part = {frozenset(v) for v in hdr_dg.values()}
    fp_only = [f for f in fp_dg.values() if frozenset(f) not in hdr_part]
    claimed = list(hdr_dg.values())
    print(f"claimed header groups: {len(claimed)}; round_end-only groups: {len(fp_only)}")

    def nonempty(p):
        return os.path.isdir(p) and len(os.listdir(p)) > 0

    def root_for(k):
        t = os.path.join(TAIL, day[k], uuid[k])
        if nonempty(t):
            return t, "tail"
        e = os.path.join(EXT, day[k], uuid[k])
        if nonempty(e):
            return e, "extracted"
        return None, "none"

    # 1. where does each claimed member have content?
    src = collections.Counter(root_for(k)[1] for ks in claimed for k in ks)
    print(f"1. claimed members (522) by raw source with content: {dict(src)}")
    d01_tail_empty = sum(1 for ks in claimed for k in ks if day[k] == "01" and not nonempty(os.path.join(TAIL, day[k], uuid[k])))
    print(f"   day-01 claimed members whose tail-export folder is empty: {d01_tail_empty}")
    groups_all_content = sum(1 for ks in claimed if all(root_for(k)[1] != "none" for k in ks))
    print(f"   claimed groups where every member has raw content: {groups_all_content} of {len(claimed)}")
    print()

    # 2. byte comparison of each member against the first member of its group that has content
    combo = collections.Counter()
    ex = []
    for ks in claimed:
        withc = [k for k in ks if root_for(k)[1] != "none"]
        if len(withc) < 2:
            continue
        ref = withc[0]
        rref = root_for(ref)[0]
        for other in withc[1:]:
            ro = root_for(other)[0]
            vdiff = (H[ref]["ppp_version"], H[ref]["rushb_version"]) != (H[other]["ppp_version"], H[other]["rushb_version"])
            eq = {}
            for name in SMALL:
                pa, pb = os.path.join(rref, name), os.path.join(ro, name)
                eq[name] = (fhash(pa) == fhash(pb)) if (os.path.isfile(pa) and os.path.isfile(pb)) else None
            combo[(eq["header"], eq["round_end"], eq["csds"], "ver_diff" if vdiff else "ver_same")] += 1
            if not eq["header"] and eq["header"] is not None and len(ex) < 3:
                ex.append((day[ref], uuid[ref][:8], day[other], uuid[other][:8]))
    print("2. copy pairs in claimed groups (member vs first member with content):")
    print("   (header_bytes_equal, round_end_bytes_equal, csds_bytes_equal, parser_versions):")
    for k, n in combo.most_common():
        print("   ", k, n)
    print("   examples where header bytes differ:", ex)
    print()

    # 3. the round_end-only groups: raw evidence
    print("3. round_end-only groups (not a header group):")
    for f in fp_only:
        print("   group:")
        for k in f:
            h = H[k]
            print("     ", (day[k], uuid[k][:8], str(h["match_date"])[:19], h["map_name"], h["t_starters_score_final"],
                         h["ct_starters_score_final"], "nop", h["number_of_points"], "ppp", h["ppp_version"],
                         "rushb", h["rushb_version"], "build", h["build_num"], "source", root_for(k)[1]))
        withc = [k for k in f if root_for(k)[1] != "none"]
        if len(withc) >= 2:
            ref = withc[0]
            rref = root_for(ref)[0]
            for other in withc[1:]:
                ro = root_for(other)[0]
                res = {}
                for name in SMALL + BIG_CHECK:
                    pa, pb = os.path.join(rref, name), os.path.join(ro, name)
                    if os.path.isfile(pa) and os.path.isfile(pb):
                        same = fhash(pa) == fhash(pb)
                        res[name] = ("eq" if same else "DIFF", os.path.getsize(pa), os.path.getsize(pb))
                print("     pair", day[ref], uuid[ref][:8], "vs", day[other], uuid[other][:8])
                for name, v in res.items():
                    print("        ", name, v)
                sz = {}
                for name in SIZES:
                    pa, pb = os.path.join(rref, name), os.path.join(ro, name)
                    if os.path.isfile(pa) and os.path.isfile(pb):
                        sz[name] = (os.path.getsize(pa), os.path.getsize(pb))
                print("         sizes (ref, other):", sz)
        else:
            print("     fewer than 2 members with content")
    print()
    print("DONE raw")


if __name__ == "__main__":
    main()
