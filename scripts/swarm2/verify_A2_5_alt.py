# verify_A2_5_alt.py
# A2-5 verification, part 2 (memory-light): event-level fingerprints and alternative explanations.
#  A. do the header key and the event keys partition the 2024 matches the same way?
#  B. timestamp-sharing pairs: which fields separate distinct matches from copies?
#  D. reproduce the cross_period.py channel filter (214 dropped, 4221 kept) and its dedup (202 groups, 271 copies)
#  E. which header fields differ inside the claimed groups
#  F. within-match duplicate round_end (match, round) rows
# Raw E: byte checks are in verify_A2_5_raw.py (kept separate to bound memory).
# Reads only: 2024 tome parquet. Writes nothing. Each channel is streamed in 65k-row batches and reduced to
# one sha1 digest per match.
import os, json, hashlib, collections, gc
import pyarrow.parquet as pq

BASE = r"G:\pureskill\csds-cs2"
TOME = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
HDR = os.path.join(TOME, "header", "dataframe_00000")
HCOLS = ["key", "match_id", "match_date", "map_name", "t_starters_score_final",
         "ct_starters_score_final", "number_of_points", "build_num", "tick_rate", "match_type",
         "platform", "providence", "rushb_version", "ppp_version", "t_starters_avg_rank",
         "t_starters_avg_wins", "ct_starters_avg_rank", "ct_starters_avg_wins"]
EXTRA = HCOLS[7:]
DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state"]
SPARSE = ["player_disconnect", "bot_takeover"]


def channel_meta(ch):
    dirs = [d for d in os.listdir(TOME) if d.split(".", 1)[0] == "basic_" + ch]
    if len(dirs) != 1:
        raise SystemExit(f"channel {ch}: {dirs}")
    dp = os.path.join(TOME, dirs[0])
    return dp, json.load(open(os.path.join(dp, "tome"), encoding="utf-8"))


def digest_channel(ch, cols):
    """Per match: sha1 of sorted row strings, and the row count. cols[0] must be match_key."""
    dp, meta = channel_meta(ch)
    rows_by = collections.defaultdict(list)
    for p in meta["pages"]:
        pf = pq.ParquetFile(os.path.join(dp, p["dataframe"]["dataframe"]))
        for batch in pf.iter_batches(batch_size=65536, columns=cols):
            d = batch.to_pydict()
            mks = d[cols[0]]
            vals = [d[c] for c in cols[1:]]
            for i in range(len(mks)):
                rows_by[mks[i]].append("|".join(str(v[i]) for v in vals))
            del d, vals
    dig, cnt = {}, {}
    for mk, lst in rows_by.items():
        lst.sort()
        dig[mk] = hashlib.sha1("\n".join(lst).encode("utf-8")).hexdigest()
        cnt[mk] = len(lst)
    del rows_by
    gc.collect()
    return dig, cnt


def key_set(ch):
    dp, meta = channel_meta(ch)
    s = set()
    for p in meta["pages"]:
        pf = pq.ParquetFile(os.path.join(dp, p["dataframe"]["dataframe"]))
        for batch in pf.iter_batches(batch_size=65536, columns=["match_key"]):
            s.update(batch.column(0).to_pylist())
    return s


def keyset_keys(ch):
    dp, meta = channel_meta(ch)
    out = set()
    for p in meta["pages"]:
        out |= set(pq.read_table(os.path.join(dp, p["keyset"]["keyset"])).column(0).to_pylist())
    return out


def sig_claim(r):
    return "|".join(str(r[c]) for c in ["match_date", "map_name", "t_starters_score_final",
                                        "ct_starters_score_final", "number_of_points"])


def groups_by(d):
    g = collections.defaultdict(list)
    for k, v in d.items():
        g[v].append(k)
    return {v: sorted(ks) for v, ks in g.items() if len(ks) > 1}


def main():
    hrows = pq.read_table(HDR, columns=HCOLS).to_pylist()
    keys = sorted(r["key"] for r in hrows)
    H = {r["key"]: r for r in hrows}
    allm = set(keys)
    hsig = {k: sig_claim(H[k]) for k in keys}
    day = {k: k.split("/")[3] for k in keys}
    uuid = {k: k.split("/")[4] for k in keys}

    fp, re_cnt = digest_channel("round_end", ["match_key", "round", "tick", "winner_team_code"])
    de, de_cnt = digest_channel("player_death", ["match_key", "round", "tick", "player_id"])
    pi, pi_cnt = digest_channel("player_info", ["match_key", "round", "player_id", "team_code"])
    fp = {k: v for k, v in fp.items() if k in allm}
    ev = {k: (fp[k], de.get(k, ""), pi.get(k, "")) for k in fp}
    print("channels digested: round_end matches", len(fp), "| player_death matches", len(de_cnt),
          "| player_info matches", len(pi_cnt))
    print()

    hdr_dg = groups_by(hsig)
    ev_dg = groups_by(ev)
    fp_dg = groups_by(fp)
    claimed = list(hdr_dg.values())

    # A. partition comparison
    hdr_part = {frozenset(v) for v in hdr_dg.values()}
    ev_part = {frozenset(v) for v in ev_dg.values()}
    fp_part = {frozenset(v) for v in fp_dg.values()}
    print("A. partitions")
    print(f"  header-sig groups {len(hdr_dg)} (excess {sum(len(v)-1 for v in hdr_dg.values())}); "
          f"event-sig groups {len(ev_dg)} (excess {sum(len(v)-1 for v in ev_dg.values())}); "
          f"identical partition: {hdr_part == ev_part}")
    print(f"  round_end-fingerprint groups {len(fp_dg)} (excess {sum(len(v)-1 for v in fp_dg.values())}); "
          f"members {sum(len(v) for v in fp_dg.values())}")
    print(f"  header groups that are exactly a round_end-fingerprint group: {len(hdr_part & fp_part)} of {len(hdr_part)}")
    fp_only = [f for f in fp_dg.values() if frozenset(f) not in hdr_part]
    print(f"  round_end-fingerprint groups NOT equal to a header group: {len(fp_only)} "
          f"(members {sum(len(f) for f in fp_only)})")
    fp_only_ev = [f for f in fp_dg.values() if frozenset(f) not in ev_part]
    print(f"  round_end-fingerprint groups NOT equal to an event-sig group: {len(fp_only_ev)}")
    print()

    # B. timestamp-sharing pairs
    print("B. pairs sharing the match_date timestamp (all 4435 matches)")
    by_ts = collections.defaultdict(list)
    for k in keys:
        by_ts[str(H[k]["match_date"])].append(k)
    cat = collections.Counter()
    fp_same = collections.Counter()
    examples = collections.defaultdict(list)
    for t, ks in by_ts.items():
        for i in range(len(ks)):
            for j in range(i + 1, len(ks)):
                a, b = ks[i], ks[j]
                same_ms = all(str(H[a][c]) == str(H[b][c]) for c in
                              ["map_name", "t_starters_score_final", "ct_starters_score_final"])
                same_nop = str(H[a]["number_of_points"]) == str(H[b]["number_of_points"])
                if same_ms and same_nop:
                    c = "same map+scores+nop (claim key)"
                elif same_ms:
                    c = "same map+scores, nop differs"
                else:
                    c = "map or score differs"
                cat[c] += 1
                same_fp = bool(fp.get(a)) and fp.get(a) == fp.get(b)
                fp_same[(c, same_fp)] += 1
                if len(examples[c]) < 8:
                    examples[c].append((day[a], uuid[a][:8], day[b], uuid[b][:8]))
    print(f"  timestamps shared by >1 match: {sum(1 for v in by_ts.values() if len(v) > 1)}")
    for c, n in cat.items():
        print(f"  {c}: pairs {n}; round_end fingerprint identical: {fp_same[(c, True)]}")
    for c in examples:
        if c != "same map+scores+nop (claim key)":
            print(f"  examples ({c}) (day_a, uuid_a, day_b, uuid_b):", examples[c])
    print()

    # D. channel filter and dedup (reproduce cross_period.py)
    print("D. cross_period.py channel filter and dedup (reproduced from keysets and channel key sets)")
    have = {"round_end": set(re_cnt), "player_death": set(de_cnt), "player_info": set(pi_cnt)}
    dropped = set()
    for ch in DENSE + SPARSE:
        ks = keyset_keys(ch)
        dense = ch in DENSE
        if dense and ch not in have:
            have[ch] = key_set(ch)
        zero = (allm - have[ch]) if dense else set()
        no_ks = allm - ks
        dropped |= (no_ks | zero)
        print(f"  {ch:<20} keyset={len(ks)} not_in_keyset={len(no_ks)} zero_row={len(zero)}")
    kept = allm - dropped
    print(f"  dropped by channel filter: {len(dropped)}; kept: {len(kept)}  "
          f"(cross_period.md table: 214 dropped, 4221 kept)")
    kg = groups_by({k: v for k, v in ev.items() if k in kept})
    kc = sum(len(v) - 1 for v in kg.values())
    print(f"  event-sig dup groups among kept: {len(kg)}, copies removed (lowest key kept): {kc}  "
          f"(cross_period.md:69 says 202 groups, 271 copies)")
    kh = groups_by({k: v for k, v in hsig.items() if k in kept})
    print(f"  header-sig dup groups among kept: {len(kh)}, excess {sum(len(v)-1 for v in kh.values())}")
    kf = groups_by({k: v for k, v in fp.items() if k in kept})
    print(f"  round_end-fingerprint dup groups among kept: {len(kf)}, excess {sum(len(v)-1 for v in kf.values())}")
    excess_hdr = [k for v in hdr_dg.values() for k in v[1:]]
    print(f"  header-sig excess copies over all matches: {len(excess_hdr)}; in channel-dropped matches: "
          f"{sum(1 for k in excess_hdr if k in dropped)}; in kept matches: {sum(1 for k in excess_hdr if k in kept)}")
    rr = sum(re_cnt.get(k, 0) for v in kg.values() for k in v[1:])
    print(f"  round_end rows in event-sig excess copies among kept: {rr} "
          f"(cross_period.md:59 reports 4559 dropped rounds, after round filters)")
    print()

    # E. header fields that differ inside the claimed groups
    print("E. header fields that differ inside the claimed groups")
    cnt = collections.Counter()
    combos = collections.Counter()
    for ks in claimed:
        diffs = tuple(c for c in EXTRA if len({str(H[k][c]) for k in ks}) > 1)
        for c in diffs:
            cnt[c] += 1
        combos[diffs] += 1
    print("  groups differing per field:", dict(cnt))
    print("  combinations (count):", combos.most_common())
    print()

    # F. within-match duplicate (match, round) rows in round_end
    print("F. within-match duplicates in round_end")
    dp, meta = channel_meta("round_end")
    seen = collections.Counter()
    for p in meta["pages"]:
        pf = pq.ParquetFile(os.path.join(dp, p["dataframe"]["dataframe"]))
        for batch in pf.iter_batches(batch_size=65536, columns=["match_key", "round"]):
            d = batch.to_pydict()
            for mk, rd in zip(d["match_key"], d["round"]):
                seen[(mk, rd)] += 1
    multi = [(k, n) for k, n in seen.items() if n > 1]
    in_claimed = set(k for v in claimed for k in v)
    print(f"  (match, round) pairs with >1 round_end row: {len(multi)}; extra rows: {sum(n-1 for _, n in multi)}")
    for (mk, rd), n in multi[:5]:
        print("   ", (day.get(mk, "?"), uuid.get(mk, "?")[:8], rd, n, "header-group member" if mk in in_claimed else "not in a header group"))
    print("DONE alt")


if __name__ == "__main__":
    main()
