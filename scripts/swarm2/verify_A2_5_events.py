# verify_A2_5_events.py
# Independent check of A2-5 at the event level: round_end fingerprints (the claimed key), the
# event signature used by cross_period.py (round_end + player_death + player_info), and the
# header signature. Reads the 2024 tome channels with projected columns only, one channel at a time.
import os, json, collections
import pyarrow.parquet as pq

BASE = r"G:\pureskill\csds-cs2"
TOME = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
HDR = os.path.join(TOME, "header", "dataframe_00000")
HCOLS = ["key", "match_id", "match_date", "map_name", "t_starters_score_final",
         "ct_starters_score_final", "number_of_points"]


def channel_pages(ch):
    dirs = [d for d in os.listdir(TOME) if d.split(".", 1)[0] == "basic_" + ch]
    if len(dirs) != 1:
        raise SystemExit(f"channel {ch}: expected 1 folder, found {dirs}")
    dp = os.path.join(TOME, dirs[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf-8"))
    return [os.path.join(dp, p["dataframe"]["dataframe"]) for p in meta["pages"]]


def read_channel(ch, cols):
    out = []
    files = channel_pages(ch)
    for f in files:
        t = pq.read_table(f, columns=cols)
        colvals = [t.column(c).to_pylist() for c in cols]
        out.extend(zip(*colvals))
        del t, colvals
    print(f"  channel {ch}: pages={len(files)} rows={len(out)}")
    return out


def sig_claim(r):
    return "|".join(str(r[c]) for c in ["match_date", "map_name", "t_starters_score_final",
                                        "ct_starters_score_final", "number_of_points"])


def groups_by(keys_to_val):
    g = collections.defaultdict(list)
    for k, v in keys_to_val.items():
        g[v].append(k)
    return {v: ks for v, ks in g.items() if len(ks) > 1}


def main():
    hrows = pq.read_table(HDR, columns=HCOLS).to_pylist()
    keys = [r["key"] for r in hrows]
    keyset = set(keys)
    hsig = {r["key"]: sig_claim(r) for r in hrows}
    print("header keys:", len(keys))

    print("loading channels (projected):")
    RE = read_channel("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    DE = read_channel("player_death", ["match_key", "round", "tick", "player_id"])
    PI = read_channel("player_info", ["match_key", "round", "player_id", "team_code"])

    re_by = collections.defaultdict(list)
    for mk, rd, tk, w, wr, pc in RE:
        re_by[mk].append((str(rd), str(tk), str(w), str(wr), str(pc)))
    de_by = collections.defaultdict(list)
    for mk, rd, tk, pid in DE:
        de_by[mk].append((str(rd), str(tk), str(pid)))
    pi_by = collections.defaultdict(list)
    for mk, rd, pid, tm in PI:
        pi_by[mk].append((str(rd), str(pid), str(tm)))
    for name, src in [("round_end", RE), ("player_death", DE), ("player_info", PI)]:
        mks = {x[0] for x in src}
        print(f"  {name}: match_keys not in header: {len(mks - keyset)}")
    print()

    # claimed fingerprint: sorted (round, tick, winner) over round_end rows
    fp_claim = {k: tuple(sorted((rd, tk, w) for rd, tk, w, _, _ in re_by[k])) for k in keys if re_by.get(k)}
    no_re = [k for k in keys if not re_by.get(k)]
    print(f"matches with >=1 round_end row: {len(fp_claim)}; with none: {len(no_re)}")
    if no_re:
        hsig_no_re = collections.Counter(hsig[k] for k in no_re)
        print("  no-round_end matches whose header signature is shared with another match:",
              sum(1 for k in no_re if sum(1 for x in keys if hsig[x] == hsig[k]) > 1))
    print()

    hdr_dg = groups_by(hsig)
    hdr_rows = sum(len(v) for v in hdr_dg.values())
    print(f"header-signature groups (claim): {len(hdr_dg)}, rows {hdr_rows}, excess {hdr_rows - len(hdr_dg)}")

    # 1. Do claimed header groups have one fingerprint across members?
    grp_fp_uniform, grp_fp_split, grp_has_empty = 0, [], 0
    for s, ks in hdr_dg.items():
        fps = [fp_claim.get(k) for k in ks]
        if any(f is None for f in fps):
            grp_has_empty += 1
            continue
        if len(set(fps)) == 1:
            grp_fp_uniform += 1
        else:
            grp_fp_split.append((s, ks))
    print(f"claimed groups: round_end fingerprint identical across members: {grp_fp_uniform}; "
          f"split: {len(grp_fp_split)}; groups with a member lacking round_end: {grp_has_empty}")
    within_pairs = sum(len(ks) * (len(ks) - 1) // 2 for ks in hdr_dg.values())
    print(f"claimed within-group pairs: {within_pairs}")
    print()

    # 2. Round-end fingerprint groups over ALL 2024 matches (no header signature used)
    fp_dg = groups_by(fp_claim)
    fp_rows = sum(len(v) for v in fp_dg.values())
    print(f"round_end-fingerprint groups over all matches: {len(fp_dg)}, rows {fp_rows}, "
          f"excess {fp_rows - len(fp_dg)}, size dist {sorted(collections.Counter(len(v) for v in fp_dg.values()).items())}")
    in_hdr = 0
    extra_members, extra_groups = [], []
    for fp, ks in fp_dg.items():
        sigs = {hsig[k] for k in ks}
        if len(sigs) == 1 and len(hdr_dg.get(hsig[ks[0]], [])) == len(ks):
            in_hdr += 1
        else:
            extra_groups.append(ks)
            for k in ks:
                if len(hdr_dg.get(hsig[k], [])) <= 1:
                    extra_members.append(k)
    print(f"  fingerprint groups equal to a header group exactly: {in_hdr}")
    print(f"  fingerprint groups NOT equal to a header group: {len(extra_groups)}; "
          f"members with a header-singleton signature: {len(extra_members)}")
    for ks in extra_groups[:8]:
        print("    fp-group members:", [(k.split('/')[3], k.split('/')[4][:8], hsig[k][:40]) for k in ks])
    # header groups with members in different fingerprint groups
    hdr_split_by_fp = [s for s, ks in hdr_dg.items() if len({fp_claim.get(k) for k in ks}) > 1]
    print(f"  header groups whose members fall in different fingerprint groups: {len(hdr_split_by_fp)}")
    print()

    # 3. Event signature (round_end + player_death + player_info) as used by cross_period.py
    ev = {}
    for k in keys:
        if re_by.get(k):
            ev[k] = (fp_claim[k], tuple(sorted(de_by.get(k, []))), tuple(sorted(pi_by.get(k, []))))
    ev_dg = groups_by(ev)
    ev_rows = sum(len(v) for v in ev_dg.values())
    print(f"event-signature (RE+death+player_info) groups over all matches: {len(ev_dg)}, rows {ev_rows}, "
          f"excess {ev_rows - len(ev_dg)}")
    print(f"  round_end-only fingerprint groups that split under the full event signature: "
          f"{sum(1 for ks in fp_dg.values() if len({ev[k] for k in ks}) > 1)}")
    print(f"  (fingerprint groups with >=2 members whose death/player_info differ = round-only over-merge)")
    print()

    # 4. Round-level size of the excess copies (all matches, before any channel filter)
    def excess_rounds(dg):
        n_matches, n_rounds = 0, 0
        for ks in dg.values():
            for k in sorted(ks)[1:]:
                n_matches += 1
                n_rounds += len(re_by.get(k, []))
        return n_matches, n_rounds
    m_h, r_h = excess_rounds(hdr_dg)
    m_e, r_e = excess_rounds(ev_dg)
    print(f"excess copies (keep lowest key) header-sig: matches {m_h}, round_end rows {r_h}")
    print(f"excess copies (keep lowest key) event-sig:  matches {m_e}, round_end rows {r_e}")
    print("DONE events")


if __name__ == "__main__":
    main()
