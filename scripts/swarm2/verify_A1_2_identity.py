# -*- coding: utf-8 -*-
"""
verify_A1_2_identity.py  (independent verifier for finding A1-2)

Claim under test: for 10 matches (max_rounds=16, days 01-03) the header t/ct starter totals are
swapped relative to round-derived starter totals, under the assumed rule "one side swap after round 8".

Finding from the probes: max_rounds=16 matches are is_wingman=True (2v2, 4 players).
Team identity below is NOT derived from any swap-rule assumption:
  - each player's side in each round comes from basic_player_info.team_code (2 = T, 3 = CT),
  - the player is linked to a pseudonymous steam_id via basic_player_personal (per match, by player_id),
  - the starter-T group = steam_ids on side 2 in the first round where both sides have the
    required head count (2 for wingman, 5 otherwise),
  - for each round, the round_end winner side (2/3) is compared with the side the starter-T group
    was actually on in that round (observed, not assumed).
Cross-checks:
  - round_end winner side vs score_update increments (engine side-coded score),
  - engine final side scores (last new_score per team_code) vs round_end side counts,
  - observed swap points vs the claimed rule (swap after max_rounds//2).
Output is printed; redirect to logs\\swarm2\\verify_A1_2.log.
"""
import os, glob, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CLAIMED = [
    "csds/2026/10/01/m_36b246426c63e2e81c3165c3/csds",
    "csds/2026/10/01/m_b2289de5d46c89d81317f2ff/csds",
    "csds/2026/10/02/m_fe71dd645987a933c259261f/csds",
    "csds/2026/10/02/m_a76f2712208e50fe59cc1082/csds",
    "csds/2026/10/02/m_ab782810212100e291053f63/csds",
    "csds/2026/10/02/m_7cc561b7c0bd47c32f7550b0/csds",
    "csds/2026/10/02/m_757b2afedab995a6dd2f429d/csds",
    "csds/2026/10/02/m_888882dcf9e1d301f91ad5b4/csds",
    "csds/2026/10/03/m_b0bf8e190e1fcd142f36e859/csds",
    "csds/2026/10/03/m_930de9b06ad0081177ea9537/csds",
]


def chan(name):
    folder = [d for d in os.listdir(T) if d.startswith(name + ".")][0]
    return sorted(glob.glob(os.path.join(T, folder, "dataframe_*")))


def scan(name, cols):
    return pl.scan_parquet(chan(name)).select(cols).collect()


pi = scan("basic_player_info", ["match_key", "round", "player_id", "team_code"])
pp = scan("basic_player_personal", ["match_key", "round", "player_id", "steam_id", "is_bot"])
su = scan("basic_score_update", ["match_key", "round", "tick", "team_code", "old_score", "new_score"])
re_ = scan("basic_round_end", ["match_key", "round", "tick", "winner_team_code"])
h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"),
                    columns=["key", "match_id", "match_date", "max_rounds", "is_wingman",
                             "t_starters_score_final", "ct_starters_score_final"])
hdr = {r[0]: r for r in h.iter_rows()}
print("header rows:", h.height, "| distinct key:", len(hdr))
print("player_info rows:", pi.height, "| round_end rows:", re_.height, "| score_update rows:", su.height)

keys_round = set(re_.get_column("match_key").unique().to_list())
print("keys: header", len(hdr), "| round_end", len(keys_round),
      "| header&round_end", len(set(hdr) & keys_round))

# ---- player id -> steam_id (per match) ----
pmap_df = pp.group_by(["match_key", "player_id"]).agg(
    pl.col("steam_id").n_unique().alias("nsid"),
    pl.col("steam_id").first().alias("sid"),
    pl.col("is_bot").any().alias("bot"))
multi = pmap_df.filter(pl.col("nsid") > 1)
print("player_id -> >1 steam_id within match:", multi.height, multi.rows())
pmap = {(r[0], r[1]): (r[2], r[3]) for r in pmap_df.select(["match_key", "player_id", "sid", "bot"]).iter_rows()}

# ---- side per (match, round, steam_id) from player_info ----
side = collections.defaultdict(dict)
side_conf = 0
unmapped = 0
tc_other = collections.Counter()
for mk, r, pid, tc in pi.iter_rows():
    if tc not in (2, 3):
        tc_other[tc] += 1
        continue
    m = pmap.get((mk, pid))
    if m is None:
        unmapped += 1
        continue
    sid = m[0]
    d = side[(mk, r)]
    if sid in d and d[sid] != tc:
        side_conf += 1
    d[sid] = tc
print("player_info team_code not 2/3 (ignored):", dict(tc_other), "| unmapped rows:", unmapped,
      "| per-round side conflicts:", side_conf)
rounds_of = collections.defaultdict(list)
for (mk, r) in side:
    rounds_of[mk].append(r)

# ---- round_end winners ----
win = collections.defaultdict(dict)
dup_end = 0
for mk, r, tick, w in re_.iter_rows():
    if r in win[mk]:
        dup_end += 1
    win[mk][r] = w
print("round_end duplicate (match,round):", dup_end)

# ---- score_update: last new_score per team_code, and per-round side increments ----
su_s = su.sort(["match_key", "round", "tick"])
eng_last = collections.defaultdict(dict)
round_inc = collections.defaultdict(set)
for mk, r, tick, tc, old, new in su_s.iter_rows():
    eng_last[mk][tc] = new
    if old is not None and new is not None and new > old:
        round_inc[(mk, r)].add(tc)

# ---- per-match identity analysis ----
rows = []
for mk in sorted(set(hdr) & keys_round):
    hk = hdr[mk]
    mr, wingman, hT, hC = hk[3], bool(hk[4]), hk[5], hk[6]
    need = 2 if wingman else 5
    rs = sorted(win[mk].keys())
    r0 = None
    for r in sorted(rounds_of.get(mk, [])):
        d = side[(mk, r)]
        if sum(1 for v in d.values() if v == 2) == need and sum(1 for v in d.values() if v == 3) == need:
            r0 = r
            break
    rec = dict(key=mk, day=mk.split("/")[3], mr=mr, wingman=wingman, n=len(rs), hT=hT, hC=hC,
               S=hT + hC, r0=r0, need=need)
    if r0 is None:
        rec.update(cls="no_full_roster_round", st_T=None, st_CT=None, seq="", swaps=[], flags=0, rw_mis=0)
        rows.append(rec)
        continue
    starterT = {sid for sid, v in side[(mk, r0)].items() if v == 2}
    seq = []
    swaps = []
    st_won = 0
    known = 0
    flags = 0
    prev = None
    rw_mis = 0
    for r in rs:
        d = side.get((mk, r), {})
        present = [d[s] for s in starterT if s in d]
        if len(present) < need:
            flags += 1
            seq.append("?")
            continue
        if len(set(present)) > 1:
            flags += 1
            seq.append("!")
            continue
        sT = present[0]
        seq.append("T" if sT == 2 else "C")
        if prev is not None and sT != prev:
            swaps.append(r)
        prev = sT
        w = win[mk][r]
        known += 1
        if w == sT:
            st_won += 1
        if (mk, r) in round_inc:
            inc = round_inc[(mk, r)]
            if len(inc) == 1 and list(inc)[0] != w:
                rw_mis += 1
    c2 = sum(1 for r in rs if win[mk][r] == 2)
    c3 = sum(1 for r in rs if win[mk][r] == 3)
    e2 = eng_last[mk].get(2)
    e3 = eng_last[mk].get(3)
    stT = st_won if known == len(rs) else None
    stCT = (known - st_won) if known == len(rs) else None
    if stT is None:
        cls = "incomplete"
    elif stT == hT and stCT == hC:
        cls = "header_matches_start_T"
    elif stT == hC and stCT == hT:
        cls = "header_swapped_vs_start_T"
    else:
        cls = "other_totals_differ"
    rec.update(cls=cls, st_T=stT, st_CT=stCT, seq="".join(seq), swaps=swaps, flags=flags, rw_mis=rw_mis,
               c2=c2, c3=c3, e2=e2, e3=e3,
               roundend_eq_engine=(c2 == e2 and c3 == e3),
               hdr_T_eq_engine_side2=(hT == e2), hdr_T_eq_engine_side3=(hT == e3))
    rows.append(rec)

print("\n=== SUMMARY over", len(rows), "matches in header & round_end ===")
print("classification (all):", dict(collections.Counter(r["cls"] for r in rows)))
for wg in (True, False):
    sub = [r for r in rows if r["wingman"] == wg]
    print("  wingman=%s n=%d:" % (wg, len(sub)), dict(collections.Counter(r["cls"] for r in sub)),
          "| mr:", dict(collections.Counter(r["mr"] for r in sub)))
swapped = [r for r in rows if r["cls"] == "header_swapped_vs_start_T"]
print("header_swapped_vs_start_T:", len(swapped), "| by day:", dict(collections.Counter(r["day"] for r in swapped)))
for r in swapped:
    print("   ", r["key"], "mr", r["mr"], "n", r["n"], "hdr(T,CT)=(%s,%s)" % (r["hT"], r["hC"]),
          "identity start-T=%s start-CT=%s" % (r["st_T"], r["st_CT"]))
other = [r for r in rows if r["cls"] in ("other_totals_differ", "incomplete", "no_full_roster_round")]
print("other (non-swap) classes:", len(other), dict(collections.Counter(r["cls"] for r in other)))
for r in other:
    if r["wingman"]:
        print("   WINGMAN", r["cls"], r["key"], "n", r["n"], "S", r["S"], "hdr(T,CT)=(%s,%s)" % (r["hT"], r["hC"]),
              "identity start-T=%s start-CT=%s" % (r["st_T"], r["st_CT"]), "flags", r.get("flags"))

# ---- claimed 10 detail ----
print("\n=== CLAIMED 10 DETAIL ===")
by_key = {r["key"]: r for r in rows}
for k in CLAIMED:
    r = by_key.get(k)
    if r is None:
        print("MISSING", k)
        continue
    print(k.split("/")[3], k.split("/")[4][:8], "mr", r["mr"], "wingman", r["wingman"], "n", r["n"],
          "hdr(T,CT)=(%s,%s)" % (r["hT"], r["hC"]),
          "identity start-T=%s start-CT=%s" % (r["st_T"], r["st_CT"]),
          "cls=%s" % r["cls"],
          "| start-T side per round=%s" % r["seq"],
          "| swap at rounds=%s" % r["swaps"],
          "| rw_mis=%s flags=%s" % (r.get("rw_mis"), r.get("flags")),
          "| engine final side2=%s side3=%s" % (r.get("e2"), r.get("e3")),
          "| roundend c2=%s c3=%s" % (r.get("c2"), r.get("c3")))

# ---- swap-rule check (observed swap points vs max_rounds//2) ----
print("\n=== OBSERVED SWAP POINTS vs EXPECTED (swap after round mr//2, i.e. first change at mr//2+1) ===")
exp_ok = collections.Counter()
exp_bad = []
for r in rows:
    if r["cls"] == "no_full_roster_round" or r["cls"] == "incomplete":
        continue
    e = r["mr"] // 2 + 1
    tag = "wingman" if r["wingman"] else "5v5"
    if len(r["swaps"]) == 0:
        exp_ok["%s no_swap_observed" % tag] += 1
    elif r["swaps"][0] == e:
        exp_ok["%s first_swap_at_expected(%d)" % (tag, e)] += 1
    else:
        exp_ok["%s first_swap_elsewhere" % tag] += 1
        if len(exp_bad) < 15:
            exp_bad.append((r["key"], r["mr"], r["swaps"]))
    exp_ok["%s n_swaps=%d" % (tag, len(r["swaps"]))] += 1
print(dict(exp_ok))
print("first-swap-elsewhere examples:", exp_bad)

# ---- engine vs round_end consistency ----
print("\n=== ENGINE vs ROUND_END ===")
print("round winner side vs score_update increment mismatches (total):", sum(r.get("rw_mis", 0) for r in rows))
print("matches with any round-winner mismatch:", sum(1 for r in rows if r.get("rw_mis", 0) > 0))
re_eq = [r for r in rows if "roundend_eq_engine" in r]
print("round_end side counts == engine final side scores:", sum(1 for r in re_eq if r["roundend_eq_engine"]), "of", len(re_eq))
print("header T == engine side2 final:", sum(1 for r in re_eq if r["hdr_T_eq_engine_side2"]),
      "| header T == engine side3 final:", sum(1 for r in re_eq if r["hdr_T_eq_engine_side3"]))
print("claimed-10: round_end counts == engine final:", sum(1 for k in CLAIMED if by_key.get(k, {}).get("roundend_eq_engine")), "of 10")
print("claimed-10: header T == engine side2 final:", sum(1 for k in CLAIMED if by_key.get(k, {}).get("hdr_T_eq_engine_side2")), "of 10")
print("claimed-10: header T == engine side3 final:", sum(1 for k in CLAIMED if by_key.get(k, {}).get("hdr_T_eq_engine_side3")), "of 10")
print("\nrows total:", len(rows))
