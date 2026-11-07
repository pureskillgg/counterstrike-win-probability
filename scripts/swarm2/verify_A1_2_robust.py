# -*- coding: utf-8 -*-
"""
verify_A1_2_robust.py  (robustness of the identity result for the claimed matches)

(a) Recompute starter-T totals with a second, independent side source: the engine side on spawn/death
    events (player_team_code, mode per player per round), instead of basic_player_info.team_code.
    Where the event side is missing for a player-round, fall back to player_info.
(b) Re-run the starter-T identity allowing a partially-observed starter group (any present member fixes
    the group's side, since teammates share a side).
(c) Per-day distribution of wingman matches and their classification; relation of header T to the engine
    final side scores (side 2 = T at end, side 3 = CT at end).
(d) Details of the player_info vs event disagreements inside the claimed matches.
"""
import os, glob, collections
import polars as pl

T = r"G:\pureskill\csds-cs2\tomes\2026-10-01_to_08\tome\csds"
CLAIMED = [
    "csds/2026/10/01/7368f3ae-7353-4f68-8d04-7c97f22ea2b8/csds",
    "csds/2026/10/01/ec8a424b-067f-4209-a42c-36d5a65cd97f/csds",
    "csds/2026/10/02/5429f41b-7721-4232-9784-bd9a3e2045a5/csds",
    "csds/2026/10/02/5f014ad2-63f8-415d-9952-5984b2924721/csds",
    "csds/2026/10/02/83d3ede2-5226-4035-8f32-87a32f083665/csds",
    "csds/2026/10/02/b2e50666-9b0b-4dcd-b8cc-59703e6a3064/csds",
    "csds/2026/10/02/d69b442a-8c83-4700-8a25-a035082e0ba8/csds",
    "csds/2026/10/02/e60396ec-5ff6-4a18-a2c9-b7187b650485/csds",
    "csds/2026/10/03/2093742d-edc0-4041-961c-a261f4f26876/csds",
    "csds/2026/10/03/cbb071c2-83a9-4167-b4c1-6fed1e145da0/csds",
]
claimed = set(CLAIMED)


def chan(name):
    folder = [d for d in os.listdir(T) if d.startswith(name + ".")][0]
    return sorted(glob.glob(os.path.join(T, folder, "dataframe_*")))


def scan(name, cols):
    return pl.scan_parquet(chan(name)).select(cols).collect()


h = pl.read_parquet(os.path.join(T, "header", "dataframe_00000"),
                    columns=["key", "max_rounds", "is_wingman", "t_starters_score_final", "ct_starters_score_final"])
hdr = {r[0]: r for r in h.iter_rows()}
pi = scan("basic_player_info", ["match_key", "round", "player_id", "team_code"])
pp = scan("basic_player_personal", ["match_key", "round", "player_id", "steam_id"])
re_ = scan("basic_round_end", ["match_key", "round", "winner_team_code"])
su = scan("basic_score_update", ["match_key", "round", "tick", "team_code", "new_score"])
sp = scan("basic_player_spawn", ["match_key", "round", "player_id", "player_team_code"])
dt = scan("basic_player_death", ["match_key", "round", "player_id", "player_team_code"])

pmap_df = pp.group_by(["match_key", "player_id"]).agg(pl.col("steam_id").first().alias("sid"))
pmap = {(r[0], r[1]): r[2] for r in pmap_df.iter_rows()}
pinfo_side = {}
for mk, r, pid, tc in pi.iter_rows():
    if tc in (2, 3) and (mk, pid) in pmap:
        pinfo_side[(mk, r, pmap[(mk, pid)])] = tc

# engine event side per (match, round, player): mode of spawn/death player_team_code
ev = pl.concat([sp, dt], how="vertical").filter(pl.col("player_team_code").is_in([2, 3]))
ev_cnt = ev.group_by(["match_key", "round", "player_id", "player_team_code"]).agg(pl.len().alias("n"))
ev_side = {}
ev_conflict = 0
best = {}
for mk, r, pid, tc, n in ev_cnt.iter_rows():
    k = (mk, r, pid)
    if k not in best or n > best[k][1]:
        if k in best and best[k][0] != tc:
            ev_conflict += 1
        best[k] = (tc, n)
for (mk, r, pid), (tc, n) in best.items():
    if (mk, pid) in pmap:
        ev_side[(mk, r, pmap[(mk, pid)])] = tc
print("event-derived (match,round,player) sides:", len(ev_side), "| player_info sides:", len(pinfo_side))

# agreement between engine-event side and player_info side, per player-round
agree = dis = 0
dis_by_match = collections.Counter()
for k, tc in ev_side.items():
    if k in pinfo_side:
        if pinfo_side[k] == tc:
            agree += 1
        else:
            dis += 1
            dis_by_match[k[0]] += 1
print("event vs player_info agree:", agree, "| disagree:", dis)
print("claimed disagreement counts:", {k.split('/')[3] + '/' + k.split('/')[4][:8]: dis_by_match.get(k, 0) for k in CLAIMED})

# round winners and score updates
win = collections.defaultdict(dict)
for mk, r, w in re_.iter_rows():
    win[mk][r] = w
su_s = su.sort(["match_key", "round", "tick"])
eng_last = collections.defaultdict(dict)
for mk, r, tick, tc, ns in su_s.iter_rows():
    eng_last[mk][tc] = ns
rounds_of = collections.defaultdict(set)
for (mk, r, sid) in pinfo_side:
    rounds_of[mk].add(r)
for (mk, r, sid) in ev_side:
    rounds_of[mk].add(r)


def side_of(mk, r, sid, src):
    if src == "pinfo":
        return pinfo_side.get((mk, r, sid))
    v = ev_side.get((mk, r, sid))
    return v if v is not None else pinfo_side.get((mk, r, sid))


members_pinfo = collections.defaultdict(lambda: collections.defaultdict(set))
for (m, r, sid) in pinfo_side:
    members_pinfo[m][r].add(sid)
members_ev = collections.defaultdict(lambda: collections.defaultdict(set))
for (m, r, sid) in ev_side:
    members_ev[m][r].add(sid)


def analyse(mk, src, partial_ok):
    hk = hdr[mk]
    wingman = bool(hk[2])
    need = 2 if wingman else 5
    members_at = (members_pinfo if src == "pinfo" else members_ev)[mk]  # round -> sids
    # starter group: side-2 members in first round with need per side (player_info based for membership)
    starterT = None
    for r in sorted(rounds_of[mk]):
        ids = [s for s in members_at[r]]
        s2 = [s for s in ids if side_of(mk, r, s, "pinfo") == 2]
        s3 = [s for s in ids if side_of(mk, r, s, "pinfo") == 3]
        if len(s2) == need and len(s3) == need:
            starterT = set(s2)
            break
    if starterT is None:
        return None
    st_won = known = 0
    seq = []
    prev = None
    swaps = []
    for r in sorted(win[mk].keys()):
        vals = [side_of(mk, r, s, src) for s in starterT]
        vals = [v for v in vals if v is not None]
        if (not partial_ok and len(vals) < need) or len(vals) == 0:
            seq.append("?")
            continue
        if len(set(vals)) > 1:
            seq.append("!")
            continue
        sT = vals[0]
        seq.append("T" if sT == 2 else "C")
        if prev is not None and sT != prev:
            swaps.append(r)
        prev = sT
        known += 1
        if win[mk][r] == sT:
            st_won += 1
    n = len(win[mk])
    return dict(n=n, known=known, stT=st_won, stCT=known - st_won, seq="".join(seq), swaps=swaps, wingman=wingman)


print("\n== (a)+(b) identity totals: source x partial rule ==")
results = {}
all_keys = sorted(set(hdr) & set(win.keys()))
for src in ("pinfo", "event"):
    for partial in (False, True):
        cls = collections.Counter()
        swapped_keys = []
        for mk in all_keys:
            a = analyse(mk, src, partial)
            if a is None:
                cls["no_roster"] += 1
                continue
            hT, hC = hdr[mk][3], hdr[mk][4]
            full = a["known"] == a["n"]
            if not full:
                cls["partial_rounds"] += 1
                continue
            if a["stT"] == hT and a["stCT"] == hC:
                cls["match_start_T"] += 1
            elif a["stT"] == hC and a["stCT"] == hT:
                cls["swapped"] += 1
                swapped_keys.append(mk)
            else:
                cls["other"] += 1
            if mk in claimed:
                results[(src, partial, mk)] = a
        print("source=%s partial_ok=%s:" % (src, partial), dict(cls))
        print("   swapped keys (complete rounds):", [k.split('/')[3] + '/' + k.split('/')[4][:8] for k in swapped_keys])

print("\n== claimed-10 detail (source=pinfo, partial_ok=True) and (source=event, partial_ok=True) ==")
for k in CLAIMED:
    row = []
    for src in ("pinfo", "event"):
        a = analyse(k, src, True)
        if a is None:
            row.append("no_roster")
            continue
        hT, hC = hdr[k][3], hdr[k][4]
        row.append("stT=%s stCT=%s known=%s/%s seq=%s swaps=%s" % (a["stT"], a["stCT"], a["known"], a["n"], a["seq"], a["swaps"]))
    print(k.split("/")[3], k.split("/")[4][:8], "hdr(T,CT)=(%s,%s)" % (hdr[k][3], hdr[k][4]), "| pinfo:", row[0], "| event:", row[1])

print("\n== (c) header T vs engine final side scores, wingman matches by classification ==")
tab = collections.Counter()
per_day = collections.defaultdict(collections.Counter)
for mk in all_keys:
    if not bool(hdr[mk][2]):
        continue
    a = analyse(mk, "pinfo", False)
    e2 = eng_last[mk].get(2)
    e3 = eng_last[mk].get(3)
    hT, hC = hdr[mk][3], hdr[mk][4]
    if a is None:
        c = "no_roster"
    elif a["known"] != a["n"]:
        c = "partial"
    elif a["stT"] == hT and a["stCT"] == hC:
        c = "match_start_T"
    elif a["stT"] == hC and a["stCT"] == hT:
        c = "swapped"
    else:
        c = "other"
    tab[(c, "hdrT==eng2" if hT == e2 else "", "hdrT==eng3" if hT == e3 else "")] += 1
    day = mk.split("/")[3]
    per_day[day][c] += 1
for k, v in sorted(tab.items(), key=lambda x: str(x)):
    print("  ", k, v)
print("wingman per day (classification):")
for d in sorted(per_day):
    print("  day", d, dict(per_day[d]))

print("\n== (d) claimed matches: which rounds have player_info vs event disagreement ==")
for k in CLAIMED:
    bad_rounds = collections.Counter()
    for (mk, r, sid), tc in ev_side.items():
        if mk == k and (mk, r, sid) in pinfo_side and pinfo_side[(mk, r, sid)] != tc:
            bad_rounds[r] += 1
    print(k.split("/")[3], k.split("/")[4][:8], "disagree by round:", dict(sorted(bad_rounds.items())))
