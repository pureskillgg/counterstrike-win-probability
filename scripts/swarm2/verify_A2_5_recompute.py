# verify_A2_5_recompute.py
# Independent verifier for finding A2-5 (2024-02-g tome: duplicate matches not removed).
# Read-only on the 2024-02-g tome. Writes only logs\swarm2\verify_A2_5.log.
# Recomputes: header-signature groups; round_end / cross_period-style content / strong
# fingerprints; the cross_period channel filter and dedupe (expects 214 dropped, 202 groups,
# 271 copies); the 4,559-round figure; and a coincidence / near-duplicate test.
import os, sys, json, glob, collections, hashlib, itertools, time
import polars as pl
import pyarrow.parquet as pq

BASE = r"G:\pureskill\csds-cs2"
TOME = os.path.join(BASE, "tomes", "2024-02-g", "tome", "csds")
SUF = ".2023-12-28,2024-02-25"
LOG_PATH = os.path.join(BASE, "logs", "swarm2", "verify_A2_5.log")
LOG = open(LOG_PATH, "w", encoding="utf-8")
T0 = time.time()
T_CODE, CT_CODE, START = 2, 3, 5


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def logonly(s):
    LOG.write(s + "\n")
    LOG.flush()


def fdir(ch):
    return os.path.join(TOME, "basic_" + ch + SUF)


def pages(ch):
    d = fdir(ch)
    meta = json.load(open(os.path.join(d, "tome"), encoding="utf-8"))
    from_meta = sorted(os.path.join(d, p["dataframe"]["dataframe"]) for p in meta["pages"])
    from_glob = sorted(glob.glob(os.path.join(d, "dataframe_*")))
    assert from_meta == from_glob, (ch, from_meta, from_glob)
    return from_glob


def read(ch, cols):
    frames = [pl.read_parquet(p, columns=cols) for p in pages(ch)]
    return pl.concat(frames, how="vertical_relaxed") if len(frames) > 1 else frames[0]


def keyset(ch):
    files = sorted(glob.glob(os.path.join(fdir(ch), "keyset_*")))
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


def sha(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def add_rows(acc, tag, df, cols):
    for row in df.select(["match_key"] + cols).iter_rows():
        acc[row[0]].append(tag + "\x1f" + "\x1f".join("~" if v is None else str(v) for v in row[1:]))


def digest_map(acc):
    return {m: sha("\x1e".join(sorted(v))) for m, v in acc.items()}


def groups(sig):
    g = collections.defaultdict(list)
    for k, s in sig.items():
        g[s].append(k)
    return sorted((sorted(v) for v in g.values() if len(v) > 1), key=lambda x: x[0])


def as_set(G):
    return {frozenset(g) for g in G}


def summarize(name, G):
    members = sum(len(g) for g in G)
    excess = members - len(G)
    sizes = collections.Counter(len(g) for g in G)
    pairs = sum(len(g) * (len(g) - 1) // 2 for g in G)
    spanning = sum(1 for g in G if len({DAY[k] for k in g}) > 1)
    out(f"[{name}] groups={len(G)} member_rows={members} excess_copies={excess} "
        f"excess_share_of_all_header_rows={excess / len(ALL):.4f} sizes={dict(sorted(sizes.items()))} "
        f"within_group_pairs={pairs} groups_spanning_folder_days={spanning}")
    return excess


def pair_identity(G, dig):
    tot = same = 0
    for g in G:
        for a, b in itertools.combinations(g, 2):
            tot += 1
            if dig.get(a) is not None and dig.get(a) == dig.get(b):
                same += 1
    return tot, same


# ---------------------------------------------------------------- header
HCOLS = ["key", "match_id", "match_date", "map_name", "t_starters_score_final",
         "ct_starters_score_final", "number_of_points", "build_num"]
hdr = pl.read_parquet(os.path.join(TOME, "header", "dataframe_00000"), columns=HCOLS)
ALL = hdr["key"].to_list()
ALLS = set(ALL)
DAY = {}
for r in hdr.iter_rows(named=True):
    parts = r["key"].split("/")
    assert len(parts) == 6 and parts[0] == "csds" and parts[4] == r["match_id"], r["key"]
    DAY[r["key"]] = parts[3]
MD = {r["key"]: r for r in hdr.iter_rows(named=True)}
out(f"header rows={len(ALL)} distinct key={len(ALLS)} distinct match_id={hdr['match_id'].n_unique()} "
    f"(key path id == match_id checked for every row)")
out(f"match_date is str: {hdr.schema['match_date']}")

out("== 1. header-signature groups (recomputed) ==")
sig_full = {k: (MD[k]["match_date"], MD[k]["map_name"], MD[k]["t_starters_score_final"],
                MD[k]["ct_starters_score_final"], MD[k]["number_of_points"]) for k in ALL}
G_hdr = groups(sig_full)
X_hdr = summarize("header sig: full match_date|map|scores|points", G_hdr)
sig_day = {k: (MD[k]["match_date"][:10],) + sig_full[k][1:] for k in ALL}
G_hdr_day = groups(sig_day)
summarize("header sig: date-only variant", G_hdr_day)
out("  header groups identical under full vs date-only key:", [sorted(g) for g in G_hdr] == [sorted(g) for g in G_hdr_day])

# ---------------------------------------------------------------- channels
out("== 2. channel reads (projection only) ==")
RE = read("round_end", ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message",
                        "legacy_code", "player_count", "second", "match_key"])
RST = read("round_start", ["round", "tick", "time_limit", "match_key"])
RS = read("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "match_key"])
DE = read("player_death", ["round", "tick", "player_id", "attacker_id", "weapon_name", "player_x_pos",
                           "player_y_pos", "player_z_pos", "player_team_code", "match_key"])
PI = read("player_info", ["round", "player_id", "team_code", "rank", "wins", "match_key"])
BS = read("bomb_state", ["round", "tick", "event_type", "player_id", "site_code", "match_key"])
DC = read("player_disconnect", ["round", "tick", "match_key"])
bt_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in pages("bot_takeover"))
out(f"round_end rows={RE.height} unique(match,round)={RE.select(['match_key','round']).unique().height} "
    f"matches_with_rows={RE['match_key'].n_unique()}")
out(f"round_start rows={RST.height} matches={RST['match_key'].n_unique()}; round_state rows={RS.height} "
    f"matches={RS['match_key'].n_unique()}; player_death rows={DE.height} matches={DE['match_key'].n_unique()}")
out(f"player_info rows={PI.height} matches={PI['match_key'].n_unique()}; bomb_state rows={BS.height} "
    f"matches={BS['match_key'].n_unique()}; player_disconnect rows={DC.height} matches={DC['match_key'].n_unique()}; "
    f"bot_takeover rows={bt_rows}")

acc_re = collections.defaultdict(list)   # all round_end columns only
acc_cp = collections.defaultdict(list)   # cross_period.signatures analogue
acc_st = collections.defaultdict(list)   # strong: many channels, many columns
add_rows(acc_re, "R", RE, ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message",
                           "legacy_code", "player_count", "second"])
add_rows(acc_cp, "R", RE, ["round", "tick", "winner_team_code"])
add_rows(acc_st, "R", RE, ["round", "tick", "winner_team_code", "win_reason_code", "win_reason_message",
                           "legacy_code", "player_count", "second"])
add_rows(acc_cp, "D", DE, ["round", "tick", "player_id"])
add_rows(acc_cp, "P", PI, ["round", "player_id", "team_code"])
add_rows(acc_st, "S", RS, ["round", "tick", "event_type", "t_score", "ct_score"])
add_rows(acc_st, "T", RST, ["round", "tick", "time_limit"])
add_rows(acc_st, "D", DE, ["round", "tick", "player_id", "attacker_id", "weapon_name", "player_x_pos",
                           "player_y_pos", "player_z_pos", "player_team_code"])
add_rows(acc_st, "P", PI, ["round", "player_id", "team_code", "rank", "wins"])
add_rows(acc_st, "B", BS, ["round", "tick", "event_type", "player_id", "site_code"])
add_rows(acc_st, "C", DC, ["round", "tick"])
dig_re = digest_map(acc_re)
dig_cp = digest_map(acc_cp)
dig_st = digest_map(acc_st)
out(f"matches with round_end digest={len(dig_re)}; with content digest={len(dig_cp)}; with strong digest={len(dig_st)}")
del acc_re, acc_cp, acc_st

# ---------------------------------------------------------------- fingerprint groups, all matches
out("== 3. fingerprint groups over all header matches ==")
G_re = groups({m: dig_re[m] for m in ALL if m in dig_re})
summarize("round_end fingerprint (all round_end cols)", G_re)
G_cp_all = groups({m: dig_cp[m] for m in ALL if m in dig_cp})
summarize("cross_period content sig (round_end+player_death+player_info)", G_cp_all)
G_st = groups({m: dig_st[m] for m in ALL if m in dig_st})
summarize("strong fingerprint (round_end+round_state+death+info+bomb+round_start)", G_st)
out("  header groups == round_end fp groups:", as_set(G_hdr) == as_set(G_re))
out("  header groups == content sig groups:", as_set(G_hdr) == as_set(G_cp_all))
out("  header groups == strong fp groups:", as_set(G_hdr) == as_set(G_st))
miss_st = as_set(G_hdr) - as_set(G_st)
extra_st = as_set(G_st) - as_set(G_hdr)
out(f"  header groups not reproduced by strong fp: {len(miss_st)}; strong-fp groups not in header groups: {len(extra_st)}")
for label, dg in [("round_end fp", dig_re), ("content sig", dig_cp), ("strong fp", dig_st)]:
    tot, same = pair_identity(G_hdr, dg)
    out(f"  within-header-group pairs: {tot}; identical {label}: {same}")

# partial copies: identical on round_end but not on the strong fingerprint
partial = 0
for g in G_hdr:
    for a, b in itertools.combinations(g, 2):
        if dig_re.get(a) == dig_re.get(b) and dig_st.get(a) != dig_st.get(b):
            partial += 1
out(f"  within-header-group pairs identical on round_end but different on strong fp (partial copies): {partial}")

# ---------------------------------------------------------------- cross_period channel filter replica
out("== 4. cross_period channel filter and dedupe replica ==")
DENSE = ["round_end", "round_start", "round_state", "player_death", "player_info", "bomb_state"]
SPARSE = ["player_disconnect", "bot_takeover"]
have = {"round_end": set(RE["match_key"].unique().to_list()),
        "round_start": set(RST["match_key"].unique().to_list()),
        "round_state": set(RS["match_key"].unique().to_list()),
        "player_death": set(DE["match_key"].unique().to_list()),
        "player_info": set(PI["match_key"].unique().to_list()),
        "bomb_state": set(BS["match_key"].unique().to_list()),
        "player_disconnect": set(DC["match_key"].unique().to_list()),
        "bot_takeover": set()}
dropped = set()
for ch in DENSE + SPARSE:
    ks = keyset(ch)
    no_ks = ALLS - ks
    zero = (ALLS - have[ch]) if ch in DENSE else set()
    dropped |= no_ks | zero
    out(f"  {ch:<18} keyset={len(ks)} equals_header={ks == ALLS} keyset_missing={len(no_ks)} "
        f"{'zero_row_dropped' if ch in DENSE else 'sparse_no_drop'}={len(zero)}")
kept = ALLS - dropped
out(f"  dropped matches={len(dropped)} (cross_period.md:55 says 214); channel-kept matches={len(kept)} (says 4221)")
G_cp_kept = groups({m: dig_cp[m] for m in kept if m in dig_cp})
X_cp_kept = summarize("cross_period content sig on channel-kept set", G_cp_kept)
G_hdr_kept = groups({m: sig_full[m] for m in kept})
summarize("header sig on channel-kept set", G_hdr_kept)
dropped_in_hdr = sum(1 for g in G_hdr for k in g if k in dropped)
excess_hdr_in_dropped = sum(1 for g in G_hdr for k in g[1:] if k in dropped)
out(f"  header-group members that are dropped by the channel filter: {dropped_in_hdr}; "
    f"header excess copies (non-first members) that are dropped: {excess_hdr_in_dropped}")
out(f"  header excess (all) = {X_hdr}; header excess after channel filter = "
    f"{sum(len(g) - 1 for g in G_hdr_kept)}; cp excess on kept = {X_cp_kept}")
drop_set_cp = {m for g in G_cp_kept for m in g[1:]}
out(f"  cp dup_drop size (kept copies removed)={len(drop_set_cp)}")

# survivors of the four relaxed round filters (replica of cross_period lines 243-265)
rd_keys = set()
for m, r in RE.select(["match_key", "round"]).iter_rows():
    if m in kept:
        rd_keys.add((m, int(r)))
players = collections.defaultdict(dict)
for m, r, pid, tc in PI.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
    if m in kept:
        players[(m, int(r))][pid] = tc
disc = {(m, int(r)) for m, r in DC.select(["match_key", "round"]).iter_rows() if m in kept}
rst_keys = {(m, int(r)) for m, r in RST.select(["match_key", "round"]).iter_rows() if m in kept}
live = {}
for m, r, t, tsc, ctsc, ev in RS.select(["match_key", "round", "tick", "t_score", "ct_score", "event_type"]).iter_rows():
    if ev == "round_freeze_end" and m in kept:
        k = (m, int(r))
        if k not in live:
            live[k] = {"tick": int(t), "t": tsc, "ct": ctsc}
rules_fail = collections.Counter()
surv = []
for k in sorted(rd_keys):
    pl_ = players.get(k, {})
    n_t = sum(1 for v in pl_.values() if v == T_CODE)
    n_ct = sum(1 for v in pl_.values() if v == CT_CODE)
    if not (n_t == START and n_ct == START):
        rules_fail["not 5T5CT"] += 1
        continue
    if k in disc:
        rules_fail["disconnect"] += 1
        continue
    if k not in rst_keys:
        rules_fail["no round_start"] += 1
        continue
    if k not in live:
        rules_fail["no live"] += 1
        continue
    if live[k]["t"] is None or live[k]["ct"] is None:
        rules_fail["null score"] += 1
        continue
    surv.append(k)
out(f"  rounds in channel-kept matches={len(rd_keys)} (cross_period says 80236); failures={dict(rules_fail)}; "
    f"after relaxed round filters={len(surv)}")
n_dup_rounds = sum(1 for k in surv if k[0] in drop_set_cp)
out(f"  rounds removed as duplicate copies (replica of cross_period 'duplicate copy' stage)={n_dup_rounds} "
    f"(cross_period.md:59 says 4559)")
out(f"  rounds after dedupe, before death-side filter={len(surv) - n_dup_rounds} (cross_period implies 55792)")

# raw round_end rows inside copies
nre = collections.Counter(RE["match_key"].to_list())
raw_hdr = sum(nre[k] for g in G_hdr for k in g[1:])
raw_cp_kept = sum(nre[m] for m in drop_set_cp)
out(f"  raw round_end rows in header excess copies={raw_hdr}; raw round_end rows in cp-removed copies={raw_cp_kept}")

# ---------------------------------------------------------------- coincidence / near-duplicate test
out("== 5. coincidence and near-duplicate test (round_end (round,tick) overlap) ==")
inv = collections.defaultdict(set)
for m, r, t in RE.select(["match_key", "round", "tick"]).iter_rows():
    inv[(int(r), int(t))].add(m)
shared_keys = sum(1 for v in inv.values() if len(v) > 1)
big_keys = sum(1 for v in inv.values() if len(v) > 300)
pc = collections.Counter()
for v in inv.values():
    if 1 < len(v) <= 300:
        for a, b in itertools.combinations(sorted(v), 2):
            pc[(a, b)] += 1
out(f"  (round,tick) keys: shared by >1 match={shared_keys}; shared by >300 matches (skipped)={big_keys}; "
    f"pairs with >=1 shared key={len(pc)}")
hdr_of = {}
for gi, g in enumerate(G_hdr):
    for k in g:
        hdr_of[k] = gi
same_pairs_full = same_pairs_part = same_pairs_none = 0
for g in G_hdr:
    for a, b in itertools.combinations(g, 2):
        c = pc.get((a, b), 0)
        mn = min(nre[a], nre[b])
        if c == mn:
            same_pairs_full += 1
        elif c > 0:
            same_pairs_part += 1
        else:
            same_pairs_none += 1
out(f"  same-header-group pairs: full round_end overlap={same_pairs_full}, partial={same_pairs_part}, none={same_pairs_none}")
nondup_ratio = []
near_nondup = []
for (a, b), c in pc.items():
    if hdr_of.get(a) is not None and hdr_of.get(a) == hdr_of.get(b):
        continue
    mn = min(nre[a], nre[b])
    if mn < 5:
        continue
    ratio = c / mn
    nondup_ratio.append(ratio)
    if ratio >= 0.5:
        near_nondup.append((a, b, c, mn))
nondup_ratio.sort(reverse=True)
out(f"  non-header-group pairs with >=5 rounds in both: {len(nondup_ratio)}; top overlap ratios: "
    f"{[round(x, 3) for x in nondup_ratio[:5]]}; pairs with ratio>=0.5: {len(near_nondup)}")
for a, b, c, mn in near_nondup[:20]:
    out(f"    near-dup not in same header group: {a.split('/')[4]} {b.split('/')[4]} shared={c} min_rounds={mn} "
        f"sig_a={sig_full[a]} sig_b={sig_full[b]} strong_equal={dig_st.get(a) == dig_st.get(b)}")

# ---------------------------------------------------------------- distinct counts and removed list
out("== 6. distinct-match counts implied by each method ==")
out(f"  header method: distinct={len(ALL) - X_hdr}; strong fp method: distinct={len(ALL) - sum(len(g) - 1 for g in G_st)}; "
    f"cp method on all header matches: distinct={len(ALL) - sum(len(g) - 1 for g in G_cp_all)}")
out(f"  cp method on channel-kept set: distinct kept after dedupe={len(kept) - X_cp_kept}")
logonly("== 7. cp-removed copies (removed match_id -> kept match_id, folder days) ==")
for g in G_cp_kept:
    for m in g[1:]:
        logonly(f"  REMOVED {m.split('/')[4]} day={DAY[m]} KEPT {g[0].split('/')[4]} day={DAY[g[0]]}")
out(f"  (full removed list written to log, {len(drop_set_cp)} lines)")
out(f"elapsed seconds={time.time() - T0:.1f}")
out("DONE")
LOG.close()
