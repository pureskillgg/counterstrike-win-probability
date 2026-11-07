# verify_A2_4 step 02: (A) missing round labels vs header total per build; (B) round-1 channel coverage;
# (C) round-1 score checks (9842 official-end state vs freeze_end(2); 9961 recovery rule vs round_end(1));
# (D) score-delta vs round_end winner by label k, per build. Read-only, 2024 tome only.
# ASSUMPTION A2: header t_starters_score_final + ct_starters_score_final = number of rounds in a complete match.
import os, json, collections
import polars as pl

TOME = r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds"


def chan_frame(name, cols):
    d = [x for x in os.listdir(TOME) if x.split(".", 1)[0] == "basic_" + name]
    assert len(d) == 1, (name, d)
    dp = os.path.join(TOME, d[0])
    meta = json.load(open(os.path.join(dp, "tome"), encoding="utf8"))
    parts = [pl.scan_parquet(os.path.join(dp, p["dataframe"]["dataframe"])).select(cols).collect()
             for p in meta["pages"]]
    return pl.concat(parts, how="vertical_relaxed") if len(parts) > 1 else parts[0]


hdir = os.path.join(TOME, "header")
hm = json.load(open(os.path.join(hdir, "tome"), encoding="utf8"))
H = pl.concat([pl.read_parquet(os.path.join(hdir, p["dataframe"]["dataframe"]))
               .select(["key", "build_num", "t_starters_score_final", "ct_starters_score_final"])
               for p in hm["pages"]], how="vertical_relaxed")
BUILD = dict(zip(H["key"].to_list(), H["build_num"].to_list()))
HSUM = {k: (t or 0) + (c or 0) for k, t, c in zip(H["key"].to_list(), H["t_starters_score_final"].to_list(),
                                                  H["ct_starters_score_final"].to_list())}
BUILDS = sorted(set(BUILD.values()))
print("== 0. builds", {b: sum(1 for v in BUILD.values() if v == b) for b in BUILDS})

st = chan_frame("round_state", ["round", "tick", "event_type", "t_score", "ct_score", "match_key"])
re_ = chan_frame("round_end", ["round", "tick", "winner_team_code", "match_key"])
pi = chan_frame("player_info", ["match_key", "round", "player_id", "team_code"])
de = chan_frame("player_death", ["match_key", "round", "tick", "player_id"])
bs = chan_frame("bomb_state", ["match_key", "round", "tick", "event_type"])

fe = st.filter(pl.col("event_type") == "round_freeze_end").unique(["match_key", "round"], keep="first")
FE = {(m, r): (t, c) for m, r, t, c in zip(fe["match_key"].to_list(), fe["round"].to_list(),
                                          fe["t_score"].to_list(), fe["ct_score"].to_list())}
REU = re_.unique(["match_key", "round"], keep="first")
RE = {(m, r): w for m, r, w in zip(REU["match_key"].to_list(), REU["round"].to_list(),
                                  REU["winner_team_code"].to_list())}
R_LAB = collections.defaultdict(set)
F_LAB = collections.defaultdict(set)
for (m, r) in RE:
    R_LAB[m].add(r)
for (m, r) in FE:
    F_LAB[m].add(r)

# (A) missing labels vs header total
print("\n== A. labels expected 1..(header total) vs observed; per build")
for b in BUILDS:
    ms = [m for m, v in BUILD.items() if v == b]
    n = len(ms)
    lenmR = collections.Counter()
    exact1 = 0
    missR = collections.Counter()
    extR = collections.Counter()
    lenmF = collections.Counter()
    missF = collections.Counter()
    fnoR = collections.Counter()
    rnoF = collections.Counter()
    zero_total = 0
    for m in ms:
        tot = HSUM[m]
        if tot == 0:
            zero_total += 1
        exp = set(range(1, tot + 1))
        R = R_LAB.get(m, set())
        F = F_LAB.get(m, set())
        mR = sorted(exp - R)
        xR = sorted(R - exp)
        mF = sorted(exp - F)
        lenmR[len(mR)] += 1
        if mR == [1]:
            exact1 += 1
        for x in mR:
            missR[x] += 1
        for x in xR:
            extR[x] += 1
        lenmF[len(mF)] += 1
        for x in mF:
            missF[x] += 1
        for x in sorted(F - R):
            fnoR[x] += 1
        for x in sorted(R - F):
            rnoF[x] += 1
    print(f"  build {b}: n={n}; header total zero={zero_total}")
    print(f"     missing round_end labels per match (count of missing: matches): {sorted(lenmR.items())}")
    print(f"     matches whose only missing round_end label is 1: {exact1}")
    print(f"     missing round_end label values (label: count, top 20): {sorted(missR.items())[:20]}")
    print(f"     extra round_end labels above header total (label: count): {sorted(extR.items())[:12]}")
    print(f"     missing freeze_end labels per match: {sorted(lenmF.items())}")
    print(f"     missing freeze_end label values (label: count, top 20): {sorted(missF.items())[:20]}")
    print(f"     freeze_end without round_end (label: count): {sorted(fnoR.items())[:20]}")
    print(f"     round_end without freeze_end (label: count): {sorted(rnoF.items())[:20]}")

# (B) round-1 coverage by channel
print("\n== B. matches with any row at round 1 / round 2, by build and channel")
for nm, df in [("round_state", st), ("player_info", pi), ("player_death", de), ("bomb_state", bs), ("round_end", re_)]:
    for lab in (1, 2):
        s = df.filter(pl.col("round") == lab)["match_key"].unique().to_list()
        cnt = collections.Counter(BUILD.get(m) for m in s)
        print(f"  {nm:<13} round {lab}: matches by build {dict(sorted(cnt.items(), key=lambda x: str(x[0])))}")
print("  round_state round-1 event types by build:")
for b in BUILDS:
    keys_b = [m for m, v in BUILD.items() if v == b]
    ev = st.filter((pl.col("round") == 1) & pl.col("match_key").is_in(keys_b))
    print(f"    build {b}:", ev.group_by("event_type").len().sort("event_type").rows())
    print(f"    build {b} round-1 state rows per match (median):",
          float(ev.group_by("match_key").len()["len"].median()) if ev.height else None)

# (C) round-1 score checks
print("\n== C. round-1 score checks")
s1 = st.filter((pl.col("round") == 1) & (pl.col("event_type") == "round_officially_ended")).unique(["match_key"], keep="first")
ok9842 = collections.Counter()
for m, t, c in zip(s1["match_key"].to_list(), s1["t_score"].to_list(), s1["ct_score"].to_list()):
    if BUILD.get(m) != 9842:
        continue
    fe2 = FE.get((m, 2))
    if fe2 is None:
        ok9842["no fe2"] += 1
        continue
    ok9842["equal fe2" if (t, c) == fe2 else "differs from fe2"] += 1
print("  9842: round_officially_ended(1) score vs freeze_end(2) score:", dict(ok9842))
# recovery rule on 9842: which side has the point at fe(2)
rec = collections.Counter()
for m, v in BUILD.items():
    if v != 9842:
        continue
    f2 = FE.get((m, 2))
    if f2 is None:
        rec["no fe2"] += 1
    elif f2 == (1, 0):
        rec["T(2)"] += 1
    elif f2 == (0, 1):
        rec["CT(3)"] += 1
    else:
        rec[f"other {f2}"] += 1
print("  9842 recovered round-1 winner from fe(2) (2=T, 3=CT):", dict(rec))
rule = collections.Counter()
for m, v in BUILD.items():
    if v != 9961:
        continue
    f1 = FE.get((m, 1))
    f2 = FE.get((m, 2))
    w = RE.get((m, 1))
    if f1 is None or f2 is None or w is None:
        rule["missing input"] += 1
        continue
    rule["fe1 is 0-0" if f1 == (0, 0) else "fe1 not 0-0"] += 1
    wr = 2 if f2 == (1, 0) else (3 if f2 == (0, 1) else 0)
    rule[f"rule {'agrees' if wr == w else 'disagrees'} with round_end(1)"] += 1
    rule[f"round_end(1) winner={w}"] += 1
print("  9961 recovery rule (fe2 point side) vs round_end(1):", dict(rule))

# (D) score-delta winner vs round_end winner by label k, per build
print("\n== D. score-delta winner (fe(k)->fe(k+1)) vs round_end(k) winner, by build and label k")
res = {b: collections.defaultdict(lambda: [0, 0, 0]) for b in BUILDS}
for (m, k), (t1, c1) in FE.items():
    b = BUILD.get(m)
    if (m, k + 1) not in FE:
        continue
    t2, c2 = FE[(m, k + 1)]
    w_end = RE.get((m, k))
    if w_end is None:
        continue
    dT, dC = (t2 or 0) - (t1 or 0), (c2 or 0) - (c1 or 0)
    w_sc = 2 if dT == 1 else (3 if dC == 1 else 0)
    if w_sc == 0:
        res[b][k][2] += 1
        continue
    res[b][k][0] += 1
    if w_sc != w_end:
        res[b][k][1] += 1
for b in BUILDS:
    print(f"  build {b} (k: n, bad, undetermined):", [(k, v[0], v[1], v[2]) for k, v in sorted(res[b].items()) if k <= 30])
print("DONE labels")
