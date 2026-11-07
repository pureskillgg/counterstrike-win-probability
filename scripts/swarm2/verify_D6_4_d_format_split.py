"""D6-4 independent verifier, part D: split the first-post-swap disagreement by build (9842 vs 9961), match format
(16-round: first score swap after round 8; 24-round: first score swap after round 12), and bomb-plant status.
Uses the same survivor arbiter as part C (tolerance 128 ticks, ASSUMPTION 64 ticks/s), the same score rule, and round_end.
Reports where round_end, score and arbiter agree or disagree. Also a raw-E: spot check is in part E (separate script).
"""
import sys, glob, collections
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
B24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
T, CT = 2, 3
TOL = 128


def load(base, ch, cols):
    if ch == "header":
        fd = base / "header"
    else:
        cands = sorted(base.glob(f"basic_{ch}.*"))
        assert len(cands) == 1, (ch, cands)
        fd = cands[0]
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f).select(cols) for f in files], how="vertical_relaxed").collect()


def label(scm, r):
    if r not in scm or (r + 1) not in scm:
        return None, None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    no = (t1 - t0, c1 - c0) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if no and sw:
        return None, "both"
    if no:
        return (T if (t1 - t0, c1 - c0) == (1, 0) else CT), "no"
    if sw:
        return (T if (t1 - c0, c1 - t0) == (0, 1) else CT), "swap"
    return None, "none"


H = load(B24, "header", ["key", "build_num"])
bm = {k: b for k, b in H.iter_rows()}

RS = load(B24, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"])
fe = RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick").unique(["match_key", "round"], keep="first")
scm = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    scm[m][r] = (t, c)

RE = load(B24, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
rek = {(m, r): (tk, w, pc) for m, r, tk, w, pc in RE.iter_rows()}

BS = load(B24, "bomb_state", ["match_key", "round", "event_type"])
planted = set()
for m, r in BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).iter_rows():
    planted.add((m, r))

PI = load(B24, "player_info", ["match_key", "round", "player_id", "team_code"]).unique(["match_key", "round", "player_id"], keep="first")
roster = collections.defaultdict(lambda: {T: set(), CT: set()})
for m, r, p, t in PI.iter_rows():
    if t in (T, CT):
        roster[(m, r)][t].add(p)
PD = load(B24, "player_death", ["match_key", "round", "tick", "player_id", "player_team_code"]).unique(["match_key", "round", "player_id"], keep="first")
deaths = collections.defaultdict(list)
for m, r, tk, p, t in PD.iter_rows():
    deaths[(m, r)].append((tk, p, t))


def arbiter(m, r):
    key = (m, r)
    if key not in rek or key in planted:
        return None
    tk = rek[key][0]
    ro = roster.get(key)
    if ro is None or len(ro[T]) != 5 or len(ro[CT]) != 5:
        return None
    dead = {T: set(), CT: set()}
    for dtk, p, t in deaths.get(key, []):
        if dtk <= tk + TOL and t in (T, CT):
            dead[t].add(p)
    aT = len(ro[T] - dead[T])
    aC = len(ro[CT] - dead[CT])
    if aT == 0 and aC == 0:
        return None
    if aT == 0:
        return CT
    if aC == 0:
        return T
    return CT  # no plant, time expired


# format per match from unambiguous score swap labels
fmt = {}
for m, s in scm.items():
    swaps = [r for r in sorted(s) if label(s, r)[1] == "swap"]
    if not swaps:
        fmt[m] = "none"
    elif swaps[0] == 8:
        fmt[m] = "f16"
    elif swaps[0] == 12:
        fmt[m] = "f24"
    else:
        fmt[m] = f"other{swaps[0]}"
print("== D1. Match format (from unambiguous score swap labels) by build ==")
fc = collections.Counter((bm.get(m), fmt[m]) for m in scm)
for k in sorted(fc, key=lambda x: (x[0] or 0, x[1])):
    print(f"  build {k[0]} format {k[1]}: matches={fc[k]}")

print("\n== D2. Rounds 8-10, 12-14, 27-28: score vs round_end vs survivor arbiter, by build and format ==")
rows = collections.defaultdict(collections.Counter)
for m, s in scm.items():
    b = bm.get(m)
    f = fmt[m]
    for r in [9, 12, 13, 14, 28]:
        if (m, r) not in rek:
            continue
        sc, _ = label(s, r)
        wre = rek[(m, r)][1]
        arb = arbiter(m, r)
        pl_flag = "planted" if (m, r) in planted else "noplant"
        key = (r, b, f, pl_flag)
        if sc is None:
            rows[key]["score_na"] += 1
        else:
            rows[key]["score=rE" if sc == wre else "score!=rE"] += 1
        if arb is not None:
            rows[key]["arb=rE" if arb == wre else "arb!=rE"] += 1
            if sc is not None:
                rows[key]["arb=score" if arb == sc else "arb!=score"] += 1
for k in sorted(rows, key=lambda x: (x[0], x[1] or 0, x[2], x[3])):
    if k[1] not in (9842, 9961):
        continue
    print(f"  r{k[0]:>2} build {k[1]} {k[2]:<5} {k[3]:<8}: " + ", ".join(f"{a}={c}" for a, c in sorted(rows[k].items())))

print("\n== D3. Round 9 (16-round first post-swap) in build 9961: where do the score-vs-round_end disagreements sit? ==")
dis = []
for m, s in scm.items():
    if bm.get(m) != 9961 or fmt[m] != "f16":
        continue
    sc, _ = label(s, 9)
    wre = rek.get((m, 9), (None, None, None))[1]
    if sc is None or wre is None:
        continue
    if sc != wre:
        dis.append((m, (m, 9) in planted, arbiter(m, 9)))
print(f"  disagreements: {len(dis)}; planted: {sum(1 for x in dis if x[1])}; no-plant: {sum(1 for x in dis if not x[1])}")
print(f"  among no-plant disagreements, arbiter result (None = unusable/unknown): {collections.Counter(x[2] for x in dis if not x[1])}")
print("  arbiter vs round_end among no-plant disagreements: "
      + str(collections.Counter(("arb=rE" if x[2] == rek[(x[0], 9)][1] else "arb!=rE") for x in dis if (not x[1]) and x[2] is not None)))
print("  arbiter vs score among no-plant disagreements: "
      + str(collections.Counter(("arb=score" if x[2] == label(scm[x[0]], 9)[0] else "arb!=score") for x in dis if (not x[1]) and x[2] is not None)))

print("\n== D4. Round 13 (24-round first post-swap) by build and planted status: score and arbiter vs round_end ==")
for b in [9842, 9961]:
    for pf in ["planted", "noplant"]:
        c = collections.Counter()
        for m, s in scm.items():
            if bm.get(m) != b or fmt[m] != "f24":
                continue
            sc, _ = label(s, 13)
            wre = rek.get((m, 13), (None, None, None))[1]
            isp = (m, 13) in planted
            if (pf == "planted") != isp:
                continue
            if sc is None or wre is None:
                c["score_na"] += 1
            else:
                c["score=rE" if sc == wre else "score!=rE"] += 1
            arb = arbiter(m, 13)
            if arb is not None and wre is not None:
                c["arb=rE" if arb == wre else "arb!=rE"] += 1
                if sc is not None:
                    c["arb=score" if arb == sc else "arb!=score"] += 1
        print(f"  build {b} {pf:<8}: {dict(sorted(c.items()))}")

print("\n== D5. Round 28 (second post-swap, 24-round) by build: score and arbiter vs round_end (all planted states) ==")
for b in [9842, 9961]:
    c = collections.Counter()
    for m, s in scm.items():
        if bm.get(m) != b or fmt[m] != "f24":
            continue
        sc, _ = label(s, 28)
        wre = rek.get((m, 28), (None, None, None))[1]
        if sc is None or wre is None:
            c["score_na"] += 1
        else:
            c["score=rE" if sc == wre else "score!=rE"] += 1
        arb = arbiter(m, 28)
        if arb is not None and wre is not None:
            c["arb=rE" if arb == wre else "arb!=rE"] += 1
            if sc is not None:
                c["arb=score" if arb == sc else "arb!=score"] += 1
    print(f"  build {b}: {dict(sorted(c.items()))}")
