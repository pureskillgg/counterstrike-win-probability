"""D6-4 independent verifier, part F: confound checks.
F1 build vs true match date (header match_date) for round-13 bomb rows: is the mismatch driven by build or by date?
F2 round 25 and rounds 24-28: survivor arbiter and bomb mismatch by build (round 25 is not covered by the claim).
F3 why the arbiter is unusable in 16-round matches: roster sizes per side in round 9.
F4 2026 header build and 2026 bomb-vs-round_end mismatch (claim: 0 of 7,427 rounds).
ASSUMPTIONS: as in part C (64 ticks/s, arbiter tolerance 128 ticks, no-plant rule C2-R).
"""
import sys, glob, collections
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
B24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
B26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T, CT = 2, 3
TOL = 128


def load(base, ch, cols):
    if ch == "header":
        fd = base / "header"
    else:
        fd = sorted(base.glob(f"basic_{ch}.*"))[0]
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f).select(cols) for f in files], how="vertical_relaxed").collect()


H = load(B24, "header", ["key", "build_num", "match_date"])
bm = {k: b for k, b, _ in H.iter_rows()}
mdate = {k: d[:10] for k, _, d in H.iter_rows()}
RE = load(B24, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
rek = {(m, r): (tk, w, pc) for m, r, tk, w, pc in RE.iter_rows()}
BS = load(B24, "bomb_state", ["match_key", "round", "event_type", "tick"])
outs = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("btick")))
outs = outs.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(CT).otherwise(T).alias("bomb_side"))
bside = {(m, r): (s, bt) for m, r, s, bt in outs.select(["match_key", "round", "bomb_side", "btick"]).iter_rows()}
planted = set(BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).iter_rows())

print("== F1. Round 13 bomb rows (player_count==20): mismatch by build and by true match date (header match_date) ==")
rows = collections.defaultdict(lambda: [0, 0])
for (m, r), (tk, w, pc) in rek.items():
    if r != 13 or pc != 20 or (m, r) not in bside:
        continue
    bs_, _ = bside[(m, r)]
    b = bm.get(m)
    d = mdate.get(m, "?")
    rows[(b, d)][0] += 1
    rows[(b, d)][1] += int(bs_ != w)
for k in sorted(rows, key=lambda x: (x[0] or 0, x[1])):
    n, mm = rows[k]
    if k[1] >= "2024-02-07" or True:
        print(f"  build {k[0]} header date {k[1]}: n={n} mismatch={mm}")
bd = collections.Counter()
for (b, d), (n, mm) in rows.items():
    bd[b] += mm
print(f"mismatch totals by build: {dict(bd)}")
aff_dates = collections.Counter(d for (b, d), (n, mm) in rows.items() if b == 9961 for _ in range(mm))
print(f"affected build-9961 header match_date range: {min(aff_dates)} .. {max(aff_dates)} (n={sum(aff_dates.values())})")
bdr = collections.defaultdict(list)
for m, b in bm.items():
    bdr[b].append(mdate.get(m, "?"))
for b in sorted(bdr, key=lambda x: x or 0):
    print(f"  all 2024 matches build {b}: n={len(bdr[b])} header match_date min={min(bdr[b])} max={max(bdr[b])}")
print(f"  build 9842 round-13 bomb rows with header date >= 2024-02-07: {sum(n for (b, d), (n, mm) in rows.items() if b == 9842 and d >= '2024-02-07')}")
print(f"  build 9961 round-13 bomb rows with header date < 2024-02-07: {sum(n for (b, d), (n, mm) in rows.items() if b == 9961 and d < '2024-02-07')}")
kd_eq = sum(1 for k, _, d in H.iter_rows() if k.split("/")[3] == d[8:10])
print(f"  key-folder day equals header match_date day for {kd_eq} of {H.height} 2024 matches")
rate = collections.defaultdict(lambda: [0, 0])
for (m, r), (tk, w, pc) in rek.items():
    if r != 13 or pc != 20 or (m, r) not in bside or bm.get(m) != 9961:
        continue
    rate[mdate.get(m, "?")][0] += 1
    rate[mdate.get(m, "?")][1] += int(bside[(m, r)][0] != w)
print("  within build 9961, round-13 bomb mismatch rate by true header date: " + ", ".join(f"{d[5:]}:{mm}/{n}" for d, (n, mm) in sorted(rate.items())))

print("\n== F2. Rounds 24-28 (2024): survivor arbiter vs round_end by build, and bomb mismatch by build ==")
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
    return CT


arb_c = collections.defaultdict(collections.Counter)
bomb_c = collections.defaultdict(collections.Counter)
for (m, r), (tk, w, pc) in rek.items():
    if r not in (24, 25, 26, 27, 28) or pc != 20:
        continue
    b = bm.get(m)
    a = arbiter(m, r)
    if a is not None:
        arb_c[(r, b)]["arb=rE" if a == w else "arb!=rE"] += 1
    if (m, r) in bside:
        bomb_c[(r, b)]["bomb=rE" if bside[(m, r)][0] == w else "bomb!=rE"] += 1
for r in [24, 25, 26, 27, 28]:
    for b in [9842, 9961]:
        print(f"  round {r} build {b}: arbiter {dict(arb_c[(r, b)])} | bomb-outcome {dict(bomb_c[(r, b)])}")

print("\n== F3. Roster sizes per side in round 9 by match format (why the arbiter is unusable in 16-round matches) ==")
fmt_swap = {}
RS = load(B24, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"])
fe = RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick").unique(["match_key", "round"], keep="first")
scm = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    scm[m][r] = (t, c)
for m, s in scm.items():
    if 8 in s and 9 in s:
        t0, c0 = s[8]
        t1, c1 = s[9]
        if (t1 - c0, c1 - t0) in [(1, 0), (0, 1)] and (t1 - t0, c1 - c0) not in [(1, 0), (0, 1)]:
            fmt_swap[m] = 8
        else:
            fmt_swap.setdefault(m, 0)
sizes = collections.Counter()
for m in list(scm.keys())[:4000]:
    if fmt_swap.get(m) == 8 and (m, 9) in rek:
        ro = roster.get((m, 9))
        sizes[("f16" if True else "", len(ro[T]) if ro else -1, len(ro[CT]) if ro else -1)] += 1
print(f"  16-round (swap after 8) round-9 roster sizes (T, CT) counts: {dict(sizes.most_common(8))}")
sizes24 = collections.Counter()
for m in list(scm.keys())[:4000]:
    if m in fmt_swap and fmt_swap[m] == 0 and (m, 13) in rek:
        ro = roster.get((m, 13))
        sizes24[(len(ro[T]) if ro else -1, len(ro[CT]) if ro else -1)] += 1
print(f"  other rounds (round 13) roster sizes (T, CT) counts: {dict(sizes24.most_common(6))}")

print("\n== F4. 2026 build and 2026 bomb-vs-round_end mismatch (claim: 0 of 7,427 rounds) ==")
H26 = load(B26, "header", ["key", "build_num"])
print(f"  2026 build_num counts: {dict(collections.Counter(H26['build_num'].to_list()))}")
RE26 = load(B26, "round_end", ["match_key", "round", "winner_team_code", "player_count"]).unique(["match_key", "round"], keep="first")
BS26 = load(B26, "bomb_state", ["match_key", "round", "event_type"])
o26 = (BS26.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
       .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
o26 = o26.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(CT).otherwise(T).alias("bomb_side"))
J26 = RE26.filter(pl.col("player_count") == 20).join(o26, on=["match_key", "round"], how="inner")
print(f"  2026 bomb-outcome rounds (player_count==20): n={J26.height}; mismatch={int((J26['bomb_side'] != J26['winner_team_code']).sum())}")
r13_26 = J26.filter(pl.col("round") == 13)
print(f"  2026 round 13 bomb rows: n={r13_26.height}; mismatch={int((r13_26['bomb_side'] != r13_26['winner_team_code']).sum())}")
