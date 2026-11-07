"""D6-4 independent verifier, part C: (C1) build-number and round-position structure of the round_end defect,
and (C2) an independent survivor arbiter for rounds with NO bomb plant (no bomb outcome to lean on).

C2 rule (ASSUMPTION C2-R: standard CS rules without a planted bomb):
  - a side with zero survivors at round end loses (other side wins);
  - if both sides still have survivors at round end, the round timed out and CT wins;
  - rounds with a bomb_planted event are skipped (a planted bomb is not ended by elimination in these rules).
  Survivors = roster from player_info (round r) minus player_death victims with tick <= round_end tick + TOL.
  TOL is 0 ticks by default; TOL = 128 and 429 ticks are reported as sensitivity (ASSUMPTION: 64 ticks/s).
  Roster must be 5 vs 5 for a round to be used (otherwise 'roster_unusable').
Score winner = round_freeze_end t_score/ct_score transition r -> r+1 with the same orientation rule as part B.
"""
import sys, glob, collections
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
B24 = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
B26 = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
T, CT = 2, 3


def load(base, ch, cols):
    if ch == "header":
        fd = base / "header"
    else:
        cands = sorted(base.glob(f"basic_{ch}.*"))
        assert len(cands) == 1, (ch, cands)
        fd = cands[0]
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.scan_parquet(f).select(cols) for f in files], how="vertical_relaxed").collect()


def score_winner(scm, r):
    if r not in scm or (r + 1) not in scm:
        return None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    no = (t1 - t0, c1 - c0) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if no and sw:
        return None
    if no:
        return T if (t1 - t0, c1 - c0) == (1, 0) else CT
    if sw:
        return T if (t1 - c0, c1 - t0) == (0, 1) else CT
    return None


def build_map(base):
    H = load(base, "header", ["key", "build_num"])
    return {k: b for k, b in H.iter_rows()}


def freeze_scores(base):
    rs = load(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"])
    fe = rs.filter(pl.col("event_type") == "round_freeze_end").sort("tick").unique(["match_key", "round"], keep="first")
    scm = collections.defaultdict(dict)
    for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
        scm[m][r] = (t, c)
    return scm


print("== C1. Build number and round-position structure (2024) ==")
bm = build_map(B24)
RE = load(B24, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
RE = RE.unique(["match_key", "round"], keep="first")
BS = load(B24, "bomb_state", ["match_key", "round", "event_type", "tick"])
outs = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("btick")))
outs = outs.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(CT).otherwise(T).alias("bomb_side"))
planted = set(zip(*[BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round"]).get_column(c).to_list() for c in ["match_key", "round"]]))
J = RE.filter(pl.col("player_count") == 20).join(outs, on=["match_key", "round"], how="inner")
J = J.with_columns(pl.col("match_key").map_elements(lambda m: bm.get(m), return_dtype=pl.Int64).alias("build"),
                   (pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
J = J.with_columns((((pl.col("tick") - pl.col("btick")) == 0)).alias("exact0"),
                   (((pl.col("tick") - pl.col("btick")) == 429)).alias("lag429"))
print("bomb-outcome rounds (player_count==20) by build and round bin:")
bins = {"r13": 13, "r25": 25}
for b in sorted({x for x in J["build"].to_list() if x is not None}):
    sub = J.filter(pl.col("build") == b)
    r13s = sub.filter(pl.col("round") == 13)
    oth = sub.filter(pl.col("round") != 13)
    print(f"  build {b}: round13 n={r13s.height} mismatch={int(r13s['mm'].sum())} | other rounds n={oth.height} mismatch={int(oth['mm'].sum())}"
          f" | exact0={int(sub['exact0'].sum())} lag429={int(sub['lag429'].sum())} of {sub.height}")
print("round_end reason codes by build, all rounds 1-30 (generic 8/9 share):")
allr = RE.filter(pl.col("round").is_between(1, 30)).with_columns(
    pl.col("match_key").map_elements(lambda m: bm.get(m), return_dtype=pl.Int64).alias("build"))
for b in sorted({x for x in allr["build"].to_list() if x is not None}):
    s = allr.filter(pl.col("build") == b)
    g = s["win_reason_code"].is_in([8, 9]).mean()
    print(f"  build {b}: n={s.height} share_8_9={g:.4f} reason counts={dict(collections.Counter(s['win_reason_code'].to_list()))}")

scm = freeze_scores(B24)
print("score winner vs round_end winner by build and round (2024, all matches with both):")
REk = {(m, r): (w, b) for m, r, w, b in RE.select(["match_key", "round", "winner_team_code", "build"]).with_columns(
    pl.col("match_key").map_elements(lambda m: bm.get(m), return_dtype=pl.Int64).alias("build")).iter_rows()} if False else None
rew = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
cnt = collections.defaultdict(collections.Counter)
for m, s in scm.items():
    b = bm.get(m)
    for r in s:
        w = score_winner(s, r)
        wre = rew.get((m, r))
        if w is None or wre is None:
            cnt[(b, r)]["unscorable"] += 1
        else:
            cnt[(b, r)]["agree" if w == wre else "disagree"] += 1
for r in [9, 12, 13, 14, 25, 28]:
    for b in [9842, 9961]:
        c = cnt[(b, r)]
        dn = c["disagree"]
        ag = c["agree"]
        rate = dn / (dn + ag) if (dn + ag) else float("nan")
        print(f"  round {r:>2} build {b}: agree={ag} disagree={dn} unscorable={c['unscorable']} rate_disagree={rate:.3f}")

print("\n== C2. Survivor arbiter for rounds with no bomb plant (2024) ==")
PI = load(B24, "player_info", ["match_key", "round", "player_id", "team_code"]).unique(["match_key", "round", "player_id"], keep="first")
roster = collections.defaultdict(lambda: {T: set(), CT: set()})
for m, r, p, t in PI.iter_rows():
    if t in (T, CT):
        roster[(m, r)][t].add(p)
PD = load(B24, "player_death", ["match_key", "round", "tick", "player_id", "player_team_code"])
PD = PD.unique(["match_key", "round", "player_id"], keep="first")
deaths = collections.defaultdict(list)
for m, r, tk, p, t in PD.iter_rows():
    deaths[(m, r)].append((tk, p, t))
print(f"roster rounds: {len(roster)}; rounds with death rows: {len(deaths)}; player_death rows (dedup): {PD.height}")

# which matches have a first post-swap at 12 (score label at 12 is 'swap'): use score transition, not the claim's label
swap12 = set()
for m, s in scm.items():
    if 12 in s and 13 in s:
        t0, c0 = s[12]
        t1, c1 = s[13]
        if (t1 - c0, c1 - t0) in [(1, 0), (0, 1)] and (t1 - t0, c1 - c0) not in [(1, 0), (0, 1)]:
            swap12.add(m)
print(f"matches with unambiguous score swap at 12: {len(swap12)}")

REd = {(m, r): (tk, w, p) for m, r, tk, w, p in RE.select(["match_key", "round", "tick", "winner_team_code", "player_count"]).with_columns(pl.lit(None).alias("x")).select(["match_key", "round", "tick", "winner_team_code", "player_count"]).iter_rows()}
planted_set = planted


def arbiter(m, r, tol):
    key = (m, r)
    if key not in REd or key in planted_set:
        return None, "skip_plant_or_missing"
    tk = REd[key][0]
    ro = roster.get(key)
    if ro is None or len(ro[T]) != 5 or len(ro[CT]) != 5:
        return None, "roster_unusable"
    dead = {T: set(), CT: set()}
    for dtk, p, t in deaths.get(key, []):
        if dtk <= tk + tol and t in (T, CT):
            dead[t].add(p)
    alive_T = len(ro[T] - dead[T])
    alive_CT = len(ro[CT] - dead[CT])
    if alive_T == 0 and alive_CT == 0:
        return None, "both_zero"
    if alive_T == 0:
        return CT, "T_wiped"
    if alive_CT == 0:
        return T, "CT_wiped"
    return CT, "timeout_no_plant"


for tol in [0, 128, 429]:
    print(f"\n-- tolerance {tol} ticks ({tol/64:.2f} s, ASSUMPTION 64 ticks/s) --")
    res = collections.Counter()
    by = collections.defaultdict(collections.Counter)
    for m, s in scm.items():
        b = bm.get(m)
        for r in [12, 13, 14]:
            if (m, r) not in REd:
                continue
            arb, why = arbiter(m, r, tol)
            if arb is None:
                by[(r, b)]["no_arb"] += 1
                continue
            sc = score_winner(s, r)
            wre = rew.get((m, r))
            key_s = "score_na" if sc is None else ("score=arb" if sc == arb else "score!=arb")
            key_r = "rE=arb" if wre == arb else "rE!=arb"
            by[(r, b)][f"{key_r}|{key_s}"] += 1
            by[(r, b)][why] += 1
    for (r, b) in sorted(by, key=lambda x: (x[0], x[1] or 0)):
        print(f"  round {r} build {b}: " + ", ".join(f"{k}={v}" for k, v in sorted(by[(r, b)].items())))

print("\n-- same arbiter, tolerance 128, all 2024 rounds 1-28 (validity: rE should match arbiter where the round is clean) --")
val = collections.defaultdict(collections.Counter)
for m, s in scm.items():
    b = bm.get(m)
    for r in range(1, 29):
        if (m, r) not in REd:
            continue
        arb, why = arbiter(m, r, 128)
        if arb is None:
            continue
        wre = rew.get((m, r))
        sc = score_winner(s, r)
        val[(b, r)]["rE=arb" if wre == arb else "rE!=arb"] += 1
        if sc is not None:
            val[(b, r)]["score=arb" if sc == arb else "score!=arb"] += 1
for b in [9842, 9961]:
    tot = collections.Counter()
    for r in range(1, 29):
        tot.update(val[(b, r)])
    print(f"  build {b} all rounds 1-28: {dict(tot)}")
for r in [9, 12, 13, 14, 28]:
    for b in [9842, 9961]:
        print(f"  round {r} build {b}: {dict(val[(b, r)])}")
