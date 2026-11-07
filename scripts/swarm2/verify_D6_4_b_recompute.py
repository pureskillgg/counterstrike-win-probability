"""D6-4 independent verifier, part B: recompute the headline D6-4 numbers from raw 2024 and 2026 tome channels.

Independent of the swarm/swarm2 D6 scripts: own loaders, own score-orientation logic, own bomb-outcome logic.
Only projected columns are read, one channel at a time.
ASSUMPTIONS (marked where used):
  A1 tick rate 64 ticks/s (ASSUMPTION; header tick_rate is checked in section B5).
  A2 round 13 = first post-swap round in 24-round matches (swap after round 12, per team-code and score evidence).
  A3 bomb outcome: bomb_defused -> CT (3), bomb_exploded -> T (2), as in the claim.
  A4 score winner of round r uses round_state round_freeze_end t_score/ct_score at r and r+1 (orientation rule below).
"""
import sys, glob, collections, statistics
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASES = {"2024": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
         "2026": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"}
T, CT = 2, 3
DAYS_HI = {f"{d:02d}" for d in range(12, 26)}


def load(base, ch, cols):
    if ch == "header":
        fd = base / "header"
    else:
        cands = sorted(base.glob(f"basic_{ch}.*"))
        assert len(cands) == 1, (ch, cands)
        fd = cands[0]
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    lazy = [pl.scan_parquet(f).select(cols) for f in files]
    return pl.concat(lazy, how="vertical_relaxed").collect()


def score_winner(scm, r):
    """A4: label transition r -> r+1. Returns (winner, label).
    label 'no' = sides kept (only one increment, same orientation), 'swap' = sides swapped,
    'both' / 'none' = unusable. ASSUMPTION: t_score/ct_score = score of the side currently on T/CT."""
    if r not in scm or (r + 1) not in scm:
        return None, "missing"
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


def freeze_scores(base):
    rs = load(base, "round_state", ["match_key", "round", "event_type", "tick", "t_score", "ct_score"])
    fe = rs.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
    dup = fe.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height
    fe = fe.unique(["match_key", "round"], keep="first")
    scm = collections.defaultdict(dict)
    for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
        scm[m][r] = (t, c)
    return scm, dup


print("== B1. 2024 round_end / bomb / score recompute (own implementation) ==")
B24 = BASES["2024"]
RE24 = load(B24, "round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count", "second"])
dupRE = RE24.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
print(f"2024 round_end rows: {RE24.height}; (match,round) pairs with >1 row: {dupRE.height} (keep=first used)")
RE24u = RE24.unique(["match_key", "round"], keep="first")
BS24 = load(B24, "bomb_state", ["match_key", "round", "event_type", "tick", "player_team_code"])
outs = (BS24.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
        .group_by(["match_key", "round"])
        .agg(pl.col("event_type").unique().alias("evs"), pl.col("tick").min().alias("btick")))
both_ev = outs.filter(pl.col("evs").list.len() > 1).height
print(f"bomb outcome (match,round) pairs: {outs.height}; with both defused and exploded: {both_ev}")
plant = (BS24.filter(pl.col("event_type") == "bomb_planted")
         .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("ptick")))
print(f"bomb_planted (match,round) pairs: {plant.height}")
outs = outs.with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(CT).otherwise(T).alias("bomb_side"))
J = (RE24u.filter(pl.col("player_count") == 20)
     .join(outs, on=["match_key", "round"], how="inner")
     .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"),
                   pl.col("match_key").str.extract(r"csds/2024/02/(\d\d)/", 1).alias("day")))
r13 = J.filter(pl.col("round") == 13)
print(f"round 13, player_count==20, with bomb outcome: n={r13.height}, mismatch={int(r13['mm'].sum())}")
d1 = r13.filter(pl.col("day").is_in(list(DAYS_HI)))
print(f"  days 12-25: n={d1.height}, mismatch={int(d1['mm'].sum())}  (claim: 605 / 434)")
d0 = r13.filter(~pl.col("day").is_in(list(DAYS_HI)))
print(f"  days 01-07: n={d0.height}, mismatch={int(d0['mm'].sum())}  (claim: 180 / 0)")
print("  round-13 rows by day:")
print(str(r13.group_by("day").agg(pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort("day")))
other = J.filter(pl.col("round") != 13)
print(f"non-13 bomb rows (player_count==20): n={other.height}, mismatch={int(other['mm'].sum())}")

scm24, dupfe = freeze_scores(B24)
print(f"2024 freeze_end: matches with scores={len(scm24)}; (match,round) with >1 freeze_end row={dupfe}")

lab13 = collections.Counter()
for m, s in scm24.items():
    if 13 in s and 14 in s:
        lab13[score_winner(s, 13)[1]] += 1
print(f"transition label census 13->14 (2024): {dict(lab13)}  (a 'swap' here would mean a swap after round 13)")

sw13 = {m: score_winner(s, 13) for m, s in scm24.items()}
r13 = r13.with_columns(
    pl.col("match_key").map_elements(lambda m: sw13.get(m, (None, "missing"))[0], return_dtype=pl.Int64).alias("score_w"),
    pl.col("match_key").map_elements(lambda m: sw13.get(m, (None, "missing"))[1], return_dtype=pl.Utf8).alias("score_lab"))
aff = r13.filter(pl.col("mm"))
ctl = r13.filter(~pl.col("mm"))
a_sc = aff.filter(pl.col("score_w").is_not_null())
print(f"affected (round_end != bomb) round 13: n={aff.height}; scorable n={a_sc.height}; "
      f"score==bomb {int((a_sc['score_w'] == a_sc['bomb_side']).sum())}; "
      f"score==round_end {int((a_sc['score_w'] == a_sc['winner_team_code']).sum())}")
c_sc = ctl.filter(pl.col("score_w").is_not_null())
print(f"control (round_end == bomb) round 13: n={ctl.height}; scorable n={c_sc.height}; "
      f"score==bomb {int((c_sc['score_w'] == c_sc['bomb_side']).sum())}; "
      f"score==round_end {int((c_sc['score_w'] == c_sc['winner_team_code']).sum())}")
print(f"affected score-label census (13->14 transition): {dict(collections.Counter(aff['score_lab'].to_list()))}")

print("\n== B2. Round 13 round_end timing vs bomb outcome (ASSUMPTION A1: 64 ticks/s) ==")
for name, sub in [("affected", aff), ("control", ctl)]:
    dt_ticks = [(tk - bt) for tk, bt in sub.select(["tick", "btick"]).iter_rows()]
    secs = [x / 64.0 for x in dt_ticks]
    print(f"{name}: n={len(dt_ticks)} median_s={statistics.median(secs):.2f} min_s={min(secs):.2f} max_s={max(secs):.2f} "
          f"exact_0={sum(1 for x in dt_ticks if x == 0)} round_end_after_bomb={sum(1 for x in dt_ticks if x > 0)} "
          f"round_end_before_bomb={sum(1 for x in dt_ticks if x < 0)}")
    print(f"   most common tick deltas (ticks, count): {collections.Counter(dt_ticks).most_common(5)}")
    sec_gap = [s - tk / 64.0 for s, tk in sub.select(["second", "tick"]).iter_rows() if s is not None]
    print(f"   round_end 'second' minus tick/64: median={statistics.median(sec_gap):.3f} min={min(sec_gap):.3f} max={max(sec_gap):.3f}")

print("\n== B3. round_end win_reason_code: affected vs control matches, rounds 12-14 ==")
RE24all = load(B24, "round_end", ["match_key", "round", "winner_team_code", "win_reason_code"]).unique(["match_key", "round"], keep="first")
aff_keys = set(aff["match_key"].to_list())
ctl_keys = set(ctl["match_key"].to_list())
rr = RE24all.with_columns(pl.col("match_key").is_in(list(aff_keys)).alias("aff"),
                          pl.col("match_key").is_in(list(ctl_keys)).alias("ctl"))
for rnd in [12, 13, 14]:
    s = rr.filter((pl.col("round") == rnd) & pl.col("aff"))
    print(f"  affected matches, round {rnd} reason codes: {dict(collections.Counter(s['win_reason_code'].to_list()))}")
s = rr.filter((pl.col("round") == 13) & pl.col("ctl"))
print(f"  control matches, round 13 reason codes: {dict(collections.Counter(s['win_reason_code'].to_list()))}")
allr = RE24all.filter(pl.col("round").is_between(1, 30))
g = allr["win_reason_code"].is_in([8, 9])
print(f"2024 all rounds (1-30): share of generic reason 8/9 = {g.mean():.4f} (n={allr.height})")
bomb_reason = collections.Counter()
bo_keys = set(zip(outs["match_key"].to_list(), outs["round"].to_list()))
for m, r, rc in RE24all.select(["match_key", "round", "win_reason_code"]).iter_rows():
    if (m, r) in bo_keys:
        bomb_reason[rc] += 1
print(f"reason codes on rounds that have a bomb outcome (all 2024): {dict(sorted(bomb_reason.items()))}")

print("\n== B4. Score field integrity: freeze_end t_score vs t_score_raw ==")
RSr = load(B24, "round_state", ["match_key", "round", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
RSr = RSr.filter(pl.col("event_type") == "round_freeze_end")
neq = RSr.filter((pl.col("t_score") != pl.col("t_score_raw")) | (pl.col("ct_score") != pl.col("ct_score_raw"))).height
print(f"2024 freeze_end rows: {RSr.height}; rows where t/ct_score != *_raw: {neq}")

print("\n== B5. Header build / tick rate (2024), affected vs control membership ==")
H24 = load(B24, "header", ["key", "match_date", "tick_rate", "build_num", "demo_version_name"])
hm = {r["key"]: r for r in H24.iter_rows(named=True)}
print(f"2024 headers: {H24.height}; tick_rate counts: {dict(collections.Counter(H24['tick_rate'].to_list()))}")
print(f"distinct build_num: {dict(collections.Counter(H24['build_num'].to_list()))}")
print(f"demo_version_name counts: {dict(collections.Counter(H24['demo_version_name'].to_list()).most_common(5))}")
byday = collections.defaultdict(collections.Counter)
for k, r in hm.items():
    byday[k.split("/")[3]][r["build_num"]] += 1
for d in sorted(byday):
    print(f"   day {d}: build_num counts {dict(byday[d])}")
print(f"affected matches by build: {dict(collections.Counter(hm[m]['build_num'] for m in aff_keys if m in hm))}; "
      f"control matches by build: {dict(collections.Counter(hm[m]['build_num'] for m in ctl_keys if m in hm))}")

print("\n== B6. 2026 clean control: score winner vs round_end winner (own implementation) ==")
B26 = BASES["2026"]
RE26 = load(B26, "round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
rew26 = {(m, r): w for m, r, w in RE26.iter_rows()}
scm26, _ = freeze_scores(B26)
c26 = collections.Counter()
for m, s in scm26.items():
    for r in s:
        w, lab = score_winner(s, r)
        wre = rew26.get((m, r))
        if w is None or wre is None:
            c26["unscorable"] += 1
        else:
            c26["agree" if w == wre else "disagree"] += 1
print(f"2026 score vs round_end (all rounds): {dict(c26)}  (claim: 25,417 agree / 0 disagree)")
