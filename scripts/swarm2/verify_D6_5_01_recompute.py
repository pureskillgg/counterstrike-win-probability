"""D6-5 verifier step 1: recompute the claim's score-vs-round_end partition on the 2024 tome, from scratch.

Score-only winner for round r: arithmetic on round_freeze_end (t_score, ct_score) at r and r+1.
  no-swap pattern: (dT, dC) in {(1,0),(0,1)}          -> winner T if dT==1 else CT
  swap pattern:    (T1-C0, C1-T0) in {(1,0),(0,1)}    -> winner by the same rule on the swapped pair
  both or neither  -> unscorable (ties, gap-1 aligned, or inconsistent)
The claim's unscorable = score not scorable OR round_end row missing. This script splits those causes.
Run: G:\pureskill\csds-cs2\.pixi\envs\default\python.exe -I -X utf8 scripts\swarm2\verify_D6_5_01_recompute.py
"""
import collections
import glob
import pickle
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(200)
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T, CT = 2, 3
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")


def files(ch):
    if ch == "header":
        return sorted(glob.glob(str(BASE / "header" / "dataframe_*")))
    fd = sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read(ch, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files(ch)], how="vertical_relaxed")


HDR = read("header", ["key", "match_id", "match_date", "map_name", "t_starters_score_final", "ct_starters_score_final"])
print(f"header rows: {HDR.height}; distinct key: {HDR['key'].n_unique()}; distinct match_id: {HDR['match_id'].n_unique()}")

RS_all = read("round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
FE = RS_all.filter(pl.col("event_type") == "round_freeze_end")
dupfe = FE.group_by(["match_key", "round"]).agg(pl.len().alias("n"),
                                              pl.col("t_score").n_unique().alias("nt"),
                                              pl.col("ct_score").n_unique().alias("nc"))
print(f"freeze_end rows: {FE.height}; (match,round) keys: {dupfe.height}; keys with >1 row: {int((dupfe['n'] > 1).sum())}; "
      f"keys with conflicting scores: {int(((dupfe['nt'] > 1) | (dupfe['nc'] > 1)).sum())}")
FE = FE.unique(["match_key", "round"], keep="first")

RE_all = read("round_end", ["match_key", "round", "winner_team_code", "win_reason_code", "tick", "player_count"])
dupre = RE_all.group_by(["match_key", "round"]).agg(pl.len().alias("n"), pl.col("winner_team_code").n_unique().alias("nw"))
print(f"round_end rows: {RE_all.height}; (match,round) keys: {dupre.height}; keys with >1 row: {int((dupre['n'] > 1).sum())}; "
      f"keys with conflicting winner: {int((dupre['nw'] > 1).sum())}")
RE = RE_all.unique(["match_key", "round"], keep="first")
rew = {(m, r): w for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows()}
pc = {(m, r): p for m, r, p in RE.select(["match_key", "round", "player_count"]).iter_rows()}

sc = collections.defaultdict(dict)
for m, r, t, c in FE.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    sc[m][r] = (t, c)
print(f"matches with freeze_end: {len(sc)}; (match,round) freeze_end rows: {sum(len(v) for v in sc.values())}")


def score_only(scm, r):
    """Return (status, winner). status in no_next, both, inconsistent, scorable."""
    if r + 1 not in scm:
        return "no_next", None
    t0, c0 = scm[r]
    t1, c1 = scm[r + 1]
    dT, dC = t1 - t0, c1 - c0
    no = (dT, dC) in [(1, 0), (0, 1)]
    sw = (t1 - c0, c1 - t0) in [(1, 0), (0, 1)]
    if no and sw:
        return "both", None
    if no:
        return "scorable", (T if (dT, dC) == (1, 0) else CT)
    if sw:
        return "scorable", (T if (t1 - c0, c1 - t0) == (0, 1) else CT)
    return "inconsistent", None


status_count = collections.Counter()
claim_cat = collections.Counter()
per_round = collections.defaultdict(collections.Counter)
dis_rows = []
gap_amb = collections.Counter()
for m, scm in sc.items():
    for r in scm:
        st, w_sc = score_only(scm, r)
        status_count[st] += 1
        w_re = rew.get((m, r))
        if st == "both":
            t0, c0 = scm[r]
            gap_amb["gap0" if t0 == c0 else f"gap{abs(t0 - c0)}"] += 1
        if w_re is None:
            claim_cat["round_end_missing"] += 1
            per_round[r]["re_missing"] += 1
            continue
        if st != "scorable":
            claim_cat[f"unscorable:{st}"] += 1
            per_round[r][f"unscorable:{st}"] += 1
            continue
        if w_sc == w_re:
            claim_cat["agree"] += 1
            per_round[r]["agree"] += 1
        else:
            claim_cat["disagree"] += 1
            per_round[r]["disagree"] += 1
            dis_rows.append((m, r))

print("\n== A. Score-only vs round_end over every (match, round) with freeze_end ==")
print(f"score status (all freeze_end rows): {dict(status_count)}")
print(f"categories (round_end missing counted separately): {dict(claim_cat)}")
n_unsc_claim = sum(v for k, v in claim_cat.items() if k.startswith("unscorable") or k == "round_end_missing")
print(f"claim-style unscorable (score not scorable OR round_end missing): {n_unsc_claim}")
print(f"gap of 'both' (ambiguous) rows: {dict(gap_amb)}")
print(f"scorable with round_end: {claim_cat['agree'] + claim_cat['disagree']}; agree {claim_cat['agree']}; disagree {claim_cat['disagree']}")

print("\n== B. Per-round split (rounds with any disagreement, or >=50 rows) ==")
print("  round  agree  disagree  unsc_both  unsc_no_next  unsc_incons  re_missing")
for r in sorted(per_round):
    d = per_round[r]
    tot = sum(d.values())
    if d["disagree"] == 0 and tot < 50:
        continue
    print(f"  {r:>5}  {d['agree']:>5}  {d['disagree']:>8}  {d['unscorable:both']:>9}  {d['unscorable:no_next']:>12}  "
          f"{d['unscorable:inconsistent']:>11}  {d['re_missing']:>10}")
print(f"  rounds with any disagreement: {sorted(r for r in per_round if per_round[r]['disagree'] > 0)}")

print("\n== C. Disagreements by absolute round and by day group ==")
dis_round = collections.Counter(r for _, r in dis_rows)
print(f"  disagreements by round: {dict(sorted(dis_round.items()))}")


def day_group(m):
    d = int(m.split("/")[3])
    return "d01-07" if d <= 7 else "d12-25" if 12 <= d <= 25 else "other"


day_grp = collections.Counter(day_group(m) for m, _ in dis_rows)
print(f"  disagreements by day group: {dict(day_grp)}")
dg = collections.defaultdict(collections.Counter)
for m, r in dis_rows:
    dg[r][day_group(m)] += 1
for r in sorted(dg):
    print(f"    round {r}: {dict(dg[r])}")

print("\n== D. Matches with a disagreement ==")
mm_dis = collections.Counter(m for m, _ in dis_rows)
print(f"  matches with >=1 disagreement: {len(mm_dis)}; disagreement rounds per match (distribution): "
      f"{dict(collections.Counter(mm_dis.values()))}")

print("\n== E. Tie check at rounds 13 and 25 (score-only cannot score ties) ==")
for rr in (9, 13, 25, 28):
    rows = [(m, scm[rr]) for m, scm in sc.items() if rr in scm]
    tie = sum(1 for _, (t, c) in rows if t == c)
    print(f"  round {rr}: freeze_end rows {len(rows)}; tied at freeze_end {tie}; "
          f"score-only status: {dict(collections.Counter(score_only(sc[m], rr)[0] for m, _ in rows))}")

print("\n== F. Match-id duplicates (copies under different keys) ==")
g = HDR.group_by("match_id").agg(pl.len().alias("n"), pl.col("key").alias("keys"))
dup_ids = g.filter(pl.col("n") > 1)
print(f"  match_id groups with >1 key: {dup_ids.height}; extra key rows: {int((dup_ids['n'] - 1).sum()) if dup_ids.height else 0}")
key_to_id = {k: i for k, i in HDR.select(["key", "match_id"]).iter_rows()}
dup_keys = set()
for row in dup_ids.iter_rows(named=True):
    for k in row["keys"][1:]:
        dup_keys.add(k)
dis_in_dup = sum(1 for m, _ in dis_rows if m in dup_keys)
print(f"  disagreement rows whose key is a later copy: {dis_in_dup} of {len(dis_rows)}")
dis_first_copy = [(m, r) for m, r in dis_rows if m not in dup_keys]
print(f"  disagreements after keeping first key per match_id: {len(dis_first_copy)}; by round: "
      f"{dict(sorted(collections.Counter(r for _, r in dis_first_copy).items()))}")

SCRATCH.mkdir(parents=True, exist_ok=True)
with open(SCRATCH / "verify_D6_5_state.pkl", "wb") as f:
    pickle.dump({"sc": dict(sc), "rew": rew, "pc": pc, "dis_rows": dis_rows,
                 "dup_keys": dup_keys, "key_to_id": key_to_id}, f)
print("\nstate cached to scratchpad: verify_D6_5_state.pkl")
