"""verify_D3_3 part A: recompute the D3-3 claim numbers independently, then cross-tab by per-match format.

Own implementation (not imported from the claimed script). Steps:
  1. Per 2024 match: max_rounds from the round_announce_last_round_half rule (2 x announce round + 2, else 24),
     the same rule cross_period/A7 use (the 2024 header has no max_rounds column). Reported, not assumed.
  2. Freeze row per (match, round): earliest freeze_end by tick; duplicates counted.
  3. Claim test replicated exactly: MAX_ROUNDS = 24 for every match (claimed script), delta = freeze(r+1) - freeze(r),
     exclude swap_expected(r, 24), keep dc==1&dt==0 (CT) or dt==1&dc==0 (T), compare with round_end winner.
  4. Same test with the per-match rule MR (ASSUMPTION: rule MR is the true halftime schedule).
Output goes to stdout; the log file is written by the caller.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


# 1. per-match rule MR
ann = (pl.scan_parquet(files("round_state"))
       .filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).group_by("match_key").agg(pl.col("round").min().alias("r_ann"))
       .collect())
MR = {m: (int(r) * 2 + 2) for m, r in ann.iter_rows()}
hdr_keys = pl.scan_parquet(str(TOME / "header" / "dataframe_*")).select(["key"]).collect()["key"].to_list()
print(f"header matches {len(hdr_keys)}; matches with announce event {len(MR)}")
print("rule MR distribution (announce-based, default 24):",
      dict(sorted(collections.Counter(MR.get(m, 24) for m in hdr_keys).items())))
print("  announce round values:", dict(sorted(collections.Counter(int(r) for r in ann["r_ann"].to_list()).items())))

# 2. freeze rows and round_end winners
fe_all = (pl.scan_parquet(files("round_state"))
          .filter(pl.col("event_type") == "round_freeze_end")
          .select(["match_key", "round", "tick", "t_score", "ct_score"]).collect())
dup_keys = fe_all.height - fe_all.select(["match_key", "round"]).unique().height
conf = (fe_all.group_by(["match_key", "round"]).agg(pl.col("t_score").n_unique().alias("nt"),
                                                    pl.col("ct_score").n_unique().alias("nc"))
        .filter((pl.col("nt") > 1) | (pl.col("nc") > 1)).height)
print(f"freeze_end rows {fe_all.height}; duplicate (match,round) keys {dup_keys}; keys with conflicting scores {conf}")
fe = (fe_all.sort("tick").group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
live = {(m, int(r)): (int(t), int(c)) for m, r, t, c in fe.iter_rows()}

re_all = (pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code", "player_count"])
          .collect())
re_dup = re_all.height - re_all.select(["match_key", "round"]).unique().height
print(f"round_end rows {re_all.height}; duplicate (match,round) keys {re_dup}")
re_u = re_all.sort("tick").group_by(["match_key", "round"]).agg(pl.col("winner_team_code").first(),
                                                              pl.col("player_count").first())
win = {(m, int(r)): int(w) for m, r, w in re_u.select(["match_key", "round", "winner_team_code"]).iter_rows()}
pc = {(m, int(r)): int(p) for m, r, p in re_u.select(["match_key", "round", "player_count"]).iter_rows()}
print("winner_team_code values:", dict(collections.Counter(win.values())))
print("player_count distribution over round_end (u):", dict(sorted(collections.Counter(pc.values()).items())))


def run(mr_of, label):
    cnt = collections.Counter()
    by = collections.Counter()
    by_tot = collections.Counter()
    dis_m = collections.defaultdict(set)
    for (m, r), (t1, c1) in live.items():
        if (m, r) not in win or (m, r + 1) not in live:
            cnt["no_next_or_no_winner"] += 1
            continue
        mr = mr_of(m)
        if swap_expected(r, mr):
            cnt["skipped_swap"] += 1
            continue
        t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if dc == 1 and dt == 0:
            ds = CT
        elif dt == 1 and dc == 0:
            ds = T
        else:
            cnt["delta_unclear"] += 1
            continue
        grp = "d12_25" if int(m.split("/")[3]) >= 12 else "d01_07"
        cnt[f"valid_{grp}"] += 1
        by_tot[(grp, r)] += 1
        if ds != win[(m, r)]:
            cnt[f"disagree_{grp}"] += 1
            by[(grp, r)] += 1
            dis_m[grp].add(m)
    print(f"\n== {label} ==")
    for k, v in sorted(cnt.items()):
        print(f"   {k:<26} {v}")
    for g in ("d01_07", "d12_25"):
        items = sorted((r, by[(g, r)], by_tot[(g, r)]) for (gg, r) in by_tot if gg == g and by[(g, r)] > 0)
        print(f"   {g} disagree/valid by round: {items}")
        print(f"   {g} matches with >=1 disagreement: {len(dis_m[g])}")
    return by, by_tot, dis_m


by24, tot24, dm24 = run(lambda m: 24, "CLAIM replica: MR=24 for every match")
byMR, totMR, dmMR = run(lambda m: MR.get(m, 24), "per-match announce-rule MR")

print("\nper-match MR in disagreement (per-match rule run), day12_25:")
print("   rounds with disagreement by MR value:")
cross = collections.Counter()
cross_tot = collections.Counter()
for (m, r), (t1, c1) in live.items():
    if (m, r) not in win or (m, r + 1) not in live:
        continue
    if int(m.split("/")[3]) < 12:
        continue
    mr = 24
    t2, c2 = live[(m, r + 1)]
    if swap_expected(r, mr):
        continue
    dt, dc = t2 - t1, c2 - c1
    if dc == 1 and dt == 0:
        ds = CT
    elif dt == 1 and dc == 0:
        ds = T
    else:
        continue
    cross_tot[(MR.get(m, 24), r)] += 1
    if ds != win[(m, r)]:
        cross[(MR.get(m, 24), r)] += 1
for k in sorted(cross_tot):
    if cross[k] > 0 or k[1] in (9, 13, 25, 28):
        print(f"   MR={k[0]} round={k[1]}: disagree {cross[k]} / valid {cross_tot[k]}")
