"""verify_D3_3 part H: dedupe-aware counts. Exact-signature groups (round_end, player_death, player_info content, as in
A3_dedupe_2024 and cross_period.signatures) keep the lowest match key. Recompute the claim count and the in-scope count
(5T/5CT roster, no disconnect/takeover) on kept matches only."""
import collections
import glob
import hashlib
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


def day(m):
    return int(m.split("/")[3])


re_all = pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code"]).collect()
de_all = pl.scan_parquet(files("player_death")).select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
pi_all = pl.scan_parquet(files("player_info")).select(["match_key", "round", "player_id", "team_code"]).collect()
parts = collections.defaultdict(list)
for name, df in (("re", re_all), ("de", de_all), ("pi", pi_all)):
    for m, sub in df.group_by("match_key"):
        body = sub.drop("match_key")
        parts[m[0]].append((name, tuple(sorted(tuple("" if v is None else str(v) for v in row) for row in body.iter_rows()))))
sig = {}
for m, lst in parts.items():
    h = hashlib.sha1(repr(sorted(lst)).encode("utf-8")).hexdigest()
    sig[m] = h
groups = collections.defaultdict(list)
for m, h in sig.items():
    groups[h].append(m)
dup_groups = [sorted(g) for g in groups.values() if len(g) > 1]
dropped = {m for g in dup_groups for m in g[1:]}
kept = set(sig) - dropped
print(f"matches {len(sig)}; exact duplicate groups {len(dup_groups)}; redundant copies dropped {len(dropped)}; kept {len(kept)}")
print("  group sizes:", dict(sorted(collections.Counter(len(g) for g in dup_groups).items())))
print("  duplicate copies by day group:", dict(collections.Counter("d12_25" if day(m) >= 12 else "d01_07" for m in dropped)))

ann = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).group_by("match_key").agg(pl.col("round").min().alias("r_ann")).collect())
MR = {m: int(r) * 2 + 2 for m, r in ann.iter_rows()}
fe = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_freeze_end")
      .select(["match_key", "round", "tick", "t_score", "ct_score"]).collect().sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("tick").first(), pl.col("t_score").first(), pl.col("ct_score").first()))
F = collections.defaultdict(dict)
FT = {}
for m, r, tk, t, c in fe.select(["match_key", "round", "tick", "t_score", "ct_score"]).iter_rows():
    F[m][int(r)] = (int(t), int(c))
    FT[(m, int(r))] = int(tk)
re_u = (re_all.sort("tick").group_by(["match_key", "round"]).agg(pl.col("tick").first(), pl.col("winner_team_code").first()))
RE = {(m, int(r)): (int(tk), int(w)) for m, r, tk, w in re_u.iter_rows()}
pl_team = collections.defaultdict(dict)
for m, r, pid, tc in pi_all.iter_rows():
    pl_team[(m, int(r))][pid] = tc
disc = set()
for ch in ("player_disconnect", "bot_takeover"):
    for f in files(ch):
        if pl.read_parquet_schema(f):
            d = pl.read_parquet(f, columns=["match_key", "round"])
            disc |= {(m, int(r)) for m, r in d.iter_rows()}
dth_g = collections.defaultdict(list)
for m, r, tk, pid, pt in de_all.iter_rows():
    dth_g[(m, int(r))].append((int(tk), int(pid), int(pt) if pt is not None else -1))


def in_scope(m, k):
    d = pl_team.get((m, k), {})
    return sum(1 for v in d.values() if v == T) == 5 and sum(1 for v in d.values() if v == CT) == 5 and (m, k) not in disc


def elim(m, k):
    t_end = RE[(m, k)][0]
    t_frz = FT[(m, k)]
    vic = {T: set(), CT: set()}
    for tk, pid, pt in dth_g.get((m, k), []):
        if t_frz <= tk <= t_end and pt in (T, CT):
            vic[pt].add(pid)
    if len(vic[T]) >= 5 and len(vic[CT]) < 5:
        return CT
    if len(vic[CT]) >= 5 and len(vic[T]) < 5:
        return T
    return None


cnt = collections.Counter()
per_round = collections.Counter()
per_round_tot = collections.Counter()
match_dis = collections.defaultdict(set)
for (m, k), (tk, w) in RE.items():
    if m not in kept or day(m) < 12:
        continue
    mr = MR.get(m, 24)
    if k not in F[m] or (k + 1) not in F[m] or swap_expected(k, 24):
        continue
    t, c = F[m][k]
    t2, c2 = F[m][k + 1]
    dT, dC = t2 - t, c2 - c
    if dC == 1 and dT == 0:
        ds = CT
    elif dT == 1 and dC == 0:
        ds = T
    else:
        cnt["claim_unclear"] += 1
        continue
    cnt["claim_valid_kept_d12_25"] += 1
    per_round_tot[(mr, k)] += 1
    if ds != w:
        cnt["claim_disagree_kept_d12_25"] += 1
        per_round[(mr, k)] += 1
        match_dis["all"].add(m)
        if in_scope(m, k):
            cnt["in_scope_disagree"] += 1
            per_round[("in", mr, k)] += 1
    if in_scope(m, k):
        cnt["in_scope_valid_delta"] += 1
print("\n== dedupe-aware claim count (MAX_ROUNDS=24 as claimed; swap 12/27 excluded; kept matches; d12_25) ==")
for kk, v in sorted(cnt.items()):
    print(f"   {kk:<32} {v}")
print("   matches with >=1 disagreement (kept):", len(match_dis["all"]))
print("   disagree/valid by (MR, round) (kept):", sorted((k, per_round[k], per_round_tot[k]) for k in per_round_tot if per_round[k] > 0))
print("   in-scope disagreements by (MR, round):", sorted((k[1], k[2], v) for k, v in per_round.items() if k[0] == "in"))
