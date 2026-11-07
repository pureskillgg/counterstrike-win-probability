"""verify_D3_3 part I: consolidated final numbers on the deduplicated, in-scope population.
Population: 2024 tome, unique matches (exact-signature duplicates removed, lowest key kept, as A3_dedupe_2024),
day 12-25, MR24 (announce rule), round in-scope = exactly 5 T and 5 CT by unique player_id in player_info and no
player_disconnect / bot_takeover in the round (ASSUMPTION: mirrors cross_period.py round filter).
For rounds 12..30: round_end winner vs (a) claim delta test (MR24 swap set, unclear excluded), (b) elimination oracle,
(c) bomb outcome. Also prints the MR16 (2v2) round-9 row for the record, outside the population.
"""
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
sig = {m: hashlib.sha1(repr(sorted(lst)).encode("utf-8")).hexdigest() for m, lst in parts.items()}
groups = collections.defaultdict(list)
for m, h in sig.items():
    groups[h].append(m)
dropped = {m for g in groups.values() if len(g) > 1 for m in sorted(g)[1:]}
kept = set(sig) - dropped
print(f"unique matches kept {len(kept)} of {len(sig)} (dropped {len(dropped)} duplicate copies)")

ann = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_announce_last_round_half")
       .select(["match_key", "round"]).group_by("match_key").agg(pl.col("round").min().alias("r_ann")).collect())
MR = {m: int(r) * 2 + 2 for m, r in ann.iter_rows()}
fe = (pl.scan_parquet(files("round_state")).filter(pl.col("event_type") == "round_freeze_end")
      .select(["match_key", "round", "tick", "t_score", "ct_score"]).collect().sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("tick").first(), pl.col("t_score").first(), pl.col("ct_score").first()))
F, FT = collections.defaultdict(dict), {}
for m, r, tk, t, c in fe.select(["match_key", "round", "tick", "t_score", "ct_score"]).iter_rows():
    F[m][int(r)] = (int(t), int(c))
    FT[(m, int(r))] = int(tk)
re_u = re_all.sort("tick").group_by(["match_key", "round"]).agg(pl.col("tick").first(), pl.col("winner_team_code").first())
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
bs = (pl.scan_parquet(files("bomb_state")).filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
      .select(["match_key", "round", "event_type"]).collect())
bo = {(m, int(r)): (T if ev == "bomb_exploded" else CT) for m, r, ev in bs.iter_rows()}
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


def delta_claim(m, k):
    if k not in F[m] or (k + 1) not in F[m] or swap_expected(k, 24):
        return "no_delta", None
    t, c = F[m][k]
    t2, c2 = F[m][k + 1]
    dT, dC = t2 - t, c2 - c
    if dC == 1 and dT == 0:
        return "valid", CT
    if dT == 1 and dC == 0:
        return "valid", T
    return "unclear", None


print("\n== in-scope (unique, MR24, day 12-25, 5v5 roster, no disconnect/takeover): round_end vs oracles ==")
tab = collections.defaultdict(collections.Counter)
for (m, k), (tk, w) in RE.items():
    if m not in kept or day(m) < 12 or MR.get(m, 24) != 24 or k > 30:
        continue
    if not in_scope(m, k):
        tab[k]["out_of_scope_rows"] += 1
        continue
    tab[k]["rows"] += 1
    st, dw = delta_claim(m, k)
    tab[k][f"delta_{st}"] += 1
    if st == "valid":
        tab[k]["delta_agree" if dw == w else "delta_disagree"] += 1
    e = elim(m, k)
    if e is not None:
        tab[k]["elim_agree" if e == w else "elim_disagree"] += 1
    if (m, k) in bo:
        tab[k]["bomb_agree" if bo[(m, k)] == w else "bomb_disagree"] += 1
keys = ["rows", "out_of_scope_rows", "delta_valid", "delta_agree", "delta_disagree", "delta_unclear", "delta_no_delta",
        "elim_agree", "elim_disagree", "bomb_agree", "bomb_disagree"]
print("   round " + " ".join(f"{x:>14}" for x in keys))
for k in sorted(tab):
    print(f"   {k:>5} " + " ".join(f"{tab[k][x]:>14}" for x in keys))
tot_dis = sum(tab[k]["delta_disagree"] for k in tab)
print(f"\n   in-scope claim-test disagreements (all rounds 12..30, unique matches): {tot_dis}")
print(f"   in-scope delta-disagree rounds: {sorted((k, tab[k]['delta_disagree']) for k in tab if tab[k]['delta_disagree'] > 0)}")
inv = [(k, tab[k]['elim_disagree'], tab[k]['elim_agree']) for k in sorted(tab)
       if tab[k]['elim_disagree'] + tab[k]['elim_agree'] > 0 and tab[k]['elim_disagree'] / (tab[k]['elim_disagree'] + tab[k]['elim_agree']) > 0.2]
print(f"   in-scope rounds with elimination disagreement > 20% (k, disagree, agree): {inv}")
print("\n   record only: MR16 (2v2 roster) round 9, unique matches, day 12-25 (outside the 2024 test population):")
c16 = collections.Counter()
for (m, k), (tk, w) in RE.items():
    if m in kept and day(m) >= 12 and MR.get(m, 24) == 16 and k == 9 and (m, k) in FT:
        st, dw = delta_claim(m, k)
        if st == "valid":
            c16["delta_disagree" if dw != w else "delta_agree"] += 1
        e = elim(m, k)
        if e is not None:
            c16["elim_disagree" if e != w else "elim_agree"] += 1
print("   ", dict(c16))
