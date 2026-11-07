"""verify_D3_3 part G: in-scope count. Apply the cross_period 2024 round filters (exactly 5 T and 5 CT by unique
player_id in player_info; no player_disconnect or bot_takeover in the round) to the MR24 day-12-25 inverted rounds,
and count: delta-based disagreements (claim test), elimination and bomb disagreements, and no-next-freeze rows.
ASSUMPTION: filters mirror scripts/model/sufficiency/cross_period.py (read, not imported).
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


def day(m):
    return int(m.split("/")[3])


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
re_df = (pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code"]).collect()
         .sort("tick").group_by(["match_key", "round"]).agg(pl.col("tick").first(), pl.col("winner_team_code").first()))
RE = {(m, int(r)): (int(tk), int(w)) for m, r, tk, w in re_df.iter_rows()}
dth = pl.scan_parquet(files("player_death")).select(["match_key", "round", "tick", "player_id", "player_team_code"]).collect()
dth_g = collections.defaultdict(list)
for m, r, tk, pid, pt in dth.iter_rows():
    dth_g[(m, int(r))].append((int(tk), int(pid), int(pt) if pt is not None else -1))
pi = pl.scan_parquet(files("player_info")).select(["match_key", "round", "player_id", "team_code"]).collect()
pl_team = collections.defaultdict(dict)
for m, r, pid, tc in pi.iter_rows():
    pl_team[(m, int(r))][pid] = tc
disc = set()
for ch in ("player_disconnect", "bot_takeover"):
    for f in files(ch):
        if not pl.read_parquet_schema(f):
            continue
        d = pl.read_parquet(f, columns=["match_key", "round"])
        disc |= {(m, int(r)) for m, r in d.iter_rows()}
bs = (pl.scan_parquet(files("bomb_state")).filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
      .select(["match_key", "round", "event_type"]).collect())
bo = {(m, int(r)): (T if ev == "bomb_exploded" else CT) for m, r, ev in bs.iter_rows()}


def in_scope(m, k):
    d = pl_team.get((m, k), {})
    n_t = sum(1 for v in d.values() if v == T)
    n_ct = sum(1 for v in d.values() if v == CT)
    return n_t == 5 and n_ct == 5 and (m, k) not in disc


def elim(m, k):
    t_end = RE[(m, k)][0]
    t_frz = FT[(m, k)]
    vic = {T: set(), CT: set()}
    for tk, pid, pt in dth_g.get((m, k), []):
        if t_frz <= tk <= t_end and pt in (T, CT):
            vic[pt].add(pid)
    dT, dC = len(vic[T]), len(vic[CT])
    if dT >= 5 and dC < 5:
        return CT
    if dC >= 5 and dT < 5:
        return T
    return None


out = collections.Counter()
for k in (13, 25, 28, 12, 14, 27):
    for m in F:
        if day(m) < 12 or MR.get(m, 24) != 24 or (m, k) not in RE:
            continue
        if not in_scope(m, k):
            out[(k, "out_of_scope")] += 1
            continue
        w = RE[(m, k)][1]
        out[(k, "in_scope")] += 1
        if k in F[m] and (k + 1) in F[m] and not swap_expected(k, 24):
            t, c = F[m][k]
            t2, c2 = F[m][k + 1]
            dT, dC = t2 - t, c2 - c
            if dC == 1 and dT == 0:
                ds = CT
            elif dT == 1 and dC == 0:
                ds = T
            else:
                ds = None
            if ds is None:
                out[(k, "delta_unclear_or_none")] += 1
            else:
                out[(k, "delta_valid")] += 1
                out[(k, "delta_disagree_rend" if ds != w else "delta_agree_rend")] += 1
        else:
            out[(k, "no_valid_delta(next freeze missing or swap)")] += 1
        e = elim(m, k)
        if e is not None:
            out[(k, "elim_disagree_rend" if e != w else "elim_agree_rend")] += 1
        if (m, k) in bo:
            out[(k, "bomb_disagree_rend" if bo[(m, k)] != w else "bomb_agree_rend")] += 1
for k in (13, 25, 28, 12, 14, 27):
    print(f"round {k} (MR24, day 12-25, in-scope = cross_period 5v5 filter):")
    for kk in sorted(x for x in out if x[0] == k):
        print(f"   {kk[1]:<45} {out[kk]}")
