"""verify_D7_5_10_main: independent check of claim D7-5 (which round-winner label is wrong after build 9961).

Sources (2024 tome, tomes/2024-02-g, 4,435 matches). Only W_code uses winner_team_code.
  W_code   : basic_round_end.winner_team_code (the label under test). Side codes 2=T, 3=CT.
  W_within : basic_round_state round_officially_ended t/c minus round_freeze_end t/c of the SAME round.
             A side swap cannot occur inside a round, so this label needs no orientation.
  W_dp     : own min-swap orientation DP over consecutive round_freeze_end rows (ledger).
             Orientation of the first round of each run fixed (team A on T at that round).
  W_bomb   : bomb_state outcome: bomb_defused -> 3 (CT), bomb_exploded -> 2 (T). Side-absolute.
Header (metadata): t_starters_score_final / ct_starters_score_final, build_num, match_date.
Period: build_num 9842 = pre, 9961 = post. Time check uses the claim onset 2024-02-07 01:01:18 UTC.
ASSUMPTION: player_count == 20 is the primary population (2024 convention); all-count results also printed.
ASSUMPTION: header score fields are final team round totals (used only for a total/parity test).
ASSUMPTION: matches with a duplicated (match, round) row in round_end or round_freeze_end are excluded.
"""
import glob
import sys
import collections
from pathlib import Path
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
SCRATCH = Path(r"C:\Users\<user>\AppData\Local\Temp\claude\C--Users-<user>-Documents-pureskill\e0a18ad2-aa9e-4a3c-9946-29641b2cde82\scratchpad")
T_SIDE, CT_SIDE = 2, 3
INF = 10 ** 9


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def scan(ch, cols, pred=None):
    lf = pl.concat([pl.scan_parquet(f) for f in files(BASE / ("basic_" + ch + SUF))])
    if pred is not None:
        lf = lf.filter(pred)
    return lf.select(cols).collect()


# ---------- header ----------
H = pl.concat([pl.read_parquet(f, columns=["key", "match_date", "build_num", "map_name",
                                           "t_starters_score_final", "ct_starters_score_final"])
               for f in files(BASE / "header")]).rename({"key": "match_key"})
H = H.with_columns(pl.col("match_date").str.replace("Z$", "")
                   .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("mdt"))
hdr = {d["match_key"]: d for d in H.to_dicts()}
print("header matches:", len(hdr))
print("build_num counts:", H.group_by("build_num").len().sort("build_num").rows())

# ---------- round_end (the label under test) ----------
RE_df = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code",
                           "win_reason_message", "player_count"])
re_map = {}
bad = set()
for d in RE_df.to_dicts():
    k = (d["match_key"], d["round"])
    if k in re_map:
        bad.add(d["match_key"])
    else:
        re_map[k] = d

# ---------- round_state: freeze_end, officially_ended, bomb events ----------
RS = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"],
          pl.col("event_type").is_in(["round_freeze_end", "round_officially_ended", "bomb_defused", "bomb_exploded"]))
fe_map, roe_map = {}, {}
rs_bomb = collections.defaultdict(set)
for d in RS.to_dicts():
    k = (d["match_key"], d["round"])
    et = d["event_type"]
    if et == "round_freeze_end":
        if k in fe_map:
            bad.add(d["match_key"])
        else:
            fe_map[k] = d
    elif et == "round_officially_ended":
        if k not in roe_map:
            roe_map[k] = d
    else:
        rs_bomb[k].add(et)

# ---------- bomb_state outcomes (side-absolute) ----------
BS = scan("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"],
          pl.col("event_type").is_in(["bomb_defused", "bomb_exploded", "bomb_planted"]))
bomb = collections.defaultdict(lambda: {"def": 0, "exp": 0, "def_tick": None, "exp_tick": None,
                                        "plant": 0, "planter": None})
for d in BS.to_dicts():
    k = (d["match_key"], d["round"])
    b = bomb[k]
    et = d["event_type"]
    if et == "bomb_defused":
        b["def"] += 1
        b["def_tick"] = d["tick"] if b["def_tick"] is None else min(b["def_tick"], d["tick"])
    elif et == "bomb_exploded":
        b["exp"] += 1
        b["exp_tick"] = d["tick"] if b["exp_tick"] is None else min(b["exp_tick"], d["tick"])
    else:
        b["plant"] += 1
        if b["planter"] is None:
            b["planter"] = d["player_team_code"]


def bomb_side(k):
    b = bomb.get(k)
    if b is None or (b["def"] == 0 and b["exp"] == 0):
        return None
    if b["def"] > 0 and b["exp"] > 0:
        return -1  # conflict marker
    return CT_SIDE if b["def"] > 0 else T_SIDE


def bomb_out_tick(k):
    b = bomb.get(k)
    if b is None:
        return None
    ts = [x for x in (b["def_tick"], b["exp_tick"]) if x is not None]
    return min(ts) if ts else None


def rs_bomb_side(k):
    s = rs_bomb.get(k, set())
    if not s:
        return None
    if "bomb_defused" in s and "bomb_exploded" in s:
        return -1
    return CT_SIDE if "bomb_defused" in s else T_SIDE


print("duplicate-row matches excluded:", len(bad), sorted(bad))

FE_BY_MATCH = collections.defaultdict(list)
for (mk, r) in fe_map.keys():
    FE_BY_MATCH[mk].append(r)


def runs_of(rounds):
    rounds = sorted(rounds)
    runs, cur = [], [rounds[0]]
    for r in rounds[1:]:
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            runs.append(cur)
            cur = [r]
    runs.append(cur)
    return runs


def ledger_for_match(mk):
    """Own min-swap orientation DP. Returns (side per round, orientation per round, status list)."""
    side_out, orient_out, status = {}, {}, []
    rounds = FE_BY_MATCH.get(mk, [])
    if not rounds:
        return side_out, orient_out, ["no_fe"]
    runs = runs_of(rounds)
    for ri, run in enumerate(runs):
        L = len(run)
        if L == 1:
            status.append("single_round_run")
            orient_out[(mk, run[0])] = None
            continue
        fe = [(fe_map[(mk, r)]["t_score"], fe_map[(mk, r)]["ct_score"]) for r in run]
        # V[i][(o0, o1)] = side of the round-winner for pair (run[i], run[i+1]) or None if invalid
        V = []
        for i in range(L - 1):
            (t0, c0), (t1, c1) = fe[i], fe[i + 1]
            m = {}
            for o0 in (0, 1):
                for o1 in (0, 1):
                    A0, B0 = (t0, c0) if o0 == 0 else (c0, t0)
                    A1, B1 = (t1, c1) if o1 == 0 else (c1, t1)
                    dA, dB = A1 - A0, B1 - B0
                    if (dA, dB) == (1, 0):
                        m[(o0, o1)] = T_SIDE if o0 == 0 else CT_SIDE
                    elif (dA, dB) == (0, 1):
                        m[(o0, o1)] = CT_SIDE if o0 == 0 else T_SIDE
                    else:
                        m[(o0, o1)] = None
            V.append(m)
        f = [[INF, INF] for _ in range(L)]
        f[0][0] = 0  # first round of run: team A on T (orientation 0); global flip gives identical sides
        for i in range(L - 1):
            for o1 in (0, 1):
                best = INF
                for o0 in (0, 1):
                    if V[i][(o0, o1)] is not None and f[i][o0] < INF:
                        best = min(best, f[i][o0] + (1 if o0 != o1 else 0))
                f[i + 1][o1] = best
        total = min(f[L - 1])
        if total >= INF:
            status.append("infeasible_run")
            for r in run:
                orient_out[(mk, r)] = None
            continue
        g = [[INF, INF] for _ in range(L)]
        g[L - 1] = [0, 0]
        for i in range(L - 2, -1, -1):
            for o0 in (0, 1):
                best = INF
                for o1 in (0, 1):
                    if V[i][(o0, o1)] is not None and g[i + 1][o1] < INF:
                        best = min(best, (1 if o0 != o1 else 0) + g[i + 1][o1])
                g[i][o0] = best
        n_amb = 0
        for i in range(L - 1):
            sides = set()
            for o0 in (0, 1):
                for o1 in (0, 1):
                    s = V[i][(o0, o1)]
                    if s is None or f[i][o0] >= INF or g[i + 1][o1] >= INF:
                        continue
                    if f[i][o0] + (1 if o0 != o1 else 0) + g[i + 1][o1] == total:
                        sides.add(s)
            if len(sides) == 1:
                side_out[(mk, run[i])] = sides.pop()
            else:
                n_amb += 1
                side_out[(mk, run[i])] = None
        # one optimal orientation path (greedy from start, keep only edges on optimal paths)
        path = [0]
        cur = 0
        for i in range(L - 1):
            nxt = cur
            for o1 in (0, 1):
                if V[i][(cur, o1)] is None or g[i + 1][o1] >= INF:
                    continue
                if f[i][cur] + (1 if cur != o1 else 0) + g[i + 1][o1] == total:
                    nxt = o1
                    break
            path.append(nxt)
            cur = nxt
        for r, o in zip(run, path):
            orient_out[(mk, r)] = o
        status.append("ok_amb%d" % n_amb if n_amb else "ok")
        if ri > 0:
            status.append("multi_run")
    return side_out, orient_out, status


LEDGER, ORIENT, LSTAT = {}, {}, {}
for mk in FE_BY_MATCH:
    if mk in bad:
        continue
    s, o, st = ledger_for_match(mk)
    LEDGER.update(s)
    ORIENT.update(o)
    LSTAT[mk] = st
print("ledger matches:", len(LSTAT))
print("ledger status counter:", collections.Counter(x for st in LSTAT.values() for x in st).most_common())
print("ledger pair sides unique:", sum(1 for v in LEDGER.values() if v is not None),
      "| ambiguous or None:", sum(1 for v in LEDGER.values() if v is None))

# ---------- round table ----------
rows = []
for (mk, r), e in re_map.items():
    if mk in bad or mk not in hdr:
        continue
    h = hdr[mk]
    fe = fe_map.get((mk, r))
    fe_n = fe_map.get((mk, r + 1))
    roe = roe_map.get((mk, r))
    dT = dC = None
    if roe is not None and fe is not None:
        dT = roe["t_score"] - fe["t_score"]
        dC = roe["ct_score"] - fe["ct_score"]
    if (dT, dC) == (1, 0):
        w_within = T_SIDE
    elif (dT, dC) == (0, 1):
        w_within = CT_SIDE
    else:
        w_within = None
    bs = bomb_side((mk, r))
    rows.append({
        "match_key": mk, "round": r, "mdt": h["mdt"], "build": h["build_num"], "map": h["map_name"],
        "player_count": e["player_count"], "W_code": e["winner_team_code"],
        "reason": e["win_reason_code"], "reason_msg": e["win_reason_message"],
        "W_within": w_within, "dT": dT, "dC": dC,
        "W_dp": LEDGER.get((mk, r)), "orient": ORIENT.get((mk, r)),
        "W_bomb": bs, "W_rs_bomb": rs_bomb_side((mk, r)),
        "bomb_out_tick": bomb_out_tick((mk, r)), "re_tick": e["tick"],
        "fe_tick": fe["tick"] if fe else None, "fe_next_tick": fe_n["tick"] if fe_n else None,
        "roe_tick": roe["tick"] if roe else None, "planter": bomb.get((mk, r), {}).get("planter"),
        "tF": h["t_starters_score_final"], "cF": h["ct_starters_score_final"],
        "fe_t": fe["t_score"] if fe else None, "fe_c": fe["ct_score"] if fe else None,
    })
R = pl.DataFrame(rows, infer_schema_length=None)
maxr = R.group_by("match_key").agg(pl.col("round").max().alias("maxr"))
R = R.join(maxr, on="match_key", how="left")
R = R.with_columns(
    pl.when(pl.col("mdt") < pl.datetime(2024, 2, 7, 1, 1, 18)).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period_time"),
    pl.when(pl.col("build") == 9961).then(pl.lit("post")).otherwise(pl.lit("pre")).alias("period"),
    pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
      .otherwise(pl.lit("other")).alias("rbin"),
)
R.write_parquet(str(SCRATCH / "verify_D7_5_rounds.parquet"))
with open(str(SCRATCH / "verify_D7_5_state.pkl"), "wb") as fh:
    import pickle
    pickle.dump({"hdr": hdr, "bad": bad, "LSTAT": LSTAT}, fh)
print("round rows:", R.height, "| matches:", R["match_key"].n_unique())
print("period (build) vs period (time) disagreement rows:", R.filter(pl.col("period_time") != pl.col("period")).height)
print("build x period_time:", R.group_by(["build", "period_time"]).len().sort(["build", "period_time"]).rows())
