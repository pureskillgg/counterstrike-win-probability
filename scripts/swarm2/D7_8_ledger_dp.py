"""D7-8: robust ledger winners via minimum-swap orientation path (replaces the schedule tie-break of D7-7).
Team A is on T at the first round of a run (orientation o=0); o=1 means A is on CT. For consecutive rounds i, i+1
with freeze-end scores (t, c), A's and B's scores follow orientation: o=0 -> A=t,B=c ; o=1 -> A=c,B=t.
A transition is valid when exactly one team gained exactly 1 point. Cost = number of orientation changes (swaps).
Minimum-cost paths are enumerated per pair: if all optimal paths agree on the round winner, the side is set;
otherwise the pair is flagged ambiguous (side=None). Runs of consecutive rounds only.
ASSUMPTION: the minimum number of side swaps is the true swap count (swaps are rare: halftime and OT).
"""
import glob, sys, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
def files(fd): return sorted(glob.glob(str(fd / "dataframe_*")))
INF = 10**9
RS = (pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_state{SUF}")])
        .filter(pl.col("event_type") == "round_freeze_end")
        .select(["match_key", "round", "t_score", "ct_score"]).collect()
        .unique(subset=["match_key", "round"], keep="first"))
led = collections.defaultdict(dict)
for k, r, t, c in RS.iter_rows():
    led[k][r] = (t, c)

def ab(t, c, o):
    return (t, c) if o == 0 else (c, t)

def valid_trans(t0, c0, o0, t1, c1, o1):
    A0, B0 = ab(t0, c0, o0); A1, B1 = ab(t1, c1, o1)
    dA, dB = A1 - A0, B1 - B0
    if (dA, dB) == (1, 0):
        return True, (2 if o0 == 0 else 3)   # A won; A's side at round i
    if (dA, dB) == (0, 1):
        return True, (2 if o0 == 1 else 3)   # B won; B on T iff o0 == 1
    return False, None

out = []
stat = collections.Counter()
for k, d in led.items():
    rs = sorted(d)
    # split into runs of consecutive round numbers
    runs, cur = [], [rs[0]]
    for r in rs[1:]:
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            runs.append(cur); cur = [r]
    runs.append(cur)
    for run in runs:
        L = len(run)
        if L == 1:
            stat["single_round_run"] += 1
            continue
        # transitions for pairs i -> i+1 (i indexes positions in run)
        V = []
        for i in range(L - 1):
            r0, r1 = run[i], run[i + 1]
            t0, c0 = d[r0]; t1, c1 = d[r1]
            m = {}
            for o0 in (0, 1):
                for o1 in (0, 1):
                    ok, side = valid_trans(t0, c0, o0, t1, c1, o1)
                    m[(o0, o1)] = (ok, side)
            V.append(m)
        # forward
        f = [[0, 0]]
        for i in range(L - 1):
            nf = [INF, INF]
            for o1 in (0, 1):
                for o0 in (0, 1):
                    ok, _ = V[i][(o0, o1)]
                    if ok and f[i][o0] < INF:
                        cost = f[i][o0] + (1 if o0 != o1 else 0)
                        nf[o1] = min(nf[o1], cost)
            f.append(nf)
        total = min(f[L - 1])
        if total >= INF:
            stat["infeasible_run"] += 1
            for i in range(L - 1):
                out.append((k, run[i], None, None, "infeasible"))
            continue
        # backward
        g = [None] * L
        g[L - 1] = [0, 0]
        for i in range(L - 2, -1, -1):
            gi = [INF, INF]
            for o0 in (0, 1):
                for o1 in (0, 1):
                    ok, _ = V[i][(o0, o1)]
                    if ok and g[i + 1][o1] < INF:
                        gi[o0] = min(gi[o0], (1 if o0 != o1 else 0) + g[i + 1][o1])
            g[i] = gi
        stat["runs_feasible"] += 1
        for i in range(L - 1):
            sides, swaps = set(), set()
            for o0 in (0, 1):
                for o1 in (0, 1):
                    ok, side = V[i][(o0, o1)]
                    if not ok or f[i][o0] >= INF or g[i + 1][o1] >= INF:
                        continue
                    c = f[i][o0] + (1 if o0 != o1 else 0) + g[i + 1][o1]
                    if c == total:
                        sides.add(side); swaps.add(o0 != o1)
            if len(sides) == 1:
                status = "unique"
                stat["pairs_unique"] += 1
                out.append((k, run[i], sides.pop(), swaps.pop(), status))
            else:
                stat["pairs_ambiguous"] += 1
                out.append((k, run[i], None, None, "ambiguous"))

print("run/pair counts:", dict(stat))
L = pl.DataFrame(out, schema={"match_key": pl.Utf8, "round": pl.Int64, "ledger_side": pl.Int64,
                              "swap": pl.Boolean, "status": pl.Utf8}, orient="row")
print("pair status counts:", L.group_by("status").len().sort("status").rows())
L.write_parquet(str(ROOT / "logs" / "swarm2" / "D7_ledger_dp.parquet"))

R = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_all.parquet"))
X = (R.filter(pl.col("player_count") == 20)
       .join(L, on=["match_key", "round"], how="left")
       .with_columns(
           pl.when(pl.col("ledger_side").is_null() | pl.col("winner_team_code").is_null()).then(None)
             .otherwise(pl.col("ledger_side") == pl.col("winner_team_code")).alias("w_ok"),
           pl.when(pl.col("ledger_side").is_null() | pl.col("bomb_side").is_null()).then(None)
             .otherwise(pl.col("ledger_side") == pl.col("bomb_side")).alias("b_ok"),
           pl.when(pl.col("mdt") < pl.datetime(2024, 2, 7)).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"),
       ))
X.write_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_with_ledger_dp.parquet"))
print("\n== winner_team_code vs ledger (unique-ledger rounds) ==")
W = X.filter(pl.col("w_ok").is_not_null())
print(W.group_by(["period", "w_ok"]).len().sort(["period", "w_ok"]))
print("== bomb-derived side vs ledger (unique-ledger rounds) ==")
B = X.filter(pl.col("b_ok").is_not_null())
print(B.group_by(["period", "rbin", "b_ok"]).len().sort(["period", "rbin", "b_ok"]))
print("== post-period winner-vs-ledger disagreements split by bomb outcome present ==")
D = X.filter((pl.col("period") == "post") & pl.col("w_ok").is_not_null())
print(D.group_by([pl.col("bomb_side").is_not_null().alias("has_bomb"), "w_ok"]).len().sort(["has_bomb", "w_ok"]))
print("== swap-observed rounds (ledger path) by round number, unique pairs ==")
print(X.filter(pl.col("swap").is_not_null()).group_by("swap").len().rows())
print(L.filter(pl.col("swap") == True).group_by("round").len().sort("round").rows())
