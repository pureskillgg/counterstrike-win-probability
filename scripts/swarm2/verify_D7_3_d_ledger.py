"""Verify D7-3 (independent): re-implements the side ledger from basic_round_state round_freeze_end scores
(not from the D7 parquet outputs), then tests the claimed flip counts.

Ledger (own re-implementation): per match, rounds sorted; freeze-end (t_score, ct_score) per round.
A transition i -> i+1 is valid under an orientation pair (o0, o1) when exactly one team gains exactly one point.
Minimum orientation-change path (DP). A pair is 'unique' when all optimal paths agree on the winner side of round i
and on whether the orientation changes between i and i+1 (swap_next).
ASSUMPTION: the minimum number of orientation changes equals the true number of side swaps (as in D7-8).
first_after_swap (fas) for round r := swap_next of pair (r-1, r), unique pairs only.
Writes nothing; prints tables to stdout (captured into logs\\swarm2\\verify_D7_3.log).
"""
import glob, sys, collections
from pathlib import Path
import polars as pl
sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1); pl.Config.set_tbl_cols(-1); pl.Config.set_tbl_width_chars(240)
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


RS_schema = pl.read_parquet_schema(files(BASE / f"basic_round_state{SUF}")[0])
print("round_state columns:", list(RS_schema.keys()))
RS = (pl.concat([pl.scan_parquet(f).select(["match_key", "round", "event_type", "t_score", "ct_score"])
                 for f in files(BASE / f"basic_round_state{SUF}")])
        .filter(pl.col("event_type") == "round_freeze_end")
        .select(["match_key", "round", "t_score", "ct_score"]).collect())
print("freeze_end rows:", RS.height, "| distinct (match,round):", RS.select(["match_key", "round"]).unique().height)
RS = RS.unique(subset=["match_key", "round"], keep="first")
print("null scores after dedupe:", RS.filter(pl.col("t_score").is_null() | pl.col("ct_score").is_null()).height)

H = pl.read_parquet(str(BASE / "header" / "dataframe_00000"), columns=["key", "build_num"]).rename({"key": "match_key"})
build = dict(H.iter_rows())

RE = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "winner_team_code", "player_count"])
                for f in files(BASE / f"basic_round_end{SUF}")]).collect()
BS = pl.concat([pl.scan_parquet(f).select(["match_key", "round", "event_type"])
                for f in files(BASE / f"basic_bomb_state{SUF}")]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"])).collect()
ev = (BS.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
        .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(3).otherwise(2).alias("bomb_side"))
        .select(["match_key", "round", "bomb_side"]))

led = collections.defaultdict(dict)
for k, r, t, c in RS.iter_rows():
    led[k][r] = (t, c)

INF = 10 ** 9


def valid(t0, c0, o0, t1, c1, o1):
    A0, B0 = (t0, c0) if o0 == 0 else (c0, t0)
    A1, B1 = (t1, c1) if o1 == 0 else (c1, t1)
    dA, dB = A1 - A0, B1 - B0
    if (dA, dB) == (1, 0):
        return True, (2 if o0 == 0 else 3)
    if (dA, dB) == (0, 1):
        return True, (2 if o0 == 1 else 3)
    return False, None


out = []
runs_stat = collections.Counter()
for k, d in led.items():
    rs = sorted(d)
    runs, cur = [], [rs[0]]
    for r in rs[1:]:
        if r == cur[-1] + 1:
            cur.append(r)
        else:
            runs.append(cur)
            cur = [r]
    runs.append(cur)
    for run in runs:
        L = len(run)
        if L < 2:
            runs_stat["single"] += 1
            continue
        V = []
        for i in range(L - 1):
            t0, c0 = d[run[i]]
            t1, c1 = d[run[i + 1]]
            V.append({(o0, o1): valid(t0, c0, o0, t1, c1, o1) for o0 in (0, 1) for o1 in (0, 1)})
        f = [[0, 0]]
        for i in range(L - 1):
            nf = [INF, INF]
            for o1 in (0, 1):
                for o0 in (0, 1):
                    ok, _ = V[i][(o0, o1)]
                    if ok and f[i][o0] < INF:
                        nf[o1] = min(nf[o1], f[i][o0] + (1 if o0 != o1 else 0))
            f.append(nf)
        total = min(f[L - 1])
        if total >= INF:
            runs_stat["infeasible"] += 1
            for i in range(L - 1):
                out.append((k, run[i], None, None))
            continue
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
        runs_stat["feasible"] += 1
        for i in range(L - 1):
            sides, swaps = set(), set()
            for o0 in (0, 1):
                for o1 in (0, 1):
                    ok, side = V[i][(o0, o1)]
                    if not ok or f[i][o0] >= INF or g[i + 1][o1] >= INF:
                        continue
                    if f[i][o0] + (1 if o0 != o1 else 0) + g[i + 1][o1] == total:
                        sides.add(side)
                        swaps.add(o0 != o1)
            if len(sides) == 1 and len(swaps) == 1:
                out.append((k, run[i], sides.pop(), swaps.pop()))
            else:
                out.append((k, run[i], None, None))
print("runs:", dict(runs_stat))
LG = pl.DataFrame(out, schema={"match_key": pl.Utf8, "round": pl.Int64, "lside": pl.Int64, "swap_next": pl.Boolean},
                  orient="row")
print("pairs:", LG.height, "| unique side:", LG.filter(pl.col("lside").is_not_null()).height,
      "| swap_next True:", LG.filter(pl.col("swap_next") == True).height,
      "| swap_next null:", LG.filter(pl.col("swap_next").is_null()).height)

# consistency check against the D7 parquet ledger (not used as a source)
D7 = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_ledger_dp.parquet")).rename({"ledger_side": "d7_side", "swap": "d7_swap"})
cmp = LG.select(["match_key", "round", "lside"]).join(D7.select(["match_key", "round", "d7_side"]), on=["match_key", "round"], how="full", coalesce=True)
both = cmp.filter(pl.col("lside").is_not_null() & pl.col("d7_side").is_not_null())
print("consistency vs D7 ledger: both non-null:", both.height, "| agree:", both.filter(pl.col("lside") == pl.col("d7_side")).height)

# round-level table for 20-player rounds
FAS = (LG.filter(pl.col("swap_next").is_not_null())
         .with_columns((pl.col("round") + 1).alias("round"))
         .select(["match_key", "round", pl.col("swap_next").alias("fas")]))
SIDE = LG.select(["match_key", "round", "lside"])
R = (RE.filter(pl.col("player_count") == 20)
       .join(SIDE, on=["match_key", "round"], how="left")
       .join(FAS, on=["match_key", "round"], how="left")
       .join(ev, on=["match_key", "round"], how="left")
       .with_columns(pl.col("match_key").replace_strict(build, default=None).alias("build_num")))
R = R.with_columns(
    pl.col("fas").fill_null(False).alias("fas_bool"),
    pl.col("fas").is_null().alias("fas_unknown"),
    pl.when(pl.col("lside").is_null()).then(None).otherwise(pl.col("lside") != pl.col("winner_team_code")).alias("w_dis"),
    pl.when(pl.col("bomb_side").is_null()).then(None).otherwise(pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"),
)
print("\nround-level rows (20 players):", R.height, "| null build:", R.filter(pl.col("build_num").is_null()).height,
      "| fas unknown:", R.filter(pl.col("fas_unknown")).height)

print("\n== ledger side vs winner_team_code, by build x fas (n, disagree) [rounds with unique ledger] ==")
print(R.filter(pl.col("w_dis").is_not_null()).group_by(["build_num", "fas_bool"]).agg(
    pl.len().alias("n"), pl.col("w_dis").sum().alias("disagree")).sort(["build_num", "fas_bool"]).rows())

print("\n== ledger vs winner disagreements by build x round (all rounds) ==")
print(R.filter(pl.col("w_dis") == True).group_by(["build_num", "round"]).len().sort(["build_num", "round"]).rows())

print("\n== fas rounds: build x round (n with ledger, disagree) ==")
print(R.filter(pl.col("fas_bool") & pl.col("w_dis").is_not_null()).group_by(["build_num", "round"]).agg(
    pl.len().alias("n"), pl.col("w_dis").sum().alias("dis")).sort(["build_num", "round"]).rows())

print("\n== bomb-based mismatch: build x fas x round (n, mm) ==")
print(R.filter(pl.col("mm").is_not_null()).group_by(["build_num", "fas_bool", "round"]).agg(
    pl.len().alias("n"), pl.col("mm").sum().alias("mm")).sort(["build_num", "fas_bool", "round"]).rows())

print("\n== ledger side vs bomb side, round 13, by build (n with both, agree) ==")
r13 = R.filter((pl.col("round") == 13) & pl.col("lside").is_not_null() & pl.col("bomb_side").is_not_null())
print(r13.with_columns((pl.col("lside") == pl.col("bomb_side")).alias("ag")).group_by("build_num").agg(
    pl.len().alias("n"), pl.col("ag").sum().alias("agree")).sort("build_num").rows())

print("\n== build 9842 fas rounds: winner vs ledger (n, dis) by round ==")
print(R.filter((pl.col("build_num") == 9842) & pl.col("fas_bool") & pl.col("w_dis").is_not_null()).group_by("round").agg(
    pl.len().alias("n"), pl.col("w_dis").sum().alias("dis")).sort("round").rows())

