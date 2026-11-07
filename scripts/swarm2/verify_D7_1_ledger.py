"""verify_D7_1_ledger.py -- independent re-derivation of the score-ledger onset behind the D7-1 claim.

Own implementation (does not import or read D7_8 output). Ledger source: round_state rows with
event_type == round_freeze_end (t_score, ct_score). Freeze-end scores at round r include the results of rounds < r.
Team identity is unknown from the tome, so each match gets an orientation path o_r in {0,1}
(o=0: team A starts round r on T; o=1: team A on CT). A transition r -> r+1 is valid when exactly one team
gains exactly 1 point and the other gains 0. The ledger side of round r is the side (2=T, 3=CT) of the winner.
Among all minimum-swap orientation paths (optimal = minimal number of side changes), the ledger side of round r
is set only when all optimal paths agree; the swap flag is set only when all optimal paths agree.
Reads: round_state (projection, lazy filter), round_end (winner_team_code, player_count), bomb_state
(defused/exploded only), header (match_date, build_num, key). Writes nothing.

ASSUMPTIONS:
  A1 match_date is UTC as recorded (trailing Z).
  A2 player_count == 20.
  A3 the minimum number of side changes is the true number of swaps (swaps are rare: halftime and OT).
  A4 freeze-end scores are cumulative through the previous round; consecutive rounds only (runs).
  A5 claim cut 2024-02-07 02:39:55 UTC for pre/post; also the ledger onset window reported by the claim is 00:24:22..01:01:18.
"""
import glob
import sys
import datetime as dt
from pathlib import Path
from collections import defaultdict

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(-1)
pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(240)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
CUT_CLAIM = dt.datetime(2024, 2, 7, 2, 39, 55)
PATH_CUT = dt.date(2024, 2, 12)
LED_LO = dt.datetime(2024, 2, 7, 0, 24, 22)
LED_HI = dt.datetime(2024, 2, 7, 1, 1, 18)
INF = 10 ** 9


def files(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def chan(ch):
    return files(BASE / f"basic_{ch}{SUF}")


def hdr(title):
    print("\n" + "=" * 110 + f"\n{title}\n" + "=" * 110)


def solve_match(run):
    """run: list of (round, t_score, ct_score) for consecutive rounds of one match.
    Returns {i: (ledger_side or None, swap_flag or None)} for transitions i -> i+1, or None if no valid path."""
    n = len(run)
    if n < 2:
        return {}
    trans = [dict() for _ in range(n - 1)]
    for i in range(n - 1):
        _, t0, c0 = run[i]
        _, t1, c1 = run[i + 1]
        for o in (0, 1):
            for o2 in (0, 1):
                sA0 = t0 if o == 0 else c0
                sA1 = t1 if o2 == 0 else c1
                sB0 = c0 if o == 0 else t0
                sB1 = c1 if o2 == 0 else t1
                gA = sA1 - sA0
                gB = sB1 - sB0
                if gA == 1 and gB == 0:
                    side = 2 if o == 0 else 3          # A won round i; A's side at round i
                elif gA == 0 and gB == 1:
                    side = 3 if o == 0 else 2          # B won round i; B's side at round i
                else:
                    continue
                trans[i][(o, o2)] = side
    F = [[INF, INF] for _ in range(n)]
    F[0] = [0, 0]
    for i in range(n - 1):
        for (o, o2) in trans[i]:
            c = 1 if o != o2 else 0
            if F[i][o] + c < F[i + 1][o2]:
                F[i + 1][o2] = F[i][o] + c
    Bk = [[INF, INF] for _ in range(n)]
    Bk[n - 1] = [0, 0]
    for i in range(n - 2, -1, -1):
        for (o, o2) in trans[i]:
            c = 1 if o != o2 else 0
            if c + Bk[i + 1][o2] < Bk[i][o]:
                Bk[i][o] = c + Bk[i + 1][o2]
    M = min(F[n - 1])
    if M >= INF:
        return None
    out = {}
    for i in range(n - 1):
        sides, swaps = set(), set()
        for (o, o2), side in trans[i].items():
            c = 1 if o != o2 else 0
            if F[i][o] + c + Bk[i + 1][o2] == M:
                sides.add(side)
                swaps.add(o != o2)
        out[i] = (sides.pop() if len(sides) == 1 else None,
                  swaps.pop() if len(swaps) == 1 else None)
    return out


print("ASSUMPTION A1: match_date UTC as recorded. A2: player_count == 20. A3: minimum-swap path is the true swap count.")

# ---------------------------------------------------------------- 1. round_state freeze-end scores
hdr("1. round_state event types and freeze-end coverage")
ev = (pl.scan_parquet(chan("round_state")).group_by("event_type").len().collect().sort("event_type"))
print(ev.rows())
FE0 = (pl.scan_parquet(chan("round_state"))
       .filter(pl.col("event_type") == "round_freeze_end")
       .select(["match_key", "round", "t_score", "ct_score"])
       .collect())
print("freeze-end rows:", FE0.height, "| unique (match,round):", FE0.select(["match_key", "round"]).unique().height)
FE = FE0.unique(subset=["match_key", "round"], keep="first", maintain_order=True)
print("round range in freeze-end rows:", FE["round"].min(), "..", FE["round"].max())

# ---------------------------------------------------------------- 2. ledger by min-swap orientation path
hdr("2. ledger derivation (own min-swap orientation DP)")
by_match = defaultdict(list)
for mk, r, t, c in FE.sort(["match_key", "round"]).select(["match_key", "round", "t_score", "ct_score"]).rows():
    by_match[mk].append((r, t, c))

led_rows = []     # (match_key, round, ledger_side, fas)
n_unsolved_match = 0
n_runs = 0
for mk, seq in by_match.items():
    # split into runs of consecutive round numbers
    runs, cur = [], []
    for item in seq:
        if cur and item[0] != cur[-1][0] + 1:
            runs.append(cur)
            cur = []
        cur.append(item)
    if cur:
        runs.append(cur)
    for run in runs:
        n_runs += 1
        res = solve_match(run)
        if res is None:
            n_unsolved_match += 1
            continue
        # row keyed on round run[i]: ledger side of round run[i]; swap = swap between run[i] and run[i+1],
        # re-keyed to round run[i+1] below (first-after-swap flag)
        for i, (side, swap) in res.items():
            led_rows.append((mk, run[i][0], side, swap))
print("runs:", n_runs, "| runs with no valid orientation path:", n_unsolved_match)
L = pl.DataFrame(led_rows, schema={"match_key": pl.String, "round": pl.Int64, "ledger_side": pl.Int64, "swap_in": pl.Boolean},
                 orient="row")
# swap_in for round r+1 is the swap flag of transition r; re-key it to round r+1
L_fas = (L.with_columns((pl.col("round") + 1).alias("round_fas"))
           .select(["match_key", "round_fas", "swap_in"])
           .rename({"round_fas": "round"}))
L = L.select(["match_key", "round", "ledger_side"]).join(L_fas, on=["match_key", "round"], how="full", coalesce=True)
print("ledger rows:", L.height, "| ledger side known:", L.filter(pl.col("ledger_side").is_not_null()).height,
      "| swap_in known True:", L.filter(pl.col("swap_in") == True).height,
      "| swap_in known False:", L.filter(pl.col("swap_in") == False).height,
      "| swap_in ambiguous (None):", L.filter(pl.col("swap_in").is_null()).height)
print("ledger side value counts:", L.group_by("ledger_side").len().sort("ledger_side").rows())

# ---------------------------------------------------------------- 3. join to round_end, bomb outcome, header
RE = (pl.scan_parquet(chan("round_end")).select(["match_key", "round", "winner_team_code", "player_count"]).collect())
RE = RE.unique(subset=["match_key", "round"], keep="first", maintain_order=True)
BS = (pl.scan_parquet(chan("bomb_state"))
      .filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
      .select(["match_key", "round", "event_type"]).collect())
OUT = (BS.group_by(["match_key", "round"])
       .agg(pl.col("event_type").eq("bomb_defused").any().alias("has_def"),
            pl.col("event_type").eq("bomb_exploded").any().alias("has_exp"))
       .filter(~(pl.col("has_def") & pl.col("has_exp")))
       .with_columns(pl.when(pl.col("has_def")).then(pl.lit(3)).otherwise(pl.lit(2)).cast(pl.Int64).alias("bomb_side")))
H = pl.read_parquet(files(BASE / "header"), columns=["key", "match_date", "build_num"]).with_columns(
    pl.col("match_date").str.replace("Z$", "").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f", strict=False).alias("md_ts"),
    pl.col("key").str.extract(r"^csds/(\d{4}/\d{2}/\d{2})/", 1).str.replace_all("/", "-")
      .str.to_date("%Y-%m-%d").alias("path_date")).rename({"key": "match_key"})
X = (RE.join(L, on=["match_key", "round"], how="inner")
       .join(OUT.select(["match_key", "round", "bomb_side"]), on=["match_key", "round"], how="left")
       .join(H.select(["match_key", "md_ts", "path_date", "build_num"]), on="match_key", how="left")
       .filter(pl.col("player_count") == 20)
       .with_columns(
           (pl.col("md_ts") < CUT_CLAIM).alias("pre"),
           (pl.col("path_date") >= PATH_CUT).alias("pg"),
           pl.when(pl.col("round") == 13).then(pl.lit("r13")).when(pl.col("round") == 25).then(pl.lit("r25"))
             .when(pl.col("round").is_in([9, 28])).then(pl.lit("r9_28")).otherwise(pl.lit("other")).alias("rbin"),
       ))
K = X.filter(pl.col("ledger_side").is_not_null())
K = K.with_columns((pl.col("winner_team_code") != pl.col("ledger_side")).alias("flip"))
print("\npc20 rounds with ledger side:", K.height, "| flips (winner != ledger):", int(K["flip"].sum()),
      "(D7_11 reported 74567 known rows over its own table, flip counts differ by scope)")

# ---------------------------------------------------------------- 4. flip rate by period and swap flag
hdr("4. winner-vs-ledger flip rate by claim period and first-after-swap (fas) flag")
print(K.group_by(["pre", "swap_in"]).agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips"),
                                         pl.col("flip").mean().round(4).alias("rate")).sort(["pre", "swap_in"]))
print("\nflip rate by round bin and fas (pc20):")
print(K.group_by(["rbin", "swap_in"]).agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips")).sort(["rbin", "swap_in"]))

# ---------------------------------------------------------------- 5. onset over fas rounds 9 and 13
hdr("5. onset over first-after-swap rounds 9 and 13 (ledger), by match_date")
F = K.filter((pl.col("swap_in") == True) & pl.col("round").is_in([9, 13])).sort("md_ts")
rows = F.select(["md_ts", "flip", "match_key", "round", "build_num", "path_date", "pg"]).rows()
n = len(rows)
flip_pre = [0] * (n + 1)
ok_pre = [0] * (n + 1)
for i, r in enumerate(rows):
    flip_pre[i + 1] = flip_pre[i] + int(r[1])
    ok_pre[i + 1] = ok_pre[i] + (0 if r[1] else 1)
errs = [flip_pre[c] + (ok_pre[n] - ok_pre[c]) for c in range(n + 1)]
best = min(range(n + 1), key=lambda c: errs[c])
print("fas 9/13 rows (pc20, ledger known):", n, "| best single cut errors:", errs[best], "| cut index", best)
print("  last agreeing before cut:", rows[best - 1][0] if best else None,
      "| first flip at/after cut:", rows[best][0] if best < n else None)
print("  all agreeing rows, latest:", max((r[0] for r in rows if not r[1]), default=None))
print("  all flipped rows, earliest:", min((r[0] for r in rows if r[1]), default=None))
print("  agreeing rows after the ledger-window end (01:01:18):",
      [(r[0], r[2], r[3], r[4]) for r in rows if (not r[1]) and r[0] > LED_HI])
print("  flipped rows before the ledger-window start (00:24:22):",
      [(r[0], r[2], r[3]) for r in rows if r[1] and r[0] < LED_LO])
print("  fas 9/13 rows inside [00:24:22, 01:01:18] (count, flips):",
      sum(1 for r in rows if LED_LO <= r[0] <= LED_HI), sum(1 for r in rows if LED_LO <= r[0] <= LED_HI and r[1]))
print("  fas 9/13 rows in [01:01:18, 02:39:55) (count, flips):",
      sum(1 for r in rows if LED_HI < r[0] < CUT_CLAIM), sum(1 for r in rows if LED_HI < r[0] < CUT_CLAIM and r[1]))
print("  fas 9/13 rows at/after 02:39:55 (count, flips):",
      sum(1 for r in rows if r[0] >= CUT_CLAIM), sum(1 for r in rows if r[0] >= CUT_CLAIM and r[1]))

print("\nfas 9/13 rows by build_num x flip:")
print(F.group_by(["build_num", "flip"]).agg(pl.len().alias("n")).sort(["build_num", "flip"]))
print("fas 9/13 rows by path >= 02/12 x pre-claim-cut x flip:")
print(F.group_by(["pre", "pg", "flip"]).agg(pl.len().alias("n")).sort(["pre", "pg", "flip"]))

# ---------------------------------------------------------------- 6. which label is consistent with the ledger
hdr("6. round-13 bomb rows (pc20): is the ledger side equal to the bomb side or to winner_team_code?")
R13B = (X.filter((pl.col("round") == 13) & pl.col("bomb_side").is_not_null() & pl.col("ledger_side").is_not_null()))
R13B = R13B.with_columns(
    (pl.col("ledger_side") == pl.col("bomb_side")).alias("ledger_eq_bomb"),
    (pl.col("ledger_side") == pl.col("winner_team_code")).alias("ledger_eq_winner"),
    (pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
print("round-13 bomb rows with ledger side known:", R13B.height)
print(R13B.group_by(["pre", "mm", "ledger_eq_bomb", "ledger_eq_winner"]).agg(pl.len().alias("n")).sort(["pre", "mm"]))
print("\nall pc20 bomb rows with ledger side known (ledger_eq_bomb, ledger_eq_winner, mismatch):")
AB = X.filter(pl.col("bomb_side").is_not_null() & pl.col("ledger_side").is_not_null()).with_columns(
    (pl.col("ledger_side") == pl.col("bomb_side")).alias("ledger_eq_bomb"),
    (pl.col("ledger_side") == pl.col("winner_team_code")).alias("ledger_eq_winner"),
    (pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
print(AB.group_by(["mm", "ledger_eq_bomb", "ledger_eq_winner"]).agg(pl.len().alias("n")).sort(["mm", "ledger_eq_bomb"]))

# ---------------------------------------------------------------- 7. build and path, all rounds with ledger
hdr("7. build_num x claim cut, all pc20 rounds with ledger side (non-fas and fas)")
print(K.group_by(["build_num", "pre", "swap_in"]).agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips"))
      .sort(["build_num", "pre", "swap_in"]))
print("\nDONE ledger")

# ---------------------------------------------------------------- 8. where are the non-swap post-onset flips, and do clean post r13 rows lack a swap
hdr("8. post-onset non-swap flips and round-13 swap status")
NF = K.filter((pl.col("pre") == False) & (pl.col("swap_in") == False))
print("post-onset non-swap rows:", NF.height, "| flips:", int(NF["flip"].sum()))
print(NF.group_by("rbin").agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips")).sort("rbin"))
NFF = NF.filter(pl.col("flip"))
print("flip round numbers (top 20 by count):", NFF.group_by("round").len().sort("len", descending=True).head(20).rows())
print("flips with a bomb outcome:", NFF.filter(pl.col("bomb_side").is_not_null()).height,
      "| without bomb outcome:", NFF.filter(pl.col("bomb_side").is_null()).height)
print("matches with such flips:", NFF["match_key"].n_unique(), "of", NF["match_key"].n_unique(), "post-onset matches with non-swap rows")
print("top matches by such flips (count):", NFF.group_by("match_key").len().sort("len", descending=True).head(8).rows())
print("pre-onset non-swap flips (baseline):", K.filter((pl.col("pre") == True) & (pl.col("swap_in") == False)).height,
      "rows,", int(K.filter((pl.col("pre") == True) & (pl.col("swap_in") == False))["flip"].sum()), "flips")
R13P = X.filter((pl.col("round") == 13) & (pl.col("pre") == False) & pl.col("bomb_side").is_not_null()
                ).with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mm"))
print("post-onset round-13 bomb rows by swap_in x mismatch (ledger-known or not):")
print(R13P.group_by(["swap_in", "mm"]).len().sort(["swap_in", "mm"]))
print("DONE section 8")

# ---------------------------------------------------------------- 9. scope: winner-vs-ledger flips by round number, pre vs post claim cut
hdr("9. scope of winner-vs-ledger flips by round number (pc20, ledger known)")
S = (K.group_by(["round", "pre"]).agg(pl.len().alias("n"), pl.col("flip").sum().alias("flips"))
       .sort(["round", "pre"]))
post = {r: (n, f) for r, p, n, f in S.rows() if p is False}
pre = {r: (n, f) for r, p, n, f in S.rows() if p is True}
print("round | pre-cut n flips | post-cut n flips | post flip rate")
for r in sorted(set(post) | set(pre)):
    pn, pf = pre.get(r, (0, 0))
    qn, qf = post.get(r, (0, 0))
    rate = (qf / qn) if qn else float("nan")
    print(f"{r:>5} | {pn:>7} {pf:>5} | {qn:>8} {qf:>5} | {rate:.3f}")
print("DONE section 9")
