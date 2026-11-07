"""D7-7: independent round-winner truth from the in-game score ledger.
Source: round_state round_freeze_end t_score / ct_score (team-attached: the score follows the team across the halftime swap).
For consecutive rounds r, r+1 in a match, two hypotheses are tested:
  no swap: team on T at r keeps T;  gains dX=(t1-t0), dY=(c1-c0)
  swap:    teams exchange sides;     gains dX=(c1-t0), dY=(t1-c0)
A hypothesis is valid when exactly one team gained exactly 1 point. If only one hypothesis is valid, it is used.
If both are valid (ties), the expected swap schedule decides: ASSUMPTION swap after rounds 12 and 24, and after
rounds 27, 33, ... (3 OT rounds per period, MR3). The schedule is checked against the unambiguous pairs (counted below).
ledger side for round r = 2 (T) if the team that was on T gained, 3 (CT) otherwise.
Compares ledger side with winner_team_code (round_end) and bomb-derived side, for player_count == 20 rounds.
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

RS = (pl.concat([pl.scan_parquet(f) for f in files(BASE / f"basic_round_state{SUF}")])
        .filter(pl.col("event_type") == "round_freeze_end")
        .select(["match_key", "round", "t_score", "ct_score"]).collect()
        .unique(subset=["match_key", "round"], keep="first"))
print("freeze_end (match,round) rows:", RS.height)

def expected_swap(r):
    return r in (12, 24) or (r >= 27 and (r - 27) % 6 == 0)

led = collections.defaultdict(dict)
for k, r, t, c in RS.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    led[k][r] = (t, c)

out = []  # (match_key, round, ledger_side, status, swap_flag)
stat = collections.Counter()
sched = collections.Counter()  # (expected, observed) for unambiguous pairs
for k, d in led.items():
    for r in sorted(d):
        if r + 1 not in d:
            continue
        t0, c0 = d[r]; t1, c1 = d[r + 1]
        nosw = (t1 - t0, c1 - c0) in [(1, 0), (0, 1)]
        sw = (c1 - t0, t1 - c0) in [(1, 0), (0, 1)]
        if nosw and sw:
            # ambiguous: resolve by expected schedule; side of the gaining team may still differ
            swap = expected_swap(r)
            status = "tie_resolved_by_schedule"
            stat["ambiguous"] += 1
        elif nosw:
            swap = False; status = "unambiguous"
        elif sw:
            swap = True; status = "unambiguous"
        else:
            stat["bad_no_hypothesis"] += 1
            out.append((k, r, None, "bad", None))
            continue
        if status == "unambiguous":
            sched[(expected_swap(r), swap)] += 1
        if not swap:
            gainT = (t1 - t0) == 1
        else:
            gainT = (c1 - t0) == 1
        side = 2 if gainT else 3
        out.append((k, r, side, status, swap))
        stat["pairs_total"] += 1

print("pair counts:", dict(stat))
print("unambiguous pairs: (expected_swap, observed_swap) -> count:", dict(sched))
L = pl.DataFrame(out, schema={"match_key": pl.Utf8, "round": pl.Int64, "ledger_side": pl.Int64,
                              "status": pl.Utf8, "swap": pl.Boolean}, orient="row")
L.write_parquet(str(ROOT / "logs" / "swarm2" / "D7_ledger_sides.parquet"))

R = pl.read_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_all.parquet"))
X = (R.filter((pl.col("player_count") == 20))
       .join(L, on=["match_key", "round"], how="left"))
X = X.with_columns(
    pl.when(pl.col("ledger_side").is_null()).then(None).otherwise(pl.col("ledger_side") == pl.col("winner_team_code")).alias("w_ok"),
    pl.when(pl.col("ledger_side").is_null() | pl.col("bomb_side").is_null()).then(None).otherwise(pl.col("ledger_side") == pl.col("bomb_side")).alias("b_ok"),
    pl.when(pl.col("mdt") < pl.datetime(2024, 2, 7)).then(pl.lit("pre")).otherwise(pl.lit("post")).alias("period"),
)
X.write_parquet(str(ROOT / "logs" / "swarm2" / "D7_rounds_with_ledger.parquet"))

print("\n== rounds with a ledger side AND a round_end winner (winner_team_code) ==")
W = X.filter(pl.col("w_ok").is_not_null())
print("rows:", W.height, "| winner agrees with ledger:", int(W["w_ok"].sum()), "| disagrees:", int((~W["w_ok"]).sum()))
print(W.group_by(["period", "w_ok"]).len().sort(["period", "w_ok"]))

print("\n== rounds with bomb outcome AND ledger side: bomb-derived side vs ledger ==")
B = X.filter(pl.col("b_ok").is_not_null())
print(B.group_by(["period", "rbin", "b_ok"]).len().sort(["period", "rbin", "b_ok"]))

print("\n== mismatch rounds (winner vs bomb): which one matches the ledger? ==")
M = X.filter(pl.col("bomb_side").is_not_null() & pl.col("ledger_side").is_not_null())
print(M.group_by(["mm", "w_ok", "b_ok"]).len().sort(["mm", "w_ok", "b_ok"]))

print("\n== mismatch rounds by rbin (ledger-known only) ==")
print(M.filter(pl.col("mm")).group_by(["rbin"]).agg(pl.len().alias("n"), pl.col("b_ok").sum().alias("bomb_agrees_ledger"),
      pl.col("w_ok").sum().alias("winner_agrees_ledger")).sort("rbin"))

print("\n== ledger-known rounds with no bomb outcome: winner vs ledger, by period ==")
NB = X.filter(pl.col("bomb_side").is_null() & pl.col("ledger_side").is_not_null() & pl.col("winner_team_code").is_not_null())
print(NB.group_by(["period", "w_ok"]).len().sort(["period", "w_ok"]))
