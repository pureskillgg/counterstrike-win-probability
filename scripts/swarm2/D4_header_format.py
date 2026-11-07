r"""D4 diagnostic: is the 2024 header misfit a format effect? Cross-tab of complete matches by winning score.

For each complete 2024 match (round count == header total, rounds 1..n contiguous):
  winner_score = max(t_starters, ct_starters) final; format hint: 13 => MR12 win, 16 => MR15 win.
  Fit under S12 (swap after round 12) and S15 (swap after round 15), with round_end winners and with bomb-side
  substitution. Affected = >= 1 label-based round-13 mismatch.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_header_format.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
OUTCOMES = ["bomb_defused", "bomb_exploded"]
T, CT = 2, 3


def read_dir(fd, cols):
    fs = sorted(glob.glob(str(fd / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in fs], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(sorted(BASE.glob(f"basic_{ch}.*"))[0], cols)


H = read_dir(BASE / "header", ["key", "t_starters_score_final", "ct_starters_score_final"])
Hd = {r["key"]: r for r in H.iter_rows(named=True)}
re_ = read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
outs = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")).with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside"))
R = re_.join(outs.select(["match_key", "round", "bside"]), on=["match_key", "round"], how="left")
mm13 = set(R.filter((pl.col("round") == 13) & pl.col("bside").is_not_null())
           .filter(pl.col("bside") != pl.col("winner_team_code"))["match_key"].to_list())
by_match = {}
for r in R.sort(["match_key", "round"]).iter_rows(named=True):
    by_match.setdefault(r["match_key"], []).append(r)
agg = {}
for m, rounds in by_match.items():
    h = Hd.get(m)
    if h is None:
        continue
    tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
    n = len(rounds)
    if not (n == tA + tB and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    ws = max(tA, tB)
    fmt = "win<=13" if ws <= 13 else ("win14-15" if ws <= 15 else "win>=16")
    grp = "affected" if m in mm13 else "unaffected"
    for sname, sw in {"S12": 12, "S15": 15}.items():
        for src in ("end", "bomb"):
            wa = wb = 0
            for x in rounds:
                w = int(x["bside"]) if (src == "bomb" and x["bside"] is not None) else int(x["winner_team_code"])
                side_A = T if int(x["round"]) <= sw else CT
                if w == side_A:
                    wa += 1
                else:
                    wb += 1
            st = "ok" if (wa, wb) == (tA, tB) else ("swapped" if (wa, wb) == (tB, tA) else "bad")
            k = (fmt, grp, sname, src)
            agg.setdefault(k, {"ok": 0, "swapped": 0, "bad": 0})
            agg[k][st] += 1
print("complete 2024 matches by winning score x group x schedule x source (ok / swapped / bad):")
for k in sorted(agg):
    v = agg[k]
    tot = sum(v.values())
    print(f"  {k[0]:<9} {k[1]:<11} {k[2]:<4} {k[3]:<5} n={tot:>5} ok={v['ok']:>5} swapped={v['swapped']:>4} bad={v['bad']:>5}")
