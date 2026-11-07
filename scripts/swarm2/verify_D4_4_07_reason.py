r"""Verifier D4-4: is round_end win_reason_code independent evidence for the round-13 label?

Compares the round-13 round_end (winner, reason) pair against the bomb outcome and the no-swap score step
(boundary 13->14, freeze-end fields) for 2024 complete regulation matches with a round-13 label.

Usage: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\verify_D4_4_07_reason.py
"""
import glob
import sys
from collections import Counter, defaultdict
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
T, CT = 2, 3


def files(ch):
    fd = BASE / "header" if ch == "header" else sorted(BASE.glob(f"basic_{ch}.*"))[0]
    return sorted(glob.glob(str(fd / "dataframe_*")))


H = pl.scan_parquet(files("header")).select(["key", "t_starters_score_final", "ct_starters_score_final"]).collect()
HD = {r["key"]: r for r in H.iter_rows(named=True)}
RE = pl.scan_parquet(files("round_end")).select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()
RE1 = RE.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)
BS = (pl.scan_parquet(files("bomb_state")).select(["match_key", "round", "tick", "event_type"])
      .filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).collect())
BO = BS.group_by(["match_key", "round"]).agg(
    (pl.col("event_type") == "bomb_exploded").any().alias("exp"),
    (pl.col("event_type") == "bomb_defused").any().alias("dfs"))
FE = (pl.scan_parquet(files("round_state")).select(["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").collect())
FE = FE.sort(["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first", maintain_order=True)

rows = defaultdict(dict)
for r in RE1.iter_rows(named=True):
    rows[r["match_key"]].setdefault(r["round"], {})["L"] = r["winner_team_code"]
    rows[r["match_key"]][r["round"]]["R"] = r["win_reason_code"]
for r in BO.iter_rows(named=True):
    d = rows[r["match_key"]].setdefault(r["round"], {})
    d["B"] = T if (r["exp"] and not r["dfs"]) else (CT if (r["dfs"] and not r["exp"]) else None)
for r in FE.iter_rows(named=True):
    rows[r["match_key"]].setdefault(r["round"], {})["fe"] = (r["t_score"], r["ct_score"])

cr = []
for m in rows:
    if m not in HD:
        continue
    rr = rows[m]
    Lr = sorted(r for r in rr if "L" in rr[r])
    n = len(Lr)
    tA = int(HD[m]["t_starters_score_final"]); tB = int(HD[m]["ct_starters_score_final"])
    if Lr == list(range(1, n + 1)) and n == tA + tB and n <= 24 and 13 in rr and "L" in rr[13]:
        cr.append(m)

tab_aff = Counter(); tab_inv = Counter(); tab_ok = Counter(); tab_bombok = Counter()
for m in cr:
    rr = rows[m]
    d = rr[13]
    L, R = d["L"], d["R"]
    B = d.get("B")
    fe13 = rr.get(13, {}).get("fe"); fe14 = rr.get(14, {}).get("fe")
    w13 = None
    if fe13 is not None and fe14 is not None:
        dt, dc = fe14[0] - fe13[0], fe14[1] - fe13[1]
        w13 = T if (dt, dc) == (1, 0) else (CT if (dt, dc) == (0, 1) else None)
    if B is not None and B != L:
        tab_aff[("bomb-exploded" if B == T else "bomb-defused", "label winner", L, "reason", R)] += 1
    elif B is not None:
        tab_bombok[("bomb-exploded" if B == T else "bomb-defused", "label winner", L, "reason", R)] += 1
    elif w13 is not None and w13 != L:
        tab_inv[("no-bomb, score says", "T" if w13 == T else "CT", "label winner", L, "reason", R)] += 1
    elif w13 is not None:
        tab_ok[("no-bomb, score agrees with label", "label winner", L, "reason", R)] += 1

print("affected round-13 (bomb side != label): bomb type x label winner x label reason (count)")
for k, v in sorted(tab_aff.items(), key=lambda kv: -kv[1]):
    print(f"  {k}  n={v}")
print("\nbomb rows that agree with label: bomb type x label winner x label reason (count)")
for k, v in sorted(tab_bombok.items(), key=lambda kv: -kv[1]):
    print(f"  {k}  n={v}")
print("\nno-bomb rounds where the no-swap score step contradicts the label: (score winner, label winner, label reason)")
for k, v in sorted(tab_inv.items(), key=lambda kv: -kv[1]):
    print(f"  {k}  n={v}")
print("\nno-bomb rounds where the no-swap score step agrees with the label: (label winner, label reason)")
for k, v in sorted(tab_ok.items(), key=lambda kv: -kv[1]):
    print(f"  {k}  n={v}")
