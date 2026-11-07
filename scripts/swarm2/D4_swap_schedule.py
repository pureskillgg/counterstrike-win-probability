r"""D4 diagnostic: halftime swap after round 12 (S12) vs after round 13 (S13), 2024 complete regulation matches.

Winners from round_end (end) or with bomb-side substitution (bomb). Classes: affected (label-based r13 bomb
mismatch), flip_nobomb (S12 misfit fixed by flipping r13, no r13 bomb row), other_misfit, fits (S12 fit).
Shows which halftime schedule reconciles each class with the header totals.
Run: "G:\pureskill\csds-cs2\.pixi\envs\default\python.exe" -I -X utf8 scripts\swarm2\D4_swap_schedule.py
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
by = {}
for r in R.sort(["match_key", "round"]).iter_rows(named=True):
    by.setdefault(r["match_key"], []).append(r)
agg = {}
for m, rounds in by.items():
    h = Hd.get(m)
    if h is None:
        continue
    tA, tB = int(h["t_starters_score_final"]), int(h["ct_starters_score_final"])
    n = len(rounds)
    if not (n == tA + tB and n <= 24 and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    has_r13_bomb = any(int(x["round"]) == 13 and x["bside"] is not None for x in rounds)
    r13 = next((x for x in rounds if int(x["round"]) == 13), None)
    fits12 = (sum(1 for x in rounds if int(x["winner_team_code"]) == (T if int(x["round"]) <= 12 else CT)), 0)
    wa12 = fits12[0]
    fits = (wa12, n - wa12) == (tA, tB)
    if m in mm13:
        cls = "affected"
    elif fits:
        cls = "fits_S12"
    elif r13 is None:
        cls = "other_misfit"
    else:
        nwa = wa12 - 1 if int(r13["winner_team_code"]) == CT else wa12 + 1
        cls = "flip_nobomb" if ((nwa, n - nwa) == (tA, tB) and not has_r13_bomb) else "other_misfit"
    for sname, sw in {"S12": 12, "S13": 13}.items():
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
            k = (cls, sname, src)
            agg.setdefault(k, {"ok": 0, "swapped": 0, "bad": 0})
            agg[k][st] += 1
print("complete regulation matches: class x halftime schedule x winner source (ok / swapped / bad)")
for k in sorted(agg):
    v = agg[k]
    tot = sum(v.values())
    print(f"  {k[0]:<12} {k[1]:<4} {k[2]:<5} n={tot:>5} ok={v['ok']:>5} swapped={v['swapped']:>4} bad={v['bad']:>5} "
          f"ok_rate={v['ok'] / tot:.3f}")
