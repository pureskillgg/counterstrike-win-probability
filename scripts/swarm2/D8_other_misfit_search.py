"""D8 follow-up: which halftime schedule and header orientation reconcile the non-round-13 misfit matches?

Universe: D4 complete regulation matches (round count == header total, n <= 24, rounds 1..n), 2024 tome, day 12-25.
For each match, every candidate configuration is tried:
  swaps   = () (no side swap) or (k,) for k in 1..23 (side that started T switches after round k)
  orient  = header as-is, or header totals reversed (t and ct swapped)
  flip    = no change, or one round-13 winner flip (as in D4)
A configuration "fits" a match when the implied starters-of-T round wins equal the header totals.
Reports, for the other_misfit class (D4 definition, 116 matches) and for all complete matches, how many fit under
each configuration, and the best configurations by map.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D8_other_misfit_search.py
"""
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(80)
pl.Config.set_tbl_cols(16)

ROOT = Path(r"G:\pureskill\csds-cs2")
BASE = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
OUTCOMES = ["bomb_defused", "bomb_exploded"]


def files_of(fd):
    return sorted(glob.glob(str(fd / "dataframe_*")))


def read_dir(fd, cols):
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files_of(fd)], how="vertical_relaxed")


def read(ch, cols):
    return read_dir(BASE / f"basic_{ch}{SUF}", cols)


H = read_dir(BASE / "header", ["key", "map_name", "t_starters_score_final", "ct_starters_score_final"])
hmap = dict(zip(H["key"].to_list(), H["map_name"].to_list()))
htot = {k: (int(a), int(b)) for k, a, b in zip(H["key"].to_list(), H["t_starters_score_final"].to_list(),
                                                H["ct_starters_score_final"].to_list())}
RX = read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
bs = read("bomb_state", ["match_key", "round", "event_type"]).filter(pl.col("event_type").is_in(OUTCOMES))
ev = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
OUT = ev.with_columns(
    pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(CT)).otherwise(pl.lit(T)).alias("bside")
).select(["match_key", "round", "bside"])
R4 = RX.join(OUT, on=["match_key", "round"], how="left")
mm13 = set(R4.filter((pl.col("round") == 13) & pl.col("bside").is_not_null()
                     & (pl.col("bside") != pl.col("winner_team_code")))["match_key"].to_list())
by = {}
for r in R4.sort(["match_key", "round"]).iter_rows(named=True):
    by.setdefault(r["match_key"], []).append(r)


def side_a(r, swaps):
    k = sum(1 for s in swaps if s < r)
    return T if k % 2 == 0 else CT


def wins_a(rounds, swaps):
    return sum(1 for x in rounds if int(x["winner_team_code"]) == side_a(int(x["round"]), swaps))


def fits(rounds, tot, swaps, orient, flip):
    tA, tB = tot if orient == 0 else (tot[1], tot[0])
    n = len(rounds)
    wa = wins_a(rounds, swaps)
    if flip:
        r13 = next((x for x in rounds if int(x["round"]) == 13), None)
        if r13 is None:
            return False
        wa = wa - 1 if int(r13["winner_team_code"]) == side_a(13, swaps) else wa + 1
    return (wa, n - wa) == (tA, tB)


CONFIGS = [((), o, f) for o in (0, 1) for f in (False, True)] + \
          [((k,), o, f) for k in range(1, 24) for o in (0, 1) for f in (False, True)]

complete = []
for m, rounds in by.items():
    if m not in htot:
        continue
    tA, tB = htot[m]
    n = len(rounds)
    if not (n == tA + tB and n <= 24 and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    complete.append((m, rounds, (tA, tB)))
print(f"complete regulation matches: {len(complete)}")

# D4 classes (MR12 baseline with one r13 flip)
cls = {}
for m, rounds, tot in complete:
    if m in mm13:
        cls[m] = "affected"
        continue
    if fits(rounds, tot, (12,), 0, False):
        cls[m] = "fits"
    elif fits(rounds, tot, (12,), 0, True):
        cls[m] = "r13_flip_misfit"
    else:
        cls[m] = "other_misfit"
cnt = {}
for c in cls.values():
    cnt[c] = cnt.get(c, 0) + 1
print("D4-style classes:", dict(sorted(cnt.items())), "(D4 log: affected 211, r13_flip_misfit 825, other_misfit 116, fits 94)")

other = [(m, rounds, tot) for m, rounds, tot in complete if cls[m] == "other_misfit"]
print(f"other_misfit matches: {len(other)}")
print("other_misfit by map:", sorted(
    {mp: sum(1 for m, _, _ in other if hmap.get(m) == mp) for mp in {hmap.get(m) for m, _, _ in other}}.items()))

res_other = []
res_all = []
for cfg in CONFIGS:
    swaps, o, f = cfg
    n_o = sum(1 for m, rounds, tot in other if fits(rounds, tot, swaps, o, f))
    n_a = sum(1 for m, rounds, tot in complete if fits(rounds, tot, swaps, o, f))
    res_other.append((n_o, cfg))
    res_all.append((n_a, cfg))
res_other.sort(key=lambda x: -x[0])
res_all.sort(key=lambda x: -x[0])
print("\nbest configurations for other_misfit matches (fits, swaps, orient(0=as-is,1=reversed), r13flip):")
for n_o, cfg in res_other[:12]:
    print(f"  {n_o:>4} / {len(other)}   swaps={cfg[0]} orient={cfg[1]} flip={cfg[2]}")
print("\nbest configurations for all complete matches:")
for n_a, cfg in res_all[:8]:
    print(f"  {n_a:>4} / {len(complete)}   swaps={cfg[0]} orient={cfg[1]} flip={cfg[2]}")

print("\nno-swap, as-is, no flip (other_misfit fits):", sum(1 for m, r, t in other if fits(r, t, (), 0, False)))
print("swap-after-8, as-is, no flip (other_misfit fits):", sum(1 for m, r, t in other if fits(r, t, (8,), 0, False)))
print("swap-after-8, as-is, flip (other_misfit fits):", sum(1 for m, r, t in other if fits(r, t, (8,), 0, True)))
print("swap-after-12 reversed header, no flip (other_misfit fits):", sum(1 for m, r, t in other if fits(r, t, (12,), 1, False)))
print("swap-after-12 reversed header, flip (other_misfit fits):", sum(1 for m, r, t in other if fits(r, t, (12,), 1, True)))

best_cfg = res_other[0][1]
print(f"\nbest config for other_misfit = {best_cfg}; fit rate by map (other_misfit matches):")
for mp in sorted({hmap.get(m) for m, _, _ in other}):
    sub = [(m, r, t) for m, r, t in other if hmap.get(m) == mp]
    k = sum(1 for m, r, t in sub if fits(r, t, best_cfg[0], best_cfg[1], best_cfg[2]))
    print(f"  {mp:<12} n={len(sub):>3} fit={k:>3}")
print("\nDONE")
