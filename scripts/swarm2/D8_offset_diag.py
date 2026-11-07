"""D8 diagnostic: one-round offset check for the non-round-13 misfit matches.

For the D4 other_misfit matches (116) and for complete matches whose detected swap signature is (8,) (109),
computes the implied starters-of-T round wins under several swap positions, as a difference from the header
tA (t_starters_score_final). Histogram of differences shows whether the header is off by a fixed amount.
Also prints the round-winner sequence and header totals for three example matches.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D8_offset_diag.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

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

# detected signature (D3 logic) for the same matches
RS = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
bym = collections.defaultdict(list)
for (m, r) in live:
    bym[m].append(r)
sig = {}
for m, rs in bym.items():
    sw = []
    for r in sorted(rs):
        if (m, r + 1) not in live:
            continue
        t1, c1 = live[(m, r)]
        t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if (dc == 1 and dt == 0) or (dt == 1 and dc == 0) or (dt == 0 and dc == 0):
            continue
        if dt * dc <= 0:
            sw.append(r)
    sig[m] = tuple(sw)
del RS, fe, live


def side_a(r, swaps):
    k = sum(1 for s in swaps if s < r)
    return T if k % 2 == 0 else CT


def wa_of(rounds, swaps):
    return sum(1 for x in rounds if int(x["winner_team_code"]) == side_a(int(x["round"]), swaps))


complete = []
for m, rounds in by.items():
    if m not in htot:
        continue
    tA, tB = htot[m]
    n = len(rounds)
    if not (n == tA + tB and n <= 24 and [int(x["round"]) for x in rounds] == list(range(1, n + 1))):
        continue
    r13 = next((x for x in rounds if int(x["round"]) == 13), None)
    wa12 = wa_of(rounds, (12,))
    misfit12 = (wa12, n - wa12) != (tA, tB)
    flip = False
    if misfit12 and r13 is not None:
        nwa = wa12 - 1 if int(r13["winner_team_code"]) == side_a(13, (12,)) else wa12 + 1
        flip = (nwa, n - nwa) == (tA, tB)
    if m in mm13:
        cls = "affected"
    elif not misfit12:
        cls = "fits"
    elif flip:
        cls = "r13_flip_misfit"
    else:
        cls = "other_misfit"
    complete.append((m, rounds, (tA, tB), cls))

other = [c for c in complete if c[3] == "other_misfit"]
s8 = [c for c in complete if sig.get(c[0]) == (8,)]
print(f"other_misfit: {len(other)}; complete matches with detected sig (8,): {len(s8)}")
print("other_misfit detected sig:", collections.Counter(str(sig.get(m, ())) for m, _, _, _ in other).most_common())

for label, group in [("other_misfit", other), ("sig (8,) complete", s8)]:
    print(f"\n{label}: difference wa - tA under swap positions (as-is header orientation)")
    for sw in [(8,), (9,), (12,)]:
        d = collections.Counter(wa_of(r, sw) - t[0] for m, r, t, _ in group)
        print(f"  swaps={sw}: " + ", ".join(f"{k:+d}:{v}" for k, v in sorted(d.items())))
    print(f"  (reversed header) difference wa - tB under swaps (9,):")
    d = collections.Counter(wa_of(r, (9,)) - t[1] for m, r, t, _ in group)
    print("  " + ", ".join(f"{k:+d}:{v}" for k, v in sorted(d.items())))

print("\nexample other_misfit matches (map, sig, header tA-tB, n, winner sequence as A/B under swap-(8,) convention):")
for m, rounds, tot, _ in other[:3]:
    seq = []
    for x in rounds:
        r = int(x["round"])
        seq.append("A" if int(x["winner_team_code"]) == side_a(r, (8,)) else "B")
    print(f"  {m.split('/')[-1][:20]} map={hmap.get(m)} sig={sig.get(m)} header=({tot[0]},{tot[1]}) n={len(rounds)}")
    print("    " + "".join(seq))
print("\nDONE")
