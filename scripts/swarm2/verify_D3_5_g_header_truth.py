"""D3-5 verifier, step G: independent truth test against header final scores (2024 G: tome).

Header t_starters_score_final / ct_starters_score_final are the final team totals (team that started on T, team
that started on CT). Two independent reconstructions of the final totals are compared with them:
  FIELD  : final side fields = round_state freeze_end of round N, plus the round-N winner (round_end).
           Compared order-free: {t_fin, c_fin} == {hdrT, hdrCT} (no swap rule needed).
  WINNER : team totals from round_end winners. Halftime round h from the ENGINE phase marker (first round with
           phase == 'second'; ASSUMPTION h=12 if missing). Side of team A (started on T): T for r<=h, CT for h<r<=2h,
           then OT blocks of 3 rounds flipping (ASSUMPTION). Compared order-free with {hdrT, hdrCT}.
Only complete matches (N == hdrT + hdrCT) are scored. Results split by halftime type (h = 8 or 12) and day group.
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3


def rd(ch, cols):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUF}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in sorted(glob.glob(str(folder / "dataframe_*")))],
                     how="vertical_relaxed")


H = rd("header", ["key", "t_starters_score_final", "ct_starters_score_final"])
hdr = {k: (t, c) for k, t, c in H.iter_rows()}
RE = rd("round_end", ["match_key", "round", "winner_team_code", "player_count"])
RE = (RE.with_columns((pl.col("player_count") == 20).cast(pl.Int8).alias("p20"))
      .sort(["match_key", "round", "p20"], descending=[False, False, True])
      .unique(["match_key", "round"], keep="first"))
RS = rd("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
sec = (RS.filter(pl.col("phase") == "second").group_by("match_key").agg(pl.col("round").min().alias("h")))
halft = {m: h for m, h in sec.iter_rows()}

W = collections.defaultdict(dict)
for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    W[m][r] = w
F = collections.defaultdict(dict)
for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows():
    F[m][r] = (t, c)
day = lambda m: int(m.split("/")[3])
grp = lambda m: "12-25" if day(m) >= 12 else "<12"


def a_side(r, h):
    if r <= h:
        return T
    if r <= 2 * h:
        return CT
    k = (r - 2 * h - 1) // 3
    return T if k % 2 == 1 else CT  # ASSUMPTION: OT block 0 starts on the side team A was not on at reg end


res = collections.Counter()
ex = collections.defaultdict(list)
for m, wins in W.items():
    if m not in hdr:
        res["no_header"] += 1
        continue
    ht, hc = hdr[m]
    if ht is None or hc is None:
        res["header_null"] += 1
        continue
    rounds = sorted(wins)
    N = len(rounds)
    if N != ht + hc or rounds != list(range(1, N + 1)):
        res["not_complete_or_gap"] += 1
        continue
    h = halft.get(m, 12)
    typ = f"h{h}" if m in halft else "h_unknown"
    key = f"{grp(m)}|{typ}"
    res[f"complete|{key}"] += 1
    # WINNER reconstruction: team A started on T
    a = sum(1 for r in rounds if wins[r] == a_side(r, h))
    b = N - a
    w_ok = (a, b) in [(ht, hc), (hc, ht)]
    res[f"winner_ok|{key}"] += int(w_ok)
    if not w_ok and len(ex[f"winner_bad|{key}"]) < 3:
        ex[f"winner_bad|{key}"].append((m, N, ht, hc, a, b, h))
    # FIELD reconstruction: freeze N fields (after N-1 rounds) plus round N winner
    if N in F[m]:
        t_n, c_n = F[m][N]
        wN = wins[N]
        t_f, c_f = (t_n + (wN == T), c_n + (wN == CT))
        f_ok = sorted([t_f, c_f]) == sorted([ht, hc])
        res[f"field_ok|{key}"] += int(f_ok)
        if not f_ok and len(ex[f"field_bad|{key}"]) < 3:
            ex[f"field_bad|{key}"].append((m, N, ht, hc, t_f, c_f))
    else:
        res[f"field_noN|{key}"] += 1
print("counts:")
for k in sorted(res):
    print(f"  {k:<32} {res[k]}")
print("\nexamples of failures:")
for k, v in ex.items():
    print(" ", k, v)
