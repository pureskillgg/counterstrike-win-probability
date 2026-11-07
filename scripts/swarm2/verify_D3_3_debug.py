"""verify D3-3 debug: per-round trace for a few matches (2024 G: tome), plus tick alignment of bomb outcomes.

Prints: round, W (round_end winner), reason, freeze (t,c) and raw, player_info side overlaps, teamOnT,
bomb outcome, and the ticks of freeze_end / round_end / bomb events so round labels can be checked
against the tick order. Read-only.
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
TRACE_MATCHES = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def chan_files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


def scan(ch):
    return pl.scan_parquet(chan_files(ch), missing_columns="insert", extra_columns="ignore")


RE = (scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()
      .sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True))
W = {(m, r): (w, rc, tk) for m, r, w, rc, tk in RE.select(
    ["match_key", "round", "winner_team_code", "win_reason_code", "tick"]).iter_rows()}

RS = scan("round_state").filter(pl.col("event_type") == "round_freeze_end").select(
    ["match_key", "round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).collect()
FE = RS.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("tick").first().alias("ftick"), pl.col("t_score").first(), pl.col("ct_score").first(),
    pl.col("t_score_raw").first(), pl.col("ct_score_raw").first())
F = {(m, r): (ft, t, c, tr, cr) for m, r, ft, t, c, tr, cr in FE.select(
    ["match_key", "round", "ftick", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).iter_rows()}

PI = scan("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
TS, CS = collections.defaultdict(set), collections.defaultdict(set)
for m, r, pid, tc in PI.iter_rows():
    if tc == T:
        TS[(m, r)].add(pid)
    elif tc == CT:
        CS[(m, r)].add(pid)

BS = scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).select(
    ["match_key", "round", "tick", "event_type"]).collect()
bomb_by = {}
for m, r, tk, ev in BS.select(["match_key", "round", "tick", "event_type"]).iter_rows():
    bomb_by.setdefault((m, r), (tk, T if ev == "bomb_exploded" else CT))

rounds_by_m = collections.defaultdict(list)
for (m, r) in W:
    rounds_by_m[m].append(r)
for m in rounds_by_m:
    rounds_by_m[m].sort()

# ---- trace a few matches that have a round-13 disagreement with the side delta
trace_pick = []
for (m, r), (ft, t, c, tr, cr) in sorted(F.items()):
    if r == 13 and (m, r + 1) in F and m.split("/")[3] >= "12" and (m, r) in W:
        t2, c2 = F[(m, r + 1)][1], F[(m, r + 1)][2]
        dt, dc = t2 - t, c2 - c
        if (dc == 1 and dt == 0 and W[(m, r)][0] != CT) or (dt == 1 and dc == 0 and W[(m, r)][0] != T):
            trace_pick.append(m)
        if len(trace_pick) >= TRACE_MATCHES:
            break

for m in trace_pick:
    print(f"\n==== match ...{m[-44:]}  rounds {rounds_by_m[m][0]}..{rounds_by_m[m][-1]}")
    print(" r   W  rsn  ftick  frz(t,c)   raw(t,c)    W_tick  Tset/Cset overlap(A=T@first)  bomb(tick,side)")
    A = None
    for r in rounds_by_m[m]:
        if (m, r) in TS and TS[(m, r)] and CS[(m, r)] and A is None:
            A = TS[(m, r)]
    for r in range(1, rounds_by_m[m][-1] + 1):
        w = W.get((m, r))
        f = F.get((m, r))
        ts, cs = TS.get((m, r), set()), CS.get((m, r), set())
        ov = (len(ts & A), len(cs & A)) if A else None
        b = bomb_by.get((m, r))
        print(f"{r:2d}  {w[0] if w else '-'}  {w[1] if w else '-':>3}  {f[0] if f else '-':>6}  "
              f"{str((f[1], f[2])) if f else '-':<10} {str((f[3], f[4])) if f else '-':<10} "
              f"{w[2] if w else '-':>7}  {len(ts)}/{len(cs)}  {str(ov) if ov else '-':<12}  "
              f"{str(b) if b else '-'}")

# ---- tick alignment of bomb outcomes: which round_end tick is the bomb event nearest to
align = collections.Counter()
ex = []
for (m, r), (tk, side) in bomb_by.items():
    if (m, r) not in W:
        continue
    # round_end of the round whose window contains tk: first round_end tick >= tk, and its label
    later = [(W[(m, q)][2], q) for q in (r - 1, r, r + 1) if (m, q) in W]
    later.sort()
    nxt = [(tt, q) for tt, q in later if tt >= tk]
    if not nxt:
        align[(m.split("/")[3] >= "12", "no_later_end")] += 1
        continue
    tt, q = nxt[0]
    grp = "day12_25" if m.split("/")[3] >= "12" else "day01_07"
    if r == 13:
        align[(grp, r, "bomb_end_label_offset", q - r)] += 1
        align[(grp, r, "bomb_side_eq_W_at_label" if side == W[(m, r)][0] else "bomb_side_ne_W_at_label")] += 1
        if side != W[(m, r)][0] and side == W[(m, q)][0] and q != r and len(ex) < 4:
            ex.append((m[-40:], r, q, side))
print("\n== bomb outcome vs the round_end whose tick is the first at or after the bomb event (round 13 bomb rows):")
for k, v in sorted(align.items(), key=lambda kv: str(kv[0])):
    print("  ", k, v)
print("  examples where bomb side equals W of a neighbouring label (match tail, label, neighbour, side):", ex)

# ---- ordering check: freeze_end(r) < round_end(r) < freeze_end(r+1)
order = collections.Counter()
for m in rounds_by_m:
    for r in rounds_by_m[m]:
        if (m, r) in F and (m, r) in W and (m, r + 1) in F:
            ok1 = F[(m, r)][0] < W[(m, r)][2]
            ok2 = W[(m, r)][2] < F[(m, r + 1)][0]
            grp = "day12_25" if m.split("/")[3] >= "12" else "day01_07"
            order[(grp, r, "ok" if ok1 and ok2 else "bad")] += 1
print("\n== tick order freeze(r) < round_end(r) < freeze(r+1), bad counts by round (rounds 8..14):")
for g in ["day01_07", "day12_25"]:
    print("  ", g, [(r, order[(g, r, 'bad')], order[(g, r, 'bad')] + order[(g, r, 'ok')]) for r in range(8, 15)])
