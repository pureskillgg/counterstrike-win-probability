"""verify_D3_4_b_round13.py: three-way arbiter for the round after a detected swap (2024 G: tome, folder days 12..25).

Compares, per round r, three independent signals:
  S  score delta side: clean delta(r) = freeze(r+1) - freeze(r) (A4 of verify_D3_4_a_scores.py)
  W  round_end winner_team_code (2 = T, 3 = CT)
  B  bomb_state outcome: bomb_exploded -> T (2), bomb_defused -> CT (3); the same rule as scripts/swarm/e2_side_check_r13.py
  R  round_end win_reason_code mapped to a side (A2 of verify_D3_4_a_scores.py)
Also reproduces the e2 round-13 bomb mismatch (claim: 434 of 605, days 12..25) with its own filters.

ASSUMPTIONS (marked):
  A8 e2 filters reproduced as written: round_end.player_count == 20 and bomb_state exploded/defused present.
  A9 sample sequences print raw freeze rows; no assumption beyond A4.
  A10 "undetected" = matches with no swap signature at 12 whose observed delta(12) is consistent with a swap
      (swap prediction equals the observed delta) but not with a plain no-swap.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_4_b_round13.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(40)
pl.Config.set_tbl_width_chars(200)

TOME = Path(r"G:\pureskill\csds-cs2\tomes\2024-02-g\tome\csds")
SUF = ".2023-12-28,2024-02-25"
SIDE_BY_REASON = {1: 2, 9: 2, 13: 2, 17: 2, 7: 3, 8: 3, 11: 3, 12: 3, 18: 3}


def scan(ch, cols):
    folder = TOME / "header" if ch == "header" else TOME / f"basic_{ch}{SUF}"
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.scan_parquet(files).select(cols).collect()


def day_of(m):
    return int(m.split("/")[3])


def clean(dt, dc):
    return (dc == 1 and dt == 0) or (dt == 1 and dc == 0)


def dside(dt, dc):
    if dt == 1 and dc == 0:
        return 2
    if dc == 1 and dt == 0:
        return 3
    return None


def pred_noswap(w):
    return (1, 0) if w == 2 else (0, 1)


def pred_swap(t, c, w):
    return (c - t + (w == 3), t - c + (w == 2))


# ---------- load ----------
RS_all = scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "t_score_raw", "ct_score_raw"])
print("round_state event types:", RS_all.group_by("event_type").len().height)
fe_rows = RS_all.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
fe = fe_rows.group_by(["match_key", "round"], maintain_order=True).agg(
    pl.col("t_score").first(), pl.col("ct_score").first(),
    pl.col("t_score_raw").first(), pl.col("ct_score_raw").first(), pl.col("tick").first().alias("fe_tick"))
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
live_raw = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score_raw", "ct_score_raw"]).iter_rows()}
fe_tick = {(m, r): tk for m, r, tk in fe.select(["match_key", "round", "fe_tick"]).iter_rows()}

RE_all = scan("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
RE = RE_all.unique(["match_key", "round"], keep="first", maintain_order=True)
win = {(m, r): (w, rc, pc, tk) for m, r, w, rc, pc, tk in
       RE.select(["match_key", "round", "winner_team_code", "win_reason_code", "player_count", "tick"]).iter_rows()}
end_tick = {(m, r): tk for (m, r), (_, _, _, tk) in win.items()}

BS = scan("bomb_state", ["match_key", "round", "tick", "event_type"])
print("bomb_state event types:", sorted(BS["event_type"].unique().to_list()))
bev = BS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
bomb_out = collections.defaultdict(set)
bomb_tick = {}
for m, r, ev, tk in bev.select(["match_key", "round", "event_type", "tick"]).iter_rows():
    bomb_out[(m, r)].add(ev)
    bomb_tick[(m, r, ev)] = tk
print(f"bomb outcome rows (m,r) with exploded/defused: {len(bomb_out)}")

by_m = collections.defaultdict(list)
for (m, r) in live:
    by_m[m].append(r)
matches = [m for m in by_m if day_of(m) >= 12]


def swaps_of(m):
    out = []
    for r in sorted(by_m[m]):
        if (m, r + 1) not in live:
            continue
        t1, c1 = live[(m, r)]
        t2, c2 = live[(m, r + 1)]
        dt, dc = t2 - t1, c2 - c1
        if clean(dt, dc) or (dt, dc) == (0, 0):
            continue
        if dt * dc <= 0:
            out.append(r)
    return tuple(out)


sig = {m: swaps_of(m) for m in matches}

# ---------- 1. reproduce e2 round-13 bomb mismatch with its own filters ----------
e2 = collections.Counter()
e2_rows = []
for (m, r), evs in bomb_out.items():
    if day_of(m) < 12 or (m, r) not in win:
        continue
    w, rc, pc, _ = win[(m, r)]
    if pc != 20:
        continue
    bside = 3 if "bomb_defused" in evs else 2
    mis = bside != w
    key = "r13" if r == 13 else "other"
    e2[(key, "n")] += 1
    e2[(key, "mis")] += int(mis)
    if r == 13:
        e2_rows.append((m, bside, w, rc))
print("\n== 1. e2 bomb mismatch reproduction (folder days 12..25, player_count==20) ==")
print(f"  round 13: {e2[('r13', 'mis')]}/{e2[('r13', 'n')]} mismatch (claim 434/605)")
print(f"  other rounds: {e2[('other', 'mis')]}/{e2[('other', 'n')]} mismatch (claim 41/14602)")

# ---------- 2. three-way cross-tab on round 13 ----------
tri = collections.Counter()
for (m, r) in [(m, 13) for m in matches]:
    if (m, 13) not in win or (m, 13) not in live or (m, 14) not in live:
        continue
    w, rc, pc, _ = win[(m, 13)]
    t1, c1 = live[(m, 13)]
    t2, c2 = live[(m, 14)]
    dt, dc = t2 - t1, c2 - c1
    ds = dside(dt, dc) if clean(dt, dc) else "nc"
    evs = bomb_out.get((m, 13), set())
    bs = ("exp" if "bomb_exploded" in evs else "") + ("def" if "bomb_defused" in evs else "")
    bs = bs or "none"
    group = str(sig[m]) if sig[m] else "()"
    tri[(group, "S" if ds != "nc" else "nc", "S=W" if ds == w else "S!=W",
         "S=R" if ds == SIDE_BY_REASON[rc] else "S!=R", "W=R" if w == SIDE_BY_REASON[rc] else "W!=R",
         bs)] += 1
print("\n== 2. round-13 three-way cross-tab (group, delta clean?, S vs W, S vs R, W vs R, bomb_state) ==")
for k, v in sorted(tri.items(), key=lambda x: -x[1]):
    if k[0] in ("(12,)", "()") and v >= 3:
        print(f"  {str(k):<90} {v}")

# ---------- 3. undetected vs detected at boundary 12: k=1 anomaly split ----------
split = collections.defaultdict(collections.Counter)
for m in matches:
    if (m, 12) not in live or (m, 13) not in live or (m, 12) not in win or (m, 13) not in win:
        continue
    t, c = live[(m, 12)]
    t2, c2 = live[(m, 13)]
    obs = (t2 - t, c2 - c)
    w12 = win[(m, 12)][0]
    rc12 = win[(m, 12)][1]
    if sig[m] == ():
        if obs != pred_noswap(w12):
            cat = "()_obs!=noswap"
        elif pred_swap(t, c, w12) != pred_noswap(w12):
            cat = "()_unambiguous_noswap"
        else:
            cat = "()_ambiguous"
    elif sig[m] == (12,):
        cat = "(12,)_detected"
    else:
        continue
    if (m, 13) in win and (m, 14) in live and (m, 13) in live:
        t1, c1 = live[(m, 13)]
        t3, c3 = live[(m, 14)]
        dt, dc = t3 - t1, c3 - c1
        if clean(dt, dc):
            split[cat]["d13_clean"] += 1
            split[cat]["d13_neq_R13"] += int(dside(dt, dc) != SIDE_BY_REASON[win[(m, 13)][1]])
            split[cat]["d13_neq_W13"] += int(dside(dt, dc) != win[(m, 13)][0])
        split[cat]["n"] += 1
print("\n== 3. boundary-12 split: is the round-13 delta anomaly tied to a swap? ==")
for cat in sorted(split):
    a = split[cat]
    print(f"  {cat:<24} n={a['n']:<6} d13_clean={a['d13_clean']:<6} d13!=R13={a['d13_neq_R13']:<6} d13!=W13={a['d13_neq_W13']}")

# ---------- 4. raw score fields at the same delta ----------
raw = collections.Counter()
for m in matches:
    if sig[m] != (12,):
        continue
    if (m, 13) not in live_raw or (m, 14) not in live_raw:
        continue
    t1, c1 = live_raw[(m, 13)]
    t2, c2 = live_raw[(m, 14)]
    dt, dc = t2 - t1, c2 - c1
    if clean(dt, dc):
        raw["raw_clean"] += 1
        raw["raw_neq_R13"] += int(dside(dt, dc) != SIDE_BY_REASON[win[(m, 13)][1]])
print(f"\n== 4. raw (t_score_raw/ct_score_raw) delta(13) on (12,) matches: {dict(raw)}")

# ---------- 5. sample sequences ----------
def show(m):
    print(f"\n  match {m}")
    for r in range(10, 16):
        f = live.get((m, r))
        e = win.get((m, r))
        ev = sorted(bomb_out.get((m, r), []))
        print(f"    r={r:<3} freeze_end tick={fe_tick.get((m, r))} score(t,c)={f}  raw={live_raw.get((m, r))}  "
              f"round_end W={e[0] if e else None} reason={e[1] if e else None} pc={e[2] if e else None} bomb={ev}")


samples = [m for m in matches if sig[m] == (12,)][:2]
print("\n== 5. sample sequences (rounds 10..15) for (12,) matches ==")
for m in samples:
    show(m)
