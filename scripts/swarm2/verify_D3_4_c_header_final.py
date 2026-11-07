"""verify_D3_4_c_header_final.py: decide whether round_end winners or freeze score deltas are right after a swap.

Independent arbiter: header t_starters_score_final / ct_starters_score_final (game-level final scoreboard,
not derived from per-round tags). For each regulation-length match (rounds 1..n, n <= 2*s) with a swap at s:
  team X = starting-T team (rounds 1..s on T side, rounds s+1..n on CT side)
  W-based totals: X wins round r if W(r) is T for r <= s, or CT for r > s
  S-based totals: side of round r from the clean score delta(r) = freeze(r+1) - freeze(r); last round uses W
Compare both totals with the header finals in both orientations.

ASSUMPTIONS (marked):
  A11 header t_starters_score_final is the final score of the team that started on T (orientation A);
      orientation B (= starting-CT team) is also tested and reported.
  A12 regulation only: sig == (s,) with n_end <= 2*s; (8,) tests the MR8 reading (swap at 8, 16 rounds).
  A13 round sequence must be contiguous 1..n in round_end.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\verify_D3_4_c_header_final.py
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


fe = (scan("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
      .filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"], maintain_order=True)
      .agg(pl.col("t_score").first(), pl.col("ct_score").first()))
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}
RE = scan("round_end", ["match_key", "round", "winner_team_code", "win_reason_code"]).unique(
    ["match_key", "round"], keep="first", maintain_order=True)
win = {(m, r): (w, rc) for m, r, w, rc in RE.select(["match_key", "round", "winner_team_code", "win_reason_code"]).iter_rows()}
n_end = collections.Counter(m for (m, _) in win)
H = scan("header", ["key", "t_starters_score_final", "ct_starters_score_final", "map_name"])
hdr = {k: (tf, cf, mp) for k, tf, cf, mp in H.iter_rows()}

by_m = collections.defaultdict(list)
for (m, r) in live:
    by_m[m].append(r)


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


DAY_LO, DAY_HI = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) == 3 else (12, 25)
print(f"folder days {DAY_LO}..{DAY_HI}")
matches = [m for m in by_m if DAY_LO <= day_of(m) <= DAY_HI]
sig = {m: swaps_of(m) for m in matches}

# ---------- size of the final-score vs round-count gap, all day 12..25 matches ----------
gap = collections.Counter()
for m in matches:
    tf, cf, _ = hdr[m]
    gap[(str(sig[m]), (tf + cf) - n_end[m])] += 1
print("== gap = (final starter sum) - (round_end rows), by signature (top) ==")
by_sig = collections.defaultdict(list)
for (lab, g), v in gap.items():
    by_sig[lab].append((g, v))
for lab in sorted(by_sig, key=lambda x: -sum(v for _, v in by_sig[x])):
    items = sorted(by_sig[lab], key=lambda x: -x[1])[:6]
    print(f"  {lab:<12} {items}")


def side_seq(m, n, s, variant):
    """Return the side (2 T / 3 CT) of rounds 1..n under a variant. Variants:
    W      : round_end winner for all rounds
    S      : clean score-delta side for rounds != s; swap-derived side for round s; W for round n
    Wflip  : W with round s+1 flipped
    Sfix1  : W for all rounds except round s+1, which takes the S side
    """
    seq = {}
    for r in range(1, n + 1):
        w, rc = win[(m, r)]
        if variant == "W":
            seq[r] = w
            continue
        if (m, r) in live and (m, r + 1) in live:
            t1, c1 = live[(m, r)]
            t2, c2 = live[(m, r + 1)]
            dt, dc = t2 - t1, c2 - c1
            if r == s:
                t, c = t1, c1
                # swap-derived: dt = c - t + [w==3], dc = t - c + [w==2]
                seq[r] = 3 if dt - (c - t) == 1 else 2
                continue
            if clean(dt, dc):
                sd = dside(dt, dc)
                seq[r] = sd if variant == "S" or (variant == "Sfix1" and r == s + 1) else w
                if variant == "Wflip" and r == s + 1:
                    seq[r] = 5 - w
                continue
        seq[r] = w
    return seq


def test(s, label):
    res = collections.Counter()
    for m in matches:
        if sig[m] != (s,):
            continue
        n = n_end[m]
        tf, cf, _ = hdr[m]
        if tf + cf != n:
            res["skip_gap"] += 1
            continue
        if any((m, r) not in win for r in range(1, n + 1)):
            continue
        res["n"] += 1
        for variant in ["W", "S", "Wflip", "Sfix1"]:
            seq = side_seq(m, n, s, variant)
            x = sum(1 for r in range(1, n + 1) if (seq[r] == 2 if r <= s else seq[r] == 3))
            ok_a = (x == tf and n - x == cf)
            ok_b = (x == cf and n - x == tf)
            res[f"{variant}_A"] += int(ok_a)
            res[f"{variant}_B"] += int(ok_b)
    print(f"\n== {label}: swap at {s}, complete matches (header total == round_end rows, rounds contiguous) ==")
    n = res["n"]
    print(f"  complete matches: {n}   skipped (header total != round_end rows): {res['skip_gap']}")
    if not n:
        return
    for variant in ["W", "S", "Wflip", "Sfix1"]:
        for o in ["A", "B"]:
            k = f"{variant}_{o}"
            print(f"  {k:<10} {res[k]}/{n} ({100.0*res[k]/n:.1f}%)")


test(12, "(12,) group")
test(8, "(8,) group")

# ---------- 2. bomb tick placement for round-13 disagreement rows (where S and W differ) ----------
bev = scan("bomb_state", ["match_key", "round", "tick", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"]))
bomb = {(m, r, ev): tk for m, r, ev, tk in bev.select(["match_key", "round", "event_type", "tick"]).iter_rows()}
RE_tick = scan("round_end", ["match_key", "round", "tick"]).unique(["match_key", "round"], keep="first")
end_tick = {(m, r): tk for m, r, tk in RE_tick.select(["match_key", "round", "tick"]).iter_rows()}
fe_tick = {(m, r): tk for m, r, tk in (scan("round_state", ["match_key", "round", "tick", "event_type"])
           .filter(pl.col("event_type") == "round_freeze_end")
           .group_by(["match_key", "round"]).agg(pl.col("tick").min()).iter_rows())}
place = collections.Counter()
for m in matches:
    if sig[m] != (12,) or (m, 13) not in win or (m, 13) not in live or (m, 14) not in live:
        continue
    w, rc = win[(m, 13)]
    t1, c1 = live[(m, 13)]
    t2, c2 = live[(m, 14)]
    dt, dc = t2 - t1, c2 - c1
    if not clean(dt, dc):
        continue
    ds = dside(dt, dc)
    if ds == w:
        continue
    for ev, side in (("bomb_exploded", 2), ("bomb_defused", 3)):
        tk = bomb.get((m, 13, ev))
        if tk is None:
            continue
        pos = ("in r13" if fe_tick.get((m, 13), 0) <= tk <= end_tick.get((m, 13), -1) else
               "after r13 end" if tk > end_tick.get((m, 13), 1 << 60) else "before r13 freeze")
        place[(ev, "S=bomb" if ds == side else "S!=bomb", pos)] += 1
print("\n== bomb_state tick placement for round-13 rows where S != W ==")
for k, v in sorted(place.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v}")
print("  rows with no bomb event among S!=W round-13 rows are not listed.")
