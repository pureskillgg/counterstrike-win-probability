"""verify D3-3: which source is wrong at round 13 (and 9, 25, 28)? 2024 G: tome, read-only.

Independent sources, none of them the claim's side-delta:
  W   round_end winner_team_code (side code) and win_reason_code
  B   bomb outcome from bomb_state: bomb_exploded -> T, bomb_defused -> CT (side)
  P   team identity per round from player_info (team_code side per player, per round)
  F   freeze-end scores (t_score, ct_score), and the raw pair (t_score_raw, ct_score_raw)

Checks:
  1. Empirical swap transitions r->r+1 from P (player sets), by day group, vs the claim's schedule (12, 27).
  2. Team-based cumulative consistency: freeze(r) t/ct vs cumulative team wins from W (rounds < r) with teamOnT from P.
  3. Team-based delta winner from freeze(r+1) - freeze(r) mapped to team via P, vs W (team form of the claim's delta).
  4. Bomb outcome B vs W and vs reason code, by round.
  5. Raw vs normalised score fields: rows where t_score != t_score_raw or ct_score != ct_score_raw, by round.
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


def chan_files(ch):
    return sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))


def scan(ch):
    return pl.scan_parquet(chan_files(ch), missing_columns="insert", extra_columns="ignore")


def day_of(m):
    return int(m.split("/")[3])


def grp(m):
    return "day12_25" if day_of(m) >= 12 else "day01_07"


# ---------- W: round_end (first by tick), with reason code
RE = (scan("round_end").select(["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).collect()
      .sort("tick").unique(["match_key", "round"], keep="first", maintain_order=True))
W = {(m, r): (w, rc) for m, r, w, rc in RE.select(["match_key", "round", "winner_team_code", "win_reason_code"]).iter_rows()}

# ---------- F: freeze_end rows (first by tick), normalised and raw
RS = scan("round_state").filter(pl.col("event_type") == "round_freeze_end").select(
    ["match_key", "round", "tick", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).collect()
FE = RS.sort("tick").group_by(["match_key", "round"]).agg(
    pl.col("t_score").first(), pl.col("ct_score").first(),
    pl.col("t_score_raw").first(), pl.col("ct_score_raw").first())
F = {(m, r): (t, c, tr, cr) for m, r, t, c, tr, cr in FE.select(
    ["match_key", "round", "t_score", "ct_score", "t_score_raw", "ct_score_raw"]).iter_rows()}

# ---------- P: side sets per round from player_info
PI = scan("player_info").select(["match_key", "round", "player_id", "team_code"]).collect()
TS, CS = collections.defaultdict(set), collections.defaultdict(set)
for m, r, pid, tc in PI.iter_rows():
    if tc == T:
        TS[(m, r)].add(pid)
    elif tc == CT:
        CS[(m, r)].add(pid)

# ---------- B: bomb outcome per round (side that won by bomb)
BS = scan("bomb_state").filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).select(
    ["match_key", "round", "tick", "event_type"]).collect()
B = {}
bomb_multi = 0
for m, r, ev in BS.select(["match_key", "round", "event_type"]).iter_rows():
    side = T if ev == "bomb_exploded" else CT
    if (m, r) in B and B[(m, r)] != side:
        bomb_multi += 1
    B.setdefault((m, r), side)
print(f"bomb outcome rows: {BS.height}, (match,round) with a bomb outcome: {len(B)}, conflicting: {bomb_multi}")

# ---------- team mapping per match: A = team on T side at first round with player info
matches = sorted({m for (m, _) in W})
rounds_by_m = collections.defaultdict(list)
for (m, r) in W:
    rounds_by_m[m].append(r)
for m in rounds_by_m:
    rounds_by_m[m].sort()

teamOnT = {}  # (m,r) -> 'A' or 'B'
Aset = {}
for m in matches:
    first = None
    for r in rounds_by_m[m]:
        if (m, r) in TS and (m, r) in CS and TS[(m, r)] and CS[(m, r)]:
            first = (m, r)
            break
    if first is None:
        continue
    A = TS[first]
    Aset[m] = A
    for r in rounds_by_m[m]:
        if (m, r) in TS and (m, r) in CS and TS[(m, r)] and CS[(m, r)]:
            tA = len(TS[(m, r)] & A)
            cA = len(CS[(m, r)] & A)
            teamOnT[(m, r)] = "A" if tA >= cA else "B"

# ---------- 1. empirical swap transitions from P
sw = collections.Counter()
for m, A in Aset.items():
    for r in rounds_by_m[m]:
        if (m, r) in teamOnT and (m, r + 1) in teamOnT:
            same = teamOnT[(m, r)] == teamOnT[(m, r + 1)]
            sw[(grp(m), r, "no_swap" if same else "swap")] += 1
print("\n== 1. empirical side changes between round r and r+1 (from player_info), rounds 6..30")
for g in ["day01_07", "day12_25"]:
    row = []
    for r in range(6, 31):
        ns = sw[(g, r, "no_swap")]
        sp = sw[(g, r, "swap")]
        if ns + sp:
            row.append((r, sp, ns + sp))
    print(f"  {g} (r, swaps, transitions): {row}")

# ---------- 2. freeze consistency with team-based cumulative W
cum_ok = collections.Counter()
cum_bad_examples = []
for m, A in Aset.items():
    team_win = {}
    for r in rounds_by_m[m]:
        if (m, r) in teamOnT and (m, r) in W:
            w = W[(m, r)][0]
            on_t = teamOnT[(m, r)]
            wt = ("A" if on_t == "A" else "B") if w == T else ("B" if on_t == "A" else "A")
            team_win[r] = wt
    # cumulative before each round, only if rounds 1..r-1 all known
    cA = cB = 0
    expect = {}
    prev_ok = True
    for r in range(1, max(rounds_by_m[m]) + 1):
        if (m, r) in teamOnT and (m, r) in F:
            if prev_ok:
                tA_exp = cA if teamOnT[(m, r)] == "A" else cB
                cB_exp = cB if teamOnT[(m, r)] == "A" else cA
                t, c, _, _ = F[(m, r)]
                ok = (t == tA_exp and c == cB_exp)
                cum_ok[(grp(m), r, "ok" if ok else "bad")] += 1
                if not ok and len(cum_bad_examples) < 6 and r in (13, 14):
                    cum_bad_examples.append((m[-40:], r, (t, c), (tA_exp, cB_exp), teamOnT[(m, r)]))
        if r in team_win:
            if team_win[r] == "A":
                cA += 1
            else:
                cB += 1
        else:
            prev_ok = False
print("\n== 2. freeze(r) vs team-cumulative from W (rounds < r known): ok / bad, rounds 1..30")
for g in ["day01_07", "day12_25"]:
    rows = []
    for r in range(1, 31):
        ok = cum_ok[(g, r, "ok")]
        bad = cum_ok[(g, r, "bad")]
        if ok + bad:
            rows.append((r, bad, ok + bad))
    print(f"  {g} (round, bad, total): {rows}")
print("  examples at rounds 13/14 (match tail, round, freeze (t,c), expected (T-team, CT-team), teamOnT):")
for e in cum_bad_examples:
    print("   ", e)

# ---------- 3. team-based delta from freeze(r+1) - freeze(r) vs W (team form)
tdel = collections.Counter()
for m, A in Aset.items():
    for r in rounds_by_m[m]:
        if (m, r) not in teamOnT or (m, r + 1) not in teamOnT:
            continue
        if (m, r) not in F or (m, r + 1) not in F or (m, r) not in W:
            continue
        t1, c1, _, _ = F[(m, r)]
        t2, c2, _, _ = F[(m, r + 1)]
        on1, on2 = teamOnT[(m, r)], teamOnT[(m, r + 1)]
        # team score at freeze: A-score and B-score
        a1 = t1 if on1 == "A" else c1
        b1 = c1 if on1 == "A" else t1
        a2 = t2 if on2 == "A" else c2
        b2 = c2 if on2 == "A" else t2
        da, db = a2 - a1, b2 - b1
        if da == 1 and db == 0:
            dteam = "A"
        elif db == 1 and da == 0:
            dteam = "B"
        else:
            tdel[(grp(m), r, "unclear")] += 1
            continue
        w = W[(m, r)][0]
        wteam = ("A" if on1 == "A" else "B") if w == T else ("B" if on1 == "A" else "A")
        tdel[(grp(m), r, "ok" if dteam == wteam else "bad")] += 1
print("\n== 3. team-based delta (freeze r+1 - freeze r via player_info mapping) vs W (team form), rounds 1..30")
for g in ["day01_07", "day12_25"]:
    rows = []
    for r in range(1, 31):
        ok = tdel[(g, r, "ok")]
        bad = tdel[(g, r, "bad")]
        unc = tdel[(g, r, "unclear")]
        if ok + bad + unc:
            rows.append((r, bad, ok + bad, unc))
    print(f"  {g} (round, bad, valid, unclear): {rows}")

# ---------- 4. bomb outcome vs W and vs reason code
bw = collections.Counter()
for (m, r), side in B.items():
    if (m, r) not in W:
        bw[(grp(m), r, "no_round_end")] += 1
        continue
    w, rc = W[(m, r)]
    bw[(grp(m), r, "bomb_eq_W" if side == w else "bomb_ne_W")] += 1
print("\n== 4. bomb outcome (bomb_state) vs round_end winner, rounds 1..30 (bad = bomb_ne_W)")
for g in ["day01_07", "day12_25"]:
    rows = []
    for r in range(1, 31):
        ne = bw[(g, r, "bomb_ne_W")]
        eq = bw[(g, r, "bomb_eq_W")]
        if ne + eq:
            rows.append((r, ne, ne + eq))
    print(f"  {g} (round, bomb_ne_W, bomb_rounds): {rows}")

rc_bad = collections.Counter()
for (m, r), side in B.items():
    if (m, r) in W:
        w, rc = W[(m, r)]
        rc_bad[(grp(m), r, "reason_T" if rc in (1, 9, 13) else ("reason_CT" if rc in (7, 8, 11, 12, 17, 18) else "other"))] += 1
print("\n   reason-code class of rounds that have a bomb outcome, day12_25, rounds 9/13/25/28:")
print("  ", {r: {k[2]: v for k, v in rc_bad.items() if k[0] == 'day12_25' and k[1] == r} for r in (9, 13, 25, 28)})

# ---------- 5. raw vs normalised freeze fields
raw_diff = collections.Counter()
for (m, r), (t, c, tr, cr) in F.items():
    raw_diff[(grp(m), r, "differs" if (t != tr or c != cr) else "same")] += 1
print("\n== 5. freeze rows where normalised (t_score, ct_score) != raw pair, rounds 1..30")
for g in ["day01_07", "day12_25"]:
    rows = []
    for r in range(1, 31):
        d = raw_diff[(g, r, "differs")]
        s = raw_diff[(g, r, "same")]
        if d + s:
            rows.append((r, d, d + s))
    print(f"  {g} (round, differs, total): {rows}")
