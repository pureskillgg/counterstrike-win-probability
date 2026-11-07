"""A13: are round_state t_score / ct_score side-attached or team-attached? (2026 tome, all matches)

Independent of the bake-off swap schedule. For each pair of consecutive rounds (r, r+1) in a match:
  1. PLAYER SWAP: did the players change sides? Compare each player's team_code in player_info at
     r and r+1 (players with codes 2/3 in both rounds). changed >= 8 and same <= 2 -> swapped;
     same >= 8 and changed <= 2 -> stayed; otherwise unclear.
  2. SCORE PATTERN: winner w = round_end.winner_team_code of round r (side code, 2 = T, 3 = CT).
     noswap_ok: (t, ct)(r+1) == (t(r) + [w==2], ct(r) + [w==3])
     swap_ok:   (t, ct)(r+1) == (ct(r) + [w==3], t(r) + [w==2])   (fields follow the sides)
     Pattern is 'noswap' / 'swap' when only one fits, 'ambig' when both fit, 'neither' otherwise.
Side-attached fields: score swap pattern exactly when the players swapped.
Team-attached fields: no score swap pattern at a player-swap boundary (the field keeps following a team).

Match classes (over unambiguous player-swap boundaries):
  side      : >=1 player-swap boundary, all unambiguous ones show the swap score pattern
  team      : >=1 player-swap boundary, all unambiguous ones show the noswap score pattern
  mixed     : both kinds
  none      : no unambiguous player-swap boundary (for example a match of 12 rounds or fewer)
Day group = the DD folder in the match key (csds/2026/10/DD/<match id>).

Read-only. Output: stdout (captured to logs/swarm2/A13_score_attach.log).
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2")
TOME = ROOT / "tomes/2026-10-01_to_08/tome/csds"
SUFFIX = ".2026-09-18,2026-10-08"


def read(channel, cols):
    folder = TOME / ("header" if channel == "header" else f"basic_{channel}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


print("== A13 score attachment (2026 tome) ==")
hdr = read("header", ["key", "match_date", "max_rounds"])
H = {r["key"]: r for r in hdr.iter_rows(named=True)}
print(f"header matches: {len(H)}")

RE = read("round_end", ["match_key", "round", "winner_team_code", "player_count"])
RS = read("round_state", ["match_key", "round", "event_type", "t_score", "ct_score"])
PI = read("player_info", ["match_key", "round", "player_id", "team_code"])
print(f"round_end rows {RE.height}; round_state rows {RS.height}; player_info rows {PI.height}")

win = {}
for x in RE.iter_rows(named=True):
    win[(x["match_key"], int(x["round"]))] = x["winner_team_code"]

frz = {}
for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
    k = (x["match_key"], int(x["round"]))
    if k not in frz:
        frz[k] = (x["t_score"], x["ct_score"])

players = collections.defaultdict(dict)
for x in PI.iter_rows(named=True):
    tc = x["team_code"]
    if tc in (2, 3):
        players[(x["match_key"], int(x["round"]))][x["player_id"]] = tc

keys_by_match = collections.defaultdict(list)
for (m, r) in frz:
    keys_by_match[m].append(r)

pair_rows = []
for m, rounds in keys_by_match.items():
    for r in sorted(rounds):
        k1, k2 = (m, r), (m, r + 1)
        if k2 not in frz or k1 not in win:
            continue
        w = win[k1]
        if w not in (2, 3):
            continue
        t1, c1 = frz[k1]
        t2, c2 = frz[k2]
        if None in (t1, c1, t2, c2):
            continue
        t1, c1, t2, c2 = int(t1), int(c1), int(t2), int(c2)
        p1, p2 = players.get(k1, {}), players.get(k2, {})
        common = set(p1) & set(p2)
        changed = sum(1 for p in common if p1[p] != p2[p])
        same = len(common) - changed
        if changed >= 8 and same <= 2:
            pmove = "swapped"
        elif same >= 8 and changed <= 2:
            pmove = "stayed"
        else:
            pmove = "unclear"
        cw, tw = int(w == 3), int(w == 2)
        noswap_ok = (t2, c2) == (t1 + tw, c1 + cw)
        swap_ok = (t2, c2) == (c1 + cw, t1 + tw)
        if noswap_ok and swap_ok:
            spat = "ambig"
        elif noswap_ok:
            spat = "noswap"
        elif swap_ok:
            spat = "swap"
        else:
            spat = "neither"
        pair_rows.append({"match": m, "day": m.split("/")[3], "r": r, "max_rounds": H[m]["max_rounds"],
                          "pmove": pmove, "spat": spat, "w": w})

print(f"\nconsecutive-round pairs with freeze rows and a winner: {len(pair_rows)}")
pc = collections.Counter(x["spat"] for x in pair_rows)
print(f"score pattern over all pairs: {dict(pc)}")
pm = collections.Counter(x["pmove"] for x in pair_rows)
print(f"player move over all pairs: {dict(pm)}")
cross = collections.Counter((x["pmove"], x["spat"]) for x in pair_rows)
print("cross-tab (player move, score pattern):")
for k, v in sorted(cross.items()):
    print(f"  {k}: {v}")

# feature_spec section 3 style: unambiguous swap-only / noswap-only / ambiguous counts, and by round
swap_only = [x for x in pair_rows if x["spat"] == "swap"]
noswap_only = [x for x in pair_rows if x["spat"] == "noswap"]
amb = [x for x in pair_rows if x["spat"] == "ambig"]
print(f"\nunambiguous score-swap pairs: {len(swap_only)}; noswap pairs: {len(noswap_only)}; ambig: {len(amb)}")
by_r = collections.Counter((x["max_rounds"], x["r"]) for x in swap_only)
print(f"score-swap-only pairs by (max_rounds, r): {sorted(by_r.items())}")

# match classes
per_match = collections.defaultdict(lambda: collections.Counter())
for x in pair_rows:
    if x["pmove"] != "swapped":
        per_match[x["match"]]["sw_no_pm"] += 0
    if x["pmove"] == "swapped" and x["spat"] == "swap":
        per_match[x["match"]]["side"] += 1
    elif x["pmove"] == "swapped" and x["spat"] == "noswap":
        per_match[x["match"]]["team"] += 1
    elif x["pmove"] == "swapped" and x["spat"] == "neither":
        per_match[x["match"]]["neither_at_swap"] += 1
    elif x["pmove"] == "swapped" and x["spat"] == "ambig":
        per_match[x["match"]]["ambig_at_swap"] += 1
    if x["pmove"] == "stayed" and x["spat"] == "swap":
        per_match[x["match"]]["swap_when_stayed"] += 1
    if x["pmove"] == "stayed" and x["spat"] == "noswap":
        per_match[x["match"]]["ok_when_stayed"] += 1
    if x["pmove"] == "unclear":
        per_match[x["match"]]["unclear_pm"] += 1


def classify(c):
    side, team = c.get("side", 0), c.get("team", 0)
    if side and team:
        return "mixed"
    if side:
        return "side"
    if team:
        return "team"
    return "none"


def summarize(label, keys):
    cls = collections.Counter(classify(per_match[m]) if m in per_match else "none" for m in keys)
    tot = len(keys)
    print(f"  {label}: matches {tot}; classes {dict(cls)}")
    return cls


all_keys = list(H.keys())
day_group = {"01-02": lambda k: k.split("/")[3] in ("01", "02"),
             "03-10": lambda k: k.split("/")[3] not in ("01", "02")}
print("\nMatch classes (header matches; 'none' includes matches with no unambiguous swap boundary):")
summarize("all 2026 matches", all_keys)
for name, fn in day_group.items():
    keys = [k for k in all_keys if fn(k)]
    summarize(f"day {name}", keys)
ge13 = [k for k in all_keys if any(x["match"] == k and x["r"] == 12 for x in pair_rows)]
print(f"matches with a round 12 -> 13 pair (halftime boundary available): {len(ge13)}")
summarize("halftime-available matches", ge13)
for name, fn in day_group.items():
    keys = [k for k in ge13 if fn(k)]
    summarize(f"halftime-available, day {name}", keys)

print("\nRed flags (should be zero if fields are side-attached):")
team_matches = [m for m in all_keys if classify(per_match[m]) == "team"]
mixed_matches = [m for m in all_keys if classify(per_match[m]) == "mixed"]
print(f"  team-only matches: {len(team_matches)}; mixed: {len(mixed_matches)}")
print(f"  pairs where players swapped and score is neither pattern: "
      f"{sum(per_match[m]['neither_at_swap'] for m in all_keys)}")
print(f"  pairs where players stayed but score showed a swap: "
      f"{sum(per_match[m]['swap_when_stayed'] for m in all_keys)}")
print(f"  pairs where the player move was unclear: {sum(per_match[m]['unclear_pm'] for m in all_keys)}")
print(f"  team-only matches by day: "
      f"{dict(collections.Counter(m.split('/')[3] for m in team_matches))}")
print(f"  example team-only keys (first 5): {[m[:40] for m in team_matches[:5]]}")
