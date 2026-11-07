"""D3 secondary: winner-derived score lead vs round_state score lead on the 2024 G: tome.

Same derivation as scripts/model/sufficiency/score_lead_fix.py (derive_leads), with max_rounds = 24
(ASSUMPTION: the 2024 header has no max_rounds column; MR12 is consistent with the side swap seen at
round 13 and the OT swap at round 28). Team A starts on T, team B on CT. The derived lead before round r
is score(CT team) - score(T team), from round_end winners and the swap rule. It is compared with
round_state ct_score - t_score at round_freeze_end of round r (the bake-off feature).

Reported: agreement by day group, first disagreeing round per match, matches with round gaps.
Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\D3_lead_derivation_2024.py
"""
import collections
import glob
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
pl.Config.set_tbl_rows(60)
pl.Config.set_tbl_width_chars(200)
ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2024-02-g" / "tome" / "csds"
SUF = ".2023-12-28,2024-02-25"
T, CT = 2, 3
MAX_ROUNDS = 24  # ASSUMPTION


def g_read(ch, cols=None):
    files = sorted(glob.glob(str(TOME / f"basic_{ch}{SUF}" / "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def swap_expected(r, mr):
    return r == mr // 2 or (r > mr and (r - mr - 3) % 6 == 0)


RE = g_read("round_end", ["match_key", "round", "winner_team_code"]).unique(["match_key", "round"], keep="first")
RS = g_read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"])
fe = (RS.filter(pl.col("event_type") == "round_freeze_end").sort("tick")
      .group_by(["match_key", "round"]).agg(pl.col("t_score").first(), pl.col("ct_score").first()))
live = {(m, r): (t, c) for m, r, t, c in fe.select(["match_key", "round", "t_score", "ct_score"]).iter_rows()}

by_match = collections.defaultdict(dict)
for m, r, w in RE.select(["match_key", "round", "winner_team_code"]).iter_rows():
    by_match[m][r] = w

stats = collections.Counter()
rows = []  # (match, day, round, derived, live, agree)
first_dis = {}
gap_matches = set()
for m, rounds in by_match.items():
    rs_sorted = sorted(rounds)
    day = int(m.split("/")[3])
    grp = "day12_25" if day >= 12 else "day01_07"
    if rs_sorted != list(range(1, len(rs_sorted) + 1)):
        gap_matches.add(m)
        stats[f"gap_{grp}"] += 1
        # do not derive across a gap; skip the match for agreement
        continue
    side_a = T
    a = b = 0
    for r in rs_sorted:
        derived = (b - a) if side_a == T else (a - b)
        if (m, r) in live:
            lt, lc = live[(m, r)]
            lead_live = lc - lt
            ok = derived == lead_live
            rows.append((m, grp, day, r, derived, lead_live, ok))
            stats[f"cmp_{grp}"] += 1
            stats[f"agree_{grp}"] += int(ok)
            if not ok and m not in first_dis:
                first_dis[m] = r
        w = rounds[r]
        if w == side_a:
            a += 1
        elif w == (CT if side_a == T else T):
            b += 1
        else:
            stats["winner_not_T_or_CT"] += 1
        if swap_expected(r, MAX_ROUNDS):
            side_a = CT if side_a == T else T

print("matches in round_end:", len(by_match), "| with round gap (excluded):", len(gap_matches))
for g in ["day01_07", "day12_25"]:
    n, ag = stats[f"cmp_{g}"], stats[f"agree_{g}"]
    print(f"{g}: rounds compared {n}, agree {ag} ({(ag / n if n else 0):.6f}), round-gap matches {stats[f'gap_{g}']}")
print("winner_not_T_or_CT:", stats["winner_not_T_or_CT"])

# matches (no gap) with any disagreement, by group
dis_by_grp = collections.Counter()
for m, r in first_dis.items():
    grp = "day12_25" if int(m.split("/")[3]) >= 12 else "day01_07"
    dis_by_grp[grp] += 1
print("matches with any lead disagreement: ", dict(dis_by_grp))
print("first disagreeing round (round: matches), day12_25:",
      sorted(collections.Counter(r for m, r in first_dis.items()
                                  if int(m.split("/")[3]) >= 12).items()))
print("first disagreeing round (round: matches), day01_07:",
      sorted(collections.Counter(r for m, r in first_dis.items()
                                 if int(m.split("/")[3]) < 12).items()))

# affected-match check: the bake-off round-13 mismatch matches (from the side check)
aff = set()
bs = g_read("bomb_state", ["match_key", "round", "event_type"]).filter(
    pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
bo = bs.group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs"))
r13 = (RE.filter(pl.col("round") == 13).join(bo.filter(pl.col("round") == 13), on=["match_key", "round"], how="inner")
       .with_columns(pl.when(pl.col("evs").list.contains("bomb_defused")).then(CT).otherwise(T).alias("bside")))
aff = set(r13.filter(pl.col("bside") != pl.col("winner_team_code"))["match_key"].to_list())
aff_nogap = [m for m in aff if m not in gap_matches]
print(f"\nround-13 mismatch matches: {len(aff)}; without round gap: {len(aff_nogap)}")
fd_aff = collections.Counter(first_dis[m] for m in aff_nogap if m in first_dis)
print("first disagreeing round for round-13 mismatch matches (round: matches):", sorted(fd_aff.items()))
print("round-13 mismatch matches with NO lead disagreement at all:", sum(1 for m in aff_nogap if m not in first_dis))

# clean matches (no round-13 mismatch) in day 12..25 with no gap
clean_late = [m for m in by_match if int(m.split("/")[3]) >= 12 and m not in aff and m not in gap_matches]
fd_clean = collections.Counter(first_dis[m] for m in clean_late if m in first_dis)
print(f"day12..25 clean matches (no r13 mismatch, no gap): {len(clean_late)}; with any disagreement: "
      f"{sum(1 for m in clean_late if m in first_dis)}; first round counts: {sorted(fd_clean.items())}")
