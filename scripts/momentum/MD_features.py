"""Momentum features for the bake-off round model (task MD, feature test).

Team identity follows the derived-lead logic in scripts/model/sufficiency/score_lead_fix.py:
team A starts on T, team B starts on CT, and after round r the sides swap when
bakeoff.swap_expected(r, max_rounds) is true. Winners come from round_end of every round of the
channel-kept matches, including rounds the bake-off model later drops, because those rounds were
played and their results are known to the teams.

Features for round r. Sign is relative to the team on CT in round r, so the same team flips sign
after the swap.
  prev_round_ct_team_won        +1 if the team on CT now won round r-1, -1 if it lost; 0 at round 1.
  signed_streak_ct_team         length of the run of consecutive round wins that ends at round r-1,
                                capped at 5, signed by the run's team (+ if it is on CT now, - if on T).
                                0 at round 1.
  pistol_r1_winner_is_ct_team   0 at round 1 (the pistol has not been played yet); +1 / -1 from
                                round 2 on, by whether the team on CT now won round 1.
  pistol_r13_winner_is_ct_team  0 through round 13; +1 / -1 from round 14 on, by whether the team
                                on CT now won round 13.

Leakage rule: feats_at(r, ...) reads winners only for rounds below r. leak_checks() tests this by
flipping the winners of round r and all later rounds and checking that the features at round r
do not move.

Run only as a module from MD_run.py (loaded with importlib). Writes nothing.
"""

import collections
import sys

sys.dont_write_bytecode = True

STREAK_CAP = 5
PISTOL_R1 = 1
PISTOL_R13 = 13
FEATURE_NAMES = ["prev_round_ct_team_won", "signed_streak_ct_team",
                 "pistol_r1_winner_is_ct_team", "pistol_r13_winner_is_ct_team"]
FLIP = {"A": "B", "B": "A"}


def team_tracks(rd, H, bk):
    """Per match: {round: (ct_team, winner_team)} with teams 'A' or 'B'.
    ct_team is the team on CT before the round; winner_team is the team that won the round.
    The side schedule depends only on the round index and max_rounds (swap rule), never on winners."""
    by_match = collections.defaultdict(dict)
    for (m, r), v in rd.items():
        by_match[m][r] = v["winner"]
    tracks = {}
    stats = collections.Counter()
    for m, rounds in by_match.items():
        mr = H[m]["max_rounds"]
        stats[f"max_rounds_{mr}_matches"] += 1
        rs = sorted(rounds)
        if rs != list(range(1, len(rs) + 1)):
            stats["matches_with_round_gap"] += 1
        side_a = bk.T
        rec = {}
        for r in rs:
            ct_team = "A" if side_a == bk.CT else "B"
            w = rounds[r]
            if w == side_a:
                wt = "A"
            elif w == (bk.CT if side_a == bk.T else bk.T):
                wt = "B"
            else:
                raise ValueError(f"unexpected winner {w!r} in {m} round {r}")
            rec[r] = (ct_team, wt)
            if bk.swap_expected(r, mr):
                side_a = bk.CT if side_a == bk.T else bk.T
                stats["swaps_applied"] += 1
        tracks[m] = rec
    return tracks, stats


def lead_crosscheck(tracks, lead):
    """Running score lead from the tracked teams must equal score_lead_fix.derive_leads."""
    n = bad = 0
    for m, rec in tracks.items():
        a = b = 0
        for r in sorted(rec):
            ct, wt = rec[r]
            mine = (a - b) if ct == "A" else (b - a)
            n += 1
            if mine != lead[(m, r)]:
                bad += 1
            if wt == "A":
                a += 1
            else:
                b += 1
    return {"rounds": n, "mismatches_vs_derive_leads": bad}


def feats_at(r, ct, winners):
    """Momentum features for round r. winners maps round -> winner team; only keys below r are read."""
    prev_ct = streak = p1 = p13 = 0
    if r >= 2 and (r - 1) in winners:
        last = winners[r - 1]
        prev_ct = 1 if last == ct else -1
        run, k = 0, r - 1
        while k >= 1 and run < STREAK_CAP and winners.get(k) == last:
            run += 1
            k -= 1
        streak = run if last == ct else -run
    if r >= 2 and PISTOL_R1 in winners:
        p1 = 1 if winners[PISTOL_R1] == ct else -1
    if r > PISTOL_R13 and PISTOL_R13 in winners:
        p13 = 1 if winners[PISTOL_R13] == ct else -1
    return {"prev_round_ct_team_won": prev_ct, "signed_streak_ct_team": streak,
            "pistol_r1_winner_is_ct_team": p1, "pistol_r13_winner_is_ct_team": p13}


def momentum_all(tracks):
    """{(match, round): feature dict} for every round of every channel-kept match."""
    out = {}
    for m, rec in tracks.items():
        winners = {r: wt for r, (ct, wt) in rec.items()}
        for r, (ct, wt) in rec.items():
            out[(m, r)] = feats_at(r, ct, winners)
    return out


def leak_checks(tracks, feats):
    res = collections.Counter()
    for m, rec in tracks.items():
        base = {r: wt for r, (ct, wt) in rec.items()}
        for r, (ct, wt) in rec.items():
            res["rounds_tested"] += 1
            # 1) winners of round r and all later rounds flipped: features at r must not move
            flipped = {k: (FLIP[v] if k >= r else v) for k, v in base.items()}
            if feats_at(r, ct, flipped) != feats[(m, r)]:
                res["changed_when_round_r_and_later_flipped"] += 1
            # 2) power check: flipping only round r-1 must move prev_round at round r
            if r >= 2 and (r - 1) in base:
                w2 = dict(base)
                w2[r - 1] = FLIP[base[r - 1]]
                res["power_rounds_checked"] += 1
                if feats_at(r, ct, w2)["prev_round_ct_team_won"] != feats[(m, r)]["prev_round_ct_team_won"]:
                    res["power_prev_round_moved"] += 1
    return dict(res)


def rule_checks(tracks, feats):
    res = collections.Counter()
    for m, rec in tracks.items():
        has = set(rec)
        for r in rec:
            f = feats[(m, r)]
            if r == 1 and any(f.values()):
                res["round1_nonzero"] += 1
            if r <= PISTOL_R13 and f["pistol_r13_winner_is_ct_team"] != 0:
                res["pistol_r13_nonzero_before_14"] += 1
            if r >= PISTOL_R13 + 1 and PISTOL_R13 in has and f["pistol_r13_winner_is_ct_team"] not in (1, -1):
                res["pistol_r13_zero_after_13_known"] += 1
            if r >= 2 and PISTOL_R1 in has and f["pistol_r1_winner_is_ct_team"] not in (1, -1):
                res["pistol_r1_zero_after_1_known"] += 1
            if abs(f["signed_streak_ct_team"]) > STREAK_CAP:
                res["streak_above_cap"] += 1
            if f["prev_round_ct_team_won"] == 0 and r >= 2 and (r - 1) in has:
                res["prev_round_zero_with_prior_round"] += 1
            res["rows"] += 1
    return dict(res)
