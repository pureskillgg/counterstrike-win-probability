"""MC step 1: team-round table from the 1 Hz freeze-end snapshot, with loss and win streak lengths.

Reads (read only):
  data/state_sample/players_1hz.parquet   (freeze-end snapshot = seconds_since_live == 0)
  data/state_sample/rounds.parquet        (round winner ct_win, split)

Writes (scratch, not the repo):
  <scratchpad>/MC_team_rounds.parquet

Definitions
  Freeze-end snapshot: the first live second (seconds_since_live == 0), equal to rounds.live_start_tick.
  Team-round: one team in one round. Each kept round gives two team-rounds (team X, team Y).
  Team identity: a match's two 5-player groups (by player_id, per match). Team X is the group that
    contains the smallest player_id in the match. Team identity is fixed across rounds, so side swaps
    do not break streaks.
  Win: team's side is CT and ct_win == 1, or side is T and ct_win == 0.
  Loss/win run length: number of consecutive prior rounds (r-1, r-2, ...) lost (or won) by the team,
    capped at 5. A run is "ok" (exact) if it ends at an observed opposite-outcome round or at match start
    (round 0). It is censored if it hits an unobserved round (gap, excluded round) before ending.
  Excluded: the flagged anomaly round (22e9c3e7..., round 16; 9 players at freeze end and absent from
    round_buy_types), and any round whose 10-player partition differs from the match's dominant partition.
"""

import collections
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path("G:/pureskill/csds-cs2")
SAMPLE = ROOT / "data/state_sample"
SCRATCH = Path("C:/Users/13144/AppData/Local/Temp/claude/C--Users-13144-Documents-pureskill/"
               "e0a18ad2-aa9e-4a3c-9946-29641b2cde82/scratchpad")
ANOMALY = ("csds/2026/10/07/22e9c3e7-a333-4ebc-a51c-19ebbcc13ef1/csds", 16)
CAP = 5


def load():
    rounds = pl.read_parquet(SAMPLE / "rounds.parquet", columns=["match_key", "round", "ct_win", "split"])
    snap = (
        pl.scan_parquet(SAMPLE / "players_1hz.parquet")
        .select(["match_key", "round", "seconds_since_live", "player_id", "team_code", "money",
                 "equipment_value_calc"])
        .filter(pl.col("seconds_since_live") == 0)
        .collect()
    )
    return rounds, snap


def main():
    rounds, snap = load()
    all_matches = sorted(rounds["match_key"].unique().to_list())
    split_of = {m: s for m, s in zip(rounds["match_key"].to_list(), rounds["split"].to_list())}
    winner = {(m, int(r)): int(c) for m, r, c in zip(rounds["match_key"], rounds["round"], rounds["ct_win"])}

    # per (match, round): side -> list of (pid, money, equip)
    rp = collections.defaultdict(lambda: {2: [], 3: []})
    for m, r, pid, tc, mo, eq in snap.select(["match_key", "round", "player_id", "team_code", "money",
                                              "equipment_value_calc"]).iter_rows():
        rp[(m, int(r))][int(tc)].append((int(pid), int(mo), int(eq)))

    reasons = collections.Counter()
    # dominant partition per match, from rounds that have 5 + 5 players
    part_count = collections.defaultdict(collections.Counter)
    for key, sides in rp.items():
        if key not in winner:
            continue
        if len(sides[2]) == 5 and len(sides[3]) == 5:
            p = frozenset([frozenset(x[0] for x in sides[2]), frozenset(x[0] for x in sides[3])])
            part_count[key[0]][p] += 1
    dominant = {m: c.most_common(1)[0][0] for m, c in part_count.items()}
    multi = [m for m, c in part_count.items() if len(c) > 1]
    print(f"matches={len(all_matches)} with_snapshot={len(part_count)} multi_partition_matches={len(multi)}")

    kept = {}  # (m, r) -> dict
    for key in sorted(winner):
        m, r = key
        if key == ANOMALY:
            reasons["anomaly_round_flagged"] += 1
            continue
        sides = rp.get(key)
        if sides is None or len(sides[2]) + len(sides[3]) == 0:
            reasons["no_snapshot"] += 1
            continue
        if len(sides[2]) != 5 or len(sides[3]) != 5:
            reasons["not_10_players"] += 1
            continue
        p = frozenset([frozenset(x[0] for x in sides[2]), frozenset(x[0] for x in sides[3])])
        if p != dominant.get(m):
            reasons["partition_mismatch"] += 1
            continue
        kept[key] = sides
    print("excluded rounds:", dict(reasons), "kept rounds:", len(kept))

    # Team X = group containing the smallest player_id in the match (fixed identity across rounds)
    rows = []
    obs = collections.defaultdict(dict)  # m -> r -> {'X': won, 'Y': won}
    side_pattern = collections.defaultdict(dict)  # m -> r -> side of X
    for key in sorted(kept):
        m, r = key
        sides = kept[key]
        gX = min(dominant[m], key=lambda g: min(g))  # group holding the smallest player_id
        # side of X
        sX = 2 if frozenset(x[0] for x in sides[2]) == gX else 3
        sY = 5 - sX
        pX = sides[sX]
        pY = sides[sY]
        eqX = sum(x[2] for x in pX)
        moX = sum(x[1] for x in pX)
        eqY = sum(x[2] for x in pY)
        moY = sum(x[1] for x in pY)
        ct = winner[key]
        wX = int((sX == 3) == (ct == 1))
        wY = 1 - wX
        side_pattern[m][r] = sX
        obs[m][r] = {"X": wX, "Y": wY}
        rows.append(dict(match_key=m, round=r, team="X", side=sX, won=wX, team_eq=eqX, team_money=moX,
                         opp_eq=eqY, opp_money=moY, n_players=5))
        rows.append(dict(match_key=m, round=r, team="Y", side=sY, won=wY, team_eq=eqY, team_money=moY,
                         opp_eq=eqX, opp_money=moX, n_players=5))

    # Side-rule check (observed, not documented): regulation flips at round 13. In overtime the sides do
    # NOT flip at round 25; they flip every 3 rounds from round 28 (rounds 25-27 keep round-24 sides).
    def blk(r):
        if r <= 12:
            return 0
        if r <= 24:
            return 1
        return (r - 25) // 3 + 1

    ok_rule = bad_rule = 0
    bad_matches = []
    for m, rr in side_pattern.items():
        s1 = rr.get(1)
        if s1 is None:
            continue
        mism = 0
        for r, s in rr.items():
            expect = s1 if blk(r) % 2 == 0 else (5 - s1)
            if expect == s:
                ok_rule += 1
            else:
                bad_rule += 1
                mism += 1
        if mism:
            bad_matches.append(m)
    print(f"side-rule check: ok_rounds={ok_rule} bad_rounds={bad_rule} bad_matches={len(bad_matches)}")

    # Round-1 bankroll check (eq + money per player should be about 800 if equipment is price paid)
    bank = collections.defaultdict(list)
    for key, sides in kept.items():
        for tc in (2, 3):
            for pid, mo, eq in sides[tc]:
                bank[key[1]].append(mo + eq)
    for r in (1, 13, 2, 12, 14):
        if r in bank:
            arr = np.array(bank[r], dtype=float)
            print(f"bankroll per player round {r}: mean={arr.mean():.1f} median={np.median(arr):.1f} n={arr.size}")

    # Streak runs
    def run_len(m, r, team, target):
        k = 0
        for j in range(1, CAP + 1):
            rr = r - j
            if rr < 1:
                return k, True  # reached match start: exact
            rec = obs[m].get(rr)
            if rec is None:
                return k, False  # censored by unobserved round
            if rec[team] == target:
                k += 1
            else:
                return k, True
        return k, True  # CAP consecutive same outcomes: exact 5+

    out = []
    for row in rows:
        m, r, team = row["match_key"], row["round"], row["team"]
        lk, lok = run_len(m, r, team, 0)
        wk, wok = run_len(m, r, team, 1)
        out.append({
            **row,
            "split": split_of[m],
            "loss_run": lk, "loss_ok": lok,
            "win_run": wk, "win_ok": wok,
            "is_pistol": r in (1, 13),
            "is_round1": r == 1,
        })
    df = pl.DataFrame(out)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    out_path = SCRATCH / "MC_team_rounds.parquet"
    df.write_parquet(out_path)
    print("wrote", out_path, "rows", df.height, "matches with rows", df["match_key"].n_unique())
    print("loss_ok rate", round(df["loss_ok"].mean(), 4), "win_ok rate", round(df["win_ok"].mean(), 4))


if __name__ == "__main__":
    main()
