"""A4 follow-up 3: does round_end.player_count agree with the player_info roster and the disconnect events?

Read-only. Cross-tabs player_count against the PI roster size and split, disconnect/takeover presence,
and the match date, and checks whether the extra players in not-5v5 rounds are bots.
Run: G:\pureskill\csds-cs2\.pixi\envs\default\python.exe -I -X utf8 scripts\swarm2\A4_filters3.py
"""
import collections
import importlib.util
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path("G:/pureskill/csds-cs2")
_spec = importlib.util.spec_from_file_location("feature_validation", ROOT / "scripts" / "model" / "feature_validation.py")
fv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fv)
T, CT, START = fv.T, fv.CT, fv.START


def main():
    hdr = fv.read("header", ["key", "match_date"])
    H = {r["key"]: r["match_date"][:10] for r in hdr.iter_rows(named=True)}
    RE = fv.read("round_end", ["match_key", "round", "tick", "player_count"])
    PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"])
    DC = fv.read("player_disconnect", ["match_key", "round", "tick", "player_id", "is_bot"])
    BT = fv.read("bot_takeover", ["match_key", "round", "tick", "player_id"])
    TC = fv.read("team_change", ["match_key", "round", "tick", "player_id", "old_team_code",
                                 "new_team_code", "is_bot"])
    rows = {}
    for x in RE.iter_rows(named=True):
        rows[(x["match_key"], int(x["round"]))] = x["player_count"]
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        players[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
    disc = set((x["match_key"], int(x["round"])) for x in DC.iter_rows(named=True))
    disc |= set((x["match_key"], int(x["round"])) for x in BT.iter_rows(named=True))
    bots_tc = collections.Counter()
    for x in TC.iter_rows(named=True):
        bots_tc[bool(x["is_bot"])] += 1
    print(f"team_change rows by is_bot: {dict(bots_tc)}")

    # per-round class: PI roster size and split
    tab = collections.Counter()
    by_month_pc = collections.Counter()
    for k, pc in rows.items():
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        exact = len(pl_) == 10 and n_t == START and n_ct == START
        tab[(pc, "PI exact 5v5" if exact else f"PI not 5v5 ({len(pl_)} rows)", k in disc)] += 1
        if pc != 10:
            by_month_pc[(H.get(k[0], "?"), pc)] += 1
    print("\nplayer_count x PI roster (exact 5v5 or not) x any disconnect/takeover in round (all 35,992 rounds):")
    agg = collections.defaultdict(int)
    for (pc, pistate, d), n in sorted(tab.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        if pc in (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12):
            print(f"  pc={pc:<3} {pistate:<26} disc={str(d):<5} rounds {n}")
    for (pc, pistate, d), n in tab.items():
        agg[(pc == 10, pistate == "PI exact 5v5", d)] += n
    print("\nsummary (pc==10?, PI exact 5v5?, disc?):")
    for k2, n in sorted(agg.items()):
        print(f"  {k2}: {n}")
    n_pc_fail_pi_ok_nodisc = sum(n for (pc, pistate, d), n in tab.items()
                                 if pc != 10 and pistate == "PI exact 5v5" and not d)
    n_pc_fail_pi_ok = sum(n for (pc, pistate, d), n in tab.items() if pc != 10 and pistate == "PI exact 5v5")
    print(f"\npc != 10 rounds where PI shows exact 5v5: {n_pc_fail_pi_ok} (with no disconnect/takeover in "
          f"round: {n_pc_fail_pi_ok_nodisc})")
    pc4_pi = collections.Counter(len(players.get(k, {})) for k, pc in rows.items() if pc == 4)
    print(f"pc=4 rounds: PI row counts {dict(sorted(pc4_pi.items()))}")
    pc11_pi = collections.Counter(len(players.get(k, {})) for k, pc in rows.items() if pc == 11)
    print(f"pc=11 rounds: PI row counts {dict(sorted(pc11_pi.items()))}")
    pc9_pi = collections.Counter(len(players.get(k, {})) for k, pc in rows.items() if pc == 9)
    print(f"pc=9 rounds: PI row counts {dict(sorted(pc9_pi.items()))}")
    print("\npc != 10 rounds by match date (date, pc, rounds):")
    for (d, pc), n in sorted(by_month_pc.items()):
        if n >= 20:
            print(f"  {d} pc={pc}: {n}")
    dates_all = collections.Counter(H.values())
    print(f"\nmatches by date (header): {dict(sorted(dates_all.items()))}")
    print("\ndone")


if __name__ == "__main__":
    main()
