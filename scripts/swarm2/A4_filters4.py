"""A4 follow-up 4: bot share of not-5v5 extras, human-only disconnect variant, kept matches by date,
and which filter attributes the 16 rounds without round_freeze_end (bake-off order).
Read-only. Run: G:\pureskill\csds-cs2\.pixi\envs\default\python.exe -I -X utf8 scripts\swarm2\A4_filters4.py
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
    RST = fv.read("round_start", ["match_key", "round", "tick"])
    RS = fv.read("round_state", ["match_key", "round", "tick", "event_type"])
    PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"])
    DC = fv.read("player_disconnect", ["match_key", "round", "tick", "player_id", "is_bot"])
    BT = fv.read("bot_takeover", ["match_key", "round", "tick", "player_id"])
    TC = fv.read("team_change", ["match_key", "round", "tick", "player_id", "is_bot"])
    BS = fv.read("bomb_state", ["match_key"])
    TCm = fv.read("team_change", ["match_key"])
    # channel-kept matches (dense channels with rows; sparse need keyset which equals header)
    all_m = set(H)
    dense_have = [set(df["match_key"].unique().to_list()) for df in (RE, RST, RS, fv.read("player_death", ["match_key"]), PI, BS, TCm)]
    kept = all_m.copy()
    for h in dense_have:
        kept &= h
    print(f"kept matches (dense channels): {len(kept)}")
    rows = {}
    for x in RE.iter_rows(named=True):
        if x["match_key"] in kept:
            rows[(x["match_key"], int(x["round"]))] = x["player_count"]
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        if x["match_key"] in kept:
            players[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
    presence = collections.Counter()
    rounds_m = collections.Counter(k[0] for k in rows)
    for k, pl_ in players.items():
        for pid in pl_:
            presence[(k[0], pid)] += 1
    tc_bot = collections.defaultdict(dict)
    for x in TC.iter_rows(named=True):
        if x["match_key"] in kept:
            tc_bot[(x["match_key"], int(x["round"]))][x["player_id"]] = bool(x["is_bot"])
    # bot share of extra players in pc==10 not-5v5 rounds
    extra_total = 0
    extra_bot = 0
    extra_unknown = 0
    not5 = []
    for k, pc in rows.items():
        if pc != 10:
            continue
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if len(pl_) == 10 and n_t == START and n_ct == START:
            continue
        not5.append(k)
        for pid in pl_:
            if presence[(k[0], pid)] < 0.8 * rounds_m[k[0]]:
                extra_total += 1
                b = tc_bot.get(k, {}).get(pid)
                if b is None:
                    extra_unknown += 1
                elif b:
                    extra_bot += 1
    print(f"not-5v5 rounds (pc==10): {len(not5)}; extra (non-core) PI players: {extra_total}; "
          f"with a team_change row marking is_bot=True: {extra_bot}; no team_change row: {extra_unknown}")

    # disconnect event composition in pass rounds
    frz = {}
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        if x["match_key"] in kept:
            frz.setdefault((x["match_key"], int(x["round"])), int(x["tick"]))
    ev_h = collections.defaultdict(lambda: {"human": 0, "bot": 0, "takeover": 0})
    for x in DC.iter_rows(named=True):
        if x["match_key"] in kept:
            k = (x["match_key"], int(x["round"]))
            ev_h[k]["bot" if x["is_bot"] else "human"] += 1
    for x in BT.iter_rows(named=True):
        if x["match_key"] in kept:
            ev_h[(x["match_key"], int(x["round"]))]["takeover"] += 1
    pass_rounds = []
    for k, pc in rows.items():
        if pc != 10:
            continue
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if len(pl_) == 10 and n_t == START and n_ct == START:
            pass_rounds.append(k)
    flagged = [k for k in pass_rounds if sum(ev_h[k].values()) > 0] if False else [k for k in pass_rounds if k in ev_h and sum(ev_h[k].values()) > 0]
    human_or_takeover = [k for k in flagged if ev_h[k]["human"] > 0 or ev_h[k]["takeover"] > 0]
    bot_only = [k for k in flagged if ev_h[k]["human"] == 0 and ev_h[k]["takeover"] == 0]
    print(f"pass rounds (pc and 5v5): {len(pass_rounds)}; flagged by any event (bake-off): {len(flagged)}; "
          f"flagged by a human disconnect or takeover: {len(human_or_takeover)}; bot-disconnect-only: {len(bot_only)}")

    # the 16 rounds with no round_freeze_end
    nofz = [k for k in rows if k[0] in kept and k not in frz]
    print(f"rounds without round_freeze_end (kept matches): {len(nofz)}; all have a disconnect/takeover event: "
          f"{sum(1 for k in nofz if k in ev_h and sum(ev_h[k].values()) > 0)}; of these, pc==10 and PI exact 5v5: "
          f"{sum(1 for k in nofz if rows[k] == 10 and len(players.get(k, {})) == 10 and sum(1 for v in players[k].values() if v == T) == START and sum(1 for v in players[k].values() if v == CT) == START)}")

    # kept matches by date
    kept_by_date = collections.Counter(H[m] for m in kept)
    all_by_date = collections.Counter(H.values())
    print("\nmatches by date: header / kept (channel) / kept after round filters (see A4_filters.py: 1521 matches)")
    for d in sorted(all_by_date):
        print(f"  {d}: {all_by_date[d]} / {kept_by_date.get(d, 0)}")
    print("done")


if __name__ == "__main__":
    main()
