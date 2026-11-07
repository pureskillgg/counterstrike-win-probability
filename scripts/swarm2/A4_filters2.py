"""A4 follow-up: where the round-filter drops come from (2026 tome, bake-off kept matches).

  a. player_count != 10: concentration by match (whole matches vs scattered rounds), pc = 4 cluster.
  b. not-5v5 rounds (player_count == 10): which player makes player_info show 11 rows; whether the
     extra player is a mid-match joiner; team_change timing in those rounds.
  c. disconnect/takeover rule window: the bake-off counts an event by its round label, anywhere in
     the round, including after round_end and before round_freeze_end. Variants:
       label (bake-off): any event carrying the round label.
       round window: tick in [round_start tick, round_end tick] (the round as defined by the start
         and end events). ASSUMPTION: the round is round_start to round_end.
       live window: tick in [round_freeze_end tick, round_end tick] (ASSUMPTION, alive counts start
         at the live start).
     For each variant: rounds dropped (among rounds passing pc and 5v5), and the surviving rounds.
  d. Rounds dropped only by events after round_end: are they the last round of the match?
  e. header match_id uniqueness (duplicate-copy cross-check, ids may be redacted).

Read-only on the tome. Run:
  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\swarm2\\A4_filters2.py
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
T, CT, START, TICK_RATE = fv.T, fv.CT, fv.START, fv.TICK_RATE


def main():
    hdr = fv.read("header", ["key", "match_id", "match_date", "max_rounds", "tick_rate"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    print(f"header rows {hdr.height}; distinct match_id {hdr['match_id'].n_unique()}; "
          f"distinct keys {hdr['key'].n_unique()}")
    RE = fv.read("round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    RST = fv.read("round_start", ["match_key", "round", "tick"])
    RS = fv.read("round_state", ["match_key", "round", "tick", "event_type"])
    PI = fv.read("player_info", ["match_key", "round", "player_id", "team_code"])
    DE = fv.read("player_death", ["match_key", "round", "tick", "player_id", "attacker_id"])
    DC = fv.read("player_disconnect", ["match_key", "round", "tick", "player_id", "is_bot"])
    BT = fv.read("bot_takeover", ["match_key", "round", "tick", "player_id"])
    TC = fv.read("team_change", ["match_key", "round", "tick", "player_id", "old_team_code",
                                 "new_team_code", "is_bot"])

    # the same channel check as the bake-off (dense channels need rows; sparse need keyset only)
    import glob
    all_matches = set(H)
    dense_frames = {"round_end": RE, "round_start": RST, "round_state": RS, "player_death": DE,
                    "player_info": PI, "team_change": TC}
    bad = set()
    from_bs = fv.read("bomb_state", ["match_key"])
    dense_frames["bomb_state"] = from_bs
    for ch, df in dense_frames.items():
        have = set(df["match_key"].unique().to_list())
        bad |= all_matches - have
    kept = all_matches - bad
    print(f"kept matches (dense-channel check, no keyset check here; keysets equal header): {len(kept)}")
    if len(kept) != 1709:
        print(f"  NOTE: expected 1709 from the bake-off log")

    rd = {}
    for x in RE.iter_rows(named=True):
        if x["match_key"] in kept:
            rd[(x["match_key"], int(x["round"]))] = {"end": int(x["tick"]), "pc": x["player_count"]}
    rst = {}
    for x in RST.iter_rows(named=True):
        if x["match_key"] in kept:
            rst.setdefault((x["match_key"], int(x["round"])), int(x["tick"]))
    frz = {}
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        if x["match_key"] in kept:
            frz.setdefault((x["match_key"], int(x["round"])), int(x["tick"]))
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        if x["match_key"] in kept:
            players[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
    rounds_per_match = collections.Counter(k[0] for k in rd)
    max_round = collections.defaultdict(int)
    for k in rd:
        max_round[k[0]] = max(max_round[k[0]], k[1])

    # ---------- a. player_count concentration
    print("\n== a. player_count != 10 by match ==")
    pc_fail = {k for k in rd if rd[k]["pc"] != 10}
    per_match = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    for k in rd:
        s = per_match[k[0]]
        s[0] += 1
        if k in pc_fail:
            s[1] += 1
            s[2][rd[k]["pc"]] += 1
    n_all_fail = sum(1 for m, s in per_match.items() if s[1] == s[0] and s[0] > 0)
    n_half = sum(1 for m, s in per_match.items() if s[1] > 0 and s[1] / s[0] >= 0.5)
    n_some = sum(1 for m, s in per_match.items() if s[1] > 0)
    print(f"  matches with any pc-failing round: {n_some}; with >=50% of rounds failing: {n_half}; "
          f"with all rounds failing: {n_all_fail}")
    fails_by_match = sorted(((s[1], m, s) for m, s in per_match.items() if s[1]), reverse=True)
    print("  top matches by pc-failing rounds (fail/rounds, pc values, date):")
    for n, m, s in fails_by_match[:10]:
        print(f"    {m[:50]:<52} {s[1]:>3}/{s[0]:<3} {dict(s[2])} {H[m]['match_date'][:10]}")
    pc4_matches = [(m, per_match[m][1]) for m in per_match if per_match[m][2].get(4, 0) > 0]
    pc4_rounds = sum(per_match[m][2].get(4, 0) for m in per_match)
    print(f"  pc=4 rounds: {pc4_rounds} in {len(pc4_matches)} matches")
    dist =collections.Counter(len(set(per_match[m][2].keys())) for m in per_match if per_match[m][1])
    print(f"  distinct pc-failing values per match (count of matches): {dict(sorted(dist.items()))}")
    fully = [(m, per_match[m]) for m in per_match if per_match[m][1] == per_match[m][0]]
    pcvals = collections.Counter()
    for m, s in fully:
        for v, n in s[2].items():
            pcvals[v] += n
    print(f"  fully failing matches: {len(fully)}; their rounds by pc value: {dict(sorted(pcvals.items()))}")
    sample_fully = sorted(fully, key=lambda x: -x[1][0])[:5]
    for m, s in sample_fully:
        print(f"    {m[:50]} rounds {s[0]} pc {dict(s[2])} date {H[m]['match_date'][:10]} max_rounds {H[m]['max_rounds']}")

    # ---------- b. not-5v5 rounds (pc == 10)
    print("\n== b. not-5v5 rounds with player_count == 10 ==")
    presence = collections.Counter()
    for k, pl_ in players.items():
        for pid in pl_:
            presence[(k[0], pid)] += 1
    not5 = []
    for k in rd:
        if rd[k]["pc"] != 10:
            continue
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if not (len(pl_) == 10 and n_t == START and n_ct == START):
            not5.append(k)
    print(f"  not-5v5 rounds with pc == 10: {len(not5)}")
    tc_by_round = collections.defaultdict(list)
    for x in TC.iter_rows(named=True):
        if x["match_key"] in kept:
            tc_by_round[(x["match_key"], int(x["round"]))].append(x)
    de_by_round = collections.defaultdict(list)
    for x in DE.iter_rows(named=True):
        if x["match_key"] in kept:
            de_by_round[(x["match_key"], int(x["round"]))].append(x)
    matches_per_pid = collections.Counter(m for (m, pid) in presence)
    extra_stats = collections.Counter()
    tc_timing = collections.Counter()
    extra_in_tc = collections.Counter()
    extra_in_de = collections.Counter()
    extra_in_disc = collections.Counter()
    examples = []
    for k in not5:
        pl_ = players[k]
        n_rows = len(pl_)
        m = k[0]
        n_rounds_m = rounds_per_match[m]
        extra_stats[f"rows={n_rows}"] += 1
        # extra players: present in this round but in fewer than 80% of the match's rounds
        extra = [pid for pid in pl_ if presence[(m, pid)] < 0.8 * n_rounds_m]
        extra_stats[f"extra_players={len(extra)}"] += 1
        for pid in extra:
            extra_stats[f"extra_team={pl_[pid]}"] += 1
            if any(x["player_id"] == pid for x in tc_by_round.get(k, [])):
                extra_in_tc[pid] += 1
            if any(x["player_id"] == pid for x in de_by_round.get(k, [])):
                extra_in_de[pid] += 1
        f = frz.get(k)
        for x in tc_by_round.get(k, []):
            if f is None:
                tc_timing["no_freeze"] += 1
            elif x["tick"] < rst.get(k, 0):
                tc_timing["before_round_start"] += 1
            elif x["tick"] < f:
                tc_timing["freeze_time"] += 1
            elif x["tick"] <= rd[k]["end"]:
                tc_timing["live"] += 1
            else:
                tc_timing["after_round_end"] += 1
        if len(examples) < 4 and n_rows == 11:
            examples.append((k, dict(pl_), [(x["tick"], x["player_id"], x["old_team_code"], x["new_team_code"],
                                              x["is_bot"]) for x in tc_by_round.get(k, [])], f, rd[k]["end"]))
    print(f"  PI row counts and extra-player stats: {dict(sorted(extra_stats.items()))}")
    print(f"  team_change timing in not-5v5 rounds: {dict(tc_timing)}")
    print(f"  extra players also in a team_change event in the round: {sum(extra_in_tc.values())}; "
          f"also in a death event: {sum(extra_in_de.values())}")
    print("  examples (round key, PI team by player, team_change events (tick, pid, old, new, is_bot), freeze, end):")
    for ex in examples:
        print(f"    {ex[0][0][:45]} r{ex[0][1]}: PI {ex[1]}")
        print(f"      TC {ex[2][:6]}; freeze {ex[3]}; end {ex[4]}")

    # rounds (pc==10 & not5) where player count 10 but PI shows 11 with 6-5 split: team_change?
    # ---------- c. disconnect/takeover window variants (rounds passing pc and 5v5)
    print("\n== c. Disconnect/takeover rule variants (rounds passing pc and 5v5) ==")
    pass_rounds = []
    for k in rd:
        if rd[k]["pc"] != 10:
            continue
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if len(pl_) == 10 and n_t == START and n_ct == START:
            pass_rounds.append(k)
    print(f"  rounds passing pc and 5v5: {len(pass_rounds)}")
    events = []  # (key, tick, src)
    for src, df in (("disc", DC), ("takeover", BT)):
        for x in df.iter_rows(named=True):
            if x["match_key"] in kept:
                events.append(((x["match_key"], int(x["round"])), int(x["tick"]), src))
    by_round = collections.defaultdict(list)
    for k, tick, src in events:
        by_round[k].append((tick, src))
    variants = {
        "label (bake-off)": lambda k, t: True,
        "round window [start, end]": lambda k, t: k in rst and rst[k] <= t <= rd[k]["end"],
        "live window [freeze, end]": lambda k, t: k in frz and frz[k] <= t <= rd[k]["end"],
    }
    surv_sets = {}
    for name, rule in variants.items():
        dropped = {k for k in pass_rounds if any(rule(k, t) for t, _ in by_round.get(k, []))}
        surv_sets[name] = set(pass_rounds) - dropped
        print(f"  {name:<28} dropped {len(dropped):>5}; kept among pass rounds {len(surv_sets[name]):>6}")
    base_drop = {k for k in pass_rounds if by_round.get(k)}
    print(f"  any-label rounds with at least one event among pass rounds: {len(base_drop)}")
    # breakdown of bake-off drops among pass rounds: after-only, freeze-only, mixed, live-only
    cls = collections.Counter()
    last_round_after_only = 0
    after_only = 0
    for k in base_drop:
        has_live = any(rst.get(k, 0) <= t <= rd[k]["end"] and k in frz and frz[k] <= t
                       for t, _ in by_round[k])
        has_after = any(t > rd[k]["end"] for t, _ in by_round[k])
        has_freeze = any(k in frz and rst.get(k, 0) <= t < frz[k] for t, _ in by_round[k])
        has_before = any(k in rst and t < rst[k] for t, _ in by_round[k])
        key_name = "+".join(n for n, f in (("live", has_live), ("freeze", has_freeze),
                                             ("after", has_after), ("before_start", has_before)) if f) or "none"
        cls[key_name] += 1
        if not has_live and not has_freeze and not has_before and has_after:
            after_only += 1
            if k[1] == max_round[k[0]]:
                last_round_after_only += 1
    print(f"  bake-off drops among pass rounds, by where the events fall: {dict(cls.most_common())}")
    print(f"  rounds dropped only by events after round_end: {after_only}; of these in the match's last "
          f"round: {last_round_after_only}")
    # states implied by kept rounds (1 + distinct in-window victims) for each variant
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        if x["match_key"] in kept:
            deaths[(x["match_key"], int(x["round"]))].append((int(x["tick"]), x["player_id"]))

    def states_of(k):
        if k not in frz:
            return 0
        seen = set()
        n = 0
        for t, pid in deaths.get(k, []):
            if frz[k] <= t <= rd[k]["end"] and pid not in seen:
                seen.add(pid)
                n += 1
        return 1 + n

    for name, s in surv_sets.items():
        print(f"  states from pass rounds kept under '{name}': {sum(states_of(k) for k in s)} "
              f"(rounds {len(s)})")

    print("\nfinished")


if __name__ == "__main__":
    main()
