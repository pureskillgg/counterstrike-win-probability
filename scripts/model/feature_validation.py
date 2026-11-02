"""Event-only feature validation for the CS2 round win-probability model.

Data: the 2026-10-01_to_08 basic tomes. Per-tick channels are not present, so every
quantity comes from event rows and their tick stamps (64 ticks = 1 second).

Steps
  1. Round lifecycle over all matches: round_state event types, live start marker.
  2. Eligibility: matches with >= 24 rounds where every round has player_count == 10.
     Sample: random.Random(0) over the sorted eligible list, capped at 200.
  3. Alive-count state machine per round. Start at 5 vs 5 at the live start
     (round_freeze_end). Each basic_player_death with tick in [live start, round_end]
     decrements the victim side. Victim side = player_team_code, else the
     basic_player_info team for (match, round, player_id).
  4. Checks (a) to (d) with failure counts and 3 examples each, plus diagnostics.

Run from the project root:
  pixi run python -I -X utf8 scripts\\model\\feature_validation.py
"""

import collections
import glob
import random
import sys
from pathlib import Path

import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
TICK_RATE = 64
MIN_ROUNDS = 24
N_SAMPLE = 200
SEED = 0
N_EX = 3
T, CT = 2, 3
START = 5
WIN_REASON = {
    1: "Target_Bombed", 7: "Bomb_Defused", 8: "CTs_Win", 9: "Terrorists_Win",
    11: "All_Hostages_Rescued", 12: "Target_Saved", 13: "Hostages_Not_Rescued",
    17: "Terrorists_Surrender", 18: "CTs_Surrender",
}


def read(channel, cols=None):
    folder = TOME / ("header" if channel == "header" else f"basic_{channel}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    if not files:
        raise SystemExit(f"no pages in {folder}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def short(match_key):
    return match_key.split("/")[-2][:8]


def task1(rs, rend, rstart):
    print("== 1. Round lifecycle ==")
    ev = (rs.group_by("event_type")
          .agg(pl.len().alias("rows"), pl.col("match_key").n_unique().alias("matches"))
          .sort("rows", descending=True))
    print(f"round_state event_type values: {ev.height}; total rows {rs.height}")
    for r in ev.iter_rows(named=True):
        print(f"  {r['event_type']:<34}{r['rows']:>8}{r['matches']:>7}")
    live = (rs.filter(pl.col("event_type") == "round_freeze_end")
            .select(["match_key", "round", "tick"]).rename({"tick": "live_tick"}))
    inferred = (rs.filter(pl.col("event_type") == "freezetime_ended_inferred")
                .select(["match_key", "round", "tick"]).rename({"tick": "inf_tick"}))
    ends = rend.select(["match_key", "round", "tick"]).rename({"tick": "end_tick"})
    starts = rstart.select(["match_key", "round", "tick"]).rename({"tick": "start_tick"})
    base = ends.join(live, on=["match_key", "round"], how="left")
    n_live = base.filter(pl.col("live_tick").is_not_null()).height
    print(f"rounds (round_end rows): {base.height}; with round_freeze_end: {n_live}; "
          f"without: {base.height - n_live}")
    j = base.filter(pl.col("live_tick").is_not_null())
    print(f"live_tick < end_tick: {(j['live_tick'] < j['end_tick']).sum()} of {j.height}")
    s = starts.join(j, on=["match_key", "round"])
    print(f"start_tick <= live_tick: {(s['start_tick'] <= s['live_tick']).sum()} of {s.height}")
    i = live.join(inferred, on=["match_key", "round"])
    gap = (i["inf_tick"] - i["live_tick"]).abs()
    print(f"freezetime_ended_inferred within 1 s (64 ticks) of round_freeze_end: "
          f"{(gap <= TICK_RATE).sum()} of {i.height}; exact tick match: {(gap == 0).sum()}")


def task_global(rs, rend):
    print("\n== 1b. Label and score checks (all matches) ==")
    xt = collections.Counter((r["win_reason_code"], r["winner_team_code"])
                             for r in rend.iter_rows(named=True))
    print("win_reason_code x winner_team_code (reason: winner counts):")
    for (rc, wt), n in sorted(xt.items()):
        print(f"  {rc:>3} {WIN_REASON.get(rc, '?'):<22} winner={wt} n={n}")
    sc = {(r["match_key"], r["round"]): r for r in
          rs.filter(pl.col("event_type") == "round_freeze_end")
          .select(["match_key", "round", "t_score", "ct_score"]).iter_rows(named=True)}
    wn = {(r["match_key"], r["round"]): r["winner_team_code"]
          for r in rend.select(["match_key", "round", "winner_team_code"]).iter_rows(named=True)}
    sum_ok = sum(1 for (m, r), v in sc.items() if v["t_score"] + v["ct_score"] == r - 1)
    print(f"freeze-end rows: {len(sc)}; t_score + ct_score == round - 1: {sum_ok}")
    kinds = collections.Counter()
    swap_after = collections.Counter()
    for (m, r), a in sc.items():
        b = sc.get((m, r + 1))
        w = wn.get((m, r))
        if b is None or w is None:
            continue
        no = (b["t_score"] == a["t_score"] + (w == T)) and (b["ct_score"] == a["ct_score"] + (w == CT))
        sw = (b["t_score"] == a["ct_score"] + (w == CT)) and (b["ct_score"] == a["t_score"] + (w == T))
        if no and sw:
            kinds["ambiguous"] += 1
        elif no:
            kinds["no_swap"] += 1
        elif sw:
            kinds["swap"] += 1
            swap_after[r] += 1
        else:
            kinds["neither"] += 1
    print(f"score transitions r -> r+1: {dict(kinds)}")
    print(f"score swap after round (round: count): {sorted(swap_after.items())}")


def main():
    rs = read("round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score", "phase"])
    rend = read("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code", "player_count"])
    rstart = read("round_start", ["match_key", "round", "tick"])
    task1(rs, rend, rstart)
    task_global(rs, rend)

    # ---------- eligibility and sample
    per_match = rend.group_by("match_key").agg(
        pl.len().alias("n_rounds"), (pl.col("player_count") == 10).all().alias("all10"))
    eligible = sorted(per_match.filter((pl.col("n_rounds") >= MIN_ROUNDS) & pl.col("all10"))
                      ["match_key"].to_list())
    print("\n== 2. Eligibility and sample ==")
    print(f"matches with round_end: {per_match.height}; all rounds player_count==10: "
          f"{int(per_match['all10'].sum())}; >=24 rounds: "
          f"{int((per_match['n_rounds'] >= MIN_ROUNDS).sum())}; eligible: {len(eligible)}")
    rng = random.Random(SEED)
    sample = sorted(rng.sample(eligible, min(N_SAMPLE, len(eligible))))
    print(f"requested {N_SAMPLE} matches; eligible pool {len(eligible)}; used {len(sample)}")

    rend_s = rend.filter(pl.col("match_key").is_in(sample))
    rs_s = rs.filter(pl.col("match_key").is_in(sample))
    deaths_df = read("player_death", ["match_key", "round", "tick", "player_id", "player_team_code",
                                      "weapon_name", "attacker_id"]).filter(pl.col("match_key").is_in(sample))
    bomb_df = read("bomb_state", ["match_key", "round", "tick", "event_type", "player_team_code"]).filter(
        pl.col("match_key").is_in(sample))
    plant_df = bomb_df.filter(pl.col("event_type") == "bomb_planted")
    explode_df = bomb_df.filter(pl.col("event_type") == "bomb_exploded")
    disc_df = read("player_disconnect", ["match_key", "round", "tick"]).filter(
        pl.col("match_key").is_in(sample))
    takeover_df = read("bot_takeover", ["match_key", "round", "tick"]).filter(
        pl.col("match_key").is_in(sample))
    info_df = read("player_info", ["match_key", "round", "player_id", "team_code"]).filter(
        pl.col("match_key").is_in(sample))
    header_df = read("header", ["key", "max_rounds"])
    max_rounds = {r["key"]: r["max_rounds"] for r in header_df.iter_rows(named=True)}
    print(f"header rows matched to sample matches: {sum(1 for m in sample if m in max_rounds)} of {len(sample)}")
    explode_tick = collections.defaultdict(list)
    for e in explode_df.iter_rows(named=True):
        explode_tick[(e["match_key"], e["round"])].append(e["tick"])
    disc_rounds = collections.defaultdict(list)
    for e in list(disc_df.iter_rows(named=True)) + list(takeover_df.iter_rows(named=True)):
        disc_rounds[(e["match_key"], e["round"])].append(e["tick"])

    R = {}
    for r in rend_s.iter_rows(named=True):
        R[(r["match_key"], r["round"])] = {"end": r["tick"], "winner": r["winner_team_code"],
                                           "reason": r["win_reason_code"], "pc": r["player_count"]}
    for r in rs_s.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        k = (r["match_key"], r["round"])
        if k in R:
            R[k].update(freeze=r["tick"], t_score=r["t_score"], ct_score=r["ct_score"], phase=r["phase"])
    for r in rstart.filter(pl.col("match_key").is_in(sample)).iter_rows(named=True):
        k = (r["match_key"], r["round"])
        if k in R:
            R[k]["start"] = r["tick"]
    deaths = collections.defaultdict(list)
    for d in deaths_df.sort("tick").iter_rows(named=True):
        deaths[(d["match_key"], d["round"])].append(d)
    plants = collections.defaultdict(list)
    for p in plant_df.iter_rows(named=True):
        plants[(p["match_key"], p["round"])].append(p["tick"])
    players = collections.defaultdict(dict)
    for p in info_df.iter_rows(named=True):
        players[(p["match_key"], p["round"])][p["player_id"]] = p["team_code"]

    n_rounds = len(R)
    n_live = sum(1 for v in R.values() if v.get("freeze") is not None)
    print(f"sample rounds: {n_rounds}; with live start: {n_live}")
    def exact_split(key):
        return sorted(collections.Counter(players[key].values()).items()) == sorted([(T, START), (CT, START)])

    split_exact = sum(1 for k in R if exact_split(k))
    print(f"sample rounds whose player_info split is exactly 5 T / 5 CT: {split_exact} of {n_rounds}")
    ct_wins = sum(1 for v in R.values() if v["winner"] == CT)
    print(f"sample round labels: ct_win (winner_team_code == 3) = {ct_wins} of {n_rounds} "
          f"({ct_wins / n_rounds:.4f}); t_win = {n_rounds - ct_wins}")

    # ---------- state machine and checks
    tally = collections.defaultdict(int)
    examples = collections.defaultdict(list)
    pre_rows = []
    diag = collections.Counter()
    feats = []
    for key, rr in R.items():
        if rr.get("freeze") is None:
            tally["rounds_no_live_start"] += 1
            continue
        frz, end = rr["freeze"], rr["end"]
        pre, post, win, unresolved = [], [], [], 0
        for d in deaths.get(key, []):
            if d["tick"] < frz:
                pre.append(d)
                continue
            if d["tick"] > end:
                post.append(d)
                continue
            side = d["player_team_code"]
            if side is None:
                diag["in_window_null_team"] += 1
                side = players[key].get(d["player_id"])
                if side is not None:
                    diag["in_window_null_team_imputed"] += 1
            if side not in (T, CT):
                unresolved += 1
                continue
            win.append({**d, "side": side})
        diag["deaths_pre_live"] += len(pre)
        diag["deaths_post_end"] += len(post)
        diag["deaths_in_window"] += len(win)
        for d in pre:
            pre_rows.append({"match_key": key[0], "round": key[1], "player_id": d["player_id"],
                             "tick": d["tick"], "weapon": d["weapon_name"]})
        if unresolved:
            tally["rounds_unresolved_victim_side"] += 1
            continue

        plant_all = plants.get(key, [])
        in_win = [t for t in plant_all if frz <= t <= end]
        diag["plants_all"] += len(plant_all)
        diag["plants_in_window"] += len(in_win)
        diag["plants_after_end"] += sum(1 for t in plant_all if t > end)
        diag["plants_before_live"] += sum(1 for t in plant_all if t < frz)
        if len(plant_all) > 1:
            diag["rounds_with_multiple_plants"] += 1
        plant_tick = min(in_win) if in_win else None

        alive = {T: START, CT: START}
        states = [{"tick": frz, "alive_t": START, "alive_ct": START, "kind": "live_start"}]
        seen = set()
        for d in win:
            if d["player_id"] in seen:
                # spec rule: each victim counts once per round (first death wins)
                diag["in_window_duplicate_victim_skipped"] += 1
                continue
            seen.add(d["player_id"])
            alive[d["side"]] -= 1
            states.append({"tick": d["tick"], "alive_t": alive[T], "alive_ct": alive[CT],
                           "kind": "death", "victim": d["side"]})
        for s in states:
            s["planted"] = int(plant_tick is not None and plant_tick <= s["tick"])

        tally["rounds_checked"] += 1
        tally["states"] += len(states)
        exact = exact_split(key)
        if exact:
            tally["rounds_exact55"] += 1
        if any(frz <= t <= end for t in disc_rounds.get(key, [])):
            tally["rounds_with_in_window_disconnect_or_takeover"] += 1
        w = rr["winner"]
        base = (short(key[0]), key[1])
        exploded = any(frz <= t <= end for t in explode_tick.get(key, []))
        last_kill = win[-1]["weapon_name"] if win else "none"

        def record(name, detail):
            tally[name] += 1
            if exact:
                tally[name + "_exact55"] += 1
            if len(examples[name]) < N_EX:
                examples[name].append(f"match {base[0]} round {base[1]} {detail}")

        def fmt(s):
            return (f"tick={s['tick']} alive_t={s['alive_t']} alive_ct={s['alive_ct']} "
                    f"planted={s['planted']} kind={s['kind']} winner={w} "
                    f"reason={WIN_REASON.get(rr['reason'], rr['reason'])}")

        # (a) negative counts
        for s in states:
            if s["alive_t"] < 0 or s["alive_ct"] < 0:
                record("a_negative_alive", fmt(s))
                break
        # (b) side with 0 alive is the round winner
        zero = {}
        for side, field in ((T, "alive_t"), (CT, "alive_ct")):
            for s in states:
                if s[field] == 0:
                    zero[side] = s
                    break
        if w in zero:
            zs = zero[w]
            if w == T and zs["planted"] == 1:
                record("b_zero_T_wins_after_plant_bomb_path", fmt(zs))
                tally["b_bomb_path_bomb_exploded"] += int(exploded)
                tally["b_bomb_path_last_kill_" + last_kill] += 1
            else:
                record("b_zero_side_wins_violation", fmt(zs))
        # (c) winner has at least one survivor at the last state
        last = states[-1]
        surv = last["alive_t"] if w == T else last["alive_ct"]
        if surv < 1:
            record("c_winner_no_survivor", fmt(last))
            tally["c_reason_" + WIN_REASON.get(rr["reason"], str(rr["reason"]))] += 1
            tally["c_no_survivor_bomb_exploded"] += int(exploded)
            tally["c_no_survivor_last_kill_" + last_kill] += 1
            if not exploded:
                tally["c_winner_no_survivor_no_explosion"] += 1
                if len(examples["c_winner_no_survivor_no_explosion"]) < 10:
                    examples["c_winner_no_survivor_no_explosion"].append(
                        f"match {base[0]} round {base[1]} {fmt(last)} last_kill={last_kill} "
                        f"last_kill_attacker_is_victim={win[-1]['attacker_id'] == win[-1]['player_id']}")
        # (d) bomb_planted flag never drops within a round
        flags = [s["planted"] for s in states]
        for prev, cur, s in zip(flags, flags[1:], states[1:]):
            if cur < prev:
                record("d_bomb_flag_drop", fmt(s))
                break

        # sample points for the feature table
        for s in states:
            planted = s["planted"]
            feats.append({
                "alive_ct": s["alive_ct"], "alive_t": s["alive_t"],
                "ct_alive_minus_t": s["alive_ct"] - s["alive_t"],
                "bomb_planted": planted,
                "seconds_since_round_live": (s["tick"] - frz) / TICK_RATE,
                "seconds_since_plant": ((s["tick"] - plant_tick) / TICK_RATE) if planted else -1.0,
                "round_number": key[1],
                "ct_score_minus_t_score": rr["ct_score"] - rr["t_score"],
                "is_pistol_round": int(key[1] in (1, max_rounds.get(key[0], 24) // 2 + 1)),
                "is_overtime": int(key[1] > max_rounds.get(key[0], 24)),
                "label_ct_win": int(w == CT),
            })

    print("\n== 3. Sample points (Task 2) ==")
    print(f"live-start points: {tally['rounds_checked']}")
    print(f"in-window player_death rows: {diag['deaths_in_window']}; death states after the "
          f"repeat-victim rule: {tally['states'] - tally['rounds_checked']}")
    print(f"total states (live start + death states): {tally['states']}")
    print(f"rounds excluded (unresolved victim side): {tally['rounds_unresolved_victim_side']}")
    print(f"rounds checked: {tally['rounds_checked']}; exactly 5v5 start: {tally['rounds_exact55']}")
    print(f"deaths before live start: {diag['deaths_pre_live']}; after round_end: {diag['deaths_post_end']}")
    print(f"null victim team in window: {diag['in_window_null_team']} "
          f"(imputed from player_info: {diag['in_window_null_team_imputed']})")
    print(f"in-window repeat deaths of the same victim (skipped by spec rule): "
          f"{diag['in_window_duplicate_victim_skipped']}")
    print(f"bomb_planted events: all {diag['plants_all']}, in window {diag['plants_in_window']}, "
          f"after round_end {diag['plants_after_end']}, before live {diag['plants_before_live']}; "
          f"rounds with >1 plant: {diag['rounds_with_multiple_plants']}")

    # pre-live deaths: are they real?
    if pre_rows:
        pre_df = pl.DataFrame(pre_rows).with_columns(
            pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64), pl.col("tick").cast(pl.Int64))
        wf = (read("weapon_fire", ["match_key", "round", "player_id", "tick"])
              .filter(pl.col("match_key").is_in(sample))
              .with_columns(pl.col("round").cast(pl.Int64), pl.col("player_id").cast(pl.Int64),
                            pl.col("tick").cast(pl.Int64).alias("fire_tick"))
              .select(["match_key", "round", "player_id", "fire_tick"]))
        fired_after = (pre_df.join(wf, on=["match_key", "round", "player_id"])
                       .filter(pl.col("fire_tick") > pl.col("tick"))
                       .select(["match_key", "round", "player_id"]).unique().height)
        world = sum(1 for r in pre_rows if r["weapon"] == "world")
        print(f"pre-live deaths: {len(pre_rows)} (weapon 'world': {world}); "
              f"victims who fire a weapon later in the same round: {fired_after}")

    print("\n== 4. Validation checks (Task 3) ==")
    names = [
        ("a_negative_alive", "(a) alive count goes negative"),
        ("b_zero_side_wins_violation", "(b) side at 0 alive wins the round, not explained by a plant"),
        ("b_zero_T_wins_after_plant_bomb_path", "(b) T at 0 alive wins after plant (bomb path)"),
        ("c_winner_no_survivor", "(c) winner has no survivor at last state"),
        ("d_bomb_flag_drop", "(d) bomb_planted flag drops within a round"),
    ]
    for key_name, label in names:
        n = tally[key_name]
        print(f"{label}: rounds={n}; exactly-5v5 rounds={tally[key_name + '_exact55']}")
        for ex in examples[key_name]:
            print(f"    e.g. {ex}")
    print("(c) failures by win reason:")
    for k, v in sorted(tally.items()):
        if k.startswith("c_reason_"):
            print(f"    {k[len('c_reason_'):]}: {v}")
    print(f"(c) failures with bomb_exploded in window: {tally['c_no_survivor_bomb_exploded']}")
    print(f"(c) failures without a bomb explosion (all listed): {tally['c_winner_no_survivor_no_explosion']}")
    for ex in examples["c_winner_no_survivor_no_explosion"]:
        print(f"    {ex}")
    print("(c) failures by last in-window kill weapon:")
    for k, v in sorted(tally.items()):
        if k.startswith("c_no_survivor_last_kill_"):
            print(f"    {k[len('c_no_survivor_last_kill_'):]}: {v}")
    print("(b) bomb-path rounds by last in-window kill weapon:")
    for k, v in sorted(tally.items()):
        if k.startswith("b_bomb_path_last_kill_"):
            print(f"    {k[len('b_bomb_path_last_kill_'):]}: {v}")
    print(f"(b) bomb-path rounds with bomb_exploded in window: {tally['b_bomb_path_bomb_exploded']}")
    print(f"rounds checked: {tally['rounds_checked']}; states checked: {tally['states']}")
    print(f"rounds with an in-window disconnect or bot takeover (alive counts may be off): "
          f"{tally['rounds_with_in_window_disconnect_or_takeover']}")

    # label and features
    if feats:
        n_states = len(feats)
        ct_rate = sum(f["label_ct_win"] for f in feats) / n_states
        print("\n== 5. Feature table summary (sample states) ==")
        print(f"states: {n_states}; CT-win share of states: {ct_rate:.4f}")
        print(f"bomb_planted share: {sum(f['bomb_planted'] for f in feats) / n_states:.4f}")
        print(f"pistol-round states: {sum(f['is_pistol_round'] for f in feats)}; "
              f"overtime states: {sum(f['is_overtime'] for f in feats)}")


if __name__ == "__main__":
    main()
