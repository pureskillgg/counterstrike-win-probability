"""Targeted follow-up checks for Task 1 goofy items (read-only).

Run: pixi run python -I scripts\\eda\\flow_checks.py
"""
import collections
import glob
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
TICK_RATE = 64
WIN_REASON = {1: "Target_Bombed", 7: "Bomb_Defused", 8: "CTs_Win", 9: "Terrorists_Win",
              11: "All_Hostages_Rescued", 12: "Target_Saved", 13: "Hostages_Not_Rescued",
              17: "Terrorists_Surrender", 18: "CTs_Surrender"}
T, CT = 2, 3


def read(ch):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def rows(df, cols):
    return list(zip(*[df[c].to_list() for c in cols]))


def main():
    re_ = read("round_end")
    rs = read("round_state")
    bs = read("bomb_state")
    ba = read("bomb_action")
    su = read("score_update")
    rst = read("round_start")
    rm = read("round_mvp")
    bd = read("bomb_defuse")
    de = read("player_death")
    hdr = read("header")

    # 1. winner vs win_reason crosstab
    print("## 1. winner_team_code x win_reason_code (counts)")
    ct = collections.Counter(rows(re_, ["winner_team_code", "win_reason_code"]))
    for (w, r), n in sorted(ct.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        side = {T: "T", CT: "CT"}.get(w, str(w))
        print(f"- winner {w} ({side}) with reason {r} ({WIN_REASON.get(r, 'UNMAPPED')}): {n}")

    # 2. plant-event presence per round vs win reason
    plant_rounds = set(rows(bs.filter(pl.col("event_type") == "bomb_planted"), ["match_key", "round"]))
    exploded_rounds = collections.Counter(rows(bs.filter(pl.col("event_type") == "bomb_exploded"),
                                               ["match_key", "round"]))
    defused_rounds = collections.Counter(rows(bs.filter(pl.col("event_type") == "bomb_defused"),
                                              ["match_key", "round"]))
    print(f"\n## 2. bomb_state event rows per round: bomb_exploded per round "
          f"{dict(sorted(collections.Counter(exploded_rounds.values()).items()))}; "
          f"bomb_defused per round {dict(sorted(collections.Counter(defused_rounds.values()).items()))}")
    print("## 2b. round win_reason x (has bomb_planted row in bomb_state)")
    ctp = collections.Counter()
    for m, r, wr in rows(re_, ["match_key", "round", "win_reason_code"]):
        ctp[(wr, (m, r) in plant_rounds)] += 1
    for (wr, planted), n in sorted(ctp.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        print(f"- reason {wr} ({WIN_REASON.get(wr, 'UNMAPPED')}) planted={planted}: {n}")
    exp_wr = collections.Counter()
    for m, r, wr in rows(re_, ["match_key", "round", "win_reason_code"]):
        if (m, r) in exploded_rounds:
            exp_wr[wr] += 1
    print(f"- rounds with a bomb_exploded row, by win_reason: {dict(exp_wr)}; "
          f"rounds with bomb_exploded row total {len(exploded_rounds)}")

    # 3. matches with zero bomb_state rows
    bs_matches = set(bs["match_key"].to_list())
    all_matches = set(hdr["key"].to_list())
    no_bs = all_matches - bs_matches
    re_no_bs = re_.filter(pl.col("match_key").is_in(list(no_bs)))
    print(f"\n## 3. matches with zero bomb_state rows: {len(no_bs)}; rounds in them: {re_no_bs.height}")
    wr3 = collections.Counter(re_no_bs["win_reason_code"].to_list())
    print(f"- win_reason in those rounds: {dict(sorted(wr3.items()))}")
    rs_planted_no_bs = rs.filter((pl.col("event_type") == "bomb_planted") &
                                 pl.col("match_key").is_in(list(no_bs)))
    print(f"- round_state bomb_planted rows in those matches: {rs_planted_no_bs.height}")
    hdr_no_bs = hdr.filter(pl.col("key").is_in(list(no_bs)))
    print(f"- header max_rounds of those matches: {dict(collections.Counter(hdr_no_bs['max_rounds'].to_list()))}; "
          f"match_type: {dict(collections.Counter(hdr_no_bs['match_type'].to_list()))}")
    # rounds in ALL matches where the round ended by bomb (1 or 7) but no plant row
    bomb_end_no_plant = sum(1 for m, r, wr in rows(re_, ["match_key", "round", "win_reason_code"])
                            if wr in (1, 7) and (m, r) not in plant_rounds)
    print(f"- rounds ended by bomb (reason 1 or 7) with no bomb_planted row (all matches): {bomb_end_no_plant}")

    # 4. round_start tick 66 sentinel
    s66 = rst.filter(pl.col("tick") == 66)
    print(f"\n## 4. round_start rows at tick 66: {s66.height}; round number counts: "
          f"{dict(sorted(collections.Counter(s66['round'].to_list()).items())[:6])} ...")
    print(f"- matches with a tick-66 round_start: {s66['match_key'].n_unique()}")
    print(f"- round_state rows at tick 66 with event_type: "
          f"{dict(collections.Counter(rs.filter(pl.col('tick') == 66)['event_type'].to_list()))}")
    # live-start tick for these rounds
    fe = rs.filter(pl.col("event_type") == "round_freeze_end")
    fe_tick = {(m, r): t for m, r, t in rows(fe, ["match_key", "round", "tick"])}
    end_tick = {(m, r): t for m, r, t in rows(re_, ["match_key", "round", "tick"])}
    s66_pairs = set(rows(s66, ["match_key", "round"]))
    d = [(end_tick[k] - fe_tick[k]) / TICK_RATE for k in s66_pairs if k in fe_tick and k in end_tick]
    d_all = [(end_tick[k] - fe_tick[k]) / TICK_RATE for k in end_tick if k in fe_tick]
    if d:
        d.sort()
        print(f"- tick-66 rounds: live-start to round_end seconds: median {d[len(d)//2]:.1f}, "
              f"max {d[-1]:.1f}; all rounds median {sorted(d_all)[len(d_all)//2]:.1f}")
    print(f"- tick-66 round_start rounds that also have a round_freeze_end row: "
          f"{sum(1 for k in s66_pairs if k in fe_tick)} of {len(s66_pairs)}")

    # 5. round_state round_end tick vs round_end tick
    rse = {(m, r): t for m, r, t in rows(rs.filter(pl.col("event_type") == "round_end"),
                                       ["match_key", "round", "tick"])}
    mism = sum(1 for k, t in end_tick.items() if k in rse and rse[k] != t)
    print(f"\n## 5. round_state round_end tick differs from round_end tick: {mism} "
          f"of {len(end_tick)} (missing in round_state: {sum(1 for k in end_tick if k not in rse)})")
    # round_state events outside round window by event type
    rst_start = {(m, r): t for m, r, t in rows(rst, ["match_key", "round", "tick"])}
    out_ct = collections.Counter()
    tot_ct = collections.Counter()
    for m, r, t, et in rows(rs, ["match_key", "round", "tick", "event_type"]):
        if (m, r) not in rst_start or (m, r) not in end_tick:
            continue
        tot_ct[et] += 1
        if t < rst_start[(m, r)] or t > end_tick[(m, r)]:
            out_ct[et] += 1
    print("- round_state rows outside [round_start, round_end] by event_type (outside/total):")
    for et, n in sorted(out_ct.items(), key=lambda kv: -kv[1]):
        print(f"    {et}: {n}/{tot_ct[et]}")

    # 6. player_id == 0 sentinel
    print("\n## 6. player_id == 0 sentinel")
    for name, df, raw, fixed in [("round_mvp", rm, "player_id", "player_id_fixed"),
                                 ("bomb_defuse", bd, "player_id", "player_id_fixed"),
                                 ("bomb_state", bs, "player_id", "player_id_fixed"),
                                 ("bomb_action", ba, "player_id", "player_id_fixed"),
                                 ("player_death", de, "player_id", "player_id_fixed")]:
        zero = df.filter(pl.col(raw) == 0)
        fx = collections.Counter(zero[fixed].to_list())
        print(f"- {name}: raw player_id==0 rows {zero.height} of {df.height}; "
              f"{fixed} for those rows: {dict(sorted(fx.items(), key=lambda kv: str(kv[0]))[:6])}")

    # 7. score_update 3-row rounds and symmetric deltas
    print("\n## 7. score_update")
    grp = collections.defaultdict(list)
    for m, r, t, tc, o, n in rows(su, ["match_key", "round", "tick", "team_code", "old_score", "new_score"]):
        grp[(m, r)].append((t, tc, o, n))
    three = [k for k, v in grp.items() if len(v) == 3]
    print(f"- rounds with 3 rows: {len(three)}")
    if three:
        ex = three[0]
        print(f"  example match {ex[0][:45]} round {ex[1]}: {sorted(grp[ex])}")
        pattern = collections.Counter(tuple(sorted(tc for _, tc, _, _ in grp[k])) for k in three)
        print(f"  team_code patterns in 3-row rounds: {dict(pattern)}")
        dpat = collections.Counter(tuple(sorted(n - o for _, _, o, n in grp[k])) for k in three)
        print(f"  delta patterns in 3-row rounds (top 8): {dpat.most_common(8)}")
    two = [k for k, v in grp.items() if len(v) == 1]
    print(f"- rounds with 1 row: {len(two)}")
    # deltas in 1-row rounds
    d1 = collections.Counter(n - o for k in two for _, _, o, n in grp[k])
    print(f"- delta counts in 1-row rounds: {dict(sorted(d1.items()))}")
    # rounds where the sum of deltas is not zero
    nz = 0
    for k in grp:
        s = sum(n - o for _, _, o, n in grp[k])
        if s != 0:
            nz += 1
    print(f"- rounds where sum of deltas != 0: {nz}")
    # are the big deltas concentrated in particular matches?
    big = collections.Counter(k[0] for k, v in grp.items() if any(abs(n - o) >= 2 for _, _, o, n in v))
    print(f"- matches with a |delta| >= 2 row: {len(big)}; top 5 match counts: "
          f"{[c for _, c in big.most_common(5)]}")
    big_rounds = [(k, grp[k]) for k in grp if any(abs(n - o) >= 2 for _, _, o, n in grp[k])]
    if big_rounds:
        k, v = big_rounds[0]
        print(f"  example |delta|>=2: round {k[1]} rows {sorted(v)}")

    # 8. round_mvp duplicate pairs
    print("\n## 8. round_mvp duplicate (match, round) pairs")
    rmp = collections.defaultdict(list)
    for m, r, t, p, mc in rows(rm, ["match_key", "round", "tick", "player_id", "mvp_count"]):
        rmp[(m, r)].append((t, p, mc))
    for k, v in rmp.items():
        if len(v) > 1:
            print(f"- match {k[0][:45]} round {k[1]}: {v}")

    # 9. bomb_action impossible velocities and site_code
    print("\n## 9. bomb_action velocity range and site_code")
    vel = ba.with_columns(
        (pl.col("player_x_vel").abs() > 1000).alias("bigx"),
        (pl.col("player_y_vel").abs() > 1000).alias("bigy"),
    )
    print(f"- bomb_action rows with |x_vel| > 1000: {vel.filter(pl.col('bigx')).height}; "
          f"|y_vel| > 1000: {vel.filter(pl.col('bigy')).height} of {ba.height}")
    print(f"- bomb_action |x_vel|>1000 by event_type: "
          f"{dict(collections.Counter(vel.filter(pl.col('bigx'))['event_type'].to_list()))}")
    print(f"- site_code distinct values (bomb_state): {bs['site_code'].n_unique()}; "
          f"values <= 2 : {bs.filter(pl.col('site_code') <= 2).height}; "
          f"bomb_planted site_code top: "
          f"{dict(collections.Counter(bs.filter(pl.col('event_type') == 'bomb_planted')['site_code'].to_list()).most_common(5))}")

    # 10. player_count != 10 concentration
    print("\n## 10. round_end player_count != 10 concentration")
    bad = re_.filter(pl.col("player_count") != 10)
    bad_m = collections.Counter(bad["match_key"].to_list())
    print(f"- rounds with player_count != 10: {bad.height} in {len(bad_m)} matches; "
          f"matches with ALL rounds != 10: "
          f"{sum(1 for m in bad_m if bad_m[m] == collections.Counter(re_['match_key'].to_list())[m])}")
    print(f"- rounds per match with any != 10: median "
          f"{sorted(collections.Counter(re_['match_key'].to_list()).get(m, 0) for m in bad_m)[len(bad_m)//2]}")

    # 11. round_end rounds in short / long matches
    rpm = collections.Counter(re_["match_key"].to_list())
    hdr_mr = dict(zip(hdr["key"].to_list(), hdr["max_rounds"].to_list()))
    print("\n## 11. rounds per match vs header max_rounds")
    ct11 = collections.Counter()
    for m, n in rpm.items():
        ct11[(hdr_mr.get(m), "in 16..30" if 16 <= n <= 30 else "outside")] += 1
    print(f"- {dict(ct11)}")
    over = collections.Counter(rpm[m] for m in rpm if rpm[m] > 30)
    print(f"- matches with more than 30 rounds, rounds count: {dict(sorted(over.items()))}")


if __name__ == "__main__":
    main()
