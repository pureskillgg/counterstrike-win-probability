"""Feasibility probe (Task A follow-up): event timing around round_end for the sampled rounds.

Read-only. Checks, per sampled round, the tick gap between round_end and (a) the last death of any
player, (b) the last losing-side death, (c) bomb_exploded and bomb_defused events, (d) round_state
round_end and round_officially_ended. Also reports where bomb events fall relative to the round's
reason code, deaths after round end, and what alive_gap_snapshots counts (from the extraction script).
Run from the repo root:
  pixi run python -I -X utf8 scripts\\eda\\feas_probe_decision_detail.py
"""

import collections
import glob
import os
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = r"G:\pureskill\csds-cs2"
TOME = os.path.join(ROOT, "tomes", "2026-10-01_to_08", "tome", "csds")
SUFFIX = ".2026-09-18,2026-10-08"
SAMPLE = os.path.join(ROOT, "data", "state_sample")
T, CT = 2, 3
KEYS = ["match_key", "round"]


def read_channel(ch, cols=None):
    files = sorted(glob.glob(os.path.join(TOME, f"basic_{ch}{SUFFIX}", "dataframe_*")))
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def stats(name, vals):
    if not vals:
        print(f"{name}: none")
        return
    a = np.asarray(vals, dtype=float)
    print(f"{name}: n={len(a)} min {a.min():.0f} median {np.median(a):.1f} p90 {np.percentile(a, 90):.1f} "
          f"max {a.max():.0f}; equal 0: {int((a == 0).sum())}")


def main():
    rounds = pl.read_parquet(os.path.join(SAMPLE, "rounds.parquet")).with_columns(
        pl.col("round").cast(pl.Int64))
    matches = set(rounds["match_key"].unique().to_list())
    rk = set(zip(rounds["match_key"].to_list(), rounds["round"].to_list()))
    re_ = (read_channel("round_end", ["match_key", "round", "tick", "winner_team_code", "win_reason_code"])
           .filter(pl.col("match_key").is_in(matches)).with_columns(pl.col("round").cast(pl.Int64)))
    re_ = {(m, r): (t, w, c) for m, r, t, w, c in re_.select(
        ["match_key", "round", "tick", "winner_team_code", "win_reason_code"]).iter_rows()
        if (m, r) in rk}
    DE = (read_channel("player_death", ["match_key", "round", "tick", "player_team_code"])
          .filter(pl.col("match_key").is_in(matches)).with_columns(pl.col("round").cast(pl.Int64)))
    deaths = collections.defaultdict(list)
    for m, r, t, tc in DE.select(["match_key", "round", "tick", "player_team_code"]).iter_rows():
        if (m, r) in rk:
            deaths[(m, r)].append((t, tc))
    BS = (read_channel("bomb_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("match_key").is_in(matches)).with_columns(pl.col("round").cast(pl.Int64)))
    bev = collections.defaultdict(list)
    for m, r, t, et in BS.select(["match_key", "round", "tick", "event_type"]).iter_rows():
        if (m, r) in rk:
            bev[(m, r, et)].append(t)
    RS = (read_channel("round_state", ["match_key", "round", "tick", "event_type"])
          .filter(pl.col("match_key").is_in(matches)
                  & pl.col("event_type").is_in(["round_end", "round_officially_ended"]))
          .with_columns(pl.col("round").cast(pl.Int64)))
    rs = collections.defaultdict(dict)
    for m, r, t, et in RS.select(["match_key", "round", "tick", "event_type"]).iter_rows():
        if (m, r) in rk:
            rs[(m, r)].setdefault(et, t)

    print("== A. round_end vs last death of any player (sample rounds) ==")
    d_any, d_lose, d_post_end, d_bomb_exp, d_bomb_def, d_rs_end, d_post_round = [], [], [], [], [], [], []
    code_by_exp = collections.Counter()
    code_by_def = collections.Counter()
    post_end_side = collections.Counter()
    post_end_elim = collections.Counter()
    for k in sorted(rk):
        end, w, code = re_[k]
        ds = deaths.get(k, [])
        inside = [t for t, _ in ds if t <= end]
        if inside:
            d_any.append(end - max(inside))
        post = [(t, tc) for t, tc in ds if t > end]
        d_post_end.append(len(post))
        if code in (8, 9):
            losing = T if w == CT else CT
            for t, tc in post:
                post_end_side[tc] += 1
                if tc == losing:
                    post_end_elim[code] += 1
        lose = [t for t, tc in ds if tc == (T if w == CT else CT) and t <= end]
        if lose:
            d_lose.append(end - max(lose))
        ex = bev.get((k[0], k[1], "bomb_exploded"), [])
        df = bev.get((k[0], k[1], "bomb_defused"), [])
        if ex:
            code_by_exp[code] += 1
            d_bomb_exp.append(end - ex[0])
        if df:
            code_by_def[code] += 1
            d_bomb_def.append(end - df[0])
        if "round_end" in rs[k]:
            d_rs_end.append(rs[k]["round_end"] - end)
        if "round_officially_ended" in rs[k]:
            d_post_round.append(rs[k]["round_officially_ended"] - end)
    stats("end - last death (any player, tick <= end)", d_any)
    stats("end - last losing-side death", d_lose)
    stats("end - bomb_exploded tick (rounds with event)", d_bomb_exp)
    stats("end - bomb_defused tick (rounds with event)", d_bomb_def)
    stats("round_state round_end tick minus round_end tick", d_rs_end)
    stats("round_officially_ended tick minus round_end tick (post-round length, ticks)", d_post_round)
    print(f"rounds with any death after round_end: {sum(1 for x in d_post_end if x > 0)}; "
          f"deaths after round_end by victim team: {dict(post_end_side)}; "
          f"post-end losing-side deaths in elimination rounds, by code: {dict(post_end_elim)}")
    print(f"bomb_exploded present, by round_end code: {dict(sorted(code_by_exp.items()))}")
    print(f"bomb_defused present, by round_end code: {dict(sorted(code_by_def.items()))}")

    print("\n== B. Sample rounds with bomb events where the round code disagrees (first 8) ==")
    shown = 0
    for k in sorted(rk):
        end, w, code = re_[k]
        ex = bev.get((k[0], k[1], "bomb_exploded"), [])
        df = bev.get((k[0], k[1], "bomb_defused"), [])
        if (ex and code != 1) or (df and code != 7):
            print(f"  {k[0][-12:]} r{k[1]}: code {code} winner {w} end {end} exploded {ex} defused {df} "
                  f"deaths(tick,team) {sorted(deaths.get(k, []))[-4:]}")
            shown += 1
            if shown >= 8:
                break

    print("\n== C. alive_gap_snapshots definition check ==")
    rounds_g = rounds.select(["alive_gap_snapshots"]).to_series()
    print(f"alive_gap_snapshots: sum {int(rounds_g.sum())}, rounds >0 {int((rounds_g > 0).sum())}, "
          f"max {int(rounds_g.max())}")
    print("extraction defines alive_gap via validation of snapshot-alive vs death-based alive; "
          "see scripts/eda/extract_state_1hz.py main() lines ~510-520")


if __name__ == "__main__":
    main()
