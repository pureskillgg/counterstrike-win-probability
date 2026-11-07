"""Task 1: profile the flow channels (read-only).

Run: pixi run python -I scripts\\eda\\flow_profile.py > scratch output (markdown-ish tables).
Reads each channel's dataframe_* pages under the tome; writes nothing.
"""
import collections
import glob
import math
from pathlib import Path

import polars as pl

ROOT = Path(r"G:\pureskill\csds-cs2")
TOME = ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds"
SUFFIX = ".2026-09-18,2026-10-08"
TICK_RATE = 64
FLOW = ["round_end", "round_start", "round_state", "bomb_state", "bomb_action",
        "bomb_defuse", "round_mvp", "score_update"]
WIN_REASON = {1: "Target_Bombed", 7: "Bomb_Defused", 8: "CTs_Win", 9: "Terrorists_Win",
              11: "All_Hostages_Rescued", 12: "Target_Saved", 13: "Hostages_Not_Rescued",
              17: "Terrorists_Surrender", 18: "CTs_Surrender"}


def read(ch):
    folder = TOME / ("header" if ch == "header" else f"basic_{ch}{SUFFIX}")
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    if not files:
        raise SystemExit(f"no pages in {folder}")
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")


def is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


def fnum(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def col_report(df, name):
    s = df.get_column(name)
    vals = s.to_list()
    n = len(vals)
    nulls = sum(1 for v in vals if v is None)
    nans = sum(1 for v in vals if isinstance(v, float) and math.isnan(v))
    good = [v for v in vals if not is_missing(v)]
    cnt = collections.Counter(good)
    nd = len(cnt)
    if not good or isinstance(good[0], bool):
        mn = mx = "-"
    elif isinstance(good[0], (int, float)):
        mn, mx = fnum(min(good)), fnum(max(good))
    else:
        lens = [len(v) for v in good]
        mn, mx = f"len {min(lens)}", f"len {max(lens)}"
    if name == "match_key":
        top = "-"
    elif isinstance(good[0] if good else None, bool):
        top = "; ".join(f"{k}={c}" for k, c in sorted(cnt.items()))
    elif nd <= 12:
        top = "; ".join(f"{fnum(k)}={c}" for k, c in sorted(cnt.items(), key=lambda kv: str(kv[0])))
    else:
        top = "top3: " + "; ".join(f"{fnum(k)}={c}" for k, c in cnt.most_common(3))
    null_pct = 100.0 * nulls / n if n else 0.0
    print(f"| {name} | {s.dtype} | {null_pct:.3f}% | {nans} | {nd} | {mn} | {mx} | {top[:160]} |")


def profile_columns(ch, df):
    print(f"\n### {ch}: rows={df.height}, columns={len(df.columns)}")
    print("| column | dtype | null % | NaN | distinct | min | max | values or top3 |")
    print("|---|---|---|---|---|---|---|---|")
    for c in df.columns:
        col_report(df, c)


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def round_checks(ch, df, round_end_pairs, start_by_pair, end_by_pair):
    """Checks shared by any channel keyed on (match_key, round)."""
    rows = list(zip(df["match_key"].to_list(), df["round"].to_list(), df["tick"].to_list()))
    n = len(rows)
    pair_ct = collections.Counter((m, r) for m, r, _ in rows)
    dup_pairs = sum(c - 1 for c in pair_ct.values() if c > 1)
    orphan = sum(1 for m, r, _ in rows if (m, r) not in round_end_pairs)
    neg_tick = sum(1 for _, _, t in rows if t is not None and t < 0)
    zero_tick = sum(1 for _, _, t in rows if t == 0)
    big_tick = sum(1 for _, _, t in rows if t is not None and t > 3600 * TICK_RATE)
    outside = 0
    checked = 0
    for m, r, t in rows:
        if t is None or (m, r) not in start_by_pair or (m, r) not in end_by_pair:
            continue
        checked += 1
        if t < start_by_pair[(m, r)] or t > end_by_pair[(m, r)]:
            outside += 1
    print(f"- {ch} checks: rows={n}; (match,round) pairs with >1 row: "
          f"{sum(1 for c in pair_ct.values() if c > 1)} (extra rows {dup_pairs}); "
          f"rows whose (match,round) is not in round_end: {orphan}; "
          f"negative tick: {neg_tick}; tick==0: {zero_tick}; tick > 1h: {big_tick}; "
          f"tick outside its round's [round_start, round_end] window: {outside} of {checked} checked")


def main():
    print("# flow profile raw output")
    header = read("header")
    print(f"header rows={header.height}, distinct key={header['key'].n_unique()}")

    frames = {ch: read(ch) for ch in FLOW}
    re_ = frames["round_end"]
    rs_ = frames["round_start"]

    # round windows from round_start and round_end (first row per pair)
    start_by_pair = {}
    for m, r, t in zip(rs_["match_key"].to_list(), rs_["round"].to_list(), rs_["tick"].to_list()):
        start_by_pair.setdefault((m, r), t)
    end_by_pair = {}
    for m, r, t in zip(re_["match_key"].to_list(), re_["round"].to_list(), re_["tick"].to_list()):
        end_by_pair.setdefault((m, r), t)
    round_end_pairs = set(end_by_pair)

    print("\n## Channel coverage")
    print("| channel | rows | matches with rows | header matches with 0 rows | rows per match (median) |")
    print("|---|---|---|---|---|")
    header_keys = set(header["key"].to_list())
    for ch in FLOW:
        df = frames[ch]
        have = set(df["match_key"].to_list())
        zero = len(header_keys - have)
        per = collections.Counter(df["match_key"].to_list())
        med = sorted(per.values())[len(per) // 2] if per else 0
        extra = len(have - header_keys)
        print(f"| {ch} | {df.height} | {len(have)} | {zero} | {med} | extra keys not in header: {extra}")

    print("\n## Column profiles")
    for ch in FLOW:
        profile_columns(ch, frames[ch])

    print("\n## Round-level counts and checks")

    # round_end: rounds per match, numbering, ordering, durations
    per_match = collections.defaultdict(list)
    for m, r, t, w, wr, pc, sec in zip(re_["match_key"].to_list(), re_["round"].to_list(),
                                       re_["tick"].to_list(), re_["winner_team_code"].to_list(),
                                       re_["win_reason_code"].to_list(), re_["player_count"].to_list(),
                                       re_["second"].to_list()):
        per_match[m].append((r, t, w, wr, pc, sec))
    rpm = collections.Counter(len(v) for v in per_match.values())
    n_matches = len(per_match)
    outside_16_30 = sum(c for k, c in rpm.items() if k < 16 or k > 30)
    print(f"- round_end matches: {n_matches}; rounds per match distribution: {dict(sorted(rpm.items()))}")
    print(f"- matches with rounds per match outside 16..30: {outside_16_30} "
          f"(rounds in them: {sum(k * c for k, c in rpm.items() if k < 16 or k > 30)})")
    gaps = 0
    first_not_1 = 0
    dup_round_matches = 0
    out_of_order = 0
    for m, lst in per_match.items():
        nums = [x[0] for x in lst]
        if min(nums) != 1:
            first_not_1 += 1
        if len(set(nums)) != len(nums):
            dup_round_matches += 1
        if set(nums) != set(range(1, max(nums) + 1)):
            gaps += 1
        by_tick = [x[0] for x in sorted(lst, key=lambda x: (x[1], x[0]))]
        if by_tick != sorted(by_tick):
            out_of_order += 1
    print(f"- round numbering: matches with min round != 1: {first_not_1}; matches with duplicate "
          f"round numbers: {dup_round_matches}; matches with gaps in 1..max: {gaps}; matches where round "
          f"order disagrees with tick order: {out_of_order}")
    neg_sec = sum(1 for lst in per_match.values() for x in lst if x[5] is not None and x[5] < 0)
    neg_tick_re = sum(1 for lst in per_match.values() for x in lst if x[1] < 0)
    print(f"- round_end negative tick: {neg_tick_re}; negative second: {neg_sec}")
    diffs = [x[5] - x[1] / TICK_RATE for lst in per_match.values() for x in lst
             if x[5] is not None]
    if diffs:
        diffs_sorted = sorted(diffs)
        print(f"- round_end second minus tick/64: min {diffs_sorted[0]:.3f}, "
              f"median {diffs_sorted[len(diffs_sorted)//2]:.3f}, max {diffs_sorted[-1]:.3f}, "
              f"|diff| > 1s: {sum(1 for d in diffs if abs(d) > 1)}")
    dur = [(end_by_pair[k] - start_by_pair[k]) / TICK_RATE for k in end_by_pair if k in start_by_pair]
    if dur:
        ds = sorted(dur)
        print(f"- round duration from round_start to round_end (s): n={len(ds)} min {ds[0]:.2f} "
              f"median {ds[len(ds)//2]:.2f} max {ds[-1]:.2f}; negative: {sum(1 for d in ds if d < 0)}; "
              f"> 200s: {sum(1 for d in ds if d > 200)}; round_end pairs without round_start: "
              f"{sum(1 for k in end_by_pair if k not in start_by_pair)}; round_start pairs without round_end: "
              f"{sum(1 for k in start_by_pair if k not in end_by_pair)}")
    print(f"- round_end winner_team_code: {dict(collections.Counter(re_['winner_team_code'].to_list()))}")
    wr = collections.Counter(re_["win_reason_code"].to_list())
    print("- round_end win_reason_code: " + "; ".join(
        f"{k} ({WIN_REASON.get(k, 'UNMAPPED')})={v}" for k, v in sorted(wr.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))))
    print(f"- round_end player_count: {dict(collections.Counter(re_['player_count'].to_list()))}")
    print(f"- round_end (match, round) duplicate rows: {re_.height - len(round_end_pairs)}")

    rs_pairs = collections.Counter(zip(rs_["match_key"].to_list(), rs_["round"].to_list()))
    print(f"- round_start (match, round) duplicate rows: {rs_.height - len(rs_pairs)}; "
          f"round_start pairs not in round_end: {sum(1 for k in rs_pairs if k not in round_end_pairs)}")
    rs_per_match = collections.Counter(k[0] for k in rs_pairs)
    mismatch = sum(1 for m in per_match if rs_per_match.get(m, 0) != len(per_match[m]))
    print(f"- matches where distinct round_start rounds != round_end rounds: {mismatch}")

    # round_start round checks
    round_checks("round_start", rs_, round_end_pairs, start_by_pair, end_by_pair)

    # round_state
    rstate = frames["round_state"]
    print("\n### round_state event_type counts")
    for k, v in sorted(collections.Counter(rstate["event_type"].to_list()).items(), key=lambda kv: -kv[1]):
        print(f"- {k}: {v}")
    fe = collections.Counter()
    for (m, r), c in collections.Counter(
            zip(*[rstate.filter(pl.col("event_type") == "round_freeze_end")[c].to_list()
                  for c in ("match_key", "round")])).items():
        fe[c] += 1
    print(f"- round_freeze_end rows per (match,round): {dict(sorted(collections.Counter(fe.values()).items()))} "
          f"(keys: rounds with 1 row / 2 rows ...)")
    rounds_with_fe = set(zip(*[rstate.filter(pl.col("event_type") == "round_freeze_end")[c].to_list()
                               for c in ("match_key", "round")]))
    print(f"- round_end rounds with no round_freeze_end row: {len(round_end_pairs - rounds_with_fe)}")
    tsc = rstate["t_score"].to_list()
    csc = rstate["ct_score"].to_list()
    print(f"- round_state t_score range {min(x for x in tsc if x is not None)}..{max(x for x in tsc if x is not None)}; "
          f"ct_score range {min(x for x in csc if x is not None)}..{max(x for x in csc if x is not None)}; "
          f"negative t/ct score: {sum(1 for x in tsc if x is not None and x < 0)}/{sum(1 for x in csc if x is not None and x < 0)}")
    print(f"- round_state null t_score: {sum(1 for x in tsc if x is None)}; null ct_score: {sum(1 for x in csc if x is None)}")
    round_checks("round_state", rstate, round_end_pairs, start_by_pair, end_by_pair)

    # bomb channels and others
    for ch in ["bomb_state", "bomb_action", "bomb_defuse"]:
        df = frames[ch]
        print(f"\n### {ch} event_type counts")
        for k, v in sorted(collections.Counter(df["event_type"].to_list()).items(), key=lambda kv: -kv[1]):
            print(f"- {k}: {v}")
        round_checks(ch, df, round_end_pairs, start_by_pair, end_by_pair)
    bs = frames["bomb_state"]
    plants = collections.Counter(zip(*[bs.filter(pl.col("event_type") == "bomb_planted")[c].to_list()
                                       for c in ("match_key", "round")]))
    print(f"- bomb_planted rows per (match,round): {dict(sorted(collections.Counter(plants.values()).items()))}")
    print(f"- bomb_planted rounds: {len(plants)} of {len(round_end_pairs)} rounds "
          f"({100.0 * len(plants) / len(round_end_pairs):.2f}%)")

    rm = frames["round_mvp"]
    print(f"\n### round_mvp: mvp_count counts {dict(collections.Counter(rm['mvp_count'].to_list()))}")
    mvp_pairs = collections.Counter(zip(rm["match_key"].to_list(), rm["round"].to_list()))
    print(f"- round_mvp rows per (match,round): {dict(sorted(collections.Counter(mvp_pairs.values()).items()))}; "
          f"rounds with an MVP row: {len(mvp_pairs)} of {len(round_end_pairs)}")
    round_checks("round_mvp", rm, round_end_pairs, start_by_pair, end_by_pair)

    su = frames["score_update"]
    print(f"\n### score_update: team_code counts {dict(collections.Counter(su['team_code'].to_list()))}")
    su_pairs = collections.Counter(zip(su["match_key"].to_list(), su["round"].to_list()))
    print(f"- score_update rows per (match,round): {dict(sorted(collections.Counter(su_pairs.values()).items()))}; "
          f"rounds with a row: {len(su_pairs)} of {len(round_end_pairs)}")
    deltas = collections.Counter(n - o for o, n in zip(su["old_score"].to_list(), su["new_score"].to_list())
                                 if o is not None and n is not None)
    print(f"- score_update new-old delta counts: {dict(sorted(deltas.items()))}")
    print(f"- score_update negative old/new score: "
          f"{sum(1 for x in su['old_score'].to_list() if x is not None and x < 0)}/"
          f"{sum(1 for x in su['new_score'].to_list() if x is not None and x < 0)}")
    round_checks("score_update", su, round_end_pairs, start_by_pair, end_by_pair)


if __name__ == "__main__":
    main()
