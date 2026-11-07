"""E2 map profile (swarm task). Per map_name: matches, rounds, CT win rate with 95% intervals, average live round
length, plant rate, plant-to-end time, kills per round. Two tomes on G: only (2026-10-01_to_08 and 2024-02-g).

Definitions (all from the tomes; no E: reads):
  - Dedupe: matches with an identical event signature (round_end, player_death, player_info), as in
    scripts\\eda\\dedupe_replicate.py. Keep the lowest match_key in each group. Matches whose round_end has a repeated
    round number after dedupe are dropped (reported).
  - Clean 5v5 round: round_end.player_count == 10 (2026) or == 20 (2024; the 2024 channel counts pawns and
    controllers, so a 5v5 round has 20). Wingman rounds (4 in 2026, 8 in 2024) are excluded.
  - CT win: round_end.winner_team_code == 3 (side-based; 2 = T). Over all clean rounds.
  - Live start: first round_state round_freeze_end tick of the round. Live end: round_end tick.
    Live length = (end - live start) / tick_rate. round_start ticks are not used (tick 66 placeholder in round 1).
  - Plant: bomb_state event bomb_planted with tick inside [live start, round end]. Plant rate denominator = clean
    rounds with a live start in matches that have at least one bomb_state row (missing bomb channel excluded).
    Plant-to-end = (round end - plant tick) / tick_rate for planted rounds.
  - Kill: player_death in [live start, round end] with a known attacker side (2 or 3), attacker slot known,
    attacker slot != victim slot, and not a team kill (same known side code). Victim slot unknown counts as enemy.
    Deaths counted per round; CT and T kills split by attacker side. Denominator: clean rounds with a live start.
  - CI: Wilson 95% on rounds (rounds treated as Bernoulli), plus a match-cluster bootstrap (B resamples of matches,
    seed 0) as a sensitivity, because rounds in one match share teams.

Run: pixi run python -I -X utf8 scripts\\swarm\\e2_map_profile.py
Writes: scripts\\swarm\\out\\e2_map_profile.json and prints Markdown tables to stdout.
"""
import collections
import glob
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(r"G:\pureskill\csds-cs2")
OUT_JSON = ROOT / "scripts" / "swarm" / "out" / "e2_map_profile.json"
ERAS = {
    "2026": {"base": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
             "suffix": ".2026-09-18,2026-10-08", "pc": 10},
    "2024": {"base": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
             "suffix": ".2023-12-28,2024-02-25", "pc": 20},
}
T_SIDE, CT_SIDE = 2, 3
Z = 1.959963984540054
B = 2000
SEED = 0
MIN_ROUNDS = 100
DEFAULT_TICK = 64


def read(base, suffix, ch, cols=None):
    fd = base / ("header" if ch == "header" else f"basic_{ch}{suffix}")
    files = sorted(glob.glob(str(fd / "dataframe_*")))
    if not files:
        raise SystemExit(f"no parquet pages for {ch} in {fd}")
    return pl.concat([pl.read_parquet(f, columns=cols) for f in files], how="vertical_relaxed")


def dedupe(RE, DE, PI):
    parts = collections.defaultdict(list)
    for m, r, t, w in RE.select(["match_key", "round", "tick", "winner_team_code"]).iter_rows():
        parts[m].append(repr(("R", int(r), int(t), int(w))))
    for m, r, t, p in DE.select(["match_key", "round", "tick", "player_id"]).iter_rows():
        parts[m].append(repr(("D", int(r), int(t), str(p))))
    for m, r, p, tc in PI.select(["match_key", "round", "player_id", "team_code"]).iter_rows():
        parts[m].append(repr(("P", int(r), str(p), str(tc))))
    groups = collections.defaultdict(list)
    for m, lst in parts.items():
        groups[tuple(sorted(lst))].append(m)
    drop, groups_out = set(), []
    for members in groups.values():
        if len(members) > 1:
            members = sorted(members)
            drop.update(members[1:])
            groups_out.append(members)
    return drop, groups_out


def load_era(name, cfg):
    base, suffix, pc = cfg["base"], cfg["suffix"], cfg["pc"]
    hdr_cols = ["key", "map_name", "match_date", "tick_rate"] + (["is_wingman"] if name == "2026" else [])
    hdr = read(base, suffix, "header", hdr_cols).rename({"key": "match_key"})
    RE = read(base, suffix, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"])
    RS = read(base, suffix, "round_state", ["match_key", "round", "tick", "event_type"])
    BS = read(base, suffix, "bomb_state", ["match_key", "round", "tick", "event_type"])
    DE = read(base, suffix, "player_death", ["match_key", "round", "tick", "player_id", "player_id_fixed",
                                             "attacker_id_fixed", "player_team_code", "attacker_team_code"])
    PI = read(base, suffix, "player_info", ["match_key", "round", "player_id", "team_code"])
    d = {"header_matches": hdr.height, "header_unique_keys": hdr["match_key"].n_unique()}

    drop, groups = dedupe(RE, DE, PI)
    d["dup_groups"] = len(groups)
    d["dup_members"] = sum(len(g) for g in groups)
    d["dup_removed"] = len(drop)
    d["dup_removed_by_map"] = dict(collections.Counter(
        hdr.filter(pl.col("match_key").is_in(list(drop)))["map_name"].to_list()))
    keep = set(hdr["match_key"].to_list()) - drop
    RE = RE.filter(pl.col("match_key").is_in(list(keep)))
    rdup = RE.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1)
    bad = set(rdup["match_key"].to_list())
    d["repeat_round_matches_dropped"] = len(bad)
    RE = RE.filter(~pl.col("match_key").is_in(list(bad)))
    keep -= bad
    d["matches_after_dedupe"] = len(keep)
    if "is_wingman" in hdr.columns:
        w = hdr.filter(pl.col("match_key").is_in(list(keep)))
        d["wingman_matches_after_dedupe"] = int(w.filter(pl.col("is_wingman")).height)
        d["wingman_clean_rounds_check"] = int(RE.join(w.select(["match_key", "is_wingman"]), on="match_key")
                                              .filter(pl.col("is_wingman") & (pl.col("player_count") == pc)).height)
    d["round_end_rows_after_dedupe"] = RE.height
    d["player_count_dist_after_dedupe"] = {int(k): int(v) for k, v in
                                           RE.group_by("player_count").len().iter_rows()}

    clean = RE.filter(pl.col("player_count") == pc).select(["match_key", "round", "tick", "winner_team_code"])
    clean = clean.rename({"tick": "end_tick"})
    d["clean_rounds"] = clean.height
    d["clean_matches"] = clean["match_key"].n_unique()

    # check: distinct player_info slots in clean rounds
    slots = (PI.filter(pl.col("match_key").is_in(list(keep)))
             .join(clean.select(["match_key", "round"]), on=["match_key", "round"], how="inner")
             .group_by(["match_key", "round"]).agg(pl.col("player_id").n_unique().alias("n")))
    d["clean_rounds_with_10_player_info_slots"] = int(slots.filter(pl.col("n") == 10).height)
    d["clean_rounds_with_player_info_slots_checked"] = int(slots.height)

    live = (RS.filter(pl.col("event_type") == "round_freeze_end")
            .group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("live_tick")))
    rounds = clean.join(live, on=["match_key", "round"], how="left")
    rounds = rounds.join(hdr.select(["match_key", "map_name", "match_date", "tick_rate"]), on="match_key", how="left")
    rounds = rounds.with_columns(pl.col("tick_rate").fill_null(DEFAULT_TICK))
    d["clean_rounds_no_live_start"] = int(rounds.filter(pl.col("live_tick").is_null()).height)
    d["tick_rate_values"] = {int(k): int(v) for k, v in rounds.group_by("tick_rate").len().iter_rows()}

    # bomb channel presence and plant
    bomb_matches = set(BS["match_key"].unique().to_list()) & keep
    d["clean_matches_without_bomb_rows"] = len(set(clean["match_key"].to_list()) - bomb_matches)
    d["clean_rounds_in_matches_without_bomb_rows"] = int(
        rounds.filter(~pl.col("match_key").is_in(list(bomb_matches))).height)
    planted = BS.filter(pl.col("event_type") == "bomb_planted").select(["match_key", "round", "tick"]).rename({"tick": "plant_tick"})
    pa = planted.join(rounds.select(["match_key", "round", "live_tick", "end_tick"]), on=["match_key", "round"], how="inner")
    d["planted_rows_total"] = planted.height
    d["planted_rows_in_clean_rounds"] = pa.height
    d["planted_rows_before_live"] = int(pa.filter(pl.col("plant_tick") < pl.col("live_tick")).height)
    d["planted_rows_after_end"] = int(pa.filter(pl.col("plant_tick") > pl.col("end_tick")).height)
    d["rounds_with_2plus_planted_rows"] = int(planted.group_by(["match_key", "round"]).len().filter(pl.col("len") > 1).height)
    pw = (pa.filter((pl.col("plant_tick") >= pl.col("live_tick")) & (pl.col("plant_tick") <= pl.col("end_tick")))
          .group_by(["match_key", "round"]).agg(pl.col("plant_tick").min()))
    rounds = rounds.join(pw, on=["match_key", "round"], how="left")

    # kills
    dj = DE.join(rounds.select(["match_key", "round", "live_tick", "end_tick"]), on=["match_key", "round"], how="inner")
    d["deaths_in_clean_rounds"] = dj.height
    dj = dj.filter((pl.col("tick") >= pl.col("live_tick")) & (pl.col("tick") <= pl.col("end_tick")))
    d["deaths_in_live_window"] = dj.height
    valid_att = (pl.col("attacker_team_code").is_in([T_SIDE, CT_SIDE]).fill_null(False)
                 & pl.col("attacker_id_fixed").is_not_null())
    self_kill = (pl.col("attacker_id_fixed") == pl.col("player_id_fixed")).fill_null(False)
    team_kill = (pl.col("attacker_team_code") == pl.col("player_team_code")).fill_null(False)
    dj = dj.with_columns(
        (valid_att & ~self_kill & ~team_kill).alias("is_kill"),
        (valid_att & ~self_kill & team_kill).alias("is_team_kill"),
        (valid_att & self_kill).alias("is_self"),
    )
    dj = dj.with_columns((pl.col("is_kill") & (pl.col("attacker_team_code") == CT_SIDE)).alias("ct_kill"),
                         (pl.col("is_kill") & (pl.col("attacker_team_code") == T_SIDE)).alias("t_kill"))
    d["kills_in_live_window"] = int(dj.filter(pl.col("is_kill")).height)
    d["team_kills_in_live_window"] = int(dj.filter(pl.col("is_team_kill")).height)
    d["self_kills_in_live_window"] = int(dj.filter(pl.col("is_self")).height)
    d["deaths_no_attacker_side_in_live_window"] = int(dj.filter(~valid_att).height)
    agg = dj.group_by(["match_key", "round"]).agg(
        pl.col("is_kill").sum().alias("kills"),
        pl.col("ct_kill").sum().alias("ct_kills"),
        pl.col("t_kill").sum().alias("t_kills"),
    )
    rounds = rounds.join(agg, on=["match_key", "round"], how="left").with_columns(
        pl.col("kills").fill_null(0), pl.col("ct_kills").fill_null(0), pl.col("t_kills").fill_null(0))

    rounds = rounds.with_columns(
        (pl.col("winner_team_code") == CT_SIDE).cast(pl.Int8).alias("ct_win"),
        pl.col("match_key").is_in(list(bomb_matches)).alias("bomb_ok"),
        ((pl.col("end_tick") - pl.col("live_tick")) / pl.col("tick_rate")).alias("live_len_s"),
        ((pl.col("end_tick") - pl.col("plant_tick")) / pl.col("tick_rate")).alias("plant_to_end_s"),
    )
    rounds = rounds.with_columns(
        pl.when(pl.col("live_tick").is_null()).then(None).otherwise(pl.col("kills")).alias("kills_live"),
        pl.when(pl.col("live_tick").is_null()).then(None).otherwise(pl.col("ct_kills")).alias("ct_kills_live"),
        pl.when(pl.col("live_tick").is_null()).then(None).otherwise(pl.col("t_kills")).alias("t_kills_live"),
    )
    return hdr, rounds, d, groups


def wilson(k, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return (c - h, c + h)


def cluster_boot(wins, ns, rng):
    m = len(wins)
    if m == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, m, size=(B, m))
    W = wins[idx].sum(axis=1)
    N = ns[idx].sum(axis=1)
    r = W / N
    return (float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5)))


def summarize(rows, rng):
    """rows: list of dicts (clean rounds of one map or of all maps)."""
    n = len(rows)
    k = sum(r["ct_win"] for r in rows)
    per_match = collections.OrderedDict()
    for r in rows:
        mk = per_match.setdefault(r["match_key"], [0, 0])
        mk[0] += r["ct_win"]
        mk[1] += 1
    wins = np.array([v[0] for v in per_match.values()], dtype=float)
    ns = np.array([v[1] for v in per_match.values()], dtype=float)
    lo, hi = wilson(k, n)
    blo, bhi = cluster_boot(wins, ns, rng)
    live = [r for r in rows if r["live_tick"] is not None]
    lens = [r["live_len_s"] for r in live]
    planted_rows = [r for r in live if r["bomb_ok"]]
    planted = [r for r in planted_rows if r["plant_tick"] is not None]
    pte = [r["plant_to_end_s"] for r in planted]
    kl = [r["kills_live"] for r in live]
    ctk = [r["ct_kills_live"] for r in live]
    tk = [r["t_kills_live"] for r in live]
    return {
        "matches": len(per_match),
        "rounds": n,
        "ct_wins": k,
        "ct_win_rate": k / n if n else None,
        "ct_ci95_wilson": [lo, hi],
        "ct_ci95_match_cluster_boot": [blo, bhi],
        "low_n_flag": n < MIN_ROUNDS,
        "rounds_with_live_start": len(live),
        "avg_live_round_len_s": statistics.fmean(lens) if lens else None,
        "median_live_round_len_s": statistics.median(lens) if lens else None,
        "rounds_bomb_ok": len(planted_rows),
        "planted_rounds": len(planted),
        "plant_rate": len(planted) / len(planted_rows) if planted_rows else None,
        "plant_to_end_mean_s": statistics.fmean(pte) if pte else None,
        "plant_to_end_median_s": statistics.median(pte) if pte else None,
        "kills_per_round_mean": statistics.fmean(kl) if kl else None,
        "kills_per_round_median": statistics.median(kl) if kl else None,
        "ct_kills_per_round": statistics.fmean(ctk) if ctk else None,
        "t_kills_per_round": statistics.fmean(tk) if tk else None,
    }


def batch_split(rows):
    a = [r for r in rows if r["match_date"] and r["match_date"][:10] == "2026-10-08"]
    b = [r for r in rows if r["match_date"] and r["match_date"][:10] != "2026-10-08"]

    def f(rs):
        return (sum(r["ct_win"] for r in rs), len(rs))

    (ka, na), (kb, nb) = f(a), f(b)
    return {"1008_rounds": na, "1008_ct_rate": ka / na if na else None,
            "other_days_rounds": nb, "other_days_ct_rate": kb / nb if nb else None}


def fmt_pct(x, nd=1):
    return "n/a" if x is None else f"{100 * x:.{nd}f}%"


def fmt_num(x, nd=1):
    return "n/a" if x is None else f"{x:.{nd}f}"


def md_table(maps):
    head = ["Map", "Matches", "Rounds", "CT win", "95% CI (Wilson)", "95% CI (match bootstrap)",
            "Avg live round (s)", "Plant rate", "Plant-to-end (s, mean / median)",
            "Kills per round (all / CT / T)", "Flag"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for mp, s in maps.items():
        flag = "n<100" if s["low_n_flag"] else ""
        ci_w = f"{100*s['ct_ci95_wilson'][0]:.1f} to {100*s['ct_ci95_wilson'][1]:.1f}%"
        ci_b = f"{100*s['ct_ci95_match_cluster_boot'][0]:.1f} to {100*s['ct_ci95_match_cluster_boot'][1]:.1f}%"
        pte = f"{fmt_num(s['plant_to_end_mean_s'])} / {fmt_num(s['plant_to_end_median_s'])}"
        kills = (f"{fmt_num(s['kills_per_round_mean'], 2)} / {fmt_num(s['ct_kills_per_round'], 2)}"
                 f" / {fmt_num(s['t_kills_per_round'], 2)}")
        lines.append("| " + " | ".join([mp, str(s["matches"]), str(s["rounds"]), fmt_pct(s["ct_win_rate"], 1),
                                       ci_w, ci_b, fmt_num(s["avg_live_round_len_s"]), fmt_pct(s["plant_rate"], 1),
                                       pte, kills, flag]) + " |")
    return "\n".join(lines)


def main():
    rng = np.random.default_rng(SEED)
    result = {"definitions": "see module docstring", "eras": {}}
    for name, cfg in ERAS.items():
        hdr, rounds, d, groups = load_era(name, cfg)
        by_map = collections.defaultdict(list)
        all_rows = []
        for r in rounds.iter_rows(named=True):
            by_map[r["map_name"]].append(r)
            all_rows.append(r)
        maps = {}
        for mp in sorted(by_map, key=lambda m: -len(by_map[m])):
            rows = by_map[mp]
            s = summarize(rows, rng)
            if name == "2026":
                s["batch_split"] = batch_split(rows)
            maps[mp] = s
        total = summarize(all_rows, rng)
        result["eras"][name] = {"diagnostics": d, "maps": maps, "all_maps": total,
                                "dup_groups_sample": groups[:5]}
        print(f"\n===== {name}: diagnostics")
        for k, v in d.items():
            print(f"  {k}: {v}")
        print(f"\n### {name} (clean 5v5 rounds; deduped matches)\n")
        print(md_table(maps))
        t = total
        print(f"\nAll maps {name}: matches {t['matches']}, rounds {t['rounds']}, CT win {fmt_pct(t['ct_win_rate'], 2)}, "
              f"Wilson {100*t['ct_ci95_wilson'][0]:.2f} to {100*t['ct_ci95_wilson'][1]:.2f}, "
              f"plant rate {fmt_pct(t['plant_rate'], 2)}, avg live round {fmt_num(t['avg_live_round_len_s'], 2)} s, "
              f"kills/round {fmt_num(t['kills_per_round_mean'], 3)}")
        if name == "2026":
            print("\nBatch split (2026-10-08 vs other days), CT rate:")
            for mp, s in maps.items():
                bs = s["batch_split"]
                print(f"  {mp}: 10-08 {bs['1008_rounds']} rounds {fmt_pct(bs['1008_ct_rate'], 1)}; "
                      f"other days {bs['other_days_rounds']} rounds {fmt_pct(bs['other_days_ct_rate'], 1)}")
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
