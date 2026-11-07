"""E3: trades, kill sequences, opening kill and first-10-second kill difference vs ct_win.

Reads G: tomes only (no E: access, no writes except logs\\swarm\\e3_trades.json).
Population and round filters are ported from scripts/model/bakeoff.py main() (not imported,
so this script does not depend on the bake-off module). The 2024 tome has no team_change
channel, so that dense-channel check is skipped for 2024 only.

Definitions
  live start      round_state event round_freeze_end (tick), as in the bake-off.
  kill            player_death in [live start, round_end] with a valid attacker (not null,
                  not 65535, not the victim), killer side != victim side. Sides: the death
                  row's team code, else player_info for that round (bake-off rule).
  state time S    live start + S seconds (S in STATE_TIMES_S). Only events with tick < S are used.
  opening kill    the earliest kill before S; ties at that tick (different sides) are excluded.
  diff10          kills by CT minus kills by T, with the kill inside the first 10 s of live play.
                  Used only with S >= 10 s, so every event used lies before S.
  trade gap       for each kill, seconds to the next kill by the other side in the same round.
                  Descriptive only (uses the whole round, so it is not a predictor here).
  AUC             roc_auc_score against ct_win (1 if round_end winner is CT, team code 3).
                  95% CI: percentile bootstrap over matches (cluster bootstrap), B resamples.

Run from the project root:
  pixi run python -I -X utf8 scripts\\swarm\\e3_trades.py
"""

import collections
import glob
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
OUT_JSON = ROOT / "logs" / "swarm" / "e3_trades.json"
T, CT, START = 2, 3, 5          # team codes verified in scripts/model/feature_validation.py (T=2, CT=3)
TICK = 64
SENTINEL = 65535
STATE_TIMES_S = [10, 20, 30]
WINDOW_S = 10
B = 1000
SEED = 0

ERAS = {
    "2026": {
        "tome": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
        "suffix": ".2026-09-18,2026-10-08",
        "dense": ["round_end", "round_start", "round_state", "player_death", "player_info",
                  "bomb_state", "team_change"],
        "sparse": ["player_disconnect", "bot_takeover"],
    },
    "2024": {
        "tome": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "suffix": ".2023-12-28,2024-02-25",
        "dense": ["round_end", "round_start", "round_state", "player_death", "player_info",
                  "bomb_state"],
        "sparse": ["player_disconnect", "bot_takeover"],
    },
}

CH_COLS = {
    "round_end": ["match_key", "round", "tick", "winner_team_code", "player_count"],
    "round_start": ["match_key", "round", "tick"],
    "round_state": ["match_key", "round", "tick", "event_type", "t_score", "ct_score"],
    "player_death": ["match_key", "round", "tick", "player_id", "attacker_id",
                     "player_team_code", "attacker_team_code"],
    "player_info": ["match_key", "round", "player_id", "team_code"],
    "bomb_state": ["match_key", "round", "tick", "event_type"],
    "player_disconnect": ["match_key", "round", "tick"],
    "bot_takeover": ["match_key", "round", "tick"],
    "team_change": ["match_key", "round", "tick"],
}


def read_dir(d, cols):
    files = sorted(glob.glob(str(d / "dataframe_*")))
    if not files:
        raise SystemExit(f"no dataframe files in {d}")
    have = pq.read_schema(files[0]).names
    use = [c for c in cols if c in have]
    if not use:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f, columns=use) for f in files], how="vertical_relaxed")


def keyset(d):
    files = sorted(glob.glob(str(d / "keyset_*")))
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


def day_of(key):
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", key)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def auc_boot(y, x, groups, rng, B):
    y = np.asarray(y, dtype=np.int64)
    x = np.asarray(x, dtype=float)
    point = float(roc_auc_score(y, x))
    uniq, inv = np.unique(np.asarray(groups), return_inverse=True)
    n_g = len(uniq)
    reps = []
    for _ in range(B):
        counts = np.bincount(rng.integers(0, n_g, n_g), minlength=n_g)
        w = counts[inv].astype(float)
        if w[y == 1].sum() == 0 or w[y == 0].sum() == 0:
            continue
        reps.append(float(roc_auc_score(y, x, sample_weight=w)))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi), len(reps)


def trade_gaps(kills):
    """kills: list of (rel_tick, side) in time order. Returns gaps in seconds to the next
    other-side kill, and the count of kills with no later other-side kill."""
    gaps, none_after = [], 0
    for i, (ti, si) in enumerate(kills):
        nxt = None
        for tj, sj in kills[i + 1:]:
            if sj != si:
                nxt = tj
                break
        if nxt is None:
            none_after += 1
        else:
            gaps.append((nxt - ti) / TICK)
    return gaps, none_after


def analyze(rounds, S, rng, keep=lambda r: True, label=""):
    S_ticks = S * TICK
    W = WINDOW_S * TICK
    assert S >= WINDOW_S, "diff10 needs S >= window, or it would use events after S"
    rows = []
    for r in rounds:
        if not keep(r) or r["end_rel"] < S_ticks:
            continue
        pre = [(rel, ks) for rel, ks in r["kills"] if rel < S_ticks]
        open_ct, tie = None, False
        if pre:
            first = min(rel for rel, _ in pre)
            sides = {ks for rel, ks in pre if rel == first}
            if len(sides) == 1:
                open_ct = int(sides.pop() == CT)
            else:
                tie = True
        in10 = [ks for rel, ks in pre if rel < W]
        diff = sum(1 for ks in in10 if ks == CT) - sum(1 for ks in in10 if ks == T)
        rows.append({"match": r["match"], "ct_win": r["ct_win"], "open_ct": open_ct,
                     "tie": tie, "had_pre_kill": bool(pre), "diff10": diff, "n10": len(in10)})
    n = len(rows)
    y_all = np.array([x["ct_win"] for x in rows])
    g_all = [x["match"] for x in rows]
    opn = [x for x in rows if x["open_ct"] is not None]
    out = {
        "label": label, "S_s": S, "n_rounds": n,
        "n_matches": len(set(g_all)), "ct_win_rate": float(y_all.mean()) if n else None,
        "rounds_no_pre_kill": sum(1 for x in rows if not x["had_pre_kill"]),
        "rounds_opening_tie": sum(1 for x in rows if x["tie"]),
        "rounds_with_opening": len(opn),
        "share_with_opening": len(opn) / n if n else None,
    }
    if opn:
        y_o = [x["ct_win"] for x in opn]
        out["open_ct_win_rate_when_CT_open"] = float(np.mean([x["ct_win"] for x in opn if x["open_ct"] == 1]))
        out["open_ct_win_rate_when_T_open"] = float(np.mean([x["ct_win"] for x in opn if x["open_ct"] == 0]))
        out["AUC_opening_CT"] = dict(zip(["auc", "ci95_lo", "ci95_hi", "B_valid"],
                                         auc_boot(y_o, [x["open_ct"] for x in opn],
                                                  [x["match"] for x in opn], rng, B)))
    if n:
        out["AUC_diff10"] = dict(zip(["auc", "ci95_lo", "ci95_hi", "B_valid"],
                                     auc_boot(y_all, [x["diff10"] for x in rows], g_all, rng, B)))
        dist = collections.defaultdict(lambda: [0, 0])
        for x in rows:
            dist[max(-4, min(4, x["diff10"]))][0] += 1
            dist[max(-4, min(4, x["diff10"]))][1] += x["ct_win"]
        out["diff10_ct_win_rate_by_value"] = {str(k): {"n": v[0], "ct_win_rate": round(v[1] / v[0], 4)}
                                              for k, v in sorted(dist.items())}
        out["mean_kills_first10s"] = float(np.mean([x["n10"] for x in rows]))
    return out


def run_era(era, rng):
    t0 = time.time()
    cfg = ERAS[era]
    res = {"era": era}
    print(f"\n########## ERA {era} ##########")
    hdr = read_dir(cfg["tome"] / "header", ["key", "match_id", "match_date", "tick_rate", "map_name"])
    H = {r["key"]: r for r in hdr.iter_rows(named=True)}
    all_matches = set(H)
    tick_rates = collections.Counter(r["tick_rate"] for r in H.values())
    print(f"header matches: {len(all_matches)}; tick_rate: {dict(tick_rates)}")
    assert set(tick_rates) == {TICK}, "unexpected tick rate"
    res["header_matches"] = len(all_matches)

    frames = {}
    for ch in cfg["dense"] + cfg["sparse"]:
        d = cfg["tome"] / f"basic_{ch}{cfg['suffix']}"
        frames[ch] = read_dir(d, CH_COLS[ch])
    print(f"channels loaded in {time.time() - t0:.1f}s")

    # 1. match-level channel check (bake-off rule)
    missing_by_match = collections.defaultdict(list)
    chan_report = {}
    for ch in cfg["dense"] + cfg["sparse"]:
        df = frames[ch]
        ks = keyset(cfg["tome"] / f"basic_{ch}{cfg['suffix']}")
        have = set(df["match_key"].unique().to_list()) if df.width else set()
        dense = ch in cfg["dense"]
        no_ks = all_matches - ks
        zero = all_matches - have
        miss = no_ks | (zero if dense else set())
        for m in miss:
            missing_by_match[m].append(ch)
        chan_report[ch] = {"rows": df.height, "keyset": len(ks), "zero_row_matches": len(zero),
                           "counted_missing": len(miss), "dense": dense}
    kept_matches = all_matches - set(missing_by_match)
    res["channel_check"] = chan_report
    res["matches_after_channel_filter"] = len(kept_matches)
    print(f"matches dropped for a missing channel: {len(set(missing_by_match))}; kept: {len(kept_matches)}")
    kept_list = list(kept_matches)

    def keep(df):
        # zero-width frame = channel file with no columns (2024 bot_takeover has 0 rows): no events
        if df.width == 0:
            return df
        return df.filter(pl.col("match_key").is_in(kept_list))

    RE = keep(frames["round_end"])
    RS = keep(frames["round_state"])
    DE = keep(frames["player_death"]).sort("tick")
    PI = keep(frames["player_info"])
    RST = keep(frames["round_start"])
    DC = keep(frames["player_disconnect"])
    BT = keep(frames["bot_takeover"])

    rd = {}
    for x in RE.iter_rows(named=True):
        rd[(x["match_key"], int(x["round"]))] = {"end": int(x["tick"]), "winner": x["winner_team_code"],
                                                 "pc": x["player_count"]}
    live, live_dups = {}, 0
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        k = (x["match_key"], int(x["round"]))
        if k in live:
            live_dups += 1
            continue
        live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True)}
    players = collections.defaultdict(dict)
    for x in PI.iter_rows(named=True):
        players[(x["match_key"], int(x["round"]))][x["player_id"]] = x["team_code"]
    disc_rounds = set()
    for df in (DC, BT):
        if df.width:
            for x in df.iter_rows(named=True):
                disc_rounds.add((x["match_key"], int(x["round"])))
    bomb_ev = collections.defaultdict(set)
    for x in RS.filter(pl.col("event_type").is_in(["bomb_exploded", "bomb_defused"])).iter_rows(named=True):
        bomb_ev[(x["match_key"], int(x["round"]))].add(x["event_type"])
    print(f"live-start rows: {len(live)} (duplicates skipped {live_dups})")

    # 2. round filters, in bake-off order
    reasons = collections.Counter()
    survivors = []
    for k in sorted(rd):
        rr = rd[k]
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if rr["pc"] != 10:
            reasons["round_end.player_count != 10"] += 1
            continue
        if not (len(pl_) == 10 and n_t == START and n_ct == START):
            reasons["not exactly 5 T and 5 CT at start (player_info)"] += 1
            continue
        if k in disc_rounds:
            reasons["disconnect or bot takeover in round"] += 1
            continue
        if k not in rst_keys:
            reasons["no round_start row"] += 1
            continue
        if k not in live:
            reasons["no round_freeze_end row"] += 1
            continue
        if live[k]["t"] is None or live[k]["ct"] is None:
            reasons["null score at live start"] += 1
            continue
        survivors.append(k)
    res["rounds_in_kept_matches"] = len(rd)
    res["round_filter_drops"] = dict(reasons)
    res["rounds_after_round_filters"] = len(survivors)
    print(f"rounds in kept matches: {len(rd)}; after round filters: {len(survivors)}")
    for name, n in reasons.most_common():
        print(f"  dropped: {name}: {n}")

    # 3. duplicate matches (identical event signature under two keys; lowest key kept)
    sig_parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]), str(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]), str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]), str(x["team_code"]))))
    sigs = {m: tuple(sorted(v)) for m, v in sig_parts.items()}
    sig_groups = collections.defaultdict(list)
    for m in sorted(kept_matches):
        sig_groups[sigs.get(m, ())].append(m)
    dedup_groups = [g for g in sig_groups.values() if len(g) > 1]
    dup_drop = set()
    for g in dedup_groups:
        dup_drop.update(g[1:])
    n_before = len(survivors)
    dup_with_rounds = len({k[0] for k in survivors if k[0] in dup_drop})
    survivors = [k for k in survivors if k[0] not in dup_drop]
    res["duplicate_groups"] = len(dedup_groups)
    res["duplicate_matches_removed"] = len(dup_drop)
    res["duplicate_matches_removed_with_kept_rounds"] = dup_with_rounds
    res["rounds_after_dedupe"] = len(survivors)
    print(f"duplicate groups: {len(dedup_groups)}; duplicate matches removed: {len(dup_drop)}; "
          f"rounds {n_before} -> {len(survivors)}")

    # 4. kills per round (live window), with the bake-off unresolved-victim rule
    deaths = collections.defaultdict(list)
    for x in DE.iter_rows(named=True):
        deaths[(x["match_key"], int(x["round"]))].append(x)
    kchk = collections.Counter()
    rounds = []
    for k in survivors:
        frz, end = live[k]["tick"], rd[k]["end"]
        pmap = players.get(k, {})
        kills, seen = [], set()
        unresolved = False
        for d in deaths.get(k, []):
            tick = int(d["tick"])
            if tick < frz or tick > end:
                continue
            vteam = d["player_team_code"]
            vside = vteam if vteam is not None else pmap.get(d["player_id"])
            kchk["deaths_in_window"] += 1
            if vside not in (T, CT):
                unresolved = True
                break
            if vteam is not None and pmap.get(d["player_id"]) is not None:
                kchk["victim_side_agree" if vteam == pmap[d["player_id"]] else "victim_side_disagree"] += 1
            vid = d["player_id"]
            if vid in seen:
                kchk["duplicate_victim_rows"] += 1
                continue
            seen.add(vid)
            aid = d["attacker_id"]
            if aid is None:
                kchk["no_attacker_null"] += 1
                continue
            if aid == SENTINEL:
                kchk["no_attacker_sentinel_65535"] += 1
                continue
            if aid == vid:
                kchk["suicide"] += 1
                continue
            ateam = d["attacker_team_code"]
            aside = ateam if ateam is not None else pmap.get(aid)
            if ateam is not None and pmap.get(aid) is not None:
                kchk["attacker_side_agree" if ateam == pmap[aid] else "attacker_side_disagree"] += 1
            if aside not in (T, CT):
                kchk["attacker_unresolved"] += 1
                continue
            if aside == vside:
                kchk["teamkill"] += 1
                continue
            kills.append((tick - frz, aside))
            kchk["kill"] += 1
        if unresolved:
            reasons["death with unresolved victim side in live window"] += 1
            kchk["rounds_dropped_unresolved_victim"] += 1
            continue
        kills.sort(key=lambda t: t[0])
        rounds.append({"match": k[0], "day": day_of(k[0]), "round": k[1],
                       "ct_win": int(rd[k]["winner"] == CT), "end_rel": end - frz,
                       "kills": kills, "winner": rd[k]["winner"], "live_tick": frz})
    res["kill_checks"] = dict(kchk)
    # per-round dump for the independent check (scripts/swarm/e3_verify_duckdb.py)
    dump = pl.DataFrame({
        "era": [era] * len(rounds),
        "match": [r["match"] for r in rounds],
        "round": [int(r["round"]) for r in rounds],
        "ct_win": [int(r["ct_win"]) for r in rounds],
        "live_tick": [int(r["live_tick"]) for r in rounds],
        "end_rel": [int(r["end_rel"]) for r in rounds],
        "kills_json": [json.dumps([[int(a), int(b)] for a, b in r["kills"]]) for r in rounds],
    })
    dump.write_parquet(ROOT / "logs" / "swarm" / f"e3_rounds_{era}.parquet")
    res["rounds_final"] = len(rounds)
    res["matches_final"] = len({r["match"] for r in rounds})
    print(f"final rounds: {len(rounds)} in {res['matches_final']} matches")
    print(f"kill checks: {dict(kchk)}")

    # 5. ct_win label check: bomb events vs winner (team code 3 should be CT)
    lab = collections.Counter()
    for r in rounds:
        ev = bomb_ev.get((r["match"], r["round"]), set())
        if "bomb_exploded" in ev:
            lab["exploded_rounds"] += 1
            lab["exploded_winner_T"] += int(r["winner"] == T)
        if "bomb_defused" in ev:
            lab["defused_rounds"] += 1
            lab["defused_winner_CT"] += int(r["winner"] == CT)
        if {"bomb_exploded", "bomb_defused"} <= ev:
            lab["both_events"] += 1
    res["label_check"] = dict(lab)
    print(f"label check (team 3 = CT): {dict(lab)}")

    # 6. kill sequence summaries (whole live round, descriptive)
    first_kill_s = [r["kills"][0][0] / TICK for r in rounds if r["kills"]]
    res["first_kill_s_quantiles"] = {q: float(np.quantile(first_kill_s, p)) for q, p in
                                     [("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9)]}
    res["share_rounds_first_kill_before_s"] = {str(s): float(np.mean([t < s for t in first_kill_s]))
                                               for s in (10, 20, 30, 45)}
    res["rounds_with_no_kill"] = sum(1 for r in rounds if not r["kills"])
    res["kills_per_round_mean"] = float(np.mean([len(r["kills"]) for r in rounds]))
    res["kills_per_round_max"] = max(len(r["kills"]) for r in rounds)
    all_gaps, none_after, per_round_med, per_round_n5 = [], 0, [], []
    for r in rounds:
        g, na = trade_gaps(r["kills"])
        all_gaps.extend(g)
        none_after += na
        if g:
            per_round_med.append(float(np.median(g)))
            per_round_n5.append(sum(1 for x in g if x <= 5))
    ag = np.array(all_gaps)
    res["trade_gap"] = {
        "kills_with_later_other_side_kill": int(len(ag)), "kills_without": int(none_after),
        "median_s": float(np.median(ag)), "p25_s": float(np.quantile(ag, .25)),
        "p75_s": float(np.quantile(ag, .75)), "share_le_2s": float(np.mean(ag <= 2)),
        "share_le_5s": float(np.mean(ag <= 5)), "rounds_with_gap": len(per_round_med),
        "per_round_median_gap_s": float(np.median(per_round_med)),
        "per_round_mean_trades_le_5s": float(np.mean(per_round_n5)),
    }
    print(f"trade gaps: {res['trade_gap']}")
    print(f"first kill (s after live) quantiles: {res['first_kill_s_quantiles']}; "
          f"share before s: {res['share_rounds_first_kill_before_s']}")

    # 7. AUCs at state times (primary S = 10 s; 20 and 30 s as sensitivity)
    res["analysis"] = []
    for S in STATE_TIMES_S:
        a = analyze(rounds, S, rng, label=f"{era} all")
        res["analysis"].append(a)
        print(f"[{era}] S={S}s rounds={a['n_rounds']} matches={a['n_matches']} "
              f"open_defined={a['rounds_with_opening']} ({a['share_with_opening']:.3f}) "
              f"AUC_open={a.get('AUC_opening_CT')} AUC_diff10={a.get('AUC_diff10')}")
    if era == "2026":
        for S in (10, 30):
            a = analyze(rounds, S, rng, keep=lambda r: r["day"] != "2026-10-08",
                        label=f"{era} excl 2026-10-08")
            res["analysis"].append(a)
            print(f"[{era} excl 10-08] S={S}s rounds={a['n_rounds']} "
                  f"AUC_open={a.get('AUC_opening_CT')} AUC_diff10={a.get('AUC_diff10')}")
    res["day_counts"] = dict(collections.Counter(r["day"] for r in rounds))
    print(f"elapsed {time.time() - t0:.1f}s")
    return res


def main():
    rng = np.random.default_rng(SEED)
    results = {}
    for era in ("2026", "2024"):
        results[era] = run_era(era, rng)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                        encoding="utf-8")
    print(f"\nwrote {OUT_JSON}")


if __name__ == "__main__":
    main()
