"""M1 map inventory for the bake-off kept set (2026 tome) and the same pipeline on the 2024 tome.

Ports the kept-set filters from scripts/model/bakeoff.py (match channel check, round filters,
duplicate-match removal, build_states) without importing that module, so the tome can change.
The 2026 run must reproduce the bake-off counts (1,521 matches, 28,683 rounds, 234,265 states).

Per tome, reads only the columns it needs, one channel at a time:
  - per map_name: header matches, channel-kept matches, kept matches, kept rounds, states,
    CT round win rate;
  - residual test: does map_name predict the CT round win beyond the features the model already
    uses (score lead, alive counts, plant, round time)? Residuals are clustered by match and the
    global test is a permutation test that shuffles map labels across matches;
  - 2024 round-13 side mismatch (same definition as scripts/swarm/e2_side_check_r13.py) by map.

Run: "G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe" -I -X utf8 scripts\\swarm2\\M1_map_inventory.py
Log: logs\\swarm2\\M1_map_inventory.log
"""

import collections
import glob
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
LOG = ROOT / "logs" / "swarm2" / "M1_map_inventory.log"
T, CT, START, TICK_RATE = 2, 3, 5, 64
MIN_ROUNDS, MIN_MATCHES = 100, 10
N_PERM = 2000
SEED = 0
REFERENCE_2026 = {"matches": 1521, "rounds": 28683, "states": 234265}

CONFIGS = {
    "2026": {
        "dir": ROOT / "tomes" / "2026-10-01_to_08" / "tome" / "csds",
        "suffix": ".2026-09-18,2026-10-08",
        "dense": ["round_end", "round_start", "round_state", "player_death", "player_info",
                  "bomb_state", "team_change"],
        "sparse": ["player_disconnect", "bot_takeover"],
        "default_max_rounds": None,
        "pc_full": 10,
    },
    "2024": {
        "dir": ROOT / "tomes" / "2024-02-g" / "tome" / "csds",
        "suffix": ".2023-12-28,2024-02-25",
        # ASSUMPTION: the 2024 tome has no basic_team_change folder, so team_change cannot be a
        # required dense channel. Without this change no 2024 match passes the channel check.
        "dense": ["round_end", "round_start", "round_state", "player_death", "player_info",
                  "bomb_state"],
        "sparse": ["player_disconnect", "bot_takeover"],
        # ASSUMPTION: the 2024 header has no max_rounds column. It is used only for the
        # is_pistol_round / is_overtime flags and does not change any count below.
        "default_max_rounds": 24,
        # ASSUMPTION: 2024 round_end.player_count is 20 for a full 10-player round (78,596 rounds
        # have 20; no round has 10). The bake-off's player_count == 10 rule is applied as == 20.
        # The 5 T + 5 CT check on player_info is unchanged and is the real test.
        "pc_full": 20,
    },
}

LOG.parent.mkdir(parents=True, exist_ok=True)
_LOG_FH = open(LOG, "w", encoding="utf-8")


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


sys.stdout = Tee(sys.stdout, _LOG_FH)


def files_of(folder):
    files = sorted(glob.glob(str(folder / "dataframe_*")))
    if not files:
        raise SystemExit(f"no pages in {folder}")
    return files


EMPTY_DTYPES = {"match_key": pl.Utf8, "round": pl.Int64, "tick": pl.Int64}


def read(cfg, ch, cols):
    folder = cfg["dir"] / ("header" if ch == "header" else f"basic_{ch}{cfg['suffix']}")
    frames = []
    for f in files_of(folder):
        if len(pl.read_parquet_schema(f)) == 0:
            # Empty page with no columns (2024 basic_bot_takeover has one). Zero rows.
            print(f"  note: empty page with no columns skipped: {folder.name} / {Path(f).name}")
            continue
        frames.append(pl.read_parquet(f, columns=cols))
    if not frames:
        return pl.DataFrame(schema={c: EMPTY_DTYPES[c] for c in cols})
    return pl.concat(frames, how="vertical_relaxed")


def keyset_keys(cfg, ch):
    folder = cfg["dir"] / f"basic_{ch}{cfg['suffix']}"
    files = sorted(glob.glob(str(folder / "keyset_*")))
    if not files:
        raise SystemExit(f"no keyset files in {folder}")
    return set(pl.concat([pl.read_parquet(f) for f in files])["_"].to_list())


def clip(x, lo, hi):
    return max(lo, min(hi, x))


def md_table(headers, rows):
    lines = ["| " + " | ".join(str(h) for h in headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def build_states(key, rr, deaths, plants, players, max_rounds, diag):
    """Ported from scripts/model/bakeoff.py build_states. Returns state dicts for one round, or
    None if a death in the live window has an unresolved victim side."""
    frz, end = rr["freeze"], rr["end"]
    win, unresolved = [], 0
    for d in deaths:
        if d["tick"] < frz:
            diag["deaths_pre_live"] += 1
            continue
        if d["tick"] > end:
            diag["deaths_post_end"] += 1
            continue
        side = d["team"]
        if side is None:
            diag["in_window_null_team"] += 1
            side = players.get(d["pid"])
            if side is not None:
                diag["in_window_null_team_imputed"] += 1
        if side not in (T, CT):
            unresolved += 1
            continue
        win.append({**d, "side": side})
    if unresolved:
        return None
    diag["deaths_in_window"] += len(win)
    in_win = [t for t in plants if frz <= t <= end]
    plant_tick = min(in_win) if in_win else None
    alive = {T: START, CT: START}
    states = [{"tick": frz, "alive_t": START, "alive_ct": START, "kind": "live_start"}]
    seen = set()
    for d in win:
        if d["pid"] in seen:
            diag["in_window_duplicate_victim_skipped"] += 1
            continue
        seen.add(d["pid"])
        alive[d["side"]] -= 1
        states.append({"tick": d["tick"], "alive_t": alive[T], "alive_ct": alive[CT],
                       "kind": "death"})
    out = []
    for s in states:
        planted = int(plant_tick is not None and plant_tick <= s["tick"])
        out.append({
            "match_key": key[0], "round": key[1], "tick": s["tick"], "kind": s["kind"],
            "alive_ct": s["alive_ct"], "alive_t": s["alive_t"],
            "ct_alive_minus_t": s["alive_ct"] - s["alive_t"],
            "bomb_planted": planted,
            "seconds_since_round_live": (s["tick"] - frz) / TICK_RATE,
            "seconds_since_plant": ((s["tick"] - plant_tick) / TICK_RATE) if planted else -1.0,
            "round_number": key[1],
            "ct_score_minus_t_score": rr["ct_score"] - rr["t_score"],
            "is_pistol_round": int(key[1] in (1, max_rounds // 2 + 1)),
            "is_overtime": int(key[1] > max_rounds),
            "label_ct_win": int(rr["winner"] == CT),
        })
    return out


def defect_check(cfg, map_of, tag):
    """2024 round-13 side mismatch, same definition as scripts/swarm/e2_side_check_r13.py:
    rows with player_count == 20 and a bomb_defused / bomb_exploded event. Full tome, all matches.
    Mismatch = bomb outcome side (defused -> CT 3, exploded -> T 2) != winner_team_code."""
    RE = read(cfg, "round_end", ["match_key", "round", "winner_team_code", "player_count"])
    BS = read(cfg, "bomb_state", ["match_key", "round", "event_type"])
    outcome = (BS.filter(pl.col("event_type").is_in(["bomb_defused", "bomb_exploded"]))
               .group_by(["match_key", "round"]).agg(pl.col("event_type").unique().alias("evs")))
    mapdf = pl.DataFrame({"match_key": list(map_of.keys()),
                          "map": [map_of[m] for m in map_of.keys()]})
    j = (RE.filter(pl.col("player_count") == 20)
         .join(outcome, on=["match_key", "round"], how="inner")
         .with_columns(
             pl.when(pl.col("evs").list.contains("bomb_defused")).then(pl.lit(3))
             .otherwise(pl.lit(2)).alias("bomb_side"))
         .with_columns((pl.col("bomb_side") != pl.col("winner_team_code")).alias("mismatch"))
         .join(mapdf, on="match_key", how="left")
         .with_columns(
             pl.when(pl.col("round") == 13).then(pl.lit("r13"))
             .when(pl.col("round") == 25).then(pl.lit("r25"))
             .otherwise(pl.lit("other")).alias("rbin")))
    total_rows = j.height
    total_mm = int(j["mismatch"].sum())
    print(f"[{tag}] round_end rows with 20 players and a bomb outcome: {total_rows}; mismatches: "
          f"{total_mm}")
    for rb in ["r13", "r25", "other"]:
        sub = j.filter(pl.col("rbin") == rb)
        print(f"[{tag}]   {rb}: n={sub.height} mismatches={int(sub['mismatch'].sum())}")
    mm_side = {(x["match_key"], int(x["round"])): int(x["bomb_side"])
               for x in j.filter(pl.col("mismatch") == True).iter_rows(named=True)}  # noqa: E712
    mm_set = set(mm_side)
    agg = (j.group_by(["map", "rbin"])
           .agg(pl.len().alias("n"), pl.col("mismatch").sum().alias("mm"))
           .sort(["map", "rbin"]))
    per = collections.defaultdict(lambda: collections.Counter())
    for x in agg.iter_rows(named=True):
        per[x["map"]][x["rbin"] + "_n"] += x["n"]
        per[x["map"]][x["rbin"] + "_mm"] += int(x["mm"] or 0)
    return mm_set, per, {"rows": total_rows, "mismatches": total_mm}, mm_side


def residual_test(rows, n_perm, rng):
    """rows: (match_key, y, stratum, map_name). Expected y per stratum is the pooled rate over
    rows. Residuals are summed per match (cluster). Global statistic: sum over maps of
    (sum of match residual sums)^2 / (sum of squared match residual sums). Null: map labels are
    exchangeable across matches (permutation)."""
    if not rows:
        return None
    ys = np.array([r[1] for r in rows], dtype=float)
    strat_ids = {}
    sid = np.array([strat_ids.setdefault(r[2], len(strat_ids)) for r in rows])
    n_s = np.bincount(sid, minlength=len(strat_ids)).astype(float)
    sum_s = np.bincount(sid, weights=ys, minlength=len(strat_ids))
    p = (sum_s / n_s)[sid]
    e = ys - p
    matches = sorted({r[0] for r in rows})
    m_idx = {m: i for i, m in enumerate(matches)}
    mi = np.array([m_idx[r[0]] for r in rows])
    M = np.bincount(mi, weights=e, minlength=len(matches))
    map_names = sorted({str(r[3]) for r in rows})
    map_idx = {mn: i for i, mn in enumerate(map_names)}
    n_map = len(map_names)
    match_map = np.zeros(len(matches), dtype=int)
    for r in rows:
        match_map[m_idx[r[0]]] = map_idx[str(r[3])]
    ri_map = match_map[mi]
    O = np.bincount(ri_map, weights=ys, minlength=n_map)
    E = np.bincount(ri_map, weights=p, minlength=n_map)
    NR = np.bincount(ri_map, minlength=n_map)
    NM = np.bincount(match_map, minlength=n_map)
    D = np.bincount(match_map, weights=M, minlength=n_map)
    Vc = np.bincount(match_map, weights=M * M, minlength=n_map)

    def chi(mm):
        d = np.bincount(mm, weights=M, minlength=n_map)
        vc = np.bincount(mm, weights=M * M, minlength=n_map)
        ok = vc > 0
        return float((d[ok] ** 2 / vc[ok]).sum())

    obs = chi(match_map)
    ge = 0
    for _ in range(n_perm):
        if chi(rng.permutation(match_map)) >= obs:
            ge += 1
    p_perm = (ge + 1) / (n_perm + 1)
    return {"rows": len(rows), "matches": len(matches), "maps": map_names, "O": O, "E": E,
            "NR": NR, "NM": NM, "D": D, "Vc": Vc, "chi2": obs, "p_perm": p_perm,
            "n_strata": len(strat_ids)}


K_BASE = 5


def map_holdout(rows, k_cell=50, k_off=200, folds=5, seed=SEED):
    """Out-of-sample proxies for a map feature, match-level 5-fold CV. Base: rate per stratum (the
    model's own features as strata). Cell model: rate per (stratum, map), shrunk toward base with
    k_cell pseudo-rows. Offset model: base plus one additive shift per map, estimated on the
    training folds and shrunk by k_off rows (ASSUMPTIONS k_cell=50, k_off=200). Gain = LL(base) -
    LL(model), summed per match; z uses per-match sums. Not an XGBoost retrain."""
    if not rows:
        return None
    matches = sorted({r[0] for r in rows})
    perm = np.random.default_rng(seed).permutation(len(matches))
    fold_of = {matches[i]: int(j % folds) for j, i in enumerate(perm)}
    eps = 1e-6
    gain_cell = collections.defaultdict(float)
    gain_off = collections.defaultdict(float)
    ll0_tot = ll_cell_tot = ll_off_tot = 0.0
    n_tot = 0
    for f in range(folds):
        tr = [r for r in rows if fold_of[r[0]] != f]
        te = [r for r in rows if fold_of[r[0]] == f]
        s_n, s_y = collections.Counter(), collections.Counter()
        sm_n, sm_y = collections.Counter(), collections.Counter()
        for r in tr:
            s_n[r[2]] += 1
            s_y[r[2]] += r[1]
            sm_n[(r[2], r[3])] += 1
            sm_y[(r[2], r[3])] += r[1]
        gmean = sum(s_y.values()) / max(1, sum(s_n.values()))

        def base_of(strat):
            # smoothed toward the global rate (K_BASE pseudo-rows, ASSUMPTION) so that no stratum
            # gets a probability of exactly 0 or 1 from a few training rows
            n_s = s_n.get(strat, 0)
            return (s_y[strat] + K_BASE * gmean) / (n_s + K_BASE)

        resid_m = collections.Counter()
        n_m = collections.Counter()
        for r in tr:
            resid_m[r[3]] += r[1] - base_of(r[2])
            n_m[r[3]] += 1
        delta = {mp: resid_m[mp] / (n_m[mp] + k_off) for mp in n_m}
        for r in te:
            base = base_of(r[2])
            key = (r[2], r[3])
            n_sm = sm_n.get(key, 0)
            cell = (sm_y[key] + k_cell * base) / (n_sm + k_cell) if n_sm else base
            off = base + delta.get(r[3], 0.0)
            y = r[1]
            b = min(max(base, eps), 1 - eps)
            c = min(max(cell, eps), 1 - eps)
            o = min(max(off, eps), 1 - eps)
            l0 = -(y * np.log(b) + (1 - y) * np.log(1 - b))
            lc = -(y * np.log(c) + (1 - y) * np.log(1 - c))
            lo = -(y * np.log(o) + (1 - y) * np.log(1 - o))
            gain_cell[r[0]] += float(l0 - lc)
            gain_off[r[0]] += float(l0 - lo)
            ll0_tot += float(l0)
            ll_cell_tot += float(lc)
            ll_off_tot += float(lo)
            n_tot += 1

    def summ(gd, ll_tot):
        d = np.array(list(gd.values()))
        gain = float(d.sum())
        se = float(np.sqrt((d ** 2).sum()))
        return {"ll": ll_tot / n_tot, "gain_per_row": gain / n_tot,
                "z": gain / se if se > 0 else float("nan")}

    return {"n": n_tot, "ll0": ll0_tot / n_tot,
            "cell": summ(gain_cell, ll_cell_tot), "offset": summ(gain_off, ll_off_tot)}


def run(tag, cfg):
    print(f"\n{'#' * 10} {tag} tome {'#' * 10}")
    print(f"dir: {cfg['dir']}\nsuffix: {cfg['suffix']}")
    print(f"required dense channels: {cfg['dense']}\nsparse channels: {cfg['sparse']}")
    hdr_cols = list(pl.read_parquet_schema(files_of(cfg["dir"] / "header")[0]).keys())
    has_mr = "max_rounds" in hdr_cols
    want = ["key", "match_id", "match_date", "map_name", "tick_rate"] + (["max_rounds"] if has_mr else [])
    hdr = read(cfg, "header", want)
    H = {}
    n_default_mr = 0
    for r in hdr.iter_rows(named=True):
        mr = r["max_rounds"] if (has_mr and r["max_rounds"] is not None) else None
        if mr is None:
            mr = cfg["default_max_rounds"]
            n_default_mr += 1
        H[r["key"]] = {"map": r["map_name"], "tick_rate": r["tick_rate"], "max_rounds": mr}
    all_matches = set(H)
    print(f"header rows: {hdr.height}; unique keys: {len(all_matches)}")
    print(f"tick_rate counts: {dict(collections.Counter(v['tick_rate'] for v in H.values()))}")
    assert set(v["tick_rate"] for v in H.values()) == {TICK_RATE}, "unexpected tick rate"
    if has_mr:
        print(f"max_rounds from header; counts: {dict(collections.Counter(v['max_rounds'] for v in H.values()))}")
    else:
        print(f"max_rounds: not in header; ASSUMPTION default {cfg['default_max_rounds']} for all "
              f"{n_default_mr} matches (affects flags only)")
    raw = collections.Counter(v["map"] for v in H.values())
    print("map_name exact header values (count):")
    for name, n in sorted(raw.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"    {name!r}: {n}")
    map_of = {m: H[m]["map"] for m in all_matches}

    # ---- 1. match-level channel check (bakeoff section 1)
    print("\n== 1. Match-level channel check ==")
    missing = collections.defaultdict(list)
    for ch in cfg["dense"] + cfg["sparse"]:
        dense = ch in cfg["dense"]
        mk = read(cfg, ch, ["match_key"])
        have = set(mk["match_key"].unique().to_list())
        ks = keyset_keys(cfg, ch)
        no_ks = all_matches - ks
        zero = all_matches - have
        miss = no_ks | (zero if dense else set())
        for m in miss:
            missing[m].append(ch)
        print(f"  {ch:<20} {'dense ' if dense else 'sparse'} rows={mk.height:>8} "
              f"matches_with_rows={len(have):>5} keyset={len(ks):>5} zero_rows={len(zero):>5} "
              f"not_in_keyset={len(no_ks):>5} counted_missing={len(miss):>5}")
        del mk
    dropped = set(missing)
    kept_matches = all_matches - dropped
    combos = collections.Counter(tuple(sorted(v)) for v in missing.values())
    print(f"matches dropped for a missing channel: {len(dropped)}")
    for combo, n in combos.most_common():
        print(f"    missing {list(combo)}: {n}")
    print(f"channel-kept matches: {len(kept_matches)}")

    kept_list = sorted(kept_matches)

    def kf():
        return pl.col("match_key").is_in(kept_list)

    RE = read(cfg, "round_end", ["match_key", "round", "tick", "winner_team_code", "player_count"]).filter(kf())
    RST = read(cfg, "round_start", ["match_key", "round"]).filter(kf())
    RS = read(cfg, "round_state", ["match_key", "round", "tick", "event_type", "t_score", "ct_score"]).filter(kf())
    DE = read(cfg, "player_death", ["match_key", "round", "tick", "player_id", "player_team_code"]).filter(kf())
    PI = read(cfg, "player_info", ["match_key", "round", "player_id", "team_code"]).filter(kf())
    BS = read(cfg, "bomb_state", ["match_key", "round", "tick", "event_type"]).filter(kf())
    DC = read(cfg, "player_disconnect", ["match_key", "round", "tick"]).filter(kf())
    BT = read(cfg, "bot_takeover", ["match_key", "round", "tick"]).filter(kf())

    # ---- duplicate matches (bakeoff: identical round_end, player_death, player_info signatures)
    sig_parts = collections.defaultdict(list)
    for x in RE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("R", int(x["round"]), int(x["tick"]),
                                               str(x["winner_team_code"]))))
    for x in DE.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("D", int(x["round"]), int(x["tick"]),
                                               str(x["player_id"]))))
    for x in PI.iter_rows(named=True):
        sig_parts[x["match_key"]].append(repr(("P", int(x["round"]), str(x["player_id"]),
                                               str(x["team_code"]))))
    sigs = {m: tuple(sorted(v)) for m, v in sig_parts.items()}
    sig_groups = collections.defaultdict(list)
    for m in kept_list:
        sig_groups[sigs.get(m, ())].append(m)
    dedup_groups = [g for g in sig_groups.values() if len(g) > 1]
    dup_drop = set()
    for g in dedup_groups:
        dup_drop.update(g[1:])
    dup_by_map = collections.Counter(map_of[m] for m in dup_drop)
    print(f"\nduplicate groups among channel-kept matches: {len(dedup_groups)} groups; "
          f"matches flagged as duplicate copies: {len(dup_drop)}")

    # ---- 2. round filters (bakeoff section 2)
    print("\n== 2. Round filters (bakeoff order) ==")
    rd = {}
    for x in RE.iter_rows(named=True):
        rd[(x["match_key"], int(x["round"]))] = {"end": int(x["tick"]),
                                                 "winner": x["winner_team_code"],
                                                 "pc": x["player_count"]}
    rst_keys = {(x["match_key"], int(x["round"])) for x in RST.iter_rows(named=True)}
    live = {}
    live_dups = 0
    for x in RS.filter(pl.col("event_type") == "round_freeze_end").iter_rows(named=True):
        k = (x["match_key"], int(x["round"]))
        if k in live:
            live_dups += 1
            continue
        live[k] = {"tick": int(x["tick"]), "t": x["t_score"], "ct": x["ct_score"]}
    players = collections.defaultdict(dict)
    pi_dups = 0
    for x in PI.iter_rows(named=True):
        k = (x["match_key"], int(x["round"]))
        if x["player_id"] in players[k]:
            pi_dups += 1
        players[k][x["player_id"]] = x["team_code"]
    disc_rounds = set()
    for df in (DC, BT):
        for x in df.iter_rows(named=True):
            disc_rounds.add((x["match_key"], int(x["round"])))
    plants = collections.defaultdict(list)
    for x in BS.filter(pl.col("event_type") == "bomb_planted").iter_rows(named=True):
        plants[(x["match_key"], int(x["round"]))].append(int(x["tick"]))
    deaths = collections.defaultdict(list)
    for x in DE.sort("tick").iter_rows(named=True):
        deaths[(x["match_key"], int(x["round"]))].append(
            {"tick": int(x["tick"]), "pid": x["player_id"], "team": x["player_team_code"]})
    print(f"channel-kept rounds (round_end rows): {len(rd)}; live-start rows: {len(live)} "
          f"(duplicate freeze_end skipped {live_dups}); player_info duplicate rows {pi_dups}")

    reasons = collections.Counter()
    survivors = []
    pc_only_keys = []
    for k in sorted(rd):
        rr = rd[k]
        pl_ = players.get(k, {})
        n_t = sum(1 for v in pl_.values() if v == T)
        n_ct = sum(1 for v in pl_.values() if v == CT)
        if rr["pc"] != cfg["pc_full"]:
            reasons[f"round_end.player_count != {cfg['pc_full']}"] += 1
            if len(pl_) == 10 and n_t == START and n_ct == START:
                pc_only_keys.append(k)
            continue
        if not (len(pl_) == 10 and n_t == START and n_ct == START):
            reasons["not exactly 5 T and 5 CT at start (player_info)"] += 1
            continue
        if k in disc_rounds:
            reasons["disconnect or bot takeover event in round"] += 1
            continue
        if k not in rst_keys:
            reasons["no round_start row"] += 1
            continue
        if k not in live:
            reasons["no round_freeze_end (live start) row"] += 1
            continue
        if live[k]["t"] is None or live[k]["ct"] is None:
            reasons["null score at live start"] += 1
            continue
        survivors.append(k)
    dup_rounds = sum(1 for k in survivors if k[0] in dup_drop)
    reasons["duplicate copy of a kept match"] += dup_rounds
    survivors = [k for k in survivors if k[0] not in dup_drop]

    diag = collections.Counter()
    rounds_ok = []
    state_rows = []
    for k in survivors:
        rr = {**rd[k], "freeze": live[k]["tick"], "t_score": live[k]["t"], "ct_score": live[k]["ct"]}
        local = collections.Counter()
        out = build_states(k, rr, deaths.get(k, []), plants.get(k, []), players.get(k, {}),
                           H[k[0]]["max_rounds"], local)
        if out is None:
            reasons["death with unresolved victim side in live window"] += 1
            continue
        diag.update(local)
        state_rows.extend(out)
        rounds_ok.append(k)
    final_matches = {k[0] for k in rounds_ok}
    print(f"rounds in channel-kept matches before round filters: {len(rd)}")
    for name, n in reasons.most_common():
        print(f"  dropped: {name}: {n}")
    print(f"rounds kept: {len(rounds_ok)} in {len(final_matches)} matches")
    print(f"states: {len(state_rows)} (live start "
          f"{sum(1 for s in state_rows if s['kind'] == 'live_start')}, death "
          f"{sum(1 for s in state_rows if s['kind'] == 'death')})")
    print(f"diagnostics: {dict(diag)}")
    null_winner = sum(1 for k in rounds_ok if rd[k]["winner"] is None)
    print(f"kept rounds with null winner_team_code (counted as T win): {null_winner}")
    print(f"kept rounds in dup-removed matches: 0 (duplicates removed before build_states)")

    if tag == "2026":
        got = {"matches": len(final_matches), "rounds": len(rounds_ok), "states": len(state_rows)}
        ok = got == REFERENCE_2026
        print(f"\nREPRODUCTION CHECK vs bake-off kept set {REFERENCE_2026}: got {got} -> "
              f"{'PASS' if ok else 'FAIL'}")

    # ---- per-map counts
    print("\n== 3. Per-map counts in the kept set ==")
    hdr_m = collections.Counter(map_of.values())
    chan_m = collections.Counter(map_of[m] for m in kept_matches)
    final_m = collections.Counter(map_of[m] for m in final_matches)
    rounds_m = collections.Counter(map_of[k[0]] for k in rounds_ok)
    states_m = collections.Counter(map_of[s["match_key"]] for s in state_rows)
    ct_m = collections.Counter(map_of[k[0]] for k in rounds_ok if rd[k]["winner"] == CT)
    pc_m = collections.Counter(map_of[k[0]] for k in pc_only_keys)
    rows = []
    for mp in sorted(hdr_m, key=lambda x: (-rounds_m.get(x, 0), str(x))):
        r_n = rounds_m.get(mp, 0)
        m_n = final_m.get(mp, 0)
        flags = []
        if r_n < MIN_ROUNDS:
            flags.append(f"rounds<{MIN_ROUNDS}")
        if m_n < MIN_MATCHES:
            flags.append(f"matches<{MIN_MATCHES}")
        ctp = f"{100 * ct_m.get(mp, 0) / r_n:.1f}" if r_n else "-"
        spr = f"{states_m.get(mp, 0) / r_n:.2f}" if r_n else "-"
        rows.append([mp, hdr_m[mp], chan_m.get(mp, 0), dup_by_map.get(mp, 0), m_n, r_n,
                     states_m.get(mp, 0), spr, ctp, pc_m.get(mp, 0), ", ".join(flags) or "-"])
    tot = ["ALL", sum(hdr_m.values()), sum(chan_m.values()), sum(dup_by_map.values()),
           len(final_matches), len(rounds_ok), len(state_rows),
           f"{len(state_rows) / max(1, len(rounds_ok)):.2f}",
           f"{100 * sum(ct_m.values()) / max(1, len(rounds_ok)):.1f}", sum(pc_m.values()), ""]
    rows.append(tot)
    print(md_table(["map_name", "header matches", "channel-kept matches", "dup copies removed",
                    "kept matches", "kept rounds", "states", "states/round", "CT round win %",
                    "5v5 rounds dropped by player_count rule", "flag"], rows))

    # ---- 4. map residual test (beyond score lead, alive counts, plant, round time)
    print("\n== 4. Does map_name predict CT round win beyond the model's features? ==")
    rng = np.random.default_rng(SEED)
    mismatch_probe = None  # filled below for 2024 sensitivity
    results = {}
    mm_set, defect_per_map, defect_tot, mm_side = defect_check(cfg, map_of, tag)
    mm_kept = {k for k in rounds_ok if k in mm_set}
    print(f"[{tag}] kept rounds with a bomb-outcome/winner mismatch: {len(mm_kept)} "
          f"(of which bomb outcome is CT: {sum(1 for k in mm_kept if mm_side[k] == CT)})")
    srows = []
    for mp in sorted(hdr_m, key=lambda x: (-rounds_m.get(x, 0), str(x))):
        ks = [k for k in rounds_ok if map_of[k[0]] == mp]
        if not ks:
            continue
        n = len(ks)
        mk = [k for k in ks if k in mm_side]
        raw = sum(1 for k in ks if rd[k]["winner"] == CT)
        excl_ks = [k for k in ks if k not in mm_side]
        excl = sum(1 for k in excl_ks if rd[k]["winner"] == CT)
        corr = sum(1 for k in ks if (mm_side[k] == CT if k in mm_side else rd[k]["winner"] == CT))
        srows.append([mp, n, len(mk), sum(1 for k in mk if mm_side[k] == CT),
                      f"{100 * raw / n:.1f}",
                      f"{100 * excl / max(1, len(excl_ks)):.1f}",
                      f"{100 * corr / n:.1f}"])
    print(f"\n[{tag}] label sensitivity by map (kept rounds): raw CT win vs excluding the mismatched "
          f"rounds vs labels replaced by the bomb outcome")
    print(md_table(["map_name", "kept rounds", "mismatched kept rounds",
                    "of which bomb says CT", "CT % raw", "CT % excl mismatch",
                    "CT % bomb-outcome labels"], srows))

    def rows_for(kind, exclude):
        out_rows = []
        for s in state_rows:
            if kind == "live" and s["kind"] != "live_start":
                continue
            key = (s["match_key"], s["round"])
            if exclude and key in mm_set:
                continue
            lead = clip(s["ct_score_minus_t_score"], -6, 6)
            if kind == "live":
                strat = (lead, s["is_pistol_round"], s["is_overtime"])
            else:
                strat = (s["alive_ct"], s["alive_t"], s["bomb_planted"], lead,
                         s["is_pistol_round"], s["is_overtime"],
                         int(clip(int(s["seconds_since_round_live"] // 15), 0, 8)))
            out_rows.append((s["match_key"], s["label_ct_win"], strat, map_of[s["match_key"]]))
        return out_rows

    print(f"[{tag}] sensitivity: rounds failing only the player_count rule but with exactly 5 T "
          f"and 5 CT on 10 player_info slots: {len(pc_only_keys)} (not kept by the bake-off rule)")
    for kind in ["live", "state"]:
        for exclude in [False, True]:
            name = f"{kind}{'-excl-mismatch' if exclude else ''}"
            res = residual_test(rows_for(kind, exclude), N_PERM, rng)
            results[name] = res
            if res is None:
                print(f"[{tag}] {name}: no rows")
                continue
            print(f"[{tag}] {name}: rows={res['rows']} matches={res['matches']} "
                  f"strata={res['n_strata']} chi2={res['chi2']:.2f} "
                  f"perm p={res['p_perm']:.4f} (N_PERM={N_PERM}, maps={len(res['maps'])})")

    for kind in ["live", "state"]:
        ho = map_holdout(rows_for(kind, False))
        if ho is None:
            continue
        for mname in ["cell", "offset"]:
            hm = ho[mname]
            print(f"[{tag}] {kind}: 5-fold match CV, base LL={ho['ll0']:.5f}; +{mname} map LL="
                  f"{hm['ll']:.5f}; gain per row={hm['gain_per_row'] * 1000:+.3f} x1e-3 "
                  f"(cluster z={hm['z']:+.2f}, rows={ho['n']})")

    if results["live"] is None or results["state"] is None:
        print(f"[{tag}] no kept rows: residual tables skipped")
        summary = {"rows": rows, "rounds_ok": len(rounds_ok), "matches": len(final_matches),
                   "states": len(state_rows), "res": results}
        del RE, RST, RS, DE, PI, BS, DC, BT, rd, live, players, deaths, plants, state_rows
        return summary

    res_live = results["live"]
    rowsA = []
    for i, mn in enumerate(res_live["maps"]):
        nr = int(res_live["NR"][i])
        z = res_live["D"][i] / np.sqrt(res_live["Vc"][i]) if res_live["Vc"][i] > 0 else float("nan")
        rowsA.append([mn, int(res_live["NM"][i]), nr,
                      f"{100 * res_live['O'][i] / nr:.1f}", f"{100 * res_live['E'][i] / nr:.1f}",
                      f"{100 * res_live['D'][i] / nr:+.2f}", f"{z:+.2f}"])
    print("\nRound-level (live start, one row per round). Strata: score lead (clipped +-6), pistol, OT.")
    print(md_table(["map_name", "matches", "rounds", "CT win % raw", "CT win % expected",
                    "CT win pp vs expected", "cluster z"], rowsA))
    res_state = results["state"]
    rowsB = []
    for i, mn in enumerate(res_state["maps"]):
        ns = int(res_state["NR"][i])
        z = res_state["D"][i] / np.sqrt(res_state["Vc"][i]) if res_state["Vc"][i] > 0 else float("nan")
        rowsB.append([mn, int(res_state["NM"][i]), ns, f"{100 * res_state['D'][i] / ns:+.3f}",
                      f"{z:+.2f}"])
    print("\nState-level (all states). Strata: alive CT/T, planted, score lead, pistol, OT, 15 s bin.")
    print(md_table(["map_name", "matches", "states", "mean residual (pp per state)", "cluster z"], rowsB))

    # ---- 5. defect cross-check by map
    print("\n== 5. Bomb-outcome vs winner mismatch by map (full tome, e2 definition) ==")
    drows = []
    for mp in sorted(hdr_m, key=lambda x: -(defect_per_map[x]["r13_n"] + defect_per_map[x]["other_n"])):
        c = defect_per_map.get(mp, collections.Counter())
        drows.append([mp, c["r13_n"], c["r13_mm"],
                      f"{c['r13_mm'] / c['r13_n']:.2f}" if c["r13_n"] else "-",
                      c["r25_n"], c["r25_mm"], c["other_n"], c["other_mm"]])
    print(md_table(["map_name", "r13 rows", "r13 mismatch", "r13 rate", "r25 rows", "r25 mismatch",
                    "other rows", "other mismatch"], drows))
    try:
        from scipy.stats import chi2_contingency
        tab = [[defect_per_map[mp]["r13_mm"], defect_per_map[mp]["r13_n"] - defect_per_map[mp]["r13_mm"]]
               for mp in defect_per_map if defect_per_map[mp]["r13_n"] > 0]
        if len(tab) > 1:
            stat, pv, dof, _ = chi2_contingency(np.array(tab))
            print(f"[{tag}] round-13 mismatch rate homogeneity across maps: chi2={stat:.2f} "
                  f"df={dof} p={pv:.4f}")
    except Exception as ex:  # scipy optional
        print(f"[{tag}] chi2 homogeneity skipped: {ex}")

    summary = {"rows": rows, "rounds_ok": len(rounds_ok), "matches": len(final_matches),
               "states": len(state_rows), "res": results}
    del RE, RST, RS, DE, PI, BS, DC, BT, rd, live, players, deaths, plants, state_rows
    return summary


def main():
    summaries = {}
    for tag, cfg in CONFIGS.items():
        summaries[tag] = run(tag, cfg)
    print("\n== Done ==")
    for tag in CONFIGS:
        s = summaries[tag]
        print(f"{tag}: kept matches {s['matches']}, kept rounds {s['rounds_ok']}, states {s['states']}")


if __name__ == "__main__":
    main()
