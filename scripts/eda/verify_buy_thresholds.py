"""THRESHOLDS lens: independent check of the buy-type pilot's cut-offs (read-only on inputs).

Re-derives freeze-end side equipment from players_1hz.parquet without importing
buy_type_pilot.py. Checks: 1000-dollar histogram and valleys at the cuts; class counts with
flags; side-level win-rate ordering under each threshold set and over a sensitivity grid;
the anti_eco pairing; and a row-level cross-check against round_buy_types.parquet.
ct_win comes from rounds.parquet. ct_score and t_score are never read.
Writes only logs/buy_types/verify_thresholds.txt.

Run from the repo root:
  pixi run python -I -X utf8 scripts\\eda\\verify_buy_thresholds.py
"""

import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(r"G:\pureskill\csds-cs2")
STATE = ROOT / "data" / "state_sample"
PLAYERS = STATE / "players_1hz.parquet"
ROUNDS = STATE / "rounds.parquet"
BUY_OUT = STATE / "round_buy_types.parquet"
LOG = ROOT / "logs" / "buy_types" / "verify_thresholds.txt"

T_CODE, CT_CODE = 2, 3
PISTOL = (1, 13)
OT_FIRST = 25
BOOT_B = 1000
BOOT_SEED = 0
MIN_ROUNDS = 30
MIN_MATCHES = 5

# (name, eco_max, full_min). Side class: eco if eq < eco_max; force if eco_max <= eq < full_min; full otherwise.
SETS = [
    ("conservative", 5000, 25000),
    ("default", 10000, 20000),
    ("aggressive", 10000, 20000),
    ("paper_2109_team", 3000, 20000),  # supplementary: eco cut from the CS:GO Table 1 note
]
GRID_ECO = [3000, 5000, 7000, 10000, 12000, 15000]
GRID_FULL = [15000, 18000, 20000, 22000, 25000]

CLASSES = ["pistol", "eco_vs_eco", "anti_eco", "force_vs_eco", "full_vs_force", "force_vs_force", "full_vs_full"]
PAIR = {
    ("eco", "eco"): "eco_vs_eco",
    ("eco", "force"): "force_vs_eco",
    ("eco", "full"): "anti_eco",
    ("force", "force"): "force_vs_force",
    ("force", "full"): "full_vs_force",
    ("full", "full"): "full_vs_full",
}


def bucket(eq, eco_max, full_min):
    if eq < eco_max:
        return "eco"
    if eq < full_min:
        return "force"
    return "full"


def pair_class(a, b):
    return PAIR[tuple(sorted((a, b)))]


def boot_ci(wins, ns):
    wins = np.asarray(wins, dtype=float)
    ns = np.asarray(ns, dtype=float)
    m = ns.size
    if m == 0 or ns.sum() == 0:
        return None, None
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, m, size=(BOOT_B, m))
    ws = wins[idx].sum(axis=1)
    nn = ns[idx].sum(axis=1)
    keep = nn > 0
    rates = ws[keep] / nn[keep]
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(lo), float(hi)


def pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}"


def ci(lo, hi):
    return "n/a" if lo is None else f"[{100 * lo:.1f}, {100 * hi:.1f}]"


def side_records(rows):
    recs = []
    for r in rows:
        ot = r["round"] >= OT_FIRST
        for side, n, eq, money in (
            ("CT", r["ct_n"], r["ct_eq"], r["ct_money"]),
            ("T", r["t_n"], r["t_eq"], r["t_money"]),
        ):
            won = (r["ct_win"] == 1) if side == "CT" else (r["ct_win"] == 0)
            recs.append({"match_key": r["match_key"], "round": r["round"], "side": side, "n": n,
                         "eq": eq, "money": money, "won": won, "ot": ot})
    return recs


def block(recs):
    by_m = defaultdict(lambda: [0, 0])
    for x in recs:
        by_m[x["match_key"]][0] += int(x["won"])
        by_m[x["match_key"]][1] += 1
    ms = sorted(by_m)
    wins = [by_m[m][0] for m in ms]
    ns = [by_m[m][1] for m in ms]
    n = sum(ns)
    w = sum(wins)
    lo, hi = boot_ci(wins, ns) if n else (None, None)
    return {"n": n, "w": w, "rate": (w / n) if n else None, "lo": lo, "hi": hi, "matches": len(ms)}


def order_verdict(stats):
    r = {k: stats[k]["rate"] for k in ("eco", "force", "full")}
    if any(v is None for v in r.values()):
        return "not evaluable (an empty class)"
    ok1 = r["eco"] < r["force"]
    ok2 = r["force"] < r["full"]
    if ok1 and ok2:
        return "HOLDS eco < force < full"
    broke = []
    if not ok1:
        broke.append("eco >= force")
    if not ok2:
        broke.append("force >= full")
    return "FAILS: " + "; ".join(broke)


def main():
    out = []

    def emit(s=""):
        print(s)
        out.append(s)

    # ---- load --------------------------------------------------------------------
    players = pl.read_parquet(
        PLAYERS,
        columns=["match_key", "round", "tick", "team_code", "equipment_value_calc", "money",
                 "has_defuser", "inv_c4"],
    ).with_columns(
        pl.col("round").cast(pl.Int64),
        pl.col("team_code").cast(pl.Int64),
        pl.col("equipment_value_calc").cast(pl.Int64),
        pl.col("money").cast(pl.Int64),
    )
    rd = pl.read_parquet(ROUNDS, columns=["match_key", "round", "ct_win", "split", "live_start_tick"]).with_columns(
        pl.col("round").cast(pl.Int64)
    )

    # ---- 0. Re-derive freeze-end snapshot and sides --------------------------------
    emit("== 0. Freeze-end snapshot and side sums (re-derived) ==")
    first = players.group_by(["match_key", "round"]).agg(pl.col("tick").min().alias("ft"))
    miss_r = first.select(["match_key", "round"]).join(rd.select(["match_key", "round"]), on=["match_key", "round"], how="anti").height
    miss_p = rd.select(["match_key", "round"]).join(first.select(["match_key", "round"]), on=["match_key", "round"], how="anti").height
    emit(f"rounds with players: {first.height} | rounds in rounds.parquet: {rd.height} | unmatched either way: {miss_r + miss_p}")
    chk = first.join(rd.select(["match_key", "round", "live_start_tick"]), on=["match_key", "round"], how="inner")
    n_eq = chk.filter(pl.col("ft") == pl.col("live_start_tick")).height
    n_ge = chk.filter(pl.col("ft") >= pl.col("live_start_tick")).height
    emit(f"first snapshot == live_start_tick: {n_eq} of {chk.height} | first snapshot >= live_start_tick: {n_ge}")
    tcs = sorted(players["team_code"].unique().to_list())
    emit(f"team_code values {tcs}")
    fz = players.join(first, on=["match_key", "round"]).filter(pl.col("tick") == pl.col("ft"))
    side_df = fz.group_by(["match_key", "round", "team_code"]).agg(
        pl.len().alias("n"),
        pl.col("equipment_value_calc").sum().alias("eq"),
        pl.col("money").sum().alias("money"),
    )
    sides = defaultdict(dict)
    for r in side_df.iter_rows(named=True):
        sides[(r["match_key"], r["round"])][r["team_code"]] = (r["n"], r["eq"], r["money"])
    rd_map = {(r["match_key"], r["round"]): r for r in rd.iter_rows(named=True)}
    assert set(rd_map) == set(sides), "key sets differ"
    rounds = []
    for key, rr in rd_map.items():
        s = sides[key]
        ct = s.get(CT_CODE, (0, 0, 0))
        t = s.get(T_CODE, (0, 0, 0))
        rounds.append({
            "match_key": key[0], "round": key[1], "ct_win": int(rr["ct_win"]), "split": rr["split"],
            "ct_n": ct[0], "ct_eq": ct[1], "ct_money": ct[2],
            "t_n": t[0], "t_eq": t[1], "t_money": t[2],
        })
    rounds.sort(key=lambda r: (r["match_key"], r["round"]))
    kept = [r for r in rounds if r["ct_n"] == 5 and r["t_n"] == 5]
    dropped = [r for r in rounds if not (r["ct_n"] == 5 and r["t_n"] == 5)]
    emit(f"rounds {len(rounds)} | kept (both sides 5 players) {len(kept)} | dropped {len(dropped)}")
    for r in dropped:
        emit(f"  dropped {r['match_key']} round {r['round']}: CT n={r['ct_n']} eq={r['ct_eq']:,} | T n={r['t_n']} eq={r['t_eq']:,}")
    nonp = [r for r in kept if r["round"] not in PISTOL]
    pistol_rows = [r for r in kept if r["round"] in PISTOL]
    emit(f"kept pistol rounds {len(pistol_rows)} | kept non-pistol rounds {len(nonp)}")
    rounds_by_key = {(r["match_key"], r["round"]): r for r in kept}

    recs_np = side_records(nonp)
    eqs = [x["eq"] for x in recs_np]

    # ---- 1. Histogram and cuts ----------------------------------------------------
    emit()
    emit("== 1. Histogram of non-pistol side equipment (kept rounds, 1,000-dollar bins) ==")
    emit(f"non-pistol side-rounds {len(recs_np)} (CT {sum(x['side'] == 'CT' for x in recs_np)}, "
         f"T {sum(x['side'] == 'T' for x in recs_np)}); OT (25+) side-rounds {sum(x['ot'] for x in recs_np)}")
    emit(f"min side equipment {min(eqs):,} | max {max(eqs):,} | sides exactly on a 1,000 boundary {sum(1 for e in eqs if e % 1000 == 0)} (bin = floor(eq/1000); a side at exactly 10,000 counts as at-or-above 10,000)")
    hb = Counter(x["eq"] // 1000 for x in recs_np)
    hct = Counter(x["eq"] // 1000 for x in recs_np if x["side"] == "CT")
    ht = Counter(x["eq"] // 1000 for x in recs_np if x["side"] == "T")
    hot = Counter(x["eq"] // 1000 for x in recs_np if x["ot"])
    emit("bin (k$)     total    CT     T     OT   histogram (# = 5 side-rounds)")
    top = max(hb)
    for b in range(0, top + 1):
        emit(f"{b:>2}-{b + 1:<2}      {hb.get(b, 0):>6}{hct.get(b, 0):>6}{ht.get(b, 0):>6}{hot.get(b, 0):>6}   {'#' * (hb.get(b, 0) // 5)}")
    series = [hb.get(b, 0) for b in range(0, top + 1)]
    minima = [b for b in range(1, len(series) - 1) if series[b] < series[b - 1] and series[b] < series[b + 1]]
    emit("strict local minima (bin below both neighbours), bin k$: " + ", ".join(
        f"{b}-{b + 1} ({series[b]}; neighbours {series[b - 1]}, {series[b + 1]})" for b in minima))
    emit()
    emit("== 1b. Side counts on each side of each cut (non-pistol, kept, OT included) ==")
    for c in [3000, 5000, 10000, 20000, 25000]:
        lo_s = [x for x in recs_np if x["eq"] < c]
        hi_s = [x for x in recs_np if x["eq"] >= c]
        lo_ct = sum(x["side"] == "CT" for x in lo_s)
        lo_t = sum(x["side"] == "T" for x in lo_s)
        emit(f"cut {c:,}: below {len(lo_s)} (CT {lo_ct}, T {lo_t}) | at or above {len(hi_s)} (CT {len(recs_np) // 2 - lo_ct if False else sum(x['side'] == 'CT' for x in hi_s)}, T {sum(x['side'] == 'T' for x in hi_s)})")
        LL = sum(1 for x in recs_np if c - 2000 <= x["eq"] < c - 1000)
        L = sum(1 for x in recs_np if c - 1000 <= x["eq"] < c)
        U = sum(1 for x in recs_np if c <= x["eq"] < c + 1000)
        UU = sum(1 for x in recs_np if c + 1000 <= x["eq"] < c + 2000)
        valley = (L < LL) and (U < UU)
        emit(f"   bins [{c - 2000:,},{c - 1000:,}) {LL} | [{c - 1000:,},{c:,}) {L} | [{c:,},{c + 1000:,}) {U} | [{c + 1000:,},{c + 2000:,}) {UU}")
        emit(f"   moving the cut down 1,000 reclassifies {L} sides; up 1,000 reclassifies {U} sides; "
             f"both bins beside the cut lower than their outer neighbours (valley): {'yes' if valley else 'no'}")

    # ---- 2. Class counts per set ------------------------------------------------------
    emit()
    emit("== 2. Class counts by threshold set (kept rounds; flags: <30 rounds, <5 matches, <30 matches) ==")
    per_set = {}
    for name, eco_max, full_min in SETS:
        assert eco_max < full_min
        rows = []
        for r in kept:
            cb = bucket(r["ct_eq"], eco_max, full_min)
            tb = bucket(r["t_eq"], eco_max, full_min)
            cls = "pistol" if r["round"] in PISTOL else pair_class(cb, tb)
            rows.append({**r, "ct_buy": cb, "t_buy": tb, "cls": cls})
        per_set[name] = (eco_max, full_min, rows)
        cnt = Counter(x["cls"] for x in rows)
        assert sum(cnt.values()) == len(kept)
        emit(f"{name}: eco_max {eco_max:,}, full_min {full_min:,}")
        for c in CLASSES:
            sub = [x for x in rows if x["cls"] == c]
            m = len({x["match_key"] for x in sub})
            flags = []
            if len(sub) < MIN_ROUNDS:
                flags.append(f"FLAG <{MIN_ROUNDS} rounds")
            if m < MIN_MATCHES:
                flags.append(f"FLAG <{MIN_MATCHES} matches")
            elif m < 30:
                flags.append("note <30 matches")
            ot = sum(1 for x in sub if x["round"] >= OT_FIRST)
            emit(f"   {c:<15} rounds {len(sub):>5}  matches {m:>4}  OT {ot:>3}  {' | '.join(flags)}")

    # ---- 3. Win-rate ordering -----------------------------------------------------------
    emit()
    emit("== 3. Side-level win rate by side class (non-pistol sides; side won = its team won the round) ==")
    emit("   CI: match-cluster bootstrap, 1,000 resamples, seed 0. Ordering tested as eco < force < full.")
    for name, eco_max, full_min in SETS:
        emit(f"{name}: eco_max {eco_max:,}, full_min {full_min:,}")
        subs = {
            "all non-pistol sides": recs_np,
            "excluding OT (rounds 25+)": [x for x in recs_np if not x["ot"]],
            "CT sides only": [x for x in recs_np if x["side"] == "CT"],
            "T sides only": [x for x in recs_np if x["side"] == "T"],
        }
        for sub_name, sub in subs.items():
            stats = {}
            for c in ("eco", "force", "full"):
                members = [x for x in sub if bucket(x["eq"], eco_max, full_min) == c]
                stats[c] = block(members)
            emit(f"   [{sub_name}]")
            for c in ("eco", "force", "full"):
                s = stats[c]
                emit(f"      {c:<6} sides {s['n']:>5}  matches {s['matches']:>4}  win {pct(s['rate']):>5}%  95% CI {ci(s['lo'], s['hi'])}")
            emit(f"      ordering: {order_verdict(stats)}")

    emit()
    emit("== 3b. Sensitivity grid: side-level win rate, all non-pistol sides (eco_max rows x full_min columns) ==")
    emit("   cell = eco n/rate | force n/rate | full n/rate | ordering. Classes under 30 sides marked *.")
    for eco_max in GRID_ECO:
        for full_min in GRID_FULL:
            if eco_max >= full_min:
                continue
            stats = {}
            for c in ("eco", "force", "full"):
                members = [x for x in recs_np if bucket(x["eq"], eco_max, full_min) == c]
                stats[c] = block(members)
            parts = []
            for c in ("eco", "force", "full"):
                s = stats[c]
                star = "*" if s["n"] < 30 else ""
                parts.append(f"{c} {s['n']}{star}/{pct(s['rate'])}")
            emit(f"   eco {eco_max:>6,} full {full_min:>6,} | " + " | ".join(parts) + f" | {order_verdict(stats)}")

    emit()
    emit("== 3c. Pair-level hi-side win rate by class (higher-equipment side; ties excluded) ==")
    for name, eco_max, full_min in SETS:
        rows_s = per_set[name][2]
        emit(f"{name}: eco_max {eco_max:,}, full_min {full_min:,}")
        for c in CLASSES[1:]:
            members = [x for x in rows_s if x["cls"] == c]
            recs_c = []
            ties = 0
            for x in members:
                if x["ct_eq"] == x["t_eq"]:
                    ties += 1
                    continue
                hs_ct = x["ct_eq"] > x["t_eq"]
                won = (x["ct_win"] == 1) if hs_ct else (x["ct_win"] == 0)
                recs_c.append({"match_key": x["match_key"], "won": won})
            b = block(recs_c)
            emit(f"   {c:<15} rounds {len(members):>5} ties {ties:>2} | hi-side n {b['n']:>5} matches {b['matches']:>4} "
                 f"hi-side win {pct(b['rate']):>5}% CI {ci(b['lo'], b['hi'])}")

    # ---- 4. anti_eco pairing ------------------------------------------------------------
    emit()
    emit("== 4. anti_eco pairing check (default set) ==")
    eco_max, full_min, rows_def = per_set["default"]
    pair_counts = Counter((x["ct_buy"], x["t_buy"]) for x in rows_def if x["round"] not in PISTOL)
    emit("ordered (CT buy, T buy) counts, non-pistol kept rounds, and the class each maps to:")
    for a in ("eco", "force", "full"):
        for b in ("eco", "force", "full"):
            emit(f"   CT {a:<5} T {b:<5} n={pair_counts.get((a, b), 0):>4}  -> {pair_class(a, b)}")
    ae = [x for x in rows_def if x["cls"] == "anti_eco"]
    emit(f"anti_eco rounds (default): {len(ae)}; CT is full and T is eco: {sum(1 for x in ae if x['ct_buy'] == 'full' and x['t_buy'] == 'eco')}; "
         f"CT is eco and T is full: {sum(1 for x in ae if x['ct_buy'] == 'eco' and x['t_buy'] == 'full')}")
    bad_pair = [x for x in ae if {x["ct_buy"], x["t_buy"]} != {"eco", "full"}]
    emit(f"anti_eco rounds whose pair is not exactly {{eco, full}}: {len(bad_pair)}")
    eco_bad = 0
    full_bad = 0
    hi_not_full = 0
    full_won = 0
    eco_won = 0
    for x in ae:
        full_is_ct = x["ct_buy"] == "full"
        full_eq = x["ct_eq"] if full_is_ct else x["t_eq"]
        eco_eq = x["t_eq"] if full_is_ct else x["ct_eq"]
        if not (full_eq >= full_min):
            full_bad += 1
        if not (eco_eq < eco_max):
            eco_bad += 1
        if not (full_eq > eco_eq):
            hi_not_full += 1
        full_side_won = (x["ct_win"] == 1) if full_is_ct else (x["ct_win"] == 0)
        full_won += int(full_side_won)
        eco_won += int(not full_side_won)
    emit(f"full side below full_min: {full_bad} | eco side at or above eco_max: {eco_bad} | higher-equipment side is not the full side: {hi_not_full}")
    emit(f"anti_eco: full side won {full_won} of {len(ae)} ({100 * full_won / len(ae):.1f}%); eco side won {eco_won} ({100 * eco_won / len(ae):.1f}%)")
    # opposing side: each kept round has exactly one CT and one T side row
    opp_ok = all(sides[(x["match_key"], x["round"])].keys() == {CT_CODE, T_CODE} for x in ae)
    emit(f"each anti_eco round has exactly one CT row and one T row (opposing sides): {opp_ok}")
    # per-player recount for three anti_eco rounds
    per_player = defaultdict(list)
    for r in fz.select(["match_key", "round", "team_code", "equipment_value_calc", "money"]).iter_rows(named=True):
        per_player[(r["match_key"], r["round"], r["team_code"])].append((r["equipment_value_calc"], r["money"]))
    sample_idx = [0, len(ae) // 2, len(ae) - 1] if ae else []
    for i in sample_idx:
        x = ae[i]
        emit(f"   sample {x['match_key']} round {x['round']}: CT buy {x['ct_buy']} (sum {x['ct_eq']:,}), T buy {x['t_buy']} (sum {x['t_eq']:,}), ct_win {x['ct_win']}")
        for code, lab in ((CT_CODE, "CT"), (T_CODE, "T")):
            vals = sorted(per_player[(x["match_key"], x["round"], code)], reverse=True)
            emit(f"      {lab} per-player equipment {[v[0] for v in vals]} sum {sum(v[0] for v in vals):,} | money sum {sum(v[1] for v in vals):,}")

    # cross-check against the pilot output (read only)
    bp = pl.read_parquet(BUY_OUT)
    emit(f"round_buy_types.parquet: rows {bp.height}, columns {bp.width}")
    mism = Counter()
    bp_map = {(r["match_key"], r["round"]): r for r in bp.iter_rows(named=True)}
    emit(f"pilot keys == kept keys: {set(bp_map) == set(rounds_by_key)}")
    for x in rows_def:
        b = bp_map.get((x["match_key"], x["round"]))
        if b is None:
            mism["missing"] += 1
            continue
        if b["ct_buy"] != x["ct_buy"]:
            mism["ct_buy"] += 1
        if b["t_buy"] != x["t_buy"]:
            mism["t_buy"] += 1
        if b["round_class"] != x["cls"]:
            mism["round_class"] += 1
        if int(b["ct_equipment"]) != x["ct_eq"] or int(b["t_equipment"]) != x["t_eq"]:
            mism["equipment"] += 1
        if int(b["ct_win"]) != x["ct_win"]:
            mism["ct_win"] += 1
    emit(f"pilot vs re-derived (default set) mismatches: {dict(mism) if mism else 'none'}")

    # ---- 5. Pistol rounds and side mapping ---------------------------------------------
    emit()
    emit("== 5. Pistol rounds and side mapping (re-derived) ==")
    pf = fz.filter(pl.col("round").is_in(list(PISTOL)))
    emit(f"pistol freeze-end player rows {pf.height}; per-player equipment max {pf['equipment_value_calc'].max():,}; "
         f"per-player money max {pf['money'].max():,}")
    p_recs = side_records(pistol_rows)
    big = sorted([x for x in p_recs if x["eq"] >= 3000], key=lambda x: -x["eq"])
    emit(f"pistol-round sides with equipment >= $3,000: {len(big)} of {len(p_recs)}")
    for x in big[:8]:
        vals = sorted([v[0] for v in per_player[(x["match_key"], x["round"], CT_CODE if x['side'] == 'CT' else T_CODE)]], reverse=True)
        emit(f"   {x['match_key']} round {x['round']} {x['side']}: sum {x['eq']:,} per-player {vals}")
    ct_p = fz.filter(pl.col("team_code") == CT_CODE)
    t_p = fz.filter(pl.col("team_code") == T_CODE)
    emit(f"side mapping: CT players with defuser {ct_p['has_defuser'].mean():.3f} | T players with defuser {t_p.filter(pl.col('has_defuser')).height} (expect 0) | "
         f"CT players holding C4 {ct_p.filter(pl.col('inv_c4') > 0).height} (expect 0) | T sides with a C4 holder {t_p.group_by(['match_key','round']).agg((pl.col('inv_c4') > 0).any().alias('h'))['h'].mean():.3f}")
    # pistol rounds: classes under the default set are all pistol by construction
    emit(f"pistol rounds by number: round 1 {sum(1 for r in pistol_rows if r['round'] == 1)}, round 13 {sum(1 for r in pistol_rows if r['round'] == 13)}")

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {LOG}")


if __name__ == "__main__":
    main()
