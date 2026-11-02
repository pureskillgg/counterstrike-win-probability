"""Look at single rounds: the model's win probability second by second, beside what happened (kills, plant, defuse, explosion).

  python -I -X utf8 scripts\\model\\inspect_rounds.py --pred data\\combined\\export_winprob_v1_test_predictions.parquet [--out data\\combined\\round_inspection] [--n 4]

--pred is written by export_model.py: one row per second of every HELD-OUT round (matches never used to fit or to stop the model), with the
probability P(CT wins). Nothing is fitted here. Outputs (in --out):
  index.html    self-contained page: how to read it, summary tables, and one probability curve per selected round (SVG, no JavaScript)
  swings.csv    every one-second jump of the probability of 0.20 or more in the held-out rounds, with the nearest event (the swing audit)
  summary.json  the same numbers as the tables

Selected rounds (each group is the first --n by its criterion; a round already shown in an earlier group is skipped, so each appears once):
  confidently wrong  highest mean log loss over the round, among rounds of at least 20 seconds (the model was sure, the round went the other way)
  biggest swings     largest one-second change of the probability
  comebacks          the winner's probability fell lowest (the model had written the eventual winner off)
  clutches           a side reached 1 alive against 2 or more, and won
  random             rounds drawn at random (seed 0): an unbiased look
Behaviour table: for seconds that contain exactly ONE event, the average change of P(CT wins) after a CT death, a T death and a plant,
and the share of those seconds where the probability moved the expected way (CT death down, T death up, plant down). The grid stops before the decisive event, so defuses and explosions are not measured here. A model that fails this is wrong in a
way no aggregate score shows.
"""
import os

os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
import argparse
import html
import json
import sys

import numpy as np
import polars as pl

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
TICK = 64
SIDE_T, SIDE_CT = 2, 3
SWING = 0.20
BLUE, ORANGE, GREY = "#2a6fdb", "#e07b00", "#8a8f98"


def load(pred, store):
    P = pl.read_parquet(pred)
    keys = P.select("match_key", "round").unique()
    base = os.path.join(ROOT, "data", store)
    R = pl.read_parquet(os.path.join(base, "rounds.parquet"), columns=["match_key", "round", "plant_tick", "defuse_tick", "explode_tick"]).join(keys, on=["match_key", "round"])
    D = pl.read_parquet(os.path.join(base, "deaths.parquet"), columns=["match_key", "round", "tick", "side"]).join(keys, on=["match_key", "round"])
    return P, R, D


def events_of(r, deaths):
    """[(tick, kind)] with kind in death_ct, death_t, plant, defuse, explode."""
    ev = []
    for tick, side in deaths:
        if side == SIDE_CT:
            ev.append((int(tick), "death_ct"))
        elif side == SIDE_T:
            ev.append((int(tick), "death_t"))
    for col, kind in (("plant_tick", "plant"), ("defuse_tick", "defuse"), ("explode_tick", "explode")):
        v = r.get(col)
        if v is not None:
            ev.append((int(v), kind))
    return sorted(ev)


def analyse(P, R, D):
    rinfo = {(r["match_key"], r["round"]): r for r in R.iter_rows(named=True)}
    dmap = {}
    for mk, rd, tick, side in D.iter_rows():
        dmap.setdefault((mk, rd), []).append((tick, side))
    rounds, swings = [], []
    beh = {k: [] for k in ("death_ct", "death_t", "plant")}  # the grid ends before the decisive event, so defuse and explosion never fall inside a step
    for (mk, rd), g in P.sort(["match_key", "round", "second"]).partition_by(["match_key", "round"], as_dict=True, maintain_order=True).items():
        sec = g["seconds_since_round_live"].to_numpy().astype(float)
        p = np.clip(g["p"].to_numpy().astype(float), 1e-6, 1 - 1e-6)
        tick = g["tick"].to_numpy().astype(np.int64)
        y = int(g["label_ct_win"][0])
        n = len(p)
        if n < 3:
            continue
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
        ac, at = g["alive_ct"].to_numpy().astype(np.int64), g["alive_t"].to_numpy().astype(np.int64)  # int8 here would overflow in the chart arithmetic
        ev = events_of(rinfo.get((mk, rd), {}), dmap.get((mk, rd), []))
        ev_sec = [(sec[0] + (t - tick[0]) / TICK, k) for t, k in ev]
        dp = np.diff(p)
        j = int(np.argmax(np.abs(dp)))
        wp = p if y == 1 else 1 - p
        lone_ct = bool(((ac == 1) & (at >= 2)).any())
        lone_t = bool(((at == 1) & (ac >= 2)).any())
        clutch = (y == 1 and lone_ct) or (y == 0 and lone_t)
        rounds.append({"match_key": mk, "round": rd, "map": g["map_name"][0] if "map_name" in g.columns else "", "y": y, "n": n, "mean_ll": float(ll.mean()),
                       "max_step": float(abs(dp[j])), "max_step_at": float(sec[j + 1]), "winner_min_p": float(wp.min()), "clutch": clutch,
                       "round_number": int(g["round_number"][0]) if "round_number" in g.columns else rd})
        # events by one-second step: step k covers (sec[k-1], sec[k]]
        idx = np.searchsorted(sec, [s for s, _ in ev_sec], side="left") if ev_sec else np.array([], dtype=int)
        per_step = {}
        for (s, kind), k in zip(ev_sec, idx):
            if 1 <= k < n:
                per_step.setdefault(int(k), []).append(kind)
        for k, kinds in per_step.items():
            if len(kinds) == 1 and kinds[0] in beh:
                beh[kinds[0]].append(float(dp[k - 1]))
        for k in np.nonzero(np.abs(dp) >= SWING)[0]:
            kinds = per_step.get(int(k) + 1, [])
            swings.append({"match_key": mk, "round": rd, "second": float(sec[k + 1]), "p_before": float(p[k]), "p_after": float(p[k + 1]), "delta": float(dp[k]),
                           "alive_before": f"{ac[k]}v{at[k]}", "alive_after": f"{ac[k + 1]}v{at[k + 1]}", "event": "+".join(kinds) or "none"})
        rounds[-1]["_g"] = (sec, p, ac, at, ev_sec)
    return rounds, swings, beh


def pick(rounds, n, seed=0):
    used, out = set(), {}

    def take(name, items):
        got = []
        for r in items:
            key = (r["match_key"], r["round"])
            if key in used:
                continue
            used.add(key)
            got.append(r)
            if len(got) == n:
                break
        out[name] = got

    long_enough = [r for r in rounds if r["n"] >= 20]
    take("Confidently wrong", sorted(long_enough, key=lambda r: -r["mean_ll"]))
    take("Biggest swings", sorted(rounds, key=lambda r: -r["max_step"]))
    take("Comebacks (the winner was written off)", sorted(rounds, key=lambda r: r["winner_min_p"]))
    take("Clutches (1 alive against 2 or more, won)", sorted([r for r in rounds if r["clutch"]], key=lambda r: -r["mean_ll"]))
    rng = np.random.default_rng(seed)
    take("Random rounds", [rounds[i] for i in rng.permutation(len(rounds))])
    return out


def svg_round(r):
    sec, p, ac, at, ev = r["_g"]
    W_, H_, ml, mr, mt, mb = 920, 300, 46, 16, 14, 78
    xmax = max(float(sec[-1]), 30.0)
    X = lambda s: ml + (W_ - ml - mr) * s / xmax
    Y = lambda v: mt + (H_ - mt - mb) * (1 - v)
    ph = H_ - mb + 12  # alive panel top
    PH = 52
    YA = lambda v: ph + PH - PH * v / 5.0
    s = [f'<svg viewBox="0 0 {W_} {H_}" role="img" aria-label="win probability curve" class="rc">']
    for v in (0, 0.25, 0.5, 0.75, 1):
        s.append(f'<line x1="{ml}" x2="{W_ - mr}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="g{" mid" if v == 0.5 else ""}"/><text x="{ml - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="t">{v:.2f}</text>')
    for t in range(0, int(xmax) + 1, 15):
        s.append(f'<line x1="{X(t):.1f}" x2="{X(t):.1f}" y1="{mt}" y2="{H_ - mb}" class="g v"/><text x="{X(t):.1f}" y="{H_ - mb + 11}" text-anchor="middle" class="t">{t}s</text>')
    for st, kind in ev:
        if kind in ("plant", "defuse", "explode"):
            col = {"plant": "#111", "defuse": "#1a9c4b", "explode": "#d33"}[kind]
            s.append(f'<line x1="{X(st):.1f}" x2="{X(st):.1f}" y1="{mt}" y2="{H_ - mb}" stroke="{col}" stroke-dasharray="4 3" stroke-width="1.4"/><text x="{X(st) + 3:.1f}" y="{mt + 10}" class="t" fill="{col}">{kind}</text>')
    pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(sec, p))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{BLUE}" stroke-width="2.2"/>')
    for st, kind in ev:
        if kind in ("death_ct", "death_t"):
            k = int(np.clip(np.searchsorted(sec, st), 0, len(p) - 1))
            col = BLUE if kind == "death_ct" else ORANGE
            s.append(f'<circle cx="{X(st):.1f}" cy="{Y(p[k]):.1f}" r="4.5" fill="{col}" stroke="#fff" stroke-width="1.2"><title>{"CT" if kind == "death_ct" else "T"} player dies at {st:.0f}s</title></circle>')
    # alive counts panel
    s.append(f'<text x="{ml - 6}" y="{ph + 8}" text-anchor="end" class="t">alive</text>')
    for arr, col in ((ac, BLUE), (at, ORANGE)):
        s.append(f'<polyline points="{" ".join(f"{X(a):.1f},{YA(b):.1f}" for a, b in zip(sec, arr))}" fill="none" stroke="{col}" stroke-width="1.8"/>')
    s.append(f'<line x1="{ml}" x2="{W_ - mr}" y1="{YA(0):.1f}" y2="{YA(0):.1f}" class="g"/>')
    s.append("</svg>")
    return "".join(s)


def card(r):
    won = "CT won" if r["y"] == 1 else "T won"
    return (f'<div class="card"><h3>{html.escape(str(r["map"]))} round {r["round_number"]} - {won}</h3>'
            f'<p class="m">mean log loss {r["mean_ll"]:.2f} - biggest one-second move {r["max_step"]:.2f} at {r["max_step_at"]:.0f}s - winner\'s lowest probability {r["winner_min_p"]:.2f} - '
            f'match {html.escape(r["match_key"].split("/")[-2][:8])}</p>{svg_round(r)}</div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--store", default="combined")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=4)
    a = ap.parse_args()
    if a.n < 1:
        sys.exit("--n must be at least 1")
    out = a.out or os.path.join(ROOT, "data", a.store, "round_inspection")
    os.makedirs(out, exist_ok=True)
    P, R, D = load(a.pred, a.store)
    print(f"{P.height:,} held-out seconds, {P.select('match_key').n_unique()} matches, {P.select('match_key', 'round').n_unique():,} rounds", flush=True)
    rounds, swings, beh = analyse(P, R, D)
    sel = pick(rounds, a.n)
    sw = pl.DataFrame(swings).sort(pl.col("delta").abs(), descending=True) if swings else pl.DataFrame()
    if swings:
        sw.write_csv(os.path.join(out, "swings.csv"))
    n_sw = len(swings)
    explained = sum(1 for s in swings if s["event"] != "none")
    by_event = {}  # a jump that contains several events counts once under each of them
    for s in swings:
        for kind in set(s["event"].split("+")) - {"none"}:
            by_event[kind] = by_event.get(kind, 0) + 1
    by_event["no event in that second"] = sum(1 for s in swings if s["event"] == "none")
    expected = {"death_ct": -1, "death_t": 1, "plant": -1}
    brows = {}
    for k, v in beh.items():
        v = np.asarray(v)
        if len(v):
            brows[k] = {"seconds_with_only_this_event": int(len(v)), "mean_change": float(v.mean()), "median_change": float(np.median(v)),
                        "share_expected_direction": float(np.mean(np.sign(v) == expected[k])), "share_no_change_over_0.005": float(np.mean(np.abs(v) < 0.005))}
    ll_all = np.array([r["mean_ll"] for r in rounds])
    summary = {"rounds": len(rounds), "swing_threshold": SWING, "swings": n_sw, "swings_with_an_event": explained,
               "swings_per_1000_rounds": 1000 * n_sw / max(1, len(rounds)), "swings_by_event": by_event, "behaviour": brows,
               "round_mean_logloss_quantiles": {q: float(np.quantile(ll_all, q)) for q in (0.5, 0.9, 0.99)}}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w", encoding="utf-8"), indent=1)

    css = ("body{font:15px/1.5 system-ui,sans-serif;max-width:980px;margin:24px auto;padding:0 16px;background:#fff;color:#1c1f24}"
           "h1{font-size:24px}h2{margin-top:34px;font-size:19px}h3{margin:0;font-size:15px}.m{margin:2px 0 6px;color:#555;font-size:13px}"
           ".card{border:1px solid #d5d9df;border-radius:8px;padding:10px 12px;margin:14px 0}table{border-collapse:collapse;font-size:14px}"
           "td,th{border-bottom:1px solid #e3e6ea;padding:4px 10px;text-align:right}th:first-child,td:first-child{text-align:left}"
           ".rc{width:100%;height:auto}.g{stroke:#e1e4e8;stroke-width:1}.g.mid{stroke:#9aa0a8;stroke-dasharray:3 3}.t{font-size:10.5px;fill:#6b7280}"
           "@media(prefers-color-scheme:dark){body{background:#14161a;color:#e6e8eb}.card{border-color:#343a42}.m{color:#a3a9b2}td,th{border-color:#2c3138}.g{stroke:#2a2f36}.t{fill:#9aa0a8}}")
    h = [f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Round inspection</title><style>{css}</style><body>",
         "<h1>Looking at single rounds</h1>",
         "<p>Each chart is one held-out round (a match the model never saw while it was being fitted). The blue line is the model's probability that the CT side wins the round, second by second. "
         "<b style=color:#2a6fdb>Blue dots</b> mark a CT player dying, <b style=color:#e07b00>orange dots</b> a T player dying; dashed lines mark the bomb being planted, defused or exploding. "
         "The strip under the chart shows how many players each side has alive. A good curve moves when something happens, in the direction that makes sense, and ends near the real result.</p>",
         f"<h2>Does the probability move the right way?</h2><p>Seconds that contain exactly one event, over {len(rounds):,} held-out rounds.</p><table><tr><th>event</th><th>seconds</th><th>average change of P(CT wins)</th><th>moved the expected way</th><th>barely moved (&lt;0.005)</th></tr>"]
    names = {"death_ct": "CT player dies (expect down)", "death_t": "T player dies (expect up)", "plant": "bomb planted (expect down)"}
    for k, v in brows.items():
        h.append(f"<tr><td>{names[k]}</td><td>{v['seconds_with_only_this_event']:,}</td><td>{v['mean_change']:+.3f}</td><td>{v['share_expected_direction']:.0%}</td><td>{v['share_no_change_over_0.005']:.0%}</td></tr>")
    h.append("</table>")
    h.append(f"<h2>Large jumps (swing audit)</h2><p>{n_sw:,} one-second jumps of {SWING:.2f} or more ({summary['swings_per_1000_rounds']:.0f} per 1,000 rounds); "
             f"{explained / max(1, n_sw):.0%} contain a recorded kill or plant in that second. The rest have no recorded event in that second: the player table changed some other way (damage, purchases or drops, the clock) and the tool does not attribute them further. "
             "The biggest ones are in swings.csv; the first rows:</p><table><tr><th>round</th><th>second</th><th>before</th><th>after</th><th>alive</th><th>event</th></tr>")
    for s in (sw.head(12).iter_rows(named=True) if swings else []):
        h.append(f"<tr><td>{s['match_key'].split('/')[-2][:8]} r{s['round']}</td><td>{s['second']:.0f}</td><td>{s['p_before']:.2f}</td><td>{s['p_after']:.2f}</td><td>{s['alive_before']} to {s['alive_after']}</td><td>{s['event']}</td></tr>")
    h.append("</table>")
    for name, items in sel.items():
        h.append(f"<h2>{html.escape(name)}</h2>")
        h.extend(card(r) for r in items)
    h.append("</body></html>")
    open(os.path.join(out, "index.html"), "w", encoding="utf-8").write("\n".join(h))
    print(json.dumps({k: summary[k] for k in ("rounds", "swings", "swings_per_1000_rounds", "swings_by_event", "behaviour")}, indent=1))
    print("->", os.path.join(out, "index.html"))


if __name__ == "__main__":
    main()
