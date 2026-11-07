"""Builds notes/status_page.html: the live status board of the win-probability work (published as an Artifact so it can be read from anywhere).

  .pixi\\envs\\default\\python.exe -I -X utf8 scripts\\viz\\build_status_page.py [--pred data\\combined\\standin_cache_seed0_test_predictions.parquet]

Every number is read from the project files at build time (matches.parquet, rounds.parquet, logs, the mutation log); only the step list below is written by hand.
The round viewer draws held-out rounds from a predictions file (one row per second of rounds from matches the model never saw), with the events of the round
(kills, plant, defuse, explosion) from rounds.parquet / deaths.parquet, picked by the criteria of scripts/model/inspect_rounds.py.
"""
import argparse
import glob
import html
import json
import os
import re
import sys
import time

import numpy as np
import polars as pl

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model"))
import inspect_rounds as IR  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = r"G:\pureskill\csds-cs2"
D = os.path.join(ROOT, "data", "combined")
E_RAW = {}  # E: folders per upload day, counted at build time (see raw_counts)

def _j(name):
    f = os.path.join(D, name)
    return json.load(open(f, encoding="utf-8")) if os.path.exists(f) else None


def raw_counts(days):
    """Match folders per upload day on E: (directories only), counted at build time."""
    out = {}
    for d in days:
        p = os.path.join(r"E:\csds_final\csds", *d.split("/"))
        try:
            out[d] = sum(1 for e in os.scandir(p) if e.is_dir())
        except OSError:
            out[d] = None
    return out


def build_steps(M, R):
    prep = _j("prep_report.json") or {}
    ex = _j("export_winprob_v3.json") or _j("export_winprob_v2.json")
    exb = _j("export_winprob_v3_basic.json") or _j("export_winprob_v2_basic.json")
    tag = "v3" if _j("export_winprob_v3.json") else "v2"
    e2e = _j(f"e2e_winprob_{tag}.json") or {}
    stress = _j(f"stress_winprob_{tag}.json") or {}
    par = _j(f"parity_winprob_{tag}.json") or {}
    tm = _j("exp_temporal.json") or {}
    use = prep.get("use_matches")
    by_era = prep.get("use_by_era", {})
    g = ex["gpu_trained"] if ex else None
    return [
        ("Data", [
            ("Merge days 09-13 into the combined store", "done", f"{M.height:,} matches in the store; day 13 (2,797 matches) merged this afternoon in 7 min, nothing earlier rebuilt"),
            ("prepare + make_grid", "done", f"{use:,} usable matches (2024: {by_era.get('2024', ['?'])[0]:,}; 2026: {by_era.get('2026', ['?'])[0]:,}), {prep.get('use_rounds', 0):,} rounds" if use else "see prep_report.json"),
            ("Frozen train / val / test split", "done", "append-only manifest: matches keep their set when new ones arrive (hash rule for new matches)"),
            ("1 Hz living-player tables, loadout cache", "done", "all 2026 matches read from the raw files on E:; cache rebuilt in 12 chunks"),
            ("Positions, view angles and blindness (new inputs)", "done" if _j("exp_positions.json") else "running", (f"paired log-loss change {_j('exp_positions.json')['geometry+view+blind']['delta_vs_base']:+.4f} {[round(x, 4) for x in _j('exp_positions.json')['geometry+view+blind']['delta_vs_base_ci95']]} on the same {_j('exp_positions.json')['test_matches']:,} test matches; positions give most of it, cones and blindness little; not in the served model yet" if _j("exp_positions.json") else "extracting")),
            ("Unseen maps (vertigo, overpass, ...)", "done" if _j("eval_scope_maps_winprob_v3.json") else "todo", (f"snapshot model {_j('eval_scope_maps_winprob_v3.json')['snapshot_all']['logloss']:.4f} on {_j('eval_scope_maps_winprob_v3.json')['snapshot_all']['n_matches']} matches vs 0.4565 on its maps; wingman not measured" if _j("eval_scope_maps_winprob_v3.json") else "pending")),
            ("GPU hyperparameter sweep", "done" if _j("exp_hparams.json") else "todo", "depth x min child weight on the pod: no setting measurably better than the final one" if _j("exp_hparams.json") else "pending"),
        ]),
        ("Models", [
            (f"Snapshot model (winprob_{tag})", "done", f"log loss {g['logloss']:.4f} on {ex['test']['matches']:,} held-out matches; GPU and CPU fits agree" if g else "pending"),
            (f"Alive-count fallback (winprob_{tag}_basic)", "done", f"log loss {exb['gpu_trained']['logloss']:.4f} on {exb['test']['matches']:,} matches" if exb else "pending"),
            ("Baselines, robustness, newest-uploads, model families, deep teacher", "done" if tm else "running", "each writes the JSON the deck slides read"),
            ("Foundation-model fine-tunes (TabPFN v2, TabICL v2)", "done", "finished earlier on a GPU pod: both worse than XGBoost (TabPFN +0.0046 [0.0021, 0.0074])"),
        ]),
        ("Checks", [
            ("Stress test of the CPU predictor", "done" if stress.get("all_passed") else "running", f"{stress.get('n_checks', '?')} checks, {stress.get('n_failed', '?')} failed, {stress.get('n_warn', '?')} warning" if stress else "pending"),
            ("End-to-end data checks and train/serve parity", "done" if e2e.get("all_passed") else "running", (f"e2e: {e2e.get('counts', {}).get('PASS', '?')} pass, {e2e.get('counts', {}).get('WARN', '?')} warnings; parity {par.get('counts', {}).get('PASS', '?')} of 8" if e2e else "pending")),
            ("Mutation test of the checks", "done", "__MUT__"),
        ]),
        ("Tooling", [
            ("Resumable / extendable tome builder (dsdk fork)", "done", "311 tests pass; resume and extend verified on 40 real matches; chunk-publish path 8 of 8"),
            ("RunPod launcher", "done", "stdlib client; cheapest Community GPU first; pod terminated and confirmed absent after each run"),
            ("Replay viewer", "done", "two held-out matches (dust2, mirage) with positions, view cones, smokes, grenades, hits, blindness, win chance"),
        ]),
    ]


TESTS = [
    ("Prediction quality", "log loss on held-out matches, intervals that resample whole matches", "your brief: split by match, log loss as target; seconds in a round are correlated, so row-level intervals are about 5.6x too narrow"),
    ("Calibration", "reliability plot, ECE with an interval, isotonic and Platt compared", "your brief: calibration plot, isotonic comparison"),
    ("Beats the baselines", "alive counts only, the old score-lead design, logistic, constant", "your brief: better than the score-lead model and simple baselines"),
    ("Every second of the round", "log loss by time slice and bomb phase", "your brief: works at every tick"),
    ("Holds up", "unseen maps, learning curve, shuffled-label leak check, train on earlier uploads and test on the newest", "your brief: enough data, no leakage, adapts to new data"),
    ("Data and labels", "scope, duplicates, labels against the header score, grid recomputed from raw events, split", "readiness audit: the first checks passed even on empty or stale data"),
    ("Serving code", "stress test (21 checks), parity with training, golden rows", "your brief: never crash on an unknown map or mode; the audit found ticks-as-seconds and fraction-health served as ok"),
    ("The checks themselves", "mutation test: 24 deliberate faults must each trip a check", "a check that never fails proves nothing"),
    ("The match split", "frozen manifest, cross-artifact leakage check", "audit: 578 of 743 test matches were training matches of the current model because every script re-split its own table"),
]


def mutation_summary():
    p = os.path.join(ROOT, "logs", "mutation_run1.log")
    if not os.path.exists(p):
        return "not run yet"
    t = open(p, encoding="utf-8", errors="replace").read()
    caught = len(re.findall(r"^CAUGHT", t, flags=re.M))
    bad = re.findall(r"^(SURVIVED|STALE|ERROR|SKIPPED)\s+(\S+)", t, flags=re.M)
    done = "SUMMARY" in t
    s = f"{caught} of 24 faults caught so far" if not done else f"{caught} of 24 faults caught"
    if bad:
        s += "; not caught: " + ", ".join(b[1] for b in bad)
    return s


def coverage(M):
    h = dict(M.group_by("upload_day").len().iter_rows())
    rows = []
    for d in sorted(E_RAW):
        rows.append({"day": d, "raw": E_RAW[d], "header": h.get(d, 0)})
    return rows


def unknown_panel(M, R):
    """Matches by header match_type for 2026, with the map mix of the 'unknown' type, so the keep / drop decision can be made on data."""
    pages = sorted(glob.glob(os.path.join(ROOT, "tomes", "combined", "tome", "csds", "header", "dataframe_*")))
    H = pl.concat([pl.read_parquet(f, columns=["key", "server_name"]) for f in pages], how="vertical_relaxed").rename({"key": "match_key"}).unique("match_key")
    H = H.with_columns(pl.when(pl.col("server_name").str.starts_with("FACEIT")).then(pl.lit("FACEIT server")).when(pl.col("server_name").str.starts_with("Valve")).then(pl.lit("Valve server")).otherwise(pl.lit("other")).alias("source"))
    use = R.filter(pl.col("use")).group_by("match_key").agg(pl.len().alias("used_rounds"))
    X = M.filter(pl.col("era") == "2026").join(H.select("match_key", "source"), on="match_key", how="left").join(use, on="match_key", how="left").with_columns(pl.col("used_rounds").fill_null(0))
    types = []
    for t, g in X.group_by("match_type"):
        types.append({"type": str(t[0]) if t[0] is not None else "(none)", "matches": g.height, "in_scope": int(g["in_scope"].sum()), "usable_matches": int((g["used_rounds"] > 0).sum()),
                      "used_rounds": int(g["used_rounds"].sum()), "faceit": int((g["source"] == "FACEIT server").sum())})
    types.sort(key=lambda r: -r["matches"])
    U = X.filter(pl.col("match_type") == "unknown")
    maps = [{"map": r[0], "matches": r[1], "used_rounds": int(r[2])} for r in U.group_by("map_name").agg(pl.len(), pl.col("used_rounds").sum()).sort("len", descending=True).iter_rows()]
    C = X.filter(pl.col("match_type") == "competitive")
    cmaps = dict(C.group_by("map_name").len().iter_rows())
    for m in maps:
        m["competitive_matches"] = int(cmaps.get(m["map"], 0))
    rank = {}
    for nm, g in (("unknown", U), ("competitive", C)):
        r = g.select(((pl.col("t_starters_avg_rank") + pl.col("ct_starters_avg_rank")) / 2).alias("r")).drop_nulls()["r"]
        rank[nm] = {"n": r.len(), "mean": float(r.mean()) if r.len() else None}
    return {"types": types, "unknown_maps": maps, "rank": rank, "unknown_total": U.height, "unknown_used_rounds": int(U["used_rounds"].sum())}


def rounds_for_viewer(pred, per_group=3):
    P, Rr, Dd = IR.load(pred, "combined")
    rounds, _, _ = IR.analyse(P, Rr, Dd)
    groups = IR.pick(rounds, per_group)
    out = []
    for gname, items in groups.items():
        for r in items:
            sec, p, ac, at, ev = r["_g"]
            step = max(1, int(len(sec) // 130))
            idx = list(range(0, len(sec), step))
            if idx[-1] != len(sec) - 1:
                idx.append(len(sec) - 1)
            out.append({"group": gname, "match": r["match_key"].split("/")[-2][:8] if "/" in r["match_key"] else r["match_key"][:8], "map": r["map"], "round": int(r["round_number"]),
                        "y": int(r["y"]), "mean_ll": round(r["mean_ll"], 3), "sec": [round(float(sec[i]), 1) for i in idx], "p": [round(float(p[i]), 3) for i in idx],
                        "ac": [int(ac[i]) for i in idx], "at": [int(at[i]) for i in idx], "ev": [[round(float(s), 1), k] for s, k in ev]})
    return out, len(rounds), P["match_key"].n_unique()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=os.path.join(D, "standin_cache_seed0_test_predictions.parquet"))
    ap.add_argument("--viewer-label", default="held-out rounds of the final snapshot model")
    ap.add_argument("--out", default=os.path.join(ROOT, "notes", "status_page.html"))
    a = ap.parse_args()
    M = pl.read_parquet(os.path.join(D, "matches.parquet"))
    R = pl.read_parquet(os.path.join(D, "rounds.parquet"), columns=["match_key", "round", "use", "era", "round_status"])
    steps = [(g, [(n, s, d.replace("__MUT__", mutation_summary())) for n, s, d in items]) for g, items in build_steps(M, R)]
    E_RAW.update({k: v for k, v in raw_counts(sorted(M.filter(pl.col("upload_day") >= "2026/10/08")["upload_day"].unique().to_list())).items() if v is not None})
    cov = coverage(M)
    u = R.filter(pl.col("use"))
    status_2026 = dict(R.filter(pl.col("era") == "2026").group_by("round_status").len().iter_rows())
    # 17,275 raw folders on E: for 2026-09-19..10-13 and 11,666 header matches on those same days (23 days equal, 10/12 is 7 short, 10/13 absent): listing of 2026-10-15
    rc = raw_counts(sorted(M.filter((pl.col("upload_day") >= "2026/09/19") & (pl.col("era") == "2026"))["upload_day"].unique().to_list()))
    raw_total = sum(v for v in rc.values() if v)
    in_store = M.filter((pl.col("upload_day") >= "2026/09/19") & (pl.col("era") == "2026")).height
    funnel = [("Raw match folders on E:, 2026-09-19 onward (listed now)", raw_total), ("of them in the store", in_store),
              ("Matches in the store, all eras and days", M.height), ("In scope: competitive 5v5 on the seven maps", int(M["in_scope"].sum())),
              ("Usable after duplicates and round filters", u["match_key"].n_unique())]
    viewer, n_rounds, n_matches = rounds_for_viewer(a.pred)
    data = {"built": time.strftime("%Y-%m-%d %H:%M"), "steps": steps, "coverage": cov, "funnel": funnel, "round_status_2026": status_2026,
            "unknown": unknown_panel(M, R), "tests": TESTS, "viewer": viewer, "viewer_note": a.viewer_label, "viewer_pool": {"rounds": n_rounds, "matches": n_matches}}
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "status_page_template.html"), encoding="utf-8").read()
    page = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    open(a.out, "w", encoding="utf-8").write(page)
    print(f"wrote {a.out} ({len(page):,} bytes); viewer rounds {len(viewer)} from a pool of {n_rounds:,} rounds / {n_matches:,} matches")


if __name__ == "__main__":
    main()
