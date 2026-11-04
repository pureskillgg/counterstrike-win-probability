"""Report for a validation sprint run (experiments/sprint): reads results/<id>.json and preds/<id>.parquet from the run folder and
writes docs/sprint/<run name>/report.md and report.json. Validation only; holdout B and the frozen test are never read here.

    python analysis/sprint_report.py --run <run dir> --out docs/sprint/<run name> [--pinned <positions_validation dir>]

Pinned rules (see the sprint pin):
- Selection: the full-feature setting with the lowest mean validation log loss over seeds 0 to 2 is selected, and only when all three
  seeds exist. Seeds 3 and 4 of that setting are a check, reported separately and not used to select.
- Robustness bar (pin 773d5c12, reused): a difference counts as robust in one direction if its pooled mean (mean of the seed means)
  is at most -0.0005 (or at least +0.0005) and every seed's 95% interval is on the same side of 0. A verdict is given only when every
  seed in its list exists for both arms. Otherwise the verdict says how many seeds are missing.
- Differences are per-second log loss, paired on (match, round, second) with the match-cluster interval from paired(); a difference
  is arm a minus arm b, so a negative value means arm a is better.
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts", "model"))

import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

from spatial_feature_eval import paired  # noqa: E402

KEY = ["match_key", "round", "second"]
TIER1, TIER2 = [0, 1, 2], []  # three seeds per amendment ddd564b1: no separate seed 3-4 check
SEEDS = TIER1 + TIER2
DEFAULT = (4, 0.05, 20)
ABLATIONS = ["base+GEO", "base+VIEW", "base+BLIND", "base+VIEW+BLIND", "base+GEO+BLIND", "base+GEO+VIEW"]
BAR = 0.0005
POS_NAME = "cache_pos_2026_trainctr.parquet"  # the train-centre table, the one the pinned validation runs used
PRED_TOL = 1e-5  # largest absolute difference allowed between a reproduced prediction and its pinned value
# validation log losses of the earlier positions runs at the default setting, seed 0 (positions_validation.py)
REPRO = {"full": 0.45032, "base": 0.45381}
PINNED_FILES = {"full": "full_trainctr_seed0.parquet", "base": "base_seed0.parquet"}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def folder_manifest(run):
    path = os.path.join(run, "manifest.json")
    if not os.path.exists(path):
        raise SystemExit(f"{run} has no manifest.json; the sprint worker writes one into each run folder")
    return json.load(open(path))


def load_results(run):
    res, errors = {}, []
    for path in sorted(glob.glob(os.path.join(run, "results", "*.json"))):
        r, err = read_json(path)
        if err:
            errors.append({"id": os.path.basename(path), "message": f"unreadable results file: {err}"})
            continue
        if r.get("status") != "ok":
            errors.append({"id": r.get("id"), "message": (r.get("traceback") or "").strip().splitlines()[-1:] or ["error"]})
            continue
        try:
            res[(r["set"], r["depth"], r["eta"], r["mcw"], r["seed"])] = r
        except (KeyError, TypeError) as exc:
            errors.append({"id": os.path.basename(path), "message": f"result file lacks a cell field: {exc}"})
    return res, errors


def seed_diffs(run, res, ka, kb, seeds):
    """Per-seed paired per-second log-loss difference, arm ka minus arm kb, on the listed seeds that both arms have."""
    rows = []
    for s in seeds:
        a, b = res.get(ka + (s,)), res.get(kb + (s,))
        if a is None or b is None:
            continue
        pa_path, pb_path = os.path.join(run, "preds", a["id"] + ".parquet"), os.path.join(run, "preds", b["id"] + ".parquet")
        if not (os.path.exists(pa_path) and os.path.exists(pb_path)):
            continue  # a finished cell whose prediction file is missing counts as a missing seed, not a crash
        pa, pb = pl.read_parquet(pa_path), pl.read_parquet(pb_path)
        j = pa.join(pb, on=KEY, how="inner", suffix="_b")
        assert j.height == pa.height == pb.height, f"{a['id']} vs {b['id']}: row sets differ"
        assert (j["label_ct_win"] == j["label_ct_win_b"]).all(), f"{a['id']} vs {b['id']}: labels disagree"
        r = paired(j["match_key"].to_numpy(), j["label_ct_win"].to_numpy().astype(np.float64),
                   j["p"].to_numpy().astype(np.float64), j["p_b"].to_numpy().astype(np.float64), seed=0)
        rows.append({"seed": s, "diff": r["diff"], "ci95": r["ci95"]})
    return rows


def summarize(rows, required):
    """Pooled mean over the seed means, the per-seed bounds, and whether exactly the required seeds are present."""
    have = sorted(r["seed"] for r in rows)
    base = {"seeds_present": have, "required": required, "complete": have == sorted(required), "pooled": None, "per_seed": rows}
    if not rows:
        return base
    d = np.array([r["diff"] for r in rows])
    base.update({"pooled": float(d.mean()), "all_upper_below_0": all(r["ci95"][1] < 0 for r in rows),
                 "all_lower_above_0": all(r["ci95"][0] > 0 for r in rows)})
    return base


def verdict(s):
    if not s["complete"]:
        return f"incomplete ({len(s['seeds_present'])} of {len(s['required'])} seeds)"
    if s["pooled"] <= -BAR and s["all_upper_below_0"]:
        return "first arm better (bar met)"
    if s["pooled"] >= BAR and s["all_lower_above_0"]:
        return "first arm worse (bar met)"
    return "no claim"


def mean_ll(res, key, seeds):
    vals = [res[key + (s,)]["val_logloss"] for s in seeds if key + (s,) in res]
    return (float(np.mean(vals)) if vals else None), len(vals)


def reproduce(run, res, pinned_dir, kind):
    r = res.get((kind,) + DEFAULT + (0,))
    out = {"pinned_ll": REPRO[kind]}
    if r is None:
        return {**out, "status": "not run"}
    out.update({"got_ll": r["val_logloss"], "ll_gap": r["val_logloss"] - REPRO[kind], "pos": r.get("pos"), "device": r.get("device")})
    out["ll_within_1e-4"] = abs(out["ll_gap"]) < 1e-4
    out["pos_is_train_centre_table"] = r.get("pos") == POS_NAME
    if pinned_dir:
        path = os.path.join(pinned_dir, PINNED_FILES[kind])
        if not os.path.exists(path):
            raise SystemExit(f"--pinned folder lacks {PINNED_FILES[kind]}: {pinned_dir}")
        pinned = pl.read_parquet(path)
        got_path = os.path.join(run, "preds", r["id"] + ".parquet")
        if not os.path.exists(got_path):
            out["status"] = "prediction file missing"
            return out
        got = pl.read_parquet(got_path)
        j = got.join(pinned, on=KEY, how="inner", suffix="_pin")
        out["rows_match"] = j.height == got.height == pinned.height
        out["max_abs_pred_diff"] = float(np.abs(j["p"].to_numpy() - j["p_pin"].to_numpy()).max()) if j.height else None
        out["predictions_reproduced"] = bool(out["rows_match"] and out["max_abs_pred_diff"] is not None
                                             and out["max_abs_pred_diff"] <= PRED_TOL)
    return out


def cross_device(run, res, other, res_o):
    """Per arm and seed: the default-setting cell on this device minus the same cell on the other device, paired per second."""
    rows = []
    for kind in ("full", "base"):
        for s in SEEDS:
            a, b = res.get((kind,) + DEFAULT + (s,)), res_o.get((kind,) + DEFAULT + (s,))
            if a is None or b is None:
                continue
            pa_path, pb_path = os.path.join(run, "preds", a["id"] + ".parquet"), os.path.join(other, "preds", b["id"] + ".parquet")
            if not (os.path.exists(pa_path) and os.path.exists(pb_path)):
                rows.append({"arm": kind, "seed": s, "error": "prediction file missing"})
                continue
            pa, pb = pl.read_parquet(pa_path), pl.read_parquet(pb_path)
            j = pa.join(pb, on=KEY, how="inner", suffix="_b")
            if j.height != pa.height or j.height != pb.height:
                rows.append({"arm": kind, "seed": s, "error": "row sets differ"})
                continue
            r = paired(j["match_key"].to_numpy(), j["label_ct_win"].to_numpy().astype(np.float64),
                       j["p"].to_numpy().astype(np.float64), j["p_b"].to_numpy().astype(np.float64), seed=0)
            rows.append({"arm": kind, "seed": s, "diff": r["diff"], "ci95": r["ci95"],
                         "max_abs_pred_diff": float(np.abs(j["p"].to_numpy() - j["p_b"].to_numpy()).max())})
    return rows


def fmt_ll(v, n, of):
    return "none" if v is None else f"{v:.5f} ({n}/{of})"


def main():
    ap = argparse.ArgumentParser(description="Report for a validation sprint run")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pinned", default=None, help="the positions_validation output folder, for the prediction-level check")
    ap.add_argument("--other", default=None, help="the run folder of the other device, for the device check")
    a = ap.parse_args()
    run_manifest = folder_manifest(a.run)
    res, errors = load_results(a.run)
    if any(r.get("pos") != POS_NAME for r in res.values()):
        raise SystemExit(f"{a.run} holds cells fitted on a positions table other than {POS_NAME}; the report refuses them")
    res_o = {}
    if a.other:
        if os.path.abspath(a.other) == os.path.abspath(a.run):
            raise SystemExit("--other is the same folder as --run; the device check needs the other device's folder")
        res_o, _ = load_results(a.other)
        if not res_o:
            raise SystemExit(f"--other holds no finished results: {a.other}")
        if any(r.get("pos") != POS_NAME for r in res_o.values()):
            raise SystemExit(f"{a.other} holds cells fitted on a positions table other than {POS_NAME}; the report refuses them")
        man_o = folder_manifest(a.other)
        if run_manifest["device"] == man_o["device"]:
            raise SystemExit("both folders were run on the same device; the device check needs one folder of each")
        for key in ("definitions", "pos_bytes", "cache_bytes", "rows"):
            if run_manifest[key] != man_o[key]:
                raise SystemExit(f"the two folders differ in {key}; the device check would mix that in")
    cross = cross_device(a.run, res, a.other, res_o) if a.other else None

    cfgs = sorted({(k[1], k[2], k[3]) for k in res if k[0] in ("full", "base")})
    grid = []
    for d, e, m in cfgs:
        fk, bk = ("full", d, e, m), ("base", d, e, m)
        fl, fn = mean_ll(res, fk, TIER1)
        bl, bn = mean_ll(res, bk, TIER1)
        diff = summarize(seed_diffs(a.run, res, fk, bk, TIER1), TIER1)
        grid.append({"depth": d, "eta": e, "mcw": m, "full_ll": fl, "full_n": fn, "base_ll": bl, "base_n": bn,
                     "full_minus_base": diff, "verdict": verdict(diff)})

    candidates = [g for g in grid if g["full_n"] == len(TIER1)]
    selected = min(candidates, key=lambda g: (g["full_ll"], g["depth"], g["eta"], g["mcw"])) if candidates else None
    check = None
    if selected and TIER2:
        d, e, m = selected["depth"], selected["eta"], selected["mcw"]
        fl, fn = mean_ll(res, ("full", d, e, m), TIER2)
        bl, bn = mean_ll(res, ("base", d, e, m), TIER2)
        diff = summarize(seed_diffs(a.run, res, ("full", d, e, m), ("base", d, e, m), TIER2), TIER2)
        check = {"full_ll": fl, "full_n": fn, "base_ll": bl, "base_n": bn, "full_minus_base": diff, "verdict": verdict(diff)}

    abl = []
    for name in ABLATIONS:
        s = summarize(seed_diffs(a.run, res, (name,) + DEFAULT, ("full",) + DEFAULT, SEEDS), SEEDS)
        abl.append({"set": name, "set_minus_full": s, "verdict": verdict(s)})

    repro = {kind: reproduce(a.run, res, a.pinned, kind) for kind in ("full", "base")}

    fit_s = [r["fit_seconds"] for r in res.values()]
    by_set = {}
    for r in res.values():
        by_set.setdefault(r["set"], []).append(r["fit_seconds"])
    timing = {"cells_ok": len(res), "cells_error_or_unreadable": len(errors),
              "fit_seconds_total": round(float(sum(fit_s)), 1) if fit_s else 0.0,
              "fit_seconds_median": round(float(np.median(fit_s)), 1) if fit_s else None,
              "median_fit_seconds_by_set": {k: round(float(np.median(v)), 1) for k, v in sorted(by_set.items())}}

    out = {"run": a.run, "grid": grid, "selected": selected, "selected_check_seeds_3_4": check, "ablations": abl,
           "reproduction_default_seed0": repro, "device_check": cross, "timing": timing, "errors": errors}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(out, f, indent=2, default=str)

    L = ["# Validation sprint report", "",
         "Validation only: 1,168 matches, the split of experiments/positions/positions_validation.py. Holdout B and the frozen test were not read.",
         "Selection on validation is optimistic for the selected setting. Nothing here is a test result.", "",
         "## Reproduction (default setting, seed 0)", "",
         "| Arm | Device | Log loss | Pinned | Gap | Position table | Predictions within 1e-5 of pinned |",
         "|---|---|---|---|---|---|---|"]
    for kind in ("full", "base"):
        r = repro[kind]
        if r.get("status") == "not run":
            L.append(f"| {kind} | not run | not run | {r['pinned_ll']:.5f} | | | |")
            continue
        if "status" in r:
            preds = r["status"]
        elif "predictions_reproduced" not in r:
            preds = "not checked"
        elif r["predictions_reproduced"]:
            preds = "yes"
        elif r["max_abs_pred_diff"] is None:
            preds = "no overlapping rows"
        else:
            preds = f"no (max {r['max_abs_pred_diff']:.2e})"
        L.append(f"| {kind} | {r['device']} | {r['got_ll']:.5f} | {r['pinned_ll']:.5f} | {r['ll_gap']:+.5f} | "
                 f"{r['pos']} | {preds} |")
    L += ["", "## Grid: full minus base, paired on validation, seeds 0-2", "",
          "| depth | eta | min_child_weight | full LL (seeds) | base LL (seeds) | full minus base (pooled) | verdict |",
          "|---|---|---|---|---|---|---|"]
    for g in grid:
        pooled_s = "none" if g["full_minus_base"]["pooled"] is None else f"{g['full_minus_base']['pooled']:+.5f}"
        L.append(f"| {g['depth']} | {g['eta']} | {g['mcw']} | {fmt_ll(g['full_ll'], g['full_n'], 3)} | "
                 f"{fmt_ll(g['base_ll'], g['base_n'], 3)} | {pooled_s} | {g['verdict']} |")
    if not grid:
        L.append("| no results yet | | | | | | |")
    L += ["", "## Selected full setting", ""]
    if selected:
        L.append(f"Selected by the pinned rule (lowest full LL over seeds 0-2): depth {selected['depth']}, eta {selected['eta']}, "
                 f"min_child_weight {selected['mcw']}.")
        if check is not None:
            c = check
            L.append(f"Check on seeds 3 and 4, not used to select: full LL {fmt_ll(c['full_ll'], c['full_n'], 2)}, base LL "
                 f"{fmt_ll(c['base_ll'], c['base_n'], 2)}; full minus base {('none' if c['full_minus_base']['pooled'] is None else format(c['full_minus_base']['pooled'], '+.5f'))}; "
                 f"verdict: {c['verdict']}.")
    else:
        L.append("No setting has all three seeds 0-2 for the full set yet, so nothing is selected.")
    L += ["", "## Feature-group ablations at the default setting, seeds 0-2", "",
          "Difference is ablation set minus full, per second. Negative: the ablation is better, so the dropped groups hurt. "
          "Positive: the full set is better, so the group helps.", "",
          "| Set | Seeds | Set minus full (pooled) | Reading |", "|---|---|---|---|"]
    reading = {"first arm better (bar met)": "ablation better (dropped groups hurt)",
               "first arm worse (bar met)": "full better (group helps)", "no claim": "no claim"}
    for r in abl:
        s = r["set_minus_full"]
        pooled_s = "none" if s["pooled"] is None else f"{s['pooled']:+.5f}"
        L.append(f"| {r['set']} | {len(s['seeds_present'])}/{len(SEEDS)} | {pooled_s} | {reading.get(r['verdict'], r['verdict'])} |")
    if cross is not None:
        L += ["", "## Device check: the same default-setting cell on this device minus the other device", "",
              "A device offset is material if its pooled mean is at least 0.0005 in size. Both arms and every seed should sit near 0.", "",
              "| Arm | Seed | This minus other (per second) | 95% interval | Max abs prediction difference |", "|---|---|---|---|---|"]
        for r in cross:
            if "error" in r:
                L.append(f"| {r['arm']} | {r['seed']} | {r['error']} | | |")
            else:
                L.append(f"| {r['arm']} | {r['seed']} | {r['diff']:+.6f} | {r['ci95'][0]:+.6f} to {r['ci95'][1]:+.6f} | "
                         f"{r['max_abs_pred_diff']:.2e} |")
        pooled_ok = [r["diff"] for r in cross if "diff" in r]
        if not cross:
            L.append("")
            L.append("No default-setting cell is finished on both devices yet, so there is no device check.")
        if pooled_ok:
            L.append("")
            L.append(f"Mean device offset over the paired cells: {np.mean(pooled_ok):+.6f} (n = {len(pooled_ok)}).")
    L += ["", "## Timing", "",
          f"Cells ok: {timing['cells_ok']}; cells with errors or unreadable files: {timing['cells_error_or_unreadable']}. "
          f"Summed fit time {timing['fit_seconds_total']:,.0f} s; median fit {timing['fit_seconds_median']} s.", "",
          "Median fit seconds by set: " + (", ".join(f"{k} {v}" for k, v in timing["median_fit_seconds_by_set"].items()) or "none"), ""]
    if errors:
        L += ["## Errors", ""] + [f"- {e['id']}: {e['message'][-1] if isinstance(e['message'], list) else e['message']}" for e in errors] + [""]
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
