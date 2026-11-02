"""Round-quality evaluation in the intern's protocol: GDT against the fixed situation table, GDT_sit, and per-situation differences.

Writes one markdown report and the grader's raw outputs. Per-second log loss and AUC are kept as diagnostics, not as the headline.

    python evaluation/evaluate.py --rows <rows.parquet> --reference p_v6 --models p_v6,p_v4,p_exp --tag intern_oct14_25

Rows: one row per second, with the grader's columns (see evaluation/intern_gdt/README.md) and one column per model.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import polars as pl

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
GRADER = os.path.join(REPO, "evaluation", "intern_gdt")
sys.path.insert(0, os.path.join(REPO, "analysis"))
sys.path.insert(0, GRADER)
import group_gdt as G  # noqa: E402
import score_gdt as S  # noqa: E402


def fmt_ci(v, ci, sign=""):
    if v is None or ci is None:
        return "none"
    return f"{v:{sign}.4f} ({ci[0]:{sign}.4f} to {ci[1]:{sign}.4f})"


def auc(y, p):
    r = pl.Series(p).rank("average").to_numpy()
    n1 = y.sum()
    n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser(description="Round-quality evaluation in the intern's protocol")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--reference", default="p_v6")
    ap.add_argument("--models", required=True, help="comma-separated columns; the reference is added first if missing")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--window", default="", help="a sentence naming the window and which matches it holds")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1000)
    a = ap.parse_args()
    models = [m.strip() for m in a.models.split(",") if m.strip()]
    if a.reference not in models:
        models = [a.reference] + models
    models = [a.reference] + [m for m in models if m != a.reference]

    out_dir = os.path.join(REPO, "docs", "eval_reports", a.tag)
    os.makedirs(out_dir, exist_ok=True)
    grader = subprocess.run([sys.executable, os.path.join(GRADER, "score_gdt.py"), "--rows", a.rows, "--models", ",".join(models),
                             "--seed", str(a.seed), "--reps", str(a.reps), "--out", out_dir],
                            capture_output=True, text=True)
    print(grader.stdout)
    if grader.returncode != 0:
        print(grader.stderr)
        raise SystemExit("grader failed")
    summ = json.load(open(os.path.join(out_dir, "summary.json")))

    rows = pl.read_parquet(a.rows).drop_nulls(models)
    rounds = rows.select(["match", "round"]).n_unique()
    diag = {}
    y = rows["y"].to_numpy().astype(np.float64)
    for m in models:
        p = np.clip(rows[m].to_numpy().astype(np.float64), 1e-6, 1 - 1e-6)
        diag[m] = {"ll": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), "auc": auc(y, p)}

    pts = G.add_groups(G.build_points(a.rows, models, os.path.join(GRADER, "table.csv"), a.seed))
    groups = pl.DataFrame(G.group_rows(pts, models, a.reference, a.reps, a.seed))
    groups.write_csv(os.path.join(out_dir, "groups.csv"))

    c = summ["counts"]
    lines = [f"# Round-quality evaluation: {a.tag}", ""]
    if a.window:
        lines += [a.window, ""]
    lines += [
        f"Reference: `{a.reference}`. Rows: {rows.height:,} seconds, {c['matches']:,} matches, {rounds:,} rounds, "
        f"{c['datapoints']:,} grader datapoints in {c['situations']:,} situations ({c[f'situations_with_{S.SIT_MIN}_or_more']} with {S.SIT_MIN} or more). "
        f"Seconds per round: {rows.height / rounds:.1f}. Seed {a.seed}, {a.reps} resamples of matches.",
        "",
        "Positive differences mean the model beats the reference. Intervals resample matches.",
        "",
        "## Headline: GDT against the table",
        "",
        "| Model | GDT (95%) | GDT_sit (95%) | rGDT | Matches better than table | GDT minus reference (95%) | GDT_sit minus reference (95%) |",
        "|---|---|---|---|---|---|---|",
    ]
    vs = summ["vs_first"]
    for m in models:
        r = summ["models"][m]
        d = vs.get(m, {})
        dv = fmt_ci(d.get("GDT_difference"), d.get("GDT_difference_ci"), "+") if d else "reference"
        ds = fmt_ci(d.get("GDT_sit_difference"), d.get("GDT_sit_difference_ci"), "+") if d else "reference"
        lines.append(f"| `{m}` | {fmt_ci(r['GDT'], r['GDT_ci'])} | {fmt_ci(r['GDT_sit'], r['GDT_sit_ci'])} | {r['rGDT']:.1%} | "
                     f"{r['share_of_matches_model_better']:.1%} | {dv} | {ds} |")
    lines += ["", "## Diagnostics (per second, all rows)", "", "| Model | Log loss | AUC |", "|---|---|---|"]
    for m in models:
        lines.append(f"| `{m}` | {diag[m]['ll']:.5f} | {diag[m]['auc']:.5f} |")
    lines += ["", "## By situation: difference from the reference (model better when positive)", ""]
    for fam in G.FAMILIES:
        name = fam[0]
        sub = groups.filter(pl.col("family") == name)
        if sub.height == 0:
            continue
        lines += [f"### {name}", "", "| Group | Datapoints | Matches | " + " | ".join(f"`{m}`" for m in models[1:]) + " |",
                  "|---|---|---|" + "---|" * (len(models) - 1)]
        for g in sub["group"].unique(maintain_order=True).to_list():
            s = sub.filter(pl.col("group") == g)
            cells = []
            for m in models[1:]:
                r = s.filter(pl.col("model") == m)
                cells.append(f"{r['diff'][0]:+.4f} ({r['ci_low'][0]:+.4f} to {r['ci_high'][0]:+.4f})" if r.height else "none")
            lines.append(f"| {g} | {s['datapoints'][0]:,} | {s['matches'][0]} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Notes", "",
              "- The table is fixed (`evaluation/intern_gdt/table.csv`, built on upload days Oct 9 to 13, 2026).",
              "- Groups were defined before the run (`analysis/group_gdt.py`). Read any single interval with care: many groups are shown at once.",
              "- Models with predictions on only every other second (positions, graph, set) are scored on that subsample. The intern's protocol asks for every second.",
              ""]
    report = os.path.join(REPO, "docs", "eval_reports", f"{a.tag}.md")
    open(report, "w").write("\n".join(lines))
    print("report:", report)


if __name__ == "__main__":
    main()
