"""Figures for docs/explainability/README.md: GDT ablations, calibration deciles and one waterfall.

Reads the committed tables and writes PNGs to docs/explainability/figures/. Run from the repo root with
the pixi Python:

    G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe scripts/explain/readme_figs.py
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2] / "docs" / "explainability"
TABLES = ROOT / "tables"
LOCAL = ROOT / "local"
FIGS = ROOT / "figures"

INK = "#1B2430"
MUTED = "#4A5563"
ACCENT = "#2F6F8F"
WARN = "#B5542D"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": INK,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

GDT_LABEL = {
    "full": "Full model (83)",
    "minus_alive": "Minus alive counts",
    "minus_armor_sums": "Minus armour sums",
    "minus_blind": "Minus blind",
    "minus_round_state": "Minus round state",
    "minus_view": "Minus view",
    "minus_bomb_geo": "Minus bomb geometry",
    "loadout_only": "Loadout only (46)",
    "round_state_only": "Round state only (6)",
}


def gdt_figure():
    g = pl.read_csv(TABLES / "drop_ablation_83_gdt.csv")
    rows = [r for r in g.iter_rows(named=True) if r["set"] in GDT_LABEL]
    full = next(r for r in rows if r["set"] == "full")
    rows.sort(key=lambda r: r["mean_GDT"])

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    for i, r in enumerate(rows):
        mean = r["mean_GDT"]
        if r["set"] == "full":
            color = INK
        elif mean < 0:
            color = WARN
        else:
            color = ACCENT
        ax.barh(i, mean, color=color, height=0.62, zorder=2)
        ax.errorbar(mean, i, xerr=[[mean - r["GDT_min_seed"]], [r["GDT_max_seed"] - mean]],
                    fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
        ax.text(max(mean, 0) + 0.0012, i, f"{mean:.4f}", va="center", ha="left", color=INK, fontsize=9)
        if r["set"] == "full":
            diff = "reference"
        else:
            diff = f"{r['mean_paired_GDT_difference_vs_full']:+.4f}"
        ax.text(0.0395, i, diff, va="center", ha="right", color=INK, fontsize=9)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([GDT_LABEL[r["set"]] for r in rows])
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.axvline(full["mean_GDT"], color=INK, lw=0.8, ls="--", zorder=1)
    ax.text(0.0002, -0.85, "lookup table = 0", color=MUTED, fontsize=8, va="center")
    ax.text(0.0395, len(rows) - 0.35, "GDT vs full", color=INK, fontsize=9, ha="right", va="bottom",
            fontweight="bold")
    ax.set_xlim(-0.022, 0.0405)
    ax.set_ylim(-1.1, len(rows) - 0.1)
    ax.set_xlabel("GDT, the team's metric (higher is better)")
    ax.grid(axis="x", color="#E6E3DA", lw=0.6, zorder=0)
    fig.suptitle("Drop-group retrains, scored on GDT", x=0.02, ha="left", fontsize=14, fontweight="bold",
                 color=INK, y=0.98)
    fig.text(0.02, 0.91, "Mean of 3 seeds on 295 matches. Whiskers show the range over seeds. "
             "Paired change is against the full model on the same datapoints.",
             fontsize=9, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = FIGS / "gdt_ablation.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def calibration_figure():
    d = pl.read_csv(TABLES / "calibration_deciles_shipped.csv")
    x = d["mean_predicted_p_ct_win"].to_list()
    y = d["observed_ct_win_rate"].to_list()
    lo = [b - a for a, b in zip(d["observed_ci_low"].to_list(), y)]
    hi = [a - b for a, b in zip(d["observed_ci_high"].to_list(), y)]
    gap = max(abs(a - b) for a, b in zip(x, y))

    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="perfect calibration", zorder=1)
    ax.errorbar(x, y, yerr=[lo, hi], fmt="o", color=ACCENT, ecolor=INK, elinewidth=1, capsize=3, ms=6,
                label="decile (Wilson interval)", zorder=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted P(CT wins)")
    ax.set_ylabel("Observed CT-win rate")
    ax.grid(color="#E6E3DA", lw=0.6, zorder=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.text(0.98, 0.04, f"largest gap {gap:.3f}", ha="right", va="bottom", fontsize=9, color=INK)
    fig.suptitle("Calibration by decile", x=0.02, ha="left", fontsize=14, fontweight="bold", color=INK, y=0.98)
    fig.text(0.02, 0.915, "All 1,515,228 validation seconds. The intervals treat seconds as independent, "
             "so they are too narrow.", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = FIGS / "calibration_deciles.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out, gap


def fmt_value(v):
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    if abs(v) >= 10:
        return f"{v:,.1f}"
    return f"{v:.3g}"


def waterfall_figure():
    w = pl.read_csv(LOCAL / "waterfall_largest_loss.csv")
    meta = json.loads((LOCAL / "local_explain_meta.json").read_text(encoding="utf-8"))
    base = meta["base_logit"]
    feats = w["feature"].to_list()
    vals = w["value"].to_list()
    contrib = w["logit_contribution"].to_list()

    order = sorted(range(len(contrib)), key=lambda i: -abs(contrib[i]))
    top = order[:10]
    rest = order[10:]
    total = base + sum(contrib)
    p = 1 / (1 + math.exp(-total))
    assert abs(p * 100 - 0.11) < 0.005, f"prediction {p:.5f} does not match the 0.11% example"

    steps = [("base value (average over the sample)", None, base)]
    steps += [(f"{feats[i]} = {fmt_value(vals[i])}", vals[i], contrib[i]) for i in top]
    steps += [(f"other {len(rest)} inputs, summed", None, sum(contrib[i] for i in rest))]

    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    cur = 0.0
    ys = []
    for row, (label, _, c) in enumerate(steps):
        if row == 0:
            ax.plot([base], [row], marker="o", color=INK, ms=7, zorder=3)
            cur = base
            ax.text(base + 0.12, row, f"{base:+.3f}  (P = {1 / (1 + math.exp(-base)) * 100:.1f}%)",
                    va="center", fontsize=9, color=INK)
        else:
            left, right = min(cur, cur + c), max(cur, cur + c)
            color = WARN if c < 0 else ACCENT
            ax.barh(row, right - left, left=left, color=color, height=0.62, zorder=2)
            if c < 0:
                ax.text(left - 0.12, row, f"{c:+.2f}", va="center", ha="right", fontsize=9, color=INK)
            else:
                ax.text(right + 0.12, row, f"{c:+.2f}", va="center", ha="left", fontsize=9, color=INK)
            cur += c
        ys.append(label)

    last = len(steps)
    ax.barh(last, total, color=INK, height=0.62, zorder=2)
    ax.text(total + 0.15, last, f"{total:+.2f}  (P = {p * 100:.2f}%)", va="center", ha="left",
            fontsize=9, color="white", fontweight="bold")
    ys.append("prediction: P(CT wins)")

    ax.set_yticks(range(len(ys)))
    ax.set_yticklabels(ys, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.text(0.05, -0.75, "50%", color=MUTED, fontsize=8, va="center")
    ax.set_xlabel("logit: log-odds of CT winning (0 = 50%)")
    ax.grid(axis="x", color="#E6E3DA", lw=0.6, zorder=0)
    ax.set_xlim(min(total, cur) - 0.8, 0.9)
    fig.suptitle("The largest-loss second: model P(CT wins) = 0.11%, and CT won the round", x=0.02,
                 ha="left", fontsize=13, fontweight="bold", color=INK, y=0.985)
    fig.text(0.02, 0.92, "Each bar is one input's contribution on the logit scale, largest first. "
             "Top 10 shown; the rest are summed.", fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = FIGS / "waterfall_largest_loss.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out, p


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    g = gdt_figure()
    c, gap = calibration_figure()
    wf, p = waterfall_figure()
    print("wrote", g)
    print("wrote", c, "| largest decile gap", round(gap, 4))
    print("wrote", wf, "| prediction P(CT wins)", round(p, 6))


if __name__ == "__main__":
    main()
