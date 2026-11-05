"""Figures for the per-input effect curves (reads docs/explainability/tables from feature_curves.py)."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from xgb83_explain import DOCS  # noqa: E402

TABLES = os.path.join(DOCS, "tables")
FIGS = os.path.join(DOCS, "figures")
LOADOUT_PANELS = ["st_hp_diff", "st_equip_diff", "st_armor_n_diff", "st_rifle_diff", "st_awp_diff", "st_smg_diff",
                  "st_hp_ct", "st_hp_t", "st_equip_ct", "st_equip_t", "st_armor_n_ct", "st_armor_n_t"]


def panel(ax, long, summ, feat):
    d = long.filter(pl.col("feature") == feat).filter(pl.col("mean_value").is_not_null()).sort("mean_value")
    s = summ.filter(pl.col("feature") == feat).row(0, named=True)
    x = d["mean_value"].to_numpy()
    y = d["mean_signed_effect_pp"].to_numpy()
    ax.axhline(0, color="#999999", lw=0.8)
    ax.plot(x, y, marker="o", ms=3, lw=1.2, color="#4C72B0")
    title = feat
    if s["slope_pp_per_unit"] is not None:
        title += f"\nslope {s['slope_pp_per_unit']:+.3g} pp/unit"
    if s["nan_rate"] and s["nan_rate"] > 0:
        title += f"  (missing {s['nan_rate']:.0%})"
    ax.set_title(title, fontsize=7.5)
    ax.set_xlabel(s["unit"][:48], fontsize=6)
    ax.set_ylabel("effect, pp", fontsize=6)
    ax.tick_params(labelsize=6)
    if s["note"]:
        ax.set_facecolor("#fff4e6")


def main():
    long = pl.read_csv(os.path.join(TABLES, "feature_curves_all83.csv"))
    summ = pl.read_csv(os.path.join(TABLES, "feature_slopes_all83.csv"))
    os.makedirs(FIGS, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(9.5, 7.5))
    for ax, f in zip(axes.ravel(), LOADOUT_PANELS):
        panel(ax, long, summ, f)
    fig.suptitle("Economy and health inputs: signed effect on P(CT wins) against the input's value (binned, validation sample)",
                 fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "curves_loadout.png"), dpi=150)
    plt.close(fig)

    top = summ.sort("mean_abs_effect_pp", descending=True).head(24)["feature"].to_list()
    for part, chunk in (("a", top[:12]), ("b", top[12:24])):
        fig, axes = plt.subplots(3, 4, figsize=(9.5, 7.5))
        for ax, f in zip(axes.ravel(), chunk):
            panel(ax, long, summ, f)
        ranks = "1 to 12" if part == "a" else "13 to 24"
        fig.suptitle(f"Largest inputs by mean absolute effect, ranks {ranks}: signed effect against the value (binned)", fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, f"curves_top_{part}.png"), dpi=150)
        plt.close(fig)
    print("wrote curves_loadout.png, curves_top_a.png, curves_top_b.png")


if __name__ == "__main__":
    main()
