"""Examples from the team's GDT: where the model gains on the table, and three matches to look at.

Reads the rows that xgb83_gdt.py writes (WORK/gdt/rows_population.parquet: every second of the 295-match population, each
model's P(CT wins) in its own column) and uses the team's scorer functions from evaluation/intern_gdt/score_gdt.py, so the
datapoints are the scorer's own draw for seed 0. Writes tables/gdt_*.csv and figures/gdt_*.png under docs/explainability/.

Run from the repo root with the pixi Python:
    G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe scripts\\explain\\gdt_examples.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "evaluation" / "intern_gdt"))
import score_gdt as sg  # noqa: E402

OUT = REPO / "docs" / "explainability"
TABLES = OUT / "tables"
FIGS = OUT / "figures"
WORK = Path("C:/Users/13144/sprint/explain83_ablation/gdt")
ROWS = WORK / "rows_population.parquet"
TABLE = REPO / "evaluation" / "intern_gdt" / "table.csv"
CACHE = "G:/pureskill/csds-cs2/data/combined/cache_snapshot_2026.parquet"
SEED = 0
MODELS = ["full", "no_position", "checklist_77", "loadout_only", "round_state_only", "minus_view", "minus_bomb_geo"]
MIN_POINTS = 60  # matches with fewer sampled datapoints are not picked as examples

INK = "#1B2430"
MUTED = "#4A5563"
ACCENT = "#2F6F8F"
WARN = "#B5542D"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})


def short(match):
    return f"{match.split('/')[-1][:8]}…"  # the id is the match folder name; the path holds the upload date


def load():
    rows = sg.situations(sg.read_rows(str(ROWS)))
    table = pl.read_csv(TABLE, schema_overrides={"tb": pl.Utf8, "econ_ct": pl.Utf8, "econ_t": pl.Utf8})
    table = table.select(sg.CELL + [pl.col("p_ct").alias("p_table")])
    have = [m for m in MODELS if f"{m}_s{SEED}" in rows.columns]
    missing = sorted(set(MODELS) - set(have))
    if missing:
        print("not in the rows file yet (skipped):", missing)
    return rows, table, have


def gains(rows, table, have):
    """Sampled datapoints (the scorer's draw) with each model's gain over the table: table log loss minus model log loss."""
    cols = [f"{m}_s{SEED}" for m in have]
    pts = sg.draw(rows, cols, SEED).join(table, on=sg.CELL, how="left").filter(pl.col("p_table").is_not_null())
    y = pts["y"].to_numpy()
    lt = sg.logloss(pts["p_table"].to_numpy(), y)
    add = {f"gain_{m}": lt - sg.logloss(pts[f"{m}_s{SEED}"].to_numpy(), y) for m in have}
    pts = pts.with_columns([pl.Series(k, v) for k, v in add.items()])
    if "gain_full" in add and "gain_no_position" in add:
        pts = pts.with_columns((pl.col("gain_full") - pl.col("gain_no_position")).alias("diff_position"))
    return pts


def per_match(pts, have):
    gcols = [f"gain_{m}" for m in have] + (["diff_position"] if "diff_position" in pts.columns else [])
    pm = (pts.group_by("match").agg([pl.len().alias("datapoints")] + [pl.col(c).mean() for c in gcols])
              .sort("gain_full", descending=True))
    return pm


def scorer_check(pts, have):
    """The plain mean of the gains, next to the scorer's own GDT for the same seed, as a check of the datapoints."""
    summary = json.load(open(WORK / f"score_seed{SEED}" / "summary.json"))
    out = {}
    for m in have:
        out[m] = {"plain_mean_gain": float(pts[f"gain_{m}"].mean()),
                  "scorer_GDT": summary["models"][f"{m}_s{SEED}"]["GDT"] if f"{m}_s{SEED}" in summary["models"] else None}
    return out


def fig_by_alive(pts):
    agg = (pts.group_by(["alive_ct", "alive_t"]).agg([pl.len().alias("n"), pl.col("gain_full").mean().alias("g")]))
    grid = np.full((6, 6), np.nan)
    n = np.zeros((6, 6), dtype=int)
    for r in agg.iter_rows(named=True):
        grid[int(r["alive_ct"]), int(r["alive_t"])] = r["g"]
        n[int(r["alive_ct"]), int(r["alive_t"])] = r["n"]
    lim = np.nanmax(np.abs(grid))
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, origin="lower")
    for i in range(6):
        for j in range(6):
            if n[i, j] == 0:
                continue
            ax.text(j, i, f"{grid[i, j]:+.3f}\n{n[i, j]:,}", ha="center", va="center", fontsize=7.5,
                    color="white" if abs(grid[i, j]) > 0.6 * lim else INK)
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xlabel("Alive on T")
    ax.set_ylabel("Alive on CT")
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Mean gain over the table, full model (log-loss points)")
    fig.suptitle("Where the full model beats the table, by alive count", x=0.02, ha="left", fontsize=13, fontweight="bold",
                 color=INK, y=0.98)
    fig.text(0.02, 0.91, "Seed-0 datapoints of the 295 matches. Each cell: mean gain (top) and datapoints (bottom). "
             "Blue beats the table, red does worse.", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = FIGS / "gdt_by_alive.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out, agg


def fig_per_match(pm):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    ax = axes[0]
    ax.hist(pm["gain_full"].to_numpy(), bins=40, color=ACCENT, edgecolor="white")
    ax.axvline(0, color=INK, lw=1)
    ax.axvline(float(np.median(pm["gain_full"].to_numpy())), color=WARN, lw=1.2, ls="--", label="median match")
    share = float((pm["gain_full"] > 0).mean())
    ax.set_title(f"Full model vs table, per match\nbeats the table in {share:.0%} of matches", fontsize=10, color=INK, loc="left")
    ax.set_xlabel("Mean gain per sampled datapoint")
    ax.set_ylabel("Matches")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    if "diff_position" in pm.columns:
        d = pm["diff_position"].to_numpy()
        ax.hist(d, bins=40, color=WARN, edgecolor="white")
        ax.axvline(0, color=INK, lw=1)
        ax.axvline(float(np.median(d)), color=ACCENT, lw=1.2, ls="--", label="median match")
        ax.set_title(f"Full minus non-positional model, per match\nfull is better in {(d > 0).mean():.0%} of matches",
                     fontsize=10, color=INK, loc="left")
        ax.set_xlabel("Gain difference per sampled datapoint")
        ax.set_ylabel("Matches")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = FIGS / "gdt_per_match.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def example_round(rows, table, have, match):
    r = (rows.filter(pl.col("match") == match).join(table, on=sg.CELL, how="left").sort(["round", "second"]))
    if "no_position_s0" in r.columns and "full_s0" in r.columns:
        per_round = (r.with_columns((pl.col("full_s0") - pl.col("no_position_s0")).alias("d"))
                     .group_by("round").agg(pl.col("d").abs().sum().alias("score"), pl.col("y").first()))
        rnd = int(per_round.sort("score", descending=True)["round"][0])
    else:
        rnd = int(r["round"][0])
    rr = r.filter(pl.col("round") == rnd).sort("second")
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    t = rr["second"].to_numpy()
    ax.plot(t, rr["p_table"].to_numpy(), color=MUTED, lw=1.4, label="team's lookup table")
    if "no_position_s0" in rr.columns:
        ax.plot(t, rr["no_position_s0"].to_numpy(), color=ACCENT, lw=1.4, ls="--", label="non-positional (59 inputs)")
    if "full_s0" in rr.columns:
        ax.plot(t, rr["full_s0"].to_numpy(), color=INK, lw=2, label="full model (83 inputs)")
    y = int(rr["y"][0])
    ax.axhline(y, color=WARN if y == 0 else ACCENT, lw=0.8, ls=":")
    ax.text(t.max(), y + (0.03 if y == 1 else -0.03), f"CT {'won' if y == 1 else 'lost'} the round", ha="right",
            va="bottom" if y == 1 else "top", fontsize=8.5, color=INK)
    planted = rr.filter(pl.col("planted") == 1)
    if planted.height:
        ax.axvline(float(planted["second"][0]), color=MUTED, lw=0.8, ls="--")
        ax.text(float(planted["second"][0]) + 1, 0.97, "bomb planted", fontsize=8, color=MUTED, va="top")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Seconds since the round went live")
    ax.set_ylabel("P(CT wins the round)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(f"One round, second by second: match {short(match)}, round {rnd}", x=0.02, ha="left", fontsize=13,
                 fontweight="bold", color=INK, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = FIGS / "gdt_example_round.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out, rnd, rr


def dates_for(ids):
    d = (pl.scan_parquet(CACHE).select(["match_key", "match_date"]).unique(subset=["match_key"])
         .filter(pl.col("match_key").is_in(ids)).collect())
    return {r["match_key"]: str(r["match_date"])[:10] for r in d.iter_rows(named=True)}


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    rows, table, have = load()
    pts = gains(rows, table, have)
    pm = per_match(pts, have)
    pm.write_csv(TABLES / "gdt_per_match.csv")

    check = scorer_check(pts, have)
    print("datapoints", pts.height, "matches", pm.height)
    for m, v in check.items():
        print(f"  {m}: plain mean gain {v['plain_mean_gain']:.5f}  scorer GDT {v['scorer_GDT']}")

    eligible = pm.filter(pl.col("datapoints") >= MIN_POINTS)
    best = eligible.sort("gain_full", descending=True).row(0, named=True)
    worst = eligible.sort("gain_full").row(0, named=True)
    picks = [("beats the table most", best), ("does worst against the table", worst)]
    if "diff_position" in pm.columns:
        picks.append(("gains most from the position inputs", eligible.sort("diff_position", descending=True).row(0, named=True)))
        picks.append(("loses most from the position inputs", eligible.sort("diff_position").row(0, named=True)))
    dates = dates_for([p[1]["match"] for p in picks])
    ex_rows = []
    for label, r in picks:
        ex_rows.append({"example": label, "match": r["match"], "date": dates.get(r["match"], ""), "datapoints": r["datapoints"],
                        "gain_full": round(r["gain_full"], 5),
                        "gain_no_position": round(r["gain_no_position"], 5) if "gain_no_position" in r else None,
                        "diff_position": round(r["diff_position"], 5) if "diff_position" in r else None})
    pl.DataFrame(ex_rows).write_csv(TABLES / "gdt_examples.csv")
    print(pl.DataFrame(ex_rows))

    fa, agg = fig_by_alive(pts)
    agg.sort(["alive_ct", "alive_t"]).write_csv(TABLES / "gdt_by_alive.csv")
    fp = fig_per_match(pm)

    pos_pick = next((p for p in picks if p[0] == "gains most from the position inputs"), None)
    if pos_pick:
        fr, rnd, rr = example_round(rows, table, have, pos_pick[1]["match"])
        print("example round", rnd, "seconds", rr.height)
    summary = {"matches": pm.height, "datapoints_seed0": pts.height, "share_matches_full_beats_table": float((pm["gain_full"] > 0).mean()),
               "median_match_gain_full": float(pm["gain_full"].median()), "check": check}
    if "diff_position" in pm.columns:
        summary["share_matches_full_beats_no_position"] = float((pm["diff_position"] > 0).mean())
        summary["median_match_diff_position"] = float(pm["diff_position"].median())
    json.dump(summary, open(TABLES / "gdt_examples_summary.json", "w"), indent=1)
    print("wrote", fa, fp, "| summary", TABLES / "gdt_examples_summary.json")


if __name__ == "__main__":
    main()
