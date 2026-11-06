"""Explains given matches second by second with TreeSHAP, grouped into input bundles.

For each match: every matrix row of the match (validation rows, so these are out-of-sample predictions), the contribution of each
bundle to the log-odds, and the largest moves in P(CT wins) between consecutive rows with the bundle that moved most. Writes:
  tables/match_<name>_timeline.csv   one row per matrix row: round, second, label, P(CT wins), log-odds, bundle contributions
  tables/match_swings.csv            the largest moves in P(CT wins) per match, with the bundles behind each move
  tables/match_bundle_summary.csv    per match and bundle: mean contribution (log-odds) and share of absolute contribution
  figures/match_<name>.png           P(CT wins) over the match, and the bundle contributions as a heatmap

Rows are the seconds the matrix holds for the match. They are not every second of the round.

Run with the pixi Python from the repo root:
    G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe scripts\\explain\\match_explain.py
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import xgboost as xgb  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "explain"))
import xgb83_explain as XE  # noqa: E402  (GROUPS: the bundle definitions the report uses)

MATRIX = Path("C:/Users/13144/sprint/matrix_trainctr")
WORK = Path("C:/Users/13144/sprint/explain83")
MODEL = REPO / "models" / "winprob_xgb83_final.json"
# The file holds best_iteration + 100 boosted rounds. Predictions and explanations use the rounds up to the best iteration,
# as the explanation stage and the calibration do; a plain predict (all rounds) differs by up to 0.02.
BEST = json.load(open(REPO / "models" / "winprob_xgb83_final.meta.json", encoding="utf-8"))["delivered_file"]["best_iteration"]
ITER = (0, BEST + 1)
TABLES = REPO / "docs" / "explainability" / "tables"
FIGS = REPO / "docs" / "explainability" / "figures"

MATCHES = {  # match_key as stored in the matrix; the three examples of the report and the GDT examples
    "largest_loss": "csds/2026/10/09/706c83dd-bd8b-4523-8ad6-fd609988d8c3/csds",
    "positional_help": "csds/2026/09/22/f1ed1bfb-9910-43f2-8ca8-5d81ff15d013/csds",
    "worst_vs_table": "csds/2026/10/05/27c71f8f-1c97-4eff-b8f4-2ddbab993b01/csds",
}
BUNDLES = ["LOADOUT_DIFF", "LOADOUT_CT", "LOADOUT_T", "ROUND_STATE", "MAP", "BOMB_GEO", "GEO_SITES_SPREAD", "GEO_CONTACT",
           "VIEW", "BLIND"]
LABEL = {"LOADOUT_DIFF": "loadout difference", "LOADOUT_CT": "CT loadout", "LOADOUT_T": "T loadout",
         "ROUND_STATE": "round state", "MAP": "map", "BOMB_GEO": "bomb geometry", "GEO_SITES_SPREAD": "site and spread",
         "GEO_CONTACT": "contact", "VIEW": "view", "BLIND": "blindness"}
PLAIN = {"LOADOUT_DIFF": "loadout difference", "LOADOUT_CT": "CT loadout", "LOADOUT_T": "T loadout"}


def family(group):
    if group.startswith("LOADOUT_CT_"):
        return "LOADOUT_CT"
    if group.startswith("LOADOUT_T_"):
        return "LOADOUT_T"
    return group


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(MATRIX / "meta.json"))
    feats = meta["feats_all"]
    fam = {}
    for g, members in XE.GROUPS.items():
        if g.startswith("ALL_"):
            continue
        for m in members:
            fam.setdefault(m, family(g))
    missing = [f for f in feats if f not in fam]
    assert not missing, f"features without a bundle: {missing}"
    bundle_of = [fam[f] for f in feats]
    assert set(bundle_of) <= set(BUNDLES), set(bundle_of) - set(BUNDLES)

    a2 = meta["n_fit"] + meta["n_es"]
    keys = pl.read_parquet(MATRIX / "val_keys.parquet").with_row_index("vi")
    p_saved = np.load(WORK / "p_val_shipped.npy")
    X = np.load(MATRIX / "X.npy", mmap_mode="r")
    bst = xgb.Booster()
    bst.load_model(str(MODEL))

    timeline_rows, swing_rows, summary_rows = [], [], []
    for name, match in MATCHES.items():
        sub = keys.filter(pl.col("match_key") == match).sort(["round", "second"])
        assert sub.height > 0, f"match {match} not in the validation rows"
        vi = sub["vi"].to_numpy()
        rows = np.asarray(X[a2 + vi], dtype=np.float32)
        c = bst.predict(xgb.DMatrix(rows), pred_contribs=True, iteration_range=ITER)  # last column is the bias (base log-odds)
        z = c.sum(axis=1)
        p = 1.0 / (1.0 + np.exp(-z))
        gap = float(np.max(np.abs(p - p_saved[vi])))
        assert gap < 1e-4, f"{name}: explanation does not reproduce the saved predictions (max gap {gap})"
        bundle_cols = {b: c[:, [j for j in range(len(feats)) if bundle_of[j] == b]].sum(axis=1) for b in BUNDLES}
        bias = c[:, -1]
        assert np.max(np.abs(z - (bias + sum(bundle_cols.values())))) < 1e-4, "bundles do not add up to the log-odds"

        r = sub["round"].to_numpy()
        s = sub["second"].to_numpy()
        y = sub["label_ct_win"].to_numpy()
        df = pl.DataFrame({"match": [match] * sub.height, "round": r, "second": s, "label_ct_win": y,
                           "p_ct": p, "logit": z, "base_logit": bias})
        df = df.with_columns([pl.Series(b, bundle_cols[b]) for b in BUNDLES])
        df.write_csv(TABLES / f"match_{name}_timeline.csv")
        timeline_rows.append(name)

        # largest moves between consecutive rows of the same round, and the bundle that moved most
        same = np.r_[False, r[1:] == r[:-1]]
        idx = np.flatnonzero(same)
        dp = p[idx] - p[idx - 1]
        dc = np.stack([bundle_cols[b][idx] - bundle_cols[b][idx - 1] for b in BUNDLES], axis=1)
        order = np.argsort(-np.abs(dp))[:6]
        for k in order:
            i = idx[k]
            moved = dc[k]
            top = np.argsort(-np.abs(moved))[:2]
            swing_rows.append({
                "match": name, "round": int(r[i]), "second_from": int(s[i - 1]), "second_to": int(s[i]),
                "p_from": round(float(p[i - 1]), 4), "p_to": round(float(p[i]), 4), "change_pp": round(float(dp[k]) * 100, 2),
                "biggest_mover": LABEL[BUNDLES[top[0]]], "mover_logit": round(float(moved[top[0]]), 3),
                "second_mover": LABEL[BUNDLES[top[1]]], "second_logit": round(float(moved[top[1]]), 3),
                "label_ct_win": int(y[i]),
            })

        total_abs = sum(np.abs(bundle_cols[b]) for b in BUNDLES).sum()
        for b in BUNDLES:
            summary_rows.append({"match": name, "bundle": LABEL[b], "mean_logit": round(float(bundle_cols[b].mean()), 4),
                                 "share_of_abs": round(float(np.abs(bundle_cols[b]).sum() / total_abs), 4)})
        print(f"{name}: {sub.height} rows, {sub['round'].n_unique()} rounds, max gap to saved predictions {gap:.2e}", flush=True)

        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 7.2), gridspec_kw={"height_ratios": [1, 1.5]}, sharex=True)
        x = np.arange(sub.height)
        ax0.plot(x, p, color="#1B2430", lw=1.6)
        ax0.set_ylim(-0.02, 1.02)
        ax0.set_ylabel("P(CT wins)")
        ax0.grid(axis="y", color="#E6E3DA", lw=0.6)
        starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]])
        for st in starts:
            ax0.axvline(st - 0.5, color="#9AA3AD", lw=0.6, ls=":")
            ax1.axvline(st - 0.5, color="#9AA3AD", lw=0.6, ls=":")
            end = min(st + np.flatnonzero(r[st:] != r[st])[0] if np.any(r[st:] != r[st]) else sub.height, sub.height)
            ax0.text(st + 0.3, 1.0, f"R{int(r[st])}", fontsize=7, va="top", color="#4A5563")
            outcome_ct = int(y[st])
            ax0.plot([st, end - 1], [outcome_ct, outcome_ct], color="#2F6F8F" if outcome_ct else "#B5542D", lw=3, alpha=0.5)
        mat = np.stack([bundle_cols[b] for b in BUNDLES])
        lim = float(np.percentile(np.abs(mat), 99)) or 1.0
        im = ax1.imshow(mat, aspect="auto", cmap="RdBu", vmin=-lim, vmax=lim, interpolation="nearest",
                        extent=(-0.5, sub.height - 0.5, len(BUNDLES) - 0.5, -0.5))
        ax1.set_yticks(range(len(BUNDLES)))
        ax1.set_yticklabels([LABEL[b] for b in BUNDLES], fontsize=8)
        ax1.set_xlabel("rows of the match in round order (not every second; round starts dotted, the bar on the top panel is the round's outcome)")
        cb = fig.colorbar(im, ax=ax1, shrink=0.9, pad=0.01)
        cb.set_label("contribution to log-odds of CT winning", fontsize=8)
        fig.suptitle(f"Match {match.split('/')[-2][:8]}…: each input bundle's contribution, row by row", x=0.02, ha="left",
                     fontsize=13, fontweight="bold", color="#1B2430", y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(FIGS / f"match_{name}.png", dpi=150)
        plt.close(fig)

    pl.DataFrame(swing_rows).write_csv(TABLES / "match_swings.csv")
    pl.DataFrame(summary_rows).write_csv(TABLES / "match_bundle_summary.csv")
    print("wrote timelines", timeline_rows, "swings", len(swing_rows), "summary", len(summary_rows))


if __name__ == "__main__":
    main()
