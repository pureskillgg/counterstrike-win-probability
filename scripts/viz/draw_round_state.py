"""Draw the round-state arrays for one map: a PCA picture, nearest neighbours, and one traced round.

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\viz\\draw_round_state.py --map de_mirage

Reads data/round_state/<map>.npz and .json (from scripts/model/build_round_state_map.py). The embedding here is a PCA baseline
(numpy, fitted on a train subsample), not the learned set encoder: it is the first picture, to see what the checks can show.
Colours: round winner (data/combined/rounds.parquet, column final: 2 = T, 3 = CT), time into the round, alive count.
Writes PNGs to deliverables/stage2/figs/ and a text report to deliverables/stage2/ROUND_STATE_CHECKS_<MAP>.txt.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "model"))
import embed_round_state as ers  # noqa: E402

TRAIN, VAL, TEST = 0, 1, 2
SPLIT_NAME = {TRAIN: "train", VAL: "val", TEST: "test"}
FIG_DIR = ROOT / "deliverables" / "stage2" / "figs"
N_COMPONENTS = ers.LATENT_DIM


def load(map_name):
    npz = np.load(ROOT / "data" / "round_state" / f"{map_name}.npz", allow_pickle=False)
    arrays = {k: npz[k] for k in npz.files}
    meta = json.loads((ROOT / "data" / "round_state" / f"{map_name}.json").read_text(encoding="utf-8"))
    return arrays, meta


def row_table(arrays):
    X = arrays["X"]
    df = pl.DataFrame(
        {
            "match_idx": arrays["match_idx"],
            "round": arrays["round"].astype(np.int32),
            "second": arrays["second"].astype(np.int32),
            "split": arrays["split"],
            "alive_ct": X[:, :5, 0].sum(axis=1).astype(np.int8),
            "alive_t": X[:, 5:, 0].sum(axis=1).astype(np.int8),
        }
    ).with_columns(pl.col("match_idx").cast(pl.Int64))
    keys = pl.DataFrame({"match_idx": np.arange(len(arrays["match_keys"]), dtype=np.int64),
                         "match_key": arrays["match_keys"]})
    rounds = (
        pl.read_parquet(ROOT / "data" / "combined" / "rounds.parquet", columns=["match_key", "round", "final"])
        .with_columns(pl.col("round").cast(pl.Int32))
        .unique(["match_key", "round"], keep="first")
        .rename({"final": "winner"})
    )
    df = df.join(keys, on="match_idx", how="left").join(rounds, on=["match_key", "round"], how="left")
    return df


def pca_embed(X, split):
    train_idx = np.flatnonzero(split == TRAIN)
    rng = np.random.default_rng(0)
    sub = rng.choice(train_idx, size=min(300_000, train_idx.size), replace=False)
    model = ers.fit_pca_baseline(X[sub], n_components=N_COMPONENTS)
    Z = np.empty((len(X), N_COMPONENTS), dtype=np.float32)
    step = 500_000
    for start in range(0, len(X), step):
        Z[start : start + step] = ers.transform_pca_baseline(model, X[start : start + step])
    return Z, model


def world_xy(X, bounds):
    span = bounds["xy_max"] - bounds["xy_min"]
    xs = bounds["xy_min"] + X[:, :, 1] * span
    ys = bounds["xy_min"] + X[:, :, 2] * span
    return xs, ys


def draw_pca(df, Z, path):
    rng = np.random.default_rng(1)
    pick = rng.choice(len(Z), size=min(60_000, len(Z)), replace=False)
    z1, z2 = Z[pick, 0], Z[pick, 1]
    winner = df["winner"].to_numpy()[pick]
    second = df["second"].to_numpy()[pick]
    alive = (df["alive_ct"] + df["alive_t"]).to_numpy()[pick]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    t_colour = np.where(winner == 2, "#d9622b", np.where(winner == 3, "#2b6fd9", "#999999"))
    axes[0].scatter(z1, z2, c=t_colour, s=2, alpha=0.5, linewidths=0)
    axes[0].set_title("coloured by round winner (orange T, blue CT, grey unknown)")
    sc = axes[1].scatter(z1, z2, c=second, s=2, alpha=0.5, linewidths=0, cmap="viridis")
    axes[1].set_title("coloured by seconds into the round")
    fig.colorbar(sc, ax=axes[1], shrink=0.8)
    sc2 = axes[2].scatter(z1, z2, c=alive, s=2, alpha=0.5, linewidths=0, cmap="plasma", vmin=0, vmax=10)
    axes[2].set_title("coloured by players alive (both sides)")
    fig.colorbar(sc2, ax=axes[2], shrink=0.8)
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.suptitle(f"de_mirage round states, PCA baseline ({len(pick)} random rows of all splits)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def neighbours(Z, arrays, q, k=5):
    d = np.linalg.norm(Z - Z[q], axis=1)
    d[arrays["match_idx"] == arrays["match_idx"][q]] = np.inf  # same match: consecutive seconds are near-copies
    order = np.argpartition(d, k)[:k]
    order = order[np.argsort(d[order])]
    return order, d[order]


def draw_neighbours(arrays, bounds, queries, path, k=5):
    X = arrays["X"]
    xs, ys = world_xy(X, bounds)
    fig, axes = plt.subplots(len(queries), k + 1, figsize=(2.3 * (k + 1), 2.5 * len(queries)))
    for r, (q, nbr, dist) in enumerate(queries):
        for c, idx in enumerate([q] + list(nbr)):
            ax = axes[r, c]
            alive = X[idx, :, 0] == 1
            ax.scatter(xs[idx][alive & (np.arange(10) < 5)], ys[idx][alive & (np.arange(10) < 5)], c="#2b6fd9", s=18)
            ax.scatter(xs[idx][alive & (np.arange(10) >= 5)], ys[idx][alive & (np.arange(10) >= 5)], c="#d9622b", s=18)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            title = "query" if c == 0 else f"nbr {c} d={dist[c - 1]:.2f}"
            ax.set_title(title, fontsize=8)
    fig.suptitle("query state (left) and its 5 nearest neighbours; blue CT, orange T; positions only (no map image)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_path(arrays, df, Z, path):
    d = df.filter(pl.col("split") == TEST).group_by(["match_idx", "round"]).agg(pl.len().alias("n")).sort("n", descending=True)
    midx, rnd = d["match_idx"][0], d["round"][0]
    rows = np.flatnonzero((arrays["match_idx"] == midx) & (arrays["round"] == rnd))
    rows = rows[np.argsort(arrays["second"][rows])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8))
    a1.plot(Z[rows, 0], Z[rows, 1], "-", color="#888888", lw=0.8)
    sc = a1.scatter(Z[rows, 0], Z[rows, 1], c=arrays["second"][rows], cmap="viridis", s=14)
    a1.scatter(Z[rows[0], 0], Z[rows[0], 1], c="red", s=60, marker="*", label="start")
    a1.scatter(Z[rows[-1], 0], Z[rows[-1], 1], c="black", s=60, marker="X", label="end")
    a1.legend(loc="best")
    fig.colorbar(sc, ax=a1, shrink=0.8, label="seconds into round")
    a1.set_title("one test round traced through the PCA picture")
    a1.set_xlabel("PC1")
    a1.set_ylabel("PC2")
    a2.plot(arrays["second"][rows], (arrays["X"][rows, :5, 0]).sum(1), label="CT alive")
    a2.plot(arrays["second"][rows], (arrays["X"][rows, 5:, 0]).sum(1), label="T alive")
    a2.set_xlabel("seconds into round")
    a2.set_ylabel("alive")
    a2.legend()
    a2.set_title(f"same round: match {arrays['match_keys'][midx]}, round {rnd}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return int(midx), int(rnd), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    a = ap.parse_args()
    arrays, meta = load(a.map)
    bounds = meta["window"]
    X, split = arrays["X"], arrays["split"]
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = [f"ROUND-STATE CHECKS, {a.map} (PCA baseline; not the learned encoder)", ""]
    lines.append(f"rows {X.shape[0]}, shape {X.shape[1:]}, window {bounds}")
    lines.append(f"rows per split {meta['stats']['rows_per_split']}")
    lines.append(f"matches per split {meta['stats']['matches_per_split']}")
    lines.append(f"out-of-window fraction per split {meta['stats']['frac_alive_coords_outside_window']}")
    lines.append(f"rows dropped as recording gaps {meta['stats']['rows_dropped_bad_coordinates']}; "
                 f"position rows with no snapshot match {meta['stats']['position_rows_unmapped_match_key']}")
    lines.append("")

    df = row_table(arrays)
    cov = df.select((pl.col("winner").is_not_null()).mean()).item()
    dist = df.group_by("winner").len().sort("winner").to_dicts()
    lines.append(f"round winner known for {cov:.4%} of rows; winner codes {dist}")
    lines.append("")

    Z, model = pca_embed(X, split)
    lines.append(f"PCA baseline on flattened (10 x 6) states, fitted on a train subsample: explained variance "
                 f"ratio {np.round(model.explained_variance_ratio, 4).tolist()}")
    draw_pca(df, Z, FIG_DIR / f"round_state_{a.map}_pca.png")

    test_rows = np.flatnonzero((split == TEST) & (X[:, :, 0].sum(1) >= 6))
    picks = np.random.default_rng(0).choice(test_rows, size=4, replace=False)
    queries = []
    lines.append("")
    lines.append("NEAREST NEIGHBOURS (same match excluded; 5 neighbours; distance in PCA space)")
    for q in picks:
        nbr, dist = neighbours(Z, arrays, q)
        queries.append((q, nbr, dist))
        qa = (int(X[q, :5, 0].sum()), int(X[q, 5:, 0].sum()))
        qw = df["winner"][int(q)]
        lines.append(f"  query row {q}: CT alive {qa[0]}, T alive {qa[1]}, second {arrays['second'][q]}, "
                     f"winner {qw}, match {arrays['match_keys'][arrays['match_idx'][q]]}")
        same_state = 0
        same_winner = 0
        for n_i, dd in zip(nbr, dist):
            na = (int(X[n_i, :5, 0].sum()), int(X[n_i, 5:, 0].sum()))
            nw = df["winner"][int(n_i)]
            same_state += int(na == qa)
            same_winner += int(nw == qw)
            lines.append(f"      nbr row {n_i}: d={dd:.3f}, CT {na[0]} T {na[1]}, second {arrays['second'][n_i]}, winner {nw}")
        lines.append(f"    neighbours with the same alive counts: {same_state}/5; same winner: {same_winner}/5")
    draw_neighbours(arrays, bounds, queries, FIG_DIR / f"round_state_{a.map}_neighbours.png")

    midx, rnd, n = draw_path(arrays, df, Z, FIG_DIR / f"round_state_{a.map}_path.png")
    lines.append("")
    lines.append(f"TRACED ROUND: test round with the most seconds: match_idx {midx}, round {rnd}, {n} seconds")
    lines.append("figures: round_state_%s_pca.png, round_state_%s_neighbours.png, round_state_%s_path.png in %s"
                 % (a.map, a.map, a.map, FIG_DIR))

    report = ROOT / "deliverables" / "stage2" / f"ROUND_STATE_CHECKS_{a.map.upper()}.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
