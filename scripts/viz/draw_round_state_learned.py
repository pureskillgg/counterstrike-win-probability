"""Draw the learned round-state embedding (z from train_round_state_encoder.py) for one map, with the same checks as the PCA baseline.

  G:\\pureskill\\csds-cs2\\.pixi\\envs\\default\\python.exe -I -X utf8 scripts\\viz\\draw_round_state_learned.py --map de_mirage --emb h3

Reads data/round_state/<map>.npz and .json (states) and data/round_state/<map>_<emb>.npz (z, same row order).
The drawing uses the two principal directions of z only to place the points on a plane. Neighbour search uses all eight z numbers.
Colours: round winner (data/combined/rounds.parquet, final: 2 = T, 3 = CT), seconds into the round, alive count.
Writes deliverables/stage2/figs/round_state_<map>_learned_<emb>_*.png and deliverables/stage2/ROUND_STATE_CHECKS_<MAP>_LEARNED_<EMB>.txt.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import draw_round_state as drs  # noqa: E402

TEST = drs.TEST


def plane(Z):
    """Two directions of Z for drawing only (PCA of z, centred on its mean)."""
    rng = np.random.default_rng(0)
    sub = Z[rng.choice(len(Z), size=min(200_000, len(Z)), replace=False)]
    mu = sub.mean(axis=0)
    _, _, vt = np.linalg.svd(sub - mu, full_matrices=False)
    return ((Z - mu) @ vt[:2].T).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", required=True)
    ap.add_argument("--emb", default="h3")
    a = ap.parse_args()

    arrays, meta = drs.load(a.map)
    emb = np.load(drs.ROOT / "data" / "round_state" / f"{a.map}_{a.emb}.npz")
    Z = emb["z"]
    X, split = arrays["X"], arrays["split"]
    if Z.shape[0] != X.shape[0]:
        raise SystemExit(f"embedding has {Z.shape[0]} rows, states have {X.shape[0]}: they must be the same rows")
    bounds = meta["window"]
    drs.FIG_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"learned_{a.emb}"
    lines = [f"ROUND-STATE CHECKS, {a.map}, LEARNED embedding ({a.emb}: set encoder, next-state prediction, 8 numbers)", ""]
    lines.append(f"rows {X.shape[0]}; embedding z shape {Z.shape}; z mean {np.round(Z.mean(axis=0), 3).tolist()}")

    df = drs.row_table(arrays)
    cov = df.select(drs.pl.col("winner").is_not_null().mean()).item()
    dist = df.group_by("winner").len().sort("winner").to_dicts()
    lines.append(f"round winner known for {cov:.4%} of rows; winner codes {dist}")
    lines.append("")

    P = plane(Z)
    drs.draw_pca(df, P, drs.FIG_DIR / f"round_state_{a.map}_{tag}_pca.png")

    test_rows = np.flatnonzero((split == TEST) & (X[:, :, 0].sum(1) >= 6))
    picks = np.random.default_rng(0).choice(test_rows, size=4, replace=False)
    queries = []
    lines.append("NEAREST NEIGHBOURS in z (same match excluded; 5 neighbours; Euclidean distance in z)")
    for q in picks:
        nbr, dd = drs.neighbours(Z, arrays, q)
        queries.append((q, nbr, dd))
        qa = (int(X[q, :5, 0].sum()), int(X[q, 5:, 0].sum()))
        qw = df["winner"][int(q)]
        lines.append(f"  query row {q}: CT alive {qa[0]}, T alive {qa[1]}, second {arrays['second'][q]}, winner {qw}")
        same_state = same_winner = 0
        for n_i, d_i in zip(nbr, dd):
            na = (int(X[n_i, :5, 0].sum()), int(X[n_i, 5:, 0].sum()))
            nw = df["winner"][int(n_i)]
            same_state += int(na == qa)
            same_winner += int(nw == qw)
            lines.append(f"      nbr row {n_i}: d={d_i:.3f}, CT {na[0]} T {na[1]}, second {arrays['second'][n_i]}, winner {nw}")
        lines.append(f"    neighbours with the same alive counts: {same_state}/5; same winner: {same_winner}/5")
    drs.draw_neighbours(arrays, bounds, queries, drs.FIG_DIR / f"round_state_{a.map}_{tag}_neighbours.png")

    midx, rnd, n = drs.draw_path(arrays, df, P, drs.FIG_DIR / f"round_state_{a.map}_{tag}_path.png")
    lines.append("")
    lines.append(f"TRACED ROUND: test round with the most seconds: match_idx {midx}, round {rnd}, {n} seconds")

    report = drs.ROOT / "deliverables" / "stage2" / f"ROUND_STATE_CHECKS_{a.map.upper()}_LEARNED_{a.emb.upper()}.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
