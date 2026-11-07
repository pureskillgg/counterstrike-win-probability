"""R2 industry benchmark arithmetic.

Compares publicly reported CS win-probability benchmark numbers against
coin-flip baselines. Inputs are copied from pages opened in this session
(see notes/research/swarm_R2_industry.md for URLs). Prints a markdown table.

Baselines:
  Brier score of constant 0.5 forecast      = 0.25
  Log loss of constant 0.5 forecast         = ln 2 = 0.693147...
  Brier skill score vs 0.5 = 1 - Brier / 0.25 (assumes balanced outcomes)
"""

import math

COIN_BRIER = 0.25
COIN_LOGLOSS = math.log(2)

# (source, model, reported log loss, reported Brier, reported ROC-AUC or None)
ROWS = [
    # Xenopoulos et al. 2020, arXiv 2011.01324 (CS:GO, 18M test game states)
    ("Xenopoulos 2020 (arXiv 2011.01324)", "XGBoost", 0.5353, 0.1842, 0.7913),
    ("Xenopoulos 2020 (arXiv 2011.01324)", "CatBoost", 0.5443, 0.1875, 0.7851),
    ("Xenopoulos 2020 (arXiv 2011.01324)", "Logistic regression", 0.5539, 0.1912, 0.7743),
    ("Xenopoulos 2020 (arXiv 2011.01324)", "Map-average baseline", 0.6917, 0.2493, 0.5303),
    # lc-leonardo/cs2-match-prediction README (hobby project, CS2 BO3.gg data)
    ("lc-leonardo GitHub README", "LightGBM", 0.659, 0.231, 0.72),
    ("lc-leonardo GitHub README", "Random Forest", 0.665, 0.233, 0.71),
    ("lc-leonardo GitHub README", "XGBoost", 0.672, 0.235, 0.70),
    ("lc-leonardo GitHub README", "MLP", 0.675, 0.238, 0.69),
]


def main() -> None:
    print("| Source | Model | Log loss | Gain vs ln2 (nats) | Brier | Brier skill vs 0.25 | ROC-AUC |")
    print("|---|---|---|---|---|---|---|")
    for source, model, ll, brier, auc in ROWS:
        ll_gain = COIN_LOGLOSS - ll
        brier_skill = 1.0 - brier / COIN_BRIER
        auc_txt = "n/a" if auc is None else f"{auc:.4f}"
        print(
            f"| {source} | {model} | {ll:.4f} | {ll_gain:+.4f} | {brier:.4f} | "
            f"{brier_skill:+.3f} | {auc_txt} |"
        )
    print()
    print(f"coin-flip log loss = {COIN_LOGLOSS:.6f}; coin-flip Brier = {COIN_BRIER:.2f}")

    # Accuracy-only claims with no stated definition, for reference (not computed).
    print("Thesis (Rocha 2026, UFSCar, pre-match): holdout ROC-AUC 0.6828, accuracy 0.6293; base rate not stated.")


if __name__ == "__main__":
    main()
