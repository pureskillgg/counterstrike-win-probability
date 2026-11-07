"""Verify the slide 11 claims: does every arm beat the lookup table in every segment, and how far apart are the GDT-trained arms?

Uses failure_analysis.py's loader on the GDT-trained proxy rows (validation only; no holdout, frozen-test or Oct 26-31 rows).
"""
import importlib.util
import sys
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
FA = r"C:/Users/<user>/AppData/Local/Temp/claude/44708acea14da7d4/047ee5fa-b35b-4b25-97b1-beaf96834ff2/scratchpad/failure_analysis.py"
spec = importlib.util.spec_from_file_location("fa", FA)
fa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa)

sit = fa.load(fa.GDT_ROWS)
y = sit["y"].to_numpy().astype(np.float64)
tab = fa.ll(sit["p_table"].to_numpy(), y)
print(f"rows: {len(y):,}, matches: {sit['match'].n_unique()}")

arms = {}
for arm in fa.ALL_ARMS:
    cols = [f"{arm}_s{s}" for s in fa.SEEDS]
    cols = [c for c in cols if c in sit.columns]
    if cols:
        arms[arm] = np.mean([fa.ll(sit[c].to_numpy(), y) for c in cols], axis=0)
print("arms with columns:", sorted(arms))

five = [a for a in fa.FIVE if a in arms]
seg_cols = {
    "planted": sit["planted"].to_numpy(),
    "tb": sit["tb"].to_numpy(),
    "alive": sit["alive_ct"].to_numpy() * 10 + sit["alive_t"].to_numpy(),
    "econ_ct": sit["econ_ct"].to_numpy(),
    "econ_t": sit["econ_t"].to_numpy(),
}

total_segments = 0
arm_table_fail = []
spread_max = []
for seg_name, seg in seg_cols.items():
    for k in np.unique(seg):
        m = seg == k
        n = int(m.sum())
        if n < 20000:
            continue
        total_segments += 1
        gains = {a: float((tab[m] - arms[a][m]).mean()) for a in arms}
        losers = [a for a, g in gains.items() if g <= 0]
        if losers:
            arm_table_fail.append((seg_name, str(k), n, losers))
        five_gains_vs_gru = [float((arms["gru"][m] - arms[a][m]).mean()) for a in five]
        spread_max.append((seg_name, str(k), n, max(five_gains_vs_gru) - min(five_gains_vs_gru)))

print(f"\nsegments with n >= 20,000: {total_segments}")
print(f"segments where at least one arm does NOT beat the table: {len(arm_table_fail)}")
for row in arm_table_fail[:20]:
    print("  ", row)
if spread_max:
    worst = max(spread_max, key=lambda r: r[3])
    print(f"\nlargest spread between five-seed GDT arms (relative to GRU) in one segment: {worst[3]:.5f} at {worst[0]}={worst[1]} (n={worst[2]:,})")
    print(f"median segment spread: {float(np.median([r[3] for r in spread_max])):.5f}")
