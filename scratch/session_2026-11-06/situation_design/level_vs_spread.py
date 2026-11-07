import csv
PATH = "C:/Users/<user>/Documents/pureskill/counterstrike-win-probability/docs/eval_reports/positions_validation_gdt/by_situation.csv"
rows = list(csv.DictReader(open(PATH, encoding="utf-8")))
N = sum(int(r["datapoints"]) for r in rows)
print("cells", len(rows), "datapoints", N)
for model in ["p_full_trainctr", "p_base"]:
    weak = [r for r in rows if float(r["gain_" + model]) < 0]
    wdp = sum(int(r["datapoints"]) for r in weak)
    level_wrong = spread_wrong = both = 0
    lw_dp = sw_dp = 0
    for r in weak:
        share = float(r["ct_win_share"]); pm = float(r["mean_" + model]); pt = float(r["p_table"])
        dm = abs(pm - share); dt = abs(pt - share)
        n = int(r["datapoints"])
        if dm > dt:
            level_wrong += 1; lw_dp += n   # model's mean sits further from observed share than the table's
        else:
            spread_wrong += 1; sw_dp += n  # model's mean is at least as close as the table's; loss is in the spread
    print(model, "weak cells", len(weak), "datapoints", wdp, f"({wdp/N:.2%})")
    print("  model mean further from observed share than table:", level_wrong, "cells,", lw_dp, "datapoints")
    print("  model mean at least as close as table (spread/noise):", spread_wrong, "cells,", sw_dp, "datapoints")
    for lo, hi in [(1, 10), (11, 50), (51, 499), (500, 10**9)]:
        sub = [r for r in weak if lo <= int(r["datapoints"]) <= hi]
        lwc = sum(1 for r in sub if abs(float(r["mean_" + model]) - float(r["ct_win_share"])) > abs(float(r["p_table"]) - float(r["ct_win_share"])))
        print(f"  weak n {lo}-{hi}: {len(sub)} cells, model level worse than table in {lwc}")
